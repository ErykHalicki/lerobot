#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging
import math
import multiprocessing
import queue
import signal
import threading
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from . import gravity_model
from .config_rebot_b601_follower import RebotB601FollowerRobotConfig
from .thermal_monitor import COMPONENTS, ThermalMonitor

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)

# Joint controlled in FORCE_POS mode; every other joint runs in POS_VEL mode.
GRIPPER_MOTOR = "gripper"
# Wrist pitch, held clear of the shoulder during the ramp home. Negative angles
# point the gripper up on this arm.
WRIST_MOTOR = "wrist_flex"
# Per-joint Damiao motor models for the B601-DM (passed to motorbridge).
MOTOR_MODELS = {
    "shoulder_pan": "4340P",
    "shoulder_lift": "4340P",
    "elbow_flex": "4340P",
    "wrist_flex": "4310",
    "wrist_yaw": "4310",
    "wrist_roll": "4310",
    "gripper": "4310",
}
# Shared-array stand-in for "no time-to-overheat estimate", since a shared
# array of doubles has no room for None. Any real estimate is positive.
_NO_ETA = -1.0
_ENSURE_MODE_RETRIES = 9
_SETTLE_SEC = 0.01
_ZERO_SETTLE_SEC = 0.1
# Held at the end of go_home(). The motor (and the smoother ahead of it) trails
# the goal, and disconnect() stops the follower process the instant go_home()
# returns, which would otherwise cut the gripper off part-closed.
_GRIPPER_SETTLE_SEC = 0.5


class _SCurveAxis:
    """Causal per-axis S-curve filter: 3 cascaded first-order (exponential)
    filters tracking a target that can jump or arrive at irregular intervals.

    An explicit accel/jerk-clamped trajectory was tried first but chatters
    once the clamp saturates (a bang-bang/anti-windup problem). Cascaded
    first-order filters are monotonic and non-overshooting by construction
    for any dt, at the cost of no hard-clamped peak acceleration.
    """

    def __init__(self, tau_s: float):
        self.tau_s = tau_s
        self._s1: float | None = None
        self._s2: float = 0.0
        self._s3: float = 0.0

    def step(self, target_pos: float, dt: float) -> tuple[float, float]:
        """Advance by `dt` seconds toward `target_pos`, returning (position,
        velocity). Snaps directly to the target on the first call."""
        if self._s1 is None:
            self._s1 = self._s2 = self._s3 = target_pos
            return target_pos, 0.0

        # Clamp to 1.0 (full pass-through) once dt >= tau_s, instead of
        # letting dt/tau_s exceed 1 and overshoot each stage's bound.
        a = min(1.0, dt / self.tau_s) if self.tau_s > 0 else 1.0
        self._s1 += a * (target_pos - self._s1)
        self._s2 += a * (self._s1 - self._s2)
        prev_s3 = self._s3
        self._s3 += a * (self._s2 - self._s3)
        vel = (self._s3 - prev_s3) / dt if dt > 0 else 0.0
        return self._s3, vel


class _GravityCompensator:
    """Turns the joint angles read each tick into the torque that holds the arm
    up, in the follower process's motor order.

    Resolves the joint order, trim and clamp once at construction, so the send
    loop only pays for the model itself.

    Disabled config, a non-MIT arm, or a motor layout the model does not cover
    all yield zeros, so the send loop needs no special case.
    """

    def __init__(self, config: RebotB601FollowerRobotConfig, motor_names: list[str]):
        self._zeros = [0.0] * len(motor_names)
        self.enabled = config.gravity_compensation and config.control_mode == "mit"
        if self.enabled and not set(gravity_model.JOINT_NAMES) <= set(motor_names):
            missing = sorted(set(gravity_model.JOINT_NAMES) - set(motor_names))
            logger.warning(f"Gravity compensation off: no {missing} in motor_can_ids.")
            self.enabled = False
        if not self.enabled:
            return

        unknown = set(config.gravity_scale) - set(gravity_model.JOINT_NAMES)
        if unknown:
            raise ValueError(
                f"gravity_scale names {sorted(unknown)} are not arm joints; "
                f"pick from {list(gravity_model.JOINT_NAMES)}"
            )
        self._scale = np.array(
            [config.gravity_scale.get(name, 1.0) for name in gravity_model.JOINT_NAMES]
        ) * config.gravity_gain
        self._limit = np.minimum(gravity_model.EFFORT_LIMITS, config.gravity_max_torque)
        self._payload = config.gravity_payload_kg
        self._payload_com = tuple(config.gravity_payload_com)
        # Where each modelled joint sits in the motor order the loop indexes by.
        self._slots = [motor_names.index(name) for name in gravity_model.JOINT_NAMES]

    def torques(self, present_pos_deg) -> list[float]:
        """Holding torque per motor, N.m, from present positions in degrees."""
        if not self.enabled:
            return self._zeros
        q = np.radians([present_pos_deg[slot] for slot in self._slots])
        try:
            tau = gravity_model.gravity_torque(
                q, payload=self._payload, payload_com=self._payload_com
            )
        except Exception as e:
            logger.warning(f"Gravity compensation failed, commanding zero: {e}")
            return self._zeros
        tau = np.clip(tau * self._scale, -self._limit, self._limit)
        out = list(self._zeros)
        for value, slot in zip(tau, self._slots, strict=True):
            out[slot] = float(value)
        return out


class _HomeRamp:
    """The path back to the calibration zero pose, as a function of how long
    the ramp has been running.

    Sampled by elapsed time rather than stepped, so the arm arrives on schedule
    whatever the driving loop does: a late tick resumes where the clock says,
    instead of stretching the ramp out. Eased, so a joint that starts far from
    zero doesn't travel any faster than one that starts close.

    Three legs, held in one object so whoever drives it keeps no state:

    1. every joint eases to 0 over `home_duration_s`, except the gripper, which
       opens all the way (so it can't be gripping anything while the arm moves)
       and the wrist, which holds `home_wrist_flex_deg` to keep whatever is on
       the end clear of the shoulder -- coming in flat swings it into the
       shoulder at full ramp speed.
    2. the gripper closes to 0 while the wrist lowers to 0 over
       `gripper_close_duration_s`, so the load is set down over the whole close
       rather than dropped at the end of it.
    3. the settled pose is held for _GRIPPER_SETTLE_SEC, since both are still
       travelling when their goals stop moving.
    """

    def __init__(
        self,
        config: RebotB601FollowerRobotConfig,
        motor_names: list[str],
        start: dict[str, float],
    ):
        self._limits = config.joint_limits
        gripper_open = config.joint_limits[GRIPPER_MOTOR][0]
        wrist_up = config.home_wrist_flex_deg

        self._approach = {name: (start[name], 0.0) for name in motor_names}
        self._approach[GRIPPER_MOTOR] = (start[GRIPPER_MOTOR], gripper_open)
        self._approach[WRIST_MOTOR] = (start[WRIST_MOTOR], wrist_up)
        # Every joint stays in the close leg, holding the zero it reached, so
        # each sample is a complete goal rather than a partial one.
        self._close = dict.fromkeys(motor_names, (0.0, 0.0))
        self._close[GRIPPER_MOTOR] = (gripper_open, 0.0)
        self._close[WRIST_MOTOR] = (wrist_up, 0.0)

        self._approach_s = config.home_duration_s
        self._close_s = config.gripper_close_duration_s
        self.duration = self._approach_s + self._close_s + _GRIPPER_SETTLE_SEC

    def target(self, elapsed: float) -> dict[str, float]:
        """The goal position (degrees) for every joint `elapsed` seconds in."""
        if elapsed < self._approach_s:
            leg, span, t = self._approach, self._approach_s, elapsed
        else:
            leg, span, t = self._close, self._close_s, elapsed - self._approach_s
        frac = min(1.0, t / span) if span > 0 else 1.0
        eased = frac * frac * (3.0 - 2.0 * frac)  # smoothstep: zero velocity at both ends
        return {name: self._clip(name, s + eased * (e - s)) for name, (s, e) in leg.items()}

    def _clip(self, name: str, value: float) -> float:
        """Same soft joint limits send_action() applies, since the ramp reaches
        the shared goal without passing through it."""
        if name not in self._limits:
            return value
        low, high = self._limits[name]
        return max(low, min(high, value))


def _configured_gains(
    config: RebotB601FollowerRobotConfig, motor_names: list[str]
) -> tuple[list[float], list[float]]:
    """The MIT gains from the config, one per motor and the gripper's own pair
    included, in motor order.

    The home ramp drives with these rather than whatever is in shared memory,
    so a caller that softened the arm cannot leave it unable to answer its own
    goal. Resolved once, since the send loop reads them every tick.
    """

    def per_motor(value, gripper_value) -> list[float]:
        resolved = [
            float(value[i]) if isinstance(value, (list, tuple)) else float(value)
            for i in range(len(motor_names))
        ]
        for i, name in enumerate(motor_names):
            if name == GRIPPER_MOTOR:
                resolved[i] = float(gripper_value)
        return resolved

    return (
        per_motor(config.mit_kp, config.gripper_mit_kp),
        per_motor(config.mit_kd, config.gripper_mit_kd),
    )


def _follower_process_main(
    config: RebotB601FollowerRobotConfig,
    motor_names: list[str],
    goal_pos_shared,
    goal_ready,
    present_pos_shared,
    present_torq_shared,
    present_vel_shared,
    present_temp_mos_shared,
    present_temp_rotor_shared,
    present_temp_eta_shared,
    present_pos_ts,
    last_sent_shared,
    has_sent_ever,
    mit_kp_shared,
    mit_kd_shared,
    tau_ff_shared,
    command_queue: multiprocessing.Queue,
    ready_event,
    stop_event,
    error_queue: multiprocessing.Queue,
    home_done,
    homing_flag,
    thermal_shutdown,
) -> None:
    """Sole owner of the motor connection: runs the read/smooth/send loop at
    config.send_rate_hz in its own process, so nothing in the caller's
    process can delay it by holding the GIL.

    Communicates via shared memory only: `goal_pos_shared` (target),
    `mit_kp_shared`/`mit_kd_shared`/`tau_ff_shared` (how hard to chase it),
    `present_pos_shared`/`present_torq_shared`/`present_vel_shared`/
    `present_temp_mos_shared`/`present_temp_rotor_shared`/`present_pos_ts`
    (latest reading), `last_sent_shared` (what was actually sent).
    `command_queue` carries disable/enable/clear_error/home, `home_done` and
    `homing_flag` report the state of a home ramp back to the parent, and
    `thermal_shutdown` asks it to disconnect an overheating arm.

    Thermal watching lives here rather than in the parent because this loop
    reads the temperatures anyway and keeps its rate regardless of what the
    caller's process is doing -- a protection that a long GIL hold in the
    parent (a dataset writer flushing a video, say) could stall is no
    protection.
    """
    # Ignore SIGINT: a raw Ctrl-C could interrupt a send_mit() call
    # mid-transaction and leave a motor comm-faulted. Shutdown goes through
    # stop_event or SIGTERM instead.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if config.profile_log_path:
        # Separate process: attach its own FileHandler to the same file.
        root_logger = logging.getLogger()
        handler = logging.FileHandler(config.profile_log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        root_logger.addHandler(handler)
        # Importing lerobot adds an unrestricted console StreamHandler via an
        # implicit logging.basicConfig(); cap it at WARNING so per-tick
        # DEBUG logs only go to the file.
        for h in root_logger.handlers:
            if h is not handler and isinstance(h, logging.StreamHandler):
                h.setLevel(logging.WARNING)
        logger.setLevel(logging.DEBUG)

    # The setup connection's close() may not release the serial device
    # instantly; retry briefly instead of failing on "Device or resource busy".
    bus = None
    connect_attempts = 10
    for attempt in range(connect_attempts):
        try:
            if config.can_adapter == "damiao":
                bus = MotorBridgeController.from_dm_serial(
                    serial_port=config.port, baud=config.dm_serial_baud
                )
            elif config.can_adapter == "socketcan":
                bus = MotorBridgeController(channel=config.port)
            else:
                raise ValueError(
                    f"Unsupported can_adapter '{config.can_adapter}'. Use 'damiao' or 'socketcan'."
                )
            break
        except Exception as e:
            if attempt == connect_attempts - 1:
                error_queue.put(str(e))
                return
            time.sleep(0.2)

    try:
        motors = {
            name: bus.add_damiao_motor(send_id, recv_id, MOTOR_MODELS[name])
            for name, (send_id, recv_id) in config.motor_can_ids.items()
        }
        # Enabling here (not in configure()) starts each motor's comm-timeout
        # watchdog right before it starts receiving continuous traffic.
        for motor in motors.values():
            motor.enable()
    except Exception as e:
        error_queue.put(str(e))
        return

    ready_event.set()

    smoothers: dict[str, _SCurveAxis] = {}
    last_send_time: float | None = None
    prev_tick: float | None = None
    home_ramp: _HomeRamp | None = None
    home_started: float = 0.0
    home_requested = False
    use_mit = config.control_mode == "mit"
    interval = 1.0 / config.send_rate_hz
    gravity = _GravityCompensator(config, motor_names)
    home_kp, home_kd = _configured_gains(config, motor_names)
    thermal = ThermalMonitor(config, motor_names)
    eta_interval = 1.0 / config.temp_sample_hz
    next_eta_refresh = 0.0

    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            while True:
                try:
                    cmd = command_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd == "disable":
                    bus.disable_all()
                elif cmd == "enable":
                    for motor in motors.values():
                        motor.enable()
                elif cmd == "clear_error":
                    for motor in motors.values():
                        motor.clear_error()
                elif cmd == "home":
                    # Built below instead of here, so the ramp starts from the
                    # reading this tick is about to take rather than the last one.
                    home_requested = True

            for motor in motors.values():
                motor.request_feedback()
            try:
                bus.poll_feedback_once()
            except Exception:
                logger.warning("CAN bus poll feedback failed.")
            temps: dict[str, tuple[float, float] | None] = {}
            for i, name in enumerate(motor_names):
                state = motors[name].get_state()
                present_pos_shared[i] = math.degrees(state.pos) if state is not None else 0.0
                present_torq_shared[i] = state.torq if state is not None else 0.0
                present_vel_shared[i] = math.degrees(state.vel) if state is not None else 0.0
                # A motor that missed this tick keeps its last known
                # temperature rather than reporting 0C: a dropout is not a cold
                # motor, and feeding one to the fit would read as a steep cool.
                if state is None:
                    temps[name] = None
                else:
                    present_temp_mos_shared[i] = state.t_mos
                    present_temp_rotor_shared[i] = state.t_rotor
                    temps[name] = (state.t_mos, state.t_rotor)
            # From the angles just read, so the arm is carried at the pose it is
            # actually in rather than the one it was asked for.
            gravity_torques = gravity.torques(present_pos_shared)
            present_pos_ts.value = time.perf_counter()

            # Latches on the first motor to reach the shutdown threshold. The
            # parent does the disconnecting; this loop has to stay running to
            # carry the home ramp that disconnect() starts.
            if thermal.update(time.perf_counter(), temps) and not thermal_shutdown.is_set():
                thermal_shutdown.set()

            # Refitting all seven motors is far too costly to do every tick, and
            # the estimate cannot move faster than the history behind it anyway.
            if config.temp_debug and time.perf_counter() >= next_eta_refresh:
                next_eta_refresh = time.perf_counter() + eta_interval
                for i, name in enumerate(motor_names):
                    eta = thermal.seconds_to_shutdown(name)
                    present_temp_eta_shared[i] = _NO_ETA if eta is None else eta

            if home_requested:
                home_ramp = _HomeRamp(
                    config,
                    motor_names,
                    {name: present_pos_shared[i] for i, name in enumerate(motor_names)},
                )
                home_started = time.perf_counter()
                home_requested = False

            # Driven here rather than in go_home() so nothing holding the
            # caller's GIL can freeze the goal mid-travel.
            goal_pos: dict[str, float] | None = None
            homing_now = home_ramp is not None
            if home_ramp is not None:
                elapsed = time.perf_counter() - home_started
                target = home_ramp.target(elapsed)
                # Still published, so that when the ramp ends the shared goal
                # holds the pose the arm actually reached instead of the one
                # send_action() left there before homing started.
                for i, name in enumerate(motor_names):
                    goal_pos_shared[i] = target[name]
                goal_ready.value = 1
                # Dispatched from the ramp's own dict rather than read back out
                # of shared memory: while homing, no other process gets to put
                # a position anywhere near the motors.
                goal_pos = target
                if elapsed >= home_ramp.duration:
                    home_ramp = None
                    homing_flag.value = 0
                    home_done.set()
            elif goal_ready.value:
                goal_pos = {name: goal_pos_shared[i] for i, name in enumerate(motor_names)}

            if goal_pos is not None:
                if config.enable_trajectory_smoothing:
                    now = time.perf_counter()
                    dt = now - last_send_time if last_send_time is not None else 0.0
                    last_send_time = now
                    smoothed_pos: dict[str, float] = {}
                    smoothed_vel: dict[str, float] = {}
                    for name, target_deg in goal_pos.items():
                        if name not in smoothers:
                            idx = motor_names.index(name)
                            tau = config.smoothing_time_constant_s
                            smoothers[name] = _SCurveAxis(tau_s=tau[idx] if isinstance(tau, list) else tau)
                        pos, vel = smoothers[name].step(target_deg, dt)
                        smoothed_pos[name] = pos
                        smoothed_vel[name] = vel
                else:
                    smoothed_pos = goal_pos
                    smoothed_vel = dict.fromkeys(goal_pos, 0.0)

                for i, name in enumerate(motor_names):
                    motor = motors.get(name)
                    if motor is None:
                        continue
                    idx = i
                    pos_rad = math.radians(smoothed_pos[name])
                    vel_rad_s = math.radians(smoothed_vel[name])
                    tau = tau_ff_shared[i] + gravity_torques[i]
                    # The ramp drives on the configured gains, not the caller's.
                    # A softened joint (kp=0, as under gravity compensation)
                    # ignores its goal entirely, so homing on the shared gains
                    # moves only the joints that never used kp -- a FORCE_POS
                    # gripper -- and leaves the arm to drop when torque goes.
                    kp = home_kp[i] if homing_now else mit_kp_shared[i]
                    kd = home_kd[i] if homing_now else mit_kd_shared[i]
                    try:
                        if name == GRIPPER_MOTOR:
                            if config.gripper_control_mode == "mit":
                                motor.send_mit(pos_rad, vel_rad_s, kp, kd, tau)
                            else:
                                vel_deg_s = (
                                    config.pos_vel_velocity[idx]
                                    if isinstance(config.pos_vel_velocity, list)
                                    else config.pos_vel_velocity
                                )
                                motor.send_force_pos(pos_rad, math.radians(vel_deg_s), config.gripper_torque_ratio)
                        elif use_mit:
                            motor.send_mit(pos_rad, vel_rad_s, kp, kd, tau)
                        else:
                            vel_deg_s = (
                                config.pos_vel_velocity[idx]
                                if isinstance(config.pos_vel_velocity, list)
                                else config.pos_vel_velocity
                            )
                            motor.send_pos_vel(pos_rad, math.radians(vel_deg_s))
                    except Exception as e:
                        logger.warning(f"Follower process failed to send to {name}: {e}")
                    last_sent_shared[i] = smoothed_pos[name]
                has_sent_ever.value = 1

            if logger.isEnabledFor(logging.DEBUG) and prev_tick is not None:
                logger.debug(f"follower process tick interval: {(t0 - prev_tick) * 1e3:.1f}ms")
            prev_tick = t0

            elapsed = time.perf_counter() - t0
            stop_event.wait(max(0.0, interval - elapsed))
    finally:
        # A go_home() still waiting has nothing left to wait for; let it return
        # and find the arm wherever the ramp got to, rather than time out.
        homing_flag.value = 0
        home_done.set()
        for motor in motors.values():
            if config.disable_torque_on_disconnect:
                motor.disable()
            motor.clear_error()
            motor.close()
        bus.close()


class RebotB601Follower(Robot):
    """Seeed Studio reBot B601-DM follower arm (6-DOF + gripper, Damiao CAN motors).

    Motor communication is handled by the ``motorbridge`` package over a CAN bus,
    reached either through a Damiao serial bridge or a SocketCAN adapter.
    """

    config_class = RebotB601FollowerRobotConfig
    name = "rebot_b601_follower"

    def __init__(self, config: RebotB601FollowerRobotConfig):
        require_package("motorbridge", extra="rebot")
        super().__init__(config)
        self.config = config
        # Only set during the temporary setup connection in connect(); None
        # once ownership hands off to the follower process.
        self.bus: MotorBridgeController | None = None
        self.motors: dict = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self.cameras = make_cameras_from_configs(config.cameras)
        # Last successfully-read frame per camera key, used as a fallback on
        # read failure instead of crashing the recording session.
        self._last_camera_frames: dict[str, Any] = {}
        # Keys currently served from _last_camera_frames, so recovery logs once.
        self._stale_camera_keys: set[str] = set()
        # Consecutive-failure count per camera key, so a prolonged outage logs
        # on the 1st failure and every 30th, not at full loop rate.
        self._camera_failure_counts: dict[str, int] = {}
        # The follower process is the sole owner of the connection once
        # connect() hands off to it; its own interpreter/GIL means nothing
        # here can delay it.
        #
        # "spawn" starts the child from a clean interpreter instead of
        # forking, so it never inherits file descriptors opened here.
        self._mp_ctx = multiprocessing.get_context("spawn")
        n = len(self.motor_names)
        # lock=False: none of these need atomicity (a torn read on a
        # continuous position target is harmless), and a lock adds syscall
        # overhead and can leak a semaphore warning at shutdown.
        # Written by send_action(), read every tick by the follower process.
        self._goal_pos_shared = self._mp_ctx.Array("d", n, lock=False)
        # Set on send_action()'s first call; until then the array is all
        # zeros, which the follower process must not treat as a real target.
        self._goal_ready = self._mp_ctx.Value("b", 0, lock=False)
        # Written every tick by the follower process, read by _present_pos().
        self._present_pos_shared = self._mp_ctx.Array("d", n, lock=False)
        # Written every tick alongside present_pos_shared, read by _present_torq().
        self._present_torq_shared = self._mp_ctx.Array("d", n, lock=False)
        # Written every tick alongside present_pos_shared, read by _present_vel().
        self._present_vel_shared = self._mp_ctx.Array("d", n, lock=False)
        # MOSFET and rotor temperature (C), from the same feedback frame. Unlike
        # the others these hold their last good value through a dropout, since a
        # missing frame is not a cold motor.
        self._present_temp_mos_shared = self._mp_ctx.Array("d", n, lock=False)
        self._present_temp_rotor_shared = self._mp_ctx.Array("d", n, lock=False)
        # Seconds until each motor reaches temp_max_c, or _NO_ETA. Only written
        # when config.temp_debug is set.
        self._present_temp_eta_shared = self._mp_ctx.Array("d", n, lock=False)
        self._reset_temp_etas()
        self._present_pos_ts = self._mp_ctx.Value("d", 0.0, lock=False)
        # Smoothed position actually sent, read by send_action()'s return value.
        self._last_sent_shared = self._mp_ctx.Array("d", n, lock=False)
        self._has_sent_ever = self._mp_ctx.Value("b", 0, lock=False)
        # MIT gains and feedforward torque, read every tick alongside the goal.
        # Resolved to one entry per motor here, so the send loop is a plain
        # indexed read and set_mit_gains() can retune a live arm.
        self._mit_kp_shared = self._mp_ctx.Array("d", n, lock=False)
        self._mit_kd_shared = self._mp_ctx.Array("d", n, lock=False)
        self._tau_ff_shared = self._mp_ctx.Array("d", n, lock=False)
        self._reset_mit_gains()
        # Rare control messages; doesn't need shared-memory speed.
        self._command_queue = self._mp_ctx.Queue()
        self._ready_event = self._mp_ctx.Event()
        self._stop_event = self._mp_ctx.Event()
        self._error_queue = self._mp_ctx.Queue()
        # Set by the follower process when a home ramp finishes; go_home()'s
        # only way of knowing, since it no longer drives the ramp itself.
        self._home_done = self._mp_ctx.Event()
        # Raised by go_home() before it asks for a ramp and dropped by the
        # follower process when the ramp ends. While it is up send_action()
        # refuses to write, so nothing can steer the arm on its way home.
        self._homing_flag = self._mp_ctx.Value("b", 0, lock=False)
        # Set by the follower process's thermal monitor when a motor reaches
        # the shutdown threshold, and watched by _thermal_watch() below.
        self._thermal_shutdown = self._mp_ctx.Event()
        self._thermal_thread: threading.Thread | None = None
        self._thermal_thread_stop = threading.Event()
        self._follower_process: multiprocessing.process.BaseProcess | None = None
        # Parent-side mirror of the follower process's compensator, built on
        # first use by gravity_torques(). Stateless given the config, so the two
        # cannot disagree.
        self._gravity: _GravityCompensator | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _motors_torq_ft(self) -> dict[str, type]:
        return {f"{motor}.torq": float for motor in self.motor_names}

    @property
    def _motors_vel_ft(self) -> dict[str, type]:
        return {f"{motor}.vel": float for motor in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._motors_torq_ft, **self._motors_vel_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._follower_process is not None
            and self._follower_process.is_alive()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self} on {self.config.port} (adapter={self.config.can_adapter})...")
        # Temporary connection: only used for calibration and configure().
        # Closed below and handed off to the follower process, which reopens
        # its own.
        self._open_bus()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        self._close_bus()

        self._start_follower_process()
        self._start_thermal_watch()
        logger.info(f"{self} connected.")

    def _open_bus(self) -> None:
        """Open a direct connection to the motors and register them."""
        if self.config.can_adapter == "damiao":
            self.bus = MotorBridgeController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        elif self.config.can_adapter == "socketcan":
            self.bus = MotorBridgeController(channel=self.config.port)
        else:
            raise ValueError(
                f"Unsupported can_adapter '{self.config.can_adapter}'. Use 'damiao' or 'socketcan'."
            )

        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            self.motors[motor_name] = self.bus.add_damiao_motor(send_id, recv_id, MOTOR_MODELS[motor_name])

    def _close_bus(self) -> None:
        """Release the direct connection, leaving the device free to reopen.

        Each motor is closed before the bus: skipping that can leave the serial
        device open, so the next open fails with "Device or resource busy". The
        short sleep gives the OS time to release it -- the follower process also
        retries on its own.
        """
        for motor in self.motors.values():
            with contextlib.suppress(Exception):
                motor.close()
        if self.bus is not None:
            self.bus.close()
        self.bus = None
        self.motors = {}
        time.sleep(0.3)

    @contextlib.contextmanager
    def _direct_connection(self):
        """Own the motors directly for the duration of the block.

        connect() already holds such a connection while it calibrates and
        configures, so this is a no-op there. Anything called *after* connect()
        has to take the device back from the follower process, which owns it by
        then -- lerobot-calibrate does exactly that: connect(calibrate=False)
        followed by calibrate().
        """
        if self.bus is not None:
            yield
            return

        running = self._follower_process is not None and self._follower_process.is_alive()
        if running:
            self._stop_follower_process()
            time.sleep(0.3)
        try:
            self._open_bus()
            yield
        finally:
            self._close_bus()
            if running:
                self._start_follower_process()

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        # Zeroing writes to the motors directly, so this needs the connection
        # back from the follower process when called on an already-connected
        # robot (lerobot-calibrate's connect-then-calibrate order).
        with self._direct_connection():
            self.bus.disable_all()
            print(
                "\nCalibration: set zero position.\n"
                "Manually move the reBot B601 to its ZERO POSITION and close the gripper.\n"
                "See the B601 manual for the zero pose (the default sit-down position).\n"
            )
            input("Press ENTER when ready...")

            for motor in self.motors.values():
                motor.set_zero_position()
                time.sleep(_ZERO_SETTLE_SEC)
            logger.info("Arm zero position set.")

        self.calibration = {}
        for motor_name, (send_id, _recv_id) in self.config.motor_can_ids.items():
            range_min, range_max = self.config.joint_limits[motor_name]
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        if self.config.control_mode not in ("pos_vel", "mit"):
            raise ValueError(
                f"Unsupported control_mode '{self.config.control_mode}'. Use 'pos_vel' or 'mit'."
            )
        if self.config.gripper_control_mode not in ("force_pos", "mit"):
            raise ValueError(
                f"Unsupported gripper_control_mode '{self.config.gripper_control_mode}'. "
                "Use 'force_pos' or 'mit'."
            )
        use_mit = self.config.control_mode == "mit"
        gripper_use_mit = self.config.gripper_control_mode == "mit"

        # Clear any latched comm-timeout fault before touching mode or enable
        # state, or a previously-faulted motor won't come back up from
        # enable() alone.
        for motor in self.motors.values():
            motor.clear_error()

        # Verify/set every motor's mode before enabling any of them: enabling
        # makes a motor expect continuous commands immediately, and checking
        # motors one at a time would leave earlier-enabled ones idle long
        # enough to fault.
        for motor_name, motor in self.motors.items():
            if motor_name == GRIPPER_MOTOR:
                target_mode = MotorBridgeMode.MIT if gripper_use_mit else MotorBridgeMode.FORCE_POS
            elif use_mit:
                target_mode = MotorBridgeMode.MIT
            else:
                target_mode = MotorBridgeMode.POS_VEL
            for attempt in range(_ENSURE_MODE_RETRIES + 1):
                try:
                    motor.ensure_mode(target_mode)
                    break
                except Exception:
                    if attempt == _ENSURE_MODE_RETRIES:
                        raise
                    time.sleep(_SETTLE_SEC)
            logger.debug(f"{motor_name} mode set to {target_mode}")

        # Not enabling here: this connection closes right after, and the
        # follower process reopens its own before sending any traffic.
        # Enabling here would start the comm-timeout watchdog during that
        # gap and fault the motor. _follower_process_main() enables instead,
        # right before its send loop.

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand.

        Also call before any pause with no continuous commands, since an
        enabled-but-uncommanded motor faults on its own comm-timeout. Just
        queues the command; the follower process keeps looping regardless.
        """
        self._command_queue.put("disable")
        logger.info(f"{self} torque disabled.")

    @check_if_not_connected
    def enable_torque(self) -> None:
        """Re-enable torque after disable_torque(), without re-verifying mode.

        Clears any latched fault first: a motor that faulted while disabled
        won't come back from enable() alone.
        """
        self._command_queue.put("clear_error")
        self._command_queue.put("enable")
        logger.info(f"{self} torque enabled.")

    def _write_per_motor(self, shared, value: float | list[float] | dict[str, float]) -> None:
        """Write a scalar / motor-ordered list / name-keyed dict into a shared
        array. A dict updates only the motors it names, leaving the others as
        they are, so a caller can retune one joint without restating the rest.
        """
        if isinstance(value, dict):
            unknown = set(value) - set(self.motor_names)
            if unknown:
                raise ValueError(f"{self}: unknown motor(s) {sorted(unknown)}")
            for i, name in enumerate(self.motor_names):
                if name in value:
                    shared[i] = float(value[name])
            return
        if isinstance(value, (list, tuple)):
            if len(value) != len(self.motor_names):
                raise ValueError(f"{self}: expected {len(self.motor_names)} values, got {len(value)}")
            for i, v in enumerate(value):
                shared[i] = float(v)
            return
        for i in range(len(self.motor_names)):
            shared[i] = float(value)

    def _reset_mit_gains(self) -> None:
        """Load the configured MIT gains into shared memory, the gripper's own
        pair included, so the send loop never has to consult the config."""
        self._write_per_motor(self._mit_kp_shared, self.config.mit_kp)
        self._write_per_motor(self._mit_kd_shared, self.config.mit_kd)
        self._write_per_motor(self._mit_kp_shared, {GRIPPER_MOTOR: self.config.gripper_mit_kp})
        self._write_per_motor(self._mit_kd_shared, {GRIPPER_MOTOR: self.config.gripper_mit_kd})

    def set_mit_gains(
        self,
        kp: float | list[float] | dict[str, float] | None = None,
        kd: float | list[float] | dict[str, float] | None = None,
    ) -> None:
        """Retune the MIT gains the follower process sends, without a reconnect.

        Takes effect on the process's next tick. Passing neither restores the
        configured gains, so a caller that lowered them can always put them
        back. Only meaningful under control_mode="mit" (and, for the gripper,
        gripper_control_mode="mit"); POS_VEL and FORCE_POS ignore these.

        Dropping kp to 0 leaves a joint carried by kd damping and whatever
        set_torque_feedforward() supplies -- that is gravity compensation.
        """
        if kp is None and kd is None:
            self._reset_mit_gains()
            logger.info(f"{self} MIT gains restored to their configured values.")
            return
        if kp is not None:
            self._write_per_motor(self._mit_kp_shared, kp)
        if kd is not None:
            self._write_per_motor(self._mit_kd_shared, kd)

    def gravity_torques(self) -> dict[str, float]:
        """The holding torque the follower process is adding right now, by motor.

        Recomputed here from the same config and the latest reading, rather than
        read back from the process, so a caller sees exactly what is being
        applied without another shared array. All zeros when compensation is off.
        """
        if self._gravity is None:
            self._gravity = _GravityCompensator(self.config, self.motor_names)
        present = self._present_pos()
        torques = self._gravity.torques([present[name] for name in self.motor_names])
        return dict(zip(self.motor_names, torques, strict=True))

    def get_mit_gains(self) -> tuple[dict[str, float], dict[str, float]]:
        """The (kp, kd) the follower process is currently sending, by motor
        name. Read before lowering them to have something to restore to."""
        return (
            {name: self._mit_kp_shared[i] for i, name in enumerate(self.motor_names)},
            {name: self._mit_kd_shared[i] for i, name in enumerate(self.motor_names)},
        )

    def set_torque_feedforward(self, torque: dict[str, float] | None = None) -> None:
        """Set the feedforward torque (N.m) added to every MIT command, by
        motor name. Passing nothing clears it back to zero.

        Like send_action(), this only updates shared memory: the follower
        process keeps applying the last value at its own fixed rate, so a
        stall in the caller's loop holds the arm rather than dropping it.
        Motors named here keep their previous feedforward, so a caller can
        drive one joint without restating the rest.
        """
        self._write_per_motor(self._tau_ff_shared, 0.0 if torque is None else torque)

    def _present_pos(self) -> dict[str, float]:
        """Read present joint positions in degrees from the follower
        process's latest reading, not a direct hardware read. Falls back to
        0.0 per joint before the follower process's first tick."""
        if self._present_pos_ts.value == 0.0:
            return dict.fromkeys(self.motor_names, 0.0)
        return {name: self._present_pos_shared[i] for i, name in enumerate(self.motor_names)}

    def _present_torq(self) -> dict[str, float]:
        """Read present joint torque (Nm) from the follower process's latest
        reading. Same source tick as _present_pos(): the CAN feedback frame
        motorbridge's get_state() already polls every tick reports pos, vel,
        and torq together. Falls back to 0.0 per joint before the follower
        process's first tick."""
        if self._present_pos_ts.value == 0.0:
            return dict.fromkeys(self.motor_names, 0.0)
        return {name: self._present_torq_shared[i] for i, name in enumerate(self.motor_names)}

    def _present_vel(self) -> dict[str, float]:
        """Read present joint velocity (deg/s) from the follower process's
        latest reading. Same source tick as _present_pos()/_present_torq().
        Falls back to 0.0 per joint before the follower process's first
        tick."""
        if self._present_pos_ts.value == 0.0:
            return dict.fromkeys(self.motor_names, 0.0)
        return {name: self._present_vel_shared[i] for i, name in enumerate(self.motor_names)}

    def motor_temperatures(self) -> dict[str, dict[str, float]]:
        """Present MOSFET and rotor temperature (C) per motor, from the
        follower process's latest reading.

        Same source tick as _present_pos(): every Damiao feedback frame carries
        both temperatures alongside position, so reading them costs no extra
        bus traffic. Falls back to 0.0 before the follower process's first tick.
        """
        if self._present_pos_ts.value == 0.0:
            return {name: dict.fromkeys(COMPONENTS, 0.0) for name in self.motor_names}
        return {
            name: {
                "mosfet": self._present_temp_mos_shared[i],
                "rotor": self._present_temp_rotor_shared[i],
            }
            for i, name in enumerate(self.motor_names)
        }

    def _reset_temp_etas(self) -> None:
        """Clear the published estimates, so a fresh follower process is never
        read as reporting the previous one's."""
        for i in range(len(self.motor_names)):
            self._present_temp_eta_shared[i] = _NO_ETA

    def motor_overheat_etas(self) -> dict[str, float | None]:
        """Estimated seconds until each motor reaches temp_max_c, or None where
        the history does not support an estimate (too little of it, or a motor
        that is steady or cooling).

        All None unless the robot was configured with temp_debug=True: the fit
        is not cheap enough to run at loop rate for nobody.
        """
        return {
            name: (None if self._present_temp_eta_shared[i] < 0.0 else self._present_temp_eta_shared[i])
            for i, name in enumerate(self.motor_names)
        }

    def _start_thermal_watch(self) -> None:
        """Watch for the follower process's shutdown request in a thread, so an
        overheating arm comes down on its own schedule rather than whenever the
        caller next happens to call into this object."""
        if not self.config.temp_shutdown_enabled:
            return
        self._thermal_shutdown.clear()
        self._thermal_thread_stop.clear()
        self._thermal_thread = threading.Thread(
            target=self._thermal_watch, name=f"{self}_thermal_watch", daemon=True
        )
        self._thermal_thread.start()

    def _thermal_watch(self) -> None:
        while not self._thermal_thread_stop.is_set():
            if not self._thermal_shutdown.wait(0.2):
                continue
            logger.error(f"{self}: motor over maximum operating temperature; homing and disconnecting.")
            try:
                self.disconnect()
            except Exception as e:
                logger.error(f"{self}: thermal shutdown failed to disconnect cleanly: {e}")
            return

    def _stop_thermal_watch(self) -> None:
        self._thermal_thread_stop.set()
        thread = self._thermal_thread
        # A thermal shutdown reaches here from inside the watch thread itself,
        # which cannot join itself.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thermal_thread = None

    def _read_camera_or_last(self, cache_key: str, read_fn) -> Any:
        """Call `read_fn()`, falling back to the last successfully-read frame
        for `cache_key` on failure instead of crashing. Logs on the 1st
        failure and every 30th thereafter, plus once when reads resume.
        Re-raises if no frame has ever been read for this key."""
        try:
            frame = read_fn()
        except Exception as e:
            cached = self._last_camera_frames.get(cache_key)
            if cached is None:
                raise
            count = self._camera_failure_counts.get(cache_key, 0) + 1
            self._camera_failure_counts[cache_key] = count
            if count == 1 or count % 30 == 0:
                logger.warning(
                    f"{self}: {cache_key} frame not read ({count} consecutive failure(s)): {e}; "
                    "reusing last known frame."
                )
            self._stale_camera_keys.add(cache_key)
            return cached

        if cache_key in self._stale_camera_keys:
            logger.info(f"{self}: {cache_key} frame reads have resumed.")
            self._stale_camera_keys.discard(cache_key)
        self._camera_failure_counts[cache_key] = 0
        self._last_camera_frames[cache_key] = frame
        return frame

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs_dict = {f"{motor}.pos": pos for motor, pos in self._present_pos().items()}
        obs_dict.update({f"{motor}.torq": torq for motor, torq in self._present_torq().items()})
        obs_dict.update({f"{motor}.vel": vel for motor, vel in self._present_vel().items()})
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = self._read_camera_or_last(cam_key, cam.read_latest)
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = self._read_camera_or_last(
                    f"{cam_key}_depth", cam.read_latest_depth
                )
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Update the arm's target joint configuration (degrees). The relative
        action magnitude may be clipped depending on `max_relative_target`.

        Only writes the shared target; the follower process dispatches it at
        its own fixed rate, so a stall in the caller's loop doesn't stall the
        motor. Returns what the follower process actually sent, falling back
        to the raw target before it has ticked yet.

        Ignored entirely while the arm is homing: the ramp owns the arm until
        it finishes, so a teleop loop that keeps calling cannot pull it off
        course. The return value still reports what is being sent.
        """
        if self._homing_flag.value:
            logger.debug(f"{self}: ignoring action while homing.")
            return {f"{name}.pos": self._last_sent_shared[i] for i, name in enumerate(self.motor_names)}

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Clip against soft joint limits.
        for motor_name in list(goal_pos):
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped = max(min_limit, min(max_limit, goal_pos[motor_name]))
                if clipped != goal_pos[motor_name]:
                    logger.debug(f"Clipped {motor_name} from {goal_pos[motor_name]:.2f} to {clipped:.2f}")
                goal_pos[motor_name] = clipped

        # Tolerate 6-DOF leaders with no wrist_yaw joint by holding it at zero
        # (e.g. so100_leader/so101_leader teleoperating this 7-DOF follower).
        if "wrist_yaw" not in goal_pos:
            goal_pos["wrist_yaw"] = 0.0

        # Cap relative target when too far from the present position.
        if self.config.max_relative_target is not None:
            present_pos = self._present_pos()
            goal_present_pos = {key: (g, present_pos.get(key, g)) for key, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        for i, name in enumerate(self.motor_names):
            if name in goal_pos:
                self._goal_pos_shared[i] = goal_pos[name]
        self._goal_ready.value = 1

        if self._has_sent_ever.value:
            last_sent = {name: self._last_sent_shared[i] for i, name in enumerate(self.motor_names)}
        else:
            last_sent = goal_pos
        return {f"{motor}.pos": val for motor, val in last_sent.items()}

    def _start_follower_process(self) -> None:
        self._stop_event.clear()
        self._ready_event.clear()
        self._goal_ready.value = 0
        self._has_sent_ever.value = 0
        self._homing_flag.value = 0
        self._reset_temp_etas()
        self._follower_process = self._mp_ctx.Process(
            target=_follower_process_main,
            args=(
                self.config,
                self.motor_names,
                self._goal_pos_shared,
                self._goal_ready,
                self._present_pos_shared,
                self._present_torq_shared,
                self._present_vel_shared,
                self._present_temp_mos_shared,
                self._present_temp_rotor_shared,
                self._present_temp_eta_shared,
                self._present_pos_ts,
                self._last_sent_shared,
                self._has_sent_ever,
                self._mit_kp_shared,
                self._mit_kd_shared,
                self._tau_ff_shared,
                self._command_queue,
                self._ready_event,
                self._stop_event,
                self._error_queue,
                self._home_done,
                self._homing_flag,
                self._thermal_shutdown,
            ),
            name=f"{self}_follower_process",
            daemon=True,
        )
        self._follower_process.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._ready_event.is_set():
                return
            if not self._error_queue.empty():
                raise ConnectionError(f"{self}: follower process failed to start: {self._error_queue.get()}")
            if not self._follower_process.is_alive():
                raise ConnectionError(f"{self}: follower process exited unexpectedly during startup.")
            time.sleep(0.05)
        raise ConnectionError(f"{self}: follower process did not become ready within 10s.")

    def _stop_follower_process(self) -> None:
        self._stop_event.set()
        if self._follower_process is not None:
            self._follower_process.join(timeout=5.0)
            if self._follower_process.is_alive():
                logger.warning(f"{self} follower process did not terminate within timeout; terminating.")
                self._follower_process.terminate()
                self._follower_process.join(timeout=2.0)
        self._follower_process = None
        # The only reader is gone, so any command still in flight will never be
        # read. Left alone, the queue's feeder thread keeps it and the join that
        # thread gets at interpreter exit never returns -- a hang after the work
        # is done. Unsent commands are moot once the process is stopped.
        self._command_queue.cancel_join_thread()

    def go_home(self) -> None:
        """Walk the arm back to its calibration zero pose (see _HomeRamp for
        the path it takes) and return once it has settled there.

        Public capability: callers that home between episodes (recording, eval)
        detect it by this method's presence, and it must remain the same ramp
        disconnect() uses so the two can't drift.

        The ramp runs in the follower process, for the same reason the send
        loop does. Driven from here it is a 100 Hz python loop, so anything
        that holds this process's GIL for a moment -- the dataset writer
        flushing a video file between episodes, say -- freezes the goal
        mid-travel and then jumps it, and the arm follows that as a jerk. This
        side only waits.
        """
        # Checked before the put, not just in the wait loop below, so a call
        # with no process to serve it fails saying so rather than queueing a
        # command nothing will read and waiting out the loop.
        if self._follower_process is None or not self._follower_process.is_alive():
            raise ConnectionError(f"{self}: follower process is not running; cannot home.")

        self._home_done.clear()
        # Raised here rather than in the follower process so that no
        # send_action() can land in the gap between asking for the ramp and the
        # process picking the request up. The follower process drops it again
        # when the ramp ends.
        self._homing_flag.value = 1
        self._command_queue.put("home")

        # The ramp keeps its own schedule, so overrunning this means the
        # follower process is gone or wedged, not that homing is taking longer.
        deadline = time.monotonic() + (
            self.config.home_duration_s + self.config.gripper_close_duration_s + _GRIPPER_SETTLE_SEC + 5.0
        )
        try:
            while not self._home_done.wait(0.05):
                if self._follower_process is None or not self._follower_process.is_alive():
                    raise ConnectionError(f"{self}: follower process stopped while homing.")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self}: homing did not finish; the arm is part-way home.")
        except (ConnectionError, TimeoutError):
            # Homing is over either way; leaving the flag up would wedge
            # send_action() for the rest of the session.
            self._homing_flag.value = 0
            raise

    @check_if_not_connected
    def disconnect(self) -> None:
        # Stopped before homing: the request has been acted on, and the watcher
        # has nothing left to do while the arm ramps down.
        self._stop_thermal_watch()

        if self.config.return_home_on_disconnect:
            try:
                self.go_home()
            except (ConnectionError, TimeoutError) as e:
                # Leaving the arm part-way home is bad; leaving it powered and
                # holding position on a session that is over is worse, and the
                # rest of this method is what drops torque.
                logger.warning(f"{self}: homing failed on disconnect: {e}")

        # Stop the follower process first: it owns the other hardware connection.
        self._stop_follower_process()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
