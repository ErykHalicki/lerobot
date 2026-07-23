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

import logging
import math
import multiprocessing
import os
import queue
import signal
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_follower import RebotB601FollowerRobotConfig

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)

# Joint controlled in FORCE_POS mode; every other joint runs in POS_VEL mode.
GRIPPER_MOTOR = "gripper"
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
_ENSURE_MODE_RETRIES = 9
_SETTLE_SEC = 0.01
_ZERO_SETTLE_SEC = 0.1


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


def _follower_process_main(
    config: RebotB601FollowerRobotConfig,
    motor_names: list[str],
    goal_pos_shared,
    goal_ready,
    present_pos_shared,
    present_pos_ts,
    last_sent_shared,
    has_sent_ever,
    command_queue: multiprocessing.Queue,
    ready_event,
    stop_event,
    error_queue: multiprocessing.Queue,
) -> None:
    """Sole owner of the motor connection: runs the read/smooth/send loop at
    config.send_rate_hz in its own process, so nothing in the caller's
    process can delay it by holding the GIL.

    Communicates via shared memory only: `goal_pos_shared` (target),
    `present_pos_shared`/`present_pos_ts` (latest reading), `last_sent_shared`
    (what was actually sent). `command_queue` carries disable/enable/clear_error.
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
    use_mit = config.control_mode == "mit"
    interval = 1.0 / config.send_rate_hz

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

            for motor in motors.values():
                motor.request_feedback()
            try:
                bus.poll_feedback_once()
            except Exception:
                logger.warning("CAN bus poll feedback failed.")
            for i, name in enumerate(motor_names):
                state = motors[name].get_state()
                present_pos_shared[i] = math.degrees(state.pos) if state is not None else 0.0
            present_pos_ts.value = time.perf_counter()

            if goal_ready.value:
                goal_pos = {name: goal_pos_shared[i] for i, name in enumerate(motor_names)}

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
                    idx = motor_names.index(name)
                    pos_rad = math.radians(smoothed_pos[name])
                    vel_rad_s = math.radians(smoothed_vel[name])
                    try:
                        if name == GRIPPER_MOTOR:
                            if config.gripper_control_mode == "mit":
                                motor.send_mit(pos_rad, vel_rad_s, config.gripper_mit_kp, config.gripper_mit_kd, 0.0)
                            else:
                                vel_deg_s = (
                                    config.pos_vel_velocity[idx]
                                    if isinstance(config.pos_vel_velocity, list)
                                    else config.pos_vel_velocity
                                )
                                motor.send_force_pos(pos_rad, math.radians(vel_deg_s), config.gripper_torque_ratio)
                        elif use_mit:
                            kp = config.mit_kp[idx] if isinstance(config.mit_kp, list) else config.mit_kp
                            kd = config.mit_kd[idx] if isinstance(config.mit_kd, list) else config.mit_kd
                            motor.send_mit(pos_rad, vel_rad_s, kp, kd, 0.0)
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
        self._present_pos_ts = self._mp_ctx.Value("d", 0.0, lock=False)
        # Smoothed position actually sent, read by send_action()'s return value.
        self._last_sent_shared = self._mp_ctx.Array("d", n, lock=False)
        self._has_sent_ever = self._mp_ctx.Value("b", 0, lock=False)
        # Rare control messages; doesn't need shared-memory speed.
        self._command_queue = self._mp_ctx.Queue()
        self._ready_event = self._mp_ctx.Event()
        self._stop_event = self._mp_ctx.Event()
        self._error_queue = self._mp_ctx.Queue()
        self._follower_process: multiprocessing.process.BaseProcess | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

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
        return {**self._motors_ft, **self._cameras_ft}

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

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        # Close each motor before the bus: skipping this can leave the serial
        # device open, so the follower process's own open fails with "Device
        # or resource busy".
        for motor in self.motors.values():
            motor.close()
        self.bus.close()
        self.bus = None
        self.motors = {}
        # Short settle for the OS to release the serial device; the follower
        # process also retries on its own.
        time.sleep(0.3)

        self._start_follower_process()
        logger.info(f"{self} connected.")

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

    def _present_pos(self) -> dict[str, float]:
        """Read present joint positions in degrees from the follower
        process's latest reading, not a direct hardware read. Falls back to
        0.0 per joint before the follower process's first tick."""
        if self._present_pos_ts.value == 0.0:
            return dict.fromkeys(self.motor_names, 0.0)
        return {name: self._present_pos_shared[i] for i, name in enumerate(self.motor_names)}

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
        """
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
        self._follower_process = self._mp_ctx.Process(
            target=_follower_process_main,
            args=(
                self.config,
                self.motor_names,
                self._goal_pos_shared,
                self._goal_ready,
                self._present_pos_shared,
                self._present_pos_ts,
                self._last_sent_shared,
                self._has_sent_ever,
                self._command_queue,
                self._ready_event,
                self._stop_event,
                self._error_queue,
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

    def pin_follower_process_to_cores(self, cores: set[int]) -> None:
        """Best-effort: pin the follower process to the given CPU cores.
        No-op if it isn't running or CPU affinity isn't supported."""
        if self._follower_process is None or self._follower_process.pid is None:
            return
        try:
            os.sched_setaffinity(self._follower_process.pid, cores)
        except (AttributeError, OSError) as e:
            logger.warning(f"{self}: could not set follower process CPU affinity to {cores}: {e}")

    def _go_home(self) -> None:
        """Ramp every joint to 0° (the calibration zero pose) over
        `home_duration_s`, independent of starting distance: each tick's
        target is an eased interpolation between the starting position and
        0°, not 0° itself, so a joint that starts far from home doesn't move
        any faster than one that starts close."""
        start = self._present_pos()
        duration = self.config.home_duration_s
        tick = 1.0 / self.config.send_rate_hz
        t0 = time.monotonic()
        while True:
            frac = min(1.0, (time.monotonic() - t0) / duration)
            eased = frac * frac * (3.0 - 2.0 * frac)  # smoothstep: zero velocity at both ends
            self.send_action({f"{name}.pos": start[name] * (1.0 - eased) for name in self.motor_names})
            if frac >= 1.0:
                return
            time.sleep(tick)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.config.return_home_on_disconnect:
            self._go_home()

        # Stop the follower process first: it owns the other hardware connection.
        self._stop_follower_process()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
