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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class RebotB601FollowerConfig:
    """Base configuration class for the Seeed Studio reBot B601-DM follower arm.

    The B601-DM is a 6-DOF arm plus gripper driven by Damiao CAN motors. Motor
    communication goes through the ``motorbridge`` package.
    """

    # Communication port. For ``can_adapter="damiao"`` this is the Damiao serial
    # bridge device (e.g. "/dev/ttyACM0"); for ``can_adapter="socketcan"`` it is
    # the CAN channel name (e.g. "can0").
    port: str

    # CAN adapter type:
    #   "damiao"    - Damiao dedicated serial bridge (default)
    #   "socketcan" - SocketCAN based adapters (PCAN, slcan, embedded controllers, ...)
    can_adapter: str = "damiao"

    # Baud rate for the Damiao serial bridge (only used when can_adapter="damiao").
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # On disconnect(), ramp every joint back to 0° (the calibration zero
    # pose) before stopping the follower process.
    return_home_on_disconnect: bool = True

    # Time to spend ramping to the home position in disconnect(), regardless
    # of starting distance. The gripper opens fully during this ramp instead
    # of moving to 0° with the other joints, so it can't be gripping anything
    # while the arm moves.
    home_duration_s: float = 1.67

    # Time to spend closing the gripper to 0° after home_duration_s, once
    # every other joint (and the gripper itself, fully open) has arrived. The
    # wrist lowers from home_wrist_flex_deg to 0° over this same span.
    gripper_close_duration_s: float = 0.67

    # Where to hold wrist_flex while the arm ramps home, in degrees; negative
    # points the gripper up. Keeps whatever is on the end clear of the shoulder
    # on the way in, then lowers to 0° over gripper_close_duration_s rather than
    # swinging down at ramp speed. 0.0 homes the wrist with everything else.
    home_wrist_flex_deg: float = -45.0

    # `max_relative_target` limits the magnitude of the relative positional target
    # vector for safety purposes (in degrees). Set to a positive scalar to apply the
    # same value to all motors, or to a dict mapping motor names to per-motor values.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps motor names to their (send_can_id, recv_can_id) pair.
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )

    # Max speed (deg/s) per joint for POS_VEL arms and FORCE_POS gripper (motor order).
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 900.0]
    )

    # Arm control: "mit" or "pos_vel".
    control_mode: str = "mit"

    # MIT kp/kd per arm joint (motor order). Unused when control_mode="pos_vel".
    mit_kp: float | list[float] = field(default_factory=lambda: [45.0, 45.0, 45.0, 8.0, 9.0, 8.0, 8.0])
    mit_kd: float | list[float] = field(default_factory=lambda: [12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0])

    # Add tau = g(q) to every MIT command, from the angles the follower process
    # already reads each tick. Without it a joint sits g(q)/kp below target to
    # make its holding torque: a standing teleoperation error, and a gap between
    # a recorded action and the state it produced. MIT only. Assumes motor zeros
    # at the URDF zero pose, where lerobot-calibrate homes.
    gravity_compensation: bool = True

    # Trim for what a rigid-body model cannot know: joint friction and cable
    # drag. `gravity_gain` scales every joint, `gravity_scale` one of them, e.g.
    # {"shoulder_lift": 1.1}. Both are per-rig, so neither is set by default.
    gravity_gain: float = 1.0
    gravity_scale: dict[str, float] = field(default_factory=dict)

    # Mass (kg) carried at the end effector beyond the modelled gripper, and
    # where it sits in the end effector's frame.
    gravity_payload_kg: float = 0.0
    gravity_payload_com: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Ceiling on the gravity term alone, N.m, applied on top of each motor's
    # rated effort -- so it only binds on the three 27 N.m joints. shoulder_lift
    # needs 15.5 N.m at full reach; a lower ceiling makes it sag, not safer.
    gravity_max_torque: float = 20.0

    # Thermal protection, on the MOSFET and rotor temperatures every CAN
    # feedback frame already carries. Level 1 and 2 warn on the terminal;
    # level 3 homes the arm and disconnects it.
    temp_warn_c: float = 57.0
    temp_danger_c: float = 61.0
    temp_max_c: float = 65.0

    # How often a motor sitting at level 1 / level 2 warns again, in seconds.
    # Crossing into a higher level always warns immediately.
    temp_warn_repeat_s: float = 3.0
    temp_danger_repeat_s: float = 1.0

    # Temperature history behind the quadratic fit that estimates time to
    # shutdown: how long a window to keep, and how often to sample it.
    temp_history_s: float = 60.0
    temp_sample_hz: float = 10.0

    # Home and disconnect on reaching temp_max_c. Turning this off leaves the
    # warnings in place but never shuts the arm down on its own.
    temp_shutdown_enabled: bool = True

    # Publish the per-motor time-to-overheat estimate for
    # motor_overheat_etas(). Off by default: refitting every motor costs about
    # an eighth of a core at loop rate, and nothing needs it unless something
    # is watching.
    temp_debug: bool = False

    # Gripper control: "force_pos" or "mit".
    gripper_control_mode: str = "force_pos"

    # FORCE_POS only: max grip force, in [0, 1].
    gripper_torque_ratio: float = 0.07

    # MIT only.
    gripper_mit_kp: float = 8.0
    gripper_mit_kd: float = 0.3

    # send_action() only updates the target; a separate follower process
    # commands the motors at this fixed rate, independent of how often or
    # irregularly the caller calls send_action().
    send_rate_hz: float = 100.0

    # If set, the follower process attaches its own FileHandler to this path
    # and enables DEBUG on its own logger, since it's a separate process and
    # can't share the caller's logging handlers directly.
    profile_log_path: str | None = None

    # S-curve smoothing applied to the goal position before dispatch. Without
    # it, a raw teleop position is forwarded every send tick regardless of how
    # far it moved, audible as jerk on MIT-mode joints. Disable for direct
    # motor testing with enable_trajectory_smoothing=False.
    enable_trajectory_smoothing: bool = True

    # Per-joint smoothing time constant (motor order), in seconds: roughly how
    # long the smoothed position takes to catch up to a step change in
    # target. Must be meaningfully larger than the tick interval, or the
    # per-tick gain (min(1, dt/tau)) degrades toward pass-through.
    smoothing_time_constant_s: float | list[float] = 0.04

    # Soft joint limits (degrees). These are clipped against on every action.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-150.0, 150.0),
            "shoulder_lift": (-200.0, 1.0),
            "elbow_flex": (-200.0, 1.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )


@RobotConfig.register_subclass("rebot_b601_follower")
@dataclass
class RebotB601FollowerRobotConfig(RobotConfig, RebotB601FollowerConfig):
    """Registered configuration for the reBot B601-DM follower robot."""

    pass
