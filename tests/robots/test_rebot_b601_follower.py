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

import math
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower, BiRebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower import (
    RebotB601Follower,
    RebotB601FollowerConfig,
    RebotB601FollowerRobotConfig,
)
from lerobot.robots.rebot_b601_follower.rebot_b601_follower import (
    GRIPPER_MOTOR,
    WRIST_MOTOR,
    _follower_process_main,
    _HomeRamp,
)

_MODULE = "lerobot.robots.rebot_b601_follower.rebot_b601_follower"


def _make_motor_mock(position_rad: float = 0.0) -> MagicMock:
    motor = MagicMock(name="MotorMock")
    state = MagicMock()
    state.pos = position_rad
    state.vel = 0.0
    state.torq = 0.0
    motor.get_state.return_value = state
    return motor


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="MotorBridgeControllerMock")
    # add_damiao_motor returns a fresh motor mock; position encodes the call order.
    bus._motor_count = 0

    def _add_motor(_send_id, _recv_id, _model):
        bus._motor_count += 1
        return _make_motor_mock(position_rad=math.radians(bus._motor_count))

    bus.add_damiao_motor.side_effect = _add_motor
    return bus


@pytest.fixture
def follower():
    """A robot that believes it is connected, with no follower process behind it.

    connect() hands the motors off to a separate process, which respawns a fresh
    interpreter and so cannot see these mocks. The parent-side methods under test
    (get_observation, send_action) only touch the shared arrays, so the process is
    stubbed out and the shared state written directly. Homing is off: it needs a
    live process to serve it, and _HomeRamp is tested on its own below.
    """
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch.object(RebotB601Follower, "_start_follower_process", autospec=True) as start,
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = _make_bus_mock()

        def _fake_start(self):
            # is_alive() tracks the stop event, so disconnect() sees it go away
            # instead of reporting a process that would not terminate.
            self._follower_process = MagicMock(is_alive=lambda: not self._stop_event.is_set())

        start.side_effect = _fake_start
        cfg = RebotB601FollowerRobotConfig(port="/dev/null", return_home_on_disconnect=False)
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def _seed_present(robot, pos_deg: float = 0.0, torq: float = 0.0, vel_deg_s: float = 0.0) -> None:
    """Fill in a follower-process reading, including the timestamp the
    parent-side accessors use to tell "no tick yet" from a real reading."""
    for i in range(len(robot.motor_names)):
        robot._present_pos_shared[i] = pos_deg * (i + 1)
        robot._present_torq_shared[i] = torq * (i + 1)
        robot._present_vel_shared[i] = vel_deg_s * (i + 1)
    robot._present_pos_ts.value = time.perf_counter()


def _run_one_tick(config, goal_deg: dict[str, float]) -> dict[str, MagicMock]:
    """Run the follower loop in a thread until it has sent `goal_deg` once, and
    return its motor mocks so the caller can assert on how they were driven.

    A thread rather than a process so the mocks apply; signal.signal is patched
    because it only works on the main thread.
    """
    bus_mock = _make_bus_mock()
    robot = RebotB601Follower(config)
    motor_names = robot.motor_names

    for i, name in enumerate(motor_names):
        robot._goal_pos_shared[i] = goal_deg.get(name, 0.0)
    robot._goal_ready.value = 1

    captured: dict[str, MagicMock] = {}

    def _add_motor(send_id, recv_id, model):
        motor = bus_mock.add_damiao_motor(send_id, recv_id, model)
        # Recover the name from the CAN id, since the loop builds the dict itself.
        for name, (s, _r) in config.motor_can_ids.items():
            if s == send_id:
                captured[name] = motor
        return motor

    bus_for_loop = MagicMock()
    bus_for_loop.add_damiao_motor.side_effect = _add_motor

    with (
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.signal", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus_for_loop
        controller_cls.return_value = bus_for_loop
        thread = threading.Thread(
            target=_follower_process_main,
            args=(
                config,
                motor_names,
                robot._goal_pos_shared,
                robot._goal_ready,
                robot._present_pos_shared,
                robot._present_torq_shared,
                robot._present_vel_shared,
                robot._present_pos_ts,
                robot._last_sent_shared,
                robot._has_sent_ever,
                robot._mit_kp_shared,
                robot._mit_kd_shared,
                robot._tau_ff_shared,
                robot._command_queue,
                robot._ready_event,
                robot._stop_event,
                robot._error_queue,
                robot._home_done,
            ),
            daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5.0
            while not robot._has_sent_ever.value and time.monotonic() < deadline:
                time.sleep(0.01)
            assert robot._has_sent_ever.value, f"follower loop never sent: {robot._error_queue.get_nowait()}"
        finally:
            robot._stop_event.set()
            thread.join(timeout=5.0)
            assert not thread.is_alive()

    return captured


def test_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null"))
    expected_pos = {f"{m}.pos" for m in robot.motor_names}
    expected_torq = {f"{m}.torq" for m in robot.motor_names}
    expected_vel = {f"{m}.vel" for m in robot.motor_names}
    assert set(robot.action_features) == expected_pos
    assert set(robot.observation_features) == expected_pos | expected_torq | expected_vel
    assert "gripper.pos" in expected_pos
    assert "gripper.torq" in expected_torq
    assert "gripper.vel" in expected_vel


def test_connect_disconnect(follower):
    assert follower.is_connected
    follower.disconnect()
    assert not follower.is_connected


def test_get_observation_reports_pos_torq_vel(follower):
    _seed_present(follower, pos_deg=1.0, torq=0.5, vel_deg_s=2.0)
    obs = follower.get_observation()

    assert set(obs) == set(follower.observation_features)
    for idx, motor in enumerate(follower.motor_names, 1):
        assert obs[f"{motor}.pos"] == pytest.approx(1.0 * idx)
        assert obs[f"{motor}.torq"] == pytest.approx(0.5 * idx)
        assert obs[f"{motor}.vel"] == pytest.approx(2.0 * idx)


def test_get_observation_falls_back_to_zero_before_first_tick(follower):
    obs = follower.get_observation()
    assert all(value == 0.0 for value in obs.values())


def test_send_action_clips_to_joint_limits(follower):
    # shoulder_pan limit is (-150, 150); request beyond the upper bound.
    returned = follower.send_action({"shoulder_pan.pos": 999.0})

    assert returned["shoulder_pan.pos"] == 150.0
    idx = follower.motor_names.index("shoulder_pan")
    assert follower._goal_pos_shared[idx] == pytest.approx(150.0)


def test_send_action_publishes_goal_for_the_follower_process(follower):
    assert follower._goal_ready.value == 0
    follower.send_action({"shoulder_pan.pos": 10.0})

    assert follower._goal_ready.value == 1
    idx = follower.motor_names.index("shoulder_pan")
    assert follower._goal_pos_shared[idx] == pytest.approx(10.0)


def test_send_action_holds_wrist_yaw_for_6dof_leaders(follower):
    # A leader with no wrist_yaw must not leave that joint's goal unwritten.
    follower._goal_pos_shared[follower.motor_names.index("wrist_yaw")] = 42.0
    returned = follower.send_action({"shoulder_pan.pos": 5.0})

    assert returned["wrist_yaw.pos"] == 0.0
    assert follower._goal_pos_shared[follower.motor_names.index("wrist_yaw")] == pytest.approx(0.0)


def test_send_action_returns_what_the_process_last_sent(follower):
    follower._has_sent_ever.value = 1
    for i in range(len(follower.motor_names)):
        follower._last_sent_shared[i] = 7.0

    returned = follower.send_action({"shoulder_pan.pos": 120.0})

    # The smoothed value the process actually sent, not the raw target.
    assert returned["shoulder_pan.pos"] == pytest.approx(7.0)


def test_follower_process_routes_arm_joints_to_send_mit():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null", control_mode="mit")
    motors = _run_one_tick(cfg, {"shoulder_pan.pos": 0.0})

    motors["shoulder_pan"].send_mit.assert_called()
    motors["shoulder_pan"].send_pos_vel.assert_not_called()


def test_follower_process_routes_arm_joints_to_send_pos_vel():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null", control_mode="pos_vel")
    motors = _run_one_tick(cfg, {"shoulder_pan.pos": 0.0})

    motors["shoulder_pan"].send_pos_vel.assert_called()
    motors["shoulder_pan"].send_mit.assert_not_called()


def test_follower_process_routes_gripper_to_force_pos():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null")
    motors = _run_one_tick(cfg, {"gripper.pos": -10.0})

    motors[GRIPPER_MOTOR].send_force_pos.assert_called()
    motors[GRIPPER_MOTOR].send_mit.assert_not_called()


def test_follower_process_routes_gripper_to_send_mit_when_configured():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null", gripper_control_mode="mit")
    motors = _run_one_tick(cfg, {"gripper.pos": -10.0})

    motors[GRIPPER_MOTOR].send_mit.assert_called()
    motors[GRIPPER_MOTOR].send_force_pos.assert_not_called()


def test_follower_process_sends_degrees_as_radians():
    cfg = RebotB601FollowerRobotConfig(
        port="/dev/null", control_mode="mit", enable_trajectory_smoothing=False
    )
    motors = _run_one_tick(cfg, {"shoulder_pan": 90.0})

    pos_rad = motors["shoulder_pan"].send_mit.call_args[0][0]
    assert pos_rad == pytest.approx(math.radians(90.0))


def test_home_ramp_starts_where_the_arm_is_and_ends_at_zero():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null")
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        motor_names = RebotB601Follower(cfg).motor_names
    start = dict.fromkeys(motor_names, 40.0)
    ramp = _HomeRamp(cfg, motor_names, start)

    at_start = ramp.target(0.0)
    assert at_start["shoulder_pan"] == pytest.approx(40.0)

    # Every joint is present in every sample, so each one is a complete goal.
    assert set(ramp.target(cfg.home_duration_s + 0.01)) == set(motor_names)

    settled = ramp.target(ramp.duration)
    assert all(value == pytest.approx(0.0) for value in settled.values())


def test_home_ramp_opens_gripper_and_lifts_wrist_before_lowering_them():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null")
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        motor_names = RebotB601Follower(cfg).motor_names
    ramp = _HomeRamp(cfg, motor_names, dict.fromkeys(motor_names, 40.0))

    # End of the approach leg: gripper wide open, wrist held clear of the shoulder.
    approach_end = ramp.target(cfg.home_duration_s)
    assert approach_end[GRIPPER_MOTOR] == pytest.approx(cfg.joint_limits[GRIPPER_MOTOR][0])
    assert approach_end[WRIST_MOTOR] == pytest.approx(cfg.home_wrist_flex_deg)

    # Both come down together over the close leg, rather than dropping at its end.
    mid_close = ramp.target(cfg.home_duration_s + cfg.gripper_close_duration_s / 2)
    assert abs(mid_close[GRIPPER_MOTOR]) < abs(approach_end[GRIPPER_MOTOR])
    assert abs(mid_close[WRIST_MOTOR]) < abs(approach_end[WRIST_MOTOR])


def test_home_ramp_is_monotonic_and_eased():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null")
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        motor_names = RebotB601Follower(cfg).motor_names
    ramp = _HomeRamp(cfg, motor_names, dict.fromkeys(motor_names, 100.0))

    steps = 40
    samples = [ramp.target(cfg.home_duration_s * i / steps)["shoulder_pan"] for i in range(steps + 1)]
    assert all(b <= a + 1e-9 for a, b in zip(samples[:-1], samples[1:], strict=True))

    # Smoothstep: the first step is far smaller than one at the midpoint.
    first = abs(samples[1] - samples[0])
    middle = abs(samples[steps // 2 + 1] - samples[steps // 2])
    assert first < middle / 2


def test_home_ramp_clips_to_joint_limits():
    cfg = RebotB601FollowerRobotConfig(port="/dev/null")
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        motor_names = RebotB601Follower(cfg).motor_names
    low, high = cfg.joint_limits["shoulder_pan"]
    ramp = _HomeRamp(cfg, motor_names, dict.fromkeys(motor_names, high * 10))

    assert ramp.target(0.0)["shoulder_pan"] == pytest.approx(high)
    assert low <= ramp.target(0.0)["shoulder_pan"] <= high


def test_go_home_without_a_follower_process_queues_nothing():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null"))
    robot._command_queue = MagicMock()

    with pytest.raises(ConnectionError):
        robot.go_home()

    # Not just that it raises: a command no live process will read would be
    # stranded with the queue's feeder thread and hang the interpreter at exit.
    robot._command_queue.put.assert_not_called()


def test_disconnect_releases_the_command_queue(follower):
    follower._command_queue = MagicMock()
    follower.disconnect()

    # Once the only reader is stopped, anything still in flight is unreadable;
    # left joinable it would hang the interpreter at exit, after the work is done.
    follower._command_queue.cancel_join_thread.assert_called_once()


def test_bimanual_prefixes_features():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebotB601FollowerConfig(
            left_arm_config=RebotB601FollowerConfig(port="/dev/null0"),
            right_arm_config=RebotB601FollowerConfig(port="/dev/null1"),
        )
        robot = BiRebotB601Follower(cfg)
    assert any(k.startswith("left_") for k in robot.action_features)
