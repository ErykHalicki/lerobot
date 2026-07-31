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

import io
import re

import pytest

from lerobot.robots.rebot_b601_follower import RebotB601FollowerRobotConfig
from lerobot.robots.rebot_b601_follower.thermal_monitor import ThermalMonitor, _hms

MOTORS = ["shoulder_pan", "wrist_flex", "gripper"]


def _config(**overrides) -> RebotB601FollowerRobotConfig:
    config = RebotB601FollowerRobotConfig(port="/dev/null")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _monitor(**overrides) -> tuple[ThermalMonitor, io.StringIO]:
    out = io.StringIO()
    return ThermalMonitor(_config(**overrides), MOTORS, out=out), out


def _lines(out: io.StringIO) -> list[str]:
    return [line for line in out.getvalue().split("\n") if line]


def _cool(**hot) -> dict:
    """All motors at 25C except the ones named, given as (mosfet, rotor)."""
    return {name: hot.get(name, (25.0, 25.0)) for name in MOTORS}


def test_stays_quiet_below_the_first_threshold():
    monitor, out = _monitor()
    for tick in range(50):
        assert not monitor.update(tick * 0.1, _cool(gripper=(44.0, 44.0)))
    assert _lines(out) == []


def test_level_1_names_the_single_hot_component():
    monitor, out = _monitor()
    monitor.update(0.0, _cool(wrist_flex=(46.0, 30.0)))
    (line,) = _lines(out)
    assert "wrist_flex mosfet temperature too high at 46C!" in line


def test_level_1_combines_both_components():
    monitor, out = _monitor()
    monitor.update(0.0, _cool(gripper=(45.0, 47.0)))
    (line,) = _lines(out)
    assert "gripper mosfet and rotor temperatures too high at 45C and 47C!" in line


def test_level_2_is_red_with_three_bells_and_level_1_is_yellow_with_one():
    monitor, out = _monitor()
    monitor.update(0.0, _cool(gripper=(46.0, 30.0)))
    monitor.update(1.0, _cool(gripper=(51.0, 30.0)))
    warn, danger = _lines(out)
    assert warn.startswith("\a") and not warn.startswith("\a\a")
    assert "temperature too high" in warn
    assert danger.startswith("\a\a\a")
    assert "gripper mosfet temperature dangerously high at 51C!" in danger


def test_level_3_shuts_down_and_says_so():
    monitor, out = _monitor()
    assert monitor.update(0.0, _cool(gripper=(56.0, 30.0)))
    (line,) = _lines(out)
    assert "gripper mosfet at maximum operating temperature 56C! Shutting down now!" in line
    assert monitor.shutdown_requested


def test_level_3_without_shutdown_enabled_warns_but_does_not_latch():
    monitor, out = _monitor(temp_shutdown_enabled=False)
    assert not monitor.update(0.0, _cool(gripper=(56.0, 30.0)))
    (line,) = _lines(out)
    assert "(auto shutdown disabled)" in line
    assert not monitor.shutdown_requested


def test_nothing_is_printed_once_shutdown_has_latched():
    monitor, out = _monitor()
    monitor.update(0.0, _cool(gripper=(56.0, 30.0)))
    for tick in range(1, 100):
        monitor.update(tick * 0.1, _cool(gripper=(57.0, 30.0), wrist_flex=(52.0, 52.0)))
    assert len(_lines(out)) == 1


def test_repeat_rate_differs_per_level():
    warn_monitor, warn_out = _monitor()
    danger_monitor, danger_out = _monitor()
    for tick in range(101):  # 10s at 10Hz
        warn_monitor.update(tick * 0.1, _cool(gripper=(46.0, 30.0)))
        danger_monitor.update(tick * 0.1, _cool(gripper=(51.0, 30.0)))
    # 3s repeat over 10s: t=0, 3, 6, 9. 1s repeat: t=0..10.
    assert len(_lines(warn_out)) == 4
    assert len(_lines(danger_out)) == 11


def test_escalating_warns_immediately_rather_than_waiting_out_the_interval():
    monitor, out = _monitor()
    monitor.update(0.0, _cool(gripper=(46.0, 30.0)))
    monitor.update(0.1, _cool(gripper=(51.0, 30.0)))
    assert len(_lines(out)) == 2


def test_dropping_back_below_every_threshold_rearms_the_warning():
    monitor, out = _monitor()
    monitor.update(0.0, _cool(gripper=(46.0, 30.0)))
    monitor.update(0.1, _cool(gripper=(30.0, 30.0)))
    monitor.update(0.2, _cool(gripper=(46.0, 30.0)))
    assert len(_lines(out)) == 2


def test_a_motor_that_missed_its_frame_is_skipped():
    monitor, out = _monitor()
    monitor.update(0.0, {"shoulder_pan": None, "wrist_flex": None, "gripper": None})
    assert _lines(out) == []


def test_colour_is_off_for_a_non_tty():
    monitor, out = _monitor()
    monitor.update(0.0, _cool(gripper=(46.0, 30.0)))
    assert "\033[" not in out.getvalue()


def test_steady_ramp_predicts_its_own_arrival():
    """A motor climbing 1C/s from 40C should reach the 55C shutdown threshold
    15s in, so the estimate offered at 46C is ~9s."""
    monitor, _ = _monitor()
    for tick in range(61):
        t = tick * 0.1
        monitor.update(t, _cool(gripper=(40.0 + t, 30.0)))
    eta = monitor.seconds_to_shutdown("gripper")
    assert eta is not None
    assert 8.0 < eta < 10.0


def test_no_estimate_offered_for_a_motor_holding_temperature():
    monitor, _ = _monitor()
    for tick in range(200):
        monitor.update(tick * 0.1, _cool(gripper=(46.0, 30.0)))
    assert monitor.seconds_to_shutdown("gripper") is None


def test_no_estimate_offered_for_a_cooling_motor():
    monitor, _ = _monitor()
    for tick in range(200):
        t = tick * 0.1
        monitor.update(t, _cool(gripper=(46.0 - t * 0.05, 30.0)))
    assert monitor.seconds_to_shutdown("gripper") is None


def test_no_estimate_before_enough_history_has_accumulated():
    monitor, out = _monitor()
    for tick in range(5):
        monitor.update(tick * 0.1, _cool(gripper=(45.0 + tick, 30.0)))
    assert monitor.seconds_to_shutdown("gripper") is None
    assert "auto shutdown" not in _lines(out)[0]


def test_level_1_message_carries_the_shutdown_estimate():
    monitor, out = _monitor()
    for tick in range(61):
        t = tick * 0.1
        monitor.update(t, _cool(gripper=(40.0 + t, 30.0)))
    warned = [line for line in _lines(out) if "too high" in line]
    assert warned, "expected at least one level 1 warning"
    assert re.search(r"Turn off arm to prevent auto shutdown in \d\d:\d\d:\d\d", warned[-1])


def test_level_2_message_carries_the_shutdown_estimate_in_seconds():
    monitor, out = _monitor()
    for tick in range(111):
        t = tick * 0.1
        monitor.update(t, _cool(gripper=(40.0 + t, 30.0)))
    dangerous = [line for line in _lines(out) if "dangerously high" in line]
    assert dangerous, "expected at least one level 2 warning"
    assert re.search(r"~\d+ seconds until auto shutdown", dangerous[-1])


def test_next_level_estimate_targets_the_threshold_above_the_current_one():
    """Climbing 1C/s and sitting at level 1, the next level (50C) is closer
    than shutdown (55C)."""
    monitor, _ = _monitor()
    for tick in range(61):
        t = tick * 0.1
        monitor.update(t, _cool(gripper=(40.0 + t, 30.0)))
    next_level = monitor.seconds_to_next_level("gripper")
    shutdown = monitor.seconds_to_shutdown("gripper")
    assert next_level is not None and shutdown is not None
    assert next_level < shutdown
    assert 3.0 < next_level < 5.0


def test_history_is_capped_at_the_configured_window():
    monitor, _ = _monitor(temp_history_s=5.0, temp_sample_hz=10.0)
    for tick in range(600):
        monitor.update(tick * 0.1, _cool(gripper=(30.0, 30.0)))
    track = monitor._tracks["gripper"]["mosfet"]
    assert len(track._t) == 50


def test_history_is_decimated_to_the_sample_rate():
    """Fed at 100Hz but sampling at 10Hz, one second of ticks stores ~10."""
    monitor, _ = _monitor()
    for tick in range(100):
        monitor.update(tick * 0.01, _cool(gripper=(30.0, 30.0)))
    track = monitor._tracks["gripper"]["mosfet"]
    assert 9 <= len(track._t) <= 11


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00"), (5, "00:00:05"), (92, "00:01:32"), (3661, "01:01:01"), (-5, "00:00:00")],
)
def test_hms(seconds, expected):
    assert _hms(seconds) == expected
