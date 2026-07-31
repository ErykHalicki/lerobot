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

"""Thermal protection for the Damiao motors on the reBot B601-DM.

Each CAN feedback frame carries the motor's MOSFET and rotor temperatures
alongside its position, so the follower process gets both for free on every
tick. This module turns that stream into escalating terminal warnings and,
at the top level, a request to shut the arm down.
"""

import contextlib
import sys
import warnings
from collections import deque

import numpy as np

# The two temperatures a Damiao feedback frame carries, in the order
# ThermalMonitor.update() expects them.
COMPONENTS: tuple[str, ...] = ("mosfet", "rotor")

_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"

# Extrapolation from a short, 1-degree-quantised history is only worth
# reporting over a modest horizon; past this the fit says more about sensor
# noise than about where the motor is heading.
_MAX_HORIZON_S = 3600.0
# Below this many samples the fit is chasing quantisation steps rather than a
# trend, so no estimate is offered.
_MIN_SAMPLES = 20


def _hms(seconds: float) -> str:
    """Format a duration as HH:MM:SS."""
    total = max(0, int(round(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class _Track:
    """Recent temperature history for one component of one motor, and the
    quadratic extrapolation of when it will reach a given temperature.

    Samples are kept at a fixed rate over a fixed window, so the fit always
    sees the same span of time regardless of how fast the caller ticks.

    The curve is fitted over the whole window rather than differenced across
    it, so the 1-degree quantisation the motor reports in averages out instead
    of stepping the estimate.
    """

    def __init__(self, max_samples: int):
        self._t: deque[float] = deque(maxlen=max_samples)
        self._y: deque[float] = deque(maxlen=max_samples)

    def add(self, t: float, temp: float) -> None:
        self._t.append(t)
        self._y.append(temp)

    def seconds_to(self, target_c: float, now: float) -> float | None:
        """Seconds until this component reaches `target_c`, or None when no
        useful estimate exists.

        Returns None when there is too little history, when the component is
        already at or above the target, when the fit is not rising, or when the
        crossing lands beyond the horizon the history supports.
        """
        if len(self._t) < _MIN_SAMPLES or self._y[-1] >= target_c:
            return None

        # Time relative to now, so a crossing is simply a positive root.
        xs = np.fromiter(self._t, dtype=float) - now
        ys = np.fromiter(self._y, dtype=float)
        try:
            with warnings.catch_warnings():
                # A flat or near-flat history is rank-deficient for a quadratic;
                # the guards below catch the fits that are not usable.
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(xs, ys, 2)
        except Exception:
            return None

        # Slope at x=0: a motor that is steady or cooling right now has no
        # arrival time, whatever curvature the fit found behind it.
        if coeffs[1] <= 0:
            return None

        shifted = coeffs.copy()
        shifted[-1] -= target_c
        try:
            # A fit that curves over and plateaus below the target has no real
            # root, which is the honest answer rather than an extrapolated one.
            roots = np.roots(shifted)
        except Exception:
            return None

        crossings = [float(r.real) for r in roots if abs(r.imag) < 1e-6 and 0.0 < r.real <= _MAX_HORIZON_S]
        return min(crossings) if crossings else None


class ThermalMonitor:
    """Watches every motor's temperatures and escalates as they climb.

    Three configured thresholds define three levels: level 1 and 2 print a
    warning at their own repeat rate, level 3 prints once and latches a
    shutdown request. A motor that crosses into a higher level warns
    immediately rather than waiting out the previous level's interval.

    Warnings name only the components actually at the motor's current level,
    so one line covers a motor whose MOSFET and rotor are both hot. Levels 1
    and 2 also carry an estimated time to shutdown from the quadratic fit, when
    the history supports one.

    Owns no hardware: the caller feeds it readings and acts on the returned
    shutdown request.
    """

    def __init__(self, config, motor_names: list[str], out=None):
        self.warn_c = config.temp_warn_c
        self.danger_c = config.temp_danger_c
        self.max_c = config.temp_max_c
        self.shutdown_enabled = config.temp_shutdown_enabled
        self._repeat_s = {
            1: config.temp_warn_repeat_s,
            2: config.temp_danger_repeat_s,
            3: config.temp_danger_repeat_s,
        }
        self._sample_interval = 1.0 / config.temp_sample_hz
        max_samples = max(_MIN_SAMPLES, int(round(config.temp_history_s * config.temp_sample_hz)))
        self._tracks = {name: {comp: _Track(max_samples) for comp in COMPONENTS} for name in motor_names}
        self._last_level: dict[str, int] = dict.fromkeys(motor_names, 0)
        self._last_emit: dict[str, float] = dict.fromkeys(motor_names, 0.0)
        self._start: float | None = None
        self._last_sample: float | None = None
        self._shutdown = False

        self._out = out if out is not None else sys.stderr
        try:
            self._color = self._out.isatty()
        except Exception:
            self._color = False

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown

    def update(self, now: float, temps: dict[str, tuple[float, float] | None]) -> bool:
        """Record this tick's readings and print any warnings that are due.

        `temps` maps motor name to its (mosfet, rotor) temperatures in degrees
        Celsius, or to None for a motor that did not answer this tick. Returns
        True once any motor has reached the shutdown threshold; once that
        latches, nothing further is printed, so the home ramp that follows is
        not buried in repeats.
        """
        if self._start is None:
            self._start = now

        # History is sampled at its own rate; levels are checked every tick, so
        # an escalation is caught as soon as the frame carrying it arrives.
        record = self._last_sample is None or now - self._last_sample >= self._sample_interval
        if record:
            self._last_sample = now

        for name, reading in temps.items():
            if reading is None or name not in self._tracks:
                continue
            values = dict(zip(COMPONENTS, reading, strict=True))
            if record:
                for comp, temp in values.items():
                    self._tracks[name][comp].add(now, temp)

            if self._shutdown:
                continue

            level = max(self._level(temp) for temp in values.values())
            if level == 0:
                self._last_level[name] = 0
                continue

            hot = [comp for comp in COMPONENTS if self._level(values[comp]) == level]
            if self._should_emit(name, level, now):
                self._emit(name, level, hot, [values[comp] for comp in hot], now)
            if level >= 3 and self.shutdown_enabled:
                self._shutdown = True

        return self._shutdown

    def seconds_to_shutdown(self, motor: str) -> float | None:
        """Estimated seconds until `motor` reaches the shutdown threshold, from
        whichever of its components gets there first."""
        etas = [
            eta
            for comp in COMPONENTS
            if (eta := self._tracks[motor][comp].seconds_to(self.max_c, self._now_ref())) is not None
        ]
        return min(etas) if etas else None

    def seconds_to_next_level(self, motor: str) -> float | None:
        """Estimated seconds until `motor` reaches the threshold above the one
        it currently sits at. None once it is already at the top level."""
        level = self._last_level.get(motor, 0)
        target = {0: self.warn_c, 1: self.danger_c, 2: self.max_c}.get(level)
        if target is None:
            return None
        etas = [
            eta
            for comp in COMPONENTS
            if (eta := self._tracks[motor][comp].seconds_to(target, self._now_ref())) is not None
        ]
        return min(etas) if etas else None

    def _now_ref(self) -> float:
        """The timestamp of the most recent sample, which every fit measures
        its horizon from."""
        return self._last_sample if self._last_sample is not None else 0.0

    def _level(self, temp: float) -> int:
        if temp >= self.max_c:
            return 3
        if temp >= self.danger_c:
            return 2
        if temp >= self.warn_c:
            return 1
        return 0

    def _should_emit(self, name: str, level: int, now: float) -> bool:
        if level != self._last_level.get(name, 0):
            return True
        return now - self._last_emit.get(name, 0.0) >= self._repeat_s[level]

    def _emit(self, name: str, level: int, comps: list[str], temps: list[float], now: float) -> None:
        self._last_level[name] = level
        self._last_emit[name] = now
        elapsed = now - self._start if self._start is not None else 0.0
        line = f"{_hms(elapsed)} {self._message(name, level, comps, temps)}"
        color, bells = (_YELLOW, "\a") if level == 1 else (_RED, "\a\a\a")
        if self._color:
            line = f"{color}{line}{_RESET}"
        self._out.write(f"{bells}{line}\n")
        # Unbuffered, so a warning reaches the terminal before the arm reaches
        # the next threshold.
        with contextlib.suppress(Exception):
            self._out.flush()

    def _message(self, name: str, level: int, comps: list[str], temps: list[float]) -> str:
        parts = " and ".join(comps)
        values = " and ".join(f"{temp:.0f}C" for temp in temps)
        noun = "temperatures" if len(comps) > 1 else "temperature"

        if level == 3:
            tail = "Shutting down now!" if self.shutdown_enabled else "(auto shutdown disabled)"
            return f"{name} {parts} at maximum operating {noun} {values}! {tail}"

        eta = self.seconds_to_shutdown(name) if self.shutdown_enabled else None
        if level == 1:
            msg = f"{name} {parts} {noun} too high at {values}!"
            if eta is not None:
                msg += f" Turn off arm to prevent auto shutdown in {_hms(eta)}"
            return msg

        msg = f"{name} {parts} {noun} dangerously high at {values}!"
        if eta is not None:
            whole = max(1, round(eta))
            msg += f" ~{whole} second{'' if whole == 1 else 's'} until auto shutdown"
        return msg
