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

from dataclasses import dataclass
from typing import Literal

from ..configs import CameraConfig

# (width, height) per eye -> py_zed_open_capture.Resolution name. These are the
# camera's only native modes -- there's no continuous resolution negotiation like a
# generic UVC webcam.
RESOLUTIONS = {
    (2208, 1242): "HD2K",
    (1920, 1080): "HD1080",
    (1280, 720): "HD720",
    (672, 376): "VGA",
}
FPS_CHOICES = (15, 30, 60, 100)


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration for one eye (or the raw stereo pair) of a ZED / ZED-Mini /
    ZED-2 / ZED-2i camera.

    The camera delivers a single side-by-side stereo frame per capture. Requesting
    `side="left"` or `side="right"` exposes just that eye, cropped from the raw
    frame -- useful for treating each eye as its own dataset camera key via two
    `ZedCameraConfig`/`ZedCamera` instances. `side="stereo"` instead returns the
    full, uncropped side-by-side frame (both eyes, at `2 * width` pixels wide) from
    a single instance. All instances of the same physical camera must share the
    same `serial_number` (or all leave it as the default `None`, for auto-detecting
    the sole/first connected ZED) -- the underlying capture device is then opened
    exactly once and shared between them (see `ZedCamera`/`_ZedDevice` in
    `camera_zed.py`), since the physical /dev/video node can't be opened twice.

    Example:
        ```python
        cameras = {
            "zed_left": ZedCameraConfig(side="left", serial_number="13925480"),
            "zed_right": ZedCameraConfig(side="right", serial_number="13925480"),
        }
        # or, for the raw undivided stereo frame:
        cameras = {"zed_stereo": ZedCameraConfig(side="stereo", serial_number="13925480")}
        ```

    Attributes:
        side: Which eye this camera instance reads ("left" or "right"), or "stereo"
            for the full undivided side-by-side frame.
        serial_number: ZED serial number to open a specific camera. `None` (default)
            auto-detects the first ZED found -- fine for a single-camera setup.
        fps: One of (15, 30, 60, 100), subject to the resolution's own fps ceiling
            (see zed-open-capture's RESOLUTION/FPS docs). Defaults to 30.
        width, height: Per-eye pixel size; must be one of the pairs in `RESOLUTIONS`
            (2208x1242, 1920x1080, 1280x720, 672x376). Defaults to 1280x720 (HD720).
            In "stereo" mode the returned frame is `2 * width` pixels wide.
    """

    side: Literal["left", "right", "stereo"]
    serial_number: str | None = None

    def __post_init__(self) -> None:
        # fps/width/height stay `int | None` (unset = ZedCamera fills in the
        # 1280x720@30fps default) -- kept as inherited from CameraConfig (rather than
        # overridden with concrete defaults here) because redeclaring them on this
        # subclass's non-kw_only @dataclass would strip their kw_only-ness inherited
        # from CameraConfig, breaking field ordering against `side` (no default).
        if self.fps is not None and self.fps not in FPS_CHOICES:
            raise ValueError(f"Unsupported ZED fps {self.fps}. Must be one of {FPS_CHOICES}.")
        if self.width is not None and self.height is not None and (self.width, self.height) not in RESOLUTIONS:
            raise ValueError(
                f"Unsupported ZED resolution {self.width}x{self.height}. "
                f"Must be one of {sorted(RESOLUTIONS)}."
            )
