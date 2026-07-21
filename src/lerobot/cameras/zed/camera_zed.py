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

"""
Provides the ZedCamera class for capturing frames from ZED/ZED-Mini/ZED-2/ZED-2i
stereo cameras via py-zed-open-capture (no CUDA/ZED SDK required).
"""

import logging
import threading
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _py_zed_open_capture_available, require_package

from ..camera import Camera
from .configuration_zed import RESOLUTIONS, ZedCameraConfig

if TYPE_CHECKING or _py_zed_open_capture_available:
    import py_zed_open_capture as zoc
else:
    zoc = None

logger = logging.getLogger(__name__)


class _ZedDevice:
    """Shared handle to one physical ZED camera's underlying VideoCapture.

    A ZED delivers one side-by-side stereo frame per capture, but each eye is exposed
    as its own `ZedCamera` (see below) so it can be recorded as its own dataset camera
    key. Both `ZedCamera` instances for the same physical camera resolve to the same
    `_ZedDevice` (keyed by serial number, or "auto" when unspecified) so the device is
    only opened once and reference-counted across the two `ZedCamera.connect()`/
    `disconnect()` calls, rather than trying to open the same /dev/video node twice.

    The decoded BGR frame is cached by `frame_id` so polling both eyes back-to-back
    (the common case in a control loop) only pays for one YUV->BGR conversion.
    """

    _registry: dict[str, "_ZedDevice"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, serial_number: str | None, width: int, height: int, fps: int):
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps

        self._lock = threading.Lock()
        self._refcount = 0
        self._cap: Any = None
        self._cached_frame_id: int | None = None
        self._cached_bgr: NDArray[Any] | None = None

    @classmethod
    def acquire(cls, key: str, serial_number: str | None, width: int, height: int, fps: int) -> "_ZedDevice":
        with cls._registry_lock:
            device = cls._registry.get(key)
            if device is None:
                device = cls(serial_number, width, height, fps)
                cls._registry[key] = device
            elif (device.width, device.height, device.fps) != (width, height, fps):
                raise ValueError(
                    f"ZedCamera(serial={serial_number!r}) already opened with "
                    f"{device.width}x{device.height}@{device.fps}fps; both sides of the "
                    f"same physical camera must use matching width/height/fps."
                )
            device._refcount += 1
            return device

    @classmethod
    def release(cls, key: str) -> None:
        with cls._registry_lock:
            device = cls._registry.get(key)
            if device is None:
                return
            device._refcount -= 1
            if device._refcount <= 0:
                device._close()
                del cls._registry[key]

    def connect(self) -> None:
        with self._lock:
            if self._cap is not None:
                return
            dev_id = self._resolve_dev_id()
            params = zoc.VideoParams()
            params.res = getattr(zoc.Resolution, RESOLUTIONS[(self.width, self.height)])
            params.fps = getattr(zoc.Fps, f"FPS_{self.fps}")
            cap = zoc.VideoCapture(params)
            if not cap.initialize_video(dev_id):
                raise ConnectionError(
                    f"Failed to open ZED camera (serial={self.serial_number!r}, dev_id={dev_id})."
                )
            self._cap = cap
            self._cached_frame_id = None
            self._cached_bgr = None

    def _resolve_dev_id(self) -> int:
        if self.serial_number is None:
            return -1
        for info in find_zed_cameras():
            if info["id"] == self.serial_number:
                return info["index"]
        raise ConnectionError(f"No ZED camera found with serial number {self.serial_number!r}.")

    def _close(self) -> None:
        with self._lock:
            self._cap = None  # no explicit close in the native API; dropping the last
            # reference stops its capture thread and releases the device fd
            self._cached_frame_id = None
            self._cached_bgr = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None

    def get_frame(self, timeout_ms: int) -> NDArray[Any]:
        with self._lock:
            cap = self._cap
        if cap is None:
            raise RuntimeError("_ZedDevice.get_frame() called before connect().")

        frame = cap.get_last_frame(timeout_ms)
        if frame is None:
            raise TimeoutError(f"Timed out waiting for a ZED frame after {timeout_ms} ms.")

        with self._lock:
            if self._cached_frame_id != frame.frame_id:
                self._cached_bgr = zoc.to_bgr(frame)
                self._cached_frame_id = frame.frame_id
            return self._cached_bgr


def find_zed_cameras() -> list[dict[str, Any]]:
    """Detects connected ZED cameras by probing a small range of /dev/video indices.

    zed-open-capture has no lightweight bus-enumeration API (unlike e.g. pyrealsense2's
    `context.query_devices()`) -- identifying a camera means actually opening it, so
    each candidate index is briefly initialized (at the cheapest mode) just to read its
    serial number/name, then released. This only tries even indices 0..14, matching
    the two-node-per-camera pattern observed on Linux (metadata node + capture node);
    a ZED enumerated at an odd/other index would be missed.
    """
    require_package("py-zed-open-capture", extra="zed", import_name="py_zed_open_capture")

    found = []
    for dev_id in range(0, 16, 2):
        try:
            params = zoc.VideoParams()
            params.res = zoc.Resolution.VGA
            params.fps = zoc.Fps.FPS_15
            cap = zoc.VideoCapture(params)
            if not cap.initialize_video(dev_id):
                continue
            found.append(
                {
                    "name": cap.get_device_name(),
                    "type": "ZED",
                    "id": str(cap.get_serial_number()),
                    "index": dev_id,
                }
            )
        except Exception:
            continue
    return found


class ZedCamera(Camera):
    """One eye (left or right) of a ZED/ZED-Mini/ZED-2/ZED-2i stereo camera.

    See `ZedCameraConfig` for why this is split into two per-camera objects instead of
    one camera returning a double-wide image, and `_ZedDevice` for how the two share
    one underlying capture device.

    Example:
        ```python
        from lerobot.cameras.zed import ZedCamera, ZedCameraConfig

        left = ZedCamera(ZedCameraConfig(side="left", serial_number="13925480"))
        right = ZedCamera(ZedCameraConfig(side="right", serial_number="13925480"))
        left.connect()
        right.connect()  # shares the device `left` already opened
        left_image = left.read()
        right_image = right.read()
        left.disconnect()
        right.disconnect()  # device only actually closes here
        ```
    """

    def __init__(self, config: ZedCameraConfig):
        require_package("py-zed-open-capture", extra="zed", import_name="py_zed_open_capture")
        super().__init__(config)

        self.config = config
        self.side = config.side
        self.fps = config.fps if config.fps is not None else 30
        self.width = config.width if config.width is not None else 1280
        self.height = config.height if config.height is not None else 720
        self._key = config.serial_number or "auto"
        self._device: _ZedDevice | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.side}, serial={self.config.serial_number or 'auto'})"

    @property
    def is_connected(self) -> bool:
        return self._device is not None and self._device.is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return find_zed_cameras()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        # ZedCameraConfig.__post_init__ guarantees these are set (defaults filled in).
        assert self.width is not None and self.height is not None and self.fps is not None
        device = _ZedDevice.acquire(self._key, self.config.serial_number, self.width, self.height, self.fps)
        try:
            device.connect()
        except Exception:
            _ZedDevice.release(self._key)
            raise
        self._device = device

        if warmup:
            try:
                self.read()
            except Exception as e:
                self.disconnect()
                raise ConnectionError(f"{self} failed to capture a frame during warmup.") from e

        logger.info(f"{self} connected.")

    def _crop(self, bgr: NDArray[Any]) -> NDArray[Any]:
        half = bgr.shape[1] // 2
        return bgr[:, :half] if self.side == "left" else bgr[:, half:]

    @check_if_not_connected
    def read(self, timeout_ms: int = 1000) -> NDArray[Any]:
        """Reads a single frame (this eye only) synchronously. Blocking."""
        assert self._device is not None
        return self._crop(self._device.get_frame(timeout_ms))

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Returns the latest frame (this eye only). See `Camera.async_read`."""
        assert self._device is not None
        return self._crop(self._device.get_frame(int(timeout_ms)))

    def disconnect(self) -> None:
        if self._device is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )
        _ZedDevice.release(self._key)
        self._device = None
        logger.info(f"{self} disconnected.")
