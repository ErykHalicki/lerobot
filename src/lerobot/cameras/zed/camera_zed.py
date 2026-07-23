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
import time
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

    A background thread is the sole caller of the native `get_last_frame()` (which
    blocks until the SDK has a new frame, ~1/fps per call): it continuously grabs
    and decodes frames into a shared buffer, so both eyes' `read_latest()` calls
    are non-blocking peeks of that buffer (matching `OpenCVCamera`'s async-read
    pattern) instead of each independently blocking for a fresh SDK frame. Without
    this, polling both eyes back-to-back in a control loop -- the common case --
    roughly halved the achievable loop rate, since each blocked ~1/fps in turn.
    `read()`/`async_read()` still block, but now by waiting on the background
    thread's buffer update rather than calling the SDK directly, so only one
    thread ever touches the native capture handle.
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
        self._cached_rgb: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None
        self._new_frame_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def __str__(self) -> str:
        return f"_ZedDevice(serial={self.serial_number or 'auto'})"

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
            self._cached_rgb = None
            self._latest_timestamp = None
        self._start_read_thread()

    def _resolve_dev_id(self) -> int:
        if self.serial_number is None:
            return -1
        for info in find_zed_cameras():
            if info["id"] == self.serial_number:
                return info["index"]
        raise ConnectionError(f"No ZED camera found with serial number {self.serial_number!r}.")

    def _close(self) -> None:
        # Stopped without holding self._lock: the read loop takes self._lock itself
        # to snapshot self._cap each iteration, so holding it across the join here
        # would deadlock against a thread blocked trying to acquire it.
        self._stop_read_thread()
        with self._lock:
            self._cap = None  # no explicit close in the native API; dropping the last
            # reference stops its capture thread and releases the device fd
            self._cached_frame_id = None
            self._cached_rgb = None
            self._latest_timestamp = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None

    def _read_loop(self) -> None:
        """Background thread: the only caller of the native (blocking) capture API.
        Continuously grabs and decodes frames into the shared buffer that
        `get_frame()` and `get_latest_frame()` read from. Read failures are
        logged (throttled) and retried indefinitely rather than killing the
        thread, so a flaky camera self-heals once it starts responding again
        instead of permanently breaking every subsequent read."""
        stop_event = self._stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not stop_event.is_set():
            with self._lock:
                cap = self._cap
            if cap is None:
                break
            try:
                frame = cap.get_last_frame(1000)
                if frame is None:
                    continue  # no new frame within the timeout; keep polling
                rgb = zoc.to_rgb(frame)
                with self._lock:
                    self._cached_rgb = rgb
                    self._cached_frame_id = frame.frame_id
                    self._latest_timestamp = time.perf_counter()
                self._new_frame_event.set()
                if failure_count > 0:
                    logger.info(f"{self}: read succeeded again after {failure_count} failed attempt(s).")
                failure_count = 0
            except Exception as e:
                failure_count += 1
                if failure_count == 1 or failure_count % 30 == 0:
                    logger.warning(
                        f"{self}: frame read failed ({failure_count} consecutive failure(s)): {e}. "
                        "Still retrying; serving the last known frame in the meantime."
                    )
                time.sleep(0.1)

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        self._stop_read_thread()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self._thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(f"{self} read thread did not terminate within timeout.")
        self._thread = None
        self._stop_event = None
        self._new_frame_event.clear()

    def get_frame(self, timeout_ms: int) -> NDArray[Any]:
        """Blocks until the background thread delivers a frame newer than the one
        returned by the previous call (or times out). Used by `read()`/`async_read()`."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self}: read thread is not running.")

        self._new_frame_event.clear()
        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for a ZED frame after {timeout_ms} ms.")

        with self._lock:
            if self._cached_rgb is None:
                raise RuntimeError(f"Internal error: {self} event set but no frame available.")
            return self._cached_rgb

    def get_latest_frame(self, max_age_ms: int) -> NDArray[Any]:
        """Non-blocking: returns whatever the background thread most recently
        buffered, regardless of whether it's new since the last call. Used by
        `read_latest()`."""
        with self._lock:
            frame = self._cached_rgb
            timestamp = self._latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms).")
        return frame


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
    """One eye (left or right), or the raw stereo pair, of a ZED/ZED-Mini/ZED-2/
    ZED-2i stereo camera.

    See `ZedCameraConfig` for why a single eye is split into its own per-camera
    object instead of one camera returning a double-wide image (and for the
    `side="stereo"` alternative that does return the double-wide image), and
    `_ZedDevice` for how multiple `ZedCamera` instances of the same physical
    camera share one underlying capture device.

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

        # or, for the raw undivided stereo frame:
        stereo = ZedCamera(ZedCameraConfig(side="stereo", serial_number="13925480"))
        stereo.connect()
        stereo_image = stereo.read()
        stereo.disconnect()
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

    def _crop(self, rgb: NDArray[Any]) -> NDArray[Any]:
        if self.side == "stereo":
            return rgb
        half = rgb.shape[1] // 2
        return rgb[:, :half] if self.side == "left" else rgb[:, half:]

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

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent frame (this eye only), non-blocking. See
        `Camera.read_latest`; unlike the base class default, this doesn't fall
        back to `async_read()` and so never blocks on the camera hardware."""
        assert self._device is not None
        return self._crop(self._device.get_latest_frame(max_age_ms))

    def disconnect(self) -> None:
        if self._device is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )
        _ZedDevice.release(self._key)
        self._device = None
        logger.info(f"{self} disconnected.")
