"""Camera abstraction focused on Hikvision devices.

The module keeps the existing SDK integration while offering a pythonic API
that can be reused by the pipeline orchestration layer.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

try:  # pragma: no cover - SDK may be absent
    from camera import CamObj, CameraRunner  # type: ignore
except Exception:  # pragma: no cover
    CamObj = None  # type: ignore
    CameraRunner = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class Frame:
    """Simple container for image frames."""

    data: np.ndarray
    timestamp: float


class CameraManager:
    """Manage connection and streaming from Hikvision cameras.

    The implementation mirrors the earlier demo's usage of ``camera.py`` but
    exposes a stable API for the redesigned vision toolkit.
    """

    def __init__(self, config_path: str = "config.json") -> None:
        self.config_path = config_path
        self._runner: Optional[CameraRunner] = None
        self._latest_frame: Optional[Frame] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        if CamObj is None or CameraRunner is None:  # pragma: no cover - runtime guard
            raise RuntimeError("Hikvision SDK not available in this environment")

        LOGGER.info("Starting camera runner with config %s", self.config_path)
        self._runner = CameraRunner(self.config_path)

        def _on_frame(image: np.ndarray, timestamp: float) -> None:
            with self._lock:
                self._latest_frame = Frame(image, timestamp)

        self._runner.register_callback(_on_frame)
        self._runner.start()

    def close(self) -> None:
        if self._runner is not None:
            self._runner.stop()
            self._runner = None

    def grab(self) -> Optional[Frame]:
        with self._lock:
            return self._latest_frame

    @contextlib.contextmanager
    def session(self) -> Generator["CameraManager", None, None]:
        try:
            self.open()
            yield self
        finally:
            self.close()


class USBCameraManager:
    """Lightweight OpenCV-based camera manager for USB webcams."""

    def __init__(
        self,
        device_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        read_interval: float = 0.01,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.read_interval = read_interval
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[Frame] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def _reader(self) -> None:
        assert self._cap is not None
        while self._running.is_set():
            ok, frame = self._cap.read()
            if not ok:
                LOGGER.warning("USB camera %s read failed", self.device_index)
                time.sleep(max(self.read_interval, 0.05))
                continue
            with self._lock:
                self._latest_frame = Frame(frame, time.time())
            time.sleep(self.read_interval)

    def open(self) -> None:
        LOGGER.info("Opening USB camera index %s", self.device_index)
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open USB camera {self.device_index}")
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._running.set()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab(self) -> Optional[Frame]:
        with self._lock:
            return self._latest_frame

    @contextlib.contextmanager
    def session(self) -> Generator["USBCameraManager", None, None]:
        try:
            self.open()
            yield self
        finally:
            self.close()


def save_frame(frame: Frame, path: str | Path) -> None:
    """Persist a frame to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame.data)

