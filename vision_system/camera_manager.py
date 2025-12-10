"""Camera abstraction focused on Hikvision devices.

The module keeps the existing SDK integration while offering a pythonic API
that can be reused by the pipeline orchestration layer.
"""
from __future__ import annotations

import contextlib
import logging
import threading
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


def save_frame(frame: Frame, path: str | Path) -> None:
    """Persist a frame to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame.data)

