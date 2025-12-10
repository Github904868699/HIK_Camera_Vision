"""Command-line entry point for the redesigned vision system."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2

from vision_system.camera_manager import CameraManager, USBCameraManager
from vision_system.pipelines import VisionPipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def load_image(path: Path) -> Optional[cv2.Mat]:
    if not path.exists():
        LOGGER.error("Image %s not found", path)
        return None
    return cv2.imread(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIK camera vision toolkit")
    parser.add_argument("--image", type=Path, help="Path to image for offline processing")
    parser.add_argument("--template", type=Path, help="Template image for feature matching", default=None)
    parser.add_argument("--model", type=Path, help="YOLO model path", default=Path("base.pt"))
    parser.add_argument("--camera", action="store_true", help="Use Hikvision camera via config.json")
    parser.add_argument("--usb", type=int, help="USB camera index for OpenCV VideoCapture")
    parser.add_argument("--usb-width", type=int, help="Force USB capture width", default=None)
    parser.add_argument("--usb-height", type=int, help="Force USB capture height", default=None)
    parser.add_argument("--save", type=Path, help="Where to save the last grabbed frame", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = VisionPipeline(args.model)

    def _acquire_from_manager(manager) -> Optional[cv2.Mat]:
        """Grab a single frame from the given camera manager."""

        with manager.session() as cam:
            frame = cam.grab()
            if frame is None:
                LOGGER.error("No frame received from camera")
                return None
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save), frame.data)
            return frame.data

    if args.camera and args.usb is not None:
        LOGGER.error("--camera and --usb are mutually exclusive")
        return

    if args.camera:
        manager = CameraManager()
        image = _acquire_from_manager(manager)
        if image is None:
            return
    elif args.usb is not None:
        manager = USBCameraManager(
            device_index=args.usb,
            width=args.usb_width,
            height=args.usb_height,
        )
        image = _acquire_from_manager(manager)
        if image is None:
            return
    else:
        if args.image:
            image = load_image(args.image)
            if image is None:
                return
        else:
            LOGGER.info("No --image provided, attempting to open USB camera 0 for quick debugging...")
            try:
                image = _acquire_from_manager(
                    USBCameraManager(device_index=0, width=args.usb_width, height=args.usb_height)
                )
            except Exception as exc:  # pragma: no cover - runtime hardware guard
                LOGGER.warning("USB camera 0 unavailable (%s). Trying Hikvision config...", exc)
                try:
                    image = _acquire_from_manager(CameraManager())
                except Exception as hik_exc:  # pragma: no cover - runtime hardware guard
                    LOGGER.error(
                        "无法获取图像：未提供 --image，且 USB 摄像头/Hikvision 摄像头均未成功打开 (%s)",
                        hik_exc,
                    )
                    return
            if image is None:
                LOGGER.error("无法获取图像：摄像头未返回帧")
                return

    template_image = load_image(args.template) if args.template else None
    result = pipeline.run(image, template_image)
    print(result.to_json())


if __name__ == "__main__":
    main()
