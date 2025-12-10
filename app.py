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

    if args.camera and args.usb is not None:
        LOGGER.error("--camera and --usb are mutually exclusive")
        return

    if args.camera:
        manager = CameraManager()
        with manager.session() as cam:
            frame = cam.grab()
            if frame is None:
                LOGGER.error("No frame received from Hikvision camera")
                return
            image = frame.data
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save), image)
    elif args.usb is not None:
        manager = USBCameraManager(
            device_index=args.usb,
            width=args.usb_width,
            height=args.usb_height,
        )
        with manager.session() as cam:
            frame = cam.grab()
            if frame is None:
                LOGGER.error("No frame received from USB camera")
                return
            image = frame.data
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save), image)
    else:
        if not args.image:
            LOGGER.error("--image is required when --camera is not set")
            return
        image = load_image(args.image)
        if image is None:
            return

    template_image = load_image(args.template) if args.template else None
    result = pipeline.run(image, template_image)
    print(result.to_json())


if __name__ == "__main__":
    main()
