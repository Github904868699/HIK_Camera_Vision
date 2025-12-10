"""Detection utilities: YOLO inference and classical color/pose analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

SUPPORTED_LABELS = ("Red", "Green", "Blue", "Yellow")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class YOLODetector:
    """Lightweight wrapper around a YOLOv5/v8 style model.

    The loader attempts to import a torch model from ``base.pt``. When the
    dependency chain is unavailable the detector falls back to a color-based
    heuristic so the rest of the pipeline can still be exercised in CI.
    """

    def __init__(self, model_path: str | Path = "base.pt") -> None:
        self.model_path = Path(model_path)
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        try:  # pragma: no cover - optional dependency
            import torch  # type: ignore

            if not self.model_path.exists():
                LOGGER.warning("Model %s not found; using heuristics", self.model_path)
                return
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=str(self.model_path), _verbose=False)  # type: ignore
            LOGGER.info("Loaded YOLO model from %s", self.model_path)
        except Exception as exc:  # pragma: no cover - graceful degradation
            LOGGER.warning("YOLO initialization failed: %s. Using heuristic fallback.", exc)
            self.model = None

    def infer(self, image: np.ndarray) -> List[Detection]:
        if image is None:
            return []
        if self.model is None:
            return self._heuristic_detect(image)
        try:  # pragma: no cover - runtime path
            results = self.model(image)
            detections: List[Detection] = []
            for *xyxy, conf, cls in results.xyxy[0].tolist():  # type: ignore[attr-defined]
                x1, y1, x2, y2 = map(int, xyxy)
                label_idx = int(cls)
                label = SUPPORTED_LABELS[label_idx % len(SUPPORTED_LABELS)]
                detections.append(Detection(label, float(conf), (x1, y1, x2, y2)))
            return detections
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Inference failed: %s", exc)
            return self._heuristic_detect(image)

    def _heuristic_detect(self, image: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "Red": (np.array([0, 70, 50]), np.array([10, 255, 255])),
            "Green": (np.array([35, 70, 50]), np.array([85, 255, 255])),
            "Blue": (np.array([100, 70, 50]), np.array([130, 255, 255])),
            "Yellow": (np.array([20, 70, 50]), np.array([35, 255, 255])),
        }
        detections: List[Detection] = []
        for label, (lower, upper) in masks.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 150:  # skip noise
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(Detection(label, min(1.0, area / 10_000.0), (x, y, x + w, y + h)))
        return detections


def classify_orientation(contour: np.ndarray) -> str:
    """Rough orientation using contour moments."""

    moments = cv2.moments(contour)
    if moments["mu02"] + moments["mu20"] == 0:
        return "undetermined"
    theta = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
    angle = np.degrees(theta)
    if -45 <= angle <= 45:
        return "upright"
    return "flipped"


def detect_presence_and_orientation(image: np.ndarray) -> Dict[str, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    status = {"present": "no", "orientation": "undetermined"}
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            status["present"] = "yes"
            status["orientation"] = classify_orientation(largest)
    return status


def locate_objects(image: np.ndarray, detector: Optional[YOLODetector] = None) -> List[Detection]:
    detector = detector or YOLODetector()
    return detector.infer(image)


def measure_2d_size(image: np.ndarray, contour: np.ndarray) -> Tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return float(width), float(height)


def extract_text(image: np.ndarray) -> str:
    try:  # pragma: no cover - tesseract may be absent
        import pytesseract  # type: ignore

        return pytesseract.image_to_string(image, lang="eng", config="--psm 6").strip()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("OCR unavailable: %s", exc)
        return ""

