"""Task orchestration for the new vision application."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detection import (
    YOLODetector,
    detect_presence_and_orientation,
    extract_text,
    locate_objects,
)
from .tools import (
    AlignmentTools,
    CalibrationTools,
    LocalizationTools,
    LogicTools,
    MeasurementTools,
    ProcessingTools,
    RecognitionTools,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    detections: List[Dict[str, object]] = field(default_factory=list)
    presence: Dict[str, str] = field(default_factory=dict)
    measurements: Dict[str, object] = field(default_factory=dict)
    logic: Dict[str, object] = field(default_factory=dict)
    recognition: Dict[str, object] = field(default_factory=dict)
    processing: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


class VisionPipeline:
    def __init__(self, model_path: str | Path = "base.pt") -> None:
        self.detector = YOLODetector(model_path)
        self.localization = LocalizationTools()
        self.measurement = MeasurementTools()
        self.calibration = CalibrationTools()
        self.alignment = AlignmentTools()
        self.processing = ProcessingTools()
        self.logic = LogicTools()
        self.recognition = RecognitionTools()

    def run(self, image: np.ndarray, template: Optional[np.ndarray] = None) -> PipelineResult:
        result = PipelineResult()

        result.presence = detect_presence_and_orientation(image)
        result.detections = [d.__dict__ for d in locate_objects(image, self.detector)]

        edges = self.localization.edge_search(image)
        intersections = self.localization.edge_intersection(edges)
        parallels = self.localization.parallel_lines(edges)
        circles = [cm.__dict__ for cm in self.localization.find_circles(image)]
        blobs = len(self.localization.blob_analysis(image))
        quick_score = self.localization.quick_feature_match(template, image) if template is not None else 0.0
        precise_score = self.localization.high_precision_match(template, image) if template is not None else 0.0

        hist = self.measurement.histogram(image).tolist()
        stats = self.measurement.pixel_statistics(image)

        result.measurements = {
            "quick_match": quick_score,
            "precise_match": precise_score,
            "edges_intersection": intersections,
            "parallel_lines": parallels,
            "circles": circles,
            "blob_count": blobs,
            "pixel_stats": stats,
            "histogram": hist,
        }

        processed = self.processing.filtering(image)
        sharpeness = self.processing.sharpness(image)
        result.processing = {
            "filtered_preview_shape": processed.shape,
            "sharpness": sharpeness,
        }

        text = extract_text(image)
        barcodes = self.recognition.barcode(image)
        qrcodes = self.recognition.qrcode(image)
        result.recognition = {
            "ocr": text,
            "barcodes": barcodes,
            "qrcodes": qrcodes,
        }

        result.logic = {
            "pass": self.logic.conditional(sharpeness, threshold=10.0),
            "formatted_sharpness": self.logic.format_value(sharpeness, unit=" var"),
        }

        return result


__all__ = ["VisionPipeline", "PipelineResult"]
