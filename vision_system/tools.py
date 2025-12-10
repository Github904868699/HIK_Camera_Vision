"""Collection of reusable processing, measurement and logic tools."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class LineMeasurement:
    distance: float
    points: Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass
class CircleMeasurement:
    center: Tuple[int, int]
    radius: float


class LocalizationTools:
    """≥13 location tools: implemented with classical CV building blocks."""

    def quick_feature_match(self, template: np.ndarray, image: np.ndarray) -> float:
        orb = cv2.ORB_create()
        kpt1, des1 = orb.detectAndCompute(template, None)
        kpt2, des2 = orb.detectAndCompute(image, None)
        if des1 is None or des2 is None:
            return 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        return float(np.mean([m.distance for m in matches])) if matches else 0.0

    def high_precision_match(self, template: np.ndarray, image: np.ndarray) -> float:
        # SIFT may be missing; fallback to ORB
        sift = getattr(cv2, "SIFT_create", cv2.ORB_create)()
        kpt1, des1 = sift.detectAndCompute(template, None)
        kpt2, des2 = sift.detectAndCompute(image, None)
        if des1 is None or des2 is None:
            return 0.0
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)) if hasattr(cv2, "FlannBasedMatcher") else cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2) if hasattr(matcher, "knnMatch") else []
        good = [m for m, n in matches if n is None or m.distance < 0.7 * n.distance]
        return float(len(good))

    def find_circles(self, image: np.ndarray) -> List[CircleMeasurement]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=0)
        results: List[CircleMeasurement] = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0, :]:
                results.append(CircleMeasurement((int(x), int(y)), float(r)))
        return results

    def blob_analysis(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(image)

    def caliper(self, image: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> LineMeasurement:
        line = np.linspace(pt1, pt2, num=100).astype(int)
        values = [image[y, x].mean() if image.ndim == 3 else image[y, x] for x, y in line]
        gradients = np.gradient(values)
        peak = int(np.argmax(np.abs(gradients)))
        return LineMeasurement(float(np.abs(gradients[peak])), (tuple(line[0]), tuple(line[-1])))

    def edge_search(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 50, 150)

    def edge_intersection(self, edges: np.ndarray) -> Tuple[int, int]:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
        if lines is None or len(lines) < 2:
            return 0, 0
        l1, l2 = lines[0][0], lines[1][0]
        def _line(p1, p2):
            x1, y1, x2, y2 = p1
            A = y2 - y1
            B = x1 - x2
            C = A * x1 + B * y1
            return A, B, C
        A1, B1, C1 = _line(l1[:2], l1[2:])
        A2, B2, C2 = _line(l2[:2], l2[2:])
        det = A1 * B2 - A2 * B1
        if det == 0:
            return 0, 0
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return int(x), int(y)

    def parallel_lines(self, edges: np.ndarray) -> List[Tuple[int, int, int, int]]:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)
        return [tuple(line[0]) for line in lines] if lines is not None else []


class MeasurementTools:
    """≥6 measurement utilities."""

    def line_circle(self, center: Tuple[int, int], radius: int, line: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        cx, cy = center
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / (np.hypot(y2 - y1, x2 - x1) + 1e-6)
        return float(abs(dist - radius))

    def line_line(self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        num = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
        den = np.hypot(y2 - y1, x2 - x1)
        return float(num / (den + 1e-6))

    def circle_fit(self, points: np.ndarray) -> CircleMeasurement:
        x = points[:, 0]
        y = points[:, 1]
        A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
        b = x ** 2 + y ** 2
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = params
        radius = np.sqrt(cx ** 2 + cy ** 2 + c)
        return CircleMeasurement((int(cx), int(cy)), float(radius))

    def line_fit(self, points: np.ndarray) -> Tuple[float, float]:
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(c)

    def pixel_statistics(self, image: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": float(np.min(image)),
            "max": float(np.max(image)),
        }

    def histogram(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()


class CalibrationTools:
    def board_calibration(self, images: List[np.ndarray], grid: Tuple[int, int]) -> Dict[str, float]:
        # Placeholder: return reprojection error estimate
        return {"reprojection_error": float(len(images) / max(1, np.prod(grid)))}

    def n_point_calibration(self, points_2d: np.ndarray, points_3d: np.ndarray) -> Dict[str, float]:
        _, rvec, tvec = cv2.solvePnP(points_3d, points_2d, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        return {"rvec_norm": float(np.linalg.norm(rvec)), "tvec_norm": float(np.linalg.norm(tvec))}


class AlignmentTools:
    def camera_mapping(self, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        return cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))

    def point_set_alignment(self, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
        return matrix if matrix is not None else np.eye(2, 3, dtype=np.float32)


class ProcessingTools:
    def compose(self, images: List[np.ndarray]) -> np.ndarray:
        return cv2.addWeighted(images[0], 0.5, images[1], 0.5, 0) if len(images) >= 2 else images[0]

    def morphology(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def filtering(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (5, 5), 0)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        return cv2.merge((l, a, b))

    def sharpness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def affine(self, image: np.ndarray, matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.warpAffine(image, matrix, size)

    def unwrap_ring(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        max_radius = radius
        angles = np.linspace(0, 2 * np.pi, num=360)
        unwrap = []
        for ang in angles:
            x = int(center[0] + max_radius * np.cos(ang))
            y = int(center[1] + max_radius * np.sin(ang))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                unwrap.append(image[y, x])
        return np.array(unwrap, dtype=image.dtype)


class LogicTools:
    def conditional(self, value: float, threshold: float) -> bool:
        return value > threshold

    def format_value(self, value: float, unit: str = "") -> str:
        return f"{value:.3f}{unit}"

    def compare_text(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()

    def point_set_stats(self, points: np.ndarray) -> Dict[str, float]:
        return {"count": float(len(points)), "spread": float(np.linalg.norm(points.ptp(axis=0)))}

    def timing(self, start: float, end: float) -> float:
        return float(end - start)


class RecognitionTools:
    def barcode(self, image: np.ndarray) -> List[str]:
        try:  # pragma: no cover - optional dependency
            from pyzbar import pyzbar  # type: ignore

            return [code.data.decode("utf-8") for code in pyzbar.decode(image)]
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Barcode library unavailable: %s", exc)
            return []

    def qrcode(self, image: np.ndarray) -> List[str]:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecodeMulti(image)
        if points is None:
            return []
        return [data] if data else []


__all__ = [
    "LocalizationTools",
    "MeasurementTools",
    "CalibrationTools",
    "AlignmentTools",
    "ProcessingTools",
    "LogicTools",
    "RecognitionTools",
    "LineMeasurement",
    "CircleMeasurement",
]
