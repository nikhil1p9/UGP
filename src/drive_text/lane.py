from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schema import LaneSummary


@dataclass(slots=True)
class LaneEstimate:
    lane_count: int | None
    visible_lane_markings: int
    confidence: float


class HeuristicLaneEstimator:
    def estimate(self, frames: list[np.ndarray]) -> LaneSummary:
        if not frames:
            return LaneSummary()
        counts: list[int] = []
        markings: list[int] = []
        confidences: list[float] = []
        for frame in frames:
            estimate = self._estimate_single_frame(frame)
            if estimate.lane_count is not None:
                counts.append(estimate.lane_count)
            markings.append(estimate.visible_lane_markings)
            confidences.append(estimate.confidence)
        lane_count = int(round(float(np.median(counts)))) if counts else None
        marking_count = int(round(float(np.median(markings)))) if markings else 0
        confidence = float(np.mean(confidences)) if confidences else 0.0
        road_type = "center" if lane_count and lane_count >= 2 else "unknown"
        return LaneSummary(
            estimated_lane_count=lane_count,
            visible_lane_markings=marking_count,
            ego_lane_position=road_type,
            confidence=confidence,
            source="heuristic",
        )

    def _estimate_single_frame(self, frame: np.ndarray) -> LaneEstimate:
        height, width = frame.shape[:2]
        roi = frame[int(height * 0.5) :, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 180)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=max(width // 8, 40),
            maxLineGap=30,
        )
        if lines is None:
            return LaneEstimate(lane_count=None, visible_lane_markings=0, confidence=0.0)

        bottom_intersections: list[float] = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(float, line)
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.35:
                continue
            if abs(y2 - y1) < roi.shape[0] * 0.15:
                continue
            bottom_y = float(roi.shape[0] - 1)
            intercept_x = x1 + (bottom_y - y1) / slope
            if 0 <= intercept_x <= width:
                bottom_intersections.append(intercept_x)

        if not bottom_intersections:
            return LaneEstimate(lane_count=None, visible_lane_markings=0, confidence=0.0)

        bottom_intersections.sort()
        clustered = [bottom_intersections[0]]
        merge_distance = width * 0.08
        for value in bottom_intersections[1:]:
            if abs(value - clustered[-1]) > merge_distance:
                clustered.append(value)
        visible_markings = len(clustered)
        lane_count = max(1, min(visible_markings + 1, 6))
        confidence = min(1.0, visible_markings / 4.0)
        return LaneEstimate(lane_count=lane_count, visible_lane_markings=visible_markings, confidence=confidence)


class CLRLaneEstimator:
    """Stub backend that delegates to a CLRNet installation.

    To activate:
      1. Clone CLRNet: git clone https://github.com/Turoad/CLRNet
      2. Install it in the same environment (pip install -e .)
      3. Replace the body of estimate() below with:

         from clrnet.utils.config import Config
         from clrnet.models.registry import build_net
         # ... load model, run inference on each frame, count unique lane lines.
    """

    def estimate(self, frames: list[np.ndarray]) -> LaneSummary:
        raise NotImplementedError(
            "CLRNet is not installed. Either install CLRNet and implement this method, "
            "or use --lane-backend heuristic (the default)."
        )


def build_lane_estimator(backend: str):
    backend = backend.lower()
    if backend == "heuristic":
        return HeuristicLaneEstimator()
    if backend == "clrnet":
        return CLRLaneEstimator()
    raise ValueError(f"Unsupported lane backend: {backend}")
