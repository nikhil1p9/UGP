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
        # edges = cv2.Canny(blurred, 80, 180)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            # threshold=60,
            threshold=50,
            minLineLength=max(width // 8, 40),
            # maxLineGap=30,
            maxLineGap=20,
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
        # merge_distance = width * 0.08
        merge_distance = width * 0.04
        for value in bottom_intersections[1:]:
            if abs(value - clustered[-1]) > merge_distance:
                clustered.append(value)
        visible_markings = len(clustered)
        lane_count = max(1, min(visible_markings + 1, 6))
        confidence = min(1.0, visible_markings / 4.0)
        return LaneEstimate(lane_count=lane_count, visible_lane_markings=visible_markings, confidence=confidence)


class CLRLaneEstimator:
    """Backend that delegates lane detection to the CLRNet architecture."""
    
    def __init__(self, config_path: str, weight_path: str, conf_threshold: float = 0.4):
        import torch
        from clrnet.utils.config import Config
        from clrnet.models.registry import build_net
        from clrnet.utils.net_utils import load_network

        self.cfg = Config.fromfile(config_path)
        self.conf_threshold = conf_threshold
        
        # Build and load the model
        self.net = build_net(self.cfg)
        self.net = torch.nn.DataParallel(self.net).cuda()
        load_network(self.net, weight_path)
        self.net.eval()
        
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        
        # ImageNet standardization parameters expected by CLRNet backbones
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def estimate(self, frames: list[np.ndarray]) -> LaneSummary:
        import torch
        
        if not frames:
            return LaneSummary()

        # Use the middle frame of the time chunk for a stable estimate
        target_frame = frames[len(frames) // 2]
        
        # 1. Preprocess: Resize, Normalize, Standardize
        img = cv2.resize(target_frame, (self.img_w, self.img_h))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # Convert to tensor: HWC -> CHW, and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

        # 2. Run Inference
        with torch.no_grad():
            output = self.net(img_tensor)
        
        # 3. Post-process to extract lanes
        # get_lanes returns a list of lanes for each image in the batch
        predictions = self.net.module.get_lanes(output)
        
        # Extract the lanes from the first (and only) image in our batch
        valid_lanes = [lane for lane in predictions[0] if lane.metadata['conf'] > self.conf_threshold]
        lane_count = len(valid_lanes)

        # 4. Compile metrics
        ego_position = "center" if lane_count >= 2 else "unknown"
        avg_conf = float(np.mean([lane.metadata['conf'] for lane in valid_lanes])) if valid_lanes else 0.0

        return LaneSummary(
            estimated_lane_count=lane_count,
            visible_lane_markings=lane_count,
            ego_lane_position=ego_position,
            confidence=avg_conf,
            source="clrnet"
        )


def build_lane_estimator(backend: str, config_path: str | None = None, weight_path: str | None = None):
    backend = backend.lower()
    if backend == "heuristic":
        return HeuristicLaneEstimator()
    if backend == "clrnet":
        if not config_path or not weight_path:
            raise ValueError("The 'clrnet' backend requires both a config_path and a weight_path.")
        return CLRLaneEstimator(config_path, weight_path)
    raise ValueError(f"Unsupported lane backend: {backend}")