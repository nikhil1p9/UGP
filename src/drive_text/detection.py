from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees
from typing import Iterable

import numpy as np
from ultralytics import YOLO

from .schema import SpeedBucket


COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

SUPPORTED_LABELS = set(COCO_CLASS_NAMES.values())


@dataclass(slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None  # set by ByteTrack; None when using IoU fallback

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)


@dataclass(slots=True)
class TrackState:
    track_id: int
    class_name: str
    points: list[tuple[int, float, float]] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    def add(self, frame_index: int, detection: Detection) -> None:
        cx, cy = detection.center
        self.points.append((frame_index, cx, cy))
        self.bboxes.append(detection.bbox)
        self.confidences.append(detection.confidence)

    @property
    def first_seen(self) -> int:
        return self.points[0][0]

    @property
    def last_seen(self) -> int:
        return self.points[-1][0]

    def average_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    def speed_bucket(self, frame_width: int, frame_height: int) -> SpeedBucket:
        motion = normalized_motion_score(self.points, self.bboxes, frame_width, frame_height)
        if motion < 0.003:
            return "stopped"
        if motion < 0.012:
            return "slow"
        if motion < 0.03:
            return "moderate"
        return "fast"

    def infer_actions(self, frame_width: int, frame_height: int) -> list[str]:
        if len(self.points) < 2:
            return ["stopped"]
        speed = self.speed_bucket(frame_width, frame_height)
        if speed == "stopped":
            return ["stopped"]

        actions: list[str] = ["moving"]

        x0, y0 = self.points[0][1], self.points[0][2]
        x1, y1 = self.points[-1][1], self.points[-1][2]
        dx = x1 - x0
        dy = y1 - y0
        lateral_ratio = abs(dx) / max(abs(dy), 1.0)

        if lateral_ratio > 0.45 and abs(dx) > frame_width * 0.08:
            actions.append("lane_change")
        elif abs(dx) > frame_width * 0.05:
            actions.append("turning")

        # Braking: motion in the first half noticeably greater than second half
        mid = len(self.points) // 2
        if mid >= 2:
            def _avg_step(pts: list) -> float:
                return sum(
                    ((pts[i][1] - pts[i-1][1])**2 + (pts[i][2] - pts[i-1][2])**2) ** 0.5
                    for i in range(1, len(pts))
                ) / max(len(pts) - 1, 1)
            if _avg_step(self.points[:mid]) > _avg_step(self.points[mid:]) * 1.6:
                actions.append("braking")

        # Crossing: pedestrian moving primarily laterally
        if self.class_name == "person" and abs(dx) > abs(dy) and abs(dx) > frame_width * 0.05:
            if "lane_change" not in actions and "turning" not in actions:
                actions.append("crossing")

        return dedupe(actions)
    
    def estimate_ttc(self, fps: float = 30.0) -> float | None:
        """Estimates Time-To-Collision (TTC) based on bounding box expansion."""
        if len(self.bboxes) < 5:
            return None # Need history to calculate rate of change
            
        # Get width of current frame and a frame slightly in the past
        current_w = self.bboxes[-1][2] - self.bboxes[-1][0]
        past_w = self.bboxes[-5][2] - self.bboxes[-5][0]
        
        # Calculate rate of change of width
        dw = current_w - past_w
        dt = 5.0 / fps # Time elapsed over 5 frames
        
        rate_of_change = dw / dt
        
        if rate_of_change <= 0:
            return float('inf') # Object is moving away or staying at same distance
            
        ttc = current_w / rate_of_change
        return ttc


class ObjectDetector:
    def __init__(self, model_name: str, confidence: float) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = COCO_CLASS_NAMES.get(class_id)
                if class_name not in SUPPORTED_LABELS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=float(box.conf.item()),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return detections

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Detection + ByteTrack on a single frame.

        Uses Ultralytics' native ByteTrack (persist=True) so that track IDs
        stay consistent across sequential frames of the same video.
        Falls through to the caller's IoU fallback when box.id is None.
        """
        results = self.model.track(
            frame, persist=True, conf=self.confidence, verbose=False, tracker="bytetrack.yaml"
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = COCO_CLASS_NAMES.get(class_id)
                if class_name not in SUPPORTED_LABELS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id.item()) if box.id is not None else None
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=float(box.conf.item()),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        track_id=track_id,
                    )
                )
        return detections


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.2, max_frame_gap: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap
        self._next_track_id = 1

    def update_tracks(
        self,
        tracks: list[TrackState],
        detections: list[Detection],
        frame_index: int,
    ) -> list[TrackState]:
        unmatched = detections.copy()
        for track in tracks:
            if frame_index - track.last_seen > self.max_frame_gap:
                continue

            best_detection = None
            best_iou = 0.0
            for detection in unmatched:
                if detection.class_name != track.class_name:
                    continue
                overlap = iou(track.bboxes[-1], detection.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_detection = detection
            if best_detection is not None and best_iou >= self.iou_threshold:
                track.add(frame_index, best_detection)
                unmatched.remove(best_detection)

        for detection in unmatched:
            track = TrackState(track_id=self._next_track_id, class_name=detection.class_name)
            self._next_track_id += 1
            track.add(frame_index, detection)
            tracks.append(track)
        return tracks


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = max(area_a + area_b - inter_area, 1e-6)
    return inter_area / union


# def normalized_motion_score(
#     points: list[tuple[int, float, float]],
#     bboxes: list[tuple[float, float, float, float]],
#     frame_width: int,
#     frame_height: int,
# ) -> float:
#     if len(points) < 2:
#         return 0.0
#     scores: list[float] = []
#     diagonal = max((frame_width ** 2 + frame_height ** 2) ** 0.5, 1.0)
#     for idx in range(1, len(points)):
#         _, x0, y0 = points[idx - 1]
#         _, x1, y1 = points[idx]
#         pixel_distance = float(np.hypot(x1 - x0, y1 - y0))
#         bbox_height = max((bboxes[idx][3] - bboxes[idx][1] + bboxes[idx - 1][3] - bboxes[idx - 1][1]) / 2.0, 1.0)
#         scores.append(pixel_distance / max(diagonal * 0.25 + bbox_height, 1.0))
#     return float(np.mean(scores))
def normalized_motion_score(
    points: list[tuple[int, float, float]],
    bboxes: list[tuple[float, float, float, float]],
    frame_width: int,
    frame_height: int,
) -> float:
    if len(points) < 2:
        return 0.0
    scores: list[float] = []
    
    for idx in range(1, len(points)):
        _, x0, y0 = points[idx - 1]
        _, x1, y1 = points[idx]
        pixel_distance = float(np.hypot(x1 - x0, y1 - y0))
        
        # FIX: Use the Y-coordinate (bottom of the bounding box) for depth perspective
        # instead of the bounding box height.
        bottom_y = bboxes[idx][3] 
        depth_scale = max(bottom_y / frame_height, 0.1) # 0.1 to 1.0 depending on distance
        
        # Normalize distance by frame width and depth scale
        scores.append((pixel_distance / frame_width) / depth_scale)
        
    return float(np.mean(scores))


def closest_pair_distance(
    tracks: Iterable[TrackState],
    frame_width: int,
    frame_height: int,
) -> float | None:
    centers = []
    for track in tracks:
        if not track.points:
            continue
        _, x, y = track.points[-1]
        centers.append((x, y))
    if len(centers) < 2:
        return None
    diagonal = max((frame_width ** 2 + frame_height ** 2) ** 0.5, 1.0)
    best = None
    for idx in range(len(centers)):
        for jdx in range(idx + 1, len(centers)):
            distance = float(np.hypot(centers[idx][0] - centers[jdx][0], centers[idx][1] - centers[jdx][1])) / diagonal
            best = distance if best is None else min(best, distance)
    return best


def max_overlap_between_tracks(tracks: Iterable[TrackState]) -> float:
    boxes = [track.bboxes[-1] for track in tracks if track.bboxes]
    if len(boxes) < 2:
        return 0.0
    best = 0.0
    for idx in range(len(boxes)):
        for jdx in range(idx + 1, len(boxes)):
            best = max(best, iou(boxes[idx], boxes[jdx]))
    return best


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output

def depth_aware_collision_check(
    track_a: TrackState, 
    track_b: TrackState, 
    frame_width: int, 
    frame_height: int,
    fps: float = 30.0
) -> tuple[bool, bool]:
    """
    Returns (is_collision, is_near_miss) in O(1) time.
    """
    if not track_a.bboxes or not track_b.bboxes:
        return False, False

    box_a = track_a.bboxes[-1]
    box_b = track_b.bboxes[-1]

    # 1. Tire-level (Bottom-Center) Distance
    cx_a = (box_a[0] + box_a[2]) / 2.0
    bottom_y_a = box_a[3] 
    
    cx_b = (box_b[0] + box_b[2]) / 2.0
    bottom_y_b = box_b[3]
    
    # Normalize by frame dimensions
    dx = (cx_a - cx_b) / frame_width
    dy = (bottom_y_a - bottom_y_b) / frame_height
    pseudo_depth_distance = float(np.hypot(dx, dy))

    # 2. Bounding Box Overlap (IoU)
    overlap = iou(box_a, box_b)

    # 3. Time-to-Collision (TTC) - Are they rapidly expanding?
    # (Assuming you implemented the estimate_ttc method on TrackState)
    ttc_a = track_a.estimate_ttc(fps) or float('inf')
    ttc_b = track_b.estimate_ttc(fps) or float('inf')
    min_ttc = min(ttc_a, ttc_b)

    # Logic: 
    # A true collision means they overlap laterally AND their tires are at the same depth
    is_collision = overlap > 0.05 and pseudo_depth_distance < 0.08
    
    # A near miss means they are physically close, closing fast, but not touching
    is_near_miss = not is_collision and pseudo_depth_distance < 0.15 and min_ttc < 2.0

    return is_collision, is_near_miss
