from __future__ import annotations

import cv2
import numpy as np

from .config import AnalyzerConfig
from .detection import Detection, ObjectDetector, SimpleTracker, TrackState, closest_pair_distance, max_overlap_between_tracks
from .lane import build_lane_estimator
from .schema import (
    ActorDetail, ActorType, AnalysisResult, CollisionPhase, EventType,
    LanePosition, LaneSummary, RoadType, SceneInfo, SpeedBucket,
)
from .video import VideoFrame, VideoReader
from .vlm import VLMRefiner


# â”€â”€â”€ YOLO class name â†’ user-facing ActorType â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ACTOR_TYPE_MAP: dict[str, ActorType] = {
    "person":     "pedestrian",
    "bicycle":    "bicycle",
    "car":        "car",
    "motorcycle": "motorcycle",
    "bus":        "bus",
    "truck":      "truck",
}

# â”€â”€â”€ Color map for annotated frames (BGR for OpenCV) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car":        (0,   0,   255),
    "truck":      (255, 0,   0),
    "bus":        (0,   165, 255),
    "motorcycle": (0,   255, 255),
    "bicycle":    (255, 255, 0),
    "person":     (0,   255, 0),
}


def draw_annotations(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = _CLASS_COLORS.get(det.class_name, (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = det.class_name if det.track_id is None else f"{det.class_name} #{det.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, cv2.FILLED)
        cv2.putText(
            annotated, label, (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return annotated


class VideoAnalyzer:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.reader = VideoReader(config.input_path)
        self.detector = ObjectDetector(config.detector_model, config.min_detection_confidence)
        self._fallback_tracker = SimpleTracker()
        self.lane_estimator = build_lane_estimator(config.lane_backend)

    def analyze(self) -> AnalysisResult:
        info = self.reader.info()
        sampled_frames = self.reader.sample_frames(self.config.sample_every_n_frames)
        warnings: list[str] = []

        tracks_dict: dict[int, TrackState] = {}
        fallback_tracks: list[TrackState] = []
        frame_record: dict[int, list[Detection]] = {}

        for frame in sampled_frames:
            try:
                detections = self.detector.track(frame.image)
            except Exception as exc:
                warnings.append(f"ByteTrack unavailable on frame {frame.index} ({exc}); using IoU fallback.")
                detections = self.detector.detect(frame.image)

            frame_record[frame.index] = detections

            tracked   = [d for d in detections if d.track_id is not None]
            untracked = [d for d in detections if d.track_id is None]

            for det in tracked:
                if det.track_id not in tracks_dict:
                    tracks_dict[det.track_id] = TrackState(track_id=det.track_id, class_name=det.class_name)
                tracks_dict[det.track_id].add(frame.index, det)

            if untracked:
                fallback_tracks = self._fallback_tracker.update_tracks(fallback_tracks, untracked, frame.index)

        tracks = list(tracks_dict.values()) + fallback_tracks
        sampled_images = [f.image for f in sampled_frames]

        lane_summary   = self.lane_estimator.estimate(sampled_images[: min(len(sampled_images), 12)])
        scene          = infer_scene(sampled_images, lane_summary.estimated_lane_count)
        event_type, collision_phase = infer_event(tracks, info.width, info.height, self.config)
        actor_details  = build_actor_details(tracks, info.width, info.height, lane_summary.estimated_lane_count)
        ego_present    = detect_ego_vehicle(frame_record, info.height)

        # Deduplicated actor type list
        seen: list[ActorType] = []
        for a in actor_details:
            if a.type not in seen:
                seen.append(a.type)

        lane_pos = _map_lane_position(lane_summary.ego_lane_position)

        result = AnalysisResult(
            event_type=event_type,
            collision_phase=collision_phase,
            actors=seen,
            actor_count=len(actor_details),
            actor_details=actor_details,
            scene=scene,
            lane_count=lane_summary.estimated_lane_count,
            lane_position=lane_pos,
            ego_vehicle_present=ego_present,
        )

        if self.config.enable_vlm:
            try:
                refiner    = VLMRefiner()
                key_frames = _select_key_frames(sampled_frames, self.config.max_vlm_frames)
                annotated  = [
                    draw_annotations(f.image, frame_record.get(f.index, []))
                    for f in key_frames
                ]
                patch = refiner.refine(annotated, result.model_dump())
                result.event_type       = patch.event_type
                result.collision_phase  = patch.collision_phase
                result.scene            = patch.scene
                result.ego_vehicle_present = patch.ego_vehicle_present
                if patch.actors:
                    result.actor_details = [
                        ActorDetail(
                            type=a.type,
                            actions=a.actions,
                            lane_position=a.lane_position,
                            vehicle_speed=a.vehicle_speed,
                        )
                        for a in patch.actors
                    ]
                    seen_vlm: list[ActorType] = []
                    for a in result.actor_details:
                        if a.type not in seen_vlm:
                            seen_vlm.append(a.type)
                    result.actors      = seen_vlm
                    result.actor_count = len(result.actor_details)
            except Exception as exc:
                warnings.append(f"VLM refinement skipped: {exc}")

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


# â”€â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _select_key_frames(frames: list[VideoFrame], max_frames: int) -> list[VideoFrame]:
    if len(frames) <= max_frames:
        return frames
    step = max(len(frames) // max_frames, 1)
    return frames[::step][:max_frames]


def _map_lane_position(raw: str) -> LanePosition:
    mapping: dict[str, LanePosition] = {
        "left": "left", "right": "right", "center": "center",
        "shoulder": "shoulder",
    }
    return mapping.get(raw, "unknown")


def build_actor_details(
    tracks: list[TrackState],
    frame_width: int,
    frame_height: int,
    lane_count: int | None,
) -> list[ActorDetail]:
    details: list[ActorDetail] = []
    for track in tracks:
        if len(track.points) < 2:
            continue
        actor_type = _ACTOR_TYPE_MAP.get(track.class_name, "car")
        raw_actions = track.infer_actions(frame_width, frame_height)
        # Filter to only valid ActionType literals
        _valid = {"braking", "turning", "lane_change", "crossing", "overtaking", "stopped", "moving"}
        actions = [a for a in raw_actions if a in _valid]
        details.append(ActorDetail(
            type=actor_type,
            actions=actions,
            lane_position=estimate_lane_position(track, frame_width, lane_count),
            vehicle_speed=track.speed_bucket(frame_width, frame_height),
        ))
    details.sort(key=lambda d: d.type)
    return details


def estimate_lane_position(track: TrackState, frame_width: int, lane_count: int | None) -> LanePosition:
    if lane_count is None or lane_count <= 1 or not track.points:
        return "unknown"
    _, x, _ = track.points[-1]
    lane_index = int(min(max((x / max(frame_width, 1)) * lane_count, 0), lane_count - 1))
    if lane_index == 0:
        return "left"
    if lane_index == lane_count - 1:
        return "right"
    return "center"


def detect_ego_vehicle(frame_record: dict[int, list[Detection]], frame_height: int) -> bool:
    """Heuristic: ego vehicle is likely present if any actor bbox nearly fills
    the bottom of the frame (i.e., the camera is mounted on a moving vehicle)."""
    threshold = frame_height * 0.85
    for detections in frame_record.values():
        for det in detections:
            if det.bbox[3] >= threshold:
                return True
    return False


def infer_scene(frames: list[np.ndarray], lane_count: int | None) -> SceneInfo:
    if not frames:
        return SceneInfo()
    brightness = sum(float(f.mean()) for f in frames[:10]) / min(len(frames), 10)
    lighting: str = "night" if brightness < 70 else "day"
    if lane_count and lane_count >= 4:
        road_type: RoadType = "highway"
    elif lane_count and lane_count <= 2:
        road_type = "intersection"
    else:
        road_type = "urban_road"
    return SceneInfo(lighting=lighting, road_type=road_type, weather="unknown", surface="unknown")


def infer_event(
    tracks: list[TrackState],
    frame_width: int,
    frame_height: int,
    config: AnalyzerConfig,
) -> tuple[EventType, CollisionPhase]:
    overlap = max_overlap_between_tracks(tracks)
    closest = closest_pair_distance(tracks, frame_width, frame_height)
    fast_tracks = [t for t in tracks if t.speed_bucket(frame_width, frame_height) in {"moderate", "fast"}]

    if overlap >= config.collision_iou_threshold and len(tracks) >= 2:
        return "collision", "during"

    if closest is not None and closest <= config.near_miss_distance_threshold and fast_tracks:
        return "near_miss", "before"

    return "normal", "unknown"

