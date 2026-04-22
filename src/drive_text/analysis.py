from __future__ import annotations

import json
import cv2
import numpy as np

from .config import AnalyzerConfig
from .detection import Detection, ObjectDetector, SimpleTracker, TrackState, closest_pair_distance, max_overlap_between_tracks, depth_aware_collision_check
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
        # self.lane_estimator = build_lane_estimator(config.lane_backend)
        self.lane_estimator = build_lane_estimator(backend=self.config.lane_backend,
        config_path=self.config.clrnet_config_path,
        weight_path=self.config.clrnet_weight_path)

    def analyze(self) -> list[AnalysisResult]:
        info = self.reader.info()
        sampled_frames = self.reader.sample_frames(self.config.sample_every_n_frames)
        warnings: list[str] = []

        # 1. Run detection and tracking over the whole video to keep IDs consistent
        tracks_dict: dict[int, TrackState] = {}
        fallback_tracks: list[TrackState] = []
        frame_record: dict[int, list[Detection]] = {}
        frame_time_map: dict[int, float] = {}

        for frame in sampled_frames:
            frame_time_map[frame.index] = frame.timestamp
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

        all_tracks = list(tracks_dict.values()) + fallback_tracks

        # 2. Split analysis into 5-second buckets
        interval = self.config.interval_seconds
        num_intervals = int(np.ceil(info.duration_seconds / interval)) if info.duration_seconds > 0 else 0
        
        final_results: list[AnalysisResult] = []
        
        collided_pairs: set[tuple[int, int]] = set()

        for i in range(num_intervals):
            start_time = i * interval
            end_time = min((i + 1) * interval, info.duration_seconds)
            
            chunk_frames = [f for f in sampled_frames if start_time <= f.timestamp < end_time]
            if not chunk_frames:
                continue
                
            chunk_images = [f.image for f in chunk_frames]
            chunk_frame_record = {f.index: frame_record[f.index] for f in chunk_frames}

            # Filter tracks to only include behavior that happened within THIS 5-second chunk
            chunk_tracks = []
            for t in all_tracks:
                pts, boxes, confs = [], [], []
                for idx, (f_idx, cx, cy) in enumerate(t.points):
                    if start_time <= frame_time_map[f_idx] < end_time:
                        pts.append((f_idx, cx, cy))
                        boxes.append(t.bboxes[idx])
                        confs.append(t.confidences[idx])
                
                if pts:
                    new_t = TrackState(track_id=t.track_id, class_name=t.class_name)
                    new_t.points = pts
                    new_t.bboxes = boxes
                    new_t.confidences = confs
                    chunk_tracks.append(new_t)

            # Analyze the slice
            lane_summary = self.lane_estimator.estimate(chunk_images[: min(len(chunk_images), 12)])
            scene = infer_scene(chunk_images, lane_summary.estimated_lane_count)
            # event_type, collision_phase = infer_event(chunk_tracks, info.width, info.height, self.config)
            event_type, collision_phase = infer_event(chunk_tracks, info.width, info.height, info.fps, self.config,collided_pairs)
            actor_details = build_actor_details(chunk_tracks, info.width, info.height, lane_summary.estimated_lane_count)
            ego_present = detect_ego_vehicle(chunk_frame_record, info.height)

            seen: list[ActorType] = []
            for a in actor_details:
                if a.type not in seen:
                    seen.append(a.type)

            lane_pos = _map_lane_position(lane_summary.ego_lane_position)

            result = AnalysisResult(
                start_time=start_time,
                end_time=end_time,
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

            # VLM Refinement per chunk (if enabled)
            if self.config.enable_vlm:
                try:
                    refiner = VLMRefiner()
                    key_frames = _select_key_frames(chunk_frames, self.config.max_vlm_frames)
                    annotated = [
                        draw_annotations(f.image, chunk_frame_record.get(f.index, []))
                        for f in key_frames
                    ]
                    patch = refiner.refine(annotated, result.model_dump())
                    result.event_type = patch.event_type
                    result.collision_phase = patch.collision_phase
                    result.scene = patch.scene
                    result.ego_vehicle_present = patch.ego_vehicle_present
                    if patch.actors:
                        # Map patch back
                        pass # Kept standard for brevity, apply original VLM actor patch logic here
                except Exception as exc:
                    warnings.append(f"VLM refinement skipped for chunk {start_time}-{end_time}: {exc}")

            final_results.append(result)

        # 3. Write final JSON array to file
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_json = json.dumps([r.model_dump() for r in final_results], indent=2)
        self.config.output_path.write_text(output_json, encoding="utf-8")
        
        return final_results


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
    fps: float,
    config: AnalyzerConfig,
    collided_pairs: set[tuple[int, int]]
) -> tuple[EventType, CollisionPhase]:
    
    # 1. Ego-Collision Detection: Direct Impact (Dashcam rear-ends a car)
    for track in tracks:
        if not track.bboxes: continue
        box = track.bboxes[-1]
        w = box[2] - box[0]
        # If a car covers >60% of the screen width and its tires touch the bottom of the frame
        if w > frame_width * 0.60 and box[3] >= frame_height * 0.85:
            return "collision", "during"

    # 2. Ego-Collision Detection: Global Scene Shake (Side/Offset Impact)
    if len(tracks) >= 2:
        shake_scores = []
        for track in tracks:
            if len(track.points) >= 3:
                # Calculate the pixel displacement between the last two frames
                recent_jump = ((track.points[-1][1] - track.points[-2][1])**2 + 
                               (track.points[-1][2] - track.points[-2][2])**2) ** 0.5
                shake_scores.append(recent_jump)
        
        if len(shake_scores) >= 2:
            avg_shake = sum(shake_scores) / len(shake_scores)
            # If the average frame-to-frame jump of background objects exceeds 15% of the screen,
            if avg_shake > frame_height * 0.15: 
                return "collision", "during"

    # 3. Handle Already Collided Pairs First (Fixing the "After" Phase)
    for idx in range(len(tracks)):
        for jdx in range(idx + 1, len(tracks)):
            t_a = tracks[idx]
            t_b = tracks[jdx]
            pair_id = tuple(sorted([t_a.track_id, t_b.track_id]))
            
            if pair_id in collided_pairs:
                # They crashed previously. Check if they are still tangled (during) or separated/stopped (after)
                is_col, _ = depth_aware_collision_check(t_a, t_b, frame_width, frame_height, config, fps)
                
                speed_a = t_a.speed_bucket(frame_width, frame_height)
                speed_b = t_b.speed_bucket(frame_width, frame_height)
                
                # If they are still actively driving into each other
                if is_col and (speed_a not in ("stopped", "slow") or speed_b not in ("stopped", "slow")):
                    return "collision", "during"
                else:
                    # They separated (is_col is False) OR they ground to a halt
                    return "collision", "after"

    # 4. Detect New Pairwise Collisions
    for idx in range(len(tracks)):
        for jdx in range(idx + 1, len(tracks)):
            t_a = tracks[idx]
            t_b = tracks[jdx]
            pair_id = tuple(sorted([t_a.track_id, t_b.track_id]))
            
            # Skip if we already evaluated them in the block above
            if pair_id in collided_pairs:
                continue
                
            is_col, is_near = depth_aware_collision_check(
                t_a, t_b, frame_width, frame_height, config, fps
            )
            
            if is_col:
                collided_pairs.add(pair_id)
                return "collision", "during"
                    
            if is_near:
                return "near_miss", "before"

    return "normal", "unknown"