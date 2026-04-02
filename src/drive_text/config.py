# src/drive_text/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AnalyzerConfig:
    input_path: Path
    output_path: Path
    interval_seconds: int = 5      # <--- NEW FIELD
    sample_every_n_frames: int = 3
    min_detection_confidence: float = 0.35
    detector_model: str = "yolov8n.pt"
    lane_backend: str = "heuristic"
    enable_vlm: bool = False
    max_vlm_frames: int = 6
    collision_iou_threshold: float = 0.08
    near_miss_distance_threshold: float = 0.12