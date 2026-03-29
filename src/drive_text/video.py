from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class VideoFrame:
    index: int
    image: np.ndarray
    timestamp: float


@dataclass(slots=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def info(self) -> VideoInfo:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        return VideoInfo(self.path, frame_count, fps, width, height)

    def sample_frames(self, every_n_frames: int) -> list[VideoFrame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

        frames: list[VideoFrame] = []
        index = 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % max(every_n_frames, 1) == 0:
                timestamp = index / fps if fps > 0 else 0.0
                frames.append(VideoFrame(index=index, image=image, timestamp=timestamp))
            index += 1
        capture.release()
        return frames
