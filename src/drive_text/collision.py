# collision_logic.py  — complete replacement
"""
Collision inference built on three pillars:
  1. TTC (Time-To-Collision) — the physically grounded signal
  2. IOU trend — confirmatory, never the sole trigger
  3. Global scene shake — for ego-vehicle impacts only

False-positive root causes addressed:
  • Side-by-side cars: filtered by relative lateral vs vertical approach
  • Tracker ID jumps: filtered by confidence gate on the detection
  • Trucks passing close: require BOTH TTC < threshold AND IOU growth
  • Rough road vibration: shake requires multi-frame streak + object count
  • collided_pairs poisoning: pairs expire after MAX_PAIR_AGE frames
"""

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

EventType      = Literal["collision", "near_miss", "normal"]
CollisionPhase = Literal["before", "during", "after", "unknown"]

# ─── Tuneable constants ────────────────────────────────────────────────────────

# TTC thresholds (seconds)
TTC_COLLISION   = 1.5   # below this → collision imminent
TTC_NEAR_MISS   = 3.0   # below this → near miss

# IOU thresholds
IOU_COLLISION   = 0.25  # peak IOU must exceed this (confirmatory only)
IOU_NEAR_MISS   = 0.12

# Confidence gate — ignore detections below this (kills tracker-jump FPs)
MIN_CONFIDENCE  = 0.45

# Approach: fraction of frame_width the bottom-centre distance must close per frame
APPROACH_RATE   = 0.015

# Lateral filter: if horizontal closing rate >> vertical, it's a lane pass not a crash
LATERAL_RATIO_MAX = 2.5   # dx_close / dy_close must be < this for collision

# Ego-collision (camera impact) thresholds
EGO_TTC_THRESH  = 0.8   # seconds — very imminent
EGO_WIDTH_FRAC  = 0.42
EGO_BOTTOM_FRAC = 0.80
EGO_GROWTH_FRAMES = 2   # how many of last 4 intervals must show growth

# Shake
SHAKE_JUMP_FRAC       = 0.09   # fraction of frame_height for a single-object jump
SHAKE_MIN_OBJECTS     = 2
SHAKE_RATIO           = 0.40
SHAKE_CONFIRM_FRAMES  = 2

# Horizon: ignore objects entirely above this (background traffic)
HORIZON_FRAC    = 0.28

# Minimum track length before we trust any signal
MIN_TRACK_LEN   = 4

# How many frames before a collided_pair entry expires
MAX_PAIR_AGE    = 90   # ~3 seconds at 30fps

# Class pairs that can physically collide  (symmetric)
COLLIDABLE = {
    frozenset({"car", "car"}),
    frozenset({"car", "truck"}),
    frozenset({"car", "bus"}),
    frozenset({"car", "motorcycle"}),
    frozenset({"car", "person"}),
    frozenset({"truck", "truck"}),
    frozenset({"truck", "bus"}),
    frozenset({"truck", "motorcycle"}),
    frozenset({"truck", "person"}),
    frozenset({"bus", "motorcycle"}),
    frozenset({"bus", "person"}),
    frozenset({"motorcycle", "motorcycle"}),
    frozenset({"motorcycle", "person"}),
}

# ─── Session state (one instance per video) ───────────────────────────────────

@dataclass
class CollisionSessionState:
    """
    Pass one instance into every infer_event call for the same video.
    Replaces the bare `collided_pairs` set.
    """
    # pair_id → frame_index when it was first confirmed
    _pairs: dict[tuple[int, int], int] = field(default_factory=dict)
    # consecutive shake frames
    _shake_streak: int = 0

    def mark_collision(self, pair_id: tuple[int, int], frame_idx: int) -> None:
        self._pairs[pair_id] = frame_idx

    def is_known(self, pair_id: tuple[int, int]) -> bool:
        return pair_id in self._pairs

    def expire_old_pairs(self, current_frame: int) -> None:
        """Remove pairs that are older than MAX_PAIR_AGE frames."""
        expired = [k for k, v in self._pairs.items()
                   if current_frame - v > MAX_PAIR_AGE]
        for k in expired:
            del self._pairs[k]

    def update_shake(self, shaking: bool) -> bool:
        """Returns True once shake has been sustained for SHAKE_CONFIRM_FRAMES."""
        self._shake_streak = (self._shake_streak + 1) if shaking else 0
        return self._shake_streak >= SHAKE_CONFIRM_FRAMES

    @property
    def collided_pairs(self) -> set[tuple[int, int]]:
        """Back-compat: read-only view of current pair IDs."""
        return set(self._pairs.keys())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _bottom_centre(bbox: tuple) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, bbox[3]


def _recent_avg_confidence(track, n: int = 5) -> float:
    if not track.confidences:
        return 0.0
    return float(sum(track.confidences[-n:]) / len(track.confidences[-n:]))


def _pair_id(a, b) -> tuple[int, int]:
    return tuple(sorted([a.track_id, b.track_id]))


def _is_collidable_pair(a, b) -> bool:
    """Both objects must be a known collidable class combination."""
    return frozenset({a.class_name, b.class_name}) in COLLIDABLE


# ─── Main entry point ─────────────────────────────────────────────────────────

def infer_event(
    tracks: list,
    frame_width: int,
    frame_height: int,
    fps: float,
    config,
    session: CollisionSessionState,
    current_frame: int = 0,
) -> tuple[EventType, CollisionPhase]:
    """
    Parameters
    ----------
    session       : CollisionSessionState — shared across all frames of one video
    current_frame : frame index (used for pair expiry)
    """

    session.expire_old_pairs(current_frame)

    # Pre-filter: drop tracks that are too short or low-confidence
    valid = [
        t for t in tracks
        if len(t.bboxes) >= MIN_TRACK_LEN
        and _recent_avg_confidence(t) >= MIN_CONFIDENCE
    ]

    # ── 1. Ego-collision: object crashing INTO the dashcam ────────────────────
    for track in valid:
        box = track.bboxes[-1]
        w   = box[2] - box[0]

        if w < frame_width * EGO_WIDTH_FRAC:
            continue
        if box[3] < frame_height * EGO_BOTTOM_FRAC:
            continue

        # TTC check first — this is the strongest signal
        ttc = track.estimate_ttc(fps)
        if ttc is not None and ttc < EGO_TTC_THRESH:
            return "collision", "during"

        # Fallback: sustained bbox area growth over last 4 frames
        if len(track.bboxes) >= 5:
            def _area(b):
                return (b[2] - b[0]) * (b[3] - b[1])
            recent = track.bboxes[-5:]
            growth_events = sum(
                1 for i in range(1, len(recent))
                if _area(recent[i]) > _area(recent[i - 1]) * 1.06  # >6% growth
            )
            if growth_events >= EGO_GROWTH_FRAMES:
                return "collision", "during"

    # ── 2. Global scene shake ─────────────────────────────────────────────────
    nearby = [
        t for t in valid
        if t.bboxes and t.bboxes[-1][3] > frame_height * HORIZON_FRAC
    ]

    if len(nearby) >= SHAKE_MIN_OBJECTS:
        shake_votes = 0
        for track in nearby:
            if len(track.points) < 2:
                continue
            dx = track.points[-1][1] - track.points[-2][1]
            dy = track.points[-1][2] - track.points[-2][2]
            jump = math.hypot(dx, dy)
            # Must move in BOTH axes to rule out smooth camera pan
            if (jump > frame_height * SHAKE_JUMP_FRAC
                    and abs(dx) > frame_width  * 0.04
                    and abs(dy) > frame_height * 0.04):
                shake_votes += 1

        shaking = (shake_votes >= SHAKE_MIN_OBJECTS
                   and shake_votes >= len(nearby) * SHAKE_RATIO)
        if session.update_shake(shaking):
            return "collision", "during"
    else:
        session.update_shake(False)

    # ── 3. Known collided pairs ───────────────────────────────────────────────
    for idx in range(len(valid)):
        for jdx in range(idx + 1, len(valid)):
            t_a, t_b = valid[idx], valid[jdx]
            pid = _pair_id(t_a, t_b)

            if not session.is_known(pid):
                continue

            result = _score_pair(t_a, t_b, frame_width, frame_height, fps, config)

            speed_a = t_a.speed_bucket(frame_width, frame_height)
            speed_b = t_b.speed_bucket(frame_width, frame_height)
            still   = (speed_a in ("stopped", "slow")
                       and speed_b in ("stopped", "slow"))

            if result["collision"] and not still:
                return "collision", "during"
            return "collision", "after"

    # ── 4. New pairwise detection ─────────────────────────────────────────────
    best_near = False

    for idx in range(len(valid)):
        for jdx in range(idx + 1, len(valid)):
            t_a, t_b = valid[idx], valid[jdx]
            pid = _pair_id(t_a, t_b)

            if session.is_known(pid):
                continue
            if not _is_collidable_pair(t_a, t_b):
                continue

            result = _score_pair(t_a, t_b, frame_width, frame_height, fps, config)

            if result["collision"]:
                session.mark_collision(pid, current_frame)
                return "collision", "during"

            if result["near_miss"]:
                best_near = True

    if best_near:
        return "near_miss", "before"

    return "normal", "unknown"


# ─── Core pair scorer ─────────────────────────────────────────────────────────

def _score_pair(
    track_a,
    track_b,
    frame_width: int,
    frame_height: int,
    fps: float,
    config,
) -> dict:
    """
    Returns {"collision": bool, "near_miss": bool}.

    Decision tree (in priority order):
      A. TTC gate        — if either object's TTC is safe, cap the verdict
      B. Lateral filter  — closing mostly sideways → lane pass, not crash
      C. Approach gate   — must be genuinely converging before overlap
      D. IOU peak        — confirmatory overlap check
      E. Post-impact     — velocity change after peak
    """
    result = {"collision": False, "near_miss": False}

    if not track_a.bboxes or not track_b.bboxes:
        return result

    # Build frame-aligned histories
    dict_a = {pt[0]: (bbox, conf)
              for pt, bbox, conf in zip(track_a.points, track_a.bboxes, track_a.confidences)}
    dict_b = {pt[0]: (bbox, conf)
              for pt, bbox, conf in zip(track_b.points, track_b.bboxes, track_b.confidences)}

    common = sorted(set(dict_a) & set(dict_b))
    if len(common) < MIN_TRACK_LEN:
        return result

    # Horizon filter
    max_y_a = max(v[0][3] for v in dict_a.values())
    max_y_b = max(v[0][3] for v in dict_b.values())
    if max_y_a < frame_height * HORIZON_FRAC and max_y_b < frame_height * HORIZON_FRAC:
        return result

    # Per-frame arrays
    iou_hist   = []
    dist_hist  = []   # bottom-centre euclidean
    dx_hist    = []   # horizontal component (signed)
    dy_hist    = []   # vertical component (signed, positive = converging in Y)
    conf_hist  = []

    for fi in common:
        ba, ca = dict_a[fi]
        bb, cb = dict_b[fi]

        iou_hist.append(_iou(ba, bb))
        conf_hist.append(min(ca, cb))   # weakest link

        bcx_a, by_a = _bottom_centre(ba)
        bcx_b, by_b = _bottom_centre(bb)

        dist_hist.append(math.hypot(bcx_a - bcx_b, by_a - by_b))
        dx_hist.append(abs(bcx_a - bcx_b))
        dy_hist.append(abs(by_a  - by_b))

    max_iou  = max(iou_hist)
    peak_idx = iou_hist.index(max_iou)

    # ── A. Confidence gate ────────────────────────────────────────────────────
    # If detection quality is poor around the peak, don't trust geometry
    peak_conf = conf_hist[peak_idx]
    if peak_conf < MIN_CONFIDENCE:
        return result

    # ── Near-miss check (before collision logic) ──────────────────────────────
    if max_iou > IOU_NEAR_MISS:
        peak_bottom = max(
            dict_a[common[peak_idx]][0][3],
            dict_b[common[peak_idx]][0][3]
        )
        if peak_bottom > frame_height * 0.40:
            result["near_miss"] = True

    iou_threshold = getattr(config, "collision_iou_threshold", IOU_COLLISION)
    if max_iou < iou_threshold:
        return result

    # ── B. TTC gate ───────────────────────────────────────────────────────────
    # If BOTH objects have safe TTC, no collision regardless of IOU
    ttc_a = track_a.estimate_ttc(fps)
    ttc_b = track_b.estimate_ttc(fps)

    ttc_a_safe = ttc_a is None or ttc_a > TTC_NEAR_MISS
    ttc_b_safe = ttc_b is None or ttc_b > TTC_NEAR_MISS

    if ttc_a_safe and ttc_b_safe:
        # Both objects are not closing — whatever caused IOU isn't a crash
        result["near_miss"] = False
        return result

    # Near-miss upgrade: at least one object has concerning TTC
    ttc_a_alarm = ttc_a is not None and ttc_a < TTC_NEAR_MISS
    ttc_b_alarm = ttc_b is not None and ttc_b < TTC_NEAR_MISS
    if ttc_a_alarm or ttc_b_alarm:
        result["near_miss"] = True

    # Collision-level TTC: at least one object is critically close
    ttc_a_crit = ttc_a is not None and ttc_a < TTC_COLLISION
    ttc_b_crit = ttc_b is not None and ttc_b < TTC_COLLISION
    ttc_critical = ttc_a_crit or ttc_b_crit

    # ── C. Lateral filter ─────────────────────────────────────────────────────
    # Compute how much of the closing was horizontal vs vertical.
    # A car passing in an adjacent lane closes mostly laterally.
    if peak_idx > 0:
        dx_close = dx_hist[0]  - dx_hist[peak_idx]   # positive = converging
        dy_close = dy_hist[0]  - dy_hist[peak_idx]

        # If closing was overwhelmingly horizontal (lane pass), skip
        if dx_close > 0 and dy_close >= 0:
            lateral_ratio = dx_close / max(dy_close, frame_height * 0.01)
            if lateral_ratio > LATERAL_RATIO_MAX and not ttc_critical:
                return result   # lateral pass, not a crash

    # ── D. Approach gate ─────────────────────────────────────────────────────
    # Count consecutive frames of distance closing before the peak
    approach_streak = 0
    max_approach    = 0
    for i in range(1, peak_idx + 1):
        if dist_hist[i] < dist_hist[i - 1]:
            approach_streak += 1
            max_approach = max(max_approach, approach_streak)
        else:
            approach_streak = 0

    # Require at least 2 consecutive approaching frames OR critical TTC
    if max_approach < 2 and not ttc_critical:
        return result

    # ── E. Post-impact check ─────────────────────────────────────────────────
    # After a real impact: relative velocity either freezes (tangled)
    # or reverses (bounce). A clean pass-through shows continued separation.
    post_ok = False
    if peak_idx < len(dist_hist) - 1:
        pre_v  = dist_hist[peak_idx - 1] - dist_hist[peak_idx] if peak_idx > 0 else 0
        post_v = dist_hist[peak_idx] - dist_hist[peak_idx + 1]
        tangled = abs(post_v) < frame_width * 0.01
        bounced = pre_v > 0 and post_v < -pre_v * 0.4
        post_ok = tangled or bounced
    else:
        # Track dies at peak = tracker death from severe deformation
        post_ok = True

    # ── Final verdict ─────────────────────────────────────────────────────────
    # REQUIRE: (TTC critical) AND (approach confirmed) AND (post-impact OR extreme IOU)
    if ttc_critical and max_approach >= 2 and (post_ok or max_iou > 0.50):
        result["collision"] = True
        result["near_miss"] = False
        return result

    # Softer path: very strong IOU + approach + TTC warning (not yet critical)
    if max_iou > 0.40 and max_approach >= 3 and (ttc_a_alarm or ttc_b_alarm):
        result["collision"] = True
        result["near_miss"] = False

    return result

def depth_aware_collision_check(
    track_a,          # TrackState
    track_b,          # TrackState
    frame_width: int,
    frame_height: int,
    config,           # AnalyzerConfig
    fps: float,
) -> tuple[float, bool]:
    """
    Returns (collision_score [0,1], is_near_miss).

    Score components (each 0–1, weighted sum):
      S1  approach_score   – were they converging before overlap?
      S2  overlap_score    – how deep / sustained was the IOU peak?
      S3  depth_score      – are they in the same depth band (Y-position)?
      S4  post_score       – did motion change after peak (energy absorbed)?

    Side-by-side filter:
      If the vehicles share horizontal extent but have similar bottom-Y
      (adjacent lanes, same depth) AND they were not approaching each
      other vertically, the pair is flagged as "lateral" and scored 0.
    """
    if not track_a.bboxes or not track_b.bboxes:
        return 0.0, False

    dict_a = {pt[0]: bbox for pt, bbox in zip(track_a.points, track_a.bboxes)}
    dict_b = {pt[0]: bbox for pt, bbox in zip(track_b.points, track_b.bboxes)}

    common = sorted(set(dict_a) & set(dict_b))
    if len(common) < MIN_COMMON_FRAMES:
        return 0.0, False

    # Horizon filter – both tracks must have been below the horizon at some point
    max_y_a = max(b[3] for b in dict_a.values())
    max_y_b = max(b[3] for b in dict_b.values())
    if max_y_a < frame_height * HORIZON_FRAC and max_y_b < frame_height * HORIZON_FRAC:
        return 0.0, False

    # ── build per-frame histories ─────────────────────────────────────────────
    iou_hist  = []
    dist_hist = []   # euclidean distance between bottom-centres
    dy_hist   = []   # vertical component of centre approach (positive = closer)

    for frame_idx in common:
        ba, bb = dict_a[frame_idx], dict_b[frame_idx]
        iou_hist.append(_iou(ba, bb))

        cx_a = (ba[0] + ba[2]) / 2
        cx_b = (bb[0] + bb[2]) / 2
        by_a, by_b = ba[3], bb[3]

        dist_hist.append(math.hypot(cx_a - cx_b, by_a - by_b))
        # positive means b is below a (approaching from above in camera space)
        dy_hist.append(by_b - by_a)

    max_iou   = max(iou_hist)
    peak_idx  = iou_hist.index(max_iou)
    is_near   = False

    # ── near-miss detection (low IOU + proximity in lower frame) ─────────────
    for i, frame_idx in enumerate(common):
        ba, bb = dict_a[frame_idx], dict_b[frame_idx]
        if iou_hist[i] > NEAR_MISS_IOU and max(ba[3], bb[3]) > frame_height * NEAR_MISS_BOTTOM_FRAC:
            is_near = True
            break

    # ── short-circuit: IOU never reached the base threshold ──────────────────
    iou_threshold = getattr(config, "collision_iou_threshold", COLLISION_IOU_THRESH)
    if max_iou < iou_threshold:
        return 0.0, is_near

    # ── side-by-side lateral filter ───────────────────────────────────────────
    # If the two bottom-Y values are very close (same depth row) AND
    # the y-component of distance was never really changing, it's lane traffic.
    bottom_diff = abs(
        dict_a[common[peak_idx]][3] - dict_b[common[peak_idx]][3]
    ) / frame_height
    dy_change = max(dy_hist) - min(dy_hist)   # total vertical approach variance

    if bottom_diff < LATERAL_EXEMPT_FRAC and dy_change < frame_height * 0.05:
        # They are at the same depth and never approached vertically → skip
        return 0.0, is_near

    # ── S1: approach score ────────────────────────────────────────────────────
    # Count how many consecutive frames before the peak had closing distance.
    # More confirmed closing → higher confidence.
    approach_frames = 0
    for i in range(1, peak_idx + 1):
        if dist_hist[i] < dist_hist[i - 1]:
            approach_frames += 1
        else:
            # Reset on any non-approaching frame to require consecutive closure
            approach_frames = 0
    approach_score = min(1.0, approach_frames / APPROACH_CONFIRM)

    # ── S2: overlap score (sustained + deep IOU) ──────────────────────────────
    # Reward IOU that is both high and lasts more than one frame.
    frames_above_thresh = sum(1 for v in iou_hist if v > iou_threshold)
    overlap_score = min(1.0, (max_iou / 0.80) * 0.6 +
                             (frames_above_thresh / max(len(iou_hist), 1)) * 0.4)

    # ── S3: depth score (objects close in screen-Y = same real-world depth) ──
    # Objects that are far apart in Y are on different roads / levels.
    depth_score = max(0.0, 1.0 - bottom_diff / 0.25)   # full score if <5 % diff

    # ── S4: post-impact score (velocity change after peak) ────────────────────
    # After a real impact, relative distance either freezes (tangled) or
    # jumps (bounce).  A large CHANGE is the signal.
    post_score = 0.0
    if peak_idx < len(dist_hist) - 1:
        pre_velocity  = (dist_hist[peak_idx - 1] - dist_hist[peak_idx]
                         if peak_idx > 0 else 0.0)
        post_velocity = dist_hist[peak_idx] - dist_hist[peak_idx + 1]
        # Velocity reversal (bounce) or near-zero post speed (tangled)
        tangled = abs(post_velocity) < frame_width * 0.01
        bounced = pre_velocity > 0 and post_velocity < -pre_velocity * 0.5
        if tangled or bounced:
            post_score = 1.0
        else:
            post_score = 0.3   # weak signal
    else:
        # Track ended at peak – consistent with tracker death on hard impact
        post_score = 0.8

    # ── Extreme IOU shortcut ──────────────────────────────────────────────────
    # Boxes merged >55 % AND they were approaching → near-certain collision.
    if max_iou > EXTREME_IOU_THRESH and approach_score > 0:
        return min(1.0, 0.85 + approach_score * 0.15), is_near

    # ── Weighted fusion ───────────────────────────────────────────────────────
    fused = (
        approach_score * 0.35 +
        overlap_score  * 0.30 +
        depth_score    * 0.20 +
        post_score     * 0.15
    )
    return fused, is_near