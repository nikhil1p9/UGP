from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─── Vocabulary ───────────────────────────────────────────────────────────────
SpeedBucket    = Literal["stopped", "slow", "moderate", "fast"]
EventType      = Literal["normal", "near_miss", "collision"]
CollisionPhase = Literal["before", "during", "after", "unknown"]
ActorType      = Literal["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]
ActionType     = Literal["braking", "turning", "lane_change", "crossing", "overtaking", "stopped", "moving"]
LanePosition   = Literal["left", "center", "right", "shoulder", "unknown"]
RoadType       = Literal["intersection", "highway", "urban_road", "parking_lot", "unknown"]
Lighting       = Literal["day", "night", "unknown"]
Surface        = Literal["wet", "dry", "unknown"]


# ─── Public output models ─────────────────────────────────────────────────────

class ActorDetail(BaseModel):
    type: ActorType
    actions: list[ActionType] = Field(default_factory=list)
    lane_position: LanePosition = "unknown"
    vehicle_speed: SpeedBucket = "stopped"


class SceneInfo(BaseModel):
    road_type: RoadType = "unknown"
    lighting: Lighting = "unknown"
    weather: str = "unknown"
    surface: Surface = "unknown"


# class AnalysisResult(BaseModel):
#     event_type: EventType
#     collision_phase: CollisionPhase
#     actors: list[ActorType]
#     actor_count: int
#     actor_details: list[ActorDetail]
#     scene: SceneInfo
#     lane_count: int | None
#     lane_position: LanePosition
#     ego_vehicle_present: bool
class AnalysisResult(BaseModel):
    start_time: float            # <--- NEW FIELD
    end_time: float              # <--- NEW FIELD
    event_type: EventType
    collision_phase: CollisionPhase
    actors: list[ActorType]
    actor_count: int
    actor_details: list[ActorDetail]
    scene: SceneInfo
    lane_count: int | None
    lane_position: LanePosition
    ego_vehicle_present: bool


# ─── Internal: used by lane estimators ───────────────────────────────────────

class LaneSummary(BaseModel):
    estimated_lane_count: int | None = None
    visible_lane_markings: int = 0
    ego_lane_position: str = "unknown"
    confidence: float = 0.0
    source: str = "heuristic"


# ─── Internal: VLM structured output patch ───────────────────────────────────
# Only the fields the VLM is asked to refine; merged into AnalysisResult after.

class VLMActorPatch(BaseModel):
    type: ActorType
    actions: list[ActionType] = Field(default_factory=list)
    lane_position: LanePosition = "unknown"
    vehicle_speed: SpeedBucket = "stopped"


class VLMPatch(BaseModel):
    event_type: EventType
    collision_phase: CollisionPhase
    scene: SceneInfo
    actors: list[VLMActorPatch] = Field(default_factory=list)
    ego_vehicle_present: bool = False
