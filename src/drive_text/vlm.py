from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np
from openai import OpenAI

from .schema import VLMPatch


# ─── System prompt ────────────────────────────────────────────────────────
# Inspired by SeeUnsafe (arXiv:2501.10604) and TrafficVLM (arXiv:2404.09275):
# explicitly naming the colored boxes and defining each event class forces the
# model to produce grounded, schema-conformant answers.
_SYSTEM_PROMPT = """\
You are a traffic accident analyst inspecting annotated driving video frames.

Each frame has COLOR-CODED bounding boxes drawn by a CV pipeline:
  Red    = car        |  Blue   = truck       |  Orange = bus
  Green  = pedestrian |  Yellow = motorcycle  |  Cyan   = bicycle
Each box carries a label like "car #3" where the number is the persistent track ID.

Your job is to produce a structured JSON refinement of the CV analysis.

STRICT RULES
1. vehicle_speed → MUST be one of: stopped | slow | moderate | fast
   - stopped  : actor does not move across frames
   - slow     : clearly moving but low speed (pedestrian pace to ~20 km/h)
   - moderate : typical urban traffic speed (~20–60 km/h)
   - fast     : highway or aggressive speed (>60 km/h apparent)
2. event_type → MUST be one of: normal | near_miss | collision
   - normal     : routine traffic, no sudden deviations
   - near_miss  : actors come extremely close but avoid contact
   - collision  : actors make direct visible contact / impact
3. collision_phase → MUST be one of: before | during | after | unknown
4. actor actions → MUST each be one of:
   braking | turning | lane_change | crossing | overtaking | stopped | moving
5. lane_position → MUST be one of: left | center | right | shoulder | unknown
6. scene.road_type → MUST be one of: intersection | highway | urban_road | parking_lot | unknown
7. scene.lighting → MUST be one of: day | night | unknown
8. scene.surface  → MUST be one of: wet | dry | unknown
9. Only report facts directly observable in the frames.
   Do NOT invent injuries, license plates, identities, or exact speeds.
10. ego_vehicle_present: true if the camera is mounted on a vehicle in the scene.
"""


class VLMRefiner:
    """Sends annotated key frames + current CV result to an OpenAI-compatible
    VLM and returns a validated VLMPatch.

    Uses `client.beta.chat.completions.parse(response_format=VLMPatch)` which
    applies OpenAI Structured Outputs — the API enforces the Pydantic schema
    server-side so no free-text parsing or post-hoc validation is needed.
    """

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def refine(self, annotated_frames: list[np.ndarray], current_result: dict) -> VLMPatch:
        image_content = [self._encode_frame(frame) for frame in annotated_frames]
        response = self.client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            temperature=0,
            response_format=VLMPatch,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Annotated key frames from the driving video are attached below.\n"
                                "Current CV-based analysis (refine where the frames give better evidence):\n"
                                + json.dumps(current_result, indent=2)
                            ),
                        },
                        *image_content,
                    ],
                },
            ],
        )
        patch = response.choices[0].message.parsed
        if patch is None:
            raise RuntimeError("VLM returned a null parsed response")
        return patch

    def _encode_frame(self, frame: np.ndarray) -> dict:
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("cv2.imencode failed for VLM frame")
        encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"},
        }
