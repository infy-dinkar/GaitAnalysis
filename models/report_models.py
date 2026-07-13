"""Pydantic schemas for report endpoints.

Report = a saved analysis result (gait, biomech, or posture) linked to
a patient. Stores the complete metrics blob + Plotly figures so the
frontend can re-render the original report later without recomputing.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── Save payload ──────────────────────────────────────────────────
class ReportCreate(BaseModel):
    """Payload for POST /api/patients/{patient_id}/reports.

    The frontend builds this after running the analysis (gait via
    server, biomech/posture via TF.js in browser) and posts it to
    persist the result against a patient record.
    """

    module: Literal[
        "gait",
        "biomech",
        "posture",
        "trendelenburg",
        "single_leg_squat",
        "sit_to_stand",
        "chair_stand_30s",
        "single_leg_stance",
        "four_stage_balance",
        "tug",
        "sppb",
        "slr",
        "ake",
        "modified_thomas",
        "forward_lunge",
        "sts_quality",
        "tandem_walk",
        "pronator_drift",
        "functional_reach",
        # D-series functional jump tests. Both were shipped in the
        # frontend (CMJCapture / SingleLegHopCapture) but never
        # added to this allowlist — the resulting 422 was rendered
        # as "[object Object]" by the client because the Pydantic
        # error detail array can't stringify cleanly.
        "counter_movement_jump",
        "single_leg_hop",
        "tuck_jump",
        "overhead_squat",
        # Rehab game sessions — one report per exercise session
        # (K1 Squat, K5 Wall Sit, ...). Payload shape lives in
        # `metrics` (provider-agnostic dict) so all 24 exercises share
        # this single literal.
        "rehab",
    ]

    # Biomech-specific (optional for other modules)
    body_part: Optional[Literal["shoulder", "neck", "knee", "hip", "ankle"]] = None
    movement: Optional[str] = Field(default=None, max_length=64)
    side: Optional[Literal["left", "right"]] = None

    # The analysis result blob — flexible shape per module.
    # For gait:   { metrics_total, metrics_clean, joint_angles, ... }
    # For biomech: { peak_angle, peak_magnitude, status, percentage, ... }
    # For posture: { front: {...}, side: {...}, findings: [...] }
    metrics: dict[str, Any] = Field(default_factory=dict)

    # Plotly figure JSON — array of traces + layout. Optional for biomech
    # where the chart can be derived from peak + range.
    figures: list[dict[str, Any]] = Field(default_factory=list)

    # Free-text observations / interpretation (per-module shape).
    observations: dict[str, Any] = Field(default_factory=dict)

    # Metadata about the source video (gait + biomech upload modes).
    video_filename: Optional[str] = Field(default=None, max_length=200)
    video_size_bytes: Optional[int] = Field(default=None, ge=0)

    # Spec Section 2 (a): raw landmark stream as JSON. Currently used
    # by the posture module — saves the front + side keypoint arrays
    # so saved sessions can be re-rendered (annotation overlay, side-
    # by-side comparison, future trend tracking) without keeping the
    # original photo on disk. Optional for backward compatibility
    # with reports saved before this field existed.
    # Shape: { "<view-name>": [ { x, y, score?, name? }, ... ] | null }
    keypoints: Optional[dict[str, Optional[list[dict[str, Any]]]]] = None


# ─── Public report response ────────────────────────────────────────
class Report(BaseModel):
    id: str
    patient_id: str
    doctor_id: str
    module: str
    body_part: Optional[str] = None
    movement: Optional[str] = None
    side: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    observations: dict[str, Any] = Field(default_factory=dict)
    video_filename: Optional[str] = None
    video_size_bytes: Optional[int] = None
    # Optional raw-landmark stream (posture sessions). See ReportCreate.
    keypoints: Optional[dict[str, Optional[list[dict[str, Any]]]]] = None
    created_at: datetime


class ReportSummary(BaseModel):
    """Lightweight version of Report — used in list views to avoid sending
    the (potentially large) figures payload across the wire."""

    id: str
    patient_id: str
    doctor_id: str
    module: str
    body_part: Optional[str] = None
    movement: Optional[str] = None
    side: Optional[str] = None
    created_at: datetime


class ReportListResponse(BaseModel):
    success: bool = True
    data: list[ReportSummary]
    total: int


class ReportDeleteResponse(BaseModel):
    success: bool = True
    deleted_report_id: str
