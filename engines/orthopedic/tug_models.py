"""Pydantic schemas for the Timed Up and Go (TUG) test.

Mirrors the structure returned by tug_engine.analyze_tug() so the
FastAPI response model maps 1:1 to the JSON the frontend consumes.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─── Per-phase result ────────────────────────────────────────────
class TUGPhase(BaseModel):
    """One of the 5 TUG sub-phases.

    Phase-specific fields are only populated when meaningful:
      - step_count / step_length_*: only Phase B (walk-out), Phase D
        (walk-back), Phase C (turn — count only, no length)
      - cadence_steps_per_min / walking_speed_mps: Phase B + D only
    """

    phase: Literal["sit_to_stand", "walk_out", "turn", "walk_back", "stand_to_sit"]
    duration_sec: float = Field(..., ge=0)
    start_frame: int
    end_frame: int

    # Optional per-phase metrics (None when not applicable / not detected).
    step_count: Optional[int] = None
    step_length_l_px: Optional[float] = None
    step_length_r_px: Optional[float] = None
    cadence_steps_per_min: Optional[float] = None
    walking_speed_mps: Optional[float] = None

    # Per-phase truncation flag — set when the algorithm couldn't find a
    # clean end-of-phase boundary and clamped at the video end.
    truncated: bool = False


# ─── Key-frame screenshot ────────────────────────────────────────
class TUGKeyFrame(BaseModel):
    """Annotated screenshot at one of the 5 key moments in the test."""

    label: Literal[
        "test_start",
        "end_of_sit_to_stand",
        "start_of_turn",
        "end_of_turn",
        "test_end",
    ]
    frame_index: int
    # data URL — JPEG-encoded snapshot with skeleton overlay drawn on it
    image_data_url: str


# ─── Flags ──────────────────────────────────────────────────────
class TUGFlag(BaseModel):
    code: Literal[
        "turn_time_excessive",
        "turn_steps_excessive",
        "phase_truncated",
        "turn_undetected",
        "no_strikes_detected",
        "test_too_fast",
    ]
    severity: Literal["info", "warning", "concern"]
    message: str


TUGClassification = Literal[
    "normal",                # total < 10 s
    "mild_fall_risk",        # 10-13.5 s
    "elevated_fall_risk",    # 13.5-20 s
    "significant_fall_risk", # > 20 s
]


# ─── Full result ─────────────────────────────────────────────────
class TUGResult(BaseModel):
    """Complete TUG analysis result for one trial."""

    # ── Aggregate ──────────────────────────────────────────────
    total_time_sec: float = Field(..., ge=0)
    classification: TUGClassification

    # ── 5 sub-phases (in chronological order) ─────────────────
    sit_to_stand: TUGPhase
    walk_out: TUGPhase
    turn: TUGPhase
    walk_back: TUGPhase
    stand_to_sit: TUGPhase

    # ── Flags / warnings ──────────────────────────────────────
    flags: list[TUGFlag] = Field(default_factory=list)

    # ── Patient context for the report ────────────────────────
    patient_age: Optional[int] = None
    # Age-matched norm threshold (seconds) — None if age not provided
    age_norm_threshold_sec: Optional[float] = None
    age_norm_passed: Optional[bool] = None

    # ── Pre-built plain-language interpretation ───────────────
    interpretation: str

    # ── Annotated screenshots at the 5 key moments ────────────
    key_frames: list[TUGKeyFrame] = Field(default_factory=list)

    # ── Source video metadata ─────────────────────────────────
    fps: float
    total_frames: int


# ─── API response wrapper ────────────────────────────────────────
class TUGResponse(BaseModel):
    success: bool
    data: Optional[TUGResult] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None
