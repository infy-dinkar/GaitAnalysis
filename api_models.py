"""
api_models.py
Pydantic response models for the MotionLens FastAPI layer.

These models are consumed by the Next.js frontend. They expose every
field the UI needs in JSON-safe form (no numpy arrays, no NaN, no None
where a number is expected).

The actual conversion from the engine dicts happens in api_helpers.py.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Common
# ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "MotionLens API"
    version: str = "1.0.0"


class APIError(BaseModel):
    detail: str


class PatientInfo(BaseModel):
    name: Optional[str] = None
    height_cm: float = 170.0


# ══════════════════════════════════════════════════════════════════════
# GAIT — full feature-parity response
# ══════════════════════════════════════════════════════════════════════
class VideoInfo(BaseModel):
    duration_sec: float
    fps: float
    total_frames: int
    calibration_mm_per_px: Optional[float] = None
    valid_passes: int = 0
    frames_used: int = 0
    ankle_baseline_left: float = 0.0
    ankle_baseline_right: float = 0.0
    ankle_baseline_method: str = "unknown"
    ankle_baseline_n_frames: int = 0


class MetricsBlock(BaseModel):
    """One of metrics_total / metrics_clean."""
    step_count: int = 0
    cadence: Optional[float] = None              # steps/min
    symmetry: Optional[float] = None             # 0..1
    knee_peak: Optional[float] = None            # peak swing flexion (deg)
    stride_cv: Optional[float] = None            # percent
    step_length: Optional[float] = None
    step_length_unit: str = "m"
    torso_lean: Optional[float] = None           # mean (deg)
    step_time: Optional[float] = None            # mean step interval (s)
    window_seconds: float = 0.0
    window_description: str = ""
    # Clean-only fields (None on the total block):
    validated_passes: Optional[int] = None
    video_coverage_pct: Optional[float] = None


class JointDetail(BaseModel):
    peak: Optional[float] = None
    min: Optional[float] = None
    rom: Optional[float] = None
    mean: Optional[float] = None
    time_series: list[Optional[float]] = Field(default_factory=list)


class JointAnglesBlock(BaseModel):
    left_knee: JointDetail
    right_knee: JointDetail
    left_hip: JointDetail
    right_hip: JointDetail
    left_ankle: JointDetail
    right_ankle: JointDetail


class GaitCycleCurve(BaseModel):
    """Per-joint per-leg cycle curve. mean/std are length-101 arrays
    (0–100% of the gait cycle); K is the count of strides averaged."""
    mean_curve: list[Optional[float]] = Field(default_factory=list)
    std_curve: list[Optional[float]] = Field(default_factory=list)
    K: int = 0


class CycleSideBlock(BaseModel):
    cycles_accepted: int = 0
    cycles_rejected_amplitude: int = 0
    cycles_kept: int = 0
    cycles_rejected_too_long: int = 0
    cycles_rejected_too_short: int = 0
    hip: GaitCycleCurve
    knee: GaitCycleCurve
    ankle: GaitCycleCurve


class NormalRefCurve(BaseModel):
    """Healthy-adult reference: mean curve + lower/upper SD-widened band."""
    mean_curve: list[float]
    lower_band: list[float]
    upper_band: list[float]


class NormalReferenceBlock(BaseModel):
    hip: NormalRefCurve
    knee: NormalRefCurve
    ankle: NormalRefCurve


class GaitCycleData(BaseModel):
    left: CycleSideBlock
    right: CycleSideBlock
    normal_reference: NormalReferenceBlock
    stance_phase_pct: int = 60


class NormalizedSeriesPair(BaseModel):
    left: list[Optional[float]] = Field(default_factory=list)
    right: list[Optional[float]] = Field(default_factory=list)


class NormalizedOverview(BaseModel):
    time_axis: list[float] = Field(default_factory=list)
    leg_angle: NormalizedSeriesPair
    knee_flexion: NormalizedSeriesPair
    hip_flexion: NormalizedSeriesPair
    ankle_deflection: NormalizedSeriesPair


class HeelPositionTab(BaseModel):
    time_axis: list[float] = Field(default_factory=list)
    left_x: list[Optional[float]] = Field(default_factory=list)
    right_x: list[Optional[float]] = Field(default_factory=list)
    # Strike events (for scatter overlay):
    left_strikes_t: list[float] = Field(default_factory=list)
    right_strikes_t: list[float] = Field(default_factory=list)
    left_strikes_x: list[float] = Field(default_factory=list)
    right_strikes_x: list[float] = Field(default_factory=list)
    left_count: int = 0
    right_count: int = 0


class StepLengthTab(BaseModel):
    left_values: list[float] = Field(default_factory=list)
    right_values: list[float] = Field(default_factory=list)
    left_mean: float = 0.0
    right_mean: float = 0.0
    unit: str = "m"


class TimingTab(BaseModel):
    left_intervals: list[float] = Field(default_factory=list)
    right_intervals: list[float] = Field(default_factory=list)
    left_mean: float = 0.0
    right_mean: float = 0.0


class TorsoLeanTab(BaseModel):
    time_axis: list[float] = Field(default_factory=list)
    angles: list[Optional[float]] = Field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0


class AnkleTrajectoryTab(BaseModel):
    time_axis: list[float] = Field(default_factory=list)
    left_x: list[Optional[float]] = Field(default_factory=list)
    right_x: list[Optional[float]] = Field(default_factory=list)
    left_y: list[Optional[float]] = Field(default_factory=list)
    right_y: list[Optional[float]] = Field(default_factory=list)


class PassSegmentDTO(BaseModel):
    start_sec: float
    end_sec: float
    core_start_sec: float
    core_end_sec: float
    direction: int


class TabsData(BaseModel):
    heel_position: HeelPositionTab
    step_length: StepLengthTab
    timing: TimingTab
    torso_lean: TorsoLeanTab
    ankle_trajectory: AnkleTrajectoryTab
    pass_segments: list[PassSegmentDTO] = Field(default_factory=list)


class ObservationsBucketed(BaseModel):
    hip: list[str] = Field(default_factory=list)
    knee: list[str] = Field(default_factory=list)
    ankle: list[str] = Field(default_factory=list)
    overall: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class GaitData(BaseModel):
    patient_info: PatientInfo
    video_info: VideoInfo
    walking_direction: str = "Unknown"
    metrics_total: MetricsBlock
    metrics_clean: MetricsBlock
    joint_angles: JointAnglesBlock
    gait_cycle_data: Optional[GaitCycleData] = None
    normalized_overview: NormalizedOverview
    tabs_data: TabsData
    observations: ObservationsBucketed


class GaitResponse(BaseModel):
    success: bool
    data: Optional[GaitData] = None
    error: Optional[str] = None
    # Soft-warning fields surfaced when the upload validation passes but
    # one of the input quality dimensions is below the recommended bar
    # (e.g. 24-29 FPS, sub-5-sec clip). The frontend renders these as a
    # neutral banner above the report; they do not block analysis.
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# BIOMECH (unchanged)
# ══════════════════════════════════════════════════════════════════════
class BiomechData(BaseModel):
    body_part: str
    movement: str
    side: Optional[str] = None
    peak_angle: Optional[float] = None
    peak_magnitude: float = 0.0
    reference_range: list[float]
    target: float
    percentage: float
    status: str
    valid_frames: int = 0
    total_frames: int = 0
    fps: float = 0.0
    interpretation: str = ""


class BiomechResponse(BaseModel):
    success: bool
    data: Optional[BiomechData] = None
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# LIVE BIOMECH (per-frame, low-latency)
# ══════════════════════════════════════════════════════════════════════
class LandmarkPoint(BaseModel):
    x: float           # normalized [0, 1]
    y: float           # normalized [0, 1]
    visibility: float  # 0..1


class LiveBiomechFrameData(BaseModel):
    status: str  # "good" | "low_visibility" | "no_landmarks"
    landmarks: list[LandmarkPoint] = []
    current_angle: Optional[float] = None
    current_magnitude: float = 0.0


class LiveBiomechFrameResponse(BaseModel):
    success: bool
    data: Optional[LiveBiomechFrameData] = None
    error: Optional[str] = None
