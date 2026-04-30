"""
api_helpers.py
File handling + JSON-safe response formatting for the MotionLens API.

The engine modules return rich Python structures with numpy arrays and
np.nan values that don't survive json.dumps. This module is the only
place in the API layer that touches the engine output — it sanitises
everything into plain Python primitives before Pydantic validation.

Constraint: this file MUST NOT modify the engine modules. It only reads
their output.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any

import numpy as np

# Engine references (READ-ONLY). gait_plots' normal_*_reference helpers
# are PUBLIC functions used to expose the healthy-adult reference bands.
from gait_plots import (
    normal_hip_reference,
    normal_knee_reference,
    normal_ankle_reference,
)

from api_models import (
    AnkleTrajectoryTab,
    BiomechData,
    CycleSideBlock,
    GaitCycleCurve,
    GaitCycleData,
    GaitData,
    HeelPositionTab,
    JointAnglesBlock,
    JointDetail,
    MetricsBlock,
    NormalRefCurve,
    NormalReferenceBlock,
    NormalizedOverview,
    NormalizedSeriesPair,
    ObservationsBucketed,
    PassSegmentDTO,
    PatientInfo,
    StepLengthTab,
    TabsData,
    TimingTab,
    TorsoLeanTab,
    VideoInfo,
)

log = logging.getLogger(__name__)

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Visualisation widening of the normal-reference SD band — matches
# SD_SCALE in gait_plots.plot_gait_cycle_curves.
REF_SD_SCALE = 1.5


# ══════════════════════════════════════════════════════════════════════
# File handling
# ══════════════════════════════════════════════════════════════════════
def save_uploaded_video(upload_file_bytes: bytes, original_name: str) -> str:
    suffix = os.path.splitext(original_name or "video.mp4")[1].lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video extension: {suffix}. "
            f"Allowed: {sorted(ALLOWED_VIDEO_EXTS)}"
        )
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="motionlens_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(upload_file_bytes)
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise
    return path


def cleanup_temp_file(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("Failed to delete temp file %s: %s", path, e)


# ══════════════════════════════════════════════════════════════════════
# JSON sanitisation primitives
# ══════════════════════════════════════════════════════════════════════
def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _array_to_list(value)
    return value


def _nan_to_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _array_to_list(arr: Any) -> list:
    if arr is None:
        return []
    a = np.asarray(arr)
    if a.dtype.kind in ("i", "u", "b"):
        return a.tolist()
    out: list = []
    for v in a.tolist():
        if isinstance(v, list):
            out.append([_nan_to_none(x) for x in v])
        else:
            out.append(_nan_to_none(v))
    return out


def _floats_only(arr: Any) -> list[float]:
    """Same as _array_to_list but drops NaN/None entirely (for required-float lists)."""
    if arr is None:
        return []
    a = np.asarray(arr, dtype=float)
    return [float(v) for v in a.tolist() if not (math.isnan(v) or math.isinf(v))]


# ══════════════════════════════════════════════════════════════════════
# Joint summary
# ══════════════════════════════════════════════════════════════════════
def _summarise_joint_arr(arr: Any) -> JointDetail:
    series = _array_to_list(arr)
    if arr is None:
        return JointDetail()
    a = np.asarray(arr, dtype=float)
    valid = a[~np.isnan(a)]
    if valid.size == 0:
        return JointDetail(time_series=series)
    peak = float(np.max(valid))
    mn = float(np.min(valid))
    return JointDetail(
        peak=_scalar(peak),
        min=_scalar(mn),
        rom=_scalar(peak - mn),
        mean=_scalar(float(np.mean(valid))),
        time_series=series,
    )


# ══════════════════════════════════════════════════════════════════════
# Per-leg step length (NEW — splits engine's combined list by side)
# ══════════════════════════════════════════════════════════════════════
def _per_leg_step_lengths(
    ts: dict,
    left_strike_idx: np.ndarray,
    right_strike_idx: np.ndarray,
    fps: float,
    mpp: float | None,
    max_stride_sec: float = 2.0,
) -> tuple[list[float], list[float], str]:
    """Compute step length per side using the same algorithm as the
    engine (compute_metrics in gait_engine.py), but split by leg.
    Mirrors the engine's intervals < max_stride_sec cross-pass filter."""
    l_heel_px = np.asarray(ts.get("left_heel", {}).get("x_px", []), dtype=float)
    r_heel_px = np.asarray(ts.get("right_heel", {}).get("x_px", []), dtype=float)

    def _per_side(idx: np.ndarray, heel_px: np.ndarray) -> list[float]:
        if idx is None or len(idx) < 2 or heel_px.size == 0:
            return []
        idx = np.asarray(idx, dtype=int)
        # Filter out indices beyond the heel array length (defensive)
        idx = idx[(idx >= 0) & (idx < heel_px.size)]
        if len(idx) < 2:
            return []
        positions = heel_px[idx].astype(float)
        intervals = np.diff(idx) / fps
        strides = np.abs(np.diff(positions))
        keep = intervals < max_stride_sec
        return [float(s) for s in (strides[keep] / 2.0).tolist()]

    left = _per_side(left_strike_idx, l_heel_px)
    right = _per_side(right_strike_idx, r_heel_px)

    if mpp and mpp > 0:
        left = [v * float(mpp) for v in left]
        right = [v * float(mpp) for v in right]
        unit = "m"
    else:
        unit = "px"
    return left, right, unit


# ══════════════════════════════════════════════════════════════════════
# Observation bucketing
# ══════════════════════════════════════════════════════════════════════
def _bucket_observations(observations: list[str]) -> dict[str, list[str]]:
    """Keyword-bucket the engine's flat observation list into hip / knee /
    ankle / overall. Order is preserved within each bucket. The engine's
    interpret() output rarely produces hip- or ankle-specific lines, so
    those buckets are often empty — frontend shows an empty state."""
    hip: list[str] = []
    knee: list[str] = []
    ankle: list[str] = []
    overall: list[str] = []
    for raw in observations:
        text = str(raw)
        lower = text.lower()
        if "knee" in lower:
            knee.append(text)
        elif "hip" in lower:
            hip.append(text)
        elif "ankle" in lower or "heel" in lower or "foot" in lower:
            ankle.append(text)
        else:
            overall.append(text)
    return {"hip": hip, "knee": knee, "ankle": ankle, "overall": overall}


# ══════════════════════════════════════════════════════════════════════
# Normal reference curves (calls public helpers in gait_plots)
# ══════════════════════════════════════════════════════════════════════
def _normal_ref_block() -> NormalReferenceBlock:
    def _ref(joint_fn) -> NormalRefCurve:
        mean, sd = joint_fn()
        sd_widened = sd * REF_SD_SCALE
        return NormalRefCurve(
            mean_curve=[float(v) for v in mean.tolist()],
            lower_band=[float(v) for v in (mean - sd_widened).tolist()],
            upper_band=[float(v) for v in (mean + sd_widened).tolist()],
        )

    return NormalReferenceBlock(
        hip=_ref(normal_hip_reference),
        knee=_ref(normal_knee_reference),
        ankle=_ref(normal_ankle_reference),
    )


# ══════════════════════════════════════════════════════════════════════
# Block builders
# ══════════════════════════════════════════════════════════════════════
def _build_video_info(features: dict, height_cm: float) -> VideoInfo:
    fps = float(features.get("fps", 0.0) or 0.0)
    total_frames = int(features.get("total_frames", 0) or 0)
    duration_sec = float(features.get("duration_sec", 0.0) or 0.0)
    mpp = features.get("meters_per_pixel")
    cal_mm_per_px = float(mpp * 1000.0) if mpp else None
    abase = features.get("ankle_baseline", {}) or {}
    return VideoInfo(
        duration_sec=duration_sec,
        fps=fps,
        total_frames=total_frames,
        calibration_mm_per_px=cal_mm_per_px,
        valid_passes=int(features.get("num_passes", 0) or 0),
        frames_used=int(features.get("frames_used", 0) or 0),
        ankle_baseline_left=float(abase.get("offset_deg_left", 0.0) or 0.0),
        ankle_baseline_right=float(abase.get("offset_deg_right", 0.0) or 0.0),
        ankle_baseline_method=str(abase.get("method", "unknown")),
        ankle_baseline_n_frames=int(abase.get("n_frames", 0) or 0),
    )


def _build_metrics_block(
    metrics: dict,
    is_clean: bool,
    *,
    features: dict,
) -> MetricsBlock:
    """Map a metric dict (engine total_metrics or clean_metrics) into the API model."""
    step_data = metrics.get("step_data", {}) or {}
    knee = metrics.get("knee_angles", {}) or {}
    sl = metrics.get("step_length", {}) or {}
    timing = metrics.get("step_timing", {}) or {}
    torso = metrics.get("torso_lean", {}) or {}

    cadence = metrics.get("cadence", 0.0) or 0.0
    cadence_v = _scalar(cadence) if cadence > 0 else None

    sym = metrics.get("symmetry", None)
    sym_v = _scalar(sym) if (sym is not None and sym != 0) else _scalar(sym)

    knee_peak = knee.get("overall_peak", 0.0) or 0.0
    knee_peak_v = _scalar(knee_peak) if knee_peak > 0 else None

    stride_cv = metrics.get("stride_cv", 0.0) or 0.0
    stride_cv_v = _scalar(stride_cv) if stride_cv > 0 else None

    sl_mean = sl.get("mean", 0.0) or 0.0
    sl_v = _scalar(sl_mean) if sl_mean > 0 else None

    torso_v = _scalar(torso.get("mean", None))
    step_time_v = _scalar(timing.get("mean_step_time", None))
    if step_time_v == 0:
        step_time_v = None

    window_seconds = float(metrics.get("duration_sec", 0.0) or 0.0)

    if is_clean:
        total_frames = int(features.get("total_frames", 0) or 0)
        frames_used = int(features.get("frames_used", 0) or 0)
        coverage = (frames_used / total_frames * 100.0) if total_frames > 0 else 0.0
        num_passes = int(features.get("num_passes", 0) or 0)
        window_desc = (
            f"{window_seconds:.1f}s steady-state across {num_passes} validated "
            f"pass{'es' if num_passes != 1 else ''} ({coverage:.0f}% of video)"
        )
        return MetricsBlock(
            step_count=int(step_data.get("total_steps", 0) or 0),
            cadence=cadence_v,
            symmetry=sym_v,
            knee_peak=knee_peak_v,
            stride_cv=stride_cv_v,
            step_length=sl_v,
            step_length_unit=str(sl.get("unit", "m")),
            torso_lean=torso_v,
            step_time=step_time_v,
            window_seconds=window_seconds,
            window_description=window_desc,
            validated_passes=num_passes,
            video_coverage_pct=round(coverage, 1),
        )
    else:
        window_desc = f"{window_seconds:.1f}s — all frames included"
        return MetricsBlock(
            step_count=int(step_data.get("total_steps", 0) or 0),
            cadence=cadence_v,
            symmetry=sym_v,
            knee_peak=knee_peak_v,
            stride_cv=stride_cv_v,
            step_length=sl_v,
            step_length_unit=str(sl.get("unit", "m")),
            torso_lean=torso_v,
            step_time=step_time_v,
            window_seconds=window_seconds,
            window_description=window_desc,
        )


def _build_joint_angles(features: dict) -> JointAnglesBlock:
    knee = features.get("knee_angles", {}) or {}
    hip = features.get("hip_angles", {}) or {}
    ankle = features.get("ankle_angles", {}) or {}
    return JointAnglesBlock(
        left_knee=_summarise_joint_arr(knee.get("left")),
        right_knee=_summarise_joint_arr(knee.get("right")),
        left_hip=_summarise_joint_arr(hip.get("left")),
        right_hip=_summarise_joint_arr(hip.get("right")),
        left_ankle=_summarise_joint_arr(ankle.get("left")),
        right_ankle=_summarise_joint_arr(ankle.get("right")),
    )


def _curve(curve_dict: dict | None) -> GaitCycleCurve:
    d = curve_dict or {}
    return GaitCycleCurve(
        mean_curve=_array_to_list(d.get("mean")),
        std_curve=_array_to_list(d.get("std")),
        K=int(d.get("K", 0) or 0),
    )


def _build_gait_cycle_data(features: dict) -> GaitCycleData | None:
    gc = features.get("gait_cycle_curves") or {}
    if not gc:
        return None
    sr = features.get("strike_rejection", {}) or {}
    cdf = features.get("cycle_duration_filter", {}) or {}

    def _side(side: str) -> CycleSideBlock:
        sr_side = sr.get(side, {}) or {}
        cdf_side = cdf.get(side, {}) or {}
        return CycleSideBlock(
            cycles_accepted=int(sr_side.get("accepted", 0) or 0),
            cycles_rejected_amplitude=int(sr_side.get("rejected", 0) or 0),
            cycles_kept=int(cdf_side.get("kept", 0) or 0),
            cycles_rejected_too_long=int(cdf_side.get("rejected_long", 0) or 0),
            cycles_rejected_too_short=int(cdf_side.get("rejected_short", 0) or 0),
            hip=_curve(gc.get("hip", {}).get(side)),
            knee=_curve(gc.get("knee", {}).get(side)),
            ankle=_curve(gc.get("ankle", {}).get(side)),
        )

    try:
        return GaitCycleData(
            left=_side("left"),
            right=_side("right"),
            normal_reference=_normal_ref_block(),
            stance_phase_pct=60,
        )
    except Exception as e:
        log.warning("Could not build gait_cycle_data: %s", e)
        return None


def _build_normalized_overview(features: dict) -> NormalizedOverview:
    fps = float(features.get("fps", 0.0) or 0.0)
    n = int(features.get("total_frames", 0) or 0)
    time_axis = (np.arange(n) / fps).tolist() if fps > 0 else list(range(n))

    knee = features.get("knee_angles", {}) or {}
    hip = features.get("hip_angles", {}) or {}
    ankle = features.get("ankle_angles", {}) or {}
    leg = features.get("leg_angles", {}) or {}

    def _pair(d: dict) -> NormalizedSeriesPair:
        return NormalizedSeriesPair(
            left=_array_to_list(d.get("left")),
            right=_array_to_list(d.get("right")),
        )

    return NormalizedOverview(
        time_axis=[float(t) for t in time_axis],
        leg_angle=_pair(leg),
        knee_flexion=_pair(knee),
        hip_flexion=_pair(hip),
        ankle_deflection=_pair(ankle),
    )


def _build_tabs_data(features: dict, ts: dict) -> TabsData:
    fps = float(features.get("fps", 0.0) or 0.0)
    n = int(features.get("total_frames", 0) or 0)
    time_axis = (np.arange(n) / fps).tolist() if fps > 0 else list(range(n))

    clean = features.get("clean_metrics", {}) or {}
    step_data = clean.get("step_data", {}) or {}
    L_idx = np.asarray(step_data.get("left_indices", []), dtype=int)
    R_idx = np.asarray(step_data.get("right_indices", []), dtype=int)

    # ── Heel position tab ──────────────────────────────────────
    l_heel = ts.get("left_heel", {}) or {}
    r_heel = ts.get("right_heel", {}) or {}
    l_heel_x = np.asarray(l_heel.get("x", []), dtype=float)
    r_heel_x = np.asarray(r_heel.get("x", []), dtype=float)

    L_idx_clip = L_idx[(L_idx >= 0) & (L_idx < len(l_heel_x))] if len(l_heel_x) else np.array([], dtype=int)
    R_idx_clip = R_idx[(R_idx >= 0) & (R_idx < len(r_heel_x))] if len(r_heel_x) else np.array([], dtype=int)

    heel_tab = HeelPositionTab(
        time_axis=[float(t) for t in time_axis],
        left_x=_array_to_list(l_heel_x) if l_heel_x.size else [],
        right_x=_array_to_list(r_heel_x) if r_heel_x.size else [],
        left_strikes_t=[float(i / fps) for i in L_idx_clip.tolist()] if fps > 0 else [],
        right_strikes_t=[float(i / fps) for i in R_idx_clip.tolist()] if fps > 0 else [],
        left_strikes_x=[float(l_heel_x[i]) for i in L_idx_clip.tolist()] if l_heel_x.size else [],
        right_strikes_x=[float(r_heel_x[i]) for i in R_idx_clip.tolist()] if r_heel_x.size else [],
        left_count=int(step_data.get("left_count", len(L_idx_clip)) or 0),
        right_count=int(step_data.get("right_count", len(R_idx_clip)) or 0),
    )

    # ── Step length tab (per-leg split) ────────────────────────
    mpp = features.get("meters_per_pixel")
    left_sl, right_sl, sl_unit = _per_leg_step_lengths(ts, L_idx, R_idx, fps, mpp)
    left_mean = float(np.mean(left_sl)) if left_sl else 0.0
    right_mean = float(np.mean(right_sl)) if right_sl else 0.0
    sl_tab = StepLengthTab(
        left_values=left_sl, right_values=right_sl,
        left_mean=left_mean, right_mean=right_mean,
        unit=sl_unit,
    )

    # ── Timing tab ─────────────────────────────────────────────
    step_timing = clean.get("step_timing", {}) or {}
    L_t = _floats_only(step_timing.get("left", []))
    R_t = _floats_only(step_timing.get("right", []))
    timing_tab = TimingTab(
        left_intervals=L_t,
        right_intervals=R_t,
        left_mean=float(np.mean(L_t)) if L_t else 0.0,
        right_mean=float(np.mean(R_t)) if R_t else 0.0,
    )

    # ── Torso lean tab ─────────────────────────────────────────
    tl = clean.get("torso_lean", {}) or {}
    torso_tab = TorsoLeanTab(
        time_axis=[float(t) for t in time_axis],
        angles=_array_to_list(tl.get("angles")),
        mean=float(tl.get("mean", 0.0) or 0.0),
        std=float(tl.get("std", 0.0) or 0.0),
    )

    # ── Ankle trajectory tab ───────────────────────────────────
    at = clean.get("ankle_trajectory", {}) or {}
    ankle_tab = AnkleTrajectoryTab(
        time_axis=[float(t) for t in time_axis],
        left_x=_array_to_list(at.get("left_x")),
        right_x=_array_to_list(at.get("right_x")),
        left_y=_array_to_list(at.get("left_y")),
        right_y=_array_to_list(at.get("right_y")),
    )

    # ── Pass segments (frame indices → seconds) ────────────────
    passes_raw = features.get("pass_segments", []) or []
    passes: list[PassSegmentDTO] = []
    if fps > 0:
        for p in passes_raw:
            try:
                passes.append(
                    PassSegmentDTO(
                        start_sec=float(int(p["start"]) / fps),
                        end_sec=float(int(p["end"]) / fps),
                        core_start_sec=float(int(p["core_start"]) / fps),
                        core_end_sec=float(int(p["core_end"]) / fps),
                        direction=int(p.get("direction", 0)),
                    )
                )
            except Exception:
                continue

    return TabsData(
        heel_position=heel_tab,
        step_length=sl_tab,
        timing=timing_tab,
        torso_lean=torso_tab,
        ankle_trajectory=ankle_tab,
        pass_segments=passes,
    )


# ══════════════════════════════════════════════════════════════════════
# Top-level gait formatter
# ══════════════════════════════════════════════════════════════════════
def format_gait_response(
    features: dict,
    insights: dict,
    ts: dict,
    height_cm: float,
    patient_name: str | None,
) -> GaitData:
    """Convert engine output into the API's GaitData with full feature parity."""
    total = features.get("total_metrics", features) or {}
    clean = features.get("clean_metrics", features) or {}

    metrics_total = _build_metrics_block(total, is_clean=False, features=features)
    metrics_clean = _build_metrics_block(clean, is_clean=True, features=features)

    obs_buckets = _bucket_observations(insights.get("observations") or [])
    suggestions = [str(s) for s in (insights.get("suggestions") or [])]

    return GaitData(
        patient_info=PatientInfo(name=patient_name, height_cm=float(height_cm)),
        video_info=_build_video_info(features, height_cm),
        walking_direction=str(features.get("direction", "Unknown")),
        metrics_total=metrics_total,
        metrics_clean=metrics_clean,
        joint_angles=_build_joint_angles(features),
        gait_cycle_data=_build_gait_cycle_data(features),
        normalized_overview=_build_normalized_overview(features),
        tabs_data=_build_tabs_data(features, ts),
        observations=ObservationsBucketed(
            hip=obs_buckets["hip"],
            knee=obs_buckets["knee"],
            ankle=obs_buckets["ankle"],
            overall=obs_buckets["overall"],
            suggestions=suggestions,
        ),
    )


# ══════════════════════════════════════════════════════════════════════
# Biomech (unchanged)
# ══════════════════════════════════════════════════════════════════════
def _classify_biomech_status(measured: float, target: float) -> tuple[str, float]:
    if target <= 0:
        return "poor", 0.0
    pct = (measured / target) * 100.0
    if pct >= 90.0:
        return "good", pct
    if pct >= 75.0:
        return "fair", pct
    return "poor", pct


def format_biomech_response(
    body_part: str,
    movement: str,
    side: str | None,
    raw_result: dict,
    normal_range: tuple[float, float],
    target: float,
) -> BiomechData:
    peak_magnitude = float(raw_result.get("peak_magnitude", 0.0) or 0.0)
    peak_angle = raw_result.get("peak_angle", None)
    if peak_angle is not None:
        peak_angle = _scalar(peak_angle)

    status, pct = _classify_biomech_status(peak_magnitude, float(target))
    rng_low, rng_high = float(normal_range[0]), float(normal_range[1])
    rng_str = (
        f"{rng_low:.0f}°–{rng_high:.0f}°"
        if rng_low != rng_high
        else f"{rng_low:.0f}°"
    )
    label_part = movement.replace("_", " ").title()
    side_str = f" ({side})" if (body_part == "shoulder" and side) else ""
    interpretation = (
        f"{label_part}{side_str} measured {peak_magnitude:.1f}°, "
        f"which is {pct:.0f}% of the {rng_str} normal range — {status}."
    )

    return BiomechData(
        body_part=body_part,
        movement=movement,
        side=(side if body_part == "shoulder" else None),
        peak_angle=peak_angle,
        peak_magnitude=peak_magnitude,
        reference_range=[rng_low, rng_high],
        target=float(target),
        percentage=round(pct, 1),
        status=status,
        valid_frames=int(raw_result.get("valid_frames", 0) or 0),
        total_frames=int(raw_result.get("total_frames", 0) or 0),
        fps=float(raw_result.get("fps", 0.0) or 0.0),
        interpretation=interpretation,
    )
