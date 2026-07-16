"""squat_lateral_engine.py — backend B-series Lateral (Sagittal) Squat analyzer.

Clinical context:
  Patient stands SIDE-ON to a single camera (declared `side`) and
  performs 3-6 slow squats. This is the classic sagittal-plane
  physio squat screen: hip flexion, knee flexion, trunk lean at
  the squat bottom.

Only the near-side (camera-facing) leg is analysed — the far-side
leg is occluded and per-frame visibility is unreliable, so we NEVER
average both sides.

Scope — five metrics per rep, taken at the deepest frame:
  1. peak_knee_flexion_deg   — bend at knee (0° = straight)
  2. peak_hip_flexion_deg    — bend at hip  (magnitude)
  3. trunk_lean_deg          — trunk tilt from vertical
  4. hip_knee_ratio          — hip / knee flexion; high = hip-dominant
                               (posterior-chain), low = knee-dominant (quad)
  5. heel_rise: bool         — near-side heel Y lifts above the standing
                               baseline by more than the heel-rise threshold
                               (2 cm when calibration is available, else
                               1.5% of leg length)

Ankle dorsiflexion (previously #3) was REMOVED — the 2D shank-to-
vertical proxy was too noisy in practice. Near-side ankle keypoints
jitter under occlusion, and BlazePose foot landmarks aren't reliable
for a sagittal dorsi/plantar angle from a lateral view. The hip +
knee flexion angles are stable geometric readings from the shoulder-
hip-knee-ankle chain and are what's actually scored.

Aggregate:
  representative rep = deepest rep (max peak_knee_flexion_deg)
  classification (three-tier — matches the recommendation.js DSL's
  evalClassification at :247):
    good     : deepest peak_knee_flexion ≥ 90° AND no heel_rise on ANY rep
               AND trunk_lean ≤ 40°
    poor     : deepest peak_knee_flexion < 70° OR any rep has heel_rise
               OR trunk_lean > 55°
    moderate : everything in between
  0 reps or poor visibility → classification="insufficient_data",
  reps=[], and the guard fields explain why.

Pipeline reuses existing helpers by IMPORT/CALL only — this file
never mutates their state or overrides their outputs:
  • extract_poses / build_time_series      — gait_engine (unchanged)
  • _knee_angles_px                        — gait_engine (pure geometric)
  • _hip_angles_px (magnitude via abs)     — gait_engine (magnitude is
                                             pure; only the sign is
                                             walking-coupled and we
                                             discard it here)
  • height_calibration.*                   — shared calibration helpers
  • savgol_filter / find_peaks             — scipy.signal, same pattern
                                             used by gait_cycle.py

Local helpers written new (walking-coupled originals not reusable):
  • _trunk_to_vertical_deg  — sagittal trunk lean, no walking sign
  • _find_squat_bottoms     — hip-Y trough detection tailored for squats
  (_shank_to_vertical_deg is retained as dead code — see the note on
   the function itself — since removing it entirely would break any
   external callers that import it. It is no longer used internally.)

Valgus is honestly marked `not_assessed` — frontal plane not visible
from a lateral camera. Use overhead-squat or single-leg-squat for that.
"""
from __future__ import annotations

import base64
import logging
import math
from statistics import median as _stdlib_median
from typing import Any, Optional

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from engines.calibration.height_calibration import (
    build_height_calibration_dict,
    measure_body_pixel_height_from_time_series,
    probe_source_frame_dimensions,
)
from engines.gait_engine import (
    LM,  # noqa: F401 — parity with sibling orthopedic engines
    _hip_angles_px,
    _knee_angles_px,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/squatLateral.ts) ───────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 30.0
_STANDING_HOLD_SEC = 0.3
_STANDING_HOLD_SAMPLES = int(round(_STANDING_HOLD_SEC * _SAMPLE_HZ))
_MAX_SESSION_DURATION_SEC = 30.0

_STANDING_TOLERANCE_FRAC_OF_LEG = 0.04
_STANDING_TOLERANCE_FALLBACK_MULT = 3.0

# Squat-rep detection — hip Y trough on the smoothed signal.
# Prominence = trough must be at least this frac-of-leg-length deep
# below its shoulders (measured against the smoothed hip signal).
_TROUGH_PROMINENCE_FRAC_OF_LEG = 0.06
_MIN_REP_GAP_SEC = 0.6

# Heel-rise thresholds.
#   Calibrated  → any lift ≥ _HEEL_RISE_CM above standing baseline.
#   Uncalibrated → fallback: 1.5% of leg length (pixel).
# Both are conservative — real heel rise in a lateral view is
# typically 3-6 cm; 2 cm ensures we don't flag noise as heel rise.
_HEEL_RISE_CM = 2.0
_HEEL_RISE_FALLBACK_FRAC_OF_LEG = 0.015

# Classification thresholds. See the module docstring.
_KNEE_GOOD_MIN_DEG = 90.0
_KNEE_POOR_MAX_DEG = 70.0
_TRUNK_GOOD_MAX_DEG = 40.0
_TRUNK_POOR_MIN_DEG = 55.0


# ─── Landmark lookup per declared side ─────────────────────────────
_SIDES_MAP: dict[str, dict[str, str]] = {
    "left":  {
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
        "heel": "left_heel",
        "shoulder": "left_shoulder",
        "foot_index": "left_foot_index",
    },
    "right": {
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
        "heel": "right_heel",
        "shoulder": "right_shoulder",
        "foot_index": "right_foot_index",
    },
}


# ─── Local helpers (walking-coupled originals not reusable) ────────
def _shank_to_vertical_deg(
    knee_x: float, knee_y: float,
    ankle_x: float, ankle_y: float,
) -> float:
    """Ankle-dorsiflexion proxy: angle between the shank (knee→ankle)
    and screen vertical. Positive = tibia leaning FORWARD over the
    foot (dorsiflexion); negative = leaning back (plantarflexion).

    Uses image y-down convention: vertical vector = (0, +1) points
    downward on screen. A shank tilted forward (ankle x > knee x for
    a person facing right) rotates the shank vector clockwise → we
    signed-project onto the forward horizontal so positive = forward.
    Since we take absolute-value tilt magnitude for the reported
    metric, the sign here is informational only.
    """
    sx = ankle_x - knee_x
    sy = ankle_y - knee_y
    mag = math.hypot(sx, sy)
    if mag < 1e-6:
        return 0.0
    # angle from vertical-down = arctan(|horizontal|, vertical)
    tilt = math.degrees(math.atan2(abs(sx), abs(sy)))
    return round(tilt, 1)


def _trunk_to_vertical_deg(
    shoulder_x: float, shoulder_y: float,
    hip_x: float, hip_y: float,
) -> float:
    """Trunk lean magnitude from vertical. Trunk vector = hip→shoulder
    (points UP the body in image y-down → dy is negative when upright).
    Same magnitude formula as shank — sign discarded, we report
    absolute tilt from vertical.
    """
    tx = shoulder_x - hip_x
    ty = shoulder_y - hip_y  # negative when trunk points up
    mag = math.hypot(tx, ty)
    if mag < 1e-6:
        return 0.0
    tilt = math.degrees(math.atan2(abs(tx), abs(ty)))
    return round(tilt, 1)


def _find_squat_bottoms(
    hip_y: np.ndarray,
    fps: float,
    leg_length_px: float,
) -> np.ndarray:
    """Return integer frame indices of squat bottoms (deepest hip Y
    per rep). Smooths hip_y with savgol_filter then finds peaks on
    the NEGATED signal (troughs in Y = high in −Y). Prominence gate
    keeps micro-oscillations from counting as reps.
    """
    n = int(len(hip_y))
    if n < 5:
        return np.array([], dtype=int)
    # savgol window — odd, ≥5, ≤ len(signal). Use ~0.4 s or 5, whichever bigger.
    win = int(round(max(5, min(n if n % 2 == 1 else n - 1, fps * 0.4))))
    if win % 2 == 0:
        win -= 1
    win = max(5, win)
    if win >= n:
        return np.array([], dtype=int)
    try:
        smoothed = savgol_filter(hip_y, win, polyorder=3)
    except Exception:
        smoothed = hip_y
    prom = max(1.0, _TROUGH_PROMINENCE_FRAC_OF_LEG * leg_length_px)
    min_dist = max(3, int(round(_MIN_REP_GAP_SEC * fps)))
    peaks, _ = find_peaks(-smoothed, prominence=prom, distance=min_dist)
    return peaks.astype(int)


def _capture_frame(video_path: str, frame_index: int) -> Optional[str]:
    """Grab a single frame and return as base64 PNG data URL, or None."""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        log.warning("squat_lateral: frame capture failed: %s", e)
        return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


# ─── Main analyzer ─────────────────────────────────────────────────
def analyze_squat_lateral(
    video_path: str,
    pose_options: Any,
    side: str,
    calibration: Optional[dict] = None,
    patient_height_cm: Optional[float] = None,
) -> dict:
    """Analyze a lateral-view squat video. Returns a dict matching
    the frontend SquatLateralResult TypeScript interface.

    Args:
      side: "left" or "right" — the leg facing the camera.

    Never raises. When the recording is too short / too occluded /
    contains no detectable reps, returns a valid dict with
    reps=[] + classification="insufficient_data" + guard notes.
    """
    if side not in ("left", "right"):
        # Fail-safe — the endpoint should have rejected this already,
        # but if a caller passes garbage we still return a valid dict.
        return _empty_result(
            side=side,
            calibration=calibration,
            patient_height_cm=patient_height_cm,
            reason=f"invalid_side_{side!r}",
        )
    side_lms = _SIDES_MAP[side]

    # 1) Calibration pass-through
    final_calibration = calibration
    ppc: Optional[float] = None
    if final_calibration is not None:
        raw_ppc = final_calibration.get("pixels_per_cm")
        if isinstance(raw_ppc, (int, float)) and raw_ppc > 0:
            ppc = float(raw_ppc)
        else:
            final_calibration = None

    # 2) Pose extraction + time-series
    raw, fps, _ = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)
    required = [side_lms["hip"], side_lms["knee"], side_lms["ankle"],
                side_lms["heel"], side_lms["shoulder"]]
    lengths = [len(ts[k]["y"]) for k in required if k in ts]
    if not lengths or fps <= 0:
        return _empty_result(
            side=side, calibration=final_calibration,
            patient_height_cm=patient_height_cm,
            reason="poor_visibility",
        )
    n = int(min(lengths))

    # 2b) Height-calibration server-side fallback (same pattern as
    #     overhead_squat_engine + tuck_jump_engine).
    if ppc is None and patient_height_cm is not None and patient_height_cm > 0:
        body_px = measure_body_pixel_height_from_time_series(ts, fps, n)
        if body_px is not None:
            source_frame = probe_source_frame_dimensions(video_path)
            derived = build_height_calibration_dict(
                body_px, float(patient_height_cm), source_frame,
            )
            if derived is not None:
                final_calibration = derived
                ppc = float(derived["pixels_per_cm"])
                log.info(
                    "squat_lateral: height calibration: body_px=%.0f "
                    "height_cm=%.1f → %.2f px/cm",
                    body_px, patient_height_cm, ppc,
                )

    # 3) Reused gait-engine joint helpers.
    #    _knee_angles_px is pure geometric (safe as-is).
    #    _hip_angles_px magnitude is pure; sign is walking-coupled so
    #    we discard it via abs() below.
    knee_angles = _knee_angles_px(ts)[side]
    hip_angles_signed = _hip_angles_px(ts, pass_segments=None)[side]

    # 4) Side-specific px arrays
    hip_y = np.asarray(ts[side_lms["hip"]]["y_px"], dtype=float)[:n]
    knee_x = np.asarray(ts[side_lms["knee"]]["x_px"], dtype=float)[:n]
    knee_y = np.asarray(ts[side_lms["knee"]]["y_px"], dtype=float)[:n]
    ankle_x = np.asarray(ts[side_lms["ankle"]]["x_px"], dtype=float)[:n]
    ankle_y = np.asarray(ts[side_lms["ankle"]]["y_px"], dtype=float)[:n]
    heel_y = np.asarray(ts[side_lms["heel"]]["y_px"], dtype=float)[:n]
    shoulder_x = np.asarray(ts[side_lms["shoulder"]]["x_px"], dtype=float)[:n]
    shoulder_y = np.asarray(ts[side_lms["shoulder"]]["y_px"], dtype=float)[:n]
    hip_x = np.asarray(ts[side_lms["hip"]]["x_px"], dtype=float)[:n]

    # Visibility (near-side landmarks only).
    hip_vis   = np.asarray(ts[side_lms["hip"]]["vis"], dtype=float)[:n]
    knee_vis  = np.asarray(ts[side_lms["knee"]]["vis"], dtype=float)[:n]
    ankle_vis = np.asarray(ts[side_lms["ankle"]]["vis"], dtype=float)[:n]
    heel_vis  = np.asarray(ts[side_lms["heel"]]["vis"], dtype=float)[:n]
    valid = (
        (hip_vis   >= _VIS_THRESHOLD)
        & (knee_vis  >= _VIS_THRESHOLD)
        & (ankle_vis >= _VIS_THRESHOLD)
        & (heel_vis  >= _VIS_THRESHOLD)
    )
    n_valid = int(valid.sum())
    duration_seconds = float(n / fps) if fps > 0 else 0.0

    if n_valid < max(_STANDING_HOLD_SAMPLES + 5, int(n * 0.30)):
        return _empty_result(
            side=side, calibration=final_calibration,
            patient_height_cm=patient_height_cm,
            reason="poor_visibility",
            fps=float(fps), total_frames=n, valid_frames=n_valid,
            duration_seconds=duration_seconds,
        )

    # 5) Standing baseline lock — median hip Y (+ heel Y) over the
    #    first stable window. Same pattern as overhead-squat.
    provisional_leg = float(max(1.0, np.percentile(ankle_y - hip_y, 90)))
    tol_px = _STANDING_TOLERANCE_FRAC_OF_LEG * provisional_leg
    fallback_ceiling = tol_px * _STANDING_TOLERANCE_FALLBACK_MULT

    baseline_hip_y: Optional[float] = None
    baseline_heel_y: Optional[float] = None
    baseline_lock_idx: Optional[int] = None
    best_fallback: Optional[tuple[float, float, float, int]] = None

    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sample_hop = 1  # per-frame — cheap
    for end_i in range(_STANDING_HOLD_SAMPLES - 1, n, sample_hop):
        s0 = end_i - _STANDING_HOLD_SAMPLES + 1
        s1 = end_i + 1
        if not valid[s0:s1].all():
            continue
        window_hy = hip_y[s0:s1]
        med_hy = float(_stdlib_median(window_hy.tolist()))
        max_dev = float(np.max(np.abs(window_hy - med_hy)))
        med_heel_y = float(_stdlib_median(heel_y[s0:s1].tolist()))
        if max_dev <= tol_px:
            baseline_hip_y = med_hy
            baseline_heel_y = med_heel_y
            baseline_lock_idx = end_i
            break
        if best_fallback is None or max_dev < best_fallback[0]:
            best_fallback = (max_dev, med_hy, med_heel_y, end_i)

    if baseline_hip_y is None and best_fallback is not None:
        if best_fallback[0] <= fallback_ceiling:
            _, baseline_hip_y, baseline_heel_y, baseline_lock_idx = best_fallback

    if baseline_hip_y is None or baseline_heel_y is None or baseline_lock_idx is None:
        return _empty_result(
            side=side, calibration=final_calibration,
            patient_height_cm=patient_height_cm,
            reason="no_baseline",
            fps=float(fps), total_frames=n, valid_frames=n_valid,
            duration_seconds=duration_seconds,
        )

    # Refine leg length from the locked baseline.
    lock_i = int(baseline_lock_idx)
    leg_length_px = max(1.0, float(ankle_y[lock_i] - baseline_hip_y))

    # Heel-rise threshold (see module docstring).
    if ppc and ppc > 0:
        heel_rise_thresh_px = float(_HEEL_RISE_CM * ppc)
    else:
        heel_rise_thresh_px = float(_HEEL_RISE_FALLBACK_FRAC_OF_LEG * leg_length_px)

    # 6) Rep segmentation — find hip-Y troughs within [baseline_lock_idx, session_end].
    max_frames = int(round(_MAX_SESSION_DURATION_SEC * fps))
    session_start_idx = baseline_lock_idx + 1
    session_end_idx = min(n, session_start_idx + max_frames)
    session_hip = hip_y[session_start_idx:session_end_idx].copy()
    session_valid = valid[session_start_idx:session_end_idx]
    # Interpolate through invalid frames so savgol / find_peaks don't
    # chase visibility noise.
    if not session_valid.all() and session_valid.any():
        idx_arr = np.arange(len(session_hip), dtype=float)
        session_hip = np.interp(idx_arr, idx_arr[session_valid], session_hip[session_valid])
    local_troughs = _find_squat_bottoms(session_hip, fps, leg_length_px)
    bottom_frames = (local_troughs + session_start_idx).astype(int)

    if len(bottom_frames) == 0:
        return _empty_result(
            side=side, calibration=final_calibration,
            patient_height_cm=patient_height_cm,
            reason="no_reps_detected",
            fps=float(fps), total_frames=n, valid_frames=n_valid,
            duration_seconds=duration_seconds,
            leg_length_px=leg_length_px, step_hz=step,
        )

    # 7) Per-rep metrics at each bottom frame.
    reps: list[dict] = []
    for i, b in enumerate(bottom_frames):
        b = int(b)
        if b < 0 or b >= n or not bool(valid[b]):
            continue
        peak_knee = float(knee_angles[b]) if not np.isnan(knee_angles[b]) else None
        raw_hip = hip_angles_signed[b]
        peak_hip = float(abs(raw_hip)) if not np.isnan(raw_hip) else None
        # ── Ankle dorsiflexion intentionally removed ────────────
        # A lateral 2D shank-to-vertical proxy was too noisy in
        # practice — near-side ankle keypoint jitters under
        # occlusion and BlazePose foot landmarks are unreliable
        # for the dorsi/plantar angle. Only hip + knee flexion are
        # scored now, which are stable geometric angles from the
        # shoulder-hip-knee-ankle chain.
        trunk_lean = _trunk_to_vertical_deg(
            float(shoulder_x[b]), float(shoulder_y[b]),
            float(hip_x[b]),      float(hip_y[b]),
        )
        if peak_hip is not None and peak_knee is not None and peak_knee > 1e-6:
            hip_knee_ratio: Optional[float] = round(peak_hip / peak_knee, 3)
        else:
            hip_knee_ratio = None
        heel_dy = float(baseline_heel_y - heel_y[b])
        # In image y-down coords, heel LIFTING = y decreases → baseline − y_now > 0
        heel_rise = bool(heel_dy >= heel_rise_thresh_px)

        reps.append({
            "rep_index": len(reps),
            "bottom_frame_index": b,
            "bottom_t_ms": float(b / fps * 1000.0),
            "peak_knee_flexion_deg":  round(peak_knee, 1) if peak_knee is not None else None,
            "peak_hip_flexion_deg":   round(peak_hip, 1)  if peak_hip  is not None else None,
            "trunk_lean_deg":         trunk_lean,
            "hip_knee_ratio":         hip_knee_ratio,
            "heel_rise":              heel_rise,
            "heel_rise_px":           round(heel_dy, 1),
        })

    if len(reps) == 0:
        return _empty_result(
            side=side, calibration=final_calibration,
            patient_height_cm=patient_height_cm,
            reason="no_valid_reps",
            fps=float(fps), total_frames=n, valid_frames=n_valid,
            duration_seconds=duration_seconds,
            leg_length_px=leg_length_px, step_hz=step,
        )

    # 8) Representative rep = deepest peak knee flexion.
    rep_with_knee = [r for r in reps if r["peak_knee_flexion_deg"] is not None]
    if rep_with_knee:
        deepest = max(rep_with_knee, key=lambda r: r["peak_knee_flexion_deg"])
    else:
        deepest = reps[0]
    peak_knee_deepest = deepest["peak_knee_flexion_deg"] or 0.0
    trunk_lean_deepest = deepest["trunk_lean_deg"] or 0.0
    any_heel_rise = any(r["heel_rise"] for r in reps)

    # 9) Classification (three-tier). See module docstring — this
    #    matches recommendation.js evalClassification (:247).
    if peak_knee_deepest >= _KNEE_GOOD_MIN_DEG and not any_heel_rise and trunk_lean_deepest <= _TRUNK_GOOD_MAX_DEG:
        classification = "good"
    elif peak_knee_deepest < _KNEE_POOR_MAX_DEG or any_heel_rise or trunk_lean_deepest > _TRUNK_POOR_MIN_DEG:
        classification = "poor"
    else:
        classification = "moderate"

    # 10) Per-rep angle traces for the SVG chart (downsampled at ~30 Hz).
    trace_idx = np.arange(0, n, max(1, int(round(fps / _SAMPLE_HZ))), dtype=int)
    trace_t_ms = (trace_idx / fps * 1000.0).tolist()
    trace_knee = [
        None if np.isnan(knee_angles[i]) or not bool(valid[i])
        else round(float(knee_angles[i]), 1)
        for i in trace_idx
    ]
    trace_hip = [
        None if np.isnan(hip_angles_signed[i]) or not bool(valid[i])
        else round(float(abs(hip_angles_signed[i])), 1)
        for i in trace_idx
    ]

    # 11) Bottom-frame screenshot from the deepest rep.
    peak_screenshot = _capture_frame(video_path, int(deepest["bottom_frame_index"]))

    # 12) Caveats — always emitted (frontal-plane, in-plane, occlusion).
    caveats = [
        {
            "code": "frontal_not_assessed",
            "label": "Frontal-plane items (knee valgus) not assessed from lateral view",
            "detail": (
                "Add an Overhead Squat or Single-Leg Squat run for valgus / "
                "pelvic-drop scoring."
            ),
        },
        {
            "code": "in_plane_only",
            "label": "Angles trustworthy only if squat motion is in-plane",
            "detail": (
                "Roughly 15° of body twist toward or away from the camera "
                "introduces 5-10° of angle error. Position the camera "
                "perpendicular to the plane of motion."
            ),
        },
        {
            "code": "far_side_not_analysed",
            "label": "Far-side leg not analysed",
            "detail": (
                "Only the declared near-side leg is measured. Turn the patient "
                "around and re-run to assess the other side."
            ),
        },
    ]

    valgus_note = {
        "status": "not_assessed",
        "reason": "frontal plane not visible from lateral view",
    }

    return {
        "side":                     side,
        "patient_height_cm":        patient_height_cm,
        "calibration":              final_calibration,
        "baseline_hip_y_px":        float(baseline_hip_y),
        "baseline_heel_y_px":       float(baseline_heel_y),
        "leg_length_px":            float(leg_length_px),
        "heel_rise_threshold_px":   float(heel_rise_thresh_px),
        "reps":                     reps,
        "rep_count":                len(reps),
        # Summary = deepest-rep metrics (hip + knee + trunk + ratio +
        # heel-rise). Ankle dorsi removed — see note in the per-rep
        # append above.
        "peak_knee_flexion_deg":    deepest["peak_knee_flexion_deg"],
        "peak_hip_flexion_deg":     deepest["peak_hip_flexion_deg"],
        "trunk_lean_deg":           deepest["trunk_lean_deg"],
        "hip_knee_ratio":           deepest["hip_knee_ratio"],
        "heel_rise":                bool(deepest["heel_rise"]),
        "any_heel_rise":            bool(any_heel_rise),
        "deepest_rep_index":        int(deepest["rep_index"]),
        "mean_peak_knee_flexion_deg": round(_mean([r["peak_knee_flexion_deg"] for r in rep_with_knee]), 1)
            if rep_with_knee else None,
        "mean_trunk_lean_deg":      round(_mean([r["trunk_lean_deg"] for r in reps if r["trunk_lean_deg"] is not None]), 1)
            if reps else None,
        "classification":           classification,
        "valgus":                   valgus_note,
        "caveats":                  caveats,
        "peak_screenshot_data_url": peak_screenshot,
        "angle_trace": {
            "t_ms":  trace_t_ms,
            "knee":  trace_knee,
            "hip":   trace_hip,
            "bottom_frame_indices": [int(b) for b in bottom_frames.tolist()],
            "bottom_t_ms": [float(b / fps * 1000.0) for b in bottom_frames.tolist()],
        },
        "duration_seconds":         duration_seconds,
        "fps":                      float(fps),
        "total_frames":             int(n),
        "valid_frames":             int(n_valid),
        "interpretation":           None,
    }


# ─── Empty-result builder (guard path) ────────────────────────────
def _empty_result(
    *,
    side: str,
    calibration: Optional[dict],
    patient_height_cm: Optional[float],
    reason: str,
    fps: Optional[float] = None,
    total_frames: Optional[int] = None,
    valid_frames: Optional[int] = None,
    duration_seconds: float = 0.0,
    leg_length_px: float = 0.0,
    step_hz: Optional[int] = None,  # noqa: ARG001 — signature parity
) -> dict:
    """Valid-shape result when there is nothing to score. Never
    raises. classification="insufficient_data" carries a machine-
    readable `guard_reason`; downstream recommendation.js will skip
    this report cleanly.
    """
    caveats = [
        {
            "code": "insufficient_data",
            "label": "Not enough usable video to score",
            "detail": (
                "Re-record with the patient side-on, full body in frame, "
                "starting from a still upright stance and performing 3-6 "
                "slow squats."
            ),
        },
    ]
    return {
        "side":                     side,
        "patient_height_cm":        patient_height_cm,
        "calibration":              calibration,
        "baseline_hip_y_px":        0.0,
        "baseline_heel_y_px":       0.0,
        "leg_length_px":            float(leg_length_px),
        "heel_rise_threshold_px":   0.0,
        "reps":                     [],
        "rep_count":                0,
        "peak_knee_flexion_deg":    None,
        "peak_hip_flexion_deg":     None,
        "trunk_lean_deg":           None,
        "hip_knee_ratio":           None,
        "heel_rise":                False,
        "any_heel_rise":            False,
        "deepest_rep_index":        -1,
        "mean_peak_knee_flexion_deg": None,
        "mean_trunk_lean_deg":      None,
        "classification":           "insufficient_data",
        "guard_reason":             reason,
        "valgus": {
            "status": "not_assessed",
            "reason": "frontal plane not visible from lateral view",
        },
        "caveats":                  caveats,
        "peak_screenshot_data_url": None,
        "angle_trace": {
            "t_ms":  [],
            "knee":  [],
            "hip":   [],
            "bottom_frame_indices": [],
            "bottom_t_ms": [],
        },
        "duration_seconds":         float(duration_seconds),
        "fps":                      float(fps) if fps is not None else None,
        "total_frames":             int(total_frames) if total_frames is not None else None,
        "valid_frames":             int(valid_frames) if valid_frames is not None else None,
        "interpretation":           None,
    }
