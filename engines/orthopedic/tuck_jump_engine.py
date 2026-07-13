"""tuck_jump_engine.py — backend D2 Tuck Jump analyzer.

Clinical context — Myer's Tuck Jump Assessment (TJA):
  A single-camera FRONTAL screen for lower-extremity injury-risk.
  Patient performs ~10 seconds of continuous tuck jumps (knees pulled
  to chest at apex, land on both feet, immediately re-jump). Rated
  against a 10-item checklist and classified good / moderate / poor.

Myer's 10 items (scored per session):
  1. Lower-extremity valgus at landing       (measurable — KFPPA)
  2. Thighs not reach parallel (low tuck)    (measurable)
  3. Thighs not equal side-to-side           (measurable — L/R symmetry)
  4. Foot placement not shoulder-width       (measurable — ankle spread)
  5. Foot placement not parallel             (not_assessed — foot-yaw
                                              landmark unreliable frontal)
  6. Foot contact timing not equal           (measurable — takeoff /
                                              landing timing symmetry)
  7. Excessive landing contact noise         (not_assessed — needs
                                              force/audio)
  8. Pause between jumps (not continuous)    (measurable — grounded gap)
  9. Technique declines before 10 s (fatigue)(measurable — valgus trend
                                              + height fade)
  10. Does not land in same footprint (drift)(measurable — ankle-x drift)

Items 5 and 7 honestly return status="not_assessed". The remaining 8
get pass/fail flags + a per-item metric.

Pipeline mirrors counter_movement_jump_engine.py architecturally
(same extract_poses / build_time_series stack, same standing-baseline
lock-in, same takeoff/landing state machine). Deltas:
  • Frontal view (needed for KFPPA + L/R symmetry) — CMJ is lateral
  • Continuous multi-jump session (~10 s) instead of ≤3 discrete trials
  • Per-landing KFPPA + ankle spread; session-level checklist scoring
"""
from __future__ import annotations

import base64
import logging
import math
from statistics import median as _stdlib_median, stdev as _stdlib_stdev
from typing import Any, Optional

import cv2

from engines.calibration.height_calibration import (
    build_height_calibration_dict,
    measure_body_pixel_height_from_time_series,
    probe_source_frame_dimensions,
)
from engines.gait_engine import (
    LM,  # noqa: F401  — imported for parity with sibling engines
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/tuckJump.ts) ────────────
_VIS_THRESHOLD = 0.15
_ANKLE_JUMP_VIS_FLOOR = 0.05  # loosened during airborne (see CMJ engine)
_SAMPLE_HZ = 30.0
_STANDING_HOLD_SEC = 0.3
_STANDING_HOLD_SAMPLES = int(round(_STANDING_HOLD_SEC * _SAMPLE_HZ))
_MAX_TRIAL_DURATION_SEC = 15.0

_AIRBORNE_LIFT_FRAC_OF_LEG = 0.06
_LANDED_BAND_FRAC_OF_LEG = 0.03
_MIN_AIRBORNE_FRAMES_SEC = 0.08
_MIN_TRIAL_GAP_SEC = 0.15
_PAUSE_GAP_SEC = 0.6  # any grounded gap > this fails Myer item 8

_STANDING_TOLERANCE_FRAC_OF_LEG = 0.04
_STANDING_TOLERANCE_FALLBACK_MULT = 3.0

# Myer's per-item thresholds
_VALGUS_KFPPA_FAIL_DEG = 12.0
_THIGH_PARALLEL_MIN_HIP_LIFT_FRAC_OF_LEG = 0.20
_THIGH_SYMMETRY_MAX_DELTA_FRAC = 0.08
_FOOT_SHOULDERWIDTH_MIN_RATIO = 0.7
_FOOT_SHOULDERWIDTH_MAX_RATIO = 1.6
_CONTACT_TIMING_MAX_DELTA_MS = 80.0
_FATIGUE_HEIGHT_FADE_FRAC = 0.20
_FATIGUE_VALGUS_GROWTH_DEG = 5.0
_FOOTPRINT_DRIFT_MAX_FRAC_OF_LEG = 0.10

_GRAVITY_M_PER_S2 = 9.81


# ─── Small helpers ─────────────────────────────────────────────────
def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return float(_stdlib_stdev(values))
    except Exception:
        return 0.0


def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _kfppa_deg(
    hip_x: float, hip_y: float,
    knee_x: float, knee_y: float,
    ankle_x: float, ankle_y: float,
) -> float:
    """Knee frontal-plane projection angle magnitude (deg). Higher =
    more valgus deviation from a straight thigh→shank line."""
    thigh_x = knee_x - hip_x
    thigh_y = knee_y - hip_y
    shank_x = ankle_x - knee_x
    shank_y = ankle_y - knee_y
    dot = thigh_x * shank_x + thigh_y * shank_y
    mag = math.hypot(thigh_x, thigh_y) * math.hypot(shank_x, shank_y)
    if mag < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / mag))
    # _angle3 interior angle in [0, 180]; a perfectly straight leg
    # gives 180° (thigh and shank collinear). "Valgus" is the deviation
    # from straight, i.e. 180 − interior. We clamp to [0, 90] since
    # anything above 90° magnitude is not physiologically meaningful.
    interior = math.degrees(math.acos(cos_a))
    return max(0.0, min(90.0, 180.0 - interior))


def _capture_apex_frame(video_path: str, frame_index: int) -> Optional[str]:
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
        log.warning("tuck_jump: apex screenshot failed: %s", e)
        return None


def _sample_row(ts: dict, i: int, fps: float) -> Optional[dict]:
    """Read one frame from the ts dict, gated by visibility. Returns
    None when hips or knees aren't visible enough (ankle visibility is
    only floored during airborne — see _ANKLE_JUMP_VIS_FLOOR)."""
    required = ("left_hip", "right_hip", "left_knee", "right_knee")
    if not all(_visible(ts, k, i) for k in required):
        return None
    l_ank_vis = float(ts["left_ankle"]["vis"][i])
    r_ank_vis = float(ts["right_ankle"]["vis"][i])
    if max(l_ank_vis, r_ank_vis) < _ANKLE_JUMP_VIS_FLOOR:
        return None

    def _v(key: str, comp: str) -> float:
        return float(ts[key][comp][i])

    return {
        "frame_index": int(i),
        "t_ms": float(i / fps * 1000.0) if fps > 0 else 0.0,
        "hip_x_px": (_v("left_hip", "x_px") + _v("right_hip", "x_px")) / 2,
        "hip_y_px": (_v("left_hip", "y_px") + _v("right_hip", "y_px")) / 2,
        "lhip_x_px": _v("left_hip", "x_px"),
        "lhip_y_px": _v("left_hip", "y_px"),
        "rhip_x_px": _v("right_hip", "x_px"),
        "rhip_y_px": _v("right_hip", "y_px"),
        "lkn_x_px": _v("left_knee", "x_px"),
        "lkn_y_px": _v("left_knee", "y_px"),
        "rkn_x_px": _v("right_knee", "x_px"),
        "rkn_y_px": _v("right_knee", "y_px"),
        "lank_x_px": _v("left_ankle", "x_px"),
        "lank_y_px": _v("left_ankle", "y_px"),
        "rank_x_px": _v("right_ankle", "x_px"),
        "rank_y_px": _v("right_ankle", "y_px"),
        "ankle_vis": max(l_ank_vis, r_ank_vis),
    }


# ─── Main analyzer ─────────────────────────────────────────────────
def analyze_tuck_jump(
    video_path: str,
    pose_options: Any,
    calibration: Optional[dict] = None,
    patient_height_cm: Optional[float] = None,
) -> dict:
    """Analyze a tuck-jump video. Returns a dict matching the frontend
    TuckJumpResult TypeScript interface.

    Raises ValueError with a user-facing code ('poor_visibility',
    'no_baseline', 'no_jumps_detected') on hard failures.
    """
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
    n = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_knee"]["y"]),
        len(ts["right_knee"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    # 2b) Height-calibration server-side fallback
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
                    "tuck_jump: height calibration: body_px=%.0f "
                    "height_cm=%.1f → %.2f px/cm",
                    body_px, patient_height_cm, ppc,
                )

    def _px_to_cm(px: float) -> Optional[float]:
        if ppc and ppc > 0:
            return float(px / ppc)
        return None

    # 3) Sample at _SAMPLE_HZ
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_indices = list(range(0, n, step))
    samples: list[Optional[dict]] = []
    for i in sampled_indices:
        samples.append(_sample_row(ts, i, fps))

    valid_samples = [s for s in samples if s is not None]
    if len(valid_samples) < max(_STANDING_HOLD_SAMPLES + 5, int(len(samples) * 0.30)):
        raise ValueError("poor_visibility")

    duration_seconds = float(n / fps) if fps > 0 else 0.0

    # 4) Standing-baseline lock-in
    # Provisional leg length — use the maximum ankle-vs-hip gap in the
    # early samples as a proportional threshold anchor.
    leg_length_px = 0.0
    for s in valid_samples[:min(60, len(valid_samples))]:
        gap = max(s["lank_y_px"], s["rank_y_px"]) - s["hip_y_px"]
        if gap > leg_length_px:
            leg_length_px = gap
    if leg_length_px <= 0:
        leg_length_px = 300.0
    standing_tol_px = _STANDING_TOLERANCE_FRAC_OF_LEG * leg_length_px
    fallback_ceiling_px = standing_tol_px * _STANDING_TOLERANCE_FALLBACK_MULT

    baseline_hip_y: Optional[float] = None
    baseline_l_ank_y: Optional[float] = None
    baseline_r_ank_y: Optional[float] = None
    baseline_l_ank_x: Optional[float] = None
    baseline_r_ank_x: Optional[float] = None
    baseline_hip_span: Optional[float] = None
    baseline_lock_idx: Optional[int] = None
    best_fallback = None

    for end_i in range(_STANDING_HOLD_SAMPLES - 1, len(samples)):
        window = samples[end_i - _STANDING_HOLD_SAMPLES + 1: end_i + 1]
        rows = [s for s in window if s is not None]
        if len(rows) < _STANDING_HOLD_SAMPLES // 2:
            continue
        hys = [s["hip_y_px"] for s in rows]
        lays = [s["lank_y_px"] for s in rows]
        rays = [s["rank_y_px"] for s in rows]
        med_hy = _stdlib_median(hys)
        med_lay = _stdlib_median(lays)
        med_ray = _stdlib_median(rays)
        max_dev = max(
            max(abs(y - med_hy) for y in hys),
            max(abs(y - med_lay) for y in lays),
            max(abs(y - med_ray) for y in rays),
        )
        med_lax = _stdlib_median([s["lank_x_px"] for s in rows])
        med_rax = _stdlib_median([s["rank_x_px"] for s in rows])
        med_lhip_x = _stdlib_median([s["lhip_x_px"] for s in rows])
        med_rhip_x = _stdlib_median([s["rhip_x_px"] for s in rows])
        entry = (
            max_dev,
            float(med_hy), float(med_lay), float(med_ray),
            float(med_lax), float(med_rax),
            abs(float(med_rhip_x) - float(med_lhip_x)),
            end_i,
        )
        if max_dev <= standing_tol_px:
            (
                _, baseline_hip_y, baseline_l_ank_y, baseline_r_ank_y,
                baseline_l_ank_x, baseline_r_ank_x, baseline_hip_span,
                baseline_lock_idx,
            ) = entry
            break
        if best_fallback is None or entry[0] < best_fallback[0]:
            best_fallback = entry

    if baseline_hip_y is None and best_fallback is not None:
        if best_fallback[0] <= fallback_ceiling_px:
            (
                _, baseline_hip_y, baseline_l_ank_y, baseline_r_ank_y,
                baseline_l_ank_x, baseline_r_ank_x, baseline_hip_span,
                baseline_lock_idx,
            ) = best_fallback

    if (
        baseline_hip_y is None
        or baseline_l_ank_y is None
        or baseline_r_ank_y is None
        or baseline_l_ank_x is None
        or baseline_r_ank_x is None
        or baseline_hip_span is None
        or baseline_lock_idx is None
    ):
        raise ValueError(
            "no_baseline: patient did not stand still for ~0.3 s at the "
            "start of the clip. Re-record with a brief static stance."
        )

    # Recompute leg length from the locked baseline for stable thresholds.
    leg_length_px = max(
        1.0, (baseline_l_ank_y + baseline_r_ank_y) / 2 - baseline_hip_y,
    )
    if baseline_hip_span < 1.0:
        baseline_hip_span = leg_length_px * 0.3  # sane default

    airborne_lift = leg_length_px * _AIRBORNE_LIFT_FRAC_OF_LEG
    landed_band = leg_length_px * _LANDED_BAND_FRAC_OF_LEG

    # 5) Takeoff / landing state machine
    max_frames = int(round(_MAX_TRIAL_DURATION_SEC * _SAMPLE_HZ))
    session_start_idx = baseline_lock_idx + 1
    session_end_idx = min(len(samples), session_start_idx + max_frames)
    min_airborne_frames = max(1, int(round(_MIN_AIRBORNE_FRAMES_SEC * _SAMPLE_HZ)))

    state = "grounded"
    airborne_start_idx: Optional[int] = None
    apex_frame_idx: Optional[int] = None
    apex_hip_y: Optional[float] = None
    apex_row: Optional[dict] = None
    l_takeoff_frame: Optional[int] = None
    r_takeoff_frame: Optional[int] = None
    l_landing_frame: Optional[int] = None
    r_landing_frame: Optional[int] = None
    prev_landing_row: Optional[dict] = None
    jumps: list[dict] = []

    def _side_transition_frame(side_key: str, from_i: int, direction: str) -> Optional[int]:
        """Walk from from_i toward the transition frame for a single
        side. `direction` = 'takeoff' → find frame ankle rose above
        baseline − airborne_lift going forward; 'landing' → find frame
        it returned inside landed_band going forward."""
        baseline_ank_y = (
            baseline_l_ank_y if side_key == "left_ankle" else baseline_r_ank_y
        )
        for i in range(from_i, min(from_i + 8, len(samples))):
            s = samples[i]
            if s is None:
                continue
            y = s[f"{'l' if side_key == 'left_ankle' else 'r'}ank_y_px"]
            if direction == "takeoff":
                if y < baseline_ank_y - airborne_lift:
                    return i
            else:
                if abs(y - baseline_ank_y) <= landed_band:
                    return i
        return None

    for i in range(session_start_idx, session_end_idx):
        s = samples[i]
        if s is None:
            continue
        l_air = s["lank_y_px"] < baseline_l_ank_y - airborne_lift
        r_air = s["rank_y_px"] < baseline_r_ank_y - airborne_lift
        l_grounded = abs(s["lank_y_px"] - baseline_l_ank_y) <= landed_band
        r_grounded = abs(s["rank_y_px"] - baseline_r_ank_y) <= landed_band

        if state == "grounded":
            if l_air and r_air:
                airborne_start_idx = i
                apex_frame_idx = i
                apex_hip_y = s["hip_y_px"]
                apex_row = s
                # Per-side takeoff frames — walk backwards to find first
                # frame that side was airborne.
                l_takeoff_frame = i
                r_takeoff_frame = i
                for back in range(i - 1, max(session_start_idx - 1, i - 8), -1):
                    b = samples[back]
                    if b is None:
                        continue
                    if b["lank_y_px"] < baseline_l_ank_y - airborne_lift:
                        l_takeoff_frame = back
                    else:
                        break
                for back in range(i - 1, max(session_start_idx - 1, i - 8), -1):
                    b = samples[back]
                    if b is None:
                        continue
                    if b["rank_y_px"] < baseline_r_ank_y - airborne_lift:
                        r_takeoff_frame = back
                    else:
                        break
                state = "airborne"
            continue

        # airborne
        if apex_hip_y is None or s["hip_y_px"] < apex_hip_y:
            apex_hip_y = s["hip_y_px"]
            apex_frame_idx = i
            apex_row = s

        if l_grounded and r_grounded:
            if airborne_start_idx is None:
                state = "grounded"
                continue
            airborne_frames = i - airborne_start_idx
            if airborne_frames < min_airborne_frames:
                state = "grounded"
                airborne_start_idx = None
                apex_frame_idx = None
                apex_hip_y = None
                apex_row = None
                continue
            # Per-side landing frames — walk forward from the airborne
            # start looking for each ankle's return to landed band.
            l_landing_frame = _side_transition_frame(
                "left_ankle", airborne_start_idx, "landing",
            ) or i
            r_landing_frame = _side_transition_frame(
                "right_ankle", airborne_start_idx, "landing",
            ) or i

            takeoff_sample = samples[airborne_start_idx] or s
            landing_sample = s
            apex_sample = apex_row or s

            takeoff_ms = float(takeoff_sample["t_ms"])
            landing_ms = float(landing_sample["t_ms"])
            flight_sec = max(0.0, (landing_ms - takeoff_ms) / 1000.0)

            jump_height_px = max(
                0.0,
                float(baseline_hip_y - (apex_hip_y or baseline_hip_y)),
            )

            l_kfppa = _kfppa_deg(
                landing_sample["lhip_x_px"], landing_sample["lhip_y_px"],
                landing_sample["lkn_x_px"],  landing_sample["lkn_y_px"],
                landing_sample["lank_x_px"], landing_sample["lank_y_px"],
            )
            r_kfppa = _kfppa_deg(
                landing_sample["rhip_x_px"], landing_sample["rhip_y_px"],
                landing_sample["rkn_x_px"],  landing_sample["rkn_y_px"],
                landing_sample["rank_x_px"], landing_sample["rank_y_px"],
            )

            landing_ank_spread_px = abs(
                landing_sample["rank_x_px"] - landing_sample["lank_x_px"]
            )
            spread_ratio = (
                landing_ank_spread_px / baseline_hip_span
                if baseline_hip_span > 1 else 0.0
            )

            apex_l_thigh_rise = apex_sample["hip_y_px"] - apex_sample["lkn_y_px"]
            apex_r_thigh_rise = apex_sample["hip_y_px"] - apex_sample["rkn_y_px"]

            def _t_ms(idx: Optional[int], fallback: float) -> float:
                if idx is None or idx < 0 or idx >= len(samples):
                    return fallback
                s2 = samples[idx]
                return float(s2["t_ms"]) if s2 is not None else fallback

            takeoff_delta_ms = abs(
                _t_ms(l_takeoff_frame, takeoff_ms)
                - _t_ms(r_takeoff_frame, takeoff_ms)
            )
            landing_delta_ms = abs(
                _t_ms(l_landing_frame, landing_ms)
                - _t_ms(r_landing_frame, landing_ms)
            )

            grounded_since_prev_ms: Optional[float] = None
            if prev_landing_row is not None:
                grounded_since_prev_ms = takeoff_ms - float(prev_landing_row["t_ms"])

            jumps.append({
                "jump_index": len(jumps),
                "takeoff_frame_index": int(airborne_start_idx),
                "apex_frame_index": int(apex_frame_idx or airborne_start_idx),
                "landing_frame_index": int(i),
                "takeoff_t_ms": takeoff_ms,
                "landing_t_ms": landing_ms,
                "flight_time_sec": flight_sec,
                "jump_height_px": jump_height_px,
                "jump_height_cm": _px_to_cm(jump_height_px),
                "physics_height_cm": (_GRAVITY_M_PER_S2 * flight_sec * flight_sec / 8.0) * 100.0,
                "landing_kfppa_left_deg": l_kfppa,
                "landing_kfppa_right_deg": r_kfppa,
                "landing_kfppa_worse_deg": max(l_kfppa, r_kfppa),
                "landing_ank_spread_px": landing_ank_spread_px,
                "landing_ank_spread_ratio": spread_ratio,
                "apex_l_thigh_rise_px": apex_l_thigh_rise,
                "apex_r_thigh_rise_px": apex_r_thigh_rise,
                "takeoff_side_delta_ms": takeoff_delta_ms,
                "landing_side_delta_ms": landing_delta_ms,
                "grounded_since_prev_ms": grounded_since_prev_ms,
                "landing_ank_left_x_px": landing_sample["lank_x_px"],
                "landing_ank_right_x_px": landing_sample["rank_x_px"],
            })
            prev_landing_row = landing_sample
            state = "grounded"
            airborne_start_idx = None
            apex_frame_idx = None
            apex_hip_y = None
            apex_row = None

    if not jumps:
        raise ValueError(
            "no_jumps_detected: no takeoff-landing cycles were detected. "
            "Re-record with clear continuous tuck jumps — both feet leave "
            "the ground together and land together."
        )

    # 6) Session aggregates + Myer's 10-item checklist
    valgus_worse = [j["landing_kfppa_worse_deg"] for j in jumps]
    hip_rises_px = [j["jump_height_px"] for j in jumps]
    l_thighs = [j["apex_l_thigh_rise_px"] for j in jumps]
    r_thighs = [j["apex_r_thigh_rise_px"] for j in jumps]
    spread_ratios = [j["landing_ank_spread_ratio"] for j in jumps]
    takeoff_deltas = [j["takeoff_side_delta_ms"] for j in jumps]
    landing_deltas = [j["landing_side_delta_ms"] for j in jumps]
    gaps_ms = [
        j["grounded_since_prev_ms"] for j in jumps
        if isinstance(j.get("grounded_since_prev_ms"), (int, float))
    ]
    landing_x_l = [j["landing_ank_left_x_px"] for j in jumps]
    landing_x_r = [j["landing_ank_right_x_px"] for j in jumps]

    mid = max(1, len(jumps) // 2)
    valgus_first = _mean(valgus_worse[:mid])
    valgus_second = _mean(valgus_worse[mid:]) if len(valgus_worse) > mid else valgus_first
    height_first = _mean(hip_rises_px[:mid])
    height_second = _mean(hip_rises_px[mid:]) if len(hip_rises_px) > mid else height_first

    # 1. Valgus at landing — any landing over threshold
    item1_fail = any(v > _VALGUS_KFPPA_FAIL_DEG for v in valgus_worse)
    # 2. Thighs not reach parallel — mean hip-rise below the fraction floor
    thigh_parallel_min = leg_length_px * _THIGH_PARALLEL_MIN_HIP_LIFT_FRAC_OF_LEG
    mean_hip_rise = _mean(hip_rises_px)
    item2_fail = mean_hip_rise < thigh_parallel_min
    # 3. Thighs not equal side-to-side — L vs R apex thigh delta
    delta_thigh_frac = abs(_mean(l_thighs) - _mean(r_thighs)) / max(1.0, leg_length_px)
    item3_fail = delta_thigh_frac > _THIGH_SYMMETRY_MAX_DELTA_FRAC
    # 4. Foot placement not shoulder-width
    mean_spread_ratio = _mean(spread_ratios)
    item4_fail = (
        mean_spread_ratio < _FOOT_SHOULDERWIDTH_MIN_RATIO
        or mean_spread_ratio > _FOOT_SHOULDERWIDTH_MAX_RATIO
    )
    # 5. NOT ASSESSED
    # 6. Contact timing not equal
    max_takeoff_delta = max(takeoff_deltas) if takeoff_deltas else 0.0
    max_landing_delta = max(landing_deltas) if landing_deltas else 0.0
    item6_fail = (
        max_takeoff_delta > _CONTACT_TIMING_MAX_DELTA_MS
        or max_landing_delta > _CONTACT_TIMING_MAX_DELTA_MS
    )
    # 7. NOT ASSESSED
    # 8. Pause between jumps
    pause_gap_max_ms = max(gaps_ms) if gaps_ms else 0.0
    item8_fail = pause_gap_max_ms > _PAUSE_GAP_SEC * 1000.0
    # 9. Technique declines
    height_fade_frac = (
        (height_first - height_second) / height_first if height_first > 1 else 0.0
    )
    valgus_growth = valgus_second - valgus_first
    item9_fail = (
        height_fade_frac > _FATIGUE_HEIGHT_FADE_FRAC
        or valgus_growth > _FATIGUE_VALGUS_GROWTH_DEG
    )
    # 10. Footprint drift
    drift_l = _safe_stdev(landing_x_l)
    drift_r = _safe_stdev(landing_x_r)
    drift_frac = (drift_l + drift_r) / (2 * leg_length_px)
    item10_fail = drift_frac > _FOOTPRINT_DRIFT_MAX_FRAC_OF_LEG

    measurable_fails = sum(1 for f in [
        item1_fail, item2_fail, item3_fail, item4_fail,
        item6_fail, item8_fail, item9_fail, item10_fail,
    ] if f)
    if measurable_fails <= 2:
        classification = "good"
    elif measurable_fails <= 5:
        classification = "moderate"
    else:
        classification = "poor"

    checklist = [
        {
            "index": 1,
            "label": "Lower-extremity valgus at landing",
            "status": "fail" if item1_fail else "pass",
            "detail": (
                f"Worst KFPPA {max(valgus_worse):.1f}° "
                f"(threshold {_VALGUS_KFPPA_FAIL_DEG:.0f}°)"
            ),
        },
        {
            "index": 2,
            "label": "Thighs not reach parallel",
            "status": "fail" if item2_fail else "pass",
            "detail": (
                f"Mean hip rise {mean_hip_rise:.0f} px "
                f"(need ≥ {thigh_parallel_min:.0f})"
            ),
        },
        {
            "index": 3,
            "label": "Thighs not equal side-to-side",
            "status": "fail" if item3_fail else "pass",
            "detail": (
                f"L/R thigh-rise delta {delta_thigh_frac * 100:.0f}% "
                "of leg length"
            ),
        },
        {
            "index": 4,
            "label": "Foot placement not shoulder-width",
            "status": "fail" if item4_fail else "pass",
            "detail": (
                f"Mean landing ankle spread {mean_spread_ratio:.2f}× hip span"
            ),
        },
        {
            "index": 5,
            "label": "Foot placement not parallel",
            "status": "not_assessed",
            "detail": (
                "Frontal single-camera can't measure foot yaw reliably."
            ),
        },
        {
            "index": 6,
            "label": "Foot contact timing not equal",
            "status": "fail" if item6_fail else "pass",
            "detail": (
                f"Max side gap — takeoff {max_takeoff_delta:.0f} ms · "
                f"landing {max_landing_delta:.0f} ms"
            ),
        },
        {
            "index": 7,
            "label": "Excessive landing contact noise",
            "status": "not_assessed",
            "detail": "Requires force-plate or audio; not single-camera.",
        },
        {
            "index": 8,
            "label": "Pause between jumps",
            "status": "fail" if item8_fail else "pass",
            "detail": (
                f"Longest grounded gap {pause_gap_max_ms:.0f} ms "
                f"(threshold {_PAUSE_GAP_SEC * 1000:.0f} ms)"
            ),
        },
        {
            "index": 9,
            "label": "Technique declines before 10 s",
            "status": "fail" if item9_fail else "pass",
            "detail": (
                f"Height fade {height_fade_frac * 100:.0f}%; "
                f"valgus growth {valgus_growth:+.1f}°"
            ),
        },
        {
            "index": 10,
            "label": "Does not land in same footprint",
            "status": "fail" if item10_fail else "pass",
            "detail": (
                f"Landing ankle-x std {drift_frac * 100:.0f}% of leg length"
            ),
        },
    ]

    if jumps:
        best_jump = max(jumps, key=lambda j: j["jump_height_px"])
        apex_screenshot = _capture_apex_frame(video_path, best_jump["apex_frame_index"])
    else:
        apex_screenshot = None

    valid_frames = len(valid_samples)

    return {
        "patient_height_cm": patient_height_cm,
        "calibration": final_calibration,
        "baseline_hip_y_px": float(baseline_hip_y),
        "baseline_left_ankle_y_px": float(baseline_l_ank_y),
        "baseline_right_ankle_y_px": float(baseline_r_ank_y),
        "baseline_ank_spread_px": float(
            abs(baseline_r_ank_x - baseline_l_ank_x),
        ),
        "baseline_shoulder_hip_span_px": float(baseline_hip_span),
        "leg_length_px": float(leg_length_px),
        "jumps": jumps,
        "jump_count": len(jumps),
        "mean_jump_height_px": mean_hip_rise,
        "mean_jump_height_cm": _px_to_cm(mean_hip_rise),
        "mean_valgus_worse_deg": _mean(valgus_worse),
        "max_valgus_worse_deg": max(valgus_worse) if valgus_worse else 0.0,
        "height_fade_frac": height_fade_frac,
        "valgus_growth_deg": valgus_growth,
        "footprint_drift_frac": drift_frac,
        "pause_gap_max_ms": pause_gap_max_ms,
        "duration_seconds": duration_seconds,
        "checklist": checklist,
        "measurable_fails": measurable_fails,
        "classification": classification,
        "peak_screenshot_data_url": apex_screenshot,
        "fps": float(fps) if fps else None,
        "total_frames": int(n),
        "valid_frames": int(valid_frames),
        "interpretation": None,
    }
