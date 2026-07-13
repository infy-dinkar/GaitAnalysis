"""overhead_squat_engine.py — backend B2 Overhead Squat analyzer.

Clinical context — NASM / FMS-style Overhead Squat Assessment:
  Patient stands feet shoulder-width apart, arms held straight
  overhead (biceps by ears). Performs 3-5 slow squats to about
  parallel depth then returns to standing. FRONTAL single-camera
  screen — measures a subset of the classic checklist plus a
  proxy for depth.

Frontal-plane checklist (7 items, 5 measurable, 2 not_assessed):
  1. Knees cave in at bottom (valgus, KFPPA)      — measurable
  2. Hip / pelvic drop asymmetry at bottom         — measurable
  3. Foot placement not shoulder-width             — measurable
  4. Arms fall to sides (wrist Y drops toward or
     below shoulder Y — arms not staying overhead) — measurable
  5. Squat depth insufficient (hip descent below
     required fraction of leg length)              — measurable proxy
  6. Torso excessive forward lean                  — not_assessed
     (sagittal view required)
  7. Heels rise off the ground                     — not_assessed
     (needs feet close-up; MediaPipe foot dorsi-
     flexion signal at this scale is unreliable)

Classification (of the 5 measurable items):
  good     ≤ 1 fail
  moderate ≤ 3 fails
  poor     ≥ 4 fails

Pipeline structurally mirrors tuck_jump_engine.py:
  • extract_poses → build_time_series (dict keyed by landmark name)
  • Standing baseline lock (hip Y + wrist Y overhead)
  • State machine — squat-descent detection (NOT jump/airborne)
  • Per-rep bottom-frame KFPPA + pelvic tilt + arm-drop + depth
  • Best-rep screenshot via cv2

Reuses `_kfppa_deg` verbatim from tuck_jump_engine.py — imported so
there is a single source of truth for the KFPPA math.
"""
from __future__ import annotations

import base64
import logging
import math
from statistics import median as _stdlib_median
from typing import Any, Optional

import cv2

from engines.calibration.height_calibration import (
    build_height_calibration_dict,
    measure_body_pixel_height_from_time_series,
    probe_source_frame_dimensions,
)
from engines.gait_engine import (
    LM,  # noqa: F401 — parity with sibling engines
    build_time_series,
    extract_poses,
)
from engines.orthopedic.tuck_jump_engine import _kfppa_deg

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/overheadSquat.ts) ──────
_VIS_THRESHOLD = 0.15
_ARM_VIS_FLOOR = 0.10  # arms overhead often near frame edge
_SAMPLE_HZ = 30.0
_STANDING_HOLD_SEC = 0.3
_STANDING_HOLD_SAMPLES = int(round(_STANDING_HOLD_SEC * _SAMPLE_HZ))
_MAX_SESSION_DURATION_SEC = 20.0

# Squat descent detection — hip Y grows (drops) below standing baseline
# by at least this fraction of leg length to count as "descending".
_DESCENT_MIN_FRAC_OF_LEG = 0.05
# Rep is considered "at bottom" when descent depth has plateaued —
# derivative of hip Y flips sign OR sustains within this band.
_BOTTOM_PLATEAU_BAND_FRAC = 0.015
# Return to within this band of baseline hip Y closes the rep.
_STANDING_RETURN_BAND_FRAC = 0.02
_MIN_REP_DURATION_SEC = 0.6

_STANDING_TOLERANCE_FRAC_OF_LEG = 0.04
_STANDING_TOLERANCE_FALLBACK_MULT = 3.0

# Per-item thresholds
_VALGUS_KFPPA_FAIL_DEG = 12.0
_PELVIC_TILT_FAIL_FRAC_OF_HIPSPAN = 0.10  # |L hip - R hip| / hip span
_FOOT_SHOULDERWIDTH_MIN_RATIO = 0.7
_FOOT_SHOULDERWIDTH_MAX_RATIO = 1.6
# Arm overhead — wrist should be above shoulder by at least this
# fraction of the shoulder-hip trunk length. Drop below this = fail.
_ARM_OVERHEAD_MIN_FRAC_OF_TRUNK = 0.30
# Depth proxy — hip descent should reach at least this frac of leg
# length below standing to be considered "adequate depth".
_DEPTH_TARGET_FRAC_OF_LEG = 0.20


# ─── Small helpers ─────────────────────────────────────────────────
def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


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
        log.warning("overhead_squat: frame capture failed: %s", e)
        return None


def _sample_row(ts: dict, i: int, fps: float) -> Optional[dict]:
    """Read one frame from ts, gated by visibility on the required
    landmarks. Wrists / elbows are also read for the arm-drop item
    but with a looser visibility floor (arms overhead often clip)."""
    required = (
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_shoulder", "right_shoulder",
    )
    if not all(_visible(ts, k, i) for k in required):
        return None

    def _v(key: str, comp: str) -> float:
        return float(ts[key][comp][i])

    # Arms — accept lower visibility, but always read; consumer will
    # gate arm-drop item on whether both wrists cleared _ARM_VIS_FLOOR.
    def _arm_vis(key: str) -> float:
        return float(ts[key]["vis"][i])

    l_wrist_vis = _arm_vis("left_wrist")
    r_wrist_vis = _arm_vis("right_wrist")

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
        "lsh_y_px": _v("left_shoulder", "y_px"),
        "rsh_y_px": _v("right_shoulder", "y_px"),
        "shoulder_y_px": (_v("left_shoulder", "y_px") + _v("right_shoulder", "y_px")) / 2,
        "lwr_y_px": _v("left_wrist", "y_px") if l_wrist_vis >= _ARM_VIS_FLOOR else None,
        "rwr_y_px": _v("right_wrist", "y_px") if r_wrist_vis >= _ARM_VIS_FLOOR else None,
        "l_wrist_vis": l_wrist_vis,
        "r_wrist_vis": r_wrist_vis,
    }


# ─── Main analyzer ─────────────────────────────────────────────────
def analyze_overhead_squat(
    video_path: str,
    pose_options: Any,
    calibration: Optional[dict] = None,
    patient_height_cm: Optional[float] = None,
) -> dict:
    """Analyze an overhead-squat video. Returns a dict matching the
    frontend OverheadSquatResult TypeScript interface.

    Raises ValueError with a user-facing code ('poor_visibility',
    'no_baseline', 'no_reps_detected') on hard failures.
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
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
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
                    "overhead_squat: height calibration: body_px=%.0f "
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

    # 4) Provisional leg length for thresholds
    leg_length_px = 0.0
    for s in valid_samples[:min(60, len(valid_samples))]:
        gap = max(s["lank_y_px"], s["rank_y_px"]) - s["hip_y_px"]
        if gap > leg_length_px:
            leg_length_px = gap
    if leg_length_px <= 0:
        leg_length_px = 300.0
    standing_tol_px = _STANDING_TOLERANCE_FRAC_OF_LEG * leg_length_px
    fallback_ceiling_px = standing_tol_px * _STANDING_TOLERANCE_FALLBACK_MULT

    # 5) Standing baseline lock — hips + wrists overhead + shoulders
    baseline_hip_y: Optional[float] = None
    baseline_l_ank_x: Optional[float] = None
    baseline_r_ank_x: Optional[float] = None
    baseline_hip_span: Optional[float] = None
    baseline_shoulder_y: Optional[float] = None
    baseline_wrist_y: Optional[float] = None  # median of L/R wrist Y (overhead)
    baseline_lock_idx: Optional[int] = None
    best_fallback = None

    for end_i in range(_STANDING_HOLD_SAMPLES - 1, len(samples)):
        window = samples[end_i - _STANDING_HOLD_SAMPLES + 1: end_i + 1]
        rows = [s for s in window if s is not None]
        if len(rows) < _STANDING_HOLD_SAMPLES // 2:
            continue
        hys = [s["hip_y_px"] for s in rows]
        max_dev = max(abs(y - _stdlib_median(hys)) for y in hys)
        med_hy = _stdlib_median(hys)
        med_lax = _stdlib_median([s["lank_x_px"] for s in rows])
        med_rax = _stdlib_median([s["rank_x_px"] for s in rows])
        med_lhip_x = _stdlib_median([s["lhip_x_px"] for s in rows])
        med_rhip_x = _stdlib_median([s["rhip_x_px"] for s in rows])
        med_sh_y = _stdlib_median([s["shoulder_y_px"] for s in rows])
        wrist_ys: list[float] = []
        for s in rows:
            if s["lwr_y_px"] is not None:
                wrist_ys.append(s["lwr_y_px"])
            if s["rwr_y_px"] is not None:
                wrist_ys.append(s["rwr_y_px"])
        med_wrist_y = _stdlib_median(wrist_ys) if wrist_ys else None
        entry = (
            max_dev, float(med_hy),
            float(med_lax), float(med_rax),
            abs(float(med_rhip_x) - float(med_lhip_x)),
            float(med_sh_y),
            float(med_wrist_y) if med_wrist_y is not None else None,
            end_i,
        )
        if max_dev <= standing_tol_px:
            (
                _, baseline_hip_y,
                baseline_l_ank_x, baseline_r_ank_x, baseline_hip_span,
                baseline_shoulder_y, baseline_wrist_y, baseline_lock_idx,
            ) = entry
            break
        if best_fallback is None or entry[0] < best_fallback[0]:
            best_fallback = entry

    if baseline_hip_y is None and best_fallback is not None:
        if best_fallback[0] <= fallback_ceiling_px:
            (
                _, baseline_hip_y,
                baseline_l_ank_x, baseline_r_ank_x, baseline_hip_span,
                baseline_shoulder_y, baseline_wrist_y, baseline_lock_idx,
            ) = best_fallback

    if (
        baseline_hip_y is None
        or baseline_l_ank_x is None
        or baseline_r_ank_x is None
        or baseline_hip_span is None
        or baseline_shoulder_y is None
        or baseline_lock_idx is None
    ):
        raise ValueError(
            "no_baseline: patient did not stand still with arms overhead "
            "for ~0.3 s at the start of the clip. Re-record with a brief "
            "static stance holding arms straight overhead before the first "
            "squat."
        )

    # Recompute leg length from locked baseline (using ankle Y averaged).
    lock_row = samples[baseline_lock_idx] or {}
    ank_y_ref = (
        (lock_row.get("lank_y_px", baseline_hip_y + leg_length_px) +
         lock_row.get("rank_y_px", baseline_hip_y + leg_length_px)) / 2
    )
    leg_length_px = max(1.0, ank_y_ref - baseline_hip_y)
    if baseline_hip_span < 1.0:
        baseline_hip_span = leg_length_px * 0.3

    trunk_length_px = max(1.0, baseline_hip_y - baseline_shoulder_y)

    descent_min_px = leg_length_px * _DESCENT_MIN_FRAC_OF_LEG
    bottom_plateau_band_px = leg_length_px * _BOTTOM_PLATEAU_BAND_FRAC
    standing_return_band_px = leg_length_px * _STANDING_RETURN_BAND_FRAC
    depth_target_px = leg_length_px * _DEPTH_TARGET_FRAC_OF_LEG

    # 6) Squat-descent state machine
    max_frames = int(round(_MAX_SESSION_DURATION_SEC * _SAMPLE_HZ))
    session_start_idx = baseline_lock_idx + 1
    session_end_idx = min(len(samples), session_start_idx + max_frames)
    min_rep_frames = max(1, int(round(_MIN_REP_DURATION_SEC * _SAMPLE_HZ)))

    state = "standing"
    descent_start_idx: Optional[int] = None
    bottom_frame_idx: Optional[int] = None
    bottom_hip_y: Optional[float] = None
    reps: list[dict] = []

    for i in range(session_start_idx, session_end_idx):
        s = samples[i]
        if s is None:
            continue
        # Distance below standing baseline (positive = descending).
        drop = s["hip_y_px"] - baseline_hip_y

        if state == "standing":
            if drop >= descent_min_px:
                descent_start_idx = i
                bottom_frame_idx = i
                bottom_hip_y = s["hip_y_px"]
                state = "descending"
            continue

        # descending or ascending — track the deepest point
        if bottom_hip_y is None or s["hip_y_px"] > bottom_hip_y:
            bottom_hip_y = s["hip_y_px"]
            bottom_frame_idx = i

        # Close rep when hip returns within standing_return_band_px
        # of baseline (i.e. drop back near zero).
        if abs(drop) <= standing_return_band_px:
            if descent_start_idx is None:
                state = "standing"
                continue
            rep_frames = i - descent_start_idx
            if rep_frames < min_rep_frames:
                # Too fast to be a real rep — likely a wobble.
                state = "standing"
                descent_start_idx = None
                bottom_frame_idx = None
                bottom_hip_y = None
                continue

            bottom_sample = samples[bottom_frame_idx] if bottom_frame_idx is not None else s
            if bottom_sample is None:
                bottom_sample = s
            depth_px = max(0.0, (bottom_hip_y or baseline_hip_y) - baseline_hip_y)

            # KFPPA at bottom (both sides)
            l_kfppa = _kfppa_deg(
                bottom_sample["lhip_x_px"], bottom_sample["lhip_y_px"],
                bottom_sample["lkn_x_px"],  bottom_sample["lkn_y_px"],
                bottom_sample["lank_x_px"], bottom_sample["lank_y_px"],
            )
            r_kfppa = _kfppa_deg(
                bottom_sample["rhip_x_px"], bottom_sample["rhip_y_px"],
                bottom_sample["rkn_x_px"],  bottom_sample["rkn_y_px"],
                bottom_sample["rank_x_px"], bottom_sample["rank_y_px"],
            )

            # Pelvic tilt at bottom — |L hip Y − R hip Y| as fraction
            # of horizontal hip span (Trendelenburg-style).
            pelvic_tilt_px = abs(bottom_sample["lhip_y_px"] - bottom_sample["rhip_y_px"])
            pelvic_tilt_frac = (
                pelvic_tilt_px / baseline_hip_span if baseline_hip_span > 1 else 0.0
            )

            # Foot placement — ankle spread relative to baseline hip span
            ank_spread_px = abs(bottom_sample["rank_x_px"] - bottom_sample["lank_x_px"])
            ank_spread_ratio = (
                ank_spread_px / baseline_hip_span if baseline_hip_span > 1 else 0.0
            )

            # Arm drop — wrist Y at bottom vs shoulder Y at bottom. In
            # a good overhead squat, wrist Y stays well above shoulder Y
            # (i.e. wrist Y < shoulder Y − trunk_length * threshold).
            def _arm_drop_frac(wr_y: Optional[float]) -> Optional[float]:
                if wr_y is None:
                    return None
                shoulder_y_bottom = (bottom_sample["lsh_y_px"] + bottom_sample["rsh_y_px"]) / 2
                # Positive gap = wrist above shoulder (Y smaller means higher on screen).
                gap_px = shoulder_y_bottom - wr_y
                return gap_px / trunk_length_px

            l_arm_frac = _arm_drop_frac(bottom_sample["lwr_y_px"])
            r_arm_frac = _arm_drop_frac(bottom_sample["rwr_y_px"])
            arm_fracs = [f for f in (l_arm_frac, r_arm_frac) if f is not None]
            worst_arm_frac = min(arm_fracs) if arm_fracs else None

            reps.append({
                "rep_index": len(reps),
                "descent_start_frame_index": int(descent_start_idx),
                "bottom_frame_index": int(bottom_frame_idx or descent_start_idx),
                "return_frame_index": int(i),
                "descent_start_t_ms": float(samples[descent_start_idx]["t_ms"])
                if samples[descent_start_idx] else 0.0,
                "bottom_t_ms": float(bottom_sample["t_ms"]),
                "return_t_ms": float(s["t_ms"]),
                "depth_px": depth_px,
                "depth_cm": _px_to_cm(depth_px),
                "depth_frac_of_leg": depth_px / leg_length_px,
                "bottom_kfppa_left_deg": l_kfppa,
                "bottom_kfppa_right_deg": r_kfppa,
                "bottom_kfppa_worse_deg": max(l_kfppa, r_kfppa),
                "bottom_pelvic_tilt_px": pelvic_tilt_px,
                "bottom_pelvic_tilt_frac": pelvic_tilt_frac,
                "bottom_ank_spread_px": ank_spread_px,
                "bottom_ank_spread_ratio": ank_spread_ratio,
                "bottom_l_arm_overhead_frac": l_arm_frac,
                "bottom_r_arm_overhead_frac": r_arm_frac,
                "bottom_worst_arm_overhead_frac": worst_arm_frac,
            })

            state = "standing"
            descent_start_idx = None
            bottom_frame_idx = None
            bottom_hip_y = None

    if not reps:
        raise ValueError(
            "no_reps_detected: no squat descents were detected. Re-record "
            "with the patient performing 3-5 slow overhead squats to about "
            "parallel depth, returning fully to standing between reps."
        )

    # 7) Session aggregates + 7-item checklist
    valgus_worst = [r["bottom_kfppa_worse_deg"] for r in reps]
    pelvic_fracs = [r["bottom_pelvic_tilt_frac"] for r in reps]
    spread_ratios = [r["bottom_ank_spread_ratio"] for r in reps]
    depth_fracs = [r["depth_frac_of_leg"] for r in reps]
    arm_fracs_worst = [
        r["bottom_worst_arm_overhead_frac"] for r in reps
        if r["bottom_worst_arm_overhead_frac"] is not None
    ]

    # Item 1 — valgus
    max_valgus = max(valgus_worst) if valgus_worst else 0.0
    item1_fail = max_valgus > _VALGUS_KFPPA_FAIL_DEG

    # Item 2 — pelvic drop
    max_pelvic_frac = max(pelvic_fracs) if pelvic_fracs else 0.0
    item2_fail = max_pelvic_frac > _PELVIC_TILT_FAIL_FRAC_OF_HIPSPAN

    # Item 3 — foot placement
    mean_spread = _mean(spread_ratios)
    item3_fail = (
        mean_spread < _FOOT_SHOULDERWIDTH_MIN_RATIO
        or mean_spread > _FOOT_SHOULDERWIDTH_MAX_RATIO
    )

    # Item 4 — arm drop (arms not overhead throughout)
    if arm_fracs_worst:
        min_arm_frac = min(arm_fracs_worst)
        item4_fail = min_arm_frac < _ARM_OVERHEAD_MIN_FRAC_OF_TRUNK
        arm_detail = (
            f"Worst wrist-above-shoulder gap {min_arm_frac * 100:.0f}% "
            f"of trunk length (need ≥ {int(_ARM_OVERHEAD_MIN_FRAC_OF_TRUNK * 100)}%)"
        )
    else:
        item4_fail = False
        min_arm_frac = None
        arm_detail = (
            "Wrists not reliably visible for this recording — arm "
            "position not scored (raise camera or step back so wrists "
            "clear the top of frame)."
        )

    # Item 5 — depth
    max_depth_frac = max(depth_fracs) if depth_fracs else 0.0
    item5_fail = max_depth_frac < _DEPTH_TARGET_FRAC_OF_LEG

    measurable_fails = sum(
        1 for f in [item1_fail, item2_fail, item3_fail, item4_fail, item5_fail] if f
    )
    if measurable_fails <= 1:
        classification = "good"
    elif measurable_fails <= 3:
        classification = "moderate"
    else:
        classification = "poor"

    checklist = [
        {
            "index": 1,
            "label": "Knees cave in at bottom (valgus)",
            "status": "fail" if item1_fail else "pass",
            "detail": (
                f"Worst KFPPA {max_valgus:.1f}° "
                f"(threshold {_VALGUS_KFPPA_FAIL_DEG:.0f}°)"
            ),
        },
        {
            "index": 2,
            "label": "Hip / pelvic drop asymmetry at bottom",
            "status": "fail" if item2_fail else "pass",
            "detail": (
                f"Max L/R hip Y offset {max_pelvic_frac * 100:.0f}% of hip span "
                f"(threshold {int(_PELVIC_TILT_FAIL_FRAC_OF_HIPSPAN * 100)}%)"
            ),
        },
        {
            "index": 3,
            "label": "Foot placement not shoulder-width",
            "status": "fail" if item3_fail else "pass",
            "detail": (
                f"Mean ankle spread {mean_spread:.2f}× hip span "
                f"(window {_FOOT_SHOULDERWIDTH_MIN_RATIO:.1f}–{_FOOT_SHOULDERWIDTH_MAX_RATIO:.1f})"
            ),
        },
        {
            "index": 4,
            "label": "Arms fall from overhead (wrist drops toward shoulder)",
            "status": (
                "not_assessed" if not arm_fracs_worst
                else ("fail" if item4_fail else "pass")
            ),
            "detail": arm_detail,
        },
        {
            "index": 5,
            "label": "Squat depth insufficient (proxy — frontal-view only)",
            "status": "fail" if item5_fail else "pass",
            "detail": (
                f"Deepest rep hip descent {max_depth_frac * 100:.0f}% of leg length "
                f"(target ≥ {int(_DEPTH_TARGET_FRAC_OF_LEG * 100)}%)"
            ),
        },
        {
            "index": 6,
            "label": "Excessive torso forward lean",
            "status": "not_assessed",
            "detail": (
                "Sagittal view required — a single frontal camera can't "
                "resolve trunk pitch."
            ),
        },
        {
            "index": 7,
            "label": "Heels rise off the ground",
            "status": "not_assessed",
            "detail": (
                "Feet close-up / sagittal view required — MediaPipe foot "
                "dorsiflexion signal at this scale is unreliable."
            ),
        },
    ]

    # Best rep for screenshot = the rep with the WORST valgus (most
    # clinically informative frame). Falls back to deepest rep.
    if reps:
        worst_rep = max(reps, key=lambda r: r["bottom_kfppa_worse_deg"])
        peak_screenshot = _capture_frame(video_path, worst_rep["bottom_frame_index"])
    else:
        peak_screenshot = None

    valid_frames = len(valid_samples)

    return {
        "patient_height_cm": patient_height_cm,
        "calibration": final_calibration,
        "baseline_hip_y_px": float(baseline_hip_y),
        "baseline_shoulder_y_px": float(baseline_shoulder_y),
        "baseline_wrist_y_px": float(baseline_wrist_y) if baseline_wrist_y is not None else None,
        "baseline_ank_spread_px": float(abs(baseline_r_ank_x - baseline_l_ank_x)),
        "baseline_hip_span_px": float(baseline_hip_span),
        "leg_length_px": float(leg_length_px),
        "trunk_length_px": float(trunk_length_px),
        "reps": reps,
        "rep_count": len(reps),
        "mean_depth_frac": _mean(depth_fracs),
        "max_depth_frac": max_depth_frac,
        "max_depth_cm": _px_to_cm(max(r["depth_px"] for r in reps)) if reps else None,
        "mean_valgus_worse_deg": _mean(valgus_worst),
        "max_valgus_worse_deg": max_valgus,
        "max_pelvic_tilt_frac": max_pelvic_frac,
        "mean_ank_spread_ratio": mean_spread,
        "min_arm_overhead_frac": min_arm_frac,
        "duration_seconds": duration_seconds,
        "checklist": checklist,
        "measurable_fails": measurable_fails,
        "classification": classification,
        "peak_screenshot_data_url": peak_screenshot,
        "fps": float(fps) if fps else None,
        "total_frames": int(n),
        "valid_frames": int(valid_frames),
        "interpretation": None,
    }
