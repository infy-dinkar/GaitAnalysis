"""single_leg_hop_engine.py — backend D3 Single-Leg Hop test
(forward hop for distance) on the MediaPipe BlazePose Full
(33-keypoint) stack.

Clinical context:
  Standard single-leg-forward-hop functional test used for ACL
  rehab clearance and lower-extremity asymmetry screening.
  Patient stands on the test leg, hops forward as far as
  comfortable, lands and holds on the same leg. Up to 3 trials
  per leg. Best valid hop distance per leg is the primary
  outcome; the Limb Symmetry Index (LSI = weaker / stronger ×
  100) across L vs R is the asymmetry metric. ACL clearance
  convention: LSI ≥ 90 %.

Pipeline (mirrors functional_reach_engine.py architecturally):
  1. Calibration is provider-agnostic. Caller supplies a
     CalibrationResult-shaped dict OR a patient_height_cm so the
     engine can derive pixels_per_cm from the standing window
     itself. When neither is available the engine runs in
     RELATIVE-UNITS-ONLY mode (pixel distances only; no LSI
     classification).
  2. Reuse gait_engine.extract_poses() + build_time_series() to
     get smoothed 33-keypoint landmarks per frame.
  3. Per-frame samples: test-side heel.x / .y / vis,
     foot_index.x / .y / vis, ankle.y / vis, contralateral
     ankle.y / vis (single-leg validity gate).
  4. Standing-baseline lock-in — first N stable frames where the
     test-side foot is grounded (ankle.y near baseline). The
     locked baseline_ankle_y is the ground reference.
  5. Takeoff / landing event detection — same threshold-crossing
     state machine as chair_stand_30s_engine, applied to
     test-side ankle.y instead of hip.y. Each (takeoff, landing)
     pair = one trial.
  6. Per-trial hop distance = |heel.x_landing − heel.x_takeoff|.
     Falls back to foot_index.x at takeoff/landing when heel
     visibility was poor for that frame.
  7. Per-trial validity gate: contralateral ankle.y must NOT
     drop to within the ground band during the airborne window
     (otherwise patient cheated by bringing the other leg down).
  8. Best-valid screenshot at the landing frame of the longest
     valid trial.
  9. Return a dict matching the frontend SingleLegHopResult
     shape (api.py wraps it in SingleLegHopResponse).
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
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/singleLegHop.ts) ─────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 30.0     # higher than FR's 15 Hz — hop events are brief
_MAX_TRIALS = 3
_STANDING_HOLD_SEC = 0.3           # baseline-lock window — shortened
                                   # from 0.5 s so we have more candidate
                                   # positions to find a stable one
_STANDING_HOLD_SAMPLES = int(round(_STANDING_HOLD_SEC * _SAMPLE_HZ))

# Takeoff / landing detection — applied to ankle.y after baseline
# lock. Fractions of leg-length so they remain calibration-free
# (real-world cm conversion happens later via pixels_per_cm only
# for the reported hop distance, not for event detection).
_AIRBORNE_LIFT_FRAC_OF_LEG = 0.06    # ankle must rise >=6% of leg length above baseline
_LANDED_BAND_FRAC_OF_LEG   = 0.03    # back within 3% of baseline = grounded
_MIN_AIRBORNE_FRAMES_SEC   = 0.10    # >=100 ms airborne to count as a real hop
_MIN_TRIAL_GAP_SEC         = 0.5     # required quiet period between trials

# Validity — minimum hop distance (calibrated) to count as a real
# hop. Below this is either standing-jitter or a stutter-step.
_MIN_HOP_FOR_VALID_CM = 10.0
# Fallback (relative-units mode) — minimum hop distance as fraction
# of leg-length. ~30% of leg-length is roughly a 25 cm hop for an
# average adult; permissive enough that real hops register.
_MIN_HOP_FALLBACK_FRACTION_OF_LEG = 0.30

# Contralateral foot validity: if the OTHER leg's ankle.y enters
# the same grounded band during the airborne window for more than
# this many consecutive samples, the trial is marked invalid (the
# patient touched down with the wrong foot).
_CONTRALATERAL_TOUCH_GRACE_SAMPLES = 2

# Standing-baseline gate — patient is "stably standing" when test-
# side ankle.y is within this fraction of leg-length of its rolling
# median over the recent window. Raised from 0.015 → 0.04 because
# MediaPipe BlazePose's smoothed ankle landmark carries 2-3 % of
# leg-length of residual jitter even when the patient is genuinely
# stationary; the previous 1.5 % cap was tighter than the
# measurement-noise floor and was rejecting valid clips.
_STANDING_TOLERANCE_FRAC_OF_LEG = 0.04
# If no window meets the strict tolerance, accept the most-stable
# window in the clip as a fallback as long as its peak deviation
# stays under this absolute multiple of the strict tolerance. Keeps
# chaotic recordings (patient walks in and immediately hops) failing
# with the helpful "no_baseline" message while letting reasonable
# stances through.
_STANDING_TOLERANCE_FALLBACK_MULT = 3.0

# LSI classification thresholds (standard ACL clearance convention).
_LSI_CLEARED_PCT = 90.0
_LSI_WARNING_PCT = 80.0


_SIDE_INDICES = {
    "left": {
        "ankle": "left_ankle",
        "heel": "left_heel",
        "foot": "left_foot_index",
        "hip": "left_hip",
        "other_ankle": "right_ankle",
    },
    "right": {
        "ankle": "right_ankle",
        "heel": "right_heel",
        "foot": "right_foot_index",
        "hip": "right_hip",
        "other_ankle": "left_ankle",
    },
}


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _all_visible(ts: dict, keys: tuple[str, ...], i: int) -> bool:
    return all(_visible(ts, k, i) for k in keys)


def _classify_lsi(lsi_pct: Optional[float]) -> str:
    if lsi_pct is None:
        return "incomplete"
    if lsi_pct >= _LSI_CLEARED_PCT:
        return "cleared"
    if lsi_pct >= _LSI_WARNING_PCT:
        return "warning"
    return "deficit"


# ─── Peak-frame screenshot ──────────────────────────────────────
# Mirrors functional_reach_engine._grab_peak_frame: tolerant of
# keyframe-only seeking on phone-recorded MP4s by walking forward
# with cheap grab() calls when the decoder lands earlier than the
# requested frame.
def _grab_peak_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    side: str,
) -> Optional[str]:
    if frame_index < 0:
        return None
    pose_rot = int(keypoints_normalized.get("_pose_rotation") or 0)

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        max_advance = max(0, frame_index - landed) + 5
        steps = 0
        while landed < frame_index and steps < max_advance:
            if not cap.grab():
                break
            landed += 1
            steps += 1
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None:
        return None
    if pose_rot:
        frame = _apply_rot(frame, pose_rot)

    h, w = frame.shape[:2]
    target_w = min(640, w)
    if target_w < w:
        scale = target_w / w
        frame = cv2.resize(frame, (target_w, int(h * scale)))
        h, w = frame.shape[:2]

    def _kp_pixel(name: str) -> Optional[tuple[int, int]]:
        frames = keypoints_normalized.get(name, [])
        if frame_index >= len(frames):
            return None
        kp = frames[frame_index]
        if kp is None:
            return None
        x_n, y_n, _vis = kp
        return (int(x_n * w), int(y_n * h))

    dot_pos: dict[str, tuple[int, int]] = {}
    for name in LM:
        p = _kp_pixel(name)
        if p:
            dot_pos[name] = p

    edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_heel"),
        ("right_ankle", "right_heel"),
        ("left_heel", "left_foot_index"),
        ("right_heel", "right_foot_index"),
    ]
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            on_side = a.startswith(side) or b.startswith(side)
            line_color = (255, 255, 255) if on_side else (180, 180, 180)
            cv2.line(frame, dot_pos[a], dot_pos[b], line_color, 2)
    for name, (px_, py_) in dot_pos.items():
        emphasised = name.startswith(side)
        outer = (0, 0, 220) if emphasised else (150, 150, 150)
        cv2.circle(frame, (px_, py_), 5, outer, -1)
        cv2.circle(frame, (px_, py_), 7, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ─── Main entry point ───────────────────────────────────────────
def analyze_single_leg_hop(
    video_path: str,
    pose_options,
    side: str,
    calibration: Optional[dict[str, Any]] = None,
    patient_height_cm: Optional[float] = None,
) -> dict:
    """Run the Single-Leg Hop pipeline on an uploaded clip.

    Args:
        video_path:        path to the uploaded clip on disk
        pose_options:      PoseLandmarkerOptions built by
                           api._build_gait_pose_options()
        side:              'left' or 'right' — the test (hopping) leg
        calibration:       optional pre-detected CalibrationResult
                           dict from the frontend. When None and
                           patient_height_cm is provided, the
                           engine derives pixels_per_cm from the
                           standing window itself.
        patient_height_cm: optional standing height for the
                           server-side calibration fallback.

    Returns:
        Dict matching the frontend SingleLegHopResult shape — per-
        side per-trial entries plus the best valid hop and an
        interpretation string. LSI is computed CLIENT-side (the
        endpoint is per-leg; the report combines the two sides).

    Raises:
        ValueError: 'poor_visibility' / 'no_baseline' /
                    'no_hops_detected' with a user-facing message
                    that the API layer maps to HTTP 400.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    # ── 1) Calibration pass-through ───────────────────────────
    final_calibration = calibration
    ppc: Optional[float] = None
    if final_calibration is not None:
        raw_ppc = final_calibration.get("pixels_per_cm")
        if isinstance(raw_ppc, (int, float)) and raw_ppc > 0:
            ppc = float(raw_ppc)
        else:
            final_calibration = None

    # ── 2) Pose extraction ────────────────────────────────────
    raw, fps, _ = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_heel"]["y"]),
        len(ts["right_heel"]["y"]),
        len(ts["left_foot_index"]["y"]),
        len(ts["right_foot_index"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    idx = _SIDE_INDICES[side]
    required_keys = (idx["ankle"], idx["heel"], idx["foot"], idx["hip"])
    visible_frames = sum(
        1 for i in range(n) if _all_visible(ts, required_keys, i)
    )
    if visible_frames < max(3, int(n * 0.30)):
        raise ValueError("poor_visibility")

    # ── 2b) Height-calibration server-side fallback ───────────
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
                    "single_leg_hop: height calibration: body_px=%.0f "
                    "height_cm=%.1f → %.2f px/cm",
                    body_px, patient_height_cm, ppc,
                )

    # ── 3) Sample at SAMPLE_HZ ────────────────────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(0, n, step))

    samples: list[dict] = []
    for i in sampled_frames:
        t_ms = (i / fps) * 1000.0 if fps > 0 else 0.0
        ankle_y = (
            float(ts[idx["ankle"]]["y_px"][i])
            if _visible(ts, idx["ankle"], i)
            else None
        )
        heel_x = (
            float(ts[idx["heel"]]["x_px"][i])
            if _visible(ts, idx["heel"], i)
            else None
        )
        heel_y = (
            float(ts[idx["heel"]]["y_px"][i])
            if _visible(ts, idx["heel"], i)
            else None
        )
        foot_x = (
            float(ts[idx["foot"]]["x_px"][i])
            if _visible(ts, idx["foot"], i)
            else None
        )
        hip_y = (
            float(ts[idx["hip"]]["y_px"][i])
            if _visible(ts, idx["hip"], i)
            else None
        )
        other_ankle_y = (
            float(ts[idx["other_ankle"]]["y_px"][i])
            if _visible(ts, idx["other_ankle"], i)
            else None
        )
        samples.append({
            "frame_index": int(i),
            "t_ms": float(t_ms),
            "ankle_y_px": ankle_y,
            "heel_x_px": heel_x,
            "heel_y_px": heel_y,
            "foot_x_px": foot_x,
            "hip_y_px": hip_y,
            "other_ankle_y_px": other_ankle_y,
        })

    if not samples:
        raise ValueError("poor_visibility")
    duration_seconds = float(n / fps) if fps > 0 else 0.0

    # ── 4) Standing-baseline lock-in ──────────────────────────
    # Find first STANDING_HOLD_SAMPLES-long run where ankle.y +
    # hip.y are stably visible AND ankle.y barely moves (test
    # side foot grounded, patient not yet hopping).
    leg_length_px = 0.0
    for s in samples:
        if s["ankle_y_px"] is None or s["hip_y_px"] is None:
            continue
        leg_length_px = max(
            leg_length_px, abs(s["ankle_y_px"] - s["hip_y_px"])
        )
    if leg_length_px == 0.0:
        leg_length_px = 300.0   # camera-distance fallback
    standing_tol_px = _STANDING_TOLERANCE_FRAC_OF_LEG * leg_length_px

    baseline_ankle_y: Optional[float] = None
    baseline_lock_idx: Optional[int] = None
    # Best-available fallback: the window with the smallest peak
    # deviation from its own median. If no window meets the strict
    # tolerance, we'll use the most-stable one — provided its peak
    # deviation stays under the fallback multiple of tolerance.
    best_fallback: Optional[tuple[float, float, int]] = None
    fallback_ceiling_px = standing_tol_px * _STANDING_TOLERANCE_FALLBACK_MULT

    for end_i in range(_STANDING_HOLD_SAMPLES - 1, len(samples)):
        window = samples[end_i - _STANDING_HOLD_SAMPLES + 1: end_i + 1]
        ys = [s["ankle_y_px"] for s in window if s["ankle_y_px"] is not None]
        if len(ys) < _STANDING_HOLD_SAMPLES // 2:
            continue
        med_y = _stdlib_median(ys)
        max_dev = max(abs(y - med_y) for y in ys)
        if max_dev <= standing_tol_px:
            # Strict pass — first stable window wins.
            baseline_ankle_y = float(med_y)
            baseline_lock_idx = end_i
            break
        # Track the most-stable window as a graceful fallback.
        if best_fallback is None or max_dev < best_fallback[0]:
            best_fallback = (max_dev, float(med_y), end_i)

    if baseline_ankle_y is None and best_fallback is not None:
        # Accept the most-stable window if it's within the fallback
        # ceiling (3 × strict tolerance). The patient was reasonably
        # still even if not perfectly motionless.
        if best_fallback[0] <= fallback_ceiling_px:
            log.info(
                "single_leg_hop: baseline locked via fallback path — "
                "peak deviation %.1f px exceeded strict tolerance %.1f px "
                "but stayed under fallback ceiling %.1f px.",
                best_fallback[0], standing_tol_px, fallback_ceiling_px,
            )
            baseline_ankle_y = best_fallback[1]
            baseline_lock_idx = best_fallback[2]

    if baseline_ankle_y is None or baseline_lock_idx is None:
        raise ValueError(
            "no_baseline: the patient did not stand still on the test leg "
            "long enough at the start of the recording. Re-record with a "
            "~1 s static stance on the test leg before the first hop."
        )

    airborne_lift_thresh_px = _AIRBORNE_LIFT_FRAC_OF_LEG * leg_length_px
    landed_band_px = _LANDED_BAND_FRAC_OF_LEG * leg_length_px

    # ── 5) Takeoff / landing state machine ────────────────────
    # State: "grounded" → "airborne" → "grounded" repeats.
    # ankle_y is SMALLER (lower y value) when the foot rises in
    # image space (image y is top-down). So "above baseline" =
    # ankle_y < (baseline_ankle_y − threshold).
    state: str = "grounded"
    last_state_change_idx = baseline_lock_idx
    min_airborne_samples = max(2, int(round(_MIN_AIRBORNE_FRAMES_SEC * _SAMPLE_HZ)))
    min_gap_samples = max(1, int(round(_MIN_TRIAL_GAP_SEC * _SAMPLE_HZ)))

    trials: list[dict] = []
    cur_takeoff_idx: Optional[int] = None
    cur_takeoff_heel_x: Optional[float] = None
    cur_takeoff_frame: Optional[int] = None
    cur_contralateral_touch_streak = 0
    contralateral_baseline_y: Optional[float] = None
    # Lock the contralateral baseline from the same standing window.
    other_ys = [
        s["other_ankle_y_px"]
        for s in samples[
            baseline_lock_idx - _STANDING_HOLD_SAMPLES + 1: baseline_lock_idx + 1
        ]
        if s["other_ankle_y_px"] is not None
    ]
    if other_ys:
        contralateral_baseline_y = float(_stdlib_median(other_ys))

    cur_invalidation: Optional[str] = None

    def _heel_or_foot_x(s: dict) -> Optional[float]:
        # Heel preferred; foot_index fallback when heel visibility
        # was poor at takeoff/landing.
        if s["heel_x_px"] is not None:
            return s["heel_x_px"]
        return s["foot_x_px"]

    for i in range(baseline_lock_idx + 1, len(samples)):
        s = samples[i]
        ankle_y = s["ankle_y_px"]
        if ankle_y is None:
            continue

        # Single-leg validity check during airborne window.
        if (state == "airborne"
                and contralateral_baseline_y is not None
                and s["other_ankle_y_px"] is not None):
            other_drop = abs(s["other_ankle_y_px"] - contralateral_baseline_y)
            if other_drop < landed_band_px:
                cur_contralateral_touch_streak += 1
                if (cur_contralateral_touch_streak
                        > _CONTRALATERAL_TOUCH_GRACE_SAMPLES):
                    cur_invalidation = (
                        "contralateral foot touched ground during the hop"
                    )
            else:
                cur_contralateral_touch_streak = 0

        if state == "grounded":
            # Look for takeoff: ankle rises above baseline by threshold.
            rise = baseline_ankle_y - ankle_y
            if rise > airborne_lift_thresh_px:
                # Confirm we have a recent gap from the last trial.
                if i - last_state_change_idx < min_gap_samples and trials:
                    continue
                state = "airborne"
                cur_takeoff_idx = i
                # Look back one sample for the true last-grounded
                # heel.x — heel is most reliable just before liftoff.
                lookback_s = samples[i - 1] if i > 0 else s
                cur_takeoff_heel_x = _heel_or_foot_x(lookback_s) or _heel_or_foot_x(s)
                cur_takeoff_frame = lookback_s["frame_index"]
                cur_contralateral_touch_streak = 0
                cur_invalidation = None
                last_state_change_idx = i
        else:  # airborne
            # Look for landing: ankle returns to within the
            # baseline band AND stays there for at least 1 sample.
            within_band = abs(ankle_y - baseline_ankle_y) < landed_band_px
            if within_band and (i - last_state_change_idx) >= min_airborne_samples:
                # Confirmed landing.
                landing_idx_in_samples = i
                landing_heel_x = _heel_or_foot_x(s)
                landing_frame = s["frame_index"]

                if (cur_takeoff_heel_x is not None
                        and landing_heel_x is not None
                        and cur_takeoff_frame is not None):
                    hop_px = abs(landing_heel_x - cur_takeoff_heel_x)
                    hop_cm = hop_px / ppc if ppc else None

                    # Minimum-hop-for-valid gate.
                    invalidation = cur_invalidation
                    if invalidation is None:
                        if ppc is not None:
                            if hop_px < _MIN_HOP_FOR_VALID_CM * ppc:
                                invalidation = (
                                    f"hop distance {hop_cm:.1f} cm below "
                                    f"the {_MIN_HOP_FOR_VALID_CM:.0f} cm minimum"
                                )
                        else:
                            min_px_fallback = (
                                _MIN_HOP_FALLBACK_FRACTION_OF_LEG * leg_length_px
                            )
                            if hop_px < min_px_fallback:
                                invalidation = (
                                    f"hop distance {hop_px:.0f} px below "
                                    f"minimum-hop validity threshold"
                                )

                    trials.append({
                        "trial_index": len(trials) + 1,
                        "takeoff_frame_index": int(cur_takeoff_frame),
                        "landing_frame_index": int(landing_frame),
                        "takeoff_t_ms": float(
                            samples[cur_takeoff_idx]["t_ms"]
                        ) if cur_takeoff_idx is not None else 0.0,
                        "landing_t_ms": float(s["t_ms"]),
                        "hop_distance_px": float(hop_px),
                        "hop_distance_cm": (
                            float(hop_cm) if hop_cm is not None else None
                        ),
                        "valid": invalidation is None,
                        "invalidation_reason": invalidation,
                    })

                state = "grounded"
                cur_takeoff_idx = None
                cur_takeoff_heel_x = None
                cur_takeoff_frame = None
                cur_contralateral_touch_streak = 0
                cur_invalidation = None
                last_state_change_idx = landing_idx_in_samples

                if len(trials) >= _MAX_TRIALS:
                    break

    if not trials:
        raise ValueError(
            "no_hops_detected: the engine could not find any takeoff/landing "
            "events. Re-record with a clear static stance on the test leg, "
            "then a clean hop forward — both feet visible throughout."
        )

    # ── 6) Best valid trial + interpretation ──────────────────
    valid_trials = [t for t in trials if t["valid"]]
    best_valid_trial_index: Optional[int] = None
    best_valid_hop_px: Optional[float] = None
    best_valid_hop_cm: Optional[float] = None
    if valid_trials:
        best = max(valid_trials, key=lambda t: t["hop_distance_px"])
        best_valid_trial_index = int(best["trial_index"])
        best_valid_hop_px = float(best["hop_distance_px"])
        best_valid_hop_cm = (
            float(best["hop_distance_cm"])
            if best["hop_distance_cm"] is not None
            else None
        )

    # Screenshot the landing frame of the best valid trial (or the
    # best by distance if none were valid, so the operator still
    # sees what happened).
    screenshot_trial = (
        next((t for t in trials if t["trial_index"] == best_valid_trial_index), None)
        if best_valid_trial_index is not None
        else max(trials, key=lambda t: t["hop_distance_px"])
    )
    peak_screenshot = None
    if screenshot_trial is not None:
        peak_screenshot = _grab_peak_frame(
            video_path,
            int(screenshot_trial["landing_frame_index"]),
            raw,
            side,
        )

    interpretation = _build_interpretation(
        side=side,
        trials=trials,
        valid_trials=valid_trials,
        best_valid_hop_cm=best_valid_hop_cm,
        calibrated=ppc is not None,
    )

    return {
        "side_tested": side,
        "patient_height_cm": (
            float(patient_height_cm)
            if patient_height_cm is not None
            else None
        ),
        "calibration": final_calibration,
        "baseline_ankle_y_px": float(baseline_ankle_y),
        "leg_length_px": float(leg_length_px),
        "trials": trials,
        "best_valid_trial_index": best_valid_trial_index,
        "best_valid_hop_px": best_valid_hop_px,
        "best_valid_hop_cm": best_valid_hop_cm,
        "peak_screenshot_data_url": peak_screenshot,
        "duration_seconds": duration_seconds,
        "termination": "completed",
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
    }


def _build_interpretation(
    *,
    side: str,
    trials: list[dict],
    valid_trials: list[dict],
    best_valid_hop_cm: Optional[float],
    calibrated: bool,
) -> str:
    side_lbl = side.capitalize()
    if not trials:
        return (
            f"{side_lbl} — no hops were detected. Re-record with the patient "
            f"standing on the test leg, hopping forward, and landing on the "
            f"same leg."
        )
    if not valid_trials:
        reasons = sorted({
            t["invalidation_reason"] or "validity gate failed"
            for t in trials
            if not t["valid"]
        })
        reasons_str = "; ".join(reasons) if reasons else "validity gate failed"
        return (
            f"{side_lbl} — {len(trials)} hop(s) detected but none passed the "
            f"validity gate ({reasons_str}). Repeat the trial."
        )
    if best_valid_hop_cm is not None:
        return (
            f"{side_lbl} — best valid hop {best_valid_hop_cm:.1f} cm "
            f"across {len(valid_trials)} of {len(trials)} trial(s). "
            f"Limb-symmetry classification computed once the contralateral "
            f"side has been recorded."
        )
    if calibrated:
        # Should not happen — calibrated trial without cm value.
        return (
            f"{side_lbl} — {len(valid_trials)} valid hop(s) detected but "
            f"distance conversion failed. Check the calibration record."
        )
    return (
        f"{side_lbl} — {len(valid_trials)} valid hop(s) detected (relative "
        f"units only — calibration unavailable, so no centimetre value or "
        f"limb-symmetry index can be reported)."
    )


def lsi_pct(
    weaker_cm: Optional[float],
    stronger_cm: Optional[float],
) -> Optional[float]:
    """Limb Symmetry Index — (weaker / stronger) × 100. Caller is
    expected to pass the smaller of the two as `weaker_cm` and the
    larger as `stronger_cm`. Returns None when either is missing or
    `stronger_cm` is zero. Module-level helper because the test is
    per-leg at the API layer; the frontend combines L and R and
    calls this.
    """
    if weaker_cm is None or stronger_cm is None or stronger_cm <= 0:
        return None
    return float((weaker_cm / stronger_cm) * 100.0)


def lsi_classification(lsi_pct_value: Optional[float]) -> str:
    """Apply the standard ACL-clearance cutoffs:
      ≥ 90 % → cleared
      ≥ 80 % → warning
      <  80 % → deficit
      None   → incomplete (only one leg recorded).
    """
    return _classify_lsi(lsi_pct_value)
