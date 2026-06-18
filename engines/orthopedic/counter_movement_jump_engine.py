"""counter_movement_jump_engine.py — backend D4 Counter-Movement
Jump test (CMJ) on the MediaPipe BlazePose Full (33-keypoint) stack.

Clinical context:
  Standard vertical-jump power test. Patient stands upright, dips
  into a brief squat (the "counter-movement"), then jumps straight
  up as high as possible and lands on both feet. Primary outcome
  is jump height (cm); secondary is flight time (s) plus a
  gravity-based physics cross-check on the height
  (h = g·t² / 8 in metres, where t is total airborne time).

  Both legs together — NO L/R asymmetry index for CMJ. That makes
  the test architecturally simpler than D3 Single-Leg Hop: no
  side parameter, no LSI, single recording per session.

Pipeline (clones single_leg_hop_engine.py architecturally — same
takeoff/landing state-machine pattern, same calibration provider,
same screenshot helper; the per-frame measurement and validity
gates change because the axis is VERTICAL not horizontal and
BOTH legs must leave the ground together):
  1. Calibration is provider-agnostic. Caller supplies a
     CalibrationResult-shaped dict OR a patient_height_cm so the
     engine can derive pixels_per_cm from the standing window
     itself. With neither, the engine runs in RELATIVE-UNITS-ONLY
     mode (pixel heights only — no cm value).
  2. Reuse gait_engine.extract_poses() + build_time_series().
  3. Per-frame samples: hip-midpoint Y, left/right ankle Y,
     visibility gates.
  4. Standing-baseline lock-in — first N stable frames where the
     hip midpoint AND both ankles are visible AND stable. The
     locked baseline_hip_y is the ground reference for the jump-
     height computation; baseline_ankle_y_avg is the threshold
     reference for takeoff/landing event detection.
  5. Takeoff = BOTH ankles' Y values rise above their baseline by
     AIRBORNE_LIFT_FRAC_OF_LEG. (Single-leg lifts during the
     counter-movement squat don't count as takeoff.)
  6. Landing = BOTH ankles' Y values return to within
     LANDED_BAND_FRAC_OF_LEG of their baseline AND have been
     airborne for >= MIN_AIRBORNE_FRAMES.
  7. Per-trial jump_height_px = baseline_hip_y − min_hip_y_during_
     airborne (apex). Image y is downward, so the apex has the
     SMALLEST hip_y value; the difference (baseline − apex) is
     positive when the patient jumped up.
  8. Per-trial flight_time_sec = landing_t − takeoff_t.
  9. Per-trial validity: requires BOTH ankles airborne for the
     full window (inverted vs D3's single-leg gate). Trial below
     MIN_JUMP_HEIGHT_FOR_VALID is dropped as standing-jitter.
  10. Apex screenshot captured for the best valid trial.
  11. Returns a dict matching the frontend CMJResult shape.
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


# ─── Spec constants (mirror lib/orthopedic/counterMovementJump.ts) ─
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 30.0
_MAX_TRIALS = 3

_STANDING_HOLD_SEC = 0.3
_STANDING_HOLD_SAMPLES = int(round(_STANDING_HOLD_SEC * _SAMPLE_HZ))

# Takeoff / landing — same thresholds as D3 (proportional to leg
# length, calibration-free). BOTH ankles must exceed the airborne
# threshold simultaneously for the test to enter the airborne
# state; both must return to grounded for the landing to confirm.
_AIRBORNE_LIFT_FRAC_OF_LEG = 0.06
_LANDED_BAND_FRAC_OF_LEG = 0.03
_MIN_AIRBORNE_FRAMES_SEC = 0.10
_MIN_TRIAL_GAP_SEC = 0.5

# Validity — minimum calibrated jump height to count as a real
# CMJ. ~5 cm is well below any clinically meaningful jump but
# permissively above standing-noise.
_MIN_JUMP_HEIGHT_FOR_VALID_CM = 5.0
# Uncalibrated fallback (fraction of leg-length).
_MIN_JUMP_FALLBACK_FRACTION_OF_LEG = 0.05

# Standing-baseline tolerance — matches D3's loosened values.
_STANDING_TOLERANCE_FRAC_OF_LEG = 0.04
_STANDING_TOLERANCE_FALLBACK_MULT = 3.0

# Physics constant for the flight-time cross-check.
_GRAVITY_M_PER_S2 = 9.81


def _physics_jump_height_cm(flight_time_sec: float) -> float:
    """Projectile-motion estimate of vertical jump height from
    flight time alone:  h = g · t² / 8  (metres).
    Multiplied by 100 to return centimetres."""
    if flight_time_sec <= 0:
        return 0.0
    h_m = (_GRAVITY_M_PER_S2 * flight_time_sec * flight_time_sec) / 8.0
    return float(h_m * 100.0)


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _hip_mid_y(ts: dict, i: int) -> Optional[float]:
    """Hip-midpoint Y in pixel space — averages left + right hips
    when both visible. Returns None when either is below the
    visibility threshold; we'd rather skip the frame than carry a
    noisy single-hip reading."""
    if not _visible(ts, "left_hip", i):
        return None
    if not _visible(ts, "right_hip", i):
        return None
    return (
        float(ts["left_hip"]["y_px"][i])
        + float(ts["right_hip"]["y_px"][i])
    ) / 2.0


# ─── Apex-frame screenshot (clones D3's helper) ─────────────────
def _grab_peak_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
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
            cv2.line(frame, dot_pos[a], dot_pos[b], (255, 255, 255), 2)
    for _name, (px_, py_) in dot_pos.items():
        cv2.circle(frame, (px_, py_), 5, (0, 0, 220), -1)
        cv2.circle(frame, (px_, py_), 7, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ─── Main entry point ───────────────────────────────────────────
def analyze_counter_movement_jump(
    video_path: str,
    pose_options,
    calibration: Optional[dict[str, Any]] = None,
    patient_height_cm: Optional[float] = None,
) -> dict:
    """Run the D4 Counter-Movement Jump pipeline on an uploaded
    clip. Both legs together — no side parameter.

    Args:
        video_path:        path to the uploaded clip
        pose_options:      PoseLandmarkerOptions built by
                           api._build_gait_pose_options()
        calibration:       optional pre-detected CalibrationResult
        patient_height_cm: optional standing height for the
                           server-side calibration fallback

    Returns:
        Dict matching the frontend CMJResult shape — per-trial
        entries plus the best valid jump + interpretation.

    Raises:
        ValueError: 'poor_visibility' / 'no_baseline' /
                    'no_jumps_detected' with a user-facing message.
    """
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
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    required_keys = ("left_hip", "right_hip", "left_ankle", "right_ankle")
    visible_frames = sum(
        1
        for i in range(n)
        if all(_visible(ts, k, i) for k in required_keys)
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
                    "cmj: height calibration: body_px=%.0f "
                    "height_cm=%.1f → %.2f px/cm",
                    body_px, patient_height_cm, ppc,
                )

    # ── 3) Sample at SAMPLE_HZ ────────────────────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(0, n, step))

    samples: list[dict] = []
    for i in sampled_frames:
        t_ms = (i / fps) * 1000.0 if fps > 0 else 0.0
        hip_y = _hip_mid_y(ts, i)
        l_ankle_y = (
            float(ts["left_ankle"]["y_px"][i])
            if _visible(ts, "left_ankle", i)
            else None
        )
        r_ankle_y = (
            float(ts["right_ankle"]["y_px"][i])
            if _visible(ts, "right_ankle", i)
            else None
        )
        samples.append({
            "frame_index": int(i),
            "t_ms": float(t_ms),
            "hip_y_px": hip_y,
            "left_ankle_y_px": l_ankle_y,
            "right_ankle_y_px": r_ankle_y,
        })

    if not samples:
        raise ValueError("poor_visibility")
    duration_seconds = float(n / fps) if fps > 0 else 0.0

    # ── 4) Standing-baseline lock-in ──────────────────────────
    # Leg length for proportional thresholds.
    leg_length_px = 0.0
    for s in samples:
        if s["hip_y_px"] is None:
            continue
        ankle_y = (
            s["left_ankle_y_px"]
            if s["left_ankle_y_px"] is not None
            else s["right_ankle_y_px"]
        )
        if ankle_y is None:
            continue
        leg_length_px = max(leg_length_px, abs(ankle_y - s["hip_y_px"]))
    if leg_length_px == 0.0:
        leg_length_px = 300.0
    standing_tol_px = _STANDING_TOLERANCE_FRAC_OF_LEG * leg_length_px
    fallback_ceiling_px = standing_tol_px * _STANDING_TOLERANCE_FALLBACK_MULT

    baseline_hip_y: Optional[float] = None
    baseline_left_ankle_y: Optional[float] = None
    baseline_right_ankle_y: Optional[float] = None
    baseline_lock_idx: Optional[int] = None
    best_fallback: Optional[tuple[float, float, float, float, int]] = None

    for end_i in range(_STANDING_HOLD_SAMPLES - 1, len(samples)):
        window = samples[end_i - _STANDING_HOLD_SAMPLES + 1: end_i + 1]
        hys = [s["hip_y_px"] for s in window if s["hip_y_px"] is not None]
        lays = [
            s["left_ankle_y_px"]
            for s in window
            if s["left_ankle_y_px"] is not None
        ]
        rays = [
            s["right_ankle_y_px"]
            for s in window
            if s["right_ankle_y_px"] is not None
        ]
        if (
            len(hys) < _STANDING_HOLD_SAMPLES // 2
            or len(lays) < _STANDING_HOLD_SAMPLES // 2
            or len(rays) < _STANDING_HOLD_SAMPLES // 2
        ):
            continue
        med_hy = _stdlib_median(hys)
        med_lay = _stdlib_median(lays)
        med_ray = _stdlib_median(rays)
        max_dev = max(
            max(abs(y - med_hy) for y in hys),
            max(abs(y - med_lay) for y in lays),
            max(abs(y - med_ray) for y in rays),
        )
        if max_dev <= standing_tol_px:
            baseline_hip_y = float(med_hy)
            baseline_left_ankle_y = float(med_lay)
            baseline_right_ankle_y = float(med_ray)
            baseline_lock_idx = end_i
            break
        if best_fallback is None or max_dev < best_fallback[0]:
            best_fallback = (
                max_dev,
                float(med_hy),
                float(med_lay),
                float(med_ray),
                end_i,
            )

    if baseline_hip_y is None and best_fallback is not None:
        if best_fallback[0] <= fallback_ceiling_px:
            log.info(
                "cmj: baseline locked via fallback path — peak deviation "
                "%.1f px exceeded strict tolerance %.1f px but stayed "
                "under fallback ceiling %.1f px.",
                best_fallback[0], standing_tol_px, fallback_ceiling_px,
            )
            baseline_hip_y = best_fallback[1]
            baseline_left_ankle_y = best_fallback[2]
            baseline_right_ankle_y = best_fallback[3]
            baseline_lock_idx = best_fallback[4]

    if (
        baseline_hip_y is None
        or baseline_left_ankle_y is None
        or baseline_right_ankle_y is None
        or baseline_lock_idx is None
    ):
        raise ValueError(
            "no_baseline: the patient did not stand still before jumping. "
            "Re-record with a ~1 s static standing pose before the first "
            "counter-movement jump."
        )

    airborne_lift_thresh_px = _AIRBORNE_LIFT_FRAC_OF_LEG * leg_length_px
    landed_band_px = _LANDED_BAND_FRAC_OF_LEG * leg_length_px

    # ── 5) Takeoff / landing state machine ────────────────────
    # State: "grounded" → "airborne" → "grounded" repeats. BOTH
    # ankles must be airborne to enter airborne state; BOTH must
    # be back inside the landed band to confirm landing. Image y
    # decreases upward, so "above baseline" = ankle_y < baseline -
    # threshold.
    state: str = "grounded"
    min_airborne_samples = max(
        2, int(round(_MIN_AIRBORNE_FRAMES_SEC * _SAMPLE_HZ))
    )
    min_gap_samples = max(1, int(round(_MIN_TRIAL_GAP_SEC * _SAMPLE_HZ)))
    last_state_change_idx = baseline_lock_idx

    trials: list[dict] = []
    cur_takeoff_idx: Optional[int] = None
    cur_takeoff_frame: Optional[int] = None
    cur_apex_hip_y: Optional[float] = None
    cur_apex_frame: Optional[int] = None
    cur_invalidation: Optional[str] = None

    for i in range(baseline_lock_idx + 1, len(samples)):
        s = samples[i]
        lay = s["left_ankle_y_px"]
        ray = s["right_ankle_y_px"]
        if lay is None or ray is None:
            continue

        left_lift = baseline_left_ankle_y - lay
        right_lift = baseline_right_ankle_y - ray
        both_airborne = (
            left_lift > airborne_lift_thresh_px
            and right_lift > airborne_lift_thresh_px
        )
        left_landed = abs(lay - baseline_left_ankle_y) < landed_band_px
        right_landed = abs(ray - baseline_right_ankle_y) < landed_band_px
        both_landed = left_landed and right_landed

        if state == "grounded":
            if both_airborne:
                if i - last_state_change_idx < min_gap_samples and trials:
                    continue
                state = "airborne"
                cur_takeoff_idx = i
                # Last-grounded frame is i-1.
                last_grounded = samples[i - 1] if i > 0 else s
                cur_takeoff_frame = int(last_grounded["frame_index"])
                cur_apex_hip_y = (
                    last_grounded["hip_y_px"] if last_grounded["hip_y_px"] is not None else baseline_hip_y
                )
                cur_apex_frame = cur_takeoff_frame
                cur_invalidation = None
                last_state_change_idx = i
        else:  # airborne
            # Track apex (smallest hip_y = highest in image).
            if s["hip_y_px"] is not None:
                if cur_apex_hip_y is None or s["hip_y_px"] < cur_apex_hip_y:
                    cur_apex_hip_y = float(s["hip_y_px"])
                    cur_apex_frame = int(s["frame_index"])

            if (
                both_landed
                and (i - last_state_change_idx) >= min_airborne_samples
            ):
                # Confirmed landing.
                takeoff_t_ms = (
                    float(samples[cur_takeoff_idx]["t_ms"])
                    if cur_takeoff_idx is not None
                    else 0.0
                )
                landing_t_ms = float(s["t_ms"])
                landing_frame = int(s["frame_index"])
                flight_time_sec = max(
                    0.0, (landing_t_ms - takeoff_t_ms) / 1000.0,
                )

                jump_height_px = 0.0
                if cur_apex_hip_y is not None and baseline_hip_y is not None:
                    jump_height_px = max(
                        0.0, baseline_hip_y - float(cur_apex_hip_y),
                    )

                jump_height_cm = (
                    jump_height_px / ppc if ppc is not None else None
                )
                physics_height_cm = _physics_jump_height_cm(flight_time_sec)

                invalidation = cur_invalidation
                if invalidation is None:
                    if ppc is not None:
                        if jump_height_px < _MIN_JUMP_HEIGHT_FOR_VALID_CM * ppc:
                            invalidation = (
                                f"jump height {jump_height_cm:.1f} cm below "
                                f"the {_MIN_JUMP_HEIGHT_FOR_VALID_CM:.0f} cm "
                                f"minimum"
                            )
                    else:
                        min_px_fallback = (
                            _MIN_JUMP_FALLBACK_FRACTION_OF_LEG * leg_length_px
                        )
                        if jump_height_px < min_px_fallback:
                            invalidation = (
                                f"jump height {jump_height_px:.0f} px below "
                                f"minimum-jump validity threshold"
                            )

                trials.append({
                    "trial_index": len(trials) + 1,
                    "takeoff_frame_index": int(
                        cur_takeoff_frame
                        if cur_takeoff_frame is not None
                        else samples[cur_takeoff_idx]["frame_index"]
                        if cur_takeoff_idx is not None
                        else -1
                    ),
                    "apex_frame_index": int(
                        cur_apex_frame
                        if cur_apex_frame is not None
                        else -1
                    ),
                    "landing_frame_index": landing_frame,
                    "takeoff_t_ms": takeoff_t_ms,
                    "landing_t_ms": landing_t_ms,
                    "flight_time_sec": float(flight_time_sec),
                    "jump_height_px": float(jump_height_px),
                    "jump_height_cm": (
                        float(jump_height_cm)
                        if jump_height_cm is not None
                        else None
                    ),
                    "physics_height_cm": float(physics_height_cm),
                    "valid": invalidation is None,
                    "invalidation_reason": invalidation,
                })

                state = "grounded"
                cur_takeoff_idx = None
                cur_takeoff_frame = None
                cur_apex_hip_y = None
                cur_apex_frame = None
                cur_invalidation = None
                last_state_change_idx = i

                if len(trials) >= _MAX_TRIALS:
                    break

    if not trials:
        raise ValueError(
            "no_jumps_detected: the engine could not find any takeoff/landing "
            "events. Re-record with the patient performing a clear counter-"
            "movement jump — both feet leave the ground together, both feet "
            "land together, full body visible throughout."
        )

    # ── 6) Best valid + mean ──────────────────────────────────
    valid_trials = [t for t in trials if t["valid"]]
    best_valid_trial_index: Optional[int] = None
    best_valid_jump_px: Optional[float] = None
    best_valid_jump_cm: Optional[float] = None
    best_valid_flight_sec: Optional[float] = None
    mean_valid_jump_cm: Optional[float] = None
    mean_valid_flight_sec: Optional[float] = None
    if valid_trials:
        best = max(valid_trials, key=lambda t: t["jump_height_px"])
        best_valid_trial_index = int(best["trial_index"])
        best_valid_jump_px = float(best["jump_height_px"])
        best_valid_jump_cm = (
            float(best["jump_height_cm"])
            if best["jump_height_cm"] is not None
            else None
        )
        best_valid_flight_sec = float(best["flight_time_sec"])
        if all(t["jump_height_cm"] is not None for t in valid_trials):
            mean_valid_jump_cm = float(
                sum(t["jump_height_cm"] for t in valid_trials)
                / len(valid_trials),
            )
        mean_valid_flight_sec = float(
            sum(t["flight_time_sec"] for t in valid_trials)
            / len(valid_trials),
        )

    screenshot_trial = (
        next(
            (t for t in trials if t["trial_index"] == best_valid_trial_index),
            None,
        )
        if best_valid_trial_index is not None
        else max(trials, key=lambda t: t["jump_height_px"])
    )
    peak_screenshot = None
    if screenshot_trial is not None:
        peak_screenshot = _grab_peak_frame(
            video_path,
            int(screenshot_trial["apex_frame_index"]),
            raw,
        )

    interpretation = _build_interpretation(
        trials=trials,
        valid_trials=valid_trials,
        best_valid_jump_cm=best_valid_jump_cm,
        best_valid_flight_sec=best_valid_flight_sec,
        calibrated=ppc is not None,
    )

    return {
        "patient_height_cm": (
            float(patient_height_cm)
            if patient_height_cm is not None
            else None
        ),
        "calibration": final_calibration,
        "baseline_hip_y_px": float(baseline_hip_y),
        "baseline_left_ankle_y_px": float(baseline_left_ankle_y),
        "baseline_right_ankle_y_px": float(baseline_right_ankle_y),
        "leg_length_px": float(leg_length_px),
        "trials": trials,
        "best_valid_trial_index": best_valid_trial_index,
        "best_valid_jump_px": best_valid_jump_px,
        "best_valid_jump_cm": best_valid_jump_cm,
        "best_valid_flight_sec": best_valid_flight_sec,
        "mean_valid_jump_cm": mean_valid_jump_cm,
        "mean_valid_flight_sec": mean_valid_flight_sec,
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
    trials: list[dict],
    valid_trials: list[dict],
    best_valid_jump_cm: Optional[float],
    best_valid_flight_sec: Optional[float],
    calibrated: bool,
) -> str:
    if not trials:
        return (
            "No jumps detected. Re-record with the patient performing a "
            "clear counter-movement jump — both feet leave together, "
            "both feet land together."
        )
    if not valid_trials:
        reasons = sorted({
            t["invalidation_reason"] or "validity gate failed"
            for t in trials
            if not t["valid"]
        })
        reasons_str = "; ".join(reasons) if reasons else "validity gate failed"
        return (
            f"{len(trials)} jump(s) detected but none passed the validity "
            f"gate ({reasons_str}). Repeat the trial."
        )
    if best_valid_jump_cm is not None and best_valid_flight_sec is not None:
        physics_h = _physics_jump_height_cm(best_valid_flight_sec)
        return (
            f"Best jump height {best_valid_jump_cm:.1f} cm "
            f"(flight time {best_valid_flight_sec:.2f} s; physics cross-"
            f"check {physics_h:.1f} cm) across {len(valid_trials)} of "
            f"{len(trials)} valid trial(s)."
        )
    if calibrated:
        return (
            f"{len(valid_trials)} valid jump(s) detected but the cm "
            f"conversion failed. Check the calibration record."
        )
    return (
        f"{len(valid_trials)} valid jump(s) detected (relative units only — "
        f"calibration unavailable, so no centimetre height value can be "
        f"reported; flight time is still available)."
    )
