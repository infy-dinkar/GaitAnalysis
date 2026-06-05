"""functional_reach_engine.py — backend C6 Functional Reach pipeline
on the MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/functionalReach.ts so the same clip
produces the same numbers in both modes.

Pipeline:
  1. Calibration is provider-agnostic. The caller passes a
     CalibrationResult-shaped dict (currently always None — Step 1
     removed the A4 detector; Step 2 will pass a height-based
     calibration through from the frontend). When None, the engine
     runs in RELATIVE-UNITS-ONLY mode (no fall-risk classification).
  2. Reuse gait_engine.extract_poses() + build_time_series() to get
     smoothed 33-kp landmarks per frame.
  3. Per-frame: wrist_x, ankle_x, heel_y, foot_index_y, shoulder_y,
     trunk angle, arm-raised flag.
  4. Find the first BASELINE_HOLD_SAMPLES-long run where the arm is
     stably at shoulder height; median wrist_x over that window is
     the baseline (point A).
  5. Absolute displacement trace from baseline; scipy-style peak
     detection with prominence + min-distance.
  6. Top NUM_TRIALS peaks → per-trial windows → heel-rise / step
     gates → valid / invalid classification.
  7. Peak-frame screenshot for the best valid trial.
  8. Return a dict matching the frontend FunctionalReachResult shape
     (so FunctionalReachReport renders without translation).
"""
from __future__ import annotations

import base64
import logging
import math
from statistics import median as _stdlib_median
from typing import Any, Optional

import cv2
import numpy as np

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


# ─── Spec constants (mirror lib/orthopedic/functionalReach.ts) ──
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 15.0
_BASELINE_HOLD_SEC = 1.0
_BASELINE_HOLD_SAMPLES = int(round(_BASELINE_HOLD_SEC * _SAMPLE_HZ))
_SHOULDER_HEIGHT_TOLERANCE_PX = 40.0
_NUM_TRIALS = 3

_HEEL_RISE_THRESHOLD_CM = 3.5  # was 2.0 — true heel-off is >=3cm; old threshold sat in jitter band
_STEP_THRESHOLD_CM = 8.0       # was 5.0 — natural body translation during reach moves ankle 3-5cm
_MIN_REACH_FOR_VALID_CM = 3.0

_LOW_FALL_RISK_MIN_CM = 25.0
_MODERATE_FALL_RISK_MIN_CM = 15.0
_VERY_HIGH_FALL_RISK_MAX_CM = 10.0

# Fall-back validity thresholds — scaled against LEG length (hip y to
# ankle y in the baseline standing window), not torso height. Foot
# landmarks live in the leg's anatomical region; scaling against leg
# length keeps the thresholds proportional both to body anatomy AND
# to camera distance.
_HEEL_RISE_FALLBACK_FRACTION_OF_LEG = 0.04
_STEP_FALLBACK_FRACTION_OF_LEG = 0.10
_MIN_REACH_FALLBACK_FRACTION_OF_LEG = 0.06

# Validity event must persist for this many consecutive smoothed
# samples (~333 ms at SAMPLE_HZ=15) to count. Single-frame excursions
# can no longer void a trial.
_SUSTAINED_VIOLATION_FRAMES = 5

_PEAK_MIN_DISTANCE_SAMPLES = int(round(1.5 * _SAMPLE_HZ))
_PEAK_MIN_PROMINENCE_FRACTION = 0.25

# Rolling-median window applied to heel y + ankle x before the per-
# trial validity scan. Without smoothing a single MediaPipe glitch
# frame on the heel can fabricate a 10-20 cm "heel rise" that voids
# the whole trial. ~470 ms strips outliers but keeps a real heel-rise
# of a few hundred ms.
_HEEL_ANKLE_SMOOTH_WINDOW = 7

# How far around each peak the validity scan looks. ±2 s covers the
# reach + brief hold + return; "just standing" frames outside that
# window can't void the trial.
_TRIAL_VALIDITY_HALF_WINDOW_SAMPLES = int(round(2.0 * _SAMPLE_HZ))

_SIDE_INDICES = {
    "left":  {
        "wrist": "left_wrist",
        "shoulder": "left_shoulder",
        "hip": "left_hip",
        "ankle": "left_ankle",
        "heel": "left_heel",
        "foot": "left_foot_index",
    },
    "right": {
        "wrist": "right_wrist",
        "shoulder": "right_shoulder",
        "hip": "right_hip",
        "ankle": "right_ankle",
        "heel": "right_heel",
        "foot": "right_foot_index",
    },
}


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _vector_angle_deg_from_vertical(vx: float, vy: float) -> Optional[float]:
    """Angle of vector (vx, vy) from image-up (0, -1). Returns 0..180."""
    length = math.hypot(vx, vy)
    if length < 1e-6:
        return None
    cos = (-vy) / length
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _all_visible(ts: dict, keys: tuple[str, ...], i: int) -> bool:
    return all(_visible(ts, k, i) for k in keys)


def _rolling_median_filter(
    values: list[Optional[float]],
    window: int,
) -> list[Optional[float]]:
    """Centered rolling-median filter. Single-frame MediaPipe glitches
    are dropped; sustained motion across ≥ ⌈window/2⌉ samples is
    preserved. Nulls inside the window are simply omitted from that
    window's median rather than poisoning it.
    """
    n = len(values)
    out: list[Optional[float]] = [None] * n
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        buf = [v for v in values[lo:hi + 1] if v is not None]
        if not buf:
            continue
        buf.sort()
        out[i] = buf[(len(buf) - 1) // 2]
    return out


def _classify_fall_risk(reach_cm: float) -> str:
    if reach_cm < _VERY_HIGH_FALL_RISK_MAX_CM:
        return "very_high"
    if reach_cm < _MODERATE_FALL_RISK_MIN_CM:
        return "high"
    if reach_cm < _LOW_FALL_RISK_MIN_CM:
        return "moderate"
    return "low"


# ─── Peak detection ─────────────────────────────────────────────
def _find_peaks(
    values: list[Optional[float]],
    min_distance: int,
    min_prominence_abs: float,
) -> list[tuple[int, float]]:
    """Return (index, value) for local maxima passing prominence + min-distance gates."""
    n = len(values)
    candidates: list[tuple[int, float]] = []
    for i in range(1, n - 1):
        v = values[i]
        if v is None:
            continue
        prev = values[i - 1]
        nxt = values[i + 1]
        if prev is None or nxt is None:
            continue
        if not (v >= prev and v >= nxt):
            continue
        if v == prev and v == nxt:
            continue
        look = max(min_distance, 30)
        left_min = v
        for j in range(i - 1, max(-1, i - look) - 1, -1):
            x = values[j]
            if x is None:
                continue
            if x < left_min:
                left_min = x
        right_min = v
        for j in range(i + 1, min(n, i + look + 1)):
            x = values[j]
            if x is None:
                continue
            if x < right_min:
                right_min = x
        prom = v - max(left_min, right_min)
        if prom < min_prominence_abs:
            continue
        candidates.append((i, v))

    candidates.sort(key=lambda p: p[1], reverse=True)
    kept: list[tuple[int, float]] = []
    for cand in candidates:
        if any(abs(cand[0] - k[0]) < min_distance for k in kept):
            continue
        kept.append(cand)
    return kept


# ─── Peak-frame screenshot ──────────────────────────────────────
def _grab_peak_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    side: str,
) -> Optional[str]:
    """Extract the annotated peak-reach frame from the uploaded clip.

    Important: cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, X) is NOT
    reliable across codecs — for many phone-recorded MP4s it seeks
    to the nearest keyframe BEFORE X, which can land far earlier
    than requested (sometimes frame 0). That manifested as the
    "Peak frame" screenshot showing the patient's baseline standing
    pose even though the reach value was computed correctly from
    the true peak. Fix: try the direct seek first, then if the
    decoder reports we landed before the target, walk forward with
    cheap grab() calls until we hit the target frame.
    """
    if frame_index < 0:
        return None
    pose_rot = int(keypoints_normalized.get("_pose_rotation") or 0)

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Walk forward via grab() (no full decode = cheap) when the
        # backend landed earlier than target. Cap the loop at the
        # requested distance + a small safety margin so a broken
        # video file can't spin forever.
        max_advance = max(0, frame_index - landed) + 5
        steps = 0
        while landed < frame_index and steps < max_advance:
            if not cap.grab():
                break
            landed += 1
            steps += 1
        ret, frame = cap.read()
        actual_after_read = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    finally:
        cap.release()
    log.info(
        "functional_reach: peak screenshot requested frame=%d, decoder reports "
        "after-read pos=%d (landed before walk=%d, advanced %d steps)",
        frame_index, actual_after_read, landed - steps, steps,
    )
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
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_heel"),
        ("right_ankle", "right_heel"),
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


# ─── Main entry point ──────────────────────────────────────────
def analyze_functional_reach(
    video_path: str,
    pose_options,
    side: str,
    calibration: Optional[dict[str, Any]] = None,
    patient_height_cm: Optional[float] = None,
) -> dict:
    """Run the Functional Reach pipeline on an uploaded clip.

    Args:
        video_path:    path to the uploaded clip on disk
        pose_options:  PoseLandmarkerOptions built by
                       api._build_gait_pose_options()
        side:          'left' or 'right' — the test-side arm
        calibration:   optional pre-detected CalibrationResult dict
                       passed in from the frontend. When None the
                       engine runs in RELATIVE-UNITS-ONLY mode. Step 1
                       removed the A4 auto-detector; Step 2 will wire
                       in a height-based calibration provider.

    Returns:
        Dict matching the frontend FunctionalReachResult shape, plus
        diagnostic extras (fps, total_frames, valid_frames,
        interpretation). The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' / 'no_baseline' / 'no_reach'
                    with a user-facing message. The API layer maps
                    these to HTTP 400 via the response envelope.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    # ── 1) Calibration (caller-supplied pass-through) ──────
    # Provider-agnostic — accept whatever shape the API layer
    # forwarded. We only require `pixels_per_cm` to be present and
    # positive to consider the recording calibrated.
    final_calibration = calibration
    ppc: Optional[float] = None
    if final_calibration is not None:
        raw_ppc = final_calibration.get("pixels_per_cm")
        if isinstance(raw_ppc, (int, float)) and raw_ppc > 0:
            ppc = float(raw_ppc)
        else:
            final_calibration = None  # malformed → treat as uncalibrated

    # ── 2) Pose extraction ────────────────────────────────
    raw, fps, _ = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
        len(ts["left_wrist"]["y"]),
        len(ts["right_wrist"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_heel"]["y"]),
        len(ts["right_heel"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    idx = _SIDE_INDICES[side]
    required_keys = (idx["wrist"], idx["shoulder"], idx["hip"], idx["ankle"], idx["heel"])
    visible_frames = sum(1 for i in range(n) if _all_visible(ts, required_keys, i))
    if visible_frames < max(3, int(n * 0.30)):
        raise ValueError("poor_visibility")

    # ── 2b) Height-based calibration (server-side fallback) ──
    # If the caller didn't pass a complete calibration but DID send
    # patient_height_cm, derive pixels_per_cm from the standing
    # window of the uploaded clip. Mirrors the live-mode
    # HeightCalibrationStep so the same patient + same clip yields
    # the same scale in both modes.
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
                    "functional_reach: height calibration: body_px=%.0f "
                    "height_cm=%.1f → %.2f px/cm",
                    body_px, patient_height_cm, ppc,
                )

    # ── 3) Sample at SAMPLE_HZ ────────────────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(0, n, step))

    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in sampled_frames:
        t_ms = (i / fps) * 1000.0 if fps > 0 else 0.0

        wrist_x = (
            float(ts[idx["wrist"]]["x_px"][i])
            if _visible(ts, idx["wrist"], i)
            else None
        )
        wrist_y = (
            float(ts[idx["wrist"]]["y_px"][i])
            if _visible(ts, idx["wrist"], i)
            else None
        )
        shoulder_y = (
            float(ts[idx["shoulder"]]["y_px"][i])
            if _visible(ts, idx["shoulder"], i)
            else None
        )
        ankle_x = (
            float(ts[idx["ankle"]]["x_px"][i])
            if _visible(ts, idx["ankle"], i)
            else None
        )
        heel_y = (
            float(ts[idx["heel"]]["y_px"][i])
            if _visible(ts, idx["heel"], i)
            else None
        )
        foot_index_y = (
            float(ts[idx["foot"]]["y_px"][i])
            if _visible(ts, idx["foot"], i)
            else None
        )
        trunk_angle = None
        if _visible(ts, idx["hip"], i) and _visible(ts, idx["shoulder"], i):
            hx = float(ts[idx["hip"]]["x_px"][i])
            hy = float(ts[idx["hip"]]["y_px"][i])
            sx = float(ts[idx["shoulder"]]["x_px"][i])
            sy = float(ts[idx["shoulder"]]["y_px"][i])
            trunk_angle = _vector_angle_deg_from_vertical(sx - hx, sy - hy)

        arm_raised = (
            wrist_y is not None and shoulder_y is not None
            and abs(wrist_y - shoulder_y) <= _SHOULDER_HEIGHT_TOLERANCE_PX
        )

        samples.append({
            "t_ms": float(t_ms),
            "wrist_x_px": wrist_x,
            "wrist_y_px": wrist_y,
            "shoulder_y_px": shoulder_y,
            "ankle_x_px": ankle_x,
            "heel_y_px": heel_y,
            "foot_index_y_px": foot_index_y,
            "trunk_angle_deg": trunk_angle,
            "arm_raised": bool(arm_raised),
        })

        kp_frame: list[dict] = [{"x": 0.0, "y": 0.0, "score": 0.0} for _ in range(33)]
        for name, idx_n in LM.items():
            frames = raw.get(name, [])
            if i < len(frames) and frames[i] is not None:
                x_n, y_n, vis = frames[i]
                kp_frame[idx_n] = {
                    "x": float(x_n) * float(raw.get("_frame_w", 1) or 1),
                    "y": float(y_n) * float(raw.get("_frame_h", 1) or 1),
                    "score": float(vis),
                }
        keypoints_export.append(kp_frame)

    if not samples:
        raise ValueError("poor_visibility")

    # ── 4) Baseline lock-in ───────────────────────────────
    baseline_end_idx: Optional[int] = None
    consecutive = 0
    for i, s in enumerate(samples):
        ok = (
            s["arm_raised"]
            and s["wrist_x_px"] is not None
            and s["ankle_x_px"] is not None
            and s["heel_y_px"] is not None
        )
        if ok:
            consecutive += 1
            if consecutive >= _BASELINE_HOLD_SAMPLES:
                baseline_end_idx = i
                break
        else:
            consecutive = 0

    duration_seconds = float(n / fps) if fps > 0 else 0.0

    empty_payload = {
        "side_tested": side,
        "baseline_locked": False,
        "baseline_locked_at_index": None,
        "baseline_wrist_x_px": None,
        "baseline_ankle_x_px": None,
        "baseline_heel_y_px": None,
        "trials": [],
        "best_valid_trial_index": None,
        "best_valid_reach_px": None,
        "best_valid_reach_cm": None,
        "classification": None,
        "calibration": final_calibration,
        "duration_seconds": duration_seconds,
        "termination": "completed",
        "samples": samples,
        "keypoints": keypoints_export,
        "peak_screenshot_data_url": None,
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": (
            "Baseline could not be locked — the test-side arm was not held "
            "stably at shoulder height in the recording."
        ),
    }
    if baseline_end_idx is None:
        raise ValueError(
            "no_baseline: the patient's arm was never stably held at shoulder "
            "height for the required interval. Re-record with the arm at ~90° "
            "for ~1 s before the first reach."
        )

    baseline_start_idx = baseline_end_idx - _BASELINE_HOLD_SAMPLES + 1
    bw_xs = [
        samples[i]["wrist_x_px"]
        for i in range(baseline_start_idx, baseline_end_idx + 1)
        if samples[i]["wrist_x_px"] is not None
    ]
    ba_xs = [
        samples[i]["ankle_x_px"]
        for i in range(baseline_start_idx, baseline_end_idx + 1)
        if samples[i]["ankle_x_px"] is not None
    ]
    bh_ys = [
        samples[i]["heel_y_px"]
        for i in range(baseline_start_idx, baseline_end_idx + 1)
        if samples[i]["heel_y_px"] is not None
    ]
    # Local heel-rise signal = foot_index_y − heel_y. See the matching
    # comment in lib/orthopedic/functionalReach.ts.
    bf_lift = [
        samples[i]["foot_index_y_px"] - samples[i]["heel_y_px"]
        for i in range(baseline_start_idx, baseline_end_idx + 1)
        if samples[i]["heel_y_px"] is not None
        and samples[i]["foot_index_y_px"] is not None
    ]
    if not bw_xs:
        raise ValueError("no_baseline: baseline window had no valid wrist samples.")
    baseline_wrist_x = float(_stdlib_median(bw_xs))
    baseline_ankle_x = float(_stdlib_median(ba_xs)) if ba_xs else 0.0
    baseline_heel_y = float(_stdlib_median(bh_ys)) if bh_ys else 0.0
    baseline_foot_lift = float(_stdlib_median(bf_lift)) if bf_lift else 0.0

    # ── 5) Leg length (px) — fallback gate scaling ─────────
    # Hip y to ankle y across the baseline standing window. Foot
    # landmarks sit in this anatomical region; scaling the heel/ankle
    # validity gates against leg length keeps the thresholds in
    # proportion to camera distance AND to the relevant body span,
    # not the torso which has a different absolute scale.
    leg_length_px = 0.0
    for i in range(baseline_start_idx, baseline_end_idx + 1):
        if not _visible(ts, idx["hip"], sampled_frames[i]):
            continue
        if not _visible(ts, idx["ankle"], sampled_frames[i]):
            continue
        hy = float(ts[idx["hip"]]["y_px"][sampled_frames[i]])
        ay = float(ts[idx["ankle"]]["y_px"][sampled_frames[i]])
        leg_length_px = max(leg_length_px, abs(ay - hy))
    if leg_length_px == 0.0:
        leg_length_px = 300.0

    if ppc is not None:
        heel_thresh_px = _HEEL_RISE_THRESHOLD_CM * ppc
        step_thresh_px = _STEP_THRESHOLD_CM * ppc
        min_reach_thresh_px = _MIN_REACH_FOR_VALID_CM * ppc
    else:
        heel_thresh_px = _HEEL_RISE_FALLBACK_FRACTION_OF_LEG * leg_length_px
        step_thresh_px = _STEP_FALLBACK_FRACTION_OF_LEG * leg_length_px
        min_reach_thresh_px = _MIN_REACH_FALLBACK_FRACTION_OF_LEG * leg_length_px

    # ── 6) Displacement trace + peak detection ───────────
    trace: list[Optional[float]] = []
    for i, s in enumerate(samples):
        if i <= baseline_end_idx:
            trace.append(0.0)
            continue
        wx = s["wrist_x_px"]
        trace.append(None if wx is None else abs(wx - baseline_wrist_x))

    max_abs = max((v for v in trace if v is not None), default=0.0)
    prom_px = max(min_reach_thresh_px, _PEAK_MIN_PROMINENCE_FRACTION * max_abs)
    peaks_raw = _find_peaks(trace, _PEAK_MIN_DISTANCE_SAMPLES, prom_px)
    peaks = sorted(peaks_raw[:_NUM_TRIALS], key=lambda p: p[0])

    if not peaks:
        raise ValueError(
            "no_reach: no reach attempts detected. Re-record with the patient "
            "reaching forward as far as comfortable, three times."
        )

    # Pre-smooth the LOCAL foot-lift signal + ankle x. Comparing
    # absolute heel_y to a baseline far back in time fires on every
    # trial because MediaPipe shifts the heel landmark upward when
    # the trunk leans forward (body-model re-fit). The local signal
    # (foot_index_y − heel_y) cancels that bias because both
    # landmarks move together under lean / shake / depth change; it
    # only grows when the foot actually rotates around the ball.
    foot_lift_raw: list[Optional[float]] = [
        (s["foot_index_y_px"] - s["heel_y_px"])
        if s["heel_y_px"] is not None and s["foot_index_y_px"] is not None
        else None
        for s in samples
    ]
    ankle_x_raw = [s["ankle_x_px"] for s in samples]
    foot_lift_smoothed = _rolling_median_filter(foot_lift_raw, _HEEL_ANKLE_SMOOTH_WINDOW)
    ankle_x_smoothed = _rolling_median_filter(ankle_x_raw, _HEEL_ANKLE_SMOOTH_WINDOW)

    # ── 7) Per-trial windows + validity ──────────────────
    trials: list[dict] = []
    for i, (peak_idx, peak_val) in enumerate(peaks):
        prev_mid = (
            baseline_end_idx + 1
            if i == 0
            else (peaks[i - 1][0] + peak_idx) // 2
        )
        next_mid = (
            len(samples) - 1
            if i == len(peaks) - 1
            else (peak_idx + peaks[i + 1][0]) // 2
        )

        # Narrow scan window — ±2 s around the peak. The full
        # prev_mid..next_mid span is still exposed for the chart but
        # validity is judged on the actual reach motion only.
        scan_start = max(prev_mid, peak_idx - _TRIAL_VALIDITY_HALF_WINDOW_SAMPLES)
        scan_end = min(next_mid, peak_idx + _TRIAL_VALIDITY_HALF_WINDOW_SAMPLES)

        # Walk the smoothed signal once and track BOTH the bare max
        # (informational, for the trial-log magnitudes) AND the
        # max-sustained value (the largest excursion that stayed
        # above threshold for >= _SUSTAINED_VIOLATION_FRAMES
        # consecutive samples). Only the sustained value can flip
        # the trial to invalid — single-frame spikes can't.
        max_heel_rise = 0.0
        max_ankle_drift = 0.0
        sustained_heel_rise = 0.0
        sustained_ankle_drift = 0.0
        heel_run = 0
        heel_run_min = float("inf")
        ankle_run = 0
        ankle_run_min = float("inf")
        for j in range(scan_start, scan_end + 1):
            fl = foot_lift_smoothed[j]
            ax = ankle_x_smoothed[j]

            if fl is not None:
                rise = max(0.0, fl - baseline_foot_lift)
                if rise > max_heel_rise:
                    max_heel_rise = rise
                if rise >= heel_thresh_px:
                    heel_run += 1
                    if rise < heel_run_min:
                        heel_run_min = rise
                    if (heel_run >= _SUSTAINED_VIOLATION_FRAMES
                            and heel_run_min > sustained_heel_rise):
                        sustained_heel_rise = heel_run_min
                else:
                    heel_run = 0
                    heel_run_min = float("inf")
            else:
                heel_run = 0
                heel_run_min = float("inf")

            if ax is not None:
                d = abs(ax - baseline_ankle_x)
                if d > max_ankle_drift:
                    max_ankle_drift = d
                if d >= step_thresh_px:
                    ankle_run += 1
                    if d < ankle_run_min:
                        ankle_run_min = d
                    if (ankle_run >= _SUSTAINED_VIOLATION_FRAMES
                            and ankle_run_min > sustained_ankle_drift):
                        sustained_ankle_drift = ankle_run_min
                else:
                    ankle_run = 0
                    ankle_run_min = float("inf")
            else:
                ankle_run = 0
                ankle_run_min = float("inf")
        max_heel_drift = max_heel_rise  # back-compat field name

        reach_px = float(peak_val)
        reach_cm = reach_px / ppc if ppc is not None else None
        heel_drift_cm = max_heel_drift / ppc if ppc is not None else None
        ankle_drift_cm = max_ankle_drift / ppc if ppc is not None else None

        validity = "valid"
        invalidity_detail: Optional[str] = None
        if reach_px < min_reach_thresh_px:
            validity = "no_motion"
            invalidity_detail = "Reach below minimum threshold — likely a noisy detection."
        elif sustained_heel_rise > 0.0:
            validity = "heel_rise"
            sustained_cm = sustained_heel_rise / ppc if ppc is not None else None
            if sustained_cm is not None:
                invalidity_detail = (
                    f"Heel lifted {sustained_cm:.1f} cm for "
                    f">= {_SUSTAINED_VIOLATION_FRAMES} frames "
                    f"(threshold {_HEEL_RISE_THRESHOLD_CM:.1f} cm)."
                )
            else:
                invalidity_detail = (
                    f"Heel lifted {sustained_heel_rise:.0f} px for "
                    f">= {_SUSTAINED_VIOLATION_FRAMES} frames "
                    f"(threshold {heel_thresh_px:.0f} px)."
                )
        elif sustained_ankle_drift > 0.0:
            validity = "step"
            sustained_cm = sustained_ankle_drift / ppc if ppc is not None else None
            if sustained_cm is not None:
                invalidity_detail = (
                    f"Foot shifted {sustained_cm:.1f} cm for "
                    f">= {_SUSTAINED_VIOLATION_FRAMES} frames "
                    f"(threshold {_STEP_THRESHOLD_CM:.1f} cm)."
                )
            else:
                invalidity_detail = (
                    f"Foot shifted {sustained_ankle_drift:.0f} px for "
                    f">= {_SUSTAINED_VIOLATION_FRAMES} frames "
                    f"(threshold {step_thresh_px:.0f} px)."
                )

        peak_wrist_x = samples[peak_idx]["wrist_x_px"]
        signed_disp = (
            float(peak_wrist_x) - baseline_wrist_x
            if peak_wrist_x is not None else 0.0
        )

        trials.append({
            "trial_index": i,
            "peak_sample_index": int(peak_idx),
            "peak_t_ms": float(samples[peak_idx]["t_ms"]),
            "signed_displacement_px": float(signed_disp),
            "reach_px": float(reach_px),
            "reach_cm": float(reach_cm) if reach_cm is not None else None,
            "trunk_angle_at_peak_deg": (
                float(samples[peak_idx]["trunk_angle_deg"])
                if samples[peak_idx]["trunk_angle_deg"] is not None
                else None
            ),
            "validity": validity,
            "invalidity_detail": invalidity_detail,
            "window_start_index": int(prev_mid),
            "window_end_index": int(next_mid),
            "max_heel_drift_px": float(max_heel_drift),
            "max_heel_drift_cm": (
                float(heel_drift_cm) if heel_drift_cm is not None else None
            ),
            "max_ankle_drift_px": float(max_ankle_drift),
            "max_ankle_drift_cm": (
                float(ankle_drift_cm) if ankle_drift_cm is not None else None
            ),
        })

    # ── 8) Best valid trial + classification ─────────────
    valid_trials = [t for t in trials if t["validity"] == "valid"]
    best_idx: Optional[int] = None
    best_reach_px: Optional[float] = None
    if valid_trials:
        winner = max(valid_trials, key=lambda t: t["reach_px"])
        best_idx = int(winner["trial_index"])
        best_reach_px = float(winner["reach_px"])

    best_reach_cm = (
        best_reach_px / ppc
        if best_reach_px is not None and ppc is not None
        else None
    )
    classification = (
        _classify_fall_risk(best_reach_cm) if best_reach_cm is not None else None
    )

    # ── 9) Peak screenshot — best valid trial, or top peak ──
    # IMPORTANT: this MUST use the same trial whose peak value was
    # reported. Two index spaces are involved:
    #   peak_sample_index  → position inside `samples` / `trace`
    #   sampled_frames[k]  → corresponding ORIGINAL VIDEO FRAME index
    # The screenshot reads from the original video, so we must
    # always map through `sampled_frames`.
    if best_idx is not None:
        screenshot_trial = trials[best_idx]
    else:
        screenshot_trial = trials[0]
    peak_sample_index = int(screenshot_trial["peak_sample_index"])
    peak_video_frame_index = int(sampled_frames[peak_sample_index])
    expected_peak_time_sec = float(screenshot_trial["peak_t_ms"]) / 1000.0
    log.info(
        "functional_reach: peak screenshot from trial=%d "
        "sample_idx=%d video_frame=%d (t=%.2fs, reach=%.1f px) — best_valid=%s",
        int(screenshot_trial["trial_index"]),
        peak_sample_index,
        peak_video_frame_index,
        expected_peak_time_sec,
        float(screenshot_trial["reach_px"]),
        best_idx is not None,
    )
    peak_screenshot = _grab_peak_frame(
        video_path, peak_video_frame_index, raw, side,
    )

    # ── 10) Interpretation ────────────────────────────────
    interpretation = _build_interpretation(
        valid_trials=valid_trials,
        total_trials=len(trials),
        best_reach_cm=best_reach_cm,
        best_reach_px=best_reach_px,
        classification=classification,
        calibrated=ppc is not None,
        trials=trials,
    )

    return {
        "side_tested": side,
        "baseline_locked": True,
        "baseline_locked_at_index": int(baseline_end_idx),
        "baseline_wrist_x_px": float(baseline_wrist_x),
        "baseline_ankle_x_px": float(baseline_ankle_x),
        "baseline_heel_y_px": float(baseline_heel_y),
        "trials": trials,
        "best_valid_trial_index": best_idx,
        "best_valid_reach_px": best_reach_px,
        "best_valid_reach_cm": best_reach_cm,
        "classification": classification,
        "calibration": final_calibration,
        "duration_seconds": duration_seconds,
        "termination": "completed",
        "samples": samples,
        "keypoints": keypoints_export,
        "peak_screenshot_data_url": peak_screenshot,
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
    }


def _build_interpretation(
    *,
    valid_trials: list[dict],
    total_trials: int,
    best_reach_cm: Optional[float],
    best_reach_px: Optional[float],
    classification: Optional[str],
    calibrated: bool,
    trials: list[dict],
) -> str:
    if total_trials == 0:
        return "No reach attempts were detected within the recording window."
    if not valid_trials:
        reasons: list[str] = []
        if any(t["validity"] == "heel_rise" for t in trials):
            reasons.append("heel rise")
        if any(t["validity"] == "step" for t in trials):
            reasons.append("stepping")
        return (
            f"All {total_trials} trials were voided"
            + (f" ({' + '.join(reasons)})" if reasons else "")
            + ". Re-record asking the patient to keep both heels down and the feet planted."
        )

    if best_reach_cm is None:
        px = best_reach_px or 0.0
        return (
            f"Best valid reach was {px:.0f} px (relative units — no scale "
            f"calibration was applied). Fall-risk classification requires "
            f"calibration with an A4 sheet for absolute distance."
        )

    if classification == "low":
        return (
            f"Best valid reach: {best_reach_cm:.1f} cm — low fall risk "
            f"(≥ {_LOW_FALL_RISK_MIN_CM:.0f} cm). Normal functional reach "
            f"for community-dwelling adults."
        )
    if classification == "moderate":
        return (
            f"Best valid reach: {best_reach_cm:.1f} cm — moderate fall risk "
            f"({_MODERATE_FALL_RISK_MIN_CM:.0f}–{_LOW_FALL_RISK_MIN_CM:.0f} cm). "
            f"Consider balance training and review medications affecting balance."
        )
    if classification == "high":
        return (
            f"Best valid reach: {best_reach_cm:.1f} cm — high fall risk "
            f"({_VERY_HIGH_FALL_RISK_MAX_CM:.0f}–{_MODERATE_FALL_RISK_MIN_CM:.0f} cm). "
            f"Recommend a comprehensive fall-prevention programme."
        )
    return (
        f"Best valid reach: {best_reach_cm:.1f} cm — very high fall risk "
        f"(< {_VERY_HIGH_FALL_RISK_MAX_CM:.0f} cm). Significant balance impairment; "
        f"urgent fall-prevention assessment indicated."
    )
