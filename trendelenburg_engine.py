"""trendelenburg_engine.py — single-leg stance Trendelenburg test on
the backend MediaPipe (BlazePose Full, 33 keypoints) pipeline.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/trendelenburg.ts so live + upload
report the same metric:

  pelvic tilt   = angle of (right_hip → left_hip) line from horizontal
                  (positive = LEFT side down)
  trunk lean    = angle of (hip_mid → shoulder_mid) line from vertical
                  (positive = lean to patient's RIGHT)
  drop          = dropForStance(pelvic_tilt, stance) — reoriented so
                  "drop on lifted side" is positive regardless of
                  which leg is the stance leg
  classification: |max_drop| < 2°  → negative
                  |max_drop| 2-5°  → compensated
                  |max_drop| > 5°  → positive

Pipeline:
  1. Reuse gait_engine.extract_poses() + build_time_series() to
     pull smoothed 33-kp landmarks per frame.
  2. Auto-detect when the patient first achieves stance on the
     requested side (lifted ankle > 0.06 × body height above the
     stance ankle). Frames before this are the "getting ready"
     warmup and are discarded — only frames inside the actual
     hold are sampled.
  3. Sample pelvic tilt + trunk lean at 10 Hz across the hold,
     matching the frontend SAMPLE_HZ exactly so the resulting
     samples array slots into the existing TrendelenburgReport
     Plotly chart without translation.
  4. Track max drop + capture a peak-drop screenshot (skeleton
     overlay) — same per-side artefact the live mode produces.
  5. Build the TrendelenburgSideResult dict with the same shape
     `motionlens-web/lib/orthopedic/trendelenburg.ts` produces.

The video file is opened twice: once by `extract_poses` for pose
extraction, a second short pass with cv2.VideoCapture to grab the
peak-drop screenshot frame.
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Optional

import cv2
import numpy as np

from gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/trendelenburg.ts) ────
_VIS_THRESHOLD = 0.3
_SAMPLE_HZ = 10
_TARGET_HOLD_SECONDS = 30.0
_STABLE_PORTION_START_SEC = 2.0
_SHORT_HOLD_THRESHOLD_SEC = 10.0
_PELVIC_SPIKE_TERMINATION_DEG = 15.0
_COMPENSATORY_TRUNK_LEAN_DEG = 7.0
_NEGATIVE_DROP_MAX_DEG = 2.0
_COMPENSATED_DROP_MAX_DEG = 5.0
_STANCE_LIFT_RATIO = 0.06
# Allow the lifted foot to wobble back down for this many seconds
# before we call the hold terminated. Matches POST_LIFT_LOSS_GRACE_SEC
# in TrendelenburgCapture.tsx.
_POST_LIFT_LOSS_GRACE_SEC = 2.0


# ─── Geometry helpers ───────────────────────────────────────────
def _angle_from_horizontal(vx: float, vy: float) -> float:
    """atan2(vy, vx) in degrees. Y-axis points down in image coords."""
    return math.degrees(math.atan2(vy, vx))


def _angle_from_vertical(vx: float, vy: float) -> float:
    """Angle of (vx, vy) from the +Y (downward) axis, in degrees."""
    return math.degrees(math.atan2(vx, vy))


def _compute_pelvic_tilt(ts: dict, i: int) -> Optional[float]:
    """Spec-convention pelvic tilt (left-side-down positive) for
    frame i. None when either hip is below visibility."""
    if ts["left_hip"]["vis"][i] < _VIS_THRESHOLD: return None
    if ts["right_hip"]["vis"][i] < _VIS_THRESHOLD: return None
    lx = float(ts["left_hip"]["x_px"][i])
    ly = float(ts["left_hip"]["y_px"][i])
    rx = float(ts["right_hip"]["x_px"][i])
    ry = float(ts["right_hip"]["y_px"][i])
    return _angle_from_horizontal(lx - rx, ly - ry)


def _compute_trunk_lean(ts: dict, i: int) -> Optional[float]:
    """Frontal trunk lean (lean to patient's right = positive)
    for frame i. None when any of the 4 reference joints is below
    visibility."""
    for k in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        if ts[k]["vis"][i] < _VIS_THRESHOLD:
            return None
    hip_mid_x = (float(ts["left_hip"]["x_px"][i])  + float(ts["right_hip"]["x_px"][i]))  / 2.0
    hip_mid_y = (float(ts["left_hip"]["y_px"][i])  + float(ts["right_hip"]["y_px"][i]))  / 2.0
    sh_mid_x  = (float(ts["left_shoulder"]["x_px"][i]) + float(ts["right_shoulder"]["x_px"][i])) / 2.0
    sh_mid_y  = (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    vx = sh_mid_x - hip_mid_x
    vy = sh_mid_y - hip_mid_y
    magnitude = abs(_angle_from_vertical(vx, vy))
    # vx<0 means shoulder-mid is left of hip-mid → patient leaning
    # to their RIGHT (image-mirrored frontal view) → spec positive.
    return (1.0 if vx < 0 else -1.0) * magnitude


def _drop_for_stance(pelvic: float, side: str) -> float:
    """Re-orient pelvic tilt so 'drop on lifted side' = positive."""
    return pelvic if side == "right" else -pelvic


def _lean_toward_stance(lean: float, side: str) -> float:
    return lean if side == "right" else -lean


def _classify_max_drop(max_drop_deg: float) -> str:
    v = abs(max_drop_deg)
    if v < _NEGATIVE_DROP_MAX_DEG:    return "negative"
    if v <= _COMPENSATED_DROP_MAX_DEG: return "compensated"
    return "positive"


def _detect_stance_side(ts: dict, i: int) -> Optional[str]:
    """Returns 'left', 'right', or None for frame i.
    Stance = ankle with larger image-y (still on ground); lifted =
    ankle with smaller image-y. Only fires when the lift exceeds
    STANCE_LIFT_RATIO × body height, matching the live mode."""
    for k in ("left_ankle", "right_ankle", "left_shoulder", "right_shoulder"):
        if ts[k]["vis"][i] < _VIS_THRESHOLD:
            return None
    la_y = float(ts["left_ankle"]["y_px"][i])
    ra_y = float(ts["right_ankle"]["y_px"][i])
    sh_mid_y = (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    an_mid_y = (la_y + ra_y) / 2.0
    body_h = abs(an_mid_y - sh_mid_y)
    if body_h < 1e-6:
        return None
    lift_px = abs(la_y - ra_y)
    if lift_px < body_h * _STANCE_LIFT_RATIO:
        return None
    return "right" if la_y < ra_y else "left"


# ─── Peak-drop screenshot helper ────────────────────────────────
def _grab_peak_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    side: str,
) -> Optional[str]:
    """Grab the peak-drop frame with skeleton overlay, return as
    `data:image/jpeg;base64,...` — same shape live mode produces
    via window.__trendelenburgCapture. Test-side leg + trunk are
    highlighted; contralateral leg is dimmed for context."""
    if frame_index < 0:
        return None
    pose_rot = int(keypoints_normalized.get("_pose_rotation") or 0)

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
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

    # Trunk + both legs. Test-side highlighted; the other leg in grey
    # for context (matches the live skeleton overlay convention).
    edges = [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
        ("left_hip",       "left_knee"),
        ("left_knee",      "left_ankle"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
    ]
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            on_side = a.startswith(side) and b.startswith(side)
            line_color = (255, 255, 255) if on_side else (180, 180, 180)
            cv2.line(frame, dot_pos[a], dot_pos[b], line_color, 2)
    for name, (px, py) in dot_pos.items():
        emphasised = name.startswith(side)
        outer = (0, 0, 220) if emphasised else (150, 150, 150)
        cv2.circle(frame, (px, py), 5, outer, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ─── Main entry point ──────────────────────────────────────────
def analyze_trendelenburg(
    video_path: str,
    pose_options,
    side: str,
) -> dict:
    """Run the Trendelenburg pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        side:         'left' or 'right' — the STANCE leg (the leg
                      the patient is standing on).

    Returns:
        A dict matching the frontend TrendelenburgSideResult shape
        (see motionlens-web/lib/orthopedic/trendelenburg.ts) so the
        existing TrendelenburgReport renders without translation.

    Raises:
        ValueError: 'poor_visibility' for clips where the patient
                    is not consistently visible, or a user-facing
                    message when stance on the requested side is
                    never detected. api.analyze_trendelenburg_endpoint
                    maps these to HTTP 400.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    # ── Detect the hold window ────────────────────────────────
    # Hold start: first frame where stance is on the requested side.
    # Hold end:   either the spec target (30 s) is reached OR the
    #             lifted foot returns to the ground for longer than
    #             _POST_LIFT_LOSS_GRACE_SEC OR the clip runs out.
    grace_frames = max(1, int(fps * _POST_LIFT_LOSS_GRACE_SEC))
    target_frames = int(fps * _TARGET_HOLD_SECONDS)

    hold_start_idx = -1
    hold_end_idx = -1
    lost_count = 0
    foot_touched = False

    for i in range(n):
        stance = _detect_stance_side(ts, i)
        if stance == side:
            if hold_start_idx < 0:
                hold_start_idx = i
            lost_count = 0
            hold_end_idx = i
        else:
            if hold_start_idx >= 0:
                lost_count += 1
                if lost_count > grace_frames:
                    foot_touched = True
                    break
        # Hard cap at TARGET_HOLD_SECONDS after stance starts —
        # matches the live mode's voluntary 30 s completion.
        if hold_start_idx >= 0 and (i - hold_start_idx) >= target_frames:
            break

    if hold_start_idx < 0:
        raise ValueError(
            f"Single-leg stance on the {side} leg not detected. "
            f"Please record the patient standing on the {side} leg "
            f"with the opposite leg clearly lifted."
        )
    if hold_end_idx < hold_start_idx:
        hold_end_idx = hold_start_idx

    # ── Sample at 10 Hz across the hold ───────────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(hold_start_idx, hold_end_idx + 1, step))

    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    drops: list[float] = []
    stable_drops: list[float] = []
    max_drop = 0.0
    peak_frame_idx = hold_start_idx
    max_lean = 0.0
    spike_terminated = False

    for i in sampled_frames:
        t_ms = ((i - hold_start_idx) / fps) * 1000.0
        pelvic = _compute_pelvic_tilt(ts, i)
        lean = _compute_trunk_lean(ts, i)

        samples.append({
            "t_ms": float(t_ms),
            "pelvic_tilt_deg": float(pelvic) if pelvic is not None else None,
            "trunk_lean_deg":  float(lean)   if lean   is not None else None,
        })

        # Per-sample raw-keypoint export (PDF Section 2(a) parity).
        # Pixel coords matching the live mode (raw `kp.x`, `kp.y`).
        kp_frame: list[dict] = []
        for idx in range(33):
            kp_frame.append({"x": 0.0, "y": 0.0, "score": 0.0})
        for name, idx in LM.items():
            frames = raw.get(name, [])
            if i < len(frames) and frames[i] is not None:
                x_n, y_n, vis = frames[i]
                kp_frame[idx] = {
                    "x": float(x_n) * float(raw.get("_frame_w", 1) or 1),
                    "y": float(y_n) * float(raw.get("_frame_h", 1) or 1),
                    "score": float(vis),
                }
        keypoints_export.append(kp_frame)

        if pelvic is not None:
            # Same early-termination guard live mode uses.
            if abs(pelvic) > _PELVIC_SPIKE_TERMINATION_DEG:
                spike_terminated = True
                hold_end_idx = i
                break
            drop = _drop_for_stance(pelvic, side)
            drops.append(drop)
            if (t_ms / 1000.0) >= _STABLE_PORTION_START_SEC:
                stable_drops.append(drop)
            if drop > max_drop:
                max_drop = drop
                peak_frame_idx = i

        if lean is not None:
            lean_compensatory = _lean_toward_stance(lean, side)
            if lean_compensatory > max_lean:
                max_lean = lean_compensatory

    # Reject clips where no frame produced a valid pelvic reading.
    if not drops and not samples:
        raise ValueError("poor_visibility")

    hold_seconds = max(0.0, (hold_end_idx - hold_start_idx) / fps)
    max_drop = max(0.0, max_drop)
    mean_drop = (sum(stable_drops) / len(stable_drops)) if stable_drops else 0.0

    # ── Termination classification ────────────────────────────
    if spike_terminated:
        termination = "spike"
    elif foot_touched:
        termination = "foot_touch"
    else:
        # Either the 30s target was reached OR the clip ran out
        # while the patient was still in stance — both are treated
        # as a clean completion by the live mode.
        termination = "completed"

    # ── Build the result dict ─────────────────────────────────
    classification = _classify_max_drop(max_drop)
    peak_url = _grab_peak_frame(video_path, peak_frame_idx, raw, side)

    return {
        "side_tested": side,
        "hold_seconds": float(hold_seconds),
        "max_drop_deg": float(max_drop),
        "mean_drop_deg": float(mean_drop),
        "max_compensatory_lean_deg": float(max_lean),
        "classification": classification,
        "short_hold": bool(hold_seconds < _SHORT_HOLD_THRESHOLD_SEC),
        "trendelenburg_gait_pattern": bool(max_lean > _COMPENSATORY_TRUNK_LEAN_DEG),
        "termination": termination,
        "samples": samples,
        "keypoints": keypoints_export,
        "peak_screenshot_data_url": peak_url,
    }
