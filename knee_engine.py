"""knee_engine.py — clinical knee flexion+extension ROM analysis on
backend MediaPipe (BlazePose Full, 33 keypoints).

Mirrors the architectural pattern of ankle_engine.py and the merged
shoulder branches in shoulder_engine.py:

  • Reuses the shared gait pipeline (extract_poses + build_time_series)
    so the same MediaPipe pose model, frame extraction, smoothing,
    interpolation, and platform-independent rotation correction back
    everyone.
  • Inherits `_pose_rotation` from raw dict (set by extract_poses)
    — handles portrait phone videos without a separate code path.
  • Returns the existing merged BiomechData DTO shape (primary +
    secondary peaks, labels, key frames) so the frontend report
    renders without any display changes.

Math convention (matches the browser computeKneeAngle output in
motionlens-web/lib/biomech/knee.ts so live + upload modes agree):

  bent_amount = 180° − interior_knee_angle

  - 0°    = knee fully straight
  - ~140° = knee fully bent

  Peak FLEXION  = MAX of bent_amount across all frames
  Peak EXTENSION = MIN of bent_amount across all frames (residual
                   flexion at the patient's straightest position;
                   lower is better)

No direction detection, no calibration baseline — knee is the
simplest of the merged tests. Just per-frame angle + min/max
tracking with visibility gating and anatomical sanity clamps.
"""
from __future__ import annotations

import base64
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


# Clinical normal ranges (AAOS). Match the browser KNEE_MOVEMENTS
# metadata in motionlens-web/lib/biomech/knee.ts so live + upload
# report identical reference ranges. Extension target is the
# RESIDUAL flexion at the patient's straightest position (0° =
# perfectly straight); range widened from strict clinical [0, 5]
# to [0, 10] to absorb the inherent ~5–10° of 2D-pose noise on
# the hip / knee / ankle keypoint placement.
KNEE_NORMAL_RANGES = {
    "flexion":   {"range": (125, 145), "target": 135.0},
    "extension": {"range": (0, 10),    "target": 5.0},
}

_MERGED_KNEEFLEXEXT_PRIMARY_TARGET:   tuple[float, float] = (125.0, 145.0)  # Flexion
_MERGED_KNEEFLEXEXT_SECONDARY_TARGET: tuple[float, float] = (0.0, 10.0)     # Extension


# Visibility floor — looser than the per-frame default because the
# smoothed time-series tolerates brief dips better than a raw frame.
_KNEE_VIS_THRESHOLD = 0.4

# Anatomical sanity bounds on the bent_amount metric.
#   Flexion ceiling: ~150° (literature max for healthy adults).
#   Hyperextension: small negative values acceptable (knee bows
#     slightly backward); guard so jitter doesn't lock implausible
#     negatives as the "best" extension reading.
_KNEE_FLEXION_ANATOMICAL_MAX = 150.0
_KNEE_HYPEREXTENSION_LIMIT   = -5.0


# ─── Math primitive ─────────────────────────────────────────────
def _interior_knee_angle(
    hip:   tuple[float, float],
    knee:  tuple[float, float],
    ankle: tuple[float, float],
) -> Optional[float]:
    """Interior knee angle in degrees (180° = straight, smaller =
    more bent). Returns None when the vectors degenerate."""
    v1 = (hip[0] - knee[0],   hip[1] - knee[1])
    v2 = (ankle[0] - knee[0], ankle[1] - knee[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_t = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


# ─── Asymmetric ROM classifier ──────────────────────────────────
# Same logic as shoulder's _classify_in_range. Restricted ROM (below
# range) flags impairment; modest excess above range is normal
# variation / hypermobility, not a clinical concern. Kept local to
# avoid a cross-module import on the math-only helper.
def _classify_in_range(value: float, lo: float, hi: float) -> str:
    if lo <= value <= hi:
        return "good"
    width = max(1.0, hi - lo)
    if value < lo:
        dist_frac = (lo - value) / width
        return "fair" if dist_frac <= 0.30 else "poor"
    dist_frac = (value - hi) / width
    if dist_frac <= 0.30:
        return "good"
    if dist_frac <= 1.00:
        return "fair"
    return "poor"


# ─── Pre-flight: wrong-side detection ───────────────────────────
def _wrong_side_for_knee_video(ts: dict, side: str) -> Optional[str]:
    """Pre-flight check: if the requested side's knee has
    consistently lower visibility than the other side's, the user
    almost certainly picked the wrong side. Returns a user-facing
    error string, or None when selection is plausibly correct.
    Conservative — only fires when the gap is clear."""
    l_entry = ts.get("left_knee") or {}
    r_entry = ts.get("right_knee") or {}
    l_vk = l_entry.get("vis")
    r_vk = r_entry.get("vis")
    if l_vk is None or r_vk is None:
        return None
    n = min(len(l_vk), len(r_vk))
    if n == 0:
        return None
    l_mean = float(np.mean(l_vk[:n]))
    r_mean = float(np.mean(r_vk[:n]))
    requested_mean = l_mean if side == "left" else r_mean
    other_mean = r_mean if side == "left" else l_mean
    other_side = "right" if side == "left" else "left"
    if (
        requested_mean < 0.4
        and other_mean > 0.6
        and (other_mean - requested_mean) > 0.25
    ):
        return (
            f"Requested side '{side}' has lower visibility than the other "
            f"side. Please re-record showing the correct leg clearly, or "
            f"switch to the '{other_side}' side."
        )
    return None


# ─── Key-frame helper ───────────────────────────────────────────
# Mirrors _grab_ankle_key_frame / _grab_shoulder_key_frame: seek to
# the source-video frame, apply the same pose-based rotation that
# extract_poses applied to the keypoints, draw a skeleton overlay
# emphasising the leg under test, return a JPEG data URL.
def _grab_knee_key_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    label: str,
    side: str,
) -> Optional[dict]:
    if frame_index < 0:
        return None
    # extract_poses stores the pose-rotation it applied; re-apply
    # to the screenshot frame so the JPEG and the keypoint overlay
    # stay aligned.
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

    def _draw_dot(name: str) -> Optional[tuple[int, int]]:
        frames = keypoints_normalized.get(name, [])
        if frame_index >= len(frames):
            return None
        kp = frames[frame_index]
        if kp is None:
            return None
        # Keypoints already in upright space (extract_poses rotated
        # them when needed); project directly into the rotated
        # frame's pixel dimensions.
        x_n, y_n, _vis = kp
        px = int(x_n * w)
        py = int(y_n * h)
        emphasised = name.startswith(side)
        outer = (0, 0, 220) if emphasised else (150, 150, 150)
        cv2.circle(frame, (px, py), 5, outer, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        return (px, py)

    # Knee-relevant edges: trunk reference + both legs (test-side
    # highlighted in white, contralateral in grey for context).
    edges = [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
        ("left_hip",       "left_knee"),
        ("left_knee",      "left_ankle"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
        ("left_ankle",     "left_heel"),
        ("right_ankle",    "right_heel"),
        ("left_heel",      "left_foot_index"),
        ("right_heel",     "right_foot_index"),
    ]
    dot_pos: dict[str, tuple[int, int]] = {}
    for name in LM:
        p = _draw_dot(name)
        if p:
            dot_pos[name] = p
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            on_side = a.startswith(side) and b.startswith(side)
            line_colour = (255, 255, 255) if on_side else (180, 180, 180)
            cv2.line(frame, dot_pos[a], dot_pos[b], line_colour, 2)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "label": label,
        "frame_index": int(frame_index),
        "image_data_url": f"data:image/jpeg;base64,{b64}",
    }


# ─── Main entry point ───────────────────────────────────────────
def analyze_knee(
    video_path: str,
    pose_options,
    movement: str,
    side: str,
) -> dict:
    """Run the BlazePose-Full knee pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded (and optionally repaired)
                      video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        movement:     "flexion_extension" (the merged knee test;
                      the only movement currently supported).
        side:         "left" or "right".

    Returns:
        Dict matching the merged BiomechData Pydantic schema —
        secondary_peak_* + primary/secondary labels populated so
        the dual-row knee report renders without any frontend
        display changes. Caller wraps in BiomechResponse.

    Raises:
        ValueError: input arguments invalid, requested side
                    visibility is clearly wrong, or the video
                    produced fewer than ~half a second of usable
                    frames. The endpoint maps this to a
                    "poor_visibility" / "Requested side" HTTP 400.
    """
    if movement != "flexion_extension":
        raise ValueError(f"Unsupported knee movement: {movement!r}")
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # Pre-flight: wrong-side selection. Fail fast with a clear
    # message rather than running a full analysis on the unmoving
    # leg.
    wrong_side_msg = _wrong_side_for_knee_video(ts, side)
    if wrong_side_msg:
        raise ValueError(wrong_side_msg)

    hip_key   = f"{side}_hip"
    knee_key  = f"{side}_knee"
    ankle_key = f"{side}_ankle"

    hx = ts[hip_key]["x_px"];   hy = ts[hip_key]["y_px"];   vh = ts[hip_key]["vis"]
    kx = ts[knee_key]["x_px"];  ky = ts[knee_key]["y_px"];  vk = ts[knee_key]["vis"]
    ax = ts[ankle_key]["x_px"]; ay = ts[ankle_key]["y_px"]; va = ts[ankle_key]["vis"]

    n = int(min(len(hx), len(kx), len(ax)))

    # Per-frame "bent amount" (180° − interior). Matches the
    # browser computeKneeAngle output so live + upload report the
    # same metric. None for frames with bad visibility — the
    # min/max tracker just doesn't see those frames, the running
    # peak stays at the last good value. (Critical at peak flexion
    # where the ankle can briefly leave the frame.)
    angles: list[Optional[float]] = []
    valid_frames = 0
    for i in range(n):
        if (vh[i] < _KNEE_VIS_THRESHOLD
                or vk[i] < _KNEE_VIS_THRESHOLD
                or va[i] < _KNEE_VIS_THRESHOLD):
            angles.append(None)
            continue
        hip_pt   = (float(hx[i]), float(hy[i]))
        knee_pt  = (float(kx[i]), float(ky[i]))
        ankle_pt = (float(ax[i]), float(ay[i]))
        interior = _interior_knee_angle(hip_pt, knee_pt, ankle_pt)
        if interior is None:
            angles.append(None)
            continue
        bent = 180.0 - interior
        # Anatomical sanity: drop frames where the computed bent
        # angle is outside what a real knee can do. Clamp the
        # accepted values to the anatomical ceiling so a single
        # noise spike can't lock an implausible 175° "flexion".
        if bent < _KNEE_HYPEREXTENSION_LIMIT or bent > _KNEE_FLEXION_ANATOMICAL_MAX + 15.0:
            angles.append(None)
            continue
        bent = max(_KNEE_HYPEREXTENSION_LIMIT, min(_KNEE_FLEXION_ANATOMICAL_MAX, bent))
        angles.append(bent)
        valid_frames += 1

    if valid_frames < max(3, int(fps * 0.5)):
        # Less than ~half a second of usable footage. Surfaced
        # to the operator as "knee not clearly visible" rather
        # than a generic analysis failure.
        raise ValueError("poor_visibility")

    # ── Min / max tracking ────────────────────────────────────
    # No direction detection — knee is the simplest merged test.
    # Peak flexion = max bent amount (most bent).
    # Peak extension = min bent amount (residual flexion at the
    # straightest position; 0° means perfectly straight).
    peak_flexion_mag: float = -math.inf
    peak_flexion_idx: int = -1
    peak_extension_mag: float = math.inf
    peak_extension_idx: int = -1
    for i, a in enumerate(angles):
        if a is None:
            continue
        if a > peak_flexion_mag:
            peak_flexion_mag = a
            peak_flexion_idx = i
        if a < peak_extension_mag:
            peak_extension_mag = a
            peak_extension_idx = i

    # Should be unreachable given the valid_frames check above,
    # but guard anyway.
    if peak_flexion_idx < 0 or peak_extension_idx < 0:
        raise ValueError("poor_visibility")

    # ── Build response ───────────────────────────────────────
    p_lo, p_hi = _MERGED_KNEEFLEXEXT_PRIMARY_TARGET
    s_lo, s_hi = _MERGED_KNEEFLEXEXT_SECONDARY_TARGET
    p_target = p_hi
    p_pct = (peak_flexion_mag / p_target) * 100.0 if p_target > 0 else 0.0
    p_status = _classify_in_range(peak_flexion_mag, p_lo, p_hi)

    interpretation_primary = (
        f"Flexion ({side.capitalize()}) measured {peak_flexion_mag:.1f}°, "
        f"which is {p_pct:.0f}% of the {p_lo:.0f}°–{p_hi:.0f}° normal range — "
        f"{p_status}."
    )
    s_status = _classify_in_range(peak_extension_mag, s_lo, s_hi)
    interpretation_secondary = (
        f"Extension ({side.capitalize()}) — residual flexion "
        f"{peak_extension_mag:.1f}° at the straightest position, "
        f"target {s_lo:.0f}°–{s_hi:.0f}° — {s_status}."
    )
    interpretation = f"{interpretation_primary} {interpretation_secondary}"

    # Key frames: two peaks only (Flexion + Extension). Matches the
    # merged shoulder ab/ad layout the user picked — no neutral
    # frame on knee reports.
    key_frames: list[dict] = []
    if peak_flexion_idx >= 0:
        kf = _grab_knee_key_frame(
            video_path, peak_flexion_idx, raw,
            f"Flexion ({peak_flexion_mag:.1f}°)", side,
        )
        if kf:
            key_frames.append(kf)
    if peak_extension_idx >= 0:
        kf = _grab_knee_key_frame(
            video_path, peak_extension_idx, raw,
            f"Extension ({peak_extension_mag:.1f}°)", side,
        )
        if kf:
            key_frames.append(kf)

    return {
        "body_part": "knee",
        "movement": "flexion_extension",
        "side": side,
        # `peak_angle` is the primary direction's signed value;
        # for knee, flexion magnitudes are already non-negative.
        "peak_angle": float(peak_flexion_mag),
        "peak_magnitude": float(peak_flexion_mag),
        "reference_range": [float(p_lo), float(p_hi)],
        "target": float(p_target),
        "percentage": p_pct,
        "status": p_status,
        "valid_frames": valid_frames,
        "total_frames": n,
        "fps": float(fps),
        "interpretation": interpretation,
        "key_frames": key_frames,
        "secondary_peak_angle": float(peak_extension_mag),
        "secondary_peak_magnitude": float(peak_extension_mag),
        "secondary_reference_range": [float(s_lo), float(s_hi)],
        "primary_label": "Flexion",
        "secondary_label": "Extension",
    }
