"""ake_engine.py — backend Active Knee Extension pipeline on the
MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/ake.ts so live + upload report the
same numbers.

Math:
  body baseline vector = hip-midpoint → shoulder-midpoint
                         (points toward the patient's head when supine)
  thigh vector         = test-side hip → test-side knee
  shin vector          = test-side knee → test-side ankle

  hip_flex_angle       = 180° − angle(body_baseline, thigh)
                         (0° = leg flat, 90° = thigh vertical, >90° = past vertical)
  knee_angle           = inner angle at the knee between
                         (knee→hip) and (knee→ankle)
                         (180° = perfectly straight, 90° = bent)

  thigh_held           = THIGH_HELD_MIN_DEG ≤ hip_flex_angle ≤ THIGH_HELD_MAX_DEG

Result per side = max(knee_angle) across frames where thigh_held = True
AND the test-side hip/knee/ankle + both hips/shoulders were visible.
Deficit = 180° − max_knee_angle (clamped to ≥ 0).

Classification cutoffs (extension DEFICIT, degrees):
  ≤ 10°  → "normal"
  ≤ 20°  → "mild"
  ≤ 35°  → "moderate"
  > 35°  → "severe"

Pipeline:
  1. Reuse gait_engine.extract_poses() + build_time_series() to pull
     smoothed 33-kp landmarks per frame.
  2. Visibility gate — require ≥30% of frames with hip + knee + ankle
     on the test side plus both hips/shoulders (torso anchor).
  3. Per frame: compute knee_angle, hip_flex_angle, thigh_held.
  4. Aggregate: max knee_angle among thigh_held frames.
  5. 10 Hz time-series + per-frame keypoints export for the report.
  6. Peak-frame screenshot (skeleton overlay) at the frame with the
     largest valid knee extension.

Returns a dict matching the frontend AKESideResult shape so AKEReport
renders without translation.
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Optional

import cv2

from engines.gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/ake.ts) ──────────────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TRIAL_DURATION_SEC = 15.0  # report cap; engine still uses the full clip
_THIGH_HELD_MIN_DEG = 75.0
_THIGH_HELD_MAX_DEG = 105.0
_MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG = 95.0
_NORMAL_MAX_DEFICIT_DEG = 10.0
_MILD_MAX_DEFICIT_DEG = 20.0
_MODERATE_MAX_DEFICIT_DEG = 35.0

_SIDE_INDICES = {
    "left":  {"hip": "left_hip",  "knee": "left_knee",  "ankle": "left_ankle"},
    "right": {"hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle"},
}


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _vector_angle_deg(
    ax: float, ay: float, bx: float, by: float,
) -> Optional[float]:
    ma = math.hypot(ax, ay)
    mb = math.hypot(bx, by)
    if ma < 1e-6 or mb < 1e-6:
        return None
    cos = (ax * bx + ay * by) / (ma * mb)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _all_visible(ts: dict, keys: tuple[str, ...], i: int) -> bool:
    return all(_visible(ts, k, i) for k in keys)


def _compute_hip_flex_angle(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    keys = (
        idx["hip"], idx["knee"],
        "left_hip", "right_hip",
        "left_shoulder", "right_shoulder",
    )
    if not _all_visible(ts, keys, i):
        return None
    hx = float(ts[idx["hip"]]["x_px"][i]);   hy = float(ts[idx["hip"]]["y_px"][i])
    kx = float(ts[idx["knee"]]["x_px"][i]);  ky = float(ts[idx["knee"]]["y_px"][i])
    lhx = float(ts["left_hip"]["x_px"][i]);  lhy = float(ts["left_hip"]["y_px"][i])
    rhx = float(ts["right_hip"]["x_px"][i]); rhy = float(ts["right_hip"]["y_px"][i])
    lsx = float(ts["left_shoulder"]["x_px"][i]);  lsy = float(ts["left_shoulder"]["y_px"][i])
    rsx = float(ts["right_shoulder"]["x_px"][i]); rsy = float(ts["right_shoulder"]["y_px"][i])

    hip_mid_x = (lhx + rhx) / 2.0; hip_mid_y = (lhy + rhy) / 2.0
    sh_mid_x  = (lsx + rsx) / 2.0; sh_mid_y  = (lsy + rsy) / 2.0

    torso_vx = sh_mid_x - hip_mid_x
    torso_vy = sh_mid_y - hip_mid_y
    thigh_vx = kx - hx
    thigh_vy = ky - hy
    inner = _vector_angle_deg(torso_vx, torso_vy, thigh_vx, thigh_vy)
    if inner is None:
        return None
    return 180.0 - inner


def _compute_knee_angle(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _all_visible(ts, (idx["hip"], idx["knee"], idx["ankle"]), i):
        return None
    hx = float(ts[idx["hip"]]["x_px"][i]);   hy = float(ts[idx["hip"]]["y_px"][i])
    kx = float(ts[idx["knee"]]["x_px"][i]);  ky = float(ts[idx["knee"]]["y_px"][i])
    ax = float(ts[idx["ankle"]]["x_px"][i]); ay = float(ts[idx["ankle"]]["y_px"][i])
    return _vector_angle_deg(
        hx - kx, hy - ky,
        ax - kx, ay - ky,
    )


def _is_thigh_held(hip_flex_deg: Optional[float]) -> bool:
    if hip_flex_deg is None:
        return False
    return _THIGH_HELD_MIN_DEG <= hip_flex_deg <= _THIGH_HELD_MAX_DEG


# ─── Classification ─────────────────────────────────────────────
def _classify_ake(deficit_deg: float) -> str:
    if deficit_deg <= _NORMAL_MAX_DEFICIT_DEG:   return "normal"
    if deficit_deg <= _MILD_MAX_DEFICIT_DEG:     return "mild"
    if deficit_deg <= _MODERATE_MAX_DEFICIT_DEG: return "moderate"
    return "severe"


def _build_interpretation(
    side: str,
    max_knee_deg: float,
    deficit_deg: float,
    thigh_held_fraction: float,
    classification: str,
) -> str:
    side_label = "Left AKE" if side == "left" else "Right AKE"
    knee = f"{max_knee_deg:.1f}°"
    deficit = f"{deficit_deg:.1f}°"

    if max_knee_deg < _MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG:
        return (
            f"{side_label}: no extension detected (peak knee {knee}). Re-record "
            f"with the thigh held vertical and the patient slowly straightening "
            f"the knee."
        )
    if thigh_held_fraction < 0.3:
        return (
            f"{side_label}: thigh did not stay at ~90° for most of the trial "
            f"({thigh_held_fraction * 100:.0f}% of frames passed the "
            f"{_THIGH_HELD_MIN_DEG:.0f}–{_THIGH_HELD_MAX_DEG:.0f}° hip-flexion gate). "
            f"Result reflects the best qualifying moment (max knee {knee}, "
            f"deficit {deficit}) — consider a fresh attempt with the hip held stable."
        )
    if classification == "normal":
        return (
            f"{side_label}: max knee {knee}, deficit {deficit} — normal hamstring "
            f"flexibility (deficit ≤ {_NORMAL_MAX_DEFICIT_DEG:.0f}°)."
        )
    if classification == "mild":
        return (
            f"{side_label}: max knee {knee}, deficit {deficit} — mild hamstring "
            f"tightness ({_NORMAL_MAX_DEFICIT_DEG + 1:.0f}–{_MILD_MAX_DEFICIT_DEG:.0f}°)."
        )
    if classification == "moderate":
        return (
            f"{side_label}: max knee {knee}, deficit {deficit} — moderate "
            f"hamstring tightness "
            f"({_MILD_MAX_DEFICIT_DEG + 1:.0f}–{_MODERATE_MAX_DEFICIT_DEG:.0f}°). "
            f"Stretching program indicated."
        )
    return (
        f"{side_label}: max knee {knee}, deficit {deficit} — severe hamstring "
        f"tightness (> {_MODERATE_MAX_DEFICIT_DEG:.0f}°). Targeted hamstring "
        f"lengthening recommended."
    )


# ─── Peak-frame screenshot ──────────────────────────────────────
def _grab_peak_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    side: str,
) -> Optional[str]:
    """Grab the peak-extension frame with skeleton overlay, return as
    `data:image/jpeg;base64,...`. Test-side leg + trunk highlighted;
    contralateral leg dimmed for context."""
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
def analyze_ake(
    video_path: str,
    pose_options,
    side: str,
) -> dict:
    """Run the Active Knee Extension pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        side:         'left' or 'right' — the leg being tested.

    Returns:
        Dict matching the frontend AKESideResult shape plus a few
        diagnostic extras (fps, total_frames, valid_frames,
        interpretation). The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' when the patient/leg isn't
                    clearly visible in enough frames, or a user-facing
                    message when the thigh wasn't held or no extension
                    was detected. api.analyze_ake_endpoint maps these
                    to HTTP 400.
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
        len(ts["left_knee"]["y"]),
        len(ts["right_knee"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    # ── Visibility gate ─────────────────────────────────────
    idx = _SIDE_INDICES[side]
    required_keys = (
        idx["hip"], idx["knee"], idx["ankle"],
        "left_hip", "right_hip",
        "left_shoulder", "right_shoulder",
    )
    visible_frames = sum(1 for i in range(n) if _all_visible(ts, required_keys, i))
    if visible_frames < max(3, int(n * 0.30)):
        raise ValueError("poor_visibility")

    # ── Per-frame metrics ───────────────────────────────────
    knee_series:    list[Optional[float]] = []
    hipflex_series: list[Optional[float]] = []
    held_flag:      list[bool]             = []
    for i in range(n):
        kn = _compute_knee_angle(ts, i, side)
        hf = _compute_hip_flex_angle(ts, i, side)
        hd = _is_thigh_held(hf)
        knee_series.append(kn)
        hipflex_series.append(hf)
        held_flag.append(hd)

    # ── Aggregate: max valid knee + peak frame index ────────
    max_knee = 0.0
    max_idx: Optional[int] = None
    hip_at_peak: Optional[float] = None
    held_count = 0
    for i in range(n):
        if held_flag[i]:
            held_count += 1
            kn = knee_series[i]
            if kn is not None and kn > max_knee:
                max_knee = kn
                max_idx = i
                hip_at_peak = hipflex_series[i]

    thigh_held_fraction = held_count / n if n > 0 else 0.0

    if max_knee < _MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG:
        if held_count == 0:
            raise ValueError(
                "thigh_not_held: the thigh did not stay at ~90° during the "
                "trial. Re-record with the patient holding the hip flexed at "
                "90° throughout the knee extension."
            )
        raise ValueError(
            "no_extension: no knee extension detected on the "
            f"{side} leg. Please re-record the patient slowly straightening "
            "the knee while keeping the thigh vertical."
        )

    deficit = max(0.0, 180.0 - max_knee)

    # ── 10 Hz time-series + keypoints export ────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(0, n, step))
    # Re-map max_idx into the sampled series (so the frontend's
    # `max_knee_sample_index` lines up with `samples[i]`).
    max_sample_index: Optional[int] = None
    if max_idx is not None:
        sample_pos = max_idx // step
        if sample_pos < len(sampled_frames):
            max_sample_index = sample_pos

    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in sampled_frames:
        t_ms = (i / fps) * 1000.0 if fps > 0 else 0.0
        samples.append({
            "t_ms": float(t_ms),
            "knee_angle_deg":     float(knee_series[i])    if knee_series[i]    is not None else None,
            "hip_flex_angle_deg": float(hipflex_series[i]) if hipflex_series[i] is not None else None,
            "thigh_held":         bool(held_flag[i]),
        })
        kp_frame: list[dict] = [
            {"x": 0.0, "y": 0.0, "score": 0.0} for _ in range(33)
        ]
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

    duration_seconds = float(n / fps) if fps > 0 else 0.0
    termination = "completed"

    # Peak-frame screenshot
    peak_screenshot: Optional[str] = None
    if max_idx is not None:
        peak_screenshot = _grab_peak_frame(video_path, max_idx, raw, side)

    classification = _classify_ake(deficit)
    interpretation = _build_interpretation(
        side, max_knee, deficit, thigh_held_fraction, classification,
    )

    return {
        "side_tested": side,
        "max_knee_angle_deg": float(max_knee),
        "deficit_deg": float(deficit),
        "max_knee_sample_index": max_sample_index,
        "hip_flex_angle_at_peak_deg": float(hip_at_peak) if hip_at_peak is not None else None,
        "classification": classification,
        "duration_seconds": duration_seconds,
        "termination": termination,
        "thigh_held_fraction": float(thigh_held_fraction),
        "samples": samples,
        "keypoints": keypoints_export,
        "peak_screenshot_data_url": peak_screenshot,
        # Diagnostic extras
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
    }
