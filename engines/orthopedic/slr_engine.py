"""slr_engine.py — backend Straight Leg Raise pipeline on the
MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/slr.ts so live + upload report the same
numbers.

Math:
  body baseline vector = hip-midpoint → shoulder-midpoint
  leg vector           = test-side hip → test-side ankle
  inner_angle          = angle between body baseline and leg (0..180°)
  raise_angle          = 180° − inner_angle
                         (0° = leg flat, 90° = vertical, >90° = past vertical)

  knee_angle           = inner angle at the knee between (knee→hip) and (knee→ankle)
                         (180° = perfectly straight)
  knee_straight        = knee_angle >= STRAIGHT_THRESHOLD_DEG (160°)

Result per side = max(raise_angle) across frames where the knee was
straight AND the test-side hip/knee/ankle + both hips/shoulders were
visible. Not capped at 90°.

Classification cutoffs:
  raise < 30°  → "severely_limited"
  30-70°       → "positive" (possible nerve tension)
  70-90°       → "normal"
  > 90°        → "hypermobile"

Pipeline:
  1. Reuse gait_engine.extract_poses() + build_time_series() to pull
     smoothed 33-kp landmarks per frame.
  2. Visibility gate — require ≥30% of frames with hip + knee + ankle
     on the test side plus both hips/shoulders (torso anchor).
  3. Per frame: compute raise_angle, knee_angle, knee_straight.
  4. Aggregate: max raise_angle among knee_straight frames.
  5. 10 Hz time-series + per-frame keypoints export for the report.
  6. Peak-frame screenshot (skeleton overlay) at the frame with the
     largest valid raise.

Returns a dict matching the frontend SLRSideResult shape so SLRReport
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


# ─── Spec constants (mirror lib/orthopedic/slr.ts) ──────────────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TRIAL_DURATION_SEC = 15.0  # report cap; engine still uses the full clip
_STRAIGHT_THRESHOLD_DEG = 160.0
_MIN_RAISE_FOR_VALID_TRIAL_DEG = 5.0
_SEVERELY_LIMITED_MAX_DEG = 30.0
_POSITIVE_MAX_DEG = 70.0
_NORMAL_MAX_DEG = 90.0

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


def _compute_raise_angle(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    keys = (
        idx["hip"], idx["ankle"],
        "left_hip", "right_hip",
        "left_shoulder", "right_shoulder",
    )
    if not _all_visible(ts, keys, i):
        return None
    hx = float(ts[idx["hip"]]["x_px"][i]);   hy = float(ts[idx["hip"]]["y_px"][i])
    ax = float(ts[idx["ankle"]]["x_px"][i]); ay = float(ts[idx["ankle"]]["y_px"][i])
    lhx = float(ts["left_hip"]["x_px"][i]);  lhy = float(ts["left_hip"]["y_px"][i])
    rhx = float(ts["right_hip"]["x_px"][i]); rhy = float(ts["right_hip"]["y_px"][i])
    lsx = float(ts["left_shoulder"]["x_px"][i]);  lsy = float(ts["left_shoulder"]["y_px"][i])
    rsx = float(ts["right_shoulder"]["x_px"][i]); rsy = float(ts["right_shoulder"]["y_px"][i])

    hip_mid_x = (lhx + rhx) / 2.0; hip_mid_y = (lhy + rhy) / 2.0
    sh_mid_x  = (lsx + rsx) / 2.0; sh_mid_y  = (lsy + rsy) / 2.0

    torso_vx = sh_mid_x - hip_mid_x
    torso_vy = sh_mid_y - hip_mid_y
    leg_vx = ax - hx
    leg_vy = ay - hy
    inner = _vector_angle_deg(torso_vx, torso_vy, leg_vx, leg_vy)
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


# ─── Classification ─────────────────────────────────────────────
def _classify_slr(max_raise_deg: float) -> str:
    if max_raise_deg < _SEVERELY_LIMITED_MAX_DEG: return "severely_limited"
    if max_raise_deg < _POSITIVE_MAX_DEG:          return "positive"
    if max_raise_deg <= _NORMAL_MAX_DEG:           return "normal"
    return "hypermobile"


def _build_interpretation(
    side: str,
    max_raise_deg: float,
    knee_straight_fraction: float,
    classification: str,
) -> str:
    side_label = "Left SLR" if side == "left" else "Right SLR"
    angle = f"{max_raise_deg:.1f}°"
    if max_raise_deg < _MIN_RAISE_FOR_VALID_TRIAL_DEG:
        return (
            f"{side_label}: no raise detected ({angle}). Re-record with the leg lifted "
            f"from flat to as high as the patient can comfortably reach while keeping "
            f"the knee straight."
        )
    if knee_straight_fraction < 0.3:
        return (
            f"{side_label}: knee did not stay straight for most of the trial "
            f"({knee_straight_fraction * 100:.0f}% of frames passed the "
            f"{_STRAIGHT_THRESHOLD_DEG:.0f}° straightness gate). Result reflects the "
            f"best qualifying moment ({angle}) — consider a fresh attempt with the knee "
            f"held straight."
        )
    if classification == "severely_limited":
        return (
            f"{side_label}: {angle} — severely limited (< {_SEVERELY_LIMITED_MAX_DEG:.0f}°). "
            f"Marked restriction; correlate with hamstring length, lumbar pathology, "
            f"and pain provocation."
        )
    if classification == "positive":
        return (
            f"{side_label}: {angle} — positive SLR ({_SEVERELY_LIMITED_MAX_DEG:.0f}–"
            f"{_POSITIVE_MAX_DEG:.0f}°). Range suggests possible neural tension; consider "
            f"lumbar nerve-root involvement and complementary tests (Bragard's, slump)."
        )
    if classification == "normal":
        return (
            f"{side_label}: {angle} — within normal range "
            f"({_POSITIVE_MAX_DEG:.0f}–{_NORMAL_MAX_DEG:.0f}°)."
        )
    return (
        f"{side_label}: {angle} — hypermobile range (> {_NORMAL_MAX_DEG:.0f}°). "
        f"Note overall joint hypermobility if observed elsewhere."
    )


# ─── Peak-frame screenshot ──────────────────────────────────────
def _grab_peak_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    side: str,
) -> Optional[str]:
    """Grab the peak-raise frame with skeleton overlay, return as
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
def analyze_slr(
    video_path: str,
    pose_options,
    side: str,
) -> dict:
    """Run the Straight Leg Raise pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        side:         'left' or 'right' — the leg being raised.

    Returns:
        Dict matching the frontend SLRSideResult shape plus a few
        diagnostic extras (fps, total_frames, valid_frames,
        interpretation). The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' when the patient/leg isn't
                    clearly visible in enough frames, or a user-facing
                    message when no raise is detected.
                    api.analyze_slr_endpoint maps these to HTTP 400.
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
    raise_series:  list[Optional[float]] = []
    knee_series:   list[Optional[float]] = []
    straight_flag: list[bool]             = []
    for i in range(n):
        ra = _compute_raise_angle(ts, i, side)
        ka = _compute_knee_angle(ts, i, side)
        ks = (ka is not None) and (ka >= _STRAIGHT_THRESHOLD_DEG)
        raise_series.append(ra)
        knee_series.append(ka)
        straight_flag.append(ks)

    # ── Aggregate: max valid raise + peak frame index ───────
    max_raise = 0.0
    max_idx: Optional[int] = None
    knee_at_peak: Optional[float] = None
    straight_count = 0
    for i in range(n):
        if straight_flag[i]:
            straight_count += 1
            ra = raise_series[i]
            if ra is not None and ra > max_raise:
                max_raise = ra
                max_idx = i
                knee_at_peak = knee_series[i]

    knee_straight_fraction = straight_count / n if n > 0 else 0.0

    if max_raise < _MIN_RAISE_FOR_VALID_TRIAL_DEG:
        if straight_count == 0:
            raise ValueError(
                "knee_not_straight: the knee did not stay straight enough during the "
                "raise. Re-record with the patient keeping the knee fully extended "
                "throughout the lift."
            )
        raise ValueError(
            "no_raise: no leg raise detected on the "
            f"{side} leg. Please re-record the patient raising the leg from "
            "flat to as high as they can comfortably reach."
        )

    # ── 10 Hz time-series + keypoints export ────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(0, n, step))
    # Re-map max_idx into the sampled series (so the frontend's
    # `max_raise_sample_index` lines up with `samples[i]`).
    max_sample_index: Optional[int] = None
    if max_idx is not None:
        # Find the closest sampled-frame index.
        sample_pos = max_idx // step
        if sample_pos < len(sampled_frames):
            max_sample_index = sample_pos

    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in sampled_frames:
        t_ms = (i / fps) * 1000.0 if fps > 0 else 0.0
        samples.append({
            "t_ms": float(t_ms),
            "raise_angle_deg": float(raise_series[i]) if raise_series[i] is not None else None,
            "knee_angle_deg":  float(knee_series[i])  if knee_series[i]  is not None else None,
            "knee_straight":   bool(straight_flag[i]),
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
    # Termination semantics: backend always sees a fully captured clip
    # — call it "completed" when we got past the validity gates above.
    # A frontend "stopped" / "timeout" doesn't apply on the upload path.
    termination = "completed"

    # Peak-frame screenshot
    peak_screenshot: Optional[str] = None
    if max_idx is not None:
        peak_screenshot = _grab_peak_frame(video_path, max_idx, raw, side)

    classification = _classify_slr(max_raise)
    interpretation = _build_interpretation(
        side, max_raise, knee_straight_fraction, classification,
    )

    return {
        "side_tested": side,
        "max_raise_angle_deg": float(max_raise),
        "max_raise_sample_index": max_sample_index,
        "knee_angle_at_peak_deg": float(knee_at_peak) if knee_at_peak is not None else None,
        "classification": classification,
        "duration_seconds": duration_seconds,
        "termination": termination,
        "knee_straight_fraction": float(knee_straight_fraction),
        "samples": samples,
        "keypoints": keypoints_export,
        "peak_screenshot_data_url": peak_screenshot,
        # Diagnostic extras
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
    }
