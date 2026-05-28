"""single_leg_squat_engine.py — backend Single-Leg Squat / Step-Down
test (Test B1) on the MediaPipe BlazePose Full (33-keypoint) pipeline.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/singleLegSquat.ts so live + upload
report the same metrics:

  KFPPA       = acos((hip→knee) · (knee→ankle) /
                     (|hip→knee| × |knee→ankle|)) × 180/π
  pelvic tilt = atan2(lHip.y − rHip.y, lHip.x − rHip.x) × 180/π
                (positive = LEFT side down)
  trunk lean  = hip_mid → shoulder_mid vector vs vertical, signed by vx<0
                (positive = lean to patient's RIGHT)
  rep detect  = local maxima of the smoothed hip-midpoint Y trajectory
                (image y-down: bottom of squat = peak Y), via
                scipy.signal.find_peaks with `distance` and `prominence`
                gates that match the live PEAK_MIN_SEPARATION_FRAMES (30)
                and PEAK_MIN_DEPTH_PX (50) constants

Classification cutoffs (PDF Test B1):
  KFPPA  < 10°  → "good"
  KFPPA  10-15° → "borderline"
  KFPPA  > 15°  → "valgus"

Composite risk per side (same combination rule as JS):
  high     ← KFPPA > 15° OR ≥ 2 flags
  moderate ← exactly 1 flag (KFPPA 10-15°, drop > 5°, lean > 7°)
  low      ← all metrics within normal band

Pipeline:
  1. Reuse gait_engine.extract_poses() + build_time_series() to
     pull smoothed 33-kp landmarks per frame.
  2. Camera-squareness check from the median shoulder horizontal
     angle across the clip (warn-only — does not reject).
  3. Capture leg-length + baseline hip-mid Y from the first valid
     frame (matches live mode exactly).
  4. Detect rep peaks via scipy.signal.find_peaks on the smoothed
     hip-mid Y trajectory.
  5. Per rep: compute KFPPA, pelvic drop, trunk lean, depth_pct
     at the peak frame.
  6. 10 Hz time-series + per-frame keypoints export for
     TrendelenburgReport-compatible chart rendering.
  7. Worst-rep screenshot (skeleton overlay) at the peak frame
     with the largest KFPPA.

Returns a dict matching the frontend SingleLegSquatSideResult shape
so the existing SingleLegSquatReport renders without translation.
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

from engines.gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/singleLegSquat.ts) ───
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TARGET_REP_COUNT = 5
_TRIAL_TIMEOUT_SEC = 30.0
_PEAK_MIN_DEPTH_PX = 50.0          # find_peaks prominence
_PEAK_MIN_SEPARATION_FRAMES = 30   # find_peaks distance
_SQUARENESS_TOLERANCE_DEG = 5.0
_KFPPA_GOOD_MAX_DEG = 10.0
_KFPPA_BORDERLINE_MAX_DEG = 15.0
_PELVIC_DROP_THRESHOLD_DEG = 5.0
_TRUNK_LEAN_THRESHOLD_DEG = 7.0
_ASYMMETRY_THRESHOLD_DEG = 5.0

_SIDE_INDICES = {
    "left":  {"hip": "left_hip",  "knee": "left_knee",  "ankle": "left_ankle"},
    "right": {"hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle"},
}


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _compute_kfppa(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    hip_k, knee_k, ankle_k = idx["hip"], idx["knee"], idx["ankle"]
    if not (_visible(ts, hip_k, i) and _visible(ts, knee_k, i) and _visible(ts, ankle_k, i)):
        return None
    hx = float(ts[hip_k]["x_px"][i]);    hy = float(ts[hip_k]["y_px"][i])
    kx = float(ts[knee_k]["x_px"][i]);   ky = float(ts[knee_k]["y_px"][i])
    ax = float(ts[ankle_k]["x_px"][i]);  ay = float(ts[ankle_k]["y_px"][i])
    v1x, v1y = kx - hx, ky - hy
    v2x, v2y = ax - kx, ay - ky
    m1 = math.hypot(v1x, v1y)
    m2 = math.hypot(v2x, v2y)
    if m1 < 1e-6 or m2 < 1e-6:
        return None
    cos_theta = (v1x * v2x + v1y * v2y) / (m1 * m2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _compute_pelvic_tilt(ts: dict, i: int) -> Optional[float]:
    if not (_visible(ts, "left_hip", i) and _visible(ts, "right_hip", i)):
        return None
    lx = float(ts["left_hip"]["x_px"][i]);  ly = float(ts["left_hip"]["y_px"][i])
    rx = float(ts["right_hip"]["x_px"][i]); ry = float(ts["right_hip"]["y_px"][i])
    return math.degrees(math.atan2(ly - ry, lx - rx))


def _compute_trunk_lean(ts: dict, i: int) -> Optional[float]:
    for k in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        if not _visible(ts, k, i):
            return None
    hip_mid_x = (float(ts["left_hip"]["x_px"][i])  + float(ts["right_hip"]["x_px"][i]))  / 2.0
    hip_mid_y = (float(ts["left_hip"]["y_px"][i])  + float(ts["right_hip"]["y_px"][i]))  / 2.0
    sh_mid_x  = (float(ts["left_shoulder"]["x_px"][i]) + float(ts["right_shoulder"]["x_px"][i])) / 2.0
    sh_mid_y  = (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    vx = sh_mid_x - hip_mid_x
    vy = sh_mid_y - hip_mid_y
    mag = abs(math.degrees(math.atan2(vx, vy)))
    return (1.0 if vx < 0 else -1.0) * mag


def _compute_hip_mid_y(ts: dict, i: int) -> Optional[float]:
    if not (_visible(ts, "left_hip", i) and _visible(ts, "right_hip", i)):
        return None
    return (float(ts["left_hip"]["y_px"][i]) + float(ts["right_hip"]["y_px"][i])) / 2.0


def _compute_leg_length_px(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    hip_k, ankle_k = idx["hip"], idx["ankle"]
    if not (_visible(ts, hip_k, i) and _visible(ts, ankle_k, i)):
        return None
    hx = float(ts[hip_k]["x_px"][i]); hy = float(ts[hip_k]["y_px"][i])
    ax = float(ts[ankle_k]["x_px"][i]); ay = float(ts[ankle_k]["y_px"][i])
    return math.hypot(ax - hx, ay - hy)


def _compute_shoulder_horizontal_deg(ts: dict, i: int) -> Optional[float]:
    """LINE angle of shoulder-to-shoulder vs horizontal, normalised to
    [-90°, 90°]. Mirrors computeShoulderHorizontalDeg() in JS."""
    if not (_visible(ts, "left_shoulder", i) and _visible(ts, "right_shoulder", i)):
        return None
    lx = float(ts["left_shoulder"]["x_px"][i]); ly = float(ts["left_shoulder"]["y_px"][i])
    rx = float(ts["right_shoulder"]["x_px"][i]); ry = float(ts["right_shoulder"]["y_px"][i])
    deg = math.degrees(math.atan2(ry - ly, rx - lx))
    if deg > 90.0:  deg -= 180.0
    if deg < -90.0: deg += 180.0
    return deg


# ─── Classification ─────────────────────────────────────────────
def _classify_kfppa(worst_kfppa_deg: float) -> str:
    if worst_kfppa_deg < _KFPPA_GOOD_MAX_DEG:       return "good"
    if worst_kfppa_deg <= _KFPPA_BORDERLINE_MAX_DEG: return "borderline"
    return "valgus"


def _composite_risk(
    worst_kfppa: float,
    mean_pelvic_drop: float,
    mean_trunk_lean: float,
) -> str:
    if worst_kfppa > _KFPPA_BORDERLINE_MAX_DEG:
        return "high"
    flags = 0
    if worst_kfppa >= _KFPPA_GOOD_MAX_DEG: flags += 1
    if abs(mean_pelvic_drop) > _PELVIC_DROP_THRESHOLD_DEG: flags += 1
    if abs(mean_trunk_lean)  > _TRUNK_LEAN_THRESHOLD_DEG:  flags += 1
    if flags >= 2:   return "high"
    if flags == 1:   return "moderate"
    return "low"


def _build_interpretation(
    side: str,
    reps_count: int,
    incomplete: bool,
    worst_kfppa: float,
    mean_pelvic_drop: float,
    mean_trunk_lean: float,
    classification: str,
) -> str:
    side_label = "Left-leg squat" if side == "left" else "Right-leg squat"
    rep_summary = (
        f"{reps_count} of {_TARGET_REP_COUNT} reps completed"
        if incomplete
        else f"{reps_count} reps completed"
    )
    parts: list[str] = []
    if classification == "good":
        parts.append(
            f"{side_label} ({rep_summary}): good knee tracking — worst KFPPA "
            f"{worst_kfppa:.1f}° (< {_KFPPA_GOOD_MAX_DEG:.0f}°)."
        )
    elif classification == "borderline":
        parts.append(
            f"{side_label} ({rep_summary}): borderline knee tracking — worst KFPPA "
            f"{worst_kfppa:.1f}° ({_KFPPA_GOOD_MAX_DEG:.0f}–{_KFPPA_BORDERLINE_MAX_DEG:.0f}°). "
            "Monitor and reassess."
        )
    else:
        parts.append(
            f"{side_label} ({rep_summary}): dynamic valgus — worst KFPPA "
            f"{worst_kfppa:.1f}° (> {_KFPPA_BORDERLINE_MAX_DEG:.0f}°), elevated ACL/PFP risk."
        )
    if abs(mean_pelvic_drop) > _PELVIC_DROP_THRESHOLD_DEG:
        parts.append(
            f"{side_label}: mean pelvic drop {abs(mean_pelvic_drop):.1f}° "
            f"(> {_PELVIC_DROP_THRESHOLD_DEG:.0f}°) — hip abductor insufficiency on the stance side."
        )
    if abs(mean_trunk_lean) > _TRUNK_LEAN_THRESHOLD_DEG:
        parts.append(
            f"{side_label}: mean trunk lateral lean {abs(mean_trunk_lean):.1f}° "
            f"(> {_TRUNK_LEAN_THRESHOLD_DEG:.0f}°) — compensatory lean."
        )
    return " ".join(parts)


# ─── Worst-rep screenshot ───────────────────────────────────────
def _grab_worst_rep_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    side: str,
) -> Optional[str]:
    """Grab the worst-rep frame with skeleton overlay, return as
    `data:image/jpeg;base64,...`. Test-side leg + trunk highlighted;
    contralateral leg dimmed for context. Mirrors the trendelenburg
    engine's _grab_peak_frame helper."""
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
def analyze_single_leg_squat(
    video_path: str,
    pose_options,
    side: str,
) -> dict:
    """Run the Single-Leg Squat pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        side:         'left' or 'right' — the STANCE leg (the leg
                      the patient is standing on during the squat).

    Returns:
        Dict matching the frontend SingleLegSquatSideResult shape
        plus a `camera_squareness_warning` boolean and `fps` /
        `total_frames` / `valid_frames` / `interpretation` extras.
        The existing SingleLegSquatReport ignores the extras and
        renders the matched fields without translation.

    Raises:
        ValueError: 'poor_visibility' for clips where the patient
                    is not consistently visible, or a user-facing
                    message when no squat reps are detected.
                    api.analyze_single_leg_squat_endpoint maps
                    these to HTTP 400.
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

    # ── Visibility gate ──────────────────────────────────────
    # Need a healthy fraction of frames with at least the trunk + test
    # leg visible — otherwise rep detection is hopeless. Mirrors the
    # ankle/TUG visibility floor (>30% of frames).
    idx = _SIDE_INDICES[side]
    visible_frames = 0
    for i in range(n):
        if (_visible(ts, "left_hip", i)
                and _visible(ts, "right_hip", i)
                and _visible(ts, "left_shoulder", i)
                and _visible(ts, "right_shoulder", i)
                and _visible(ts, idx["knee"], i)
                and _visible(ts, idx["ankle"], i)):
            visible_frames += 1
    if visible_frames < max(3, int(n * 0.30)):
        raise ValueError("poor_visibility")

    # ── Camera-squareness check ──────────────────────────────
    # Per-frame shoulder horizontal angle; surface a soft warning
    # when the patient was rotated more than the spec tolerance for
    # the majority of the clip. Warn-only — does not reject.
    sq_angles: list[float] = []
    for i in range(n):
        deg = _compute_shoulder_horizontal_deg(ts, i)
        if deg is not None:
            sq_angles.append(deg)
    median_sq_deg = float(np.median(sq_angles)) if sq_angles else 0.0
    camera_squareness_warning = abs(median_sq_deg) > _SQUARENESS_TOLERANCE_DEG

    # ── Baselines (first valid frame) ────────────────────────
    leg_length_px: Optional[float] = None
    baseline_hip_y: Optional[float] = None
    first_valid_idx = -1
    for i in range(n):
        if leg_length_px is None:
            leg_length_px = _compute_leg_length_px(ts, i, side)
        if baseline_hip_y is None:
            baseline_hip_y = _compute_hip_mid_y(ts, i)
        if leg_length_px is not None and baseline_hip_y is not None:
            first_valid_idx = i
            break

    if leg_length_px is None or baseline_hip_y is None or first_valid_idx < 0:
        raise ValueError("poor_visibility")

    # ── Rep detection via scipy find_peaks ───────────────────
    # Smoothed hip-midpoint Y. NaN-fill missing frames with the
    # baseline so find_peaks doesn't trip on isolated dropouts —
    # the prominence gate still rejects spurious bumps.
    hip_mid_y_series = np.full(n, baseline_hip_y, dtype=float)
    for i in range(n):
        v = _compute_hip_mid_y(ts, i)
        if v is not None:
            hip_mid_y_series[i] = v
    # Scale separation by FPS so the 30-frame minimum matches the
    # live mode at variable frame rates. The live constant assumes
    # ~30 fps; at 60 fps we want 60 frames between reps, at 24 fps
    # we want 24 — i.e. keep the 1-second minimum.
    distance_frames = max(
        1, int(round(_PEAK_MIN_SEPARATION_FRAMES * (fps / 30.0)))
    )
    peaks, _props = find_peaks(
        hip_mid_y_series,
        distance=distance_frames,
        prominence=_PEAK_MIN_DEPTH_PX,
    )
    # Cap at TARGET_REP_COUNT — the first N peaks (chronological).
    rep_frames: list[int] = [int(p) for p in peaks[:_TARGET_REP_COUNT]]

    if not rep_frames:
        raise ValueError(
            f"No single-leg squat reps detected on the {side} leg. "
            "Please re-record the patient performing single-leg squats "
            "facing the camera with the test leg stance and the other "
            "leg lifted."
        )

    # ── Per-rep metrics ──────────────────────────────────────
    reps: list[dict] = []
    for r_index, frame_i in enumerate(rep_frames, start=1):
        peak_t_ms = ((frame_i - first_valid_idx) / fps) * 1000.0
        kfppa = _compute_kfppa(ts, frame_i, side)
        pelvic = _compute_pelvic_tilt(ts, frame_i)
        lean = _compute_trunk_lean(ts, frame_i)
        peak_hip_y = _compute_hip_mid_y(ts, frame_i)
        depth_pct: Optional[float] = None
        if peak_hip_y is not None and leg_length_px:
            depth_pct = (peak_hip_y - baseline_hip_y) / leg_length_px * 100.0
        reps.append({
            "rep_index": r_index,
            "t_ms": float(peak_t_ms),
            "kfppa_deg": float(kfppa) if kfppa is not None else None,
            "pelvic_drop_deg": float(pelvic) if pelvic is not None else None,
            "trunk_lean_deg":  float(lean)   if lean   is not None else None,
            "depth_pct":       float(depth_pct) if depth_pct is not None else None,
        })

    # ── Per-rep aggregates ───────────────────────────────────
    worst_idx: Optional[int] = None
    worst_kfppa = 0.0
    for i, r in enumerate(reps):
        k = r["kfppa_deg"]
        if k is not None and k > worst_kfppa:
            worst_kfppa = k
            worst_idx = i

    def _mean_of(xs: list[Optional[float]]) -> float:
        vs = [x for x in xs if x is not None]
        return float(sum(vs) / len(vs)) if vs else 0.0

    mean_drop  = _mean_of([r["pelvic_drop_deg"] for r in reps])
    mean_lean  = _mean_of([r["trunk_lean_deg"]  for r in reps])
    mean_depth = _mean_of([r["depth_pct"]       for r in reps])

    # ── 10 Hz time-series + keypoints export ─────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(first_valid_idx, n, step))
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in sampled_frames:
        t_ms = ((i - first_valid_idx) / fps) * 1000.0
        samples.append({
            "t_ms": float(t_ms),
            "hip_mid_y":       _maybe_float(_compute_hip_mid_y(ts, i)),
            "kfppa_deg":       _maybe_float(_compute_kfppa(ts, i, side)),
            "pelvic_drop_deg": _maybe_float(_compute_pelvic_tilt(ts, i)),
            "trunk_lean_deg":  _maybe_float(_compute_trunk_lean(ts, i)),
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

    # ── Termination + duration ───────────────────────────────
    duration_seconds = float(n / fps) if fps > 0 else 0.0
    incomplete = len(reps) < _TARGET_REP_COUNT
    if incomplete:
        # Clip ran out before 5 reps — treat as a natural timeout.
        termination = "timeout"
    else:
        termination = "completed"

    # ── Worst-rep screenshot ─────────────────────────────────
    worst_screenshot: Optional[str] = None
    if worst_idx is not None:
        worst_frame_i = rep_frames[worst_idx]
        worst_screenshot = _grab_worst_rep_frame(
            video_path, worst_frame_i, raw, side,
        )

    classification = _classify_kfppa(worst_kfppa)
    risk_score = _composite_risk(worst_kfppa, mean_drop, mean_lean)
    interpretation = _build_interpretation(
        side, len(reps), incomplete, worst_kfppa, mean_drop, mean_lean,
        classification,
    )

    return {
        "side_tested": side,
        "reps": reps,
        "worst_rep_index": worst_idx,
        "worst_kfppa_deg": float(worst_kfppa),
        "mean_pelvic_drop_deg": float(mean_drop),
        "mean_trunk_lean_deg":  float(mean_lean),
        "mean_depth_pct":       float(mean_depth),
        "classification": classification,
        "risk_score": risk_score,
        "duration_seconds": duration_seconds,
        "termination": termination,
        "incomplete": bool(incomplete),
        "samples": samples,
        "keypoints": keypoints_export,
        "worst_rep_screenshot_data_url": worst_screenshot,
        # Extras (not in the strict SingleLegSquatSideResult type; the
        # frontend ignores unknown keys but they're useful for the
        # response envelope + future report enrichment).
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
        "camera_squareness_warning": bool(camera_squareness_warning),
        "median_shoulder_tilt_deg": float(median_sq_deg),
    }


def _maybe_float(v: Optional[float]) -> Optional[float]:
    return float(v) if v is not None else None
