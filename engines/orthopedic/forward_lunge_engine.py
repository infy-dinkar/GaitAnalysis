"""forward_lunge_engine.py — backend Forward Lunge (B3) pipeline on
the MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/forwardLunge.ts so live + upload return
identical numbers.

Math (all on the FRONT / test leg):
  knee_angle           = inner angle at the knee between (knee→hip)
                         and (knee→ankle). Target band 85–95° at the
                         bottom of the lunge.
  knee_over_toe_ratio  = (knee.x − ankle.x) * direction_sign /
                         leg_length_px. Positive = knee forward of
                         foot. Threshold ~0.06 (≈ 5 cm fwd on an 85 cm
                         leg) triggers the ankle/quad-dominance flag.
  trunk_lean_deg       = angle of test-side hip → test-side shoulder
                         from vertical. Threshold 20° triggers the
                         posterior-chain compensation flag.
  body direction       = sign(foot_index.x − heel.x); falls back to
                         sign(foot_index.x − ankle.x) if heel is poor.

Per-rep + per-side aggregation:
  Five reps per side, segmented from local maxima of the TEST-side
  hip Y trajectory (in image y-down: hip Y rises while the patient
  descends, then falls during the push back to standing). The
  scipy.signal.find_peaks gates use distance + prominence parameters
  bit-identical to the SLS engine so the rep-detection behaviour is
  predictable across the multi-rep modules.

Classification cutoffs (mirror forwardLunge.ts):
  Per-rep flags:
    depth_out_of_band  if any rep knee at bottom < 70° or > 110°
    kot_flagged        if max KOT ratio > 0.06
    trunk_lean_flagged if max trunk lean > 20°
    fatigue_flagged    if depth_variation > 15°
  Severe single-flag escalations (force "poor"):
    trunk lean > 30°
    KOT > 0.12

Returns a dict matching the frontend ForwardLungeSideResult shape so
ForwardLungeReport renders without translation.
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


# ─── Spec constants (mirror lib/orthopedic/forwardLunge.ts) ─────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TARGET_REP_COUNT = 5
_TRIAL_TIMEOUT_SEC = 45.0
_PEAK_MIN_DEPTH_PX = 50.0          # find_peaks prominence (matches SLS)
_PEAK_MIN_SEPARATION_FRAMES = 30   # find_peaks distance (matches SLS)
_KNEE_TARGET_MIN_DEG = 85.0
_KNEE_TARGET_MAX_DEG = 95.0
_KNEE_DEPTH_HARD_MIN_DEG = 70.0
_KNEE_DEPTH_HARD_MAX_DEG = 110.0
_KOT_FLAG_RATIO = 0.06
_TRUNK_LEAN_FLAG_DEG = 20.0
_DEPTH_VARIATION_FLAG_DEG = 15.0

_SIDE_INDICES = {
    "left": {
        "shoulder": "left_shoulder",
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
        "heel": "left_heel",
        "foot_index": "left_foot_index",
    },
    "right": {
        "shoulder": "right_shoulder",
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
        "heel": "right_heel",
        "foot_index": "right_foot_index",
    },
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


def _detect_body_direction(ts: dict, i: int, side: str) -> Optional[int]:
    """+1 if the test-side foot points in image +x, −1 otherwise.
    Falls back to ankle→foot_index when the heel is poorly tracked."""
    idx = _SIDE_INDICES[side]
    if _all_visible(ts, (idx["heel"], idx["foot_index"]), i):
        hx = float(ts[idx["heel"]]["x_px"][i])
        fx = float(ts[idx["foot_index"]]["x_px"][i])
        dx = fx - hx
        if abs(dx) < 1e-3:
            return None
        return 1 if dx > 0 else -1
    if _all_visible(ts, (idx["ankle"], idx["foot_index"]), i):
        ax = float(ts[idx["ankle"]]["x_px"][i])
        fx = float(ts[idx["foot_index"]]["x_px"][i])
        dx = fx - ax
        if abs(dx) < 1e-3:
            return None
        return 1 if dx > 0 else -1
    return None


def _compute_kot_ratio(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _all_visible(ts, (idx["hip"], idx["knee"], idx["ankle"]), i):
        return None
    direction = _detect_body_direction(ts, i, side)
    if direction is None:
        return None
    hx = float(ts[idx["hip"]]["x_px"][i]);   hy = float(ts[idx["hip"]]["y_px"][i])
    kx = float(ts[idx["knee"]]["x_px"][i])
    ax = float(ts[idx["ankle"]]["x_px"][i]); ay = float(ts[idx["ankle"]]["y_px"][i])
    leg_length_px = math.hypot(ax - hx, ay - hy)
    if leg_length_px <= 0:
        return None
    knee_forward_px = (kx - ax) * direction
    return knee_forward_px / leg_length_px


def _compute_trunk_lean_deg(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _all_visible(ts, (idx["hip"], idx["shoulder"]), i):
        return None
    hx = float(ts[idx["hip"]]["x_px"][i]);     hy = float(ts[idx["hip"]]["y_px"][i])
    sx = float(ts[idx["shoulder"]]["x_px"][i]); sy = float(ts[idx["shoulder"]]["y_px"][i])
    dx = abs(sx - hx)
    dy = abs(sy - hy)
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(dx, dy))


def _compute_hip_y(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _visible(ts, idx["hip"], i):
        return None
    return float(ts[idx["hip"]]["y_px"][i])


# ─── Classification ─────────────────────────────────────────────
def _classify_lunge(
    worst_knee_deg: float,
    worst_kot_ratio: float,
    worst_trunk_deg: float,
    depth_variation_deg: float,
    incomplete: bool,
) -> str:
    if worst_trunk_deg > 30.0: return "poor"
    if worst_kot_ratio > 0.12: return "poor"

    depth_oob = (
        worst_knee_deg < _KNEE_DEPTH_HARD_MIN_DEG
        or worst_knee_deg > _KNEE_DEPTH_HARD_MAX_DEG
    )
    kot_flag   = worst_kot_ratio  > _KOT_FLAG_RATIO
    trunk_flag = worst_trunk_deg  > _TRUNK_LEAN_FLAG_DEG
    fatigue    = depth_variation_deg > _DEPTH_VARIATION_FLAG_DEG
    flag_count = (
        int(depth_oob)
        + int(kot_flag)
        + int(trunk_flag)
        + int(fatigue)
    )

    if incomplete:        return "borderline"
    if flag_count == 0:   return "good"
    if flag_count >= 3:   return "poor"
    return "borderline"


def _build_interpretation(
    side: str,
    reps_count: int,
    incomplete: bool,
    mean_knee: float,
    worst_knee: float,
    worst_kot: float,
    worst_trunk: float,
    depth_variation: float,
    classification: str,
    depth_oob: bool,
    kot_flag: bool,
    trunk_flag: bool,
    fatigue_flag: bool,
) -> str:
    side_label = "Left-leg lunge" if side == "left" else "Right-leg lunge"
    rep_summary = (
        f"{reps_count} of {_TARGET_REP_COUNT} reps captured"
        if incomplete else
        f"{reps_count} reps captured"
    )

    issues: list[str] = []
    if depth_oob:
        issues.append(
            f"depth out of band (worst rep {worst_knee:.0f}°, target "
            f"{_KNEE_TARGET_MIN_DEG:.0f}–{_KNEE_TARGET_MAX_DEG:.0f}°)"
        )
    if kot_flag:
        issues.append(
            f"knee passed forward of the foot — worst rep "
            f"{worst_kot * 100:.1f}% of leg length "
            f"(flag at {_KOT_FLAG_RATIO * 100:.0f}%) — ankle/quadriceps dominance"
        )
    if trunk_flag:
        issues.append(
            f"trunk forward lean {worst_trunk:.1f}° "
            f"(flag at {_TRUNK_LEAN_FLAG_DEG:.0f}°) — posterior-chain compensation"
        )
    if fatigue_flag:
        issues.append(
            f"depth varied {depth_variation:.1f}° across reps "
            f"(flag at {_DEPTH_VARIATION_FLAG_DEG:.0f}°) — possible fatigue / inconsistency"
        )

    if not issues:
        return (
            f"{side_label} ({rep_summary}): good. Mean knee depth "
            f"{mean_knee:.0f}°, knee never passed forward of the foot, trunk "
            f"stayed under {_TRUNK_LEAN_FLAG_DEG:.0f}° throughout."
        )
    cls = "poor" if classification == "poor" else "borderline"
    return f"{side_label} ({rep_summary}): {cls}. {'; '.join(issues)}."


# ─── Worst-rep composite scoring ────────────────────────────────
def _rep_composite_score(
    knee_deg: Optional[float],
    kot_ratio: Optional[float],
    trunk_deg: Optional[float],
) -> float:
    """Same composite as the frontend repCompositeScore — higher = worse."""
    s = 0.0
    if knee_deg is not None:
        s += abs(knee_deg - 90.0)
    if kot_ratio is not None:
        s += max(0.0, kot_ratio) * 200.0
    if trunk_deg is not None:
        s += trunk_deg
    return s


# ─── Worst-rep screenshot (skeleton-overlaid) ────────────────────
def _grab_worst_rep_frame(
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
def analyze_forward_lunge(
    video_path: str,
    pose_options,
    side: str,
) -> dict:
    """Run the Forward Lunge (B3) pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        side:         'left' or 'right' — the FRONT (test) leg.

    Returns:
        Dict matching the frontend ForwardLungeSideResult shape plus
        diagnostic extras (fps, total_frames, valid_frames,
        interpretation). The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' when the patient/leg isn't
                    clearly visible in enough frames, or a user-facing
                    message when no reps are detected.
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
    required_keys = (idx["shoulder"], idx["hip"], idx["knee"], idx["ankle"])
    visible_frames = sum(
        1 for i in range(n) if _all_visible(ts, required_keys, i)
    )
    if visible_frames < max(3, int(n * 0.30)):
        raise ValueError("poor_visibility")

    # ── Test-side hip Y trajectory for rep detection ─────────
    hip_y_series = np.full(n, np.nan, dtype=float)
    for i in range(n):
        v = _compute_hip_y(ts, i, side)
        if v is not None:
            hip_y_series[i] = v
    # Fill NaNs with the most recent valid sample (forward-fill) then
    # back-fill from the first valid sample; if the entire trace is
    # NaN (already covered by the visibility gate) we'd have raised.
    last = None
    for i in range(n):
        if not math.isnan(hip_y_series[i]):
            last = hip_y_series[i]
        elif last is not None:
            hip_y_series[i] = last
    first = None
    for i in range(n):
        if not math.isnan(hip_y_series[i]):
            first = hip_y_series[i]
            break
    if first is not None:
        for i in range(n):
            if math.isnan(hip_y_series[i]):
                hip_y_series[i] = first

    # ── Rep detection via scipy find_peaks (matches SLS) ─────
    distance_frames = max(
        1, int(round(_PEAK_MIN_SEPARATION_FRAMES * (fps / 30.0)))
    )
    peaks, _props = find_peaks(
        hip_y_series,
        distance=distance_frames,
        prominence=_PEAK_MIN_DEPTH_PX,
    )
    rep_frames: list[int] = [int(p) for p in peaks[:_TARGET_REP_COUNT]]

    if not rep_frames:
        raise ValueError(
            f"no_reps: no forward-lunge reps detected on the {side} leg. "
            "Please re-record the patient performing 5 lunges on the test "
            "leg from the side."
        )

    # ── Per-rep metrics ──────────────────────────────────────
    reps: list[dict] = []
    for r_index, frame_i in enumerate(rep_frames, start=1):
        peak_t_ms = (frame_i / fps) * 1000.0 if fps > 0 else 0.0
        knee = _compute_knee_angle(ts, frame_i, side)
        kot = _compute_kot_ratio(ts, frame_i, side)
        trunk = _compute_trunk_lean_deg(ts, frame_i, side)
        reps.append({
            "rep_index": r_index,
            "t_ms": float(peak_t_ms),
            "knee_angle_at_bottom_deg": float(knee)  if knee  is not None else None,
            "knee_over_toe_ratio":      float(kot)   if kot   is not None else None,
            "trunk_lean_deg":           float(trunk) if trunk is not None else None,
        })

    # ── Per-rep aggregates ───────────────────────────────────
    worst_idx: Optional[int] = None
    worst_score = -math.inf
    for i, r in enumerate(reps):
        s = _rep_composite_score(
            r["knee_angle_at_bottom_deg"],
            r["knee_over_toe_ratio"],
            r["trunk_lean_deg"],
        )
        if s > worst_score:
            worst_score = s
            worst_idx = i

    knee_vals  = [r["knee_angle_at_bottom_deg"] for r in reps if r["knee_angle_at_bottom_deg"] is not None]
    kot_vals   = [r["knee_over_toe_ratio"]      for r in reps if r["knee_over_toe_ratio"]      is not None]
    trunk_vals = [r["trunk_lean_deg"]           for r in reps if r["trunk_lean_deg"]           is not None]

    mean_knee = float(sum(knee_vals) / len(knee_vals)) if knee_vals else 0.0
    depth_variation = (max(knee_vals) - min(knee_vals)) if knee_vals else 0.0
    worst_knee = (
        reps[worst_idx]["knee_angle_at_bottom_deg"]
        if worst_idx is not None and reps[worst_idx]["knee_angle_at_bottom_deg"] is not None
        else 0.0
    )
    worst_kot   = max(kot_vals)   if kot_vals   else 0.0
    worst_trunk = max(trunk_vals) if trunk_vals else 0.0

    depth_oob = (
        len(knee_vals) == 0
        or any(
            v < _KNEE_DEPTH_HARD_MIN_DEG or v > _KNEE_DEPTH_HARD_MAX_DEG
            for v in knee_vals
        )
    )
    kot_flag   = worst_kot   > _KOT_FLAG_RATIO
    trunk_flag = worst_trunk > _TRUNK_LEAN_FLAG_DEG
    fatigue    = depth_variation > _DEPTH_VARIATION_FLAG_DEG

    incomplete = len(reps) < _TARGET_REP_COUNT
    termination = "timeout" if incomplete else "completed"

    classification = _classify_lunge(
        worst_knee if worst_knee != 0.0 else mean_knee,
        worst_kot,
        worst_trunk,
        depth_variation,
        incomplete,
    )
    interpretation = _build_interpretation(
        side, len(reps), incomplete,
        mean_knee, worst_knee, worst_kot, worst_trunk, depth_variation,
        classification, depth_oob, kot_flag, trunk_flag, fatigue,
    )

    # ── 10 Hz time-series + keypoints export ─────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_frames = list(range(0, n, step))
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in sampled_frames:
        t_ms = (i / fps) * 1000.0 if fps > 0 else 0.0
        hipy = _compute_hip_y(ts, i, side)
        knee = _compute_knee_angle(ts, i, side)
        kot = _compute_kot_ratio(ts, i, side)
        trunk = _compute_trunk_lean_deg(ts, i, side)
        samples.append({
            "t_ms": float(t_ms),
            "hip_y":              float(hipy)  if hipy  is not None else None,
            "knee_angle_deg":     float(knee)  if knee  is not None else None,
            "knee_over_toe_ratio": float(kot)  if kot   is not None else None,
            "trunk_lean_deg":     float(trunk) if trunk is not None else None,
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

    # ── Worst-rep screenshot ─────────────────────────────────
    worst_screenshot: Optional[str] = None
    if worst_idx is not None:
        worst_frame_i = rep_frames[worst_idx]
        worst_screenshot = _grab_worst_rep_frame(
            video_path, worst_frame_i, raw, side,
        )

    return {
        "side_tested": side,
        "reps": reps,
        "worst_rep_index": worst_idx,
        "worst_rep_knee_angle_deg":   float(worst_knee),
        "worst_rep_kot_ratio":        float(worst_kot),
        "worst_rep_trunk_lean_deg":   float(worst_trunk),
        "mean_knee_angle_deg":        float(mean_knee),
        "depth_variation_deg":        float(depth_variation),
        "depth_out_of_band":          bool(depth_oob),
        "kot_flagged":                bool(kot_flag),
        "trunk_lean_flagged":         bool(trunk_flag),
        "fatigue_flagged":            bool(fatigue),
        "classification":             classification,
        "duration_seconds":           duration_seconds,
        "termination":                termination,
        "incomplete":                 bool(incomplete),
        "samples":                    samples,
        "keypoints":                  keypoints_export,
        "worst_rep_screenshot_data_url": worst_screenshot,
        # Diagnostic extras
        "fps":            float(fps),
        "total_frames":   int(n),
        "valid_frames":   int(visible_frames),
        "interpretation": interpretation,
    }
