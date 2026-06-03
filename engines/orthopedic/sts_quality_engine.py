"""sts_quality_engine.py — backend Sit-to-Stand QUALITY (B4) pipeline
on the MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/stsQuality.ts so live + upload return
identical numbers.

SEPARATE from the 5x Sit-to-Stand (C2) speed test at
engines/orthopedic/sit_to_stand_engine.py — C2 measures total cycle
time across 5 reps, B4 measures QUALITY (phase timing per rep, trunk
lean + knee at seat-off, smoothness score, hand-use compensation
flag). They share no code; do not cross-import.

Math:
  knee_angle      = inner angle at the knee between (knee→hip) and
                    (knee→ankle). 180° = straight, 90° = bent square.
  trunk_lean_deg  = angle of test-side hip→shoulder from vertical.
                    Same formula as Forward Lunge.
  hip_y           = test-side hip pixel Y. Image y-down, so the
                    valleys (smallest y) correspond to the patient
                    standing tallest.
  smoothness      = 1 / (1 + stddev(acc) / mean(|vel|)) over the
                    rising phase. 1.0 = perfectly constant velocity.
  hand_use        = test-side wrist below test-side shoulder by more
                    than 10% of leg length during the rising phase.

Rep detection:
  scipy.signal.find_peaks on `-hip_y_series` with distance + prominence
  gates matching SLS / FL exactly. Peaks in `-hip_y` = valleys in
  hip_y = moments of standing.

Phase boundaries (per detected standing-moment frame `v`):
  seat_off          = walk backwards from v while hip_y is rising;
                      stop where velocity returns to ≈ 0 or hip_y is
                      back near baseline.
  top_of_stand      = v.
  start_of_descent  = first frame after v where velocity ≥
                      PHASE_STABLE_VELOCITY_PX (hip_y increasing
                      again, i.e. patient starts descending).
  re_seated         = first frame after start_of_descent where hip_y
                      is back at baseline AND velocity ≈ 0.

Classification mirrors classifySTSQuality():
  compensated  if hand_use_count >= 2 OR median trunk lean > 55°
  smooth       if hand_use_count == 0 AND median smoothness >= 0.7
  hesitant     otherwise.
"""
from __future__ import annotations

import base64
import logging
import math
from statistics import median as _stdlib_median, pstdev
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


# ─── Spec constants (mirror lib/orthopedic/stsQuality.ts) ───────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TARGET_REP_COUNT = 3
_TRIAL_TIMEOUT_SEC = 45.0
_PEAK_MIN_DEPTH_PX = 50.0           # find_peaks prominence
_PEAK_MIN_SEPARATION_FRAMES = 30    # find_peaks distance
_PHASE_STABLE_VELOCITY_PX = 2.0
_HAND_USE_WRIST_DROP_RATIO = 0.10
_TRUNK_LEAN_EFFICIENT_MIN_DEG = 30.0
_TRUNK_LEAN_EFFICIENT_MAX_DEG = 45.0
_TRUNK_LEAN_MOMENTUM_DEG = 55.0
_SMOOTHNESS_SMOOTH_MIN = 0.7

_SIDE_INDICES = {
    "left": {
        "shoulder": "left_shoulder",
        "wrist": "left_wrist",
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
    },
    "right": {
        "shoulder": "right_shoulder",
        "wrist": "right_wrist",
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
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


def _compute_wrist_y(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _visible(ts, idx["wrist"], i):
        return None
    return float(ts[idx["wrist"]]["y_px"][i])


def _compute_shoulder_y(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _visible(ts, idx["shoulder"], i):
        return None
    return float(ts[idx["shoulder"]]["y_px"][i])


def _compute_leg_length_px(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _all_visible(ts, (idx["hip"], idx["ankle"]), i):
        return None
    hx = float(ts[idx["hip"]]["x_px"][i]);   hy = float(ts[idx["hip"]]["y_px"][i])
    ax = float(ts[idx["ankle"]]["x_px"][i]); ay = float(ts[idx["ankle"]]["y_px"][i])
    return math.hypot(ax - hx, ay - hy)


# ─── Phase boundary detection ──────────────────────────────────
def _detect_phase_boundaries(
    hip_y_series: list[Optional[float]],
    baseline_y: float,
    top_index: int,
) -> dict:
    """Mirror of stsQuality.ts:detectPhaseBoundaries. Returns dict
    with keys seat_off, top_of_stand, start_of_descent, re_seated —
    each an int index or None."""
    v = top_index
    n = len(hip_y_series)

    # Walk backwards from v to find seat_off.
    seat_off: Optional[int] = None
    i = v - 1
    while i >= 0:
        cur = hip_y_series[i]
        nxt = hip_y_series[i + 1]
        if cur is None or nxt is None:
            i -= 1
            continue
        vel = cur - nxt  # positive = cur further from top (i.e. earlier in rise)
        if abs(vel) < _PHASE_STABLE_VELOCITY_PX:
            seat_off = i
            break
        if cur >= baseline_y - _PHASE_STABLE_VELOCITY_PX:
            seat_off = i
            break
        i -= 1
    if seat_off is None:
        seat_off = 0

    # Walk forwards to find start_of_descent.
    start_of_descent: Optional[int] = None
    for j in range(v + 1, n):
        cur = hip_y_series[j]
        prv = hip_y_series[j - 1]
        if cur is None or prv is None:
            continue
        vel = cur - prv  # positive = hip_y rising (image y-down) = patient descending
        if vel > _PHASE_STABLE_VELOCITY_PX:
            start_of_descent = j
            break

    # Walk forwards from start_of_descent to find re_seated.
    re_seated: Optional[int] = None
    if start_of_descent is not None:
        for j in range(start_of_descent + 1, n):
            cur = hip_y_series[j]
            prv = hip_y_series[j - 1]
            if cur is None or prv is None:
                continue
            at_baseline = cur >= baseline_y - _PHASE_STABLE_VELOCITY_PX
            vel = cur - prv
            if at_baseline and abs(vel) < _PHASE_STABLE_VELOCITY_PX:
                re_seated = j
                break

    return {
        "seat_off": seat_off,
        "top_of_stand": v,
        "start_of_descent": start_of_descent,
        "re_seated": re_seated,
    }


# ─── Smoothness score ───────────────────────────────────────────
def _compute_smoothness_score(
    hip_y_series: list[Optional[float]],
    start_idx: int,
    end_idx: int,
) -> Optional[float]:
    if end_idx <= start_idx + 3:
        return None
    segment: list[float] = []
    for i in range(start_idx, end_idx + 1):
        v = hip_y_series[i]
        if v is None:
            return None
        segment.append(v)
    if len(segment) < 5:
        return None
    # Central-difference velocity.
    vel = [(segment[i + 1] - segment[i - 1]) / 2.0 for i in range(1, len(segment) - 1)]
    if len(vel) < 3:
        return None
    acc = [(vel[i + 1] - vel[i - 1]) / 2.0 for i in range(1, len(vel) - 1)]
    if not acc:
        return None
    mean_abs_v = sum(abs(x) for x in vel) / len(vel)
    if mean_abs_v == 0.0:
        return None
    std_a = pstdev(acc) if len(acc) >= 2 else 0.0
    jerk_proxy = std_a / mean_abs_v
    return 1.0 / (1.0 + jerk_proxy)


# ─── Hand-use detection ─────────────────────────────────────────
def _detect_hand_use_in_phase(
    wrist_y_series: list[Optional[float]],
    shoulder_y_series: list[Optional[float]],
    leg_len_series: list[Optional[float]],
    start_idx: int,
    end_idx: int,
) -> bool:
    for i in range(start_idx, end_idx + 1):
        wr = wrist_y_series[i]
        sh = shoulder_y_series[i]
        leg = leg_len_series[i]
        if wr is None or sh is None or leg is None or leg <= 0:
            continue
        drop_px = wr - sh
        if drop_px > leg * _HAND_USE_WRIST_DROP_RATIO:
            return True
    return False


# ─── Median + classification ────────────────────────────────────
def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(_stdlib_median(values))


def _classify_sts_quality(
    median_smoothness: Optional[float],
    median_trunk_deg: Optional[float],
    hand_use_count: int,
) -> str:
    if hand_use_count >= 2:
        return "compensated"
    if median_trunk_deg is not None and median_trunk_deg > _TRUNK_LEAN_MOMENTUM_DEG:
        return "compensated"
    if (
        hand_use_count == 0
        and median_smoothness is not None
        and median_smoothness >= _SMOOTHNESS_SMOOTH_MIN
    ):
        return "smooth"
    return "hesitant"


def _build_interpretation(result: dict) -> str:
    reps = result["reps"]
    rep_count = len(reps)
    if rep_count == 0:
        return "No completed STS-quality trial to interpret."
    rep_summary = (
        f"{rep_count} of {_TARGET_REP_COUNT} reps captured"
        if rep_count < _TARGET_REP_COUNT
        else f"{rep_count} reps captured"
    )
    issues: list[str] = []
    if result["any_hand_use"]:
        issues.append(
            f"wrist pushed off below the shoulder on "
            f"{result['hand_use_count']} of {rep_count} reps — significant "
            f"lower-extremity weakness"
        )
    mt = result["median_trunk_lean_deg"]
    if mt is not None:
        if mt > _TRUNK_LEAN_MOMENTUM_DEG:
            issues.append(
                f"median trunk lean {mt:.1f}° at seat-off "
                f"(> {_TRUNK_LEAN_MOMENTUM_DEG:.0f}°) — momentum-dependent strategy, "
                f"hip/quad weakness"
            )
        elif mt < _TRUNK_LEAN_EFFICIENT_MIN_DEG:
            issues.append(
                f"median trunk lean {mt:.1f}° at seat-off "
                f"(< {_TRUNK_LEAN_EFFICIENT_MIN_DEG:.0f}°) — limited forward weight transfer"
            )
    ms = result["median_smoothness_score"]
    if ms is not None and ms < _SMOOTHNESS_SMOOTH_MIN:
        issues.append(
            f"median smoothness {ms:.2f} (< {_SMOOTHNESS_SMOOTH_MIN}) — hesitant / jerky rise"
        )

    cls = result["classification"]
    if cls == "smooth" and not issues:
        t = result["median_trunk_lean_deg"]
        k = result["median_knee_angle_deg"]
        trunk_str = f"{t:.0f}°" if t is not None else "—"
        knee_str  = f"{k:.0f}°" if k is not None else "—"
        return (
            f"{rep_summary}: smooth movement quality. Median trunk lean "
            f"{trunk_str} at seat-off, knee {knee_str} at seat-off, no "
            f"hand-use compensation detected."
        )
    return f"{rep_summary}: {cls}. {'; '.join(issues)}."


# ─── Worst/representative-rep screenshot ───────────────────────
def _grab_top_of_stand_frame(
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
def analyze_sts_quality(
    video_path: str,
    pose_options,
    side: str,
    chair_seat_height_cm: Optional[float] = None,
) -> dict:
    """Run the B4 Sit-to-Stand Quality pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        side:         'left' or 'right' — the CAMERA-FACING side.
        chair_seat_height_cm: optional metadata, recorded verbatim in
                              the result for context.

    Returns:
        Dict matching the frontend STSQualityResult shape plus a few
        diagnostic extras. The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' when the patient isn't clearly
                    visible in enough frames, or 'no_reps' when no
                    standing moments are detected.
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

    # ── Sub-sample to SAMPLE_HZ (matches frontend live cadence) ──
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_full_indices = list(range(0, n, step))
    sampled_count = len(sampled_full_indices)

    # Per-frame metrics on the sampled grid.
    hip_y_arr:     list[Optional[float]] = []
    knee_arr:      list[Optional[float]] = []
    trunk_arr:     list[Optional[float]] = []
    wrist_y_arr:   list[Optional[float]] = []
    shoulder_y_arr: list[Optional[float]] = []
    leg_arr:       list[Optional[float]] = []
    for fi in sampled_full_indices:
        hip_y_arr.append(_compute_hip_y(ts, fi, side))
        knee_arr.append(_compute_knee_angle(ts, fi, side))
        trunk_arr.append(_compute_trunk_lean_deg(ts, fi, side))
        wrist_y_arr.append(_compute_wrist_y(ts, fi, side))
        shoulder_y_arr.append(_compute_shoulder_y(ts, fi, side))
        leg_arr.append(_compute_leg_length_px(ts, fi, side))

    # ── Rep detection: find standing-moment valleys in hip_y. ─
    # Fill NaNs forward then back so find_peaks runs cleanly.
    series = np.array(
        [(v if v is not None else np.nan) for v in hip_y_arr],
        dtype=float,
    )
    last = None
    for i in range(sampled_count):
        if not math.isnan(series[i]):
            last = float(series[i])
        elif last is not None:
            series[i] = last
    first = None
    for i in range(sampled_count):
        if not math.isnan(series[i]):
            first = float(series[i])
            break
    if first is not None:
        for i in range(sampled_count):
            if math.isnan(series[i]):
                series[i] = first

    # We feed `-series` so that find_peaks fires at the valleys of
    # hip_y (the standing moments). distance and prominence gates
    # are bit-identical to SLS / FL.
    distance = max(1, int(round(_PEAK_MIN_SEPARATION_FRAMES * (_SAMPLE_HZ / 10.0))))
    peaks, _props = find_peaks(
        -series,
        distance=distance,
        prominence=_PEAK_MIN_DEPTH_PX,
    )
    top_indices: list[int] = [int(p) for p in peaks[:_TARGET_REP_COUNT]]

    if not top_indices:
        raise ValueError(
            f"no_reps: no sit-to-stand reps detected. Please re-record the "
            f"patient performing 3 sit-to-stand cycles from a standard chair."
        )

    # ── Seated baseline = median of the first 1 s of hip_y. ───
    baseline_window = min(sampled_count, max(5, _SAMPLE_HZ))
    baseline_candidates: list[float] = []
    for i in range(baseline_window):
        if hip_y_arr[i] is not None:
            baseline_candidates.append(float(hip_y_arr[i]))  # type: ignore[arg-type]
    baseline_y = float(_median(baseline_candidates)) if baseline_candidates else 0.0

    # ── Per-rep extraction ──────────────────────────────────
    reps: list[dict] = []
    for r, top_idx in enumerate(top_indices, start=1):
        phase = _detect_phase_boundaries(hip_y_arr, baseline_y, top_idx)

        def t_at(idx: Optional[int]) -> Optional[float]:
            if idx is None or idx < 0 or idx >= sampled_count:
                return None
            full = sampled_full_indices[idx]
            return (full / fps) * 1000.0 if fps > 0 else 0.0

        seat_off_t  = t_at(phase["seat_off"])
        top_t       = t_at(top_idx) or 0.0
        sod_t       = t_at(phase["start_of_descent"])
        reseated_t  = t_at(phase["re_seated"])

        sit_to_stand = (top_t - seat_off_t) if seat_off_t is not None else None
        pause        = (sod_t - top_t) if sod_t is not None else None
        stand_to_sit = (reseated_t - sod_t) if sod_t is not None and reseated_t is not None else None

        seat_off_idx = phase["seat_off"] if phase["seat_off"] is not None else top_idx
        trunk_at_so = trunk_arr[seat_off_idx]
        knee_at_so  = knee_arr[seat_off_idx]

        smoothness = None
        if phase["seat_off"] is not None:
            smoothness = _compute_smoothness_score(hip_y_arr, phase["seat_off"], top_idx)

        hand_use = False
        if phase["seat_off"] is not None:
            hand_use = _detect_hand_use_in_phase(
                wrist_y_arr, shoulder_y_arr, leg_arr,
                phase["seat_off"], top_idx,
            )

        reps.append({
            "rep_index": r,
            "seat_off_t_ms": seat_off_t,
            "top_of_stand_t_ms": top_t,
            "start_of_descent_t_ms": sod_t,
            "re_seated_t_ms": reseated_t,
            "sit_to_stand_ms": max(0.0, sit_to_stand) if sit_to_stand is not None else None,
            "pause_ms":        max(0.0, pause)        if pause        is not None else None,
            "stand_to_sit_ms": max(0.0, stand_to_sit) if stand_to_sit is not None else None,
            "trunk_lean_at_seat_off_deg": float(trunk_at_so) if trunk_at_so is not None else None,
            "knee_angle_at_seat_off_deg": float(knee_at_so)  if knee_at_so  is not None else None,
            "smoothness_score": float(smoothness) if smoothness is not None else None,
            "hand_use_detected": bool(hand_use),
        })

    # ── Per-trial aggregates (medians) ────────────────────────
    incomplete = len(reps) < _TARGET_REP_COUNT
    hand_use_count = sum(1 for r in reps if r["hand_use_detected"])

    def _med(key: str) -> Optional[float]:
        vals = [r[key] for r in reps if r[key] is not None]
        return _median(vals) if vals else None  # type: ignore[return-value]

    med_sit  = _med("sit_to_stand_ms")
    med_pau  = _med("pause_ms")
    med_sit2 = _med("stand_to_sit_ms")
    med_trunk = _med("trunk_lean_at_seat_off_deg")
    med_knee  = _med("knee_angle_at_seat_off_deg")
    med_sm    = _med("smoothness_score")

    classification = _classify_sts_quality(med_sm, med_trunk, hand_use_count)

    # ── 10 Hz samples + keypoints export ─────────────────────
    samples_out: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for sample_idx, full_i in enumerate(sampled_full_indices):
        t_ms = (full_i / fps) * 1000.0 if fps > 0 else 0.0
        samples_out.append({
            "t_ms": float(t_ms),
            "hip_y":          float(hip_y_arr[sample_idx])      if hip_y_arr[sample_idx]      is not None else None,
            "knee_angle_deg": float(knee_arr[sample_idx])       if knee_arr[sample_idx]       is not None else None,
            "trunk_lean_deg": float(trunk_arr[sample_idx])      if trunk_arr[sample_idx]      is not None else None,
            "wrist_y":        float(wrist_y_arr[sample_idx])    if wrist_y_arr[sample_idx]    is not None else None,
            "shoulder_y":     float(shoulder_y_arr[sample_idx]) if shoulder_y_arr[sample_idx] is not None else None,
            "leg_length_px":  float(leg_arr[sample_idx])        if leg_arr[sample_idx]        is not None else None,
        })
        kp_frame: list[dict] = [
            {"x": 0.0, "y": 0.0, "score": 0.0} for _ in range(33)
        ]
        for name, idx_n in LM.items():
            frames = raw.get(name, [])
            if full_i < len(frames) and frames[full_i] is not None:
                x_n, y_n, vis = frames[full_i]
                kp_frame[idx_n] = {
                    "x": float(x_n) * float(raw.get("_frame_w", 1) or 1),
                    "y": float(y_n) * float(raw.get("_frame_h", 1) or 1),
                    "score": float(vis),
                }
        keypoints_export.append(kp_frame)

    duration_seconds = float(n / fps) if fps > 0 else 0.0
    termination = "timeout" if incomplete else "completed"

    # Screenshot: the first top-of-stand moment (most representative).
    screenshot: Optional[str] = None
    if top_indices:
        full_frame_idx = sampled_full_indices[top_indices[0]]
        screenshot = _grab_top_of_stand_frame(
            video_path, full_frame_idx, raw, side,
        )

    result = {
        "camera_side": side,
        "chair_seat_height_cm":
            float(chair_seat_height_cm) if chair_seat_height_cm is not None else None,
        "reps": reps,
        "median_sit_to_stand_ms": med_sit,
        "median_pause_ms": med_pau,
        "median_stand_to_sit_ms": med_sit2,
        "median_trunk_lean_deg": med_trunk,
        "median_knee_angle_deg": med_knee,
        "median_smoothness_score": med_sm,
        "hand_use_count": int(hand_use_count),
        "any_hand_use": hand_use_count > 0,
        "classification": classification,
        "duration_seconds": duration_seconds,
        "termination": termination,
        "incomplete": bool(incomplete),
        "samples": samples_out,
        "keypoints": keypoints_export,
        "worst_rep_screenshot_data_url": screenshot,
        # Diagnostic extras
        "fps":            float(fps),
        "total_frames":   int(n),
        "valid_frames":   int(visible_frames),
    }
    result["interpretation"] = _build_interpretation(result)
    return result
