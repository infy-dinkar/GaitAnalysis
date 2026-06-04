"""pronator_drift_engine.py — backend Pronator Drift (E2) pipeline on
the MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors lib/orthopedic/pronatorDrift.ts so live + upload return
identical numbers.

The test:
  Patient holds both arms forward at shoulder height, palms up, eyes
  closed, for ~20 seconds. Frontal view. We track each wrist's
  vertical (Y) position and report the drift from a baseline taken
  in the first second of the stable hold.

Math:
  baseline_wrist_y  median wrist Y across the BASELINE window
                    (after SETTLE_DELAY_MS, lasting BASELINE_WINDOW_SEC)
  final_wrist_y     median wrist Y across the last FINAL_WINDOW_SEC
  drift_px          final - baseline. Positive = wrist dropped
                    (image y-down: drop = increasing y)
  drift_cm          (drift_px / mean_shoulder_width_px) * 40 cm
  drift_velocity    least-squares slope of drift_cm vs time across
                    the post-baseline portion of the hold

2D LIMITATION (surfaced in the report): forearm rotation /
pronation isn't measurable by a single 2D camera. This engine
captures vertical drop only.

Classification mirrors classifyPronatorDrift():
  positive_screen  max downward drift > 5 cm AND min < 2 cm  (classic
                   asymmetric pattern), OR  max/min > 3 : 1 with
                   max > 2 cm.
  borderline       max downward drift > 2 cm.
  normal           otherwise.
"""
from __future__ import annotations

import base64
import logging
import math
from statistics import median as _stdlib_median
from typing import Optional

import cv2

from engines.gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/pronatorDrift.ts) ────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TARGET_HOLD_DURATION_SEC = 20.0
_SETTLE_DELAY_MS = 500.0
_BASELINE_WINDOW_SEC = 1.0
_FINAL_WINDOW_SEC = 1.0
_ASSUMED_SHOULDER_WIDTH_CM = 40.0
# Classification thresholds.
_POSITIVE_DRIFT_CM = 5.0
_STABLE_THRESHOLD_CM = 2.0
_POSITIVE_ASYMMETRY_RATIO = 3.0
_BORDERLINE_DRIFT_CM = 2.0


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _all_visible(ts: dict, keys: tuple[str, ...], i: int) -> bool:
    return all(_visible(ts, k, i) for k in keys)


def _wrist_y(ts: dict, i: int, side: str) -> Optional[float]:
    key = f"{side}_wrist"
    if not _visible(ts, key, i):
        return None
    return float(ts[key]["y_px"][i])


def _shoulder_y(ts: dict, i: int, side: str) -> Optional[float]:
    key = f"{side}_shoulder"
    if not _visible(ts, key, i):
        return None
    return float(ts[key]["y_px"][i])


def _shoulder_width_px(ts: dict, i: int) -> Optional[float]:
    if not _all_visible(ts, ("left_shoulder", "right_shoulder"), i):
        return None
    lx = float(ts["left_shoulder"]["x_px"][i])
    ly = float(ts["left_shoulder"]["y_px"][i])
    rx = float(ts["right_shoulder"]["x_px"][i])
    ry = float(ts["right_shoulder"]["y_px"][i])
    return math.hypot(lx - rx, ly - ry)


# ─── Median + linear regression ────────────────────────────────
def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(_stdlib_median(values))


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    sx = sy = sxx = sxy = 0.0
    for i in range(n):
        sx += xs[i]; sy += ys[i]
        sxx += xs[i] * xs[i]
        sxy += xs[i] * ys[i]
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return 0.0
    return (n * sxy - sx * sy) / denom


# ─── Classification ─────────────────────────────────────────────
def _classify(left_cm: Optional[float], right_cm: Optional[float]) -> str:
    l_down = max(0.0, left_cm)  if left_cm  is not None else 0.0
    r_down = max(0.0, right_cm) if right_cm is not None else 0.0
    max_down = max(l_down, r_down)
    min_down = min(l_down, r_down)

    if max_down > _POSITIVE_DRIFT_CM and min_down < _STABLE_THRESHOLD_CM:
        return "positive_screen"
    if max_down > _BORDERLINE_DRIFT_CM and min_down > 0.1:
        ratio = max_down / min_down
        if ratio > _POSITIVE_ASYMMETRY_RATIO:
            return "positive_screen"
    if max_down > _BORDERLINE_DRIFT_CM:
        return "borderline"
    return "normal"


def _build_interpretation(result: dict) -> str:
    parts: list[str] = []
    parts.append(
        f"{result['hold_duration_seconds']:.1f} s hold. "
        f"Left arm: {_fmt_drift(result['left']['drift_cm'])}. "
        f"Right arm: {_fmt_drift(result['right']['drift_cm'])}."
    )
    cls = result["classification"]
    max_down = result["max_downward_drift_cm"]
    min_down = result["min_downward_drift_cm"]
    asym = result["asymmetry_ratio"]
    if cls == "positive_screen":
        if max_down > _POSITIVE_DRIFT_CM and min_down < _STABLE_THRESHOLD_CM:
            parts.append(
                f"Asymmetric drop pattern: one arm dropped {max_down:.1f} cm "
                f"while the other stayed within {_STABLE_THRESHOLD_CM:.0f} cm — "
                f"positive screen."
            )
        elif asym > _POSITIVE_ASYMMETRY_RATIO:
            parts.append(
                f"Asymmetry ratio {asym:.1f} : 1 "
                f"(threshold {_POSITIVE_ASYMMETRY_RATIO:.0f} : 1) — positive screen "
                f"for subtle upper-motor-neuron weakness on the affected side."
            )
        else:
            parts.append("Positive screen — see report for details.")
    elif cls == "borderline":
        parts.append(
            f"Borderline drop on at least one arm ({max_down:.1f} cm). "
            f"Repeat with a longer hold or correlate with strength testing."
        )
    else:
        parts.append(
            f"No significant drift — both arms stayed within the "
            f"{_BORDERLINE_DRIFT_CM:.0f} cm tolerance."
        )
    parts.append(
        "Note: this 2D measurement captures the vertical drop only. True "
        "clinical pronator drift also involves forearm rotation/pronation "
        "as the arm drops — that rotation is NOT assessed here. Clinical "
        "judgement required."
    )
    return " ".join(parts)


def _fmt_drift(cm: Optional[float]) -> str:
    if cm is None:
        return "not tracked"
    if cm > 0.5:
        return f"dropped {cm:.1f} cm"
    if cm < -0.5:
        return f"rose {abs(cm):.1f} cm"
    return "held position"


# ─── Capture-moment screenshot ──────────────────────────────────
def _grab_capture_frame(
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
        ("left_shoulder",  "left_elbow"),
        ("left_elbow",     "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow",    "right_wrist"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
    ]
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            cv2.line(frame, dot_pos[a], dot_pos[b], (255, 255, 255), 2)
    for _name, (px, py) in dot_pos.items():
        cv2.circle(frame, (px, py), 5, (0, 0, 220), -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


# ─── Per-arm extraction ─────────────────────────────────────────
def _extract_arm_summary(
    samples: list[dict],
    side: str,                       # "left" or "right"
    mean_shoulder_width_px: float,
    baseline_start_idx: int,
    baseline_end_idx: int,
    final_start_idx: int,
    final_end_idx: int,
) -> dict:
    key = f"{side}_wrist_y"
    baseline_vals: list[float] = []
    for i in range(baseline_start_idx, min(baseline_end_idx + 1, len(samples))):
        v = samples[i][key]
        if v is not None:
            baseline_vals.append(float(v))
    baseline_y = _median(baseline_vals)

    final_vals: list[float] = []
    for i in range(final_start_idx, min(final_end_idx + 1, len(samples))):
        v = samples[i][key]
        if v is not None:
            final_vals.append(float(v))
    final_y = _median(final_vals)

    drift_px: Optional[float] = (
        final_y - baseline_y if baseline_y is not None and final_y is not None else None
    )
    drift_cm: Optional[float] = (
        (drift_px / mean_shoulder_width_px) * _ASSUMED_SHOULDER_WIDTH_CM
        if drift_px is not None and mean_shoulder_width_px > 0
        else None
    )

    # Per-frame drift cm series for the report chart.
    drift_cm_series: list[Optional[float]] = []
    for sm in samples:
        v = sm[key]
        if v is None or baseline_y is None or mean_shoulder_width_px <= 0:
            drift_cm_series.append(None)
        else:
            drift_cm_series.append(
                ((float(v) - baseline_y) / mean_shoulder_width_px)
                * _ASSUMED_SHOULDER_WIDTH_CM
            )

    # Drift velocity — slope of cm series vs time in seconds, over
    # the post-baseline portion only.
    reg_xs: list[float] = []
    reg_ys: list[float] = []
    for i in range(baseline_end_idx + 1, len(samples)):
        cm = drift_cm_series[i]
        if cm is None:
            continue
        reg_xs.append(samples[i]["t_ms"] / 1000.0)
        reg_ys.append(cm)
    drift_velocity = _linear_slope(reg_xs, reg_ys) if len(reg_xs) >= 2 else None

    return {
        "baseline_wrist_y_px": baseline_y,
        "final_wrist_y_px": final_y,
        "drift_px": drift_px,
        "drift_cm": drift_cm,
        "drift_velocity_cm_per_sec": drift_velocity,
        "drift_cm_series": drift_cm_series,
    }


# ─── Main entry point ──────────────────────────────────────────
def analyze_pronator_drift(
    video_path: str,
    pose_options,
) -> dict:
    """Run the E2 Pronator Drift pipeline on an uploaded clip.

    Returns a dict matching the frontend PronatorDriftResult shape
    plus diagnostic extras. The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' when both arms aren't tracked in
                    enough frames; 'too_short' if the recording is
                    shorter than 5 seconds.
    """
    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n_full = int(min(
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
        len(ts["left_wrist"]["y"]),
        len(ts["right_wrist"]["y"]),
    ))
    if n_full == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    duration_seconds = float(n_full / fps) if fps > 0 else 0.0
    if duration_seconds < 5.0:
        raise ValueError("too_short")

    # ── Visibility gate ─────────────────────────────────────
    required_keys = (
        "left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
    )
    visible_frames = sum(
        1 for i in range(n_full) if _all_visible(ts, required_keys, i)
    )
    if visible_frames < max(3, int(n_full * 0.30)):
        raise ValueError("poor_visibility")

    # ── Sub-sample to SAMPLE_HZ (matches frontend live cadence) ─
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_full_indices = list(range(0, n_full, step))

    # ── Per-frame samples on the sampled grid ───────────────
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for full_i in sampled_full_indices:
        t_ms = (full_i / fps) * 1000.0 if fps > 0 else 0.0
        samples.append({
            "t_ms": float(t_ms),
            "left_wrist_y":      _wrist_y(ts, full_i, "left"),
            "right_wrist_y":     _wrist_y(ts, full_i, "right"),
            "left_shoulder_y":   _shoulder_y(ts, full_i, "left"),
            "right_shoulder_y":  _shoulder_y(ts, full_i, "right"),
            "shoulder_width_px": _shoulder_width_px(ts, full_i),
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

    # ── Baseline + final sample-index windows ──────────────
    baseline_start_ms = _SETTLE_DELAY_MS
    baseline_end_ms = baseline_start_ms + _BASELINE_WINDOW_SEC * 1000.0
    baseline_start_idx = 0
    baseline_end_idx = -1
    for i, sm in enumerate(samples):
        if sm["t_ms"] >= baseline_start_ms and baseline_end_idx < 0:
            baseline_start_idx = i
        if sm["t_ms"] <= baseline_end_ms:
            baseline_end_idx = i
    if baseline_end_idx < baseline_start_idx:
        baseline_end_idx = baseline_start_idx

    last_t = samples[-1]["t_ms"] if samples else 0.0
    final_start_ms = max(baseline_end_ms + 100, last_t - _FINAL_WINDOW_SEC * 1000.0)
    final_start_idx = baseline_end_idx + 1
    for i in range(baseline_end_idx + 1, len(samples)):
        if samples[i]["t_ms"] >= final_start_ms:
            final_start_idx = i
            break
    final_end_idx = len(samples) - 1

    # ── Mean shoulder width across the trial ────────────────
    sw_vals: list[float] = [
        float(sm["shoulder_width_px"])
        for sm in samples
        if sm["shoulder_width_px"] is not None and sm["shoulder_width_px"] > 0
    ]
    mean_shoulder_width_px = sum(sw_vals) / len(sw_vals) if sw_vals else 0.0

    left  = _extract_arm_summary(
        samples, "left",  mean_shoulder_width_px,
        baseline_start_idx, baseline_end_idx,
        final_start_idx, final_end_idx,
    )
    right = _extract_arm_summary(
        samples, "right", mean_shoulder_width_px,
        baseline_start_idx, baseline_end_idx,
        final_start_idx, final_end_idx,
    )

    l_down = max(0.0, left["drift_cm"])  if left["drift_cm"]  is not None else 0.0
    r_down = max(0.0, right["drift_cm"]) if right["drift_cm"] is not None else 0.0
    max_down = max(l_down, r_down)
    min_down = min(l_down, r_down)
    asym_ratio = (max_down / min_down) if min_down > 0.1 else 999.0
    asym_abs = (
        abs(left["drift_cm"] - right["drift_cm"])
        if left["drift_cm"] is not None and right["drift_cm"] is not None
        else 0.0
    )

    classification = _classify(left["drift_cm"], right["drift_cm"])
    incomplete = duration_seconds < _TARGET_HOLD_DURATION_SEC - 1.0
    # Backend never sees a "stopped" termination — operator can only
    # stop in live mode. Treat as completed-or-timeout based on len.
    termination = "completed" if not incomplete else "timeout"

    # Screenshot: midpoint of the hold (most representative).
    screenshot: Optional[str] = None
    if samples:
        mid_full_idx = sampled_full_indices[len(sampled_full_indices) // 2]
        screenshot = _grab_capture_frame(video_path, mid_full_idx, raw)

    t_seconds_series = [sm["t_ms"] / 1000.0 for sm in samples]

    result = {
        "hold_duration_seconds": float(duration_seconds),
        "mean_shoulder_width_px": float(mean_shoulder_width_px),
        "left": left,
        "right": right,
        "t_seconds_series": t_seconds_series,
        "max_downward_drift_cm": float(max_down),
        "min_downward_drift_cm": float(min_down),
        "asymmetry_ratio": float(asym_ratio),
        "asymmetry_absolute_cm": float(asym_abs),
        "classification": classification,
        "termination": termination,
        "incomplete": bool(incomplete),
        "samples": samples,
        "keypoints": keypoints_export,
        "capture_screenshot_data_url": screenshot,
        # Diagnostic extras
        "fps": float(fps),
        "total_frames": int(n_full),
        "valid_frames": int(visible_frames),
    }
    result["interpretation"] = _build_interpretation(result)
    return result
