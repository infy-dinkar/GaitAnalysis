"""sit_to_stand_engine.py — backend 5x Sit-to-Stand test (Test C2) on
the MediaPipe BlazePose Full (33-keypoint) pipeline.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/sitToStand.ts so live + upload report
the same metrics:

  hip_mid_y   = avg(left_hip.y, right_hip.y) with single-hip fallback
                (lateral view often drops the far hip)
  knee_angle  = interior angle hip-knee-ankle (180° = extended), best
                of L/R by min-keypoint-score
  leg_length  = hypot(ankle - hip) of the same camera-facing leg —
                sets the rep-detector threshold scale
  arms_crossed= both wrists.y ≤ shoulder_y + 0.50 × torso_height
                (sticky once violated)

Rep detection (sit↔stand 2-state machine — NOT scipy find_peaks):
  baseline_y = hip_mid_y at the first valid frame
               (test protocol: patient seated and still at "Go" click)
  sit → standing: hip_mid_y < baseline_y − 0.20 × leg_length_px
  standing → sit: hip_mid_y > baseline_y − 0.08 × leg_length_px
  each sit-back transition closes a rep + records min_knee_angle

Aggregates:
  total_time     = sum of rep durations (NOT trial wall-clock)
  cv_percent     = stdev(rep_durations) / mean(rep_durations) × 100
  classification = total_time < 12 → "normal"
                   12-15        → "borderline"
                   > 15         → "weakness"
  fatigue_flag   = last_rep > first_rep × 1.6
  arm_uncrossed_flag = sticky boolean

Returns a dict matching the frontend SitToStandResult shape (flat —
no L/R split) so the existing SitToStandReport renders without
translation. This is a SINGLE-trial test; one POST = one result.
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Optional

import cv2
import numpy as np

from engines.gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/sitToStand.ts) ────────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TARGET_REP_COUNT = 5
_TRIAL_TIMEOUT_SEC = 30.0
_STAND_DELTA_FRAC = 0.20
_SIT_DELTA_FRAC = 0.08
_FATIGUE_RATIO = 1.6
_NORMAL_MAX_SEC = 12.0
_BORDERLINE_MAX_SEC = 15.0
_ARM_UNCROSS_TORSO_FRAC = 0.50


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _compute_hip_mid_y(ts: dict, i: int) -> Optional[float]:
    """Average of L/R hip pixel-Y with single-hip fallback. Lateral
    view often loses the far hip — fall back to whichever side is
    visible."""
    l_vis = _visible(ts, "left_hip", i)
    r_vis = _visible(ts, "right_hip", i)
    if not l_vis and not r_vis:
        return None
    if l_vis and r_vis:
        return (float(ts["left_hip"]["y_px"][i]) + float(ts["right_hip"]["y_px"][i])) / 2.0
    return float(ts["left_hip"]["y_px"][i]) if l_vis else float(ts["right_hip"]["y_px"][i])


def _compute_shoulder_mid_y(ts: dict, i: int) -> Optional[float]:
    l_vis = _visible(ts, "left_shoulder", i)
    r_vis = _visible(ts, "right_shoulder", i)
    if not l_vis and not r_vis:
        return None
    if l_vis and r_vis:
        return (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    return float(ts["left_shoulder"]["y_px"][i]) if l_vis else float(ts["right_shoulder"]["y_px"][i])


def _compute_knee_angle(ts: dict, i: int) -> Optional[float]:
    """Interior knee angle (180° = extended). Picks the side with the
    highest min-score across hip+knee+ankle so the camera-facing leg
    wins in lateral view."""
    best_angle: Optional[float] = None
    best_score = 0.0
    for hip_k, knee_k, ankle_k in [
        ("left_hip",  "left_knee",  "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ]:
        if not (_visible(ts, hip_k, i) and _visible(ts, knee_k, i) and _visible(ts, ankle_k, i)):
            continue
        min_score = min(
            float(ts[hip_k]["vis"][i]),
            float(ts[knee_k]["vis"][i]),
            float(ts[ankle_k]["vis"][i]),
        )
        if min_score <= best_score:
            continue
        hx = float(ts[hip_k]["x_px"][i]); hy = float(ts[hip_k]["y_px"][i])
        kx = float(ts[knee_k]["x_px"][i]); ky = float(ts[knee_k]["y_px"][i])
        ax = float(ts[ankle_k]["x_px"][i]); ay = float(ts[ankle_k]["y_px"][i])
        v1x, v1y = hx - kx, hy - ky
        v2x, v2y = ax - kx, ay - ky
        m1 = math.hypot(v1x, v1y)
        m2 = math.hypot(v2x, v2y)
        if m1 < 1e-6 or m2 < 1e-6:
            continue
        cos_theta = (v1x * v2x + v1y * v2y) / (m1 * m2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        best_angle = math.degrees(math.acos(cos_theta))
        best_score = min_score
    return best_angle


def _compute_leg_length_px(ts: dict, i: int) -> Optional[float]:
    """hypot(ankle - hip) of the camera-facing leg. Same side-picking
    rule as `_compute_knee_angle`."""
    best_len: Optional[float] = None
    best_score = 0.0
    for hip_k, ankle_k in [
        ("left_hip",  "left_ankle"),
        ("right_hip", "right_ankle"),
    ]:
        if not (_visible(ts, hip_k, i) and _visible(ts, ankle_k, i)):
            continue
        min_score = min(
            float(ts[hip_k]["vis"][i]),
            float(ts[ankle_k]["vis"][i]),
        )
        if min_score <= best_score:
            continue
        hx = float(ts[hip_k]["x_px"][i]); hy = float(ts[hip_k]["y_px"][i])
        ax = float(ts[ankle_k]["x_px"][i]); ay = float(ts[ankle_k]["y_px"][i])
        best_len = math.hypot(ax - hx, ay - hy)
        best_score = min_score
    return best_len


def _are_arms_crossed(ts: dict, i: int) -> bool:
    """False if EITHER wrist drops more than ARM_UNCROSS_TORSO_FRAC ×
    torso-height below the shoulder line. Returns True when we can't
    measure (occluded but trial hasn't seen a clear drop) — matches
    the JS implementation."""
    shoulder_y = _compute_shoulder_mid_y(ts, i)
    hip_y = _compute_hip_mid_y(ts, i)
    if shoulder_y is None or hip_y is None:
        return True
    torso_h = abs(hip_y - shoulder_y)
    if torso_h < 1.0:
        return True
    tolerance = torso_h * _ARM_UNCROSS_TORSO_FRAC

    if _visible(ts, "left_wrist", i):
        lw_y = float(ts["left_wrist"]["y_px"][i])
        if lw_y > shoulder_y + tolerance:
            return False
    if _visible(ts, "right_wrist", i):
        rw_y = float(ts["right_wrist"]["y_px"][i])
        if rw_y > shoulder_y + tolerance:
            return False
    return True


# ─── Classification ─────────────────────────────────────────────
def _classify_total_time(total_sec: float) -> str:
    if total_sec < _NORMAL_MAX_SEC: return "normal"
    if total_sec <= _BORDERLINE_MAX_SEC: return "borderline"
    return "weakness"


def _compute_cv_percent(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return float(math.sqrt(variance) / mean * 100.0)


def _build_interpretation(
    reps: list[dict],
    incomplete: bool,
    total_time: float,
    rep_durations: list[float],
    classification: str,
    fatigue_flag: bool,
    arm_uncrossed: bool,
    termination: str,
) -> str:
    rep_count = len(reps)
    if rep_count == 0:
        return "No completed reps captured — re-run the trial."
    reps_phrase = (
        f"{rep_count} of {_TARGET_REP_COUNT} reps"
        if incomplete
        else f"{_TARGET_REP_COUNT} reps"
    )
    total_str = f"{total_time:.1f}"
    parts: list[str] = []
    if classification == "normal":
        parts.append(
            f"Completed {reps_phrase} in {total_str} s "
            f"(< {_NORMAL_MAX_SEC:.0f} s) — normal lower-extremity "
            f"strength for the 5x sit-to-stand benchmark."
        )
    elif classification == "borderline":
        parts.append(
            f"Completed {reps_phrase} in {total_str} s "
            f"({_NORMAL_MAX_SEC:.0f}–{_BORDERLINE_MAX_SEC:.0f} s) — "
            f"borderline performance; consider re-test or further evaluation."
        )
    else:
        parts.append(
            f"Completed {reps_phrase} in {total_str} s "
            f"(> {_BORDERLINE_MAX_SEC:.0f} s) — lower-extremity weakness / "
            f"elevated fall risk."
        )
    if fatigue_flag and len(rep_durations) >= 2:
        parts.append(
            f"Last rep was {rep_durations[-1]:.1f} s vs first rep "
            f"{rep_durations[0]:.1f} s "
            f"(> {round((_FATIGUE_RATIO - 1) * 100)}% slowdown) — significant fatigue."
        )
    if arm_uncrossed:
        parts.append(
            "Arms uncrossed at one or more points during the trial — "
            "strength assessment may be inflated."
        )
    if incomplete:
        term_phrase = "30 s timeout" if termination == "timeout" else "stopped early"
        parts.append(
            f"Trial ended before {_TARGET_REP_COUNT} reps were captured ({term_phrase})."
        )
    return " ".join(parts)


# ─── Deepest-knee screenshot ────────────────────────────────────
def _grab_deepest_knee_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
) -> Optional[str]:
    """Grab the deepest-knee-bend frame with a skeleton overlay,
    return as `data:image/jpeg;base64,...`. Mirrors the SLS engine's
    _grab_worst_rep_frame helper — no per-side highlighting since
    sit-to-stand has no L/R split."""
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

    # Trunk + both legs + arms (for arm-crossing inspection).
    edges = [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
        ("left_hip",       "left_knee"),
        ("left_knee",      "left_ankle"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
        ("left_shoulder",  "left_elbow"),
        ("left_elbow",     "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow",    "right_wrist"),
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
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ─── Main entry point ──────────────────────────────────────────
def analyze_sit_to_stand(
    video_path: str,
    pose_options,
) -> dict:
    """Run the 5x Sit-to-Stand (Test C2) pipeline on an uploaded
    clip. Single trial — no `side` parameter, no L/R split.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().

    Returns:
        Dict matching the frontend SitToStandResult shape (flat —
        no L/R) plus a few backend-only extras (fps, total_frames,
        valid_frames, interpretation). The existing SitToStandReport
        renders the matched fields without translation.

    Raises:
        ValueError: 'poor_visibility' when fewer than 30% of frames
                    have trunk + leg visible. User-facing message
                    when zero reps are detected. api maps both to
                    HTTP 400.
    """
    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_knee"]["y"]),
        len(ts["right_knee"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    # ── Visibility gate ──────────────────────────────────────
    # Lateral view often drops the FAR hip / knee / ankle, so we
    # gate on "trunk + at least ONE leg" rather than requiring both
    # sides.
    visible_frames = 0
    for i in range(n):
        trunk_ok = (
            (_visible(ts, "left_hip", i) or _visible(ts, "right_hip", i))
            and (_visible(ts, "left_shoulder", i) or _visible(ts, "right_shoulder", i))
        )
        leg_ok = (
            (_visible(ts, "left_hip", i) and _visible(ts, "left_knee", i) and _visible(ts, "left_ankle", i))
            or (_visible(ts, "right_hip", i) and _visible(ts, "right_knee", i) and _visible(ts, "right_ankle", i))
        )
        if trunk_ok and leg_ok:
            visible_frames += 1
    if visible_frames < max(3, int(n * 0.30)):
        raise ValueError("poor_visibility")

    # ── Baseline lock (first valid frame) ────────────────────
    baseline_y: Optional[float] = None
    leg_length_px: Optional[float] = None
    first_valid_idx = -1
    for i in range(n):
        if baseline_y is None:
            baseline_y = _compute_hip_mid_y(ts, i)
        if leg_length_px is None:
            leg_length_px = _compute_leg_length_px(ts, i)
        if baseline_y is not None and leg_length_px is not None:
            first_valid_idx = i
            break
    if baseline_y is None or leg_length_px is None or first_valid_idx < 0:
        raise ValueError("poor_visibility")

    # ── Rep state machine ─────────────────────────────────────
    # Sit ↔ Standing transitions on hip-mid Y vs baseline. Direct
    # frame-by-frame port of stepRepDetector() in sitToStand.ts.
    stand_threshold = baseline_y - leg_length_px * _STAND_DELTA_FRAC
    sit_threshold   = baseline_y - leg_length_px * _SIT_DELTA_FRAC

    current_state = "sitting"
    sit_events_ms: list[float] = [0.0]   # seed: t=0 = patient seated
    reps: list[dict] = []
    current_min_knee = 180.0
    current_min_knee_frame = first_valid_idx
    arm_uncrossed_flag = False
    deepest_knee_overall = 180.0
    deepest_knee_frame = first_valid_idx
    last_valid_frame = first_valid_idx
    sample_step = max(1, int(round(fps / _SAMPLE_HZ)))

    for i in range(first_valid_idx, n):
        hip_y = _compute_hip_mid_y(ts, i)
        if hip_y is None:
            continue
        last_valid_frame = i
        t_ms = ((i - first_valid_idx) / fps) * 1000.0

        knee = _compute_knee_angle(ts, i)
        if knee is not None and knee < current_min_knee:
            current_min_knee = knee
            current_min_knee_frame = i
        if knee is not None and knee < deepest_knee_overall:
            deepest_knee_overall = knee
            deepest_knee_frame = i

        if not _are_arms_crossed(ts, i):
            arm_uncrossed_flag = True

        if current_state == "sitting" and hip_y < stand_threshold:
            current_state = "standing"
        elif current_state == "standing" and hip_y > sit_threshold:
            current_state = "sitting"
            sit_events_ms.append(t_ms)
            # Commit rep.
            start_ms = sit_events_ms[-2]
            end_ms   = sit_events_ms[-1]
            duration_sec = max(0.0, (end_ms - start_ms) / 1000.0)
            reps.append({
                "rep_index": len(reps) + 1,
                "duration_seconds": float(duration_sec),
                "min_knee_angle_deg": float(current_min_knee),
            })
            # Reset cycle tracking.
            current_min_knee = 180.0
            if len(reps) >= _TARGET_REP_COUNT:
                break

        # Safety: hard cap on the analyzed window matching the live
        # trial timeout. After 30s we stop accruing reps.
        if (t_ms / 1000.0) >= _TRIAL_TIMEOUT_SEC:
            break

    if not reps:
        raise ValueError(
            "No sit-to-stand repetitions detected. Please ensure the "
            "patient performs full sit-to-stand movements with the "
            "camera positioned to the side (lateral view)."
        )

    # ── Aggregates ───────────────────────────────────────────
    rep_durations = [r["duration_seconds"] for r in reps]
    total_time = float(sum(rep_durations))
    cv_percent = _compute_cv_percent(rep_durations)
    fatigue_flag = (
        len(rep_durations) >= 2
        and rep_durations[-1] > rep_durations[0] * _FATIGUE_RATIO
    )

    incomplete = len(reps) < _TARGET_REP_COUNT
    if incomplete:
        # Clip ran out before 5 reps — treat as the natural timeout
        # equivalent (matches live mode's `timeout` terminator).
        termination = "timeout"
    else:
        termination = "completed"

    trial_duration_seconds = float((last_valid_frame - first_valid_idx) / fps) if fps > 0 else 0.0

    # ── 10 Hz time-series + keypoints export ─────────────────
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in range(first_valid_idx, last_valid_frame + 1, sample_step):
        t_ms = ((i - first_valid_idx) / fps) * 1000.0
        samples.append({
            "t_ms": float(t_ms),
            "hip_mid_y":      _maybe_float(_compute_hip_mid_y(ts, i)),
            "knee_angle_deg": _maybe_float(_compute_knee_angle(ts, i)),
            "arms_crossed":   bool(_are_arms_crossed(ts, i)),
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

    # ── Deepest-knee screenshot ──────────────────────────────
    last_rep_screenshot = _grab_deepest_knee_frame(
        video_path, deepest_knee_frame, raw,
    )

    classification = _classify_total_time(total_time)
    interpretation = _build_interpretation(
        reps, incomplete, total_time, rep_durations,
        classification, fatigue_flag, arm_uncrossed_flag, termination,
    )

    return {
        "total_time_seconds": total_time,
        "reps": reps,
        "rep_durations": rep_durations,
        "cv_percent": cv_percent,
        "classification": classification,
        "fatigue_flag": bool(fatigue_flag),
        "arm_uncrossed_flag": bool(arm_uncrossed_flag),
        "termination": termination,
        "incomplete": bool(incomplete),
        "trial_duration_seconds": trial_duration_seconds,
        "samples": samples,
        "keypoints": keypoints_export,
        "last_rep_screenshot_data_url": last_rep_screenshot,
        # Extras (not in the strict TS SitToStandResult shape; the
        # frontend ignores unknown keys but they're useful for the
        # response envelope + future report enrichment).
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
    }


def _maybe_float(v: Optional[float]) -> Optional[float]:
    return float(v) if v is not None else None
