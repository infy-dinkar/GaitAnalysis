"""single_leg_stance_engine.py — backend Single-Leg Stance test
(Test C5) on the MediaPipe BlazePose Full (33-keypoint) pipeline.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/singleLegStance.ts. Per-trial
analysis: each video covers ONE (side, condition) combination
(left_open, right_open, left_closed, right_closed). The frontend
uploads up to 4 trials in parallel via Promise.allSettled and
assembles the combined SessionResult client-side.
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


# ─── Spec constants ─────────────────────────────────────────────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_MAX_EYES_OPEN_SEC = 60.0
_MAX_EYES_CLOSED_SEC = 30.0
_ONSET_TIMEOUT_SEC = 8.0

# Body-height-relative termination thresholds (mirror frontend).
_FOOT_TOUCHDOWN_RATIO = 0.05
_LIFT_ONSET_RATIO = 0.06
_KNEE_LIFT_RATIO = 0.10
_LIFT_VIS_THRESHOLD = 0.10
_ARM_GRAB_DEG = 45.0
_HOP_DISPLACEMENT_RATIO = 0.06
_HOP_WINDOW_MS = 500.0


# ─── Single-Leg Stance age-banded norms ─────────────────────────
_STANCE_BANDS: list[dict] = [
    {"age_min": 0,  "age_max": 59,  "open": 10.0, "closed": 5.0},
    {"age_min": 60, "age_max": 69,  "open": 7.0,  "closed": 3.5},
    {"age_min": 70, "age_max": 200, "open": 5.0,  "closed": 2.5},
]


def _get_stance_norm(age: Optional[int], eyes_closed: bool) -> dict:
    if age is None:
        band = _STANCE_BANDS[0]
        return {
            "threshold_sec": band["closed"] if eyes_closed else band["open"],
            "comparable": False,
            "band_label": "strictest threshold (patient age not available)",
        }
    for band in _STANCE_BANDS:
        if band["age_min"] <= age <= band["age_max"]:
            cond = "eyes-closed" if eyes_closed else "eyes-open"
            age_label = (
                "under 60" if band["age_min"] == 0
                else "70+" if band["age_max"] == 200
                else f"{band['age_min']}–{band['age_max']}"
            )
            return {
                "threshold_sec": band["closed"] if eyes_closed else band["open"],
                "comparable": True,
                "band_label": f"age {age_label}, {cond}",
            }
    last = _STANCE_BANDS[-1]
    return {
        "threshold_sec": last["closed"] if eyes_closed else last["open"],
        "comparable": False,
        "band_label": "out-of-range fallback",
    }


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int, threshold: float = _VIS_THRESHOLD) -> bool:
    return float(ts[key]["vis"][i]) >= threshold


def _compute_hip_mid(ts: dict, i: int) -> Optional[tuple[float, float]]:
    if not (_visible(ts, "left_hip", i) and _visible(ts, "right_hip", i)):
        return None
    return (
        (float(ts["left_hip"]["x_px"][i])  + float(ts["right_hip"]["x_px"][i]))  / 2.0,
        (float(ts["left_hip"]["y_px"][i])  + float(ts["right_hip"]["y_px"][i]))  / 2.0,
    )


def _compute_body_height_px(ts: dict, i: int) -> Optional[float]:
    if not (_visible(ts, "left_shoulder", i) and _visible(ts, "right_shoulder", i)):
        return None
    sh_y = (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    la_vis = _visible(ts, "left_ankle", i)
    ra_vis = _visible(ts, "right_ankle", i)
    if la_vis and ra_vis:
        ankle_y = (float(ts["left_ankle"]["y_px"][i]) + float(ts["right_ankle"]["y_px"][i])) / 2.0
    elif la_vis:
        ankle_y = float(ts["left_ankle"]["y_px"][i])
    elif ra_vis:
        ankle_y = float(ts["right_ankle"]["y_px"][i])
    else:
        return None
    return abs(ankle_y - sh_y)


def _compute_trunk_lean(ts: dict, i: int) -> Optional[float]:
    for k in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        if not _visible(ts, k, i):
            return None
    hip_mx = (float(ts["left_hip"]["x_px"][i]) + float(ts["right_hip"]["x_px"][i])) / 2.0
    hip_my = (float(ts["left_hip"]["y_px"][i]) + float(ts["right_hip"]["y_px"][i])) / 2.0
    sh_mx  = (float(ts["left_shoulder"]["x_px"][i]) + float(ts["right_shoulder"]["x_px"][i])) / 2.0
    sh_my  = (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    vx = sh_mx - hip_mx
    vy = sh_my - hip_my
    mag = abs(math.degrees(math.atan2(vx, vy)))
    return (1.0 if vx < 0 else -1.0) * mag


def _detect_stance_side(ts: dict, i: int) -> Optional[str]:
    """Returns 'left' or 'right' (stance side = planted leg)."""
    la_vis = _visible(ts, "left_ankle", i, _LIFT_VIS_THRESHOLD)
    ra_vis = _visible(ts, "right_ankle", i, _LIFT_VIS_THRESHOLD)
    if not la_vis or not ra_vis:
        return None
    body_h = _compute_body_height_px(ts, i)
    if not body_h:
        return None
    la_y = float(ts["left_ankle"]["y_px"][i])
    ra_y = float(ts["right_ankle"]["y_px"][i])
    if abs(la_y - ra_y) >= body_h * _LIFT_ONSET_RATIO:
        # Lower image-y means lifted; stance is OTHER side.
        return "right" if la_y < ra_y else "left"
    # Knee fallback.
    if _visible(ts, "left_knee", i, _LIFT_VIS_THRESHOLD) and _visible(ts, "right_knee", i, _LIFT_VIS_THRESHOLD):
        lk_y = float(ts["left_knee"]["y_px"][i])
        rk_y = float(ts["right_knee"]["y_px"][i])
        if abs(lk_y - rk_y) >= body_h * _KNEE_LIFT_RATIO:
            return "right" if lk_y < rk_y else "left"
    return None


def _is_foot_touchdown(ts: dict, i: int, stance: str) -> bool:
    la_vis = _visible(ts, "left_ankle", i)
    ra_vis = _visible(ts, "right_ankle", i)
    if not la_vis or not ra_vis:
        return False
    body_h = _compute_body_height_px(ts, i)
    if not body_h:
        return False
    la_y = float(ts["left_ankle"]["y_px"][i])
    ra_y = float(ts["right_ankle"]["y_px"][i])
    stance_y = la_y if stance == "left" else ra_y
    lifted_y = ra_y if stance == "left" else la_y
    return abs(lifted_y - stance_y) < body_h * _FOOT_TOUCHDOWN_RATIO


def _is_arm_grab(ts: dict, i: int) -> bool:
    for sh_k, wr_k in [
        ("left_shoulder", "left_wrist"),
        ("right_shoulder", "right_wrist"),
    ]:
        if not (_visible(ts, sh_k, i) and _visible(ts, wr_k, i)):
            continue
        vx = float(ts[wr_k]["x_px"][i]) - float(ts[sh_k]["x_px"][i])
        vy = float(ts[wr_k]["y_px"][i]) - float(ts[sh_k]["y_px"][i])
        if math.hypot(vx, vy) < 1e-6:
            continue
        angle = abs(math.degrees(math.atan2(vx, vy)))
        if angle > _ARM_GRAB_DEG:
            return True
    return False


def _sway_path_length(positions: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(positions)):
        total += math.hypot(
            positions[i][0] - positions[i-1][0],
            positions[i][1] - positions[i-1][1],
        )
    return total


def _sway_95_ellipse_area(positions: list[tuple[float, float]]) -> float:
    n = len(positions)
    if n < 3:
        return 0.0
    mx = sum(p[0] for p in positions) / n
    my = sum(p[1] for p in positions) / n
    sxx = syy = sxy = 0.0
    for x, y in positions:
        dx = x - mx
        dy = y - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n; syy /= n; sxy /= n
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, trace * trace / 4.0 - det))
    l1 = trace / 2.0 + disc
    l2 = max(0.0, trace / 2.0 - disc)
    return math.pi * l1 * l2 * 5.991


# ─── Screenshot helper ──────────────────────────────────────────
def _grab_stance_frame(
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

    def _kp(name: str) -> Optional[tuple[int, int]]:
        frames = keypoints_normalized.get(name, [])
        if frame_index >= len(frames):
            return None
        kp = frames[frame_index]
        if kp is None:
            return None
        x_n, y_n, _ = kp
        return (int(x_n * w), int(y_n * h))

    dot_pos: dict[str, tuple[int, int]] = {}
    for name in LM:
        p = _kp(name)
        if p:
            dot_pos[name] = p

    edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            on_side = a.startswith(side) and b.startswith(side)
            color = (255, 255, 255) if on_side else (180, 180, 180)
            cv2.line(frame, dot_pos[a], dot_pos[b], color, 2)
    for name, (px, py) in dot_pos.items():
        emph = name.startswith(side)
        outer = (0, 0, 220) if emph else (150, 150, 150)
        cv2.circle(frame, (px, py), 5, outer, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ─── Main entry point ──────────────────────────────────────────
def analyze_single_leg_stance(
    video_path: str,
    pose_options,
    side: str,
    condition: str,
    patient_age: Optional[int] = None,
) -> dict:
    """Run the Single-Leg Stance pipeline on one (side, condition) trial.

    Args:
        side: 'left' or 'right' — the STANCE leg.
        condition: 'eyes_open' or 'eyes_closed'.
        patient_age: for the age-banded pass/fail norm.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")
    if condition not in ("eyes_open", "eyes_closed"):
        raise ValueError(f"Unsupported condition: {condition!r}")

    raw, fps, _ = extract_poses(video_path, pose_options)
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

    cap_sec = _MAX_EYES_OPEN_SEC if condition == "eyes_open" else _MAX_EYES_CLOSED_SEC
    cap_frames = int(cap_sec * fps)
    onset_timeout_frames = int(_ONSET_TIMEOUT_SEC * fps)
    hop_window_frames = max(1, int(_HOP_WINDOW_MS * fps / 1000.0))

    # Find onset (first frame where stance side matches expected).
    onset_frame = -1
    for i in range(min(n, onset_timeout_frames)):
        if _detect_stance_side(ts, i) == side:
            onset_frame = i
            break

    termination = "max_time"
    hold_frames = 0
    hip_path: list[tuple[float, float]] = []
    trunk_leans: list[float] = []
    stance_ankle_window: list[tuple[int, float]] = []
    screenshot_frame = onset_frame

    if onset_frame < 0:
        termination = "no_lift_detected"
        # Build a degenerate result with hold = 0.
        norm = _get_stance_norm(patient_age, condition == "eyes_closed")
        return {
            "side": side,
            "condition": condition,
            "hold_seconds": 0.0,
            "hold_capped_at": float(cap_sec),
            "termination": termination,
            "norm_threshold_sec": float(norm["threshold_sec"]),
            "norm_band_label": norm["band_label"],
            "norm_comparable": bool(norm["comparable"]),
            "classification": "fail",
            "sway_path_px": 0.0,
            "sway_95_ellipse_px2": 0.0,
            "mean_trunk_lean_deg": 0.0,
            "max_trunk_lean_deg": 0.0,
            "hip_path": [],
            "samples": [],
            "keypoints": [],
            "screenshot_data_url": None,
            "fps": float(fps),
            "total_frames": int(n),
            "interpretation": (
                f"No leg lift detected within the first {_ONSET_TIMEOUT_SEC:.0f} s "
                f"of the clip. Re-record with the patient clearly lifting "
                f"the {'right' if side == 'left' else 'left'} leg."
            ),
        }

    # Hold-phase scan.
    end_frame = onset_frame
    for i in range(onset_frame, n):
        elapsed = i - onset_frame
        if elapsed >= cap_frames:
            termination = "max_time"
            end_frame = i
            break

        # Foot touchdown.
        if _is_foot_touchdown(ts, i, side):
            termination = "foot_touchdown"
            end_frame = i
            break

        # Arm grab.
        if _is_arm_grab(ts, i):
            termination = "arm_grab"
            end_frame = i
            break

        # Hop / stance-foot reposition.
        stance_ankle_key = f"{side}_ankle"
        if _visible(ts, stance_ankle_key, i):
            body_h = _compute_body_height_px(ts, i)
            if body_h:
                stance_y = float(ts[stance_ankle_key]["y_px"][i])
                stance_ankle_window.append((i, stance_y))
                # Drop entries outside the rolling window.
                while stance_ankle_window and i - stance_ankle_window[0][0] > hop_window_frames:
                    stance_ankle_window.pop(0)
                ys = [p[1] for p in stance_ankle_window]
                if ys and (max(ys) - min(ys)) > body_h * _HOP_DISPLACEMENT_RATIO:
                    termination = "hop"
                    end_frame = i
                    break

        hip = _compute_hip_mid(ts, i)
        if hip:
            hip_path.append(hip)
        lean = _compute_trunk_lean(ts, i)
        if lean is not None:
            trunk_leans.append(lean)
        end_frame = i

    hold_frames = end_frame - onset_frame
    hold_seconds = float(hold_frames / fps) if fps > 0 else 0.0

    # Aggregates.
    sway_path = _sway_path_length(hip_path)
    sway_ellipse = _sway_95_ellipse_area(hip_path)
    mean_lean = (sum(trunk_leans) / len(trunk_leans)) if trunk_leans else 0.0
    max_lean = max((abs(v) for v in trunk_leans), default=0.0)

    norm = _get_stance_norm(patient_age, condition == "eyes_closed")
    passed = hold_seconds >= float(norm["threshold_sec"])
    classification = "pass" if passed else "fail"

    # 10 Hz samples + keypoints.
    sample_step = max(1, int(round(fps / _SAMPLE_HZ)))
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in range(onset_frame, end_frame + 1, sample_step):
        t_ms = ((i - onset_frame) / fps) * 1000.0
        hip = _compute_hip_mid(ts, i)
        lean = _compute_trunk_lean(ts, i)
        samples.append({
            "t_ms": float(t_ms),
            "hip_x": float(hip[0]) if hip else None,
            "hip_y": float(hip[1]) if hip else None,
            "trunk_lean_deg": float(lean) if lean is not None else None,
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

    screenshot = _grab_stance_frame(video_path, screenshot_frame, raw, side)

    # Interpretation.
    cond_label = "eyes-open" if condition == "eyes_open" else "eyes-closed"
    side_label = side.capitalize()
    if classification == "pass":
        interp = (
            f"{side_label}-leg stance ({cond_label}): held {hold_seconds:.1f} s "
            f"≥ {norm['threshold_sec']:.1f} s threshold for {norm['band_label']}. "
            f"Passes balance screen for this band."
        )
    else:
        interp = (
            f"{side_label}-leg stance ({cond_label}): held {hold_seconds:.1f} s "
            f"< {norm['threshold_sec']:.1f} s threshold for {norm['band_label']} — "
            f"positive screen for balance impairment / fall risk. "
            f"Termination: {termination.replace('_', ' ')}."
        )
    if not norm["comparable"]:
        interp += " (Norm comparison approximate — patient age unavailable.)"

    return {
        "side": side,
        "condition": condition,
        "hold_seconds": float(hold_seconds),
        "hold_capped_at": float(cap_sec),
        "termination": termination,
        "norm_threshold_sec": float(norm["threshold_sec"]),
        "norm_band_label": norm["band_label"],
        "norm_comparable": bool(norm["comparable"]),
        "classification": classification,
        "sway_path_px": float(sway_path),
        "sway_95_ellipse_px2": float(sway_ellipse),
        "mean_trunk_lean_deg": float(mean_lean),
        "max_trunk_lean_deg": float(max_lean),
        "hip_path": [{"x": float(x), "y": float(y)} for x, y in hip_path],
        "samples": samples,
        "keypoints": keypoints_export,
        "screenshot_data_url": screenshot,
        "fps": float(fps),
        "total_frames": int(n),
        "interpretation": interp,
    }
