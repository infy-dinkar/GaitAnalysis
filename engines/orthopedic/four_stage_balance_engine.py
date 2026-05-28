"""four_stage_balance_engine.py — backend 4-Stage Balance test
(Test C4) on the MediaPipe BlazePose Full (33-keypoint) pipeline.

Per-stage analysis: each video covers ONE of stages 1-4. The
frontend uploads stages in parallel via Promise.allSettled and
applies the stop-at-first-failure rule client-side.
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
_STAGE_HOLD_SEC = 10.0
_POSITION_LOCK_FRAMES_FRAC = 0.6   # 600 ms × fps
_POSITION_DRIFT_GRACE_FRAMES_FRAC = 0.8  # 800 ms × fps
_POSITION_TIMEOUT_SEC = 12.0
_MIN_ANKLE_SEPARATION_PX = 8.0

# Stage geometry validators (raw px — mirror fourStageBalance.ts).
_S1_Y_MAX = 20.0
_S1_X_MIN = 20.0
_S1_X_MAX = 80.0
_S2_Y_MIN = 15.0
_S2_Y_MAX = 60.0
_S2_X_MAX = 30.0
_S3_X_MAX = 15.0
_S3_Y_MIN = 40.0
_S4_LIFT_RATIO = 0.06
_S4_KNEE_LIFT_RATIO = 0.10

_ARM_GRAB_DEG = 45.0
_FOOT_TOUCHDOWN_PX = 30.0


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _read_ankles(ts: dict, i: int) -> Optional[dict]:
    if not (_visible(ts, "left_ankle", i) and _visible(ts, "right_ankle", i)):
        return None
    lx = float(ts["left_ankle"]["x_px"][i]); ly = float(ts["left_ankle"]["y_px"][i])
    rx = float(ts["right_ankle"]["x_px"][i]); ry = float(ts["right_ankle"]["y_px"][i])
    return {
        "lx": lx, "ly": ly, "rx": rx, "ry": ry,
        "dx_abs": abs(lx - rx),
        "dy_abs": abs(ly - ry),
    }


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


def _compute_hip_mid(ts: dict, i: int) -> Optional[tuple[float, float]]:
    if not (_visible(ts, "left_hip", i) and _visible(ts, "right_hip", i)):
        return None
    return (
        (float(ts["left_hip"]["x_px"][i]) + float(ts["right_hip"]["x_px"][i])) / 2.0,
        (float(ts["left_hip"]["y_px"][i]) + float(ts["right_hip"]["y_px"][i])) / 2.0,
    )


# ─── Stage-position validators ──────────────────────────────────
def _is_stage1(a: dict) -> bool:
    return a["dy_abs"] < _S1_Y_MAX and _S1_X_MIN < a["dx_abs"] < _S1_X_MAX


def _is_stage2(a: dict) -> bool:
    return _S2_Y_MIN < a["dy_abs"] < _S2_Y_MAX and a["dx_abs"] < _S2_X_MAX


def _is_stage3(a: dict) -> bool:
    return a["dx_abs"] < _S3_X_MAX and a["dy_abs"] > _S3_Y_MIN


def _detect_stage4_stance(a: dict, ts: dict, i: int) -> Optional[str]:
    body_h = _compute_body_height_px(ts, i)
    if not body_h:
        return None
    if a["dy_abs"] >= body_h * _S4_LIFT_RATIO:
        return "right" if a["ly"] < a["ry"] else "left"
    if _visible(ts, "left_knee", i) and _visible(ts, "right_knee", i):
        lk = float(ts["left_knee"]["y_px"][i])
        rk = float(ts["right_knee"]["y_px"][i])
        if abs(lk - rk) >= body_h * _S4_KNEE_LIFT_RATIO:
            return "right" if lk < rk else "left"
    return None


def _is_stage_position(stage: int, a: dict, ts: dict, i: int) -> bool:
    if a["dx_abs"] < _MIN_ANKLE_SEPARATION_PX and stage in (1, 2):
        return False
    if stage == 1: return _is_stage1(a)
    if stage == 2: return _is_stage2(a)
    if stage == 3: return _is_stage3(a)
    if stage == 4: return _detect_stage4_stance(a, ts, i) is not None
    return False


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
        if abs(math.degrees(math.atan2(vx, vy))) > _ARM_GRAB_DEG:
            return True
    return False


def _is_stage4_foot_touchdown(ts: dict, i: int, stance: str) -> bool:
    if not (_visible(ts, "left_ankle", i) and _visible(ts, "right_ankle", i)):
        return False
    la_y = float(ts["left_ankle"]["y_px"][i])
    ra_y = float(ts["right_ankle"]["y_px"][i])
    stance_y = la_y if stance == "left" else ra_y
    lifted_y = ra_y if stance == "left" else la_y
    return abs(lifted_y - stance_y) < _FOOT_TOUCHDOWN_PX


def _sway_path_length(positions: list[tuple[float, float]]) -> float:
    total = 0.0
    for j in range(1, len(positions)):
        total += math.hypot(
            positions[j][0] - positions[j-1][0],
            positions[j][1] - positions[j-1][1],
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
def _grab_stage_frame(
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
            cv2.line(frame, dot_pos[a], dot_pos[b], (255, 255, 255), 2)
    for _name, (px, py) in dot_pos.items():
        cv2.circle(frame, (px, py), 5, (0, 0, 220), -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ─── Main entry point ──────────────────────────────────────────
def analyze_four_stage_balance(
    video_path: str,
    pose_options,
    stage: int,
) -> dict:
    """Analyze ONE stage (1-4) from an uploaded clip. Returns a
    StageResult-shaped dict matching the frontend type."""
    if stage not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported stage: {stage!r}")

    raw, fps, _ = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n = int(min(
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    position_lock_frames = max(1, int(_POSITION_LOCK_FRAMES_FRAC * fps))
    drift_grace_frames = max(1, int(_POSITION_DRIFT_GRACE_FRAMES_FRAC * fps))
    position_timeout_frames = int(_POSITION_TIMEOUT_SEC * fps)
    stage_hold_frames = int(_STAGE_HOLD_SEC * fps)

    # Phase 1 — position lock.
    lock_start = -1
    continuous_count = 0
    for i in range(min(n, position_timeout_frames)):
        a = _read_ankles(ts, i)
        if a is None:
            continuous_count = 0
            continue
        if _is_stage_position(stage, a, ts, i):
            continuous_count += 1
            if continuous_count >= position_lock_frames:
                lock_start = i - position_lock_frames + 1
                break
        else:
            continuous_count = 0

    if lock_start < 0:
        return {
            "stage": stage,
            "outcome": "fail",
            "hold_seconds": 0.0,
            "failure_mode": "position_lost",
            "sway_path_px": 0.0,
            "sway_95_ellipse_px2": 0.0,
            "hip_path": [],
            "samples": [],
            "keypoints": [],
            "screenshot_data_url": None,
            "duration_seconds": 0.0,
            "fps": float(fps),
            "total_frames": int(n),
        }

    # Stage 4 — capture stance side at lock time.
    stage4_stance: Optional[str] = None
    if stage == 4:
        a = _read_ankles(ts, lock_start)
        if a:
            stage4_stance = _detect_stage4_stance(a, ts, lock_start)

    # Phase 2 — hold.
    hold_start = lock_start + position_lock_frames
    hold_end = hold_start
    failure_mode: Optional[str] = None
    last_valid_pos = hold_start
    hip_path: list[tuple[float, float]] = []

    for i in range(hold_start, n):
        elapsed = i - hold_start
        if elapsed >= stage_hold_frames:
            hold_end = i
            break

        if _is_arm_grab(ts, i):
            failure_mode = "arm_grab"
            hold_end = i
            break
        if stage == 4 and stage4_stance and _is_stage4_foot_touchdown(ts, i, stage4_stance):
            failure_mode = "foot_touchdown"
            hold_end = i
            break

        a = _read_ankles(ts, i)
        if a is not None:
            if _is_stage_position(stage, a, ts, i):
                last_valid_pos = i
            elif i - last_valid_pos > drift_grace_frames:
                failure_mode = "position_lost"
                hold_end = i
                break

        hip = _compute_hip_mid(ts, i)
        if hip:
            hip_path.append(hip)
        hold_end = i

    hold_frames = hold_end - hold_start
    hold_seconds = float(hold_frames / fps) if fps > 0 else 0.0
    outcome = "pass" if failure_mode is None and hold_frames >= stage_hold_frames else "fail"
    if failure_mode is None and outcome == "fail":
        failure_mode = "position_lost"

    # Sample export.
    sample_step = max(1, int(round(fps / _SAMPLE_HZ)))
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for i in range(hold_start, hold_end + 1, sample_step):
        t_ms = ((i - hold_start) / fps) * 1000.0
        a = _read_ankles(ts, i)
        hip = _compute_hip_mid(ts, i)
        samples.append({
            "t_ms": float(t_ms),
            "hip_x": float(hip[0]) if hip else None,
            "hip_y": float(hip[1]) if hip else None,
            "ankle_l_x": a["lx"] if a else None,
            "ankle_l_y": a["ly"] if a else None,
            "ankle_r_x": a["rx"] if a else None,
            "ankle_r_y": a["ry"] if a else None,
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

    screenshot = _grab_stage_frame(video_path, hold_start, raw)

    return {
        "stage": int(stage),
        "outcome": outcome,
        "hold_seconds": min(float(_STAGE_HOLD_SEC), hold_seconds),
        "failure_mode": failure_mode,
        "sway_path_px": float(_sway_path_length(hip_path)),
        "sway_95_ellipse_px2": float(_sway_95_ellipse_area(hip_path)),
        "hip_path": [{"x": float(x), "y": float(y)} for x, y in hip_path],
        "samples": samples,
        "keypoints": keypoints_export,
        "screenshot_data_url": screenshot,
        "duration_seconds": float(hold_seconds),
        "fps": float(fps),
        "total_frames": int(n),
    }
