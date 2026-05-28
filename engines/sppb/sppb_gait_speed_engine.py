"""sppb_gait_speed_engine.py — backend SPPB Component 2 (4-metre
gait speed) on the MediaPipe BlazePose Full pipeline.

Detects walk-start (ankle X displacement past baseline) and walk-end
(motion stops) from a side-recorded clip and computes:
  duration_sec  — time from detected start to detected stop
  speed_mps     — 4.0 m / duration_sec
  score 0-4     — per Guralnik 1994 SPPB cutoffs

The frontend SPPB orchestrator combines this with the existing
backend Balance endpoint (/api/sppb/balance) and the Sit-to-Stand
endpoint (/api/analyze-sit-to-stand) for Chair Stand to produce the
composite 0-12 SPPB result client-side.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from engines.gait_engine import (
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror sppbGaitSpeed.ts) ───────────────────
_VIS_THRESHOLD = 0.15
_PATH_LENGTH_M = 4.0
_BASELINE_WINDOW_SEC = 1.0
_START_CONFIRM_FRAMES = 5
_STOP_HOLD_SEC = 0.5
_MIN_WALK_DURATION_SEC = 2.5
_START_DISPLACEMENT_RATIO = 0.15
_STOP_VELOCITY_RATIO_PER_SEC = 0.15
_BODY_HEIGHT_FALLBACK_PX = 300.0


def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _compute_ankle_mid_x(ts: dict, i: int) -> Optional[float]:
    la_vis = _visible(ts, "left_ankle", i)
    ra_vis = _visible(ts, "right_ankle", i)
    if la_vis and ra_vis:
        return (float(ts["left_ankle"]["x_px"][i]) + float(ts["right_ankle"]["x_px"][i])) / 2.0
    if la_vis:
        return float(ts["left_ankle"]["x_px"][i])
    if ra_vis:
        return float(ts["right_ankle"]["x_px"][i])
    return None


def _compute_body_height_px(ts: dict, i: int) -> Optional[float]:
    l_sh = _visible(ts, "left_shoulder", i)
    r_sh = _visible(ts, "right_shoulder", i)
    if l_sh and r_sh:
        sh_y = (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    elif l_sh:
        sh_y = float(ts["left_shoulder"]["y_px"][i])
    elif r_sh:
        sh_y = float(ts["right_shoulder"]["y_px"][i])
    else:
        return None
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
    h = abs(ankle_y - sh_y)
    return h if h > 1.0 else None


def _score_gait_speed(speed_mps: Optional[float]) -> int:
    if speed_mps is None or speed_mps <= 0:
        return 0
    if speed_mps < 0.43:  return 1
    if speed_mps < 0.60:  return 2
    if speed_mps < 0.77:  return 3
    return 4


# ─── Main entry point ──────────────────────────────────────────
def analyze_sppb_gait_speed(
    video_path: str,
    pose_options,
) -> dict:
    """Analyze a 4-metre gait-speed clip. Mirrors the live-mode
    stepGaitTrial() state machine: baseline → walking → done."""
    raw, fps, _ = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n = int(min(
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    baseline_window_frames = int(_BASELINE_WINDOW_SEC * fps)
    stop_hold_frames = int(_STOP_HOLD_SEC * fps)
    min_walk_duration_frames = int(_MIN_WALK_DURATION_SEC * fps)

    # ── Phase 1: baseline window ─────────────────────────────
    baseline_samples: list[float] = []
    body_h_samples: list[float] = []
    first_valid = -1
    for i in range(n):
        ax = _compute_ankle_mid_x(ts, i)
        if ax is None:
            continue
        if first_valid < 0:
            first_valid = i
        baseline_samples.append(ax)
        bh = _compute_body_height_px(ts, i)
        if bh is not None:
            body_h_samples.append(bh)
        if len(baseline_samples) >= baseline_window_frames:
            break

    if not baseline_samples or first_valid < 0:
        raise ValueError("poor_visibility")

    baseline_x = sum(baseline_samples) / len(baseline_samples)
    body_h = sorted(body_h_samples)[len(body_h_samples) // 2] if len(body_h_samples) >= 3 else _BODY_HEIGHT_FALLBACK_PX
    start_displacement_px = body_h * _START_DISPLACEMENT_RATIO
    stop_velocity_px_per_sec = body_h * _STOP_VELOCITY_RATIO_PER_SEC

    # ── Phase 2: walk-start detection ────────────────────────
    baseline_end_frame = first_valid + len(baseline_samples)
    start_frame = -1
    confirm_count = 0
    for i in range(baseline_end_frame, n):
        ax = _compute_ankle_mid_x(ts, i)
        if ax is None:
            confirm_count = 0
            continue
        if abs(ax - baseline_x) >= start_displacement_px:
            confirm_count += 1
            if confirm_count >= _START_CONFIRM_FRAMES:
                start_frame = i - _START_CONFIRM_FRAMES + 1
                break
        else:
            confirm_count = 0

    if start_frame < 0:
        raise ValueError(
            "No walk start detected. Ensure the clip begins with the "
            "patient standing still and then walking 4 metres at usual pace."
        )

    # ── Phase 3: walking → done detection ────────────────────
    end_frame = -1
    prev_x: Optional[float] = None
    prev_i: Optional[int] = None
    stop_window_start: Optional[int] = None

    for i in range(start_frame, n):
        ax = _compute_ankle_mid_x(ts, i)
        if ax is None:
            continue
        elapsed_frames = i - start_frame
        if prev_x is not None and prev_i is not None and i > prev_i:
            dt_sec = (i - prev_i) / fps
            if dt_sec > 0:
                vel = abs(ax - prev_x) / dt_sec
                if vel < stop_velocity_px_per_sec and elapsed_frames > min_walk_duration_frames:
                    if stop_window_start is None:
                        stop_window_start = i
                    elif i - stop_window_start >= stop_hold_frames:
                        end_frame = i
                        break
                else:
                    stop_window_start = None
        prev_x = ax
        prev_i = i

    if end_frame < 0:
        # Clip ran out before motion stopped. Treat last valid frame
        # as the end (manual fallback behaviour from the live state).
        end_frame = n - 1

    duration_sec = (end_frame - start_frame) / fps
    speed_mps = (_PATH_LENGTH_M / duration_sec) if duration_sec > 0 else 0.0
    score = _score_gait_speed(speed_mps)

    interpretation = (
        f"4-metre walk completed in {duration_sec:.2f} s → "
        f"{speed_mps:.2f} m/s (SPPB score {score}/4)."
    )

    return {
        "duration_sec": float(duration_sec),
        "speed_mps": float(speed_mps),
        "score": int(score),
        "completed": True,
        "started_at_ms": 0,
        "fps": float(fps),
        "total_frames": int(n),
        "interpretation": interpretation,
    }
