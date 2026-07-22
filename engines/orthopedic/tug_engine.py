"""tug_engine.py — Timed Up and Go (TUG) analysis pipeline.

End-to-end TUG processing:
  1. Reuse gait_engine.extract_poses() to get MediaPipe BlazePose Full
     33-keypoint landmarks per frame (model: pose_landmarker_full.task).
  2. Reuse gait_engine.build_time_series() to smooth + interpolate.
  3. Run TUG-specific phase detection (5 sub-phases).
  4. Compute per-phase metrics (duration, step count, length, cadence,
     walking speed).
  5. Generate annotated screenshots at the 5 key moments.
  6. Build classification + flags + interpretation.

This module reuses the gait module's MediaPipe pipeline rather than
duplicating it — same model file, same smoothing helpers, same step
detection from foot-strike peaks. Only the phase boundary detection
and TUG-specific metrics are new.

The video file is opened twice: once for pose extraction (via the
shared `extract_poses`) and a second short pass with cv2.VideoCapture
to grab the 5 key frames for screenshot generation. The temp file
is deleted by the caller (api.py) in a `finally` block.
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

import os

from engines.gait_engine import (
    LM,
    build_time_series,
    extract_poses,
)
from engines.orthopedic.tug_interpretation import (
    AGE_NORMS,
    age_matched_norm,
    build_interpretation,
    classify_total_time,
    compute_flags,
)
from engines.orthopedic.tug_models import (
    TUGFlag,
    TUGKeyFrame,
    TUGPhase,
    TUGResult,
)

log = logging.getLogger(__name__)


# ─── Phase-detection thresholds (from MotionLens TUG spec) ──────
PATH_LENGTH_M = 3.0
WALK_VELOCITY_STOP_PX_PER_SEC = 50.0
TURN_ENTRY_DEG_PER_SEC = 60.0
TURN_EXIT_DEG_PER_SEC = 30.0
KNEE_EXTENDED_DEG = 170.0
ANKLE_SEPARATION_FIRST_STEP_PX = 20.0
SETTLED_HOLD_FRAMES_THRESHOLD = 5      # consecutive frames below threshold
SIT_BASELINE_TOLERANCE_PX = 20.0
MOTION_STOP_DURATION_SEC = 0.5
MIN_PHASE_DURATION_SEC = 0.3           # safety guard against zero-length phases


# ─── Geometry helpers ───────────────────────────────────────────
def _interior_angle_px(ts: dict, hip_key: str, knee_key: str, ankle_key: str) -> np.ndarray:
    """Per-frame interior angle at the knee (hip-knee-ankle), in degrees.
    Uses smoothed pixel coords. Returns array len = number of frames.
    180° = fully extended."""
    hx = ts[hip_key]["x_px"];   hy = ts[hip_key]["y_px"]
    kx = ts[knee_key]["x_px"];  ky = ts[knee_key]["y_px"]
    ax = ts[ankle_key]["x_px"]; ay = ts[ankle_key]["y_px"]
    # Vectors at the knee: knee→hip and knee→ankle
    ux = hx - kx; uy = hy - ky
    vx = ax - kx; vy = ay - ky
    dot = ux * vx + uy * vy
    mag = np.sqrt(ux * ux + uy * uy) * np.sqrt(vx * vx + vy * vy)
    cos_theta = np.where(mag > 0, dot / mag, 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _hip_midpoint_xy(ts: dict) -> tuple[np.ndarray, np.ndarray]:
    lx = ts["left_hip"]["x_px"];  ly = ts["left_hip"]["y_px"]
    rx = ts["right_hip"]["x_px"]; ry = ts["right_hip"]["y_px"]
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def _shoulder_line_angle_deg(ts: dict) -> np.ndarray:
    """Per-frame angle of the right-shoulder→left-shoulder vector from
    the image's horizontal axis, in degrees. Used to detect the 180°
    body rotation that defines the turn phase."""
    lx = ts["left_shoulder"]["x_px"];  ly = ts["left_shoulder"]["y_px"]
    rx = ts["right_shoulder"]["x_px"]; ry = ts["right_shoulder"]["y_px"]
    return np.degrees(np.arctan2(ly - ry, lx - rx))


def _velocity(arr: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference velocity (units / second). Pads ends with the
    first/last finite-difference values."""
    if len(arr) < 2:
        return np.zeros_like(arr)
    v = np.gradient(arr) * fps
    return v


def _unwrap_angle_deg(angles: np.ndarray) -> np.ndarray:
    """Unwrap a degree-valued angle series so cumulative rotation can
    be tracked without ±180° discontinuities."""
    rad = np.unwrap(np.radians(angles))
    return np.degrees(rad)


# ─── Phase boundary detection ───────────────────────────────────
def _find_phase_boundaries(
    ts: dict,
    fps: float,
    total_frames: int,
) -> tuple[dict[str, int], dict[str, bool]]:
    """Return ({phase_end_frame_map}, {phase_flag_map}).

    phase_end_frame_map keys: 'A_end', 'B_end', 'C_end', 'D_end', 'E_end'
    phase_flag_map: { 'truncated_A': bool, ..., 'turn_undetected': bool }

    Algorithm rewrite (side-view robustness):

    The previous version anchored turn detection on the shoulder
    line's angular velocity. In a true side-view TUG the left and
    right shoulders project to the SAME image-X column — the
    "shoulder line" vector is tiny and atan2 of two near-zero
    components is dominated by noise. That made shoulder_ang_vel
    spike continuously during normal walking, firing the turn
    detector immediately and collapsing walk_out to a few frames.

    The new approach uses the hip-midpoint X trajectory as the
    primary signal — robust in side view because the hip-mid sweeps
    a clean monotone arc during each walk phase:

      * Walk-out  → hip_x moves monotonically in one direction
      * Turn      → hip_x is at its global extremum (max or min)
      * Walk-back → hip_x moves monotonically in the opposite direction

    Turn center = argmax (or argmin) of smoothed hip_x within the
    walking region. We carve a small turn window around that frame
    and let walk_out / walk_back fill the surrounding intervals.

    Sit-to-stand / stand-to-sit are still detected from hip_y +
    knee extension, but now degrade gracefully when the patient was
    already standing at frame 0 (e.g. operator started recording
    after the stand) — those phases collapse to 0-frame windows
    instead of breaking the math.
    """
    # IMPORTANT: use the ACTUAL time-series length, not cv2's reported
    # frame-count metadata. cv2.get(CAP_PROP_FRAME_COUNT) often
    # over-reports by 1-2 frames (especially on re-encoded WebM /
    # MediaRecorder files) while the per-frame arrays in `ts` only
    # cover frames that were actually decoded via cap.read(). Using
    # the metadata value here would let `range(..., n)` index one past
    # the end of every keypoint array and crash with
    # `index N is out of bounds for axis 0 with size N`.
    n = int(len(ts["left_hip"]["x_px"]))
    flags = {
        "truncated_A": False, "truncated_B": False, "truncated_C": False,
        "truncated_D": False, "truncated_E": False, "turn_undetected": False,
    }
    if n < int(fps * 1.0):  # need at least 1 second
        ends = {
            "A_end": 0,
            "B_end": max(0, n // 3),
            "C_end": max(1, n // 2),
            "D_end": max(2, (2 * n) // 3),
            "E_end": max(3, n - 1),
        }
        for k in flags:
            flags[k] = True
        return ends, flags

    hip_x, hip_y = _hip_midpoint_xy(ts)
    knee_l = _interior_angle_px(ts, "left_hip", "left_knee", "left_ankle")
    knee_r = _interior_angle_px(ts, "right_hip", "right_knee", "right_ankle")
    knee_mean = (knee_l + knee_r) / 2.0

    # ── A end: sit-to-stand → walk-out ─────────────────────────
    # Patient is upright when hip_y reaches its first local minimum
    # AND knees are mostly extended. If the patient was ALREADY
    # standing at frame 0 (e.g. operator started recording mid-walk),
    # there's no chair transition to detect — collapse Phase A.
    initial_hip_y = float(np.nanmean(hip_y[: max(1, int(fps * 0.3))]))
    early_min_hip_y = float(np.nanmin(hip_y[: min(n, int(fps * 6))]))
    sit_drop_px = initial_hip_y - early_min_hip_y
    already_standing = sit_drop_px < 30.0 and float(knee_mean[0]) >= 160.0

    a_end: int
    if already_standing:
        a_end = 1  # collapse Phase A to a single frame
    else:
        a_end = -1
        look_back = max(3, int(fps * 0.1))
        for t in range(look_back, min(n - look_back, int(fps * 6))):
            if (
                hip_y[t - look_back] > hip_y[t]
                and hip_y[t + look_back] > hip_y[t]
                and knee_mean[t] >= KNEE_EXTENDED_DEG - 5.0
            ):
                a_end = t
                break
        if a_end < 0:
            # Fallback: first frame where hip_y has dropped most of
            # the way and knees are extended.
            thresh = early_min_hip_y + 0.20 * (initial_hip_y - early_min_hip_y)
            for t in range(look_back, min(n, int(fps * 6))):
                if hip_y[t] <= thresh and knee_mean[t] >= KNEE_EXTENDED_DEG * 0.95:
                    a_end = t
                    break
        if a_end < 0:
            a_end = max(1, int(fps * 0.5))
            flags["truncated_A"] = True
    a_end = max(1, min(a_end, n - 4))

    # ── Turn center: hip_x extremum within the walking region ─
    # Smooth hip_x heavily so jitter doesn't produce false extrema.
    # We look in the region [a_end, n - 1] because before a_end the
    # patient may still be standing up.
    walking = hip_x[a_end:]
    if len(walking) < 5:
        # Not enough walking footage — bail with truncated phases
        e_end = n - 1
        d_end = e_end - 1
        c_end = max(a_end + 2, (a_end + e_end) // 2)
        b_end = max(a_end + 1, c_end - 1)
        flags["truncated_B"] = flags["truncated_C"] = True
        flags["truncated_D"] = flags["truncated_E"] = True
        flags["turn_undetected"] = True
        boundaries = {"A_end": a_end, "B_end": b_end, "C_end": c_end, "D_end": d_end, "E_end": e_end}
        return _clamp_boundaries(boundaries, n, fps), flags

    # Direction of walking out: positive = patient moves toward larger
    # image-X; negative = toward smaller image-X. Determined by the
    # net displacement over the first half-second of walking.
    early_window = max(3, int(fps * 0.5))
    initial_x = float(np.nanmean(walking[:early_window]))
    later_x = (
        float(np.nanmean(walking[early_window : early_window * 2]))
        if len(walking) > early_window * 2
        else float(walking[-1])
    )
    direction = 1 if later_x > initial_x else -1

    if direction > 0:
        turn_center_local = int(np.argmax(walking))
    else:
        turn_center_local = int(np.argmin(walking))
    turn_center = a_end + turn_center_local

    # Carve out a turn window — ±0.4 s around the extremum. If the
    # extremum is at the very end of the video, the patient never
    # turned back (no walk-back) — mark turn_undetected.
    turn_half = max(int(fps * 0.4), 2)
    b_end = max(a_end + 1, turn_center - turn_half)
    c_end_candidate = min(n - 1, turn_center + turn_half)

    # ── D end: walk-back → stand-to-sit ────────────────────────
    # Walk-back ends when hip_x returns near its initial value AND
    # the patient starts descending (hip_y rising). If the patient
    # never returns to chair / never sits, d_end clamps near video end.
    initial_walk_x = float(np.nanmean(hip_x[: max(1, int(fps * 0.3))]))
    d_end = -1
    if c_end_candidate < n - 1:
        # Threshold for "near chair" — 15% of total walking distance
        excursion = abs(float(walking[turn_center_local]) - initial_walk_x)
        near_chair_thresh = max(50.0, excursion * 0.20)
        late_hip_y_baseline = float(np.nanmean(hip_y[c_end_candidate:c_end_candidate + max(3, int(fps * 0.3))]))
        for t in range(c_end_candidate + 1, n):
            near_chair = abs(float(hip_x[t]) - initial_walk_x) < near_chair_thresh
            descending = hip_y[t] > late_hip_y_baseline + 15.0
            if near_chair or descending:
                d_end = t
                break
    if d_end < 0:
        d_end = n - 1
        flags["truncated_D"] = True
    d_end = min(d_end, n - 1)

    # If the turn window's right edge was past the video end, c_end
    # collapses with d_end and we flag turn_undetected.
    c_end = min(c_end_candidate, d_end - 1) if d_end > c_end_candidate + 1 else c_end_candidate
    c_end = max(b_end + 1, c_end)
    if c_end >= n - 1:
        flags["turn_undetected"] = True

    # ── E end: test end (back to sitting baseline) ────────────
    # If the patient never sits (operator stops mid-walk-back), E
    # collapses with d_end.
    e_end = -1
    motion_stop_frames = max(int(MOTION_STOP_DURATION_SEC * fps), 3)
    if d_end < n - 1:
        for t in range(d_end + 1, n):
            near_baseline = abs(float(hip_y[t]) - initial_hip_y) < SIT_BASELINE_TOLERANCE_PX
            win_start = max(d_end, t - motion_stop_frames)
            win_y = hip_y[win_start : t + 1]
            stable = (
                (np.nanmax(win_y) - np.nanmin(win_y)) < SIT_BASELINE_TOLERANCE_PX
                if len(win_y) > 1
                else False
            )
            if near_baseline and stable:
                e_end = t
                break
    if e_end < 0:
        e_end = n - 1
        if d_end >= n - 2:
            flags["truncated_E"] = True
    e_end = min(e_end, n - 1)

    boundaries = {"A_end": a_end, "B_end": b_end, "C_end": c_end, "D_end": d_end, "E_end": e_end}
    return _clamp_boundaries(boundaries, n, fps), flags


def _clamp_boundaries(boundaries: dict[str, int], n: int, fps: float) -> dict[str, int]:
    """Enforce strictly monotonic boundaries with a minimum per-phase
    duration. Nudges forward as needed; clamps to n - 1."""
    min_frames = max(int(MIN_PHASE_DURATION_SEC * fps), 1)
    prev = 0
    for key in ("A_end", "B_end", "C_end", "D_end", "E_end"):
        if boundaries[key] <= prev:
            boundaries[key] = min(prev + min_frames, n - 1)
        if boundaries[key] >= n:
            boundaries[key] = n - 1
        prev = boundaries[key]
    return boundaries


# ─── Step detection within a phase ──────────────────────────────
def _strikes_in_range(
    ts: dict,
    fps: float,
    start: int,
    end: int,
    is_turn_phase: bool = False,
) -> dict[str, list[int]]:
    """Per-leg foot strikes (frame indices) inside [start, end).

    Two algorithms, chosen by phase:

    WALK phases (walk-out, walk-back): heel-separation peaks
      In side view, MediaPipe tracks the camera-facing foot well
      but the FAR-side foot is repeatedly occluded — its Y trace
      gets interpolated into a near-flat line and `find_peaks` on
      it returns nothing. That's exactly why a previous version of
      this code dropped one side's strikes for each walk phase.
      The original gait engine solved this with the per-frame
      heel-separation signal `|L_heel_x - R_heel_x|`: peaks of
      separation correspond to the moments when one foot is
      maximally forward = a heel strike. Because separation is a
      DIFFERENCE, it stays clean as long as ONE heel is visible.
      L vs R is then assigned by checking which heel was forward
      in the walking direction at the peak frame.

    TURN phase: per-foot foot-index Y peaks
      Heel-separation works only when the patient is TRANSLATING.
      During the turn the patient pivots in place — heel separation
      barely changes. We fall back to per-foot foot-index Y peaks
      with a generous prominence threshold to catch shuffle steps.
    """
    min_phase_frames = int(fps * 0.4) * 2
    if end - start < min_phase_frames:
        return {"left": [], "right": []}

    if is_turn_phase:
        return _strikes_per_foot_y(ts, fps, start, end)
    return _strikes_from_heel_separation(ts, fps, start, end)


def _strikes_from_heel_separation(
    ts: dict,
    fps: float,
    start: int,
    end: int,
) -> dict[str, list[int]]:
    """Heel-separation peak detection (walk phases only). Robust to
    one-foot occlusion in side view because it uses a DIFFERENCE
    signal — only one heel needs to track cleanly."""
    l_heel_x = np.asarray(ts["left_heel"]["x_px"][start:end], dtype=float)
    r_heel_x = np.asarray(ts["right_heel"]["x_px"][start:end], dtype=float)
    if len(l_heel_x) < 5:
        return {"left": [], "right": []}

    sep = np.abs(l_heel_x - r_heel_x)
    rng = float(np.ptp(sep))
    prom = max(rng * 0.10, 1.0)
    distance = max(int(fps * 0.3), 3)
    peaks, _ = find_peaks(sep, distance=distance, prominence=prom)

    # Walking direction within this phase: sign of net hip-mid X
    # displacement across the phase. +1 = patient moved toward larger
    # x in image; -1 = toward smaller x. The leading foot at a strike
    # is the one whose heel_x is further in the walking direction.
    lh = ts["left_hip"]["x_px"][start:end]
    rh = ts["right_hip"]["x_px"][start:end]
    hip_x = (np.asarray(lh, dtype=float) + np.asarray(rh, dtype=float)) / 2.0
    if len(hip_x) >= 2:
        direction = 1 if (hip_x[-1] - hip_x[0]) >= 0 else -1
    else:
        direction = 1

    L_strikes: list[int] = []
    R_strikes: list[int] = []
    for peak in peaks:
        global_frame = int(start + peak)
        # signed_diff > 0 → left heel is forward in walking direction
        signed_diff = (l_heel_x[peak] - r_heel_x[peak]) * direction
        if signed_diff > 0:
            L_strikes.append(global_frame)
        else:
            R_strikes.append(global_frame)
    return {"left": L_strikes, "right": R_strikes}


def _strikes_per_foot_y(
    ts: dict,
    fps: float,
    start: int,
    end: int,
) -> dict[str, list[int]]:
    """Per-foot Y peaks (turn phase only). Catches small shuffle
    steps where heel separation barely changes."""
    distance = max(int(fps * 0.15), 2)
    kernel = np.ones(5) / 5.0
    out: dict[str, list[int]] = {"left": [], "right": []}
    for side, key in (("left", "left_foot_index"), ("right", "right_foot_index")):
        y_slice = np.asarray(ts[key]["y_px"][start:end], dtype=float)
        if len(y_slice) < 5:
            continue
        y_smooth = np.convolve(y_slice, kernel, mode="same")
        rng = float(np.ptp(y_smooth))
        prom = max(rng * 0.04, 1.5)
        peaks, _ = find_peaks(y_smooth, distance=distance, prominence=prom)
        out[side] = [int(start + p) for p in peaks]
    return out


def _step_length_px(ts: dict, side: str, strikes: list[int]) -> Optional[float]:
    """Mean foot-index step length in pixels between consecutive same-side strikes."""
    if len(strikes) < 2:
        return None
    foot_key = "left_foot_index" if side == "left" else "right_foot_index"
    xs = ts[foot_key]["x_px"]
    diffs = [abs(float(xs[strikes[i + 1]]) - float(xs[strikes[i]])) for i in range(len(strikes) - 1)]
    if not diffs:
        return None
    return float(np.mean(diffs))


# ─── Annotated screenshots ──────────────────────────────────────
def _grab_key_frame(
    cap: cv2.VideoCapture,
    frame_index: int,
    keypoints_normalized: dict[str, list],
    label: str,
) -> Optional[TUGKeyFrame]:
    """Seek to `frame_index` in the open VideoCapture, draw the
    skeleton overlay on the frame, and return a JPEG data URL."""
    if frame_index < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]

    # Resize so the embedded image isn't huge in MongoDB
    target_w = min(640, w)
    if target_w < w:
        scale = target_w / w
        frame = cv2.resize(frame, (target_w, int(h * scale)))
        h, w = frame.shape[:2]
    else:
        scale = 1.0

    # Skeleton overlay — use the normalized keypoints from MediaPipe
    # (raw extract dict, not the smoothed time-series). Visibility-gate.
    def _draw_dot(name: str) -> Optional[tuple[int, int]]:
        frames = keypoints_normalized.get(name, [])
        if frame_index >= len(frames):
            return None
        kp = frames[frame_index]
        if kp is None:
            return None
        x_n, y_n, _vis = kp
        px = int(x_n * w)
        py = int(y_n * h)
        cv2.circle(frame, (px, py), 4, (0, 0, 220), -1)
        cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)
        return (px, py)

    edges = [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
        ("left_hip",       "left_knee"),
        ("left_knee",      "left_ankle"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
        ("left_ankle",     "left_heel"),
        ("right_ankle",    "right_heel"),
        ("left_heel",      "left_foot_index"),
        ("right_heel",     "right_foot_index"),
    ]
    dot_pos: dict[str, tuple[int, int]] = {}
    for name in LM:
        p = _draw_dot(name)
        if p:
            dot_pos[name] = p
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            cv2.line(frame, dot_pos[a], dot_pos[b], (255, 255, 255), 2)

    # Label badge
    cv2.putText(
        frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return TUGKeyFrame(
        label=label,  # type: ignore[arg-type]
        frame_index=frame_index,
        image_data_url=f"data:image/jpeg;base64,{b64}",
    )


# ─── Video preprocessing ────────────────────────────────────────
def _ensure_decodable_video(
    video_path: str,
    recording_duration_ms: Optional[int],
    force_rewrite: bool = False,
) -> tuple[str, Optional[str]]:
    """Some MediaRecorder-produced WebM containers have broken or
    missing duration metadata. There are TWO failure modes we need
    to handle, not one:

      A. cv2's CAP_PROP_FPS returns 0. Easy to detect — historically
         the only case this helper covered.
      B. cv2's CAP_PROP_FPS returns a plausible value (e.g. 30) but
         the container is internally broken so only a small subset
         of frames actually decode. cv2.CAP_PROP_FRAME_COUNT may
         report a low number, or it may report a high number but
         most frame reads fail. Symptoms downstream: extract_poses
         produces far fewer time-series entries than expected and
         the engine raises "Recording is too short". This is the
         case the SPPB balance flow has been hitting.

    Strategy:
      1. Probe FPS + total_frames from the container.
      2. If FPS == 0 → rewrite (case A).
      3. If FPS > 0 AND we have a client recording duration, compare
         the container-claimed duration (frames / fps) to the
         client-reported duration. If they disagree by more than
         ~30 %, the container is lying — rewrite (case B).
      4. Rewrite path: stream-read all decodable frames, compute
         effective FPS = decoded_frames / client_duration, write a
         new MP4 with a clean header.

    Returns:
      tuple[str, Optional[str]] — (path_to_use, path_to_cleanup_or_None)
    """
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    needs_rewrite = False
    rewrite_reason = ""

    if force_rewrite and recording_duration_ms and recording_duration_ms > 0:
        # Caller opted into unconditional normalisation (gait record
        # mode): MediaRecorder WebM metadata is never trusted, every
        # recorded clip is re-encoded to a clean constant-FPS MP4
        # whose FPS = decoded_frames / client wall-clock duration.
        # The 0.7 mismatch gate below misses the "metadata plausibly
        # wrong" band (e.g. real ~22 fps, claimed 30) which silently
        # skews every temporal gait metric downstream.
        needs_rewrite = True
        rewrite_reason = "forced normalisation (record-mode upload)"
    elif fps <= 0:
        needs_rewrite = True
        rewrite_reason = "cv2 reported FPS = 0"
    elif recording_duration_ms and recording_duration_ms > 0 and total_frames_meta > 0:
        container_duration_ms = (total_frames_meta / fps) * 1000.0
        client_duration_ms = float(recording_duration_ms)
        # If the container thinks the video is much shorter than the
        # client says it actually is, the metadata is lying. 0.7
        # threshold tolerates normal cv2 rounding (~3-5 % difference)
        # but catches the "claims 1 s, actually 30 s" failure mode.
        if container_duration_ms < client_duration_ms * 0.7:
            needs_rewrite = True
            rewrite_reason = (
                f"container claims {container_duration_ms / 1000:.1f}s "
                f"but client recorded {client_duration_ms / 1000:.1f}s"
            )

    if not needs_rewrite:
        return video_path, None

    if not recording_duration_ms or recording_duration_ms <= 0:
        # No fallback duration available — let extract_poses raise the
        # original "could not determine frame rate" error so the user
        # sees an actionable message rather than us masking it.
        return video_path, None

    log.info("video repair triggered: %s", rewrite_reason)

    # Re-open for streaming read.
    cap = cv2.VideoCapture(video_path)
    try:
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            return video_path, None
        height, width = first_frame.shape[:2]
        frames: list = [first_frame]
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()

    if len(frames) < 5:
        return video_path, None

    duration_sec = recording_duration_ms / 1000.0
    if duration_sec <= 0:
        return video_path, None
    computed_fps = len(frames) / duration_sec
    if computed_fps <= 0 or computed_fps > 240:
        # Sanity-bound — if the client-reported duration is nonsense,
        # don't write a fake FPS into the file.
        return video_path, None

    fixed_path = video_path + ".fixed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(fixed_path, fourcc, computed_fps, (width, height))
    if not writer.isOpened():
        return video_path, None
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()

    log.info(
        "tug: rewrote video with computed fps=%.2f (frames=%d, duration=%.2fs) → %s",
        computed_fps, len(frames), duration_sec, fixed_path,
    )
    return fixed_path, fixed_path


# ─── Main entry point ───────────────────────────────────────────
def analyze_tug(
    video_path: str,
    pose_options,
    patient_age: Optional[int] = None,
    recording_duration_ms: Optional[int] = None,
) -> TUGResult:
    """End-to-end TUG analysis. Returns a TUGResult ready to be wrapped
    in the API response model.

    `recording_duration_ms` is the client's wall-clock recording
    duration in record-mode; the engine uses it to repair WebM files
    whose header metadata is broken (see _ensure_decodable_video).
    """
    processed_path, fixed_path_cleanup = _ensure_decodable_video(
        video_path, recording_duration_ms,
    )

    try:
        # Stage 1+2: pose extraction + time-series build (reused).
        raw, fps, cv_total_frames = extract_poses(processed_path, pose_options)
        ts = build_time_series(raw)
        # cv2's reported frame count is sometimes off by ±1 from the
        # actual number of frames decoded into the time-series — use
        # the time-series length as the authoritative count.
        actual_total_frames = int(len(ts["left_hip"]["x_px"]))
        return _build_tug_result(
            processed_path=processed_path,
            raw=raw,
            ts=ts,
            fps=fps,
            total_frames=actual_total_frames,
            patient_age=patient_age,
        )
    finally:
        # Remove the rewritten temp file (if we made one) — the
        # original upload is cleaned up by the API endpoint's outer
        # finally block.
        if fixed_path_cleanup:
            try:
                os.unlink(fixed_path_cleanup)
            except OSError:
                pass


def _build_tug_result(
    *,
    processed_path: str,
    raw: dict,
    ts: dict,
    fps: float,
    total_frames: int,
    patient_age: Optional[int],
) -> TUGResult:
    """Run the analysis math + assemble the response. Split out of
    analyze_tug so the temp-file cleanup `finally` stays narrow."""

    # Stage 3: phase boundary detection
    boundaries, raw_flags = _find_phase_boundaries(ts, fps, total_frames)
    a_end = boundaries["A_end"]
    b_end = boundaries["B_end"]
    c_end = boundaries["C_end"]
    d_end = boundaries["D_end"]
    e_end = boundaries["E_end"]

    # Phase frame ranges
    phase_ranges = {
        "sit_to_stand": (0, a_end),
        "walk_out":     (a_end, b_end),
        "turn":         (b_end, c_end),
        "walk_back":    (c_end, d_end),
        "stand_to_sit": (d_end, e_end),
    }

    # Stage 4: per-phase metrics
    no_strikes_anywhere = True
    phase_objects: dict[str, TUGPhase] = {}
    for name, (start, end) in phase_ranges.items():
        duration = max(0.0, (end - start) / fps)
        truncated = raw_flags.get(f"truncated_{name[0].upper()}", False)
        # Map phase code letter to ABCDE for flag lookup (sit_to_stand=A,
        # walk_out=B, turn=C, walk_back=D, stand_to_sit=E)
        flag_lookup = {
            "sit_to_stand": "truncated_A",
            "walk_out":     "truncated_B",
            "turn":         "truncated_C",
            "walk_back":    "truncated_D",
            "stand_to_sit": "truncated_E",
        }
        truncated = raw_flags.get(flag_lookup[name], False)

        phase = TUGPhase(
            phase=name,  # type: ignore[arg-type]
            duration_sec=round(duration, 3),
            start_frame=start,
            end_frame=end,
            truncated=truncated,
        )

        if name in ("walk_out", "turn", "walk_back"):
            strikes = _strikes_in_range(
                ts, fps, start, end,
                is_turn_phase=(name == "turn"),
            )
            total_strikes = len(strikes["left"]) + len(strikes["right"])
            phase.step_count = total_strikes
            if total_strikes > 0:
                no_strikes_anywhere = False
            if name in ("walk_out", "walk_back"):
                phase.step_length_l_px = _step_length_px(ts, "left", strikes["left"])
                phase.step_length_r_px = _step_length_px(ts, "right", strikes["right"])
                if duration > 0:
                    phase.cadence_steps_per_min = round((total_strikes / duration) * 60.0, 1)
                    phase.walking_speed_mps = round(PATH_LENGTH_M / duration, 2) if duration > 0 else None

        phase_objects[name] = phase

    # Stage 5: aggregate + classify + flags + interpretation
    total_time = sum(p.duration_sec for p in phase_objects.values())
    classification = classify_total_time(total_time)
    norm = age_matched_norm(patient_age)
    age_threshold = norm[0] if norm else None
    age_passed = (total_time <= age_threshold) if age_threshold is not None else None

    truncated_any = any(raw_flags.get(k, False) for k in (
        "truncated_A", "truncated_B", "truncated_C", "truncated_D", "truncated_E",
    ))

    flags = compute_flags(
        turn_duration_sec=phase_objects["turn"].duration_sec,
        turn_step_count=phase_objects["turn"].step_count,
        total_time_sec=total_time,
        phase_truncated_any=truncated_any,
        turn_undetected=raw_flags.get("turn_undetected", False),
        no_strikes_anywhere=no_strikes_anywhere,
    )

    interpretation = build_interpretation(
        total_time_sec=total_time,
        classification=classification,
        turn_duration_sec=phase_objects["turn"].duration_sec,
        turn_step_count=phase_objects["turn"].step_count,
        walk_out_speed_mps=phase_objects["walk_out"].walking_speed_mps,
        walk_back_speed_mps=phase_objects["walk_back"].walking_speed_mps,
        flags=flags,
        patient_age=patient_age,
    )

    # Stage 6: grab the 5 key frames from the (possibly-rewritten)
    # video. We use `processed_path` rather than the source upload
    # because cv2.set(CAP_PROP_POS_FRAMES, idx) is unreliable on
    # broken-header WebM files, and the rewrite gives us proper seek.
    key_frames: list[TUGKeyFrame] = []
    cap = cv2.VideoCapture(processed_path)
    try:
        kf_specs = [
            ("test_start",              0),
            ("end_of_sit_to_stand",     a_end),
            ("start_of_turn",           b_end),
            ("end_of_turn",             c_end),
            ("test_end",                e_end),
        ]
        for label, idx in kf_specs:
            kf = _grab_key_frame(cap, idx, raw, label)
            if kf is not None:
                key_frames.append(kf)
    finally:
        cap.release()

    return TUGResult(
        total_time_sec=round(total_time, 2),
        classification=classification,
        sit_to_stand=phase_objects["sit_to_stand"],
        walk_out=phase_objects["walk_out"],
        turn=phase_objects["turn"],
        walk_back=phase_objects["walk_back"],
        stand_to_sit=phase_objects["stand_to_sit"],
        flags=flags,
        patient_age=patient_age,
        age_norm_threshold_sec=age_threshold,
        age_norm_passed=age_passed,
        interpretation=interpretation,
        key_frames=key_frames,
        fps=fps,
        total_frames=total_frames,
    )
