"""modified_thomas_engine.py — backend Modified Thomas Test pipeline
on the MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/modifiedThomas.ts so live + upload
return identical numbers.

Math:
  hip_angle  = inner angle at the hip between (hip→shoulder) and
               (hip→knee). 180° = thigh hanging in line with the body;
               smaller values = tight hip flexor.
  knee_angle = inner angle at the knee between (knee→hip) and
               (knee→ankle). ~90° = relaxed bent shin; >100° = rectus
               femoris tightness pulling the knee toward extension.

Static-hold reduction (not motion-peak like SLR/AKE):
  Compute (hip_angle, knee_angle) for every valid frame, then
  sub-sample to SAMPLE_HZ. Slide a window of STABILITY_WINDOW_FRAMES
  over the sub-sampled timeline; a window is "stable" when BOTH the
  hip-angle stddev AND the knee-angle stddev sit within
  STABILITY_JITTER_MAX_DEG. The LONGEST contiguous stable run is the
  capture window; the median hip and knee angles over that window are
  the result.

  If no stable run exists (e.g. patient never settled), fall back to
  the median over the LAST STABILITY_WINDOW_FRAMES samples and flag
  low_confidence=true so the report can warn the doctor.

Classification cutoffs:
  Hip (hip flexor):
    >= 170°  → "normal"
    155-170° → "mild"
    < 155°   → "significant"
  Knee (rectus femoris):
    <= 100°  → "normal"
    > 100°   → "tight"

Pipeline:
  1. Reuse gait_engine.extract_poses() + build_time_series() to pull
     smoothed 33-kp landmarks per frame.
  2. Visibility gate — require ≥30% of frames with test-side shoulder
     + hip + knee + ankle visible (the contralateral side is NOT
     required since the other leg is pulled to chest and often
     occluded).
  3. Per frame: compute hip_angle, knee_angle (None if any of the
     four required keypoints isn't visible).
  4. Sub-sample to SAMPLE_HZ for stability detection.
  5. Find the longest contiguous stable window.
  6. Take medians + stddevs over that window. Build the
     `samples` list (with `stable` flag per sample) and per-frame
     keypoints export for the report.
  7. Capture-moment screenshot (skeleton overlay) at the LAST frame
     of the chosen stable window.

Returns a dict matching the frontend ModifiedThomasSideResult shape
so ModifiedThomasReport renders without translation.
"""
from __future__ import annotations

import base64
import logging
import math
from statistics import median, pstdev
from typing import Optional

import cv2

from engines.gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/modifiedThomas.ts) ───
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TRIAL_DURATION_SEC = 30.0
_STABILITY_WINDOW_FRAMES = 15
_STABILITY_JITTER_MAX_DEG = 2.5
_HIP_NORMAL_MIN_DEG = 170.0
_HIP_MILD_MIN_DEG = 155.0
_KNEE_NORMAL_MAX_DEG = 100.0

_SIDE_INDICES = {
    "left": {
        "shoulder": "left_shoulder",
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
    },
    "right": {
        "shoulder": "right_shoulder",
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


def _compute_hip_angle(ts: dict, i: int, side: str) -> Optional[float]:
    idx = _SIDE_INDICES[side]
    if not _all_visible(ts, (idx["shoulder"], idx["hip"], idx["knee"]), i):
        return None
    sx = float(ts[idx["shoulder"]]["x_px"][i]); sy = float(ts[idx["shoulder"]]["y_px"][i])
    hx = float(ts[idx["hip"]]["x_px"][i]);      hy = float(ts[idx["hip"]]["y_px"][i])
    kx = float(ts[idx["knee"]]["x_px"][i]);     ky = float(ts[idx["knee"]]["y_px"][i])
    return _vector_angle_deg(
        sx - hx, sy - hy,
        kx - hx, ky - hy,
    )


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
def _classify_hip(hip_angle_deg: float) -> str:
    if hip_angle_deg >= _HIP_NORMAL_MIN_DEG: return "normal"
    if hip_angle_deg >= _HIP_MILD_MIN_DEG:   return "mild"
    return "significant"


def _classify_knee(knee_angle_deg: float) -> str:
    return "tight" if knee_angle_deg > _KNEE_NORMAL_MAX_DEG else "normal"


def _build_interpretation(
    side: str,
    hip_deg: float,
    knee_deg: float,
    hip_class: str,
    knee_class: str,
    low_confidence: bool,
) -> str:
    side_label = "Left MTT" if side == "left" else "Right MTT"
    hip = f"{hip_deg:.1f}°"
    knee = f"{knee_deg:.1f}°"

    if hip_deg <= 0 or knee_deg <= 0:
        return (
            f"{side_label}: no settled position captured. Re-record with the "
            f"patient lying still in the Modified Thomas position for at least "
            f"2 seconds."
        )

    if hip_class == "normal":
        hip_text = f"hip {hip} — normal hip-flexor length (≥ {_HIP_NORMAL_MIN_DEG:.0f}°)"
    elif hip_class == "mild":
        hip_text = (
            f"hip {hip} — mild hip-flexor tightness "
            f"({_HIP_MILD_MIN_DEG:.0f}–{_HIP_NORMAL_MIN_DEG:.0f}°)"
        )
    else:
        hip_text = (
            f"hip {hip} — significant hip-flexor tightness "
            f"(< {_HIP_MILD_MIN_DEG:.0f}°)"
        )

    knee_text = (
        f"knee {knee} — relaxed (≤ {_KNEE_NORMAL_MAX_DEG:.0f}°), no rectus femoris tightness"
        if knee_class == "normal"
        else f"knee {knee} — extended (> {_KNEE_NORMAL_MAX_DEG:.0f}°), rectus femoris tightness present"
    )

    confidence = (
        " (low-confidence capture — pose didn't fully settle, consider re-recording)"
        if low_confidence else ""
    )

    return f"{side_label}: {hip_text}; {knee_text}.{confidence}"


# ─── Stable-window detection ───────────────────────────────────
def _longest_stable_window(
    hips: list[Optional[float]],
    knees: list[Optional[float]],
) -> tuple[int, int] | None:
    """Walk through the timeline computing the rolling-window stddev
    over STABILITY_WINDOW_FRAMES. Return the (start, end_inclusive)
    indices of the LAST stable window in the longest contiguous run
    of windows that all qualify, or None when no window qualifies.

    Why "last" and "longest": a patient typically jostles into
    position, settles, holds — so the longest stable run reflects the
    intentional hold. Within the longest run we use the latest qualifying
    window so the medians come from the most-settled portion.
    """
    n = len(hips)
    if n < _STABILITY_WINDOW_FRAMES:
        return None

    # Compute a per-position "is window ending here stable" flag.
    # A window is stable iff all STABILITY_WINDOW_FRAMES samples have
    # non-None angles AND both stddevs are within threshold.
    stable_flags: list[bool] = []
    for end in range(_STABILITY_WINDOW_FRAMES - 1, n):
        start = end - _STABILITY_WINDOW_FRAMES + 1
        h_slice = hips[start:end + 1]
        k_slice = knees[start:end + 1]
        if any(v is None for v in h_slice) or any(v is None for v in k_slice):
            stable_flags.append(False)
            continue
        h_std = pstdev([float(v) for v in h_slice])  # type: ignore[arg-type]
        k_std = pstdev([float(v) for v in k_slice])  # type: ignore[arg-type]
        stable_flags.append(
            h_std <= _STABILITY_JITTER_MAX_DEG
            and k_std <= _STABILITY_JITTER_MAX_DEG
        )

    if not any(stable_flags):
        return None

    # Find the longest contiguous run of stable windows.
    best_len = 0
    best_end_in_flags = -1
    cur_len = 0
    for i, ok in enumerate(stable_flags):
        if ok:
            cur_len += 1
            if cur_len >= best_len:  # >= so we keep the LATEST tie
                best_len = cur_len
                best_end_in_flags = i
        else:
            cur_len = 0

    if best_end_in_flags < 0:
        return None

    # Convert back to sample indices. The flag at index `i` corresponds
    # to a window ending at sample `i + STABILITY_WINDOW_FRAMES - 1`.
    end_sample = best_end_in_flags + _STABILITY_WINDOW_FRAMES - 1
    start_sample = end_sample - _STABILITY_WINDOW_FRAMES + 1
    return (start_sample, end_sample)


# ─── Capture-moment screenshot ──────────────────────────────────
def _grab_capture_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    side: str,
) -> Optional[str]:
    """Grab the capture-moment frame with skeleton overlay, return as
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
def analyze_modified_thomas(
    video_path: str,
    pose_options,
    side: str,
) -> dict:
    """Run the Modified Thomas Test pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        side:         'left' or 'right' — the HANGING (test) leg.

    Returns:
        Dict matching the frontend ModifiedThomasSideResult shape plus
        diagnostic extras (fps, total_frames, valid_frames,
        interpretation). The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' when the patient/leg isn't
                    clearly visible in enough frames.
                    api.analyze_modified_thomas_endpoint maps this to
                    HTTP 400. (A no-stable-window clip is NOT raised;
                    it's returned with low_confidence=true so the
                    operator can still see something.)
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n_full = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
        len(ts["left_knee"]["y"]),
        len(ts["right_knee"]["y"]),
    ))
    if n_full == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    # ── Visibility gate ─────────────────────────────────────
    idx = _SIDE_INDICES[side]
    required_keys = (
        idx["shoulder"], idx["hip"], idx["knee"], idx["ankle"],
    )
    visible_frames = sum(
        1 for i in range(n_full) if _all_visible(ts, required_keys, i)
    )
    if visible_frames < max(3, int(n_full * 0.30)):
        raise ValueError("poor_visibility")

    # ── Sub-sample to SAMPLE_HZ ──────────────────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_full_indices = list(range(0, n_full, step))

    hips:  list[Optional[float]] = []
    knees: list[Optional[float]] = []
    for i in sampled_full_indices:
        hips.append(_compute_hip_angle(ts, i, side))
        knees.append(_compute_knee_angle(ts, i, side))

    # ── Find the longest stable window ─────────────────────
    window = _longest_stable_window(hips, knees)
    low_confidence = False

    if window is None:
        # Fallback: try the last STABILITY_WINDOW_FRAMES samples; if
        # they have at least 5 valid hip + knee values each, use their
        # medians but flag low_confidence.
        low_confidence = True
        tail_start = max(0, len(hips) - _STABILITY_WINDOW_FRAMES)
        tail_h = [v for v in hips[tail_start:] if v is not None]
        tail_k = [v for v in knees[tail_start:] if v is not None]
        if len(tail_h) >= 5 and len(tail_k) >= 5:
            hip_med = float(median(tail_h))
            knee_med = float(median(tail_k))
            hip_std = float(pstdev(tail_h)) if len(tail_h) >= 2 else 0.0
            knee_std = float(pstdev(tail_k)) if len(tail_k) >= 2 else 0.0
            cap_sample_idx = len(hips) - 1
        else:
            # Couldn't even build a fallback — return zeros and let the
            # frontend show "no settled position captured".
            hip_med = 0.0
            knee_med = 0.0
            hip_std = 0.0
            knee_std = 0.0
            cap_sample_idx = None
    else:
        start, end = window
        h_slice = [float(v) for v in hips[start:end + 1] if v is not None]
        k_slice = [float(v) for v in knees[start:end + 1] if v is not None]
        hip_med = float(median(h_slice))
        knee_med = float(median(k_slice))
        hip_std = float(pstdev(h_slice)) if len(h_slice) >= 2 else 0.0
        knee_std = float(pstdev(k_slice)) if len(k_slice) >= 2 else 0.0
        cap_sample_idx = end

    # ── Build samples list (with `stable` flag) + keypoints ──
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    # If we have a real stable window, mark those samples as stable
    # for the frontend chart's green shaded region.
    stable_set: set[int] = set()
    if window is not None:
        for s in range(window[0], window[1] + 1):
            stable_set.add(s)

    for sample_idx, full_i in enumerate(sampled_full_indices):
        t_ms = (full_i / fps) * 1000.0 if fps > 0 else 0.0
        samples.append({
            "t_ms": float(t_ms),
            "hip_angle_deg":  float(hips[sample_idx])  if hips[sample_idx]  is not None else None,
            "knee_angle_deg": float(knees[sample_idx]) if knees[sample_idx] is not None else None,
            "stable":         sample_idx in stable_set,
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

    duration_seconds = float(n_full / fps) if fps > 0 else 0.0

    # Capture-moment screenshot (use the LAST frame of the chosen
    # stable window, mapped back to the full-resolution frame index).
    capture_screenshot: Optional[str] = None
    if cap_sample_idx is not None and cap_sample_idx < len(sampled_full_indices):
        full_frame_idx = sampled_full_indices[cap_sample_idx]
        capture_screenshot = _grab_capture_frame(
            video_path, full_frame_idx, raw, side,
        )

    hip_class = _classify_hip(hip_med)
    knee_class = _classify_knee(knee_med)
    interpretation = _build_interpretation(
        side, hip_med, knee_med, hip_class, knee_class, low_confidence,
    )

    # Backend always sees a fully captured clip; termination reflects
    # whether the engine could lock onto a stable window.
    termination = "captured" if not low_confidence else "timeout"

    return {
        "side_tested": side,
        "hip_angle_deg":   float(hip_med),
        "knee_angle_deg":  float(knee_med),
        "hip_classification":  hip_class,
        "knee_classification": knee_class,
        "hip_angle_stddev_deg":  float(hip_std),
        "knee_angle_stddev_deg": float(knee_std),
        "low_confidence": bool(low_confidence),
        "capture_sample_index": int(cap_sample_idx) if cap_sample_idx is not None else None,
        "duration_seconds": duration_seconds,
        "termination": termination,
        "samples": samples,
        "keypoints": keypoints_export,
        "capture_screenshot_data_url": capture_screenshot,
        # Diagnostic extras
        "fps": float(fps),
        "total_frames": int(n_full),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
    }
