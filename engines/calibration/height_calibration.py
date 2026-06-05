"""engines/calibration/height_calibration.py — backend height-based
scale calibration.

Mirrors motionlens-web/lib/calibration/heightCalibration.ts so live
mode and upload mode produce the same pixels_per_cm for the same
clip + patient height.

Algorithm:
  1. Sample N frames from the early "standing window" of the clip
     (first ~3 s). The patient is expected to stand straight, fully
     in frame, before the test movement begins.
  2. For each sampled frame: compute body pixel height = lowest
     visible foot landmark y − highest visible head landmark y.
  3. Drop frames where torso landmarks aren't visible or where the
     measurement is anomalous compared to the rest.
  4. Take the MEDIAN of the surviving readings as the locked body
     pixel height — robust to single-frame jitter.
  5. pixels_per_cm = body_pixel_height / (patient_height_cm × 0.87)
     The 0.87 ratio is the anthropometric average for nose-to-floor
     fraction of total stature; we use it as the bridge between the
     pixel span we can measure and the patient's true height.
"""
from __future__ import annotations

import logging
from statistics import median as _stdlib_median
from typing import Any, Optional

import cv2

log = logging.getLogger(__name__)


NOSE_TO_FLOOR_HEIGHT_FRACTION = 0.87  # mirrors lib/calibration/types.ts
MIN_HEIGHT_CM = 80.0
MAX_HEIGHT_CM = 230.0

# Visibility gate for the head + foot landmark search.
_HEAD_FOOT_VIS_THRESHOLD = 0.30

# Sampling window — patient is assumed to be standing straight in
# the first few seconds of the clip.
_STANDING_WINDOW_SEC = 3.0
_DEFAULT_SAMPLE_FRAMES = 12

# Acceptable spread between agreeing samples (fraction of median).
_STABLE_TOLERANCE_FRACTION = 0.05

_HEAD_LANDMARKS = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
)

_FOOT_LANDMARKS = (
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

_TORSO_REQUIRED = (
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
)


def _measure_body_pixel_height(ts: dict, frame_idx: int) -> Optional[float]:
    """Body pixel height at frame_idx, computed from the smoothed
    time-series produced by engines.gait_engine.build_time_series.

    Returns None when torso isn't fully visible OR neither head nor
    feet landmarks are above the visibility threshold.
    """
    # Torso anchor must be present — otherwise the patient is
    # probably partial in frame and the height reading would be
    # meaningless.
    for name in _TORSO_REQUIRED:
        track = ts.get(name)
        if not track:
            return None
        if frame_idx >= len(track["vis"]):
            return None
        if float(track["vis"][frame_idx]) < _HEAD_FOOT_VIS_THRESHOLD:
            return None

    top_y: Optional[float] = None
    for name in _HEAD_LANDMARKS:
        track = ts.get(name)
        if not track or frame_idx >= len(track["vis"]):
            continue
        if float(track["vis"][frame_idx]) < _HEAD_FOOT_VIS_THRESHOLD:
            continue
        y = float(track["y_px"][frame_idx])
        if top_y is None or y < top_y:
            top_y = y
    if top_y is None:
        return None

    bottom_y: Optional[float] = None
    for name in _FOOT_LANDMARKS:
        track = ts.get(name)
        if not track or frame_idx >= len(track["vis"]):
            continue
        if float(track["vis"][frame_idx]) < _HEAD_FOOT_VIS_THRESHOLD:
            continue
        y = float(track["y_px"][frame_idx])
        if bottom_y is None or y > bottom_y:
            bottom_y = y
    if bottom_y is None:
        return None

    body_px = bottom_y - top_y
    return body_px if body_px > 0 else None


def measure_body_pixel_height_from_time_series(
    ts: dict,
    fps: float,
    total_frames: int,
    standing_window_sec: float = _STANDING_WINDOW_SEC,
    sample_frames: int = _DEFAULT_SAMPLE_FRAMES,
) -> Optional[float]:
    """Sample the standing-window frames and return a stable median
    body pixel height, or None when no agreeing reading was found.
    """
    if fps <= 0 or total_frames <= 0:
        return None
    window_frames = min(total_frames, int(fps * standing_window_sec))
    if window_frames < sample_frames:
        sample_frames = max(1, window_frames)
    step = max(1, window_frames // max(1, sample_frames))
    indices = list(range(0, window_frames, step))[:sample_frames]
    if not indices:
        return None

    readings: list[float] = []
    for i in indices:
        v = _measure_body_pixel_height(ts, i)
        if v is not None and v > 0:
            readings.append(v)

    if not readings:
        return None

    median = float(_stdlib_median(readings))
    # Keep only the readings within ±tolerance of the median — drops
    # outliers from partial-body frames the visibility gate let
    # through.
    inliers = [
        r for r in readings
        if abs(r - median) / median <= _STABLE_TOLERANCE_FRACTION
    ]
    if not inliers:
        # No agreement — refuse to calibrate; safer than picking a
        # noisy value.
        log.info(
            "height_calibration: readings ranged from %.0f to %.0f px "
            "(median %.0f) — no agreement, returning None.",
            min(readings), max(readings), median,
        )
        return None
    return float(_stdlib_median(inliers))


def pixels_per_cm_from_height(
    body_pixel_height_px: float,
    patient_height_cm: float,
) -> Optional[float]:
    if body_pixel_height_px <= 0:
        return None
    if patient_height_cm < MIN_HEIGHT_CM or patient_height_cm > MAX_HEIGHT_CM:
        return None
    effective_cm = patient_height_cm * NOSE_TO_FLOOR_HEIGHT_FRACTION
    if effective_cm <= 0:
        return None
    return float(body_pixel_height_px / effective_cm)


def build_height_calibration_dict(
    body_pixel_height_px: float,
    patient_height_cm: float,
    source_frame: dict[str, int],
) -> Optional[dict[str, Any]]:
    """Assemble a CalibrationResult-shaped dict for the API layer."""
    ppc = pixels_per_cm_from_height(body_pixel_height_px, patient_height_cm)
    if ppc is None:
        return None
    return {
        "pixels_per_cm": ppc,
        "detected_at_ms": 0,
        "source": "height_based",
        "patient_height_cm": float(patient_height_cm),
        "body_pixel_height_px": float(body_pixel_height_px),
        "source_frame_px": source_frame,
    }


def probe_source_frame_dimensions(video_path: str) -> dict[str, int]:
    """Best-effort read of the video's frame dimensions for the
    CalibrationResult.source_frame_px field. Returns {0,0} on failure.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return {"width": w, "height": h}
