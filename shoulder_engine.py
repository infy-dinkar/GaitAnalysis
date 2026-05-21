"""
shoulder_engine.py
Shoulder range-of-motion angle math for the Biomechanical Analysis flow.

Conventions
-----------
Coordinate system: MediaPipe Pose 2D image coords. Origin top-left,
x increases rightward, y increases DOWNWARD. All inputs are (x, y)
tuples in any consistent unit (normalised 0..1 or pixel — the angle
math is scale-invariant).

Anatomical neutral = arm hanging straight at the side. Each formula
returns angles in degrees; magnitude = degrees of motion away from
neutral; sign = forward/back (or up/down where applicable). Sign
assumes the subject is facing camera-right (+x). For camera-left
captures, sign flips — only matters for ambiguating flexion vs
extension, not for the magnitude reported in the UI.

Visibility
----------
`compute_shoulder_angle` returns None when ANY required landmark has
`visibility < vis_threshold` (default 0.5). The caller drops that
frame from the peak-tracker.

Video-pipeline orchestrator
---------------------------
`analyze_shoulder` runs the full BlazePose-Full pipeline (reused from
gait_engine) on an uploaded clip and returns a BiomechData-shaped
dict. It supports both single-direction movements (legacy) and the
merged "flexion_extension" test that captures both peaks in one
trial. Mirrors `analyze_ankle` in ankle_engine.py — same call
signature, same reuse of `extract_poses` + `build_time_series`,
same key-frame thumbnail helper pattern.
"""
from __future__ import annotations

import base64
import math
from typing import Optional, Sequence

import cv2
import numpy as np

# MediaPipe Pose landmark indices used for shoulder ROM
LEFT_SHOULDER  = 11
LEFT_ELBOW     = 13
LEFT_WRIST     = 15
LEFT_HIP       = 23
RIGHT_SHOULDER = 12
RIGHT_ELBOW    = 14
RIGHT_WRIST    = 16
RIGHT_HIP      = 24


# Clinical normal ranges for shoulder ROM (AAOS).
# `range` is the inclusive (low, high) pair displayed in the report;
# `target` is the single value used for the "% of normal" calculation
# (midpoint when the spec gives a range; the upper bound when the spec
# gives a single canonical value).
SHOULDER_NORMAL_RANGES = {
    "flexion":           {"range": (180, 180), "target": 180.0},
    "extension":         {"range": (45, 60),   "target": 52.5},
    "abduction":         {"range": (180, 180), "target": 180.0},
    "adduction":         {"range": (30, 50),   "target": 40.0},
    "external_rotation": {"range": (90, 90),   "target": 90.0},
    "internal_rotation": {"range": (70, 90),   "target": 80.0},
}


# ──────────────────────────────────────────────
# Math primitives
# ──────────────────────────────────────────────
def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Signed angle from v1 to v2 in degrees, in image-coord 2D space.
    Result is in (-180, +180]. The 2D cross product `v1.x*v2.y - v1.y*v2.x`
    is positive when v2 is rotated clockwise relative to v1 in the image
    (which corresponds to counterclockwise in math y-up convention).
    """
    dot   = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(cross, dot))


# ──────────────────────────────────────────────
# Per-movement angle formulas
# ──────────────────────────────────────────────
def shoulder_flexion_extension(
    shoulder: tuple[float, float],
    elbow:    tuple[float, float],
    hip:      tuple[float, float],
) -> float:
    """Sagittal (side view). 0° = arm at side. Positive = forward
    (flexion), negative = backward (extension). Magnitude = degrees
    away from neutral. Subject assumed to face camera-right.

    Angle is measured between trunk_down (shoulder→hip) and arm
    (shoulder→elbow); we negate the raw signed angle so the sign of
    the result matches the anatomical convention rather than the
    image-y-down convention of atan2.
    """
    trunk_down = (hip[0] - shoulder[0], hip[1] - shoulder[1])
    arm        = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
    raw = angle_between(trunk_down, arm)
    return -raw


def shoulder_abduction_adduction(
    shoulder: tuple[float, float],
    elbow:    tuple[float, float],
    hip:      tuple[float, float],
) -> float:
    """Coronal (front view). 0° = arm at side. Returns the magnitude
    of arm deviation from the trunk axis in the frontal plane.
    Adduction (arm crossing midline) and abduction (arm out to side)
    are both reported as positive magnitudes — the caller distinguishes
    the two by which movement was being recorded.
    """
    trunk_down = (hip[0] - shoulder[0], hip[1] - shoulder[1])
    arm        = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
    return abs(angle_between(trunk_down, arm))


def shoulder_rotation(
    shoulder: tuple[float, float],
    elbow:    tuple[float, float],
    wrist:    tuple[float, float],
) -> float:
    """Approximate rotation. With elbow flexed ~90° and upper arm at
    side, measures the forearm's deviation from the horizontal-forward
    reference. 2D estimation is approximate — communicate this in the
    UI. Returns positive magnitude in degrees.
    """
    forearm   = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    reference = (1.0, 0.0)
    return abs(angle_between(reference, forearm))


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────
_SIDE_INDICES = {
    "left":  {"shoulder": LEFT_SHOULDER,  "elbow": LEFT_ELBOW,
              "wrist":    LEFT_WRIST,     "hip":   LEFT_HIP},
    "right": {"shoulder": RIGHT_SHOULDER, "elbow": RIGHT_ELBOW,
              "wrist":    RIGHT_WRIST,    "hip":   RIGHT_HIP},
}


def compute_shoulder_angle(
    landmarks: Sequence,
    side: str,
    movement: str,
    vis_threshold: float = 0.5,
) -> float | None:
    """Compute the shoulder ROM angle for one frame.

    Args:
        landmarks: sequence of 33 MediaPipe Pose landmarks. Each item
                   must expose `.x`, `.y`, `.visibility` (numeric).
        side:      'left' or 'right'.
        movement:  one of SHOULDER_NORMAL_RANGES keys.

    Returns:
        Angle in degrees, or None if any required landmark has
        visibility below `vis_threshold`.
    """
    if side not in _SIDE_INDICES:
        raise ValueError(f"Unknown side: {side!r}")
    if movement not in SHOULDER_NORMAL_RANGES:
        raise ValueError(f"Unknown shoulder movement: {movement!r}")

    idx = _SIDE_INDICES[side]
    needed = [idx["shoulder"], idx["elbow"], idx["hip"]]
    if movement in ("external_rotation", "internal_rotation"):
        needed.append(idx["wrist"])

    for k in needed:
        if landmarks[k].visibility < vis_threshold:
            return None

    s = (landmarks[idx["shoulder"]].x, landmarks[idx["shoulder"]].y)
    e = (landmarks[idx["elbow"]].x,    landmarks[idx["elbow"]].y)
    h = (landmarks[idx["hip"]].x,      landmarks[idx["hip"]].y)

    if movement in ("flexion", "extension"):
        return shoulder_flexion_extension(s, e, h)
    if movement in ("abduction", "adduction"):
        return shoulder_abduction_adduction(s, e, h)
    # external/internal rotation
    w = (landmarks[idx["wrist"]].x, landmarks[idx["wrist"]].y)
    return shoulder_rotation(s, e, w)


# ══════════════════════════════════════════════════════════════════
# Video-pipeline orchestrator (BlazePose Full, reuses gait pipeline)
# ══════════════════════════════════════════════════════════════════

# Visibility floor for the math-driving joints. Looser than the
# per-frame `compute_shoulder_angle` default because the smoothed
# time-series tolerates brief dips better than a single raw frame.
_SHOULDER_VIS_THRESHOLD = 0.4


# Anatomical sanity ceiling for the EXTENSION direction. Shoulder
# extension's clinical max is ~60°; any negative-signed magnitude
# past this is almost certainly a sign flip caused by keypoint
# jitter at the overhead end of the flexion arc (where the arm
# vector is near-vertical and a few pixels of jitter on the elbow
# flips the angle's sign). Frames past this threshold get re-routed
# into the flexion peak so they can't lock an implausible 170°+
# "extension" reading.
_EXTENSION_ANATOMICAL_MAX = 75.0


# Movement metadata used by the merged flexion_extension flow when
# building the response. Source-of-truth target ranges are owned by
# the frontend metadata (SHOULDER_MOVEMENTS in shoulder.ts); these
# are kept in sync so the backend response is self-contained.
_MERGED_FLEXEXT_PRIMARY_TARGET: tuple[float, float] = (150.0, 180.0)
_MERGED_FLEXEXT_SECONDARY_TARGET: tuple[float, float] = (45.0, 60.0)


def _classify_in_range(value: float, lo: float, hi: float) -> str:
    """Range-aware classification matching AssessmentReport.classify:
    in-range = good; within 30% of range width outside = fair; else
    poor."""
    if value >= lo and value <= hi:
        return "good"
    width = max(1.0, hi - lo)
    dist = (lo - value) if value < lo else (value - hi)
    return "fair" if (dist / width) <= 0.30 else "poor"


def _grab_shoulder_key_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    label: str,
    side: str,
) -> Optional[dict]:
    """Seek to `frame_index` in the source video, draw a skeleton
    overlay highlighting the test-side arm + trunk, return a JPEG
    data URL. Mirrors ankle_engine._grab_ankle_key_frame's pattern."""
    if frame_index < 0:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None:
        return None

    h, w = frame.shape[:2]
    target_w = min(640, w)
    if target_w < w:
        scale = target_w / w
        frame = cv2.resize(frame, (target_w, int(h * scale)))
        h, w = frame.shape[:2]

    # Import LM here to avoid a circular import at module load.
    from gait_engine import LM

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
        emphasised = name.startswith(side)
        outer = (0, 0, 220) if emphasised else (150, 150, 150)
        cv2.circle(frame, (px, py), 5, outer, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        return (px, py)

    # Edges relevant to shoulder ROM: trunk, both arms, hips.
    edges = [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
        ("left_shoulder",  "left_elbow"),
        ("left_elbow",     "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow",    "right_wrist"),
    ]
    dot_pos: dict[str, tuple[int, int]] = {}
    for name in LM:
        p = _draw_dot(name)
        if p:
            dot_pos[name] = p
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            on_side = a.startswith(side) and b.startswith(side)
            line_colour = (255, 255, 255) if on_side else (180, 180, 180)
            cv2.line(frame, dot_pos[a], dot_pos[b], line_colour, 2)
    _ = label  # caption is rendered by the frontend, not baked into the JPEG

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "label": label,
        "frame_index": int(frame_index),
        "image_data_url": f"data:image/jpeg;base64,{b64}",
    }


def analyze_shoulder(
    video_path: str,
    pose_options,
    movement: str,
    side: str,
) -> dict:
    """Run the BlazePose-Full shoulder pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded (and optionally repaired)
                      video file on disk.
        pose_options: PoseLandmarkerOptions configured for VIDEO mode
                      against pose_landmarker_full.task — built by
                      api._build_gait_pose_options().
        movement:     "flexion_extension" (merged) or any single key
                      from SHOULDER_NORMAL_RANGES.
        side:         "left" or "right".

    Returns:
        Dict matching the BiomechData Pydantic schema. For merged
        movements the secondary_peak_* fields + primary/secondary
        labels are populated; for single-direction movements they
        are omitted (caller wraps in BiomechResponse).

    Raises:
        ValueError: input arguments invalid, or the video produced
                    fewer than a handful of frames with the required
                    landmarks visible. The endpoint maps this to a
                    "poor_visibility" / 400 response.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")
    is_merged = movement == "flexion_extension"
    if not is_merged and movement not in SHOULDER_NORMAL_RANGES:
        raise ValueError(f"Unsupported shoulder movement: {movement!r}")

    # Import gait pipeline here so the legacy math-only helpers above
    # don't pull in MediaPipe / OpenCV at module load (preserves the
    # existing __main__ self-test which only needs the math).
    from gait_engine import extract_poses, build_time_series

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    shoulder_key = f"{side}_shoulder"
    elbow_key    = f"{side}_elbow"
    hip_key      = f"{side}_hip"
    wrist_key    = f"{side}_wrist"

    sx = ts[shoulder_key]["x_px"]; sy = ts[shoulder_key]["y_px"]; vs = ts[shoulder_key]["vis"]
    ex = ts[elbow_key]["x_px"];    ey = ts[elbow_key]["y_px"];    ve = ts[elbow_key]["vis"]
    hx = ts[hip_key]["x_px"];      hy = ts[hip_key]["y_px"];      vh = ts[hip_key]["vis"]
    wx = ts[wrist_key]["x_px"];    wy = ts[wrist_key]["y_px"];    vw = ts[wrist_key]["vis"]

    needs_wrist = movement in ("external_rotation", "internal_rotation")
    n = int(min(len(sx), len(ex), len(hx)))
    if needs_wrist:
        n = int(min(n, len(wx)))

    # Per-frame signed angle. None when the math joints aren't all
    # confidently visible — the frame is dropped from peak tracking.
    angles: list[float | None] = []
    valid_frames = 0
    for i in range(n):
        if vs[i] < _SHOULDER_VIS_THRESHOLD or ve[i] < _SHOULDER_VIS_THRESHOLD or vh[i] < _SHOULDER_VIS_THRESHOLD:
            angles.append(None)
            continue
        if needs_wrist and vw[i] < _SHOULDER_VIS_THRESHOLD:
            angles.append(None)
            continue
        s = (float(sx[i]), float(sy[i]))
        e = (float(ex[i]), float(ey[i]))
        h = (float(hx[i]), float(hy[i]))
        if is_merged or movement in ("flexion", "extension"):
            a = shoulder_flexion_extension(s, e, h)
        elif movement in ("abduction", "adduction"):
            a = shoulder_abduction_adduction(s, e, h)
        else:
            w_pt = (float(wx[i]), float(wy[i]))
            a = shoulder_rotation(s, e, w_pt)
        angles.append(a)
        valid_frames += 1

    if valid_frames < max(3, int(fps * 0.5)):
        # Less than ~half a second of usable footage. Surfaced to
        # the operator as "arm not clearly visible" rather than a
        # generic analysis failure.
        raise ValueError("poor_visibility")

    # ── Single-direction movements ──────────────────────────────
    if not is_merged:
        peak_mag = 0.0
        peak_angle: float | None = None
        peak_frame_idx = -1
        for i, a in enumerate(angles):
            if a is None:
                continue
            if abs(a) > peak_mag:
                peak_mag = abs(a)
                peak_angle = a
                peak_frame_idx = i

        normal = SHOULDER_NORMAL_RANGES[movement]
        ref_low, ref_high = normal["range"]
        target = normal["target"]
        percentage = (peak_mag / target) * 100.0 if target > 0 else 0.0
        status = _classify_in_range(peak_mag, float(ref_low), float(ref_high))
        label = movement.replace("_", " ").title()
        interpretation = (
            f"{label} ({side.capitalize()}) measured {peak_mag:.1f}°, "
            f"which is {percentage:.0f}% of the {ref_low:.0f}°–{ref_high:.0f}° "
            f"normal range — {status}."
        )

        key_frames: list[dict] = []
        # Neutral = frame with the smallest |angle| (closest to arm-at-side).
        neutral_idx = -1
        neutral_abs = math.inf
        for i, a in enumerate(angles):
            if a is None:
                continue
            if abs(a) < neutral_abs:
                neutral_abs = abs(a)
                neutral_idx = i
        if neutral_idx >= 0:
            kf = _grab_shoulder_key_frame(
                video_path, neutral_idx, raw, "Neutral — start", side,
            )
            if kf:
                key_frames.append(kf)
        if peak_frame_idx >= 0:
            kf = _grab_shoulder_key_frame(
                video_path, peak_frame_idx, raw,
                f"Peak {label} ({peak_mag:.1f}°)", side,
            )
            if kf:
                key_frames.append(kf)

        return {
            "body_part": "shoulder",
            "movement": movement,
            "side": side,
            "peak_angle": peak_angle if valid_frames > 0 else None,
            "peak_magnitude": peak_mag,
            "reference_range": [float(ref_low), float(ref_high)],
            "target": float(target),
            "percentage": percentage,
            "status": status,
            "valid_frames": valid_frames,
            "total_frames": n,
            "fps": float(fps),
            "interpretation": interpretation,
            "key_frames": key_frames,
        }

    # ── Merged flexion + extension ─────────────────────────────
    # The signed-angle convention from shoulder_flexion_extension is:
    #   positive value → arm forward of the trunk (flexion direction)
    #   negative value → arm behind the trunk     (extension direction)
    # Track the maximum signed value for flexion and the most-negative
    # value for extension separately. Apply the anatomical sanity
    # guard to extension so a sign flip at the overhead end of the
    # flexion arc can't lock an implausible 170° "extension" peak.
    primary_peak_signed: float | None = None     # flexion (positive)
    primary_peak_idx = -1
    secondary_peak_signed: float | None = None   # extension (negative)
    secondary_peak_idx = -1
    neutral_idx = -1
    neutral_abs = math.inf
    for i, a in enumerate(angles):
        if a is None:
            continue
        if abs(a) < neutral_abs:
            neutral_abs = abs(a)
            neutral_idx = i
        if a >= 0:
            if primary_peak_signed is None or a > primary_peak_signed:
                primary_peak_signed = a
                primary_peak_idx = i
        else:
            mag = -a
            if mag > _EXTENSION_ANATOMICAL_MAX:
                # Treat as flexion (sign-flip artefact). See
                # _EXTENSION_ANATOMICAL_MAX rationale.
                if primary_peak_signed is None or mag > primary_peak_signed:
                    primary_peak_signed = mag
                    primary_peak_idx = i
                continue
            if secondary_peak_signed is None or a < secondary_peak_signed:
                secondary_peak_signed = a
                secondary_peak_idx = i

    primary_mag = float(primary_peak_signed or 0.0)
    secondary_mag = float(-secondary_peak_signed) if secondary_peak_signed is not None else 0.0

    p_lo, p_hi = _MERGED_FLEXEXT_PRIMARY_TARGET
    s_lo, s_hi = _MERGED_FLEXEXT_SECONDARY_TARGET
    p_target = p_hi
    p_pct = (primary_mag / p_target) * 100.0 if p_target > 0 else 0.0
    p_status = _classify_in_range(primary_mag, p_lo, p_hi)

    interpretation_primary = (
        f"Flexion ({side.capitalize()}) measured {primary_mag:.1f}°, "
        f"which is {p_pct:.0f}% of the {p_lo:.0f}°–{p_hi:.0f}° normal range "
        f"— {p_status}."
    )
    if secondary_mag > 0:
        s_status = _classify_in_range(secondary_mag, s_lo, s_hi)
        interpretation_secondary = (
            f"Extension ({side.capitalize()}) measured {secondary_mag:.1f}°, "
            f"which is "
            f"{(secondary_mag / s_hi) * 100.0:.0f}% of the "
            f"{s_lo:.0f}°–{s_hi:.0f}° normal range — {s_status}."
        )
    else:
        interpretation_secondary = (
            "Extension direction was not detected in this recording."
        )
    interpretation = f"{interpretation_primary} {interpretation_secondary}"

    key_frames: list[dict] = []
    if neutral_idx >= 0:
        kf = _grab_shoulder_key_frame(
            video_path, neutral_idx, raw, "Neutral — start", side,
        )
        if kf:
            key_frames.append(kf)
    if primary_peak_idx >= 0 and primary_mag > 0:
        kf = _grab_shoulder_key_frame(
            video_path, primary_peak_idx, raw,
            f"Flexion ({primary_mag:.1f}°)", side,
        )
        if kf:
            key_frames.append(kf)
    if secondary_peak_idx >= 0 and secondary_mag > 0:
        kf = _grab_shoulder_key_frame(
            video_path, secondary_peak_idx, raw,
            f"Extension ({secondary_mag:.1f}°)", side,
        )
        if kf:
            key_frames.append(kf)

    return {
        "body_part": "shoulder",
        "movement": "flexion_extension",
        "side": side,
        "peak_angle": float(primary_peak_signed) if primary_peak_signed is not None else None,
        "peak_magnitude": primary_mag,
        "reference_range": [float(p_lo), float(p_hi)],
        "target": float(p_target),
        "percentage": p_pct,
        "status": p_status,
        "valid_frames": valid_frames,
        "total_frames": n,
        "fps": float(fps),
        "interpretation": interpretation,
        "key_frames": key_frames,
        "secondary_peak_angle": (
            float(secondary_peak_signed) if secondary_peak_signed is not None else None
        ),
        "secondary_peak_magnitude": secondary_mag if secondary_mag > 0 else None,
        "secondary_reference_range": [float(s_lo), float(s_hi)],
        "primary_label": "Flexion",
        "secondary_label": "Extension",
    }


# ══════════════════════════════════════════════════════════════════
# Inline sanity tests — run with `python shoulder_engine.py`
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    def _close(actual, expected, tol=0.5):
        return abs(actual - expected) <= tol

    # 1. Arm at side: trunk_down=(0,1), arm=(0,1) → 0°.
    a = shoulder_flexion_extension((0, 0), (0, 1), (0, 1))
    assert _close(a, 0.0), f"arm-at-side flexion should be 0, got {a}"
    print(f"  flexion at-side                       : {a:+.2f}°  (expected 0)")

    # 2. Arm forward 90° (subject faces +x): elbow at (+1, 0).
    a = shoulder_flexion_extension((0, 0), (1, 0), (0, 1))
    assert _close(a, 90.0), f"arm forward 90 should be +90, got {a}"
    print(f"  flexion arm forward 90°               : {a:+.2f}°  (expected +90)")

    # 3. Arm backward 90°: elbow at (-1, 0). Negative → extension.
    a = shoulder_flexion_extension((0, 0), (-1, 0), (0, 1))
    assert _close(a, -90.0), f"arm back 90 should be -90, got {a}"
    print(f"  flexion arm backward 90° (extension)  : {a:+.2f}°  (expected -90)")

    # 4. Arm fully overhead 180°: elbow at (0, -1). Magnitude 180.
    a = shoulder_flexion_extension((0, 0), (0, -1), (0, 1))
    assert _close(abs(a), 180.0), f"arm overhead should be ±180, got {a}"
    print(f"  flexion arm overhead 180°             : {a:+.2f}°  (expected ±180)")

    # 5. Abduction 90° (arm out to side, front view).
    a = shoulder_abduction_adduction((0, 0), (1, 0), (0, 1))
    assert _close(a, 90.0), f"abduction 90 should be 90, got {a}"
    print(f"  abduction arm out 90°                 : {a:+.2f}°  (expected 90)")

    # 6. Abduction 0° (arm at side).
    a = shoulder_abduction_adduction((0, 0), (0, 1), (0, 1))
    assert _close(a, 0.0), f"abduction at-side should be 0, got {a}"
    print(f"  abduction at-side                     : {a:+.2f}°  (expected 0)")

    # 7. Rotation: forearm horizontal forward (neutral) → 0°.
    a = shoulder_rotation((0, 0), (0, 0), (1, 0))
    assert _close(a, 0.0), f"rotation neutral should be 0, got {a}"
    print(f"  rotation forearm forward              : {a:+.2f}°  (expected 0)")

    # 8. Rotation: forearm pointing up (external rotation 90°).
    a = shoulder_rotation((0, 0), (0, 0), (0, -1))
    assert _close(a, 90.0), f"rotation forearm up should be 90, got {a}"
    print(f"  rotation forearm up 90°               : {a:+.2f}°  (expected 90)")

    # 9. Visibility gate: fake landmark with low visibility → None.
    class _LM:
        def __init__(self, x, y, v): self.x, self.y, self.visibility = x, y, v

    lms = [None] * 33
    for i in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_HIP):
        lms[i] = _LM(0.5, 0.5, 0.9)
    lms[LEFT_HIP] = _LM(0.5, 0.7, 0.3)  # below threshold
    a = compute_shoulder_angle(lms, "left", "flexion")
    assert a is None, f"low visibility should return None, got {a}"
    print(f"  low-vis hip => dispatcher returns None: {a}  (expected None)")

    # 10. Dispatcher end-to-end with all visibility ok.
    lms[LEFT_HIP] = _LM(0.0, 1.0, 0.95)
    lms[LEFT_SHOULDER] = _LM(0.0, 0.0, 0.95)
    lms[LEFT_ELBOW]    = _LM(1.0, 0.0, 0.95)  # arm forward 90°
    a = compute_shoulder_angle(lms, "left", "flexion")
    assert _close(a, 90.0), f"dispatcher flex test got {a}"
    print(f"  dispatcher: left flexion forward 90°  : {a:+.2f}°  (expected +90)")

    print("\nAll shoulder_engine sanity checks passed.")
    sys.exit(0)
