"""
neck_engine.py
Neck (cervical) range-of-motion angle math for the Biomechanical
Analysis flow.

Conventions
-----------
Coordinate system: MediaPipe Pose 2D image coords. Origin top-left,
x increases rightward, y increases DOWNWARD.

The shoulder midpoint (mean of LEFT_SHOULDER and RIGHT_SHOULDER) is
the trunk-top reference. The ear midpoint (mean of LEFT_EAR and
RIGHT_EAR) is the head-centre reference. The neck vector is
shoulder_mid → ear_mid; the head's tilt and rotation are derived
from that vector and the nose position.

Neutral = head upright, level. Each formula returns degrees.
Magnitude = degrees away from neutral; sign on flexion/extension =
forward (chin to chest, +) vs back (chin up, −) for a subject
facing camera-right.

Visibility
----------
`compute_neck_angle` returns None when ANY required landmark has
`visibility < vis_threshold` (default 0.5). The caller drops that
frame from the peak-tracker.

Limitation
----------
Rotation estimation from a single 2D camera is approximate (ratio
of nose offset to ear-midline width, mapped linearly to ±90°). A
goniometer is the clinical reference for precise rotation.
"""
from __future__ import annotations

import math
from typing import Sequence

# MediaPipe Pose landmark indices used for neck ROM
NOSE            = 0
LEFT_EAR        = 7
RIGHT_EAR       = 8
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12


# Clinical normal ranges for cervical ROM (AAOS / Penning).
NECK_NORMAL_RANGES = {
    "flexion":         {"range": (45, 80), "target": 60.0},
    "extension":       {"range": (50, 70), "target": 60.0},
    "lateral_flexion": {"range": (20, 45), "target": 35.0},
    "rotation":        {"range": (70, 90), "target": 80.0},
}


# ──────────────────────────────────────────────
# Math primitives
# ──────────────────────────────────────────────
def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Signed angle from v1 to v2 in degrees, in image-coord 2D space.
    Result is in (-180, +180].
    """
    dot   = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(cross, dot))


def _midpoint(p1: tuple[float, float],
              p2: tuple[float, float]) -> tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


# ──────────────────────────────────────────────
# Per-movement angle formulas
# ──────────────────────────────────────────────
def neck_flexion_extension(
    nose:         tuple[float, float],
    ear_mid:      tuple[float, float],
    shoulder_mid: tuple[float, float],
) -> float:
    """Sagittal (side view). Tilt of the (shoulder_mid → ear_mid)
    line from vertical. Positive = forward flexion (chin toward
    chest), negative = extension (chin up). Subject assumed to face
    camera-right.

    `nose` is unused here but accepted for API symmetry with the
    other neck formulas (the dispatcher always passes it).
    """
    del nose  # unused in flexion/extension
    neck_vec = (ear_mid[0] - shoulder_mid[0], ear_mid[1] - shoulder_mid[1])
    vertical = (0.0, -1.0)  # up in image coords
    return angle_between(vertical, neck_vec)


def neck_lateral_flexion(
    ear_mid:      tuple[float, float],
    shoulder_mid: tuple[float, float],
) -> float:
    """Front view. Tilt of the (shoulder_mid → ear_mid) line from
    vertical in the frontal plane. Returns positive magnitude
    regardless of side (the caller knows which side was assessed).
    """
    neck_vec = (ear_mid[0] - shoulder_mid[0], ear_mid[1] - shoulder_mid[1])
    vertical = (0.0, -1.0)
    return abs(angle_between(vertical, neck_vec))


def neck_rotation(
    nose:      tuple[float, float],
    left_ear:  tuple[float, float],
    right_ear: tuple[float, float],
    shoulder_mid: tuple[float, float],  # accepted but unused — kept for future
) -> float:
    """Approximate. Maps the lateral nose offset (relative to the
    ear midline, normalised by half the ear-to-ear width) to degrees
    via a linear approximation: ratio ±1 → ±90°. Returns positive
    magnitude.

    The 2D estimation is sensitive to camera angle; 90° is the
    practical ceiling. Surface this in the UI.
    """
    del shoulder_mid  # accepted for API symmetry, not needed in this approximation
    ear_mid_x = (left_ear[0] + right_ear[0]) / 2.0
    ear_width = abs(left_ear[0] - right_ear[0])
    if ear_width < 1e-3:
        return 0.0
    offset_ratio = (nose[0] - ear_mid_x) / (ear_width / 2.0)
    # Clamp to reasonable degree range
    return min(abs(offset_ratio * 90.0), 120.0)


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────
def compute_neck_angle(
    landmarks: Sequence,
    movement: str,
    vis_threshold: float = 0.5,
) -> float | None:
    """Compute the cervical ROM angle for one frame.

    Args:
        landmarks: sequence of 33 MediaPipe Pose landmarks. Each item
                   must expose `.x`, `.y`, `.visibility` (numeric).
        movement:  one of NECK_NORMAL_RANGES keys.

    Returns:
        Angle in degrees (positive magnitude for lateral_flexion +
        rotation; signed for flexion / extension), or None if any
        required landmark has visibility below `vis_threshold`.

    Note: the neck has no left/right side parameter — measurements
    use both ears + the shoulder midline for every movement.
    """
    if movement not in NECK_NORMAL_RANGES:
        raise ValueError(f"Unknown neck movement: {movement!r}")

    needed = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER]
    for k in needed:
        if landmarks[k].visibility < vis_threshold:
            return None

    nose        = (landmarks[NOSE].x,           landmarks[NOSE].y)
    left_ear    = (landmarks[LEFT_EAR].x,       landmarks[LEFT_EAR].y)
    right_ear   = (landmarks[RIGHT_EAR].x,      landmarks[RIGHT_EAR].y)
    left_shldr  = (landmarks[LEFT_SHOULDER].x,  landmarks[LEFT_SHOULDER].y)
    right_shldr = (landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y)
    ear_mid     = _midpoint(left_ear,   right_ear)
    shldr_mid   = _midpoint(left_shldr, right_shldr)

    if movement in ("flexion", "extension"):
        return neck_flexion_extension(nose, ear_mid, shldr_mid)
    if movement == "lateral_flexion":
        return neck_lateral_flexion(ear_mid, shldr_mid)
    # rotation
    return neck_rotation(nose, left_ear, right_ear, shldr_mid)


# ══════════════════════════════════════════════════════════════════
# Inline sanity tests — run with `python neck_engine.py`
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    def _close(actual, expected, tol=0.5):
        return abs(actual - expected) <= tol

    # 1. Neutral (head straight up): ear_mid above shoulder_mid by 1 unit.
    a = neck_flexion_extension(nose=(0, -1), ear_mid=(0, -1), shoulder_mid=(0, 0))
    assert _close(a, 0.0), f"neutral flexion should be 0, got {a}"
    print(f"  neutral upright                        : {a:+.2f}  (expected 0)")

    # 2. Forward flexion 30° (chin toward chest, head tilts forward
    #    so ear is forward of shoulder). With subject facing +x:
    #    ear_mid = (sin30, -cos30) ≈ (0.5, -0.866).
    a = neck_flexion_extension(nose=(0.5, -0.866),
                               ear_mid=(0.5, -0.866),
                               shoulder_mid=(0, 0))
    assert _close(a, 30.0), f"forward 30 should be +30, got {a}"
    print(f"  forward flexion 30                     : {a:+.2f}  (expected +30)")

    # 3. Extension 30° (chin up, ear backward of shoulder).
    a = neck_flexion_extension(nose=(-0.5, -0.866),
                               ear_mid=(-0.5, -0.866),
                               shoulder_mid=(0, 0))
    assert _close(a, -30.0), f"extension 30 should be -30, got {a}"
    print(f"  extension 30                           : {a:+.2f}  (expected -30)")

    # 4. Lateral flexion 30° to either side — magnitude only.
    a = neck_lateral_flexion(ear_mid=(0.5, -0.866), shoulder_mid=(0, 0))
    assert _close(a, 30.0), f"lateral 30 (right) should be 30, got {a}"
    print(f"  lateral flexion 30 (right)             : {a:+.2f}  (expected 30)")
    a = neck_lateral_flexion(ear_mid=(-0.5, -0.866), shoulder_mid=(0, 0))
    assert _close(a, 30.0), f"lateral 30 (left) should be 30, got {a}"
    print(f"  lateral flexion 30 (left)              : {a:+.2f}  (expected 30)")

    # 5. Rotation neutral: nose centered between ears.
    a = neck_rotation(nose=(0, 0), left_ear=(-0.5, 0), right_ear=(0.5, 0),
                       shoulder_mid=(0, 0.5))
    assert _close(a, 0.0), f"rotation neutral should be 0, got {a}"
    print(f"  rotation neutral                       : {a:+.2f}  (expected 0)")

    # 6. Rotation 90° (nose aligned with right ear, ratio = 1.0).
    a = neck_rotation(nose=(0.5, 0), left_ear=(-0.5, 0), right_ear=(0.5, 0),
                       shoulder_mid=(0, 0.5))
    assert _close(a, 90.0), f"rotation 90 should be 90, got {a}"
    print(f"  rotation 90 (nose at right ear)        : {a:+.2f}  (expected 90)")

    # 7. Rotation 45° (nose half-way to one ear, ratio = 0.5 → 45°).
    a = neck_rotation(nose=(0.25, 0), left_ear=(-0.5, 0), right_ear=(0.5, 0),
                       shoulder_mid=(0, 0.5))
    assert _close(a, 45.0), f"rotation 45 should be 45, got {a}"
    print(f"  rotation 45 (nose half-way to ear)     : {a:+.2f}  (expected 45)")

    # 8. Visibility gate: a low-vis nose returns None.
    class _LM:
        def __init__(self, x, y, v): self.x, self.y, self.visibility = x, y, v

    lms = [None] * 33
    for k, (x, y, v) in {
        NOSE:          (0.0, 0.5, 0.30),  # below threshold
        LEFT_EAR:      (-0.5, 0.4, 0.95),
        RIGHT_EAR:     (0.5, 0.4, 0.95),
        LEFT_SHOULDER: (-0.5, 0.8, 0.95),
        RIGHT_SHOULDER:(0.5, 0.8, 0.95),
    }.items():
        lms[k] = _LM(x, y, v)
    a = compute_neck_angle(lms, "rotation")
    assert a is None, f"low-vis nose should return None, got {a}"
    print(f"  low-vis nose => dispatcher None        : {a}  (expected None)")

    # 9. Dispatcher end-to-end: visibility ok, neutral position.
    lms[NOSE] = _LM(0.0, 0.5, 0.95)
    a = compute_neck_angle(lms, "rotation")
    assert _close(a, 0.0), f"dispatcher rotation neutral got {a}"
    print(f"  dispatcher: rotation neutral           : {a:+.2f}  (expected 0)")

    print("\nAll neck_engine sanity checks passed.")
    sys.exit(0)
