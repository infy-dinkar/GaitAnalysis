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
"""
from __future__ import annotations

import math
from typing import Sequence

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
