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

# Merged abduction + adduction. Patient is in FRONTAL view (arms
# at side, camera facing torso); one recording captures both
# directions. Direction is detected geometrically (elbow position
# vs test-side shoulder, with a y-axis override for the overhead
# end of the abduction arc); magnitude is the direction-symmetric
# abs() angle the existing shoulder_abduction_adduction formula
# returns. AAOS reference ranges.
_MERGED_ABAD_PRIMARY_TARGET: tuple[float, float] = (150.0, 180.0)   # Abduction
_MERGED_ABAD_SECONDARY_TARGET: tuple[float, float] = (30.0, 50.0)   # Adduction

# Direction-detection knobs — mirror the browser merged-ab/ad code
# in motionlens-web/lib/biomech/shoulder.ts so live and upload modes
# agree frame-by-frame on what counts as abduction vs adduction.
_ABAD_OVERHEAD_FRAC = 0.20            # elbow above shoulder by 20%
                                       # of shoulder-width triggers
                                       # the overhead branch (default
                                       # "abduction" unless the
                                       # lateral signal is clearly
                                       # medial — see below).
_ABAD_DIRECTION_DEADBAND_FRAC = 0.01  # Minimum lateral offset (as
                                       # fraction of shoulder width)
                                       # before any direction is
                                       # committed at NORMAL (below-
                                       # overhead) elevation. Set to
                                       # 1% so patients with limited
                                       # mobility — who may only
                                       # achieve a few degrees of
                                       # adduction — still register
                                       # a peak rather than being
                                       # silently absorbed by the
                                       # deadband as "neutral".
_ABAD_OVERHEAD_MEDIAL_FRAC = 0.02     # At overhead elevation, declare
                                       # "adduction" when the elbow
                                       # is at least 2% of shoulder
                                       # width medial of the test-
                                       # side shoulder. Wider than
                                       # the at-shoulder deadband to
                                       # suppress true angle-sign
                                       # flip noise at the top of
                                       # the abduction arc, but
                                       # loose enough that a small-
                                       # amplitude overhead-adduction
                                       # motion (arm raised slightly
                                       # across body) classifies as
                                       # adduction rather than being
                                       # absorbed by the default
                                       # "abduction" override.
_ABAD_MIN_MOTION_MAGNITUDE_DEG = 2.0  # Don't commit a direction when
                                       # the raw arm-vs-trunk angle
                                       # is below this — postural
                                       # sway / breathing produces
                                       # sub-2° displacements at rest
                                       # that we don't want sneaking
                                       # into either peak slot. Above
                                       # this floor, even a tiny
                                       # genuine adduction (~3-5°)
                                       # registers correctly.

# Anatomical max magnitudes used as final clamps before peak update.
# The deadband + overhead override catch most flip artefacts; this
# is the belt-and-suspenders cap so an upstream noise spike never
# locks an implausibly large adduction (e.g. 90°) as the peak.
_ABDUCTION_ANATOMICAL_MAX = 180.0
_ADDUCTION_ANATOMICAL_MAX = 50.0


# Merged rotation — LATERAL-VIEW method.
# Patient stands SIDE-ON to the camera. Arm abducted to 90° (upper
# arm horizontal, projecting along the camera axis in the lateral
# view). Elbow bent 90°; forearm sweeps in a plane parallel to the
# image plane.
#
#   • Forearm horizontal forward      → 0° clinical (neutral)
#   • Forearm rotating UP toward head → External Rotation (60–90°)
#   • Forearm rotating DOWN toward floor → Internal Rotation (60–80°)
#
# Per-frame math (no calibration baseline required):
#
#     reference     = (0, 1)              # straight down, image y-down
#     forearm       = (wrist − elbow)
#     raw_angle     = |angle_between(reference, forearm)|     # 0–180°
#     magnitude_deg = |raw_angle − 90°|                       # 0–90° clinical
#     direction     = "external" if wrist.y < elbow.y else "internal"
#
# The clinical conversion (subtract 90°) reports the angle FROM
# horizontal, matching the AAOS ER 0–90° and IR 0–80° conventions
# the report header and target ranges already advertise.
#
# Mirrors motionlens-web/lib/biomech/shoulder.ts so live and upload
# modes produce the same per-frame numbers.
_MERGED_ROT_PRIMARY_TARGET: tuple[float, float] = (60.0, 90.0)   # External
_MERGED_ROT_SECONDARY_TARGET: tuple[float, float] = (60.0, 80.0) # Internal

# Direction deadband on (wrist.y − elbow.y) as a fraction of shoulder
# width — scale-invariant. Suppresses flips at the wrist≈elbow.y
# horizontal crossover.
_ROT_DIRECTION_DEADBAND_FRAC = 0.03

# Forearm vector below this many pixels is too short to derive a
# reliable angle — the peak tracker holds the last good value.
_ROT_MIN_FOREARM_PX = 4.0


def _classify_in_range(value: float, lo: float, hi: float) -> str:
    """Asymmetric range-aware classification for clinical ROM:

      • In range                       → good
      • Below range by ≤ 30% of width  → fair  (mild restriction)
      • Below range by > 30% of width  → poor  (notable restriction)
      • Above range by ≤ 30% of width  → good  (normal variation /
                                                 hypermobility — not
                                                 a clinical concern
                                                 for ROM screening)
      • Above range by 30-100% of width → fair (notable hypermobility)
      • Above range by > 100% of width  → poor (anatomically suspect;
                                                  likely a measurement
                                                  artefact / sign flip)

    The asymmetry matters: a patient with 70° shoulder extension when
    the normal-range upper bound is 60° has BETTER ROM than the spec
    — they are not impaired — so calling it "poor" gives the operator
    the opposite of the clinical truth. Restricted ROM (below range)
    is the side that flags impairment.
    """
    if lo <= value <= hi:
        return "good"
    width = max(1.0, hi - lo)
    if value < lo:
        dist_frac = (lo - value) / width
        return "fair" if dist_frac <= 0.30 else "poor"
    # value > hi  → exceeding the normal range
    dist_frac = (value - hi) / width
    if dist_frac <= 0.30:
        return "good"
    if dist_frac <= 1.00:
        return "fair"
    return "poor"


def _detect_ab_ad_direction(
    s_x: float, s_y: float,
    e_x: float, e_y: float,
    w_x: Optional[float], w_y: Optional[float],
    ls_x: float, rs_x: float,
    side: str,
) -> Optional[str]:
    """Classify one frame as 'abduction' / 'adduction' / None.

    Direction signal: WRIST position (when visible) relative to the
    test-side shoulder and body centreline. Wrist is the distal end
    of the arm chain — when a patient with limited ROM bends the
    elbow and brings the hand across the body, the wrist crosses
    the body midline well before the elbow does (the elbow can stay
    near the shoulder x). Using wrist makes the detector sensitive
    to genuine small-amplitude adduction motions that elbow-only
    detection would silently miss. Falls back to elbow when wrist
    visibility drops (e.g. at the extreme of overhead abduction).

    Elevation signal: ELBOW height above shoulder — drives the
    overhead override that suppresses dx-sign flips at the top of
    the abduction arc, where any shoulder-x/elbow-x noise can flip
    the detector. Anatomically the overhead arc is always on the
    abduction range, so we lock direction to 'abduction' there.

    Rules (in order):
      1. Overhead (dy_ratio_elbow > _ABAD_OVERHEAD_FRAC) → always
         'abduction'.
      2. Below-overhead — use the wrist's signed lateral offset
         from the test-side shoulder (scaled by shoulder width).
         Outward → abduction, medial → adduction.
      3. Inside deadband (|dx_ratio| < _ABAD_DIRECTION_DEADBAND_FRAC)
         → None (neutral pose, neither peak updated).
    """
    _ = side  # outward direction is inferred from s.x vs centreline.
    _ = w_y   # only x matters for direction; y kept in signature for
              # symmetry / future use.
    shoulder_width = abs(rs_x - ls_x)
    if shoulder_width < 1e-4:
        return None
    centre_x = (ls_x + rs_x) / 2.0
    if s_x > centre_x:
        outward_sign = 1.0
    elif s_x < centre_x:
        outward_sign = -1.0
    else:
        return None
    # Image y axis grows DOWNWARD, so (s_y - e_y) > 0 means the
    # elbow is above the shoulder on screen.
    dy_ratio = (s_y - e_y) / shoulder_width
    if dy_ratio > _ABAD_OVERHEAD_FRAC:
        # Overhead arc — locked to abduction (see docstring).
        return "abduction"
    # Direction signal: prefer WRIST (more distal, more sensitive
    # to small bent-arm adduction). Fall back to elbow when the
    # caller couldn't pass a confident wrist coordinate (low
    # visibility). The shoulder reference stays the test-side
    # shoulder either way.
    if w_x is not None:
        dx_ratio = ((w_x - s_x) * outward_sign) / shoulder_width
    else:
        dx_ratio = ((e_x - s_x) * outward_sign) / shoulder_width
    if abs(dx_ratio) < _ABAD_DIRECTION_DEADBAND_FRAC:
        return None
    return "abduction" if dx_ratio > 0 else "adduction"


# _is_rotation_neutral_pose removed — the lateral-view method does
# not require a neutral-pose calibration baseline.


def _detect_rotation_direction(
    e_y: float, w_y: float,
    shoulder_width: float,
) -> Optional[str]:
    """Lateral-view direction detector.
      • Wrist ABOVE elbow on screen (w_y < e_y)  → external
      • Wrist BELOW elbow on screen (w_y > e_y)  → internal
      • Within the y-deadband → None.
    Same left/right convention for both sides (purely vertical
    comparison; no centreline math needed)."""
    if shoulder_width < 1e-4:
        return None
    dy_ratio = (w_y - e_y) / shoulder_width
    if abs(dy_ratio) < _ROT_DIRECTION_DEADBAND_FRAC:
        return None
    # Image y grows DOWNWARD: dy_ratio < 0 ⇒ wrist above elbow.
    return "external" if dy_ratio < 0 else "internal"


def _lateral_view_for_ab_ad(ts: dict) -> Optional[str]:
    """Pre-flight check for the merged ab/ad test. The math (elbow
    position vs body centreline) assumes a FRONTAL view; in LATERAL
    profile both shoulders project to similar image-x and the
    centreline collapses, producing the spurious 50° clamp.

    Detect lateral view from body GEOMETRY rather than nose position
    (the nose moves around with head tilt and produced false-
    positive rejections on perfectly frontal videos). The signature
    of lateral view is that the projected SHOULDER WIDTH (left-to-
    right shoulder horizontal distance) is small relative to TRUNK
    HEIGHT (shoulder-to-hip vertical distance). In frontal pose the
    ratio is typically 0.45–0.70 (~half of trunk height); in lateral
    pose it drops well below 0.25 because the body's depth is much
    smaller than its width. Threshold is set conservatively so only
    clearly-lateral videos get rejected."""
    ls_entry = ts.get("left_shoulder")  or {}
    rs_entry = ts.get("right_shoulder") or {}
    lh_entry = ts.get("left_hip")  or {}
    rh_entry = ts.get("right_hip") or {}
    ls_x = ls_entry.get("x_px"); ls_y = ls_entry.get("y_px"); ls_v = ls_entry.get("vis")
    rs_x = rs_entry.get("x_px"); rs_y = rs_entry.get("y_px"); rs_v = rs_entry.get("vis")
    lh_y = lh_entry.get("y_px"); lh_v = lh_entry.get("vis")
    rh_y = rh_entry.get("y_px"); rh_v = rh_entry.get("vis")
    for arr in (ls_x, ls_y, rs_x, rs_y, lh_y, rh_y, ls_v, rs_v, lh_v, rh_v):
        if arr is None:
            return None
    n = min(len(ls_x), len(rs_x), len(lh_y), len(rh_y))
    if n == 0:
        return None
    ratios = []
    for i in range(n):
        if ls_v[i] < _SHOULDER_VIS_THRESHOLD: continue
        if rs_v[i] < _SHOULDER_VIS_THRESHOLD: continue
        if lh_v[i] < _SHOULDER_VIS_THRESHOLD: continue
        if rh_v[i] < _SHOULDER_VIS_THRESHOLD: continue
        sw = abs(float(rs_x[i]) - float(ls_x[i]))
        sm_y = (float(ls_y[i]) + float(rs_y[i])) / 2.0
        hm_y = (float(lh_y[i]) + float(rh_y[i])) / 2.0
        trunk_h = abs(hm_y - sm_y)
        if trunk_h < 1e-4:
            continue
        ratios.append(sw / trunk_h)
    if not ratios:
        return None
    ratios.sort()
    med_ratio = ratios[len(ratios) // 2]
    # Frontal: ratio 0.45–0.70. Lateral: <0.25. Threshold 0.22 is
    # conservative — only clearly-lateral profiles get rejected so
    # we don't false-positive on baggy clothes / slightly-turned
    # frontal poses.
    if med_ratio < 0.22:
        return (
            "Camera angle appears to be a side view, but the "
            "abduction/adduction test needs a FRONT-FACING view of "
            "the patient. Please re-record with the camera directly "
            "in front."
        )
    return None


def _wrong_side_for_video(ts: dict, side: str) -> Optional[str]:
    """Pre-flight sanity check: if the requested side's elbow has
    consistently lower visibility than the other side's, the user
    almost certainly picked the wrong side in the UI. Returns a
    user-facing error string, or None when the selection is
    plausibly correct.

    Thresholds are conservative — only fires when the visibility
    gap is clear, so borderline lateral views (one side modestly
    occluded) don't false-positive."""
    # `vis` is a numpy array — use explicit None / len checks (a
    # numpy `array or default` raises "ambiguous truth value").
    l_entry = ts.get("left_elbow") or {}
    r_entry = ts.get("right_elbow") or {}
    l_ve = l_entry.get("vis")
    r_ve = r_entry.get("vis")
    if l_ve is None or r_ve is None:
        return None
    n = min(len(l_ve), len(r_ve))
    if n == 0:
        return None
    l_mean = float(np.mean(l_ve[:n]))
    r_mean = float(np.mean(r_ve[:n]))
    requested_mean = l_mean if side == "left" else r_mean
    other_mean = r_mean if side == "left" else l_mean
    other_side = "right" if side == "left" else "left"
    if (
        requested_mean < 0.4
        and other_mean > 0.6
        and (other_mean - requested_mean) > 0.25
    ):
        return (
            f"Requested side '{side}' has lower visibility than the other "
            f"side. Please re-record showing the correct arm clearly, or "
            f"switch to the '{other_side}' side."
        )
    return None


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
    # Rotation handling: extract_poses now stores the pose-based
    # rotation it applied to the keypoints in raw["_pose_rotation"].
    # The screenshot frame is re-read straight from cv2, so we need
    # to apply the SAME rotation here to keep the JPEG and the
    # keypoint overlay aligned.
    from engines.gait_engine import apply_rotation as _apply_rot
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

    # Import LM here to avoid a circular import at module load.
    from engines.gait_engine import LM

    def _draw_dot(name: str) -> Optional[tuple[int, int]]:
        frames = keypoints_normalized.get(name, [])
        if frame_index >= len(frames):
            return None
        kp = frames[frame_index]
        if kp is None:
            return None
        # Keypoints are already in upright space (extract_poses
        # rotated them when needed), so just project into the
        # rotated frame's pixel dimensions directly.
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


# ─── Compensatory movement detection (Shoulder Flexion+Extension) ──
#
# Mirror of the TS tracker in lib/biomech/shoulder.ts. Three checks:
#
#   • Trunk Lean (HIGH 🔴): angle between shoulder→hip line and
#     image vertical. Baseline = mean of first 10 valid frames. Flag
#     if any frame deviates > 15° from baseline.
#
#   • Shoulder Elevation (MEDIUM 🟡): pixel distance between the
#     test-side shoulder and hip landmarks. Baseline = mean of first
#     10 valid frames. Flag if distance drops > 15 px below baseline.
#
#   • Elbow Bend (MEDIUM 🟡): interior angle at the elbow
#     (shoulder→elbow→wrist). No baseline. Flag if angle < 160°.

_COMP_TRUNK_LEAN_THRESHOLD_DEG = 15.0
_COMP_SHOULDER_ELEVATION_THRESHOLD_PX = 15.0
_COMP_ELBOW_BEND_THRESHOLD_DEG = 160.0
_COMP_BASELINE_FRAME_COUNT = 10


def _angle_from_vertical(dx: float, dy: float) -> float:
    """Angle (degrees) between vector (dx, dy) and the image-up
    reference (0, -1). Image y grows DOWNWARD, so for an upright
    trunk (shoulder above hip) dy < 0 and the result ≈ 0°. Forward
    and backward leans both increase the value (absolute deviation).
    """
    return abs(math.degrees(math.atan2(dx, -dy)))


def _interior_angle_deg(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float,
) -> float:
    """Interior angle at vertex B for points A-B-C, in degrees,
    in [0, 180]."""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    cross = v1x * v2y - v1y * v2x
    return math.degrees(math.atan2(abs(cross), dot))


_COMP_ELBOW_PEAK_WINDOW_FRAMES = 3


def _track_flexion_extension_compensations(
    sx, sy, vs,
    ex, ey, ve,
    hx, hy, vh,
    wx, wy, vw,
    nx, ny, nv,
    n: int,
    primary_peak_idx: int = -1,
    secondary_peak_idx: int = -1,
) -> list[dict]:
    """Return 3 compensation entries for the merged shoulder
    flexion+extension test. Each entry has `flagged: bool`; the
    report renderer decides green-check vs colored-card display.

    Landmark choices:
      • Trunk lean   → NOSE → HIP. The shoulder acromion moves
        substantially during shoulder ROM tests (scapular elevation
        + protraction), so a shoulder→hip line would conflate trunk
        motion with shoulder motion. The nose stays put as the arm
        moves, so a nose→hip line rotates only when the trunk does.
      • Shoulder elevation (shrug) → SHOULDER → HIP. The shrug is
        defined by the shoulder rising toward the head, so the
        shoulder landmark is the right place to measure it.
      • Elbow bend → SHOULDER → ELBOW → WRIST. Standard interior
        angle at the elbow joint."""
    trunk_baseline_samples: list[float] = []
    hip_dist_baseline_samples: list[float] = []
    trunk_baseline: float | None = None
    hip_dist_baseline: float | None = None
    trunk_lean_peak = 0.0
    shoulder_rise_peak = 0.0
    elbow_bend_peak = 180.0
    trunk_flagged = False
    elev_flagged = False
    bend_flagged = False

    for i in range(n):
        # Trunk lean — NOSE → HIP. Decoupled from shoulder motion.
        if nv[i] >= _SHOULDER_VIS_THRESHOLD and vh[i] >= _SHOULDER_VIS_THRESHOLD:
            nose_dx = float(nx[i]) - float(hx[i])
            nose_dy = float(ny[i]) - float(hy[i])
            if math.hypot(nose_dx, nose_dy) >= 1e-4:
                trunk = _angle_from_vertical(nose_dx, nose_dy)
                if len(trunk_baseline_samples) < _COMP_BASELINE_FRAME_COUNT:
                    trunk_baseline_samples.append(trunk)
                    if len(trunk_baseline_samples) == _COMP_BASELINE_FRAME_COUNT:
                        trunk_baseline = sum(trunk_baseline_samples) / _COMP_BASELINE_FRAME_COUNT
                if trunk_baseline is not None:
                    deviation = abs(trunk - trunk_baseline)
                    if deviation > trunk_lean_peak:
                        trunk_lean_peak = deviation
                    if deviation > _COMP_TRUNK_LEAN_THRESHOLD_DEG:
                        trunk_flagged = True

        # Shoulder elevation (shrug) — SHOULDER → HIP. Measures the
        # actual shrug at its anatomical location.
        if vs[i] >= _SHOULDER_VIS_THRESHOLD and vh[i] >= _SHOULDER_VIS_THRESHOLD:
            dx = float(sx[i]) - float(hx[i])
            dy = float(sy[i]) - float(hy[i])
            if math.hypot(dx, dy) >= 1e-4:
                hip_dist = math.hypot(dx, dy)
                if len(hip_dist_baseline_samples) < _COMP_BASELINE_FRAME_COUNT:
                    hip_dist_baseline_samples.append(hip_dist)
                    if len(hip_dist_baseline_samples) == _COMP_BASELINE_FRAME_COUNT:
                        hip_dist_baseline = sum(hip_dist_baseline_samples) / _COMP_BASELINE_FRAME_COUNT
                if hip_dist_baseline is not None:
                    # Shrug: shoulder rises away from hip → euclidean
                    # shoulder→hip distance INCREASES.
                    rise = hip_dist - hip_dist_baseline
                    if rise > shoulder_rise_peak:
                        shoulder_rise_peak = rise
                    if rise > _COMP_SHOULDER_ELEVATION_THRESHOLD_PX:
                        elev_flagged = True

    # Elbow bend — PEAK-WINDOWED check. Instead of flagging any single
    # frame below threshold (which fires on transit-motion bends + on
    # single-frame landmark jitter), collect per-frame elbow angles
    # and only check the small window of frames AROUND the flexion
    # and extension peaks. Clinically a compensation only matters AT
    # the peak — bending the elbow during the lowering transit is
    # not a compensation and should not be flagged. Two-pass loop.
    elbow_angles: list[float | None] = []
    for i in range(n):
        if (
            vs[i] >= _SHOULDER_VIS_THRESHOLD
            and ve[i] >= _SHOULDER_VIS_THRESHOLD
            and vw[i] >= _SHOULDER_VIS_THRESHOLD
        ):
            elbow_angles.append(_interior_angle_deg(
                float(sx[i]), float(sy[i]),
                float(ex[i]), float(ey[i]),
                float(wx[i]), float(wy[i]),
            ))
        else:
            elbow_angles.append(None)

    # Check elbow only in ±_COMP_ELBOW_PEAK_WINDOW_FRAMES around each
    # tracked peak (flexion = primary_peak_idx, extension = secondary).
    # Index < 0 means that direction wasn't captured this trial; skip.
    for peak_idx in (primary_peak_idx, secondary_peak_idx):
        if peak_idx < 0:
            continue
        lo = max(0, peak_idx - _COMP_ELBOW_PEAK_WINDOW_FRAMES)
        hi = min(n, peak_idx + _COMP_ELBOW_PEAK_WINDOW_FRAMES + 1)
        for i in range(lo, hi):
            ang = elbow_angles[i]
            if ang is None:
                continue
            if ang < elbow_bend_peak:
                elbow_bend_peak = ang
            if ang < _COMP_ELBOW_BEND_THRESHOLD_DEG:
                bend_flagged = True

    # Always populate `details` with the actual peak measured value
    # (even when not flagged), so the report can show diagnostic
    # readings under each compensation. Lets clinicians see WHERE the
    # measurement landed relative to threshold — "lean was 8° / 15°"
    # is much more actionable than a silent "not detected".
    return [
        {
            "type": "trunk_lean",
            "label": "Trunk Lean",
            "severity": "high",
            "flagged": trunk_flagged,
            "details": (
                f"Peak deviation {trunk_lean_peak:.1f}° from baseline "
                f"(threshold {_COMP_TRUNK_LEAN_THRESHOLD_DEG:.0f}°)"
            ),
        },
        {
            "type": "shoulder_elevation",
            "label": "Shoulder Elevation",
            "severity": "medium",
            "flagged": elev_flagged,
            "details": (
                f"Shoulder-hip distance rose {shoulder_rise_peak:.0f} px "
                f"above baseline (threshold "
                f"{_COMP_SHOULDER_ELEVATION_THRESHOLD_PX:.0f} px)"
            ),
        },
        {
            "type": "elbow_bend",
            "label": "Elbow Bend",
            "severity": "medium",
            "flagged": bend_flagged,
            "details": (
                f"Minimum elbow angle at peak: {elbow_bend_peak:.0f}° "
                f"(threshold {_COMP_ELBOW_BEND_THRESHOLD_DEG:.0f}°)"
            ),
        },
    ]


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
    # Two merged tests share the same upload pipeline but have
    # different direction-detection + reference-range structures.
    # Keep them as separate booleans so downstream branches stay
    # explicit (and "is_merged" can still gate the not-single-
    # direction path uniformly).
    is_merged_flex_ext = movement == "flexion_extension"
    is_merged_ab_ad    = movement == "abduction_adduction"
    is_merged_rotation = movement == "rotation"
    is_merged = is_merged_flex_ext or is_merged_ab_ad or is_merged_rotation
    if not is_merged and movement not in SHOULDER_NORMAL_RANGES:
        raise ValueError(f"Unsupported shoulder movement: {movement!r}")

    # Import gait pipeline here so the legacy math-only helpers above
    # don't pull in MediaPipe / OpenCV at module load (preserves the
    # existing __main__ self-test which only needs the math).
    from engines.gait_engine import extract_poses, build_time_series

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # Pre-flight sanity checks on the merged ab/ad and rotation
    # paths — fail fast with clear, user-actionable error messages
    # rather than running a full analysis that would produce
    # garbage values. Lateral-view rejection NOW applies to ab/ad
    # ONLY: the lateral-view rotation method REQUIRES side-on
    # framing (forearm sweeps in a plane parallel to the image),
    # so a frontal-only gate would block the legitimate case.
    if is_merged_ab_ad:
        lateral_msg = _lateral_view_for_ab_ad(ts)
        if lateral_msg:
            raise ValueError(lateral_msg)
    if is_merged_ab_ad or is_merged_rotation:
        # Wrong-side selection. If the requested side's elbow is
        # far less visible than the other side's across the run,
        # the user almost certainly picked the wrong side.
        wrong_side_msg = _wrong_side_for_video(ts, side)
        if wrong_side_msg:
            raise ValueError(wrong_side_msg)

    shoulder_key = f"{side}_shoulder"
    elbow_key    = f"{side}_elbow"
    hip_key      = f"{side}_hip"
    wrist_key    = f"{side}_wrist"

    sx = ts[shoulder_key]["x_px"]; sy = ts[shoulder_key]["y_px"]; vs = ts[shoulder_key]["vis"]
    ex = ts[elbow_key]["x_px"];    ey = ts[elbow_key]["y_px"];    ve = ts[elbow_key]["vis"]
    hx = ts[hip_key]["x_px"];      hy = ts[hip_key]["y_px"];      vh = ts[hip_key]["vis"]
    wx = ts[wrist_key]["x_px"];    wy = ts[wrist_key]["y_px"];    vw = ts[wrist_key]["vis"]

    # ══════════════════════════════════════════════════════════════
    # Merged rotation (external + internal) — LATERAL-VIEW method
    # ══════════════════════════════════════════════════════════════
    # Patient SIDE-ON, arm abducted 90° (upper arm horizontal along
    # camera axis), elbow bent 90°. Forearm sweeps in the image
    # plane. Per-frame magnitude is the angle between the elbow→
    # wrist vector and the straight-down reference (0, 1); direction
    # is a pure y-comparison between wrist and elbow. Primary slot =
    # External (matches the frontend SHOULDER_MOVEMENTS labelling
    # after STEP 1 alignment).
    # Returns early so the existing per-frame angle loop below
    # (which assumes one of the shoulder_*_extension formulas) is
    # bypassed for rotation.
    if is_merged_rotation:
        import logging as _logging
        _rotlog = _logging.getLogger("motionlens.shoulder")

        lsx_arr = ts["left_shoulder"]["x_px"]
        lsv_arr = ts["left_shoulder"]["vis"]
        rsx_arr = ts["right_shoulder"]["x_px"]
        rsv_arr = ts["right_shoulder"]["vis"]

        n_rot = int(min(len(sx), len(ex), len(wx),
                        len(lsx_arr), len(rsx_arr)))

        primary_peak_mag = 0.0      # external rotation
        primary_peak_idx = -1
        secondary_peak_mag = 0.0    # internal rotation
        secondary_peak_idx = -1
        valid_frames_rot = 0
        n_internal = 0
        n_external = 0
        n_deadband = 0
        n_skipped_vis = 0

        for i in range(n_rot):
            # Need elbow + wrist for magnitude + direction; both
            # shoulders supply the shoulder-width unit for the
            # direction deadband.
            if (ve[i] < _SHOULDER_VIS_THRESHOLD
                    or vw[i] < _SHOULDER_VIS_THRESHOLD
                    or lsv_arr[i] < _SHOULDER_VIS_THRESHOLD
                    or rsv_arr[i] < _SHOULDER_VIS_THRESHOLD):
                n_skipped_vis += 1
                continue
            e_x = float(ex[i]); e_y = float(ey[i])
            w_x = float(wx[i]); w_y = float(wy[i])
            ls_x = float(lsx_arr[i]); rs_x = float(rsx_arr[i])
            shoulder_width = abs(rs_x - ls_x)

            ax = w_x - e_x
            ay = w_y - e_y
            if math.hypot(ax, ay) < _ROT_MIN_FOREARM_PX:
                n_skipped_vis += 1
                continue
            # angle_between((0, 1), (ax, ay)):
            #   dot = ay, cross = -ax.
            # Raw angle is in [0°, 180°] from forearm-down. Clinical
            # ER/IR is measured FROM HORIZONTAL (anatomical neutral
            # at shoulder abducted 90°), so subtract 90° and take
            # |·|: forearm-horizontal → 0°, forearm-vertical-up
            # → 90° ER, forearm-vertical-down → 90° IR.
            raw_angle = abs(math.degrees(math.atan2(-ax, ay)))
            magnitude = abs(raw_angle - 90.0)

            direction = _detect_rotation_direction(e_y, w_y, shoulder_width)
            if direction is None:
                n_deadband += 1
                continue
            valid_frames_rot += 1
            if direction == "external":
                n_external += 1
                if magnitude > primary_peak_mag:
                    primary_peak_mag = magnitude
                    primary_peak_idx = i
            else:  # internal
                n_internal += 1
                if magnitude > secondary_peak_mag:
                    secondary_peak_mag = magnitude
                    secondary_peak_idx = i

        _rotlog.info(
            "rotation summary: side=%s frames=%d valid=%d external=%d "
            "internal=%d deadband=%d skip_vis=%d peak_ext=%.1f "
            "peak_int=%.1f",
            side, n_rot, valid_frames_rot, n_external, n_internal,
            n_deadband, n_skipped_vis,
            primary_peak_mag, secondary_peak_mag,
        )

        if valid_frames_rot < max(3, int(fps * 0.5)):
            raise ValueError("poor_visibility")

        p_lo, p_hi = _MERGED_ROT_PRIMARY_TARGET
        s_lo, s_hi = _MERGED_ROT_SECONDARY_TARGET
        p_target = p_hi
        p_pct = (primary_peak_mag / p_target) * 100.0 if p_target > 0 else 0.0
        p_status = _classify_in_range(primary_peak_mag, p_lo, p_hi)

        interpretation_primary = (
            f"External rotation ({side.capitalize()}) measured "
            f"{primary_peak_mag:.1f}°, which is {p_pct:.0f}% of the "
            f"{p_lo:.0f}°–{p_hi:.0f}° normal range — {p_status}."
        )
        if secondary_peak_mag > 0:
            s_status = _classify_in_range(secondary_peak_mag, s_lo, s_hi)
            interpretation_secondary = (
                f"Internal rotation ({side.capitalize()}) measured "
                f"{secondary_peak_mag:.1f}°, which is "
                f"{(secondary_peak_mag / s_hi) * 100.0:.0f}% of the "
                f"{s_lo:.0f}°–{s_hi:.0f}° normal range — {s_status}."
            )
        else:
            interpretation_secondary = (
                "Internal rotation direction was not detected in this "
                "recording."
            )
        interpretation = f"{interpretation_primary} {interpretation_secondary}"

        # Two key frames only — peak internal + peak external. No
        # neutral frame on the rotation report (matches the merged
        # ab/ad layout user picked: only the peaks matter for the
        # dual-direction comparison).
        rot_key_frames: list[dict] = []
        if primary_peak_idx >= 0 and primary_peak_mag > 0:
            kf = _grab_shoulder_key_frame(
                video_path, primary_peak_idx, raw,
                f"External rotation ({primary_peak_mag:.1f}°)", side,
            )
            if kf:
                rot_key_frames.append(kf)
        if secondary_peak_idx >= 0 and secondary_peak_mag > 0:
            kf = _grab_shoulder_key_frame(
                video_path, secondary_peak_idx, raw,
                f"Internal rotation ({secondary_peak_mag:.1f}°)", side,
            )
            if kf:
                rot_key_frames.append(kf)

        return {
            "body_part": "shoulder",
            "movement": "rotation",
            "side": side,
            "peak_angle": (
                float(primary_peak_mag) if primary_peak_mag > 0 else None
            ),
            "peak_magnitude": primary_peak_mag,
            "reference_range": [float(p_lo), float(p_hi)],
            "target": float(p_target),
            "percentage": p_pct,
            "status": p_status,
            "valid_frames": valid_frames_rot,
            "total_frames": n_rot,
            "fps": float(fps),
            "interpretation": interpretation,
            "key_frames": rot_key_frames,
            "secondary_peak_angle": (
                float(secondary_peak_mag) if secondary_peak_mag > 0 else None
            ),
            "secondary_peak_magnitude": (
                secondary_peak_mag if secondary_peak_mag > 0 else None
            ),
            "secondary_reference_range": [float(s_lo), float(s_hi)],
            "primary_label": "External rotation",
            "secondary_label": "Internal rotation",
        }

    # Facing-direction sign for flexion/extension. The signed-angle
    # math in shoulder_flexion_extension assumes the patient faces
    # camera-right (nose to the right of shoulder midpoint in image
    # space) — under that assumption, arm-forward gives + (flexion)
    # and arm-backward gives − (extension). When the patient films
    # in the opposite profile (right-side test → patient faces image-
    # left), the sign convention inverts: arm-forward becomes − and
    # the system labels flexion frames as extension. Detect facing
    # once from the median nose/shoulder positions and apply as a
    # multiplier to every per-frame angle. is_merged + single-direction
    # flex/ext both need this; ab/ad and rotation don't (those are
    # frontal-plane / axial and don't have a forward/back sign).
    is_flex_ext = is_merged_flex_ext or movement in ("flexion", "extension")
    facing_sign = 1.0
    if is_flex_ext and "nose" in ts:
        nx_arr = ts["nose"]["x_px"]; nv_arr = ts["nose"]["vis"]
        lsx_arr = ts["left_shoulder"]["x_px"]; lsv_arr = ts["left_shoulder"]["vis"]
        rsx_arr = ts["right_shoulder"]["x_px"]; rsv_arr = ts["right_shoulder"]["vis"]
        deltas = []
        for i in range(min(len(nx_arr), len(lsx_arr), len(rsx_arr))):
            if nv_arr[i] < _SHOULDER_VIS_THRESHOLD:
                continue
            if lsv_arr[i] < _SHOULDER_VIS_THRESHOLD or rsv_arr[i] < _SHOULDER_VIS_THRESHOLD:
                continue
            sm_x = (float(lsx_arr[i]) + float(rsx_arr[i])) / 2.0
            deltas.append(float(nx_arr[i]) - sm_x)
        if deltas:
            deltas.sort()
            med_delta = deltas[len(deltas) // 2]
            # Negative median delta (nose to the LEFT of shoulder
            # midpoint) → patient faces image-left → invert sign.
            if med_delta < 0:
                facing_sign = -1.0
        import logging as _logging
        _logging.getLogger("motionlens.shoulder").info(
            "facing detection: samples=%d med_nose-sm_delta=%s facing_sign=%+.0f",
            len(deltas),
            f"{deltas[len(deltas)//2]:.1f}" if deltas else "N/A",
            facing_sign,
        )

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
        if is_merged_flex_ext or movement in ("flexion", "extension"):
            a = shoulder_flexion_extension(s, e, h) * facing_sign
        elif is_merged_ab_ad or movement in ("abduction", "adduction"):
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

    # ── Merged abduction + adduction ───────────────────────────
    # Magnitude is direction-symmetric (shoulder_abduction_adduction
    # returns abs() between trunk and arm), so per-frame magnitude is
    # already in `angles` from the loop above. Direction comes from
    # _detect_ab_ad_direction (elbow position vs test-side shoulder
    # + body centreline, with a y-axis override for the overhead end
    # of the abduction arc). Mirrors the browser merged-ab/ad path in
    # shoulder.ts so live and upload modes agree on direction frame-
    # by-frame. Returns its own response so the merged flex/ext
    # branch below only runs when is_merged_flex_ext == True.
    if is_merged_ab_ad:
        import logging as _logging
        _ablog = _logging.getLogger("motionlens.shoulder")
        lsx_arr = ts["left_shoulder"]["x_px"]
        lsv_arr = ts["left_shoulder"]["vis"]
        rsx_arr = ts["right_shoulder"]["x_px"]
        rsv_arr = ts["right_shoulder"]["vis"]

        primary_peak_mag = 0.0       # abduction (positive magnitude)
        primary_peak_idx = -1
        secondary_peak_mag = 0.0     # adduction (positive magnitude)
        secondary_peak_idx = -1
        ab_neutral_idx = -1
        ab_neutral_abs = math.inf

        # Direction-detection diagnostics. Track per-frame stats so
        # the HF Space log shows exactly why a given test's
        # secondary slot ended up empty (none classified, deadband,
        # visibility, etc.) — replaces guess-and-check debugging.
        n_classified_ab = 0
        n_classified_ad = 0
        n_deadband = 0
        n_skipped_vis = 0
        n_skipped_angle = 0
        sample_logged = 0
        # Capture min dx_ratio (most-medial elbow seen) across the
        # whole run so we can tell if the adduction motion ever
        # produced a clearly medial elbow at all. Also track the
        # dy_ratio at that same frame to distinguish "medial at
        # overhead" (would hit the overhead override) from "medial
        # at chest level" (clean adduction).
        min_dx_ratio = math.inf
        min_dx_at_dyr = 0.0
        min_dx_at_mag = 0.0
        max_dx_ratio = -math.inf

        for i, a in enumerate(angles):
            if a is None:
                n_skipped_angle += 1
                continue
            if lsv_arr[i] < _SHOULDER_VIS_THRESHOLD or rsv_arr[i] < _SHOULDER_VIS_THRESHOLD:
                n_skipped_vis += 1
                continue
            s_x = float(sx[i]); s_y = float(sy[i])
            e_x = float(ex[i]); e_y = float(ey[i])
            ls_x = float(lsx_arr[i]); rs_x = float(rsx_arr[i])
            # Wrist used for direction signal when visible — see
            # _detect_ab_ad_direction docstring (more sensitive to
            # bent-arm small-amplitude adduction than elbow).
            if vw[i] >= _SHOULDER_VIS_THRESHOLD:
                w_x: Optional[float] = float(wx[i])
                w_y: Optional[float] = float(wy[i])
            else:
                w_x = None
                w_y = None
            # Recompute dx_ratio here for logging stats — use the
            # SAME (wrist-or-elbow) point the detector uses so the
            # diagnostic numbers match the classification.
            sw = abs(rs_x - ls_x)
            if sw > 1e-4:
                cx = (ls_x + rs_x) / 2.0
                outsign = 1.0 if s_x > cx else (-1.0 if s_x < cx else 0.0)
                ref_x = w_x if w_x is not None else e_x
                dxr = ((ref_x - s_x) * outsign) / sw if outsign != 0.0 else 0.0
                dyr_now = (s_y - e_y) / sw
                if dxr < min_dx_ratio:
                    min_dx_ratio = dxr
                    min_dx_at_dyr = dyr_now
                    min_dx_at_mag = float(a)
                if dxr > max_dx_ratio: max_dx_ratio = dxr
            direction = _detect_ab_ad_direction(
                s_x, s_y, e_x, e_y, w_x, w_y, ls_x, rs_x, side,
            )
            # Magnitude floor: ignore frames where the arm has
            # barely moved from rest. Without this, postural sway
            # at neutral could pick up sub-degree "adductions" once
            # we widen the deadband for limited-ROM patients.
            if direction is not None and float(a) < _ABAD_MIN_MOTION_MAGNITUDE_DEG:
                direction = None
            # Sample-log a handful of frames near the magnitude peak
            # so we can see what the detector actually sees.
            if sample_logged < 6 and float(a) > 25.0:
                _ablog.info(
                    "ab_ad frame i=%d mag=%.1f dxr=%.3f dyr=%.3f dir=%s",
                    i, float(a),
                    dxr if sw > 1e-4 else float("nan"),
                    (s_y - e_y) / sw if sw > 1e-4 else float("nan"),
                    direction,
                )
                sample_logged += 1
            if direction is None:
                n_deadband += 1
                if abs(a) < ab_neutral_abs:
                    ab_neutral_abs = abs(a)
                    ab_neutral_idx = i
                continue
            if direction == "abduction":
                n_classified_ab += 1
                mag = min(_ABDUCTION_ANATOMICAL_MAX, max(0.0, float(a)))
                if mag > primary_peak_mag:
                    primary_peak_mag = mag
                    primary_peak_idx = i
            else:  # adduction
                n_classified_ad += 1
                mag = min(_ADDUCTION_ANATOMICAL_MAX, max(0.0, float(a)))
                if mag > secondary_peak_mag:
                    secondary_peak_mag = mag
                    secondary_peak_idx = i

        _ablog.info(
            "ab_ad summary: side=%s frames=%d ab=%d ad=%d deadband=%d "
            "skip_vis=%d skip_angle=%d dxr_range=[%.3f, %.3f] "
            "min_dx@dyr=%.3f min_dx@mag=%.1f peak_ab=%.1f peak_ad=%.1f",
            side, len(angles),
            n_classified_ab, n_classified_ad, n_deadband,
            n_skipped_vis, n_skipped_angle,
            min_dx_ratio if min_dx_ratio != math.inf else 0.0,
            max_dx_ratio if max_dx_ratio != -math.inf else 0.0,
            min_dx_at_dyr, min_dx_at_mag,
            primary_peak_mag, secondary_peak_mag,
        )

        p_lo, p_hi = _MERGED_ABAD_PRIMARY_TARGET
        s_lo, s_hi = _MERGED_ABAD_SECONDARY_TARGET
        p_target = p_hi
        p_pct = (primary_peak_mag / p_target) * 100.0 if p_target > 0 else 0.0
        p_status = _classify_in_range(primary_peak_mag, p_lo, p_hi)

        interpretation_primary = (
            f"Abduction ({side.capitalize()}) measured {primary_peak_mag:.1f}°, "
            f"which is {p_pct:.0f}% of the {p_lo:.0f}°–{p_hi:.0f}° normal range "
            f"— {p_status}."
        )
        if secondary_peak_mag > 0:
            s_status = _classify_in_range(secondary_peak_mag, s_lo, s_hi)
            interpretation_secondary = (
                f"Adduction ({side.capitalize()}) measured {secondary_peak_mag:.1f}°, "
                f"which is "
                f"{(secondary_peak_mag / s_hi) * 100.0:.0f}% of the "
                f"{s_lo:.0f}°–{s_hi:.0f}° normal range — {s_status}."
            )
        else:
            interpretation_secondary = (
                "Adduction direction was not detected in this recording."
            )
        interpretation = f"{interpretation_primary} {interpretation_secondary}"

        # Merged ab/ad key frames: only the two peak frames. The
        # neutral / rest pose is omitted on user request — for this
        # test the two peaks already convey the full ROM picture
        # (single-direction tests still include neutral; merged
        # flex/ext also keeps neutral for its dual-peak context).
        ab_key_frames: list[dict] = []
        if primary_peak_idx >= 0 and primary_peak_mag > 0:
            kf = _grab_shoulder_key_frame(
                video_path, primary_peak_idx, raw,
                f"Abduction ({primary_peak_mag:.1f}°)", side,
            )
            if kf:
                ab_key_frames.append(kf)
        if secondary_peak_idx >= 0 and secondary_peak_mag > 0:
            kf = _grab_shoulder_key_frame(
                video_path, secondary_peak_idx, raw,
                f"Adduction ({secondary_peak_mag:.1f}°)", side,
            )
            if kf:
                ab_key_frames.append(kf)

        return {
            "body_part": "shoulder",
            "movement": "abduction_adduction",
            "side": side,
            "peak_angle": float(primary_peak_mag) if primary_peak_mag > 0 else None,
            "peak_magnitude": primary_peak_mag,
            "reference_range": [float(p_lo), float(p_hi)],
            "target": float(p_target),
            "percentage": p_pct,
            "status": p_status,
            "valid_frames": valid_frames,
            "total_frames": n,
            "fps": float(fps),
            "interpretation": interpretation,
            "key_frames": ab_key_frames,
            "secondary_peak_angle": (
                float(secondary_peak_mag) if secondary_peak_mag > 0 else None
            ),
            "secondary_peak_magnitude": (
                secondary_peak_mag if secondary_peak_mag > 0 else None
            ),
            "secondary_reference_range": [float(s_lo), float(s_hi)],
            "primary_label": "Abduction",
            "secondary_label": "Adduction",
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

    # Compensatory-movement detection. Runs on the same per-frame
    # arrays the peak-angle loop above already iterated, but tracks
    # an independent set of signals (trunk lean, shoulder elevation,
    # elbow bend). Trunk lean uses NOSE → HIP rather than the
    # SHOULDER → HIP line specified in the original spec — the
    # shoulder acromion moves substantially during shoulder ROM
    # (scapular elevation + protraction) and contaminates the
    # signal; the nose stays put while the trunk actually rotates.
    # See `_track_flexion_extension_compensations` docstring.
    nx_arr = ts["nose"]["x_px"]
    ny_arr = ts["nose"]["y_px"]
    nv_arr = ts["nose"]["vis"]
    compensations = _track_flexion_extension_compensations(
        sx, sy, vs, ex, ey, ve, hx, hy, vh, wx, wy, vw,
        nx_arr, ny_arr, nv_arr, n,
        primary_peak_idx=primary_peak_idx,
        secondary_peak_idx=secondary_peak_idx,
    )

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
        "compensations": compensations,
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
