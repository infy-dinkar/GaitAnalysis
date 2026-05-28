"""
neck_engine.py
Neck (cervical) range-of-motion angle math for the Biomechanical
Analysis flow.

Two distinct surfaces live in this file:

  • Single-frame helpers (`compute_neck_angle` + the per-movement
    formulas below). Used by api.analyze_live_biomech_frame for the
    live-camera path.

  • Video pipeline (`analyze_neck` near the end of the file). Used
    by api.analyze_neck for the merged flexion+extension upload
    test. Reuses the shared gait pipeline (extract_poses +
    build_time_series), inherits the `_pose_rotation` portrait-
    video correction set by extract_poses, ships its own key-frame
    screenshot helper, returns the merged BiomechData DTO shape.

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

For the video pipeline, the flex/ext math uses the ear→nose tilt
formula (rather than the cervical-only shoulder→ear formula above)
to capture the full clinical 45-80° / 50-70° ROM including the
atlanto-occipital pitch — mirroring motionlens-web/lib/biomech/
neck.ts:computeNeckAngle. This requires a LATERAL (side-profile)
camera view; the pre-flight check rejects near-frontal uploads.

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

import base64
import math
from typing import Optional, Sequence

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
# Video pipeline (BlazePose Full, reuses gait pipeline)
# ══════════════════════════════════════════════════════════════════
#
# `analyze_neck` is the merged flex+ext upload entry point. Mirrors
# knee_engine.analyze_knee / ankle_engine.analyze_ankle / the merged
# shoulder branches:
#
#   • Reuses extract_poses + build_time_series from gait_engine
#     (shared pose model, smoothing, interpolation, platform-
#     independent rotation correction).
#   • Inherits raw["_pose_rotation"] for the screenshot helper, so
#     portrait phone videos are handled automatically.
#   • Returns the merged DTO shape (secondary_peak_* + primary /
#     secondary labels) so the existing dual-row report renders
#     without any frontend display changes.
#
# Math (matches motionlens-web/lib/biomech/neck.ts:computeNeckAngle):
#
#     faceVecX = nose.x − ear_midpoint.x
#     faceVecY = nose.y − ear_midpoint.y
#     tiltDeg  = atan2(faceVecY, |faceVecX|) × 180 / π
#     signed   = tiltDeg − NEUTRAL_TILT_BASELINE (10°)
#
#     signed > 0  → flexion (chin to chest)
#     signed < 0  → extension (head back)
#
# Why ear→nose instead of the legacy shoulder→ear used by the single-
# frame compute_neck_angle helper above: shoulder→ear only captures
# the cervical-spine portion (~25° max), but full clinical neck
# flexion (45-80°) and extension (50-70°) include the atlanto-
# occipital joint + head pitch. Ear→nose tracks the face direction
# and captures the combined motion.


# Merged flex+ext reference ranges (browser NECK_MOVEMENTS verbatim
# — keep in sync so live + upload report identical reference
# bounds in the dual-row chart).
_MERGED_NECKFLEXEXT_PRIMARY_TARGET:   tuple[float, float] = (45.0, 80.0)  # Flexion
_MERGED_NECKFLEXEXT_SECONDARY_TARGET: tuple[float, float] = (50.0, 70.0)  # Extension

# Visibility floor for the smoothed time-series. Looser than the
# per-frame compute_neck_angle default because brief dips in vis
# don't kill a well-smoothed signal.
_NECK_VIS_THRESHOLD = 0.4

# 10° baseline subtraction — the typical neutral pose has the nose
# sitting slightly below the ear-axis line (ear-tragus to nose-tip
# vector points ~10° below horizontal in a relaxed head). Recenters
# the signed tilt on 0° at neutral.
_NECK_NEUTRAL_TILT_BASELINE_DEG = 10.0

# Deadband on the signed angle — below this magnitude the head is
# too close to neutral to confidently classify as flexion or
# extension. Same role as the corresponding constant in shoulder
# flex/ext.
_NECK_FLEXEXT_DEADBAND_DEG = 5.0

# Anatomical sanity ceilings — bumped slightly above the clinical
# normal-range upper bounds so hypermobile patients are not over-
# clamped, but any reading past these limits is dropped from peak
# tracking as a measurement artefact.
_NECK_FLEXION_ANATOMICAL_MAX_DEG   = 90.0
_NECK_EXTENSION_ANATOMICAL_MAX_DEG = 75.0

# Lateral-view pre-flight: median (|faceVecX| / |faceVec|) across
# visible frames must be at least this fraction. In LATERAL profile
# the nose sits clearly to one side of the ear midpoint and the
# ratio is large (~0.4-0.7); in near-FRONTAL view it collapses to
# ≈ 0 and the atan2-with-|faceVecX|-denominator math becomes
# unreliable (returns ≈ 90° regardless of head tilt). 0.25
# conservatively accepts true lateral + 3/4 profile.
_NECK_LATERAL_PROFILE_MIN_RATIO = 0.25


# ── Merged lateral flexion (left + right) ──────────────────────
# Patient stands FRONTAL to the camera and tilts the head sideways
# toward each shoulder in turn. Math (matches motionlens-web/lib/
# biomech/neck.ts:computeNeckAngle("lateral_flexion") verbatim):
#
#     neck_vec  = ear_midpoint − shoulder_midpoint
#     signed    = angle_between(vertical=(0,−1), neck_vec)
#
# Sign convention (image-space, browser-aligned):
#   • positive signed → ear_mid right of shoulder_mid in image
#                       → labelled "Left Lateral Flexion" (per
#                          the spec; matches the patient's
#                          anatomical left in selfie-mirrored
#                          view, swaps in non-mirrored uploads —
#                          same behaviour as live mode)
#   • negative signed → "Right Lateral Flexion"
_MERGED_NECKLATERAL_PRIMARY_TARGET:   tuple[float, float] = (35.0, 45.0)  # Left
_MERGED_NECKLATERAL_SECONDARY_TARGET: tuple[float, float] = (35.0, 45.0)  # Right

# Deadband on the signed angle — below this magnitude the head is
# too close to neutral to commit to a side. Same threshold the
# browser detectNeckLateralDirection uses, so live + upload agree
# on what counts as a recordable tilt.
_NECK_LATERAL_DEADBAND_DEG = 5.0

# Anatomical max per side. Lateral flexion of the cervical spine
# tops out around 45°; bumped slightly to allow for hypermobile
# patients before we drop the frame as an artefact.
_NECK_LATERAL_ANATOMICAL_MAX_DEG = 45.0

# Threshold for flagging clinically-meaningful left/right
# asymmetry. AAOS / clinical convention: > 10° side-to-side
# difference is worth a clinician's attention.
_NECK_LATERAL_ASYMMETRY_FLAG_DEG = 10.0

# Pre-flight frontal-view check for lateral_flexion (OPPOSITE
# direction from the flex/ext check above — lateral flexion
# needs the patient FACING the camera). Compare shoulder width
# to ear width; in frontal view shoulders are 2.5-4× the ear
# width, in lateral profile they project narrow and the ratio
# drops below ~1.5. 1.8 is conservative — accepts true frontal
# + 3/4 profile, rejects pure lateral.
_NECK_LATERAL_FRONTAL_MIN_SHOULDER_EAR_RATIO = 1.8


# ── Merged rotation (left + right) ─────────────────────────────
# Patient stands FRONTAL to camera, faces it squarely at the start
# of the recording (the calibration "neutral" pose), then turns
# the head to one side and the other in turn.
#
# Math (matches motionlens-web/lib/biomech/neck.ts:
# computeNeckRotationFromBaseline verbatim — ear-width foreshortening
# with per-patient anatomy correction):
#
#     baseline_ear_width = ear_width at calibration (head-sphere
#                          diameter at facing-forward)
#     current_ratio = current_ear_width / baseline_ear_width
#     ratio_clamped = max(0, min(1, current_ratio))
#     magnitude_deg = acos(ratio_clamped) × 180 / π
#
#     baseline_nose_ratio = (nose.x − ear_mid.x) / (baseline_ear_width / 2)
#                          at calibration (anatomical asymmetry baseline)
#     current_nose_ratio  = (nose.x − ear_mid.x) / (current_ear_width / 2)
#                          per-frame
#     direction_sign = sign(current_nose_ratio − baseline_nose_ratio)
#
#     signed = direction_sign × magnitude_deg
#
# Sign convention (image-space, browser-aligned):
#   • positive → nose moved RIGHT in image after baseline (patient
#                turned right in mirrored selfie view; patient
#                turned left in non-mirrored upload)
#   • negative → nose moved LEFT in image
#
# Labels follow spec convention (image-space):
#   primary   = "Left Rotation"   ← negative signed peaks
#   secondary = "Right Rotation"  ← positive signed peaks
#
# Why foreshortening beats the nose-displacement approximation:
# nose protrusion varies per patient (3-5 cm forward of ear axis,
# vs shoulder width ~ 30-45 cm), so nose-displacement / shoulder-
# width gives a ratio that saturates at ~0.15-0.25 even at full
# 90° rotation. arcsin of that returns just ~10-15°, severely
# under-reporting ROM. The ear-width foreshortening cancels nose
# anatomy and is geometrically faithful (ear-to-ear chord shrinks
# as cos(θ) when the head rotates around its vertical axis).
_MERGED_NECKROT_PRIMARY_TARGET:   tuple[float, float] = (60.0, 80.0)  # Left
_MERGED_NECKROT_SECONDARY_TARGET: tuple[float, float] = (60.0, 80.0)  # Right

# Maximum tolerated baseline ratio when auto-detecting "facing
# forward". 0.15 = nose sits within 15% of half-ear-width of the
# ear midline. Larger than that and we suspect the patient isn't
# actually facing the camera squarely. Matches browser constant.
_NECK_ROT_FACING_FORWARD_RATIO = 0.15

# Minimum ear-to-ear pixel width to accept calibration. Below this
# the patient is too far from camera and keypoint noise dominates.
_NECK_ROT_MIN_EAR_WIDTH_PX = 30.0

# Consecutive stable facing-forward frames required before the
# baseline locks. Matches the shoulder-rotation pattern; once
# locked, never relocked mid-trial (would destabilise magnitudes).
_NECK_ROT_CALIBRATION_STABLE_FRAMES = 5

# Deadband on the signed angle — below this magnitude the head is
# too close to neutral to commit to a side.
_NECK_ROT_DEADBAND_DEG = 5.0

# Anatomical clamp per spec; the foreshortening formula's arccos
# naturally saturates at 90° anyway, so this is a conservative
# clinical reporting cap rather than a math guard.
_NECK_ROT_ANATOMICAL_MAX_DEG = 80.0

# Threshold for flagging meaningful left/right asymmetry.
_NECK_ROT_ASYMMETRY_FLAG_DEG = 10.0


def _classify_in_range(value: float, lo: float, hi: float) -> str:
    """Asymmetric range classifier — mirror of the version in
    knee_engine / shoulder_engine. Below range = fair / poor (mild
    vs notable restriction); above range = good (normal variation)
    up to ~30%, then fair, then poor."""
    if lo <= value <= hi:
        return "good"
    width = max(1.0, hi - lo)
    if value < lo:
        dist_frac = (lo - value) / width
        return "fair" if dist_frac <= 0.30 else "poor"
    dist_frac = (value - hi) / width
    if dist_frac <= 0.30:
        return "good"
    if dist_frac <= 1.00:
        return "fair"
    return "poor"


def _frontal_view_for_neck(ts: dict) -> Optional[str]:
    """Pre-flight rejection of near-frontal uploads. Returns a
    user-facing error message when the median nose-to-ear-midpoint
    horizontal component ratio (|faceVecX| / |faceVec|) is below
    the threshold across visible frames — that case the atan2-
    with-|faceVecX|-denominator math collapses and the resulting
    tilt readings are noise. Conservative threshold so true
    lateral profile + 3/4 profile pass cleanly."""
    nose_entry = ts.get("nose")      or {}
    le_entry   = ts.get("left_ear")  or {}
    re_entry   = ts.get("right_ear") or {}
    nx  = nose_entry.get("x_px"); ny  = nose_entry.get("y_px"); nv  = nose_entry.get("vis")
    lex = le_entry.get("x_px");   ley = le_entry.get("y_px");   lev = le_entry.get("vis")
    rex = re_entry.get("x_px");   rey = re_entry.get("y_px");   rev = re_entry.get("vis")
    for arr in (nx, ny, nv, lex, ley, lev, rex, rey, rev):
        if arr is None:
            return None
    n = min(len(nx), len(lex), len(rex))
    if n == 0:
        return None
    ratios = []
    for i in range(n):
        if nv[i]  < _NECK_VIS_THRESHOLD: continue
        if lev[i] < _NECK_VIS_THRESHOLD: continue
        if rev[i] < _NECK_VIS_THRESHOLD: continue
        ear_mid_x = (float(lex[i]) + float(rex[i])) / 2.0
        ear_mid_y = (float(ley[i]) + float(rey[i])) / 2.0
        fx = float(nx[i]) - ear_mid_x
        fy = float(ny[i]) - ear_mid_y
        mag = math.hypot(fx, fy)
        if mag < 1e-4:
            continue
        ratios.append(abs(fx) / mag)
    if not ratios:
        return None
    ratios.sort()
    med_ratio = ratios[len(ratios) // 2]
    if med_ratio < _NECK_LATERAL_PROFILE_MIN_RATIO:
        return (
            "Camera angle appears to be a front view, but the neck "
            "flexion/extension test needs a SIDE (lateral) profile — "
            "place the camera to the patient's side so the face is "
            "seen from the ear. Please re-record."
        )
    return None


def _poor_nose_visibility(ts: dict) -> bool:
    """True when nose visibility is below 0.5 for more than half
    the frames. The ear→nose tilt math needs the nose landmark, so
    a run where the nose is hidden most of the time can't produce
    a meaningful peak."""
    # Local import — numpy is only needed by the video pipeline,
    # not the single-frame compute_neck_angle helper.
    import numpy as _np
    nose_entry = ts.get("nose") or {}
    nv = nose_entry.get("vis")
    if nv is None or len(nv) == 0:
        return False
    nv_arr = _np.asarray(nv, dtype=float)
    visible_frac = float(_np.mean(nv_arr >= 0.5))
    return visible_frac < 0.50


def _grab_neck_key_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    label: str,
) -> Optional[dict]:
    """Seek to `frame_index`, apply the same pose-based rotation
    extract_poses applied to the keypoints, draw a face-emphasised
    skeleton overlay, return a JPEG data URL. Mirrors the other
    engines' _grab_*_key_frame helpers."""
    if frame_index < 0:
        return None
    # Local imports — keep the live-frame surface above import-free.
    import cv2 as _cv2
    from engines.gait_engine import LM as _LM, apply_rotation as _apply_rot

    pose_rot = int(keypoints_normalized.get("_pose_rotation") or 0)

    cap = _cv2.VideoCapture(video_path)
    try:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_index)
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
        frame = _cv2.resize(frame, (target_w, int(h * scale)))
        h, w = frame.shape[:2]

    # Face / neck landmarks emphasised; the rest of the upper-body
    # skeleton stays in grey for context.
    EMPHASISED = {"nose", "left_ear", "right_ear"}

    def _draw_dot(name: str):
        frames = keypoints_normalized.get(name, [])
        if frame_index >= len(frames):
            return None
        kp = frames[frame_index]
        if kp is None:
            return None
        x_n, y_n, _vis = kp
        px = int(x_n * w)
        py = int(y_n * h)
        outer = (0, 0, 220) if name in EMPHASISED else (150, 150, 150)
        _cv2.circle(frame, (px, py), 5, outer, -1)
        _cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        return (px, py)

    edges = [
        ("left_shoulder",  "right_shoulder"),
        ("left_ear",       "right_ear"),
        ("left_ear",       "nose"),
        ("right_ear",      "nose"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
    ]
    dot_pos: dict[str, tuple[int, int]] = {}
    for name in _LM:
        p = _draw_dot(name)
        if p:
            dot_pos[name] = p
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            highlight = a in EMPHASISED and b in EMPHASISED
            line_colour = (255, 255, 255) if highlight else (180, 180, 180)
            _cv2.line(frame, dot_pos[a], dot_pos[b], line_colour, 2)

    ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "label": label,
        "frame_index": int(frame_index),
        "image_data_url": f"data:image/jpeg;base64,{b64}",
    }


def _lateral_profile_for_neck_lateral_flexion(ts: dict) -> Optional[str]:
    """Pre-flight FRONTAL-view requirement for the lateral_flexion
    test. Compare shoulder-width to ear-width: in frontal view
    shoulders project clearly wider than the head (ratio ≥ ~2),
    in lateral profile both project narrow and the ratio drops
    toward 1. Conservative threshold rejects only clearly-lateral
    uploads so true frontal + 3/4 profile pass."""
    ls = ts.get("left_shoulder")  or {}
    rs = ts.get("right_shoulder") or {}
    le = ts.get("left_ear")  or {}
    re = ts.get("right_ear") or {}
    ls_x = ls.get("x_px"); ls_v = ls.get("vis")
    rs_x = rs.get("x_px"); rs_v = rs.get("vis")
    le_x = le.get("x_px"); le_v = le.get("vis")
    re_x = re.get("x_px"); re_v = re.get("vis")
    for arr in (ls_x, rs_x, le_x, re_x, ls_v, rs_v, le_v, re_v):
        if arr is None:
            return None
    n = min(len(ls_x), len(rs_x), len(le_x), len(re_x))
    if n == 0:
        return None
    ratios = []
    for i in range(n):
        if ls_v[i] < _NECK_VIS_THRESHOLD: continue
        if rs_v[i] < _NECK_VIS_THRESHOLD: continue
        if le_v[i] < _NECK_VIS_THRESHOLD: continue
        if re_v[i] < _NECK_VIS_THRESHOLD: continue
        sw = abs(float(rs_x[i]) - float(ls_x[i]))
        ew = abs(float(re_x[i]) - float(le_x[i]))
        if ew < 1e-4:
            continue
        ratios.append(sw / ew)
    if not ratios:
        return None
    ratios.sort()
    med_ratio = ratios[len(ratios) // 2]
    if med_ratio < _NECK_LATERAL_FRONTAL_MIN_SHOULDER_EAR_RATIO:
        return (
            "Camera angle appears to be a side view, but the neck "
            "lateral flexion test needs a FRONT-FACING view of the "
            "patient. Please face the camera directly and re-record."
        )
    return None


def _analyze_neck_lateral_flexion(
    video_path: str,
    pose_options,
) -> dict:
    """Backend pipeline for the merged neck lateral_flexion test
    (left + right tilt captured in one trial). Math + sign
    convention mirror motionlens-web/lib/biomech/neck.ts:
    computeNeckAngle("lateral_flexion") + detectNeckLateralDirection
    verbatim so live + upload report the same metric / direction
    frame-by-frame.

    Math: signed angle between the vertical reference (0, −1) and
    the shoulder_mid → ear_mid vector, computed in image-coordinate
    2D space.

    Sign convention:
      positive → ear_mid right of shoulder_mid in image → labelled
                  "Left Lateral Flexion" (selfie-mirrored
                  convention; in non-mirrored uploads the labels
                  reflect image-space, same as live mode).
      negative → "Right Lateral Flexion".

    No direction detector — the sign is the direction. Deadband
    on the angle magnitude (5°) suppresses neutral-pose noise.
    """
    from engines.gait_engine import build_time_series, extract_poses

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # Pre-flight: FRONTAL view required (the test relies on the
    # frontal-plane tilt of the shoulder→ear axis; in pure
    # lateral profile both shoulders project to similar x and
    # the math collapses).
    lateral_msg = _lateral_profile_for_neck_lateral_flexion(ts)
    if lateral_msg:
        raise ValueError(lateral_msg)

    le_entry = ts["left_ear"]
    re_entry = ts["right_ear"]
    ls_entry = ts["left_shoulder"]
    rs_entry = ts["right_shoulder"]

    lex = le_entry["x_px"]; ley = le_entry["y_px"]; lev = le_entry["vis"]
    rex = re_entry["x_px"]; rey = re_entry["y_px"]; rev = re_entry["vis"]
    lsx = ls_entry["x_px"]; lsy = ls_entry["y_px"]; lsv = ls_entry["vis"]
    rsx = rs_entry["x_px"]; rsy = rs_entry["y_px"]; rsv = rs_entry["vis"]

    n = int(min(len(lex), len(rex), len(lsx), len(rsx)))

    # Per-frame signed lateral angle. None for frames with any
    # required landmark below the visibility threshold — the
    # min/max tracker simply skips those frames so the running
    # peak holds its last good value (avoids the 0° fall-back
    # at extreme tilt where one ear can briefly occlude).
    signed_angles: list[Optional[float]] = []
    valid_frames = 0
    for i in range(n):
        if (lev[i] < _NECK_VIS_THRESHOLD
                or rev[i] < _NECK_VIS_THRESHOLD
                or lsv[i] < _NECK_VIS_THRESHOLD
                or rsv[i] < _NECK_VIS_THRESHOLD):
            signed_angles.append(None)
            continue
        ear_mid_x = (float(lex[i]) + float(rex[i])) / 2.0
        ear_mid_y = (float(ley[i]) + float(rey[i])) / 2.0
        sh_mid_x = (float(lsx[i]) + float(rsx[i])) / 2.0
        sh_mid_y = (float(lsy[i]) + float(rsy[i])) / 2.0
        nvx = ear_mid_x - sh_mid_x
        nvy = ear_mid_y - sh_mid_y
        if math.hypot(nvx, nvy) < 1e-4:
            signed_angles.append(None)
            continue
        # Signed angle from vertical (0, −1) to neck_vec in image
        # coords (y grows downward). atan2(cross, dot) where:
        #   cross = vert.x*vec.y − vert.y*vec.x = vec.x
        #   dot   = vert.x*vec.x + vert.y*vec.y = −vec.y
        signed = math.degrees(math.atan2(nvx, -nvy))
        # Anatomical sanity: drop frames where the signed tilt
        # exceeds physical limits (artefact — typically an ear
        # occluded by hand / clothing during the tilt).
        if abs(signed) > _NECK_LATERAL_ANATOMICAL_MAX_DEG + 15.0:
            signed_angles.append(None)
            continue
        signed_angles.append(signed)
        valid_frames += 1

    if valid_frames < max(3, int(fps * 0.5)):
        raise ValueError("poor_visibility")

    # ── Dual-peak tracking ───────────────────────────────────
    # No direction detector — the sign is the direction. Inside
    # the deadband (|signed| < 5°) the frame is neutral and
    # updates neither peak.
    primary_peak_signed:   Optional[float] = None  # left  (positive)
    primary_peak_idx   = -1
    secondary_peak_signed: Optional[float] = None  # right (negative)
    secondary_peak_idx = -1
    for i, a in enumerate(signed_angles):
        if a is None:
            continue
        if a > _NECK_LATERAL_DEADBAND_DEG:
            if primary_peak_signed is None or a > primary_peak_signed:
                primary_peak_signed = a
                primary_peak_idx = i
        elif a < -_NECK_LATERAL_DEADBAND_DEG:
            if secondary_peak_signed is None or a < secondary_peak_signed:
                secondary_peak_signed = a
                secondary_peak_idx = i
        # Else in deadband → neither peak updated.

    # Clamp the REPORTED magnitudes to anatomical limits (45°
    # each side). The raw signed peak is still preserved on the
    # `peak_angle` field for downstream debugging.
    primary_mag = (
        min(_NECK_LATERAL_ANATOMICAL_MAX_DEG, max(0.0, float(primary_peak_signed)))
        if primary_peak_signed is not None else 0.0
    )
    secondary_mag = (
        min(_NECK_LATERAL_ANATOMICAL_MAX_DEG, max(0.0, float(-secondary_peak_signed)))
        if secondary_peak_signed is not None else 0.0
    )

    # ── Asymmetry detection ──────────────────────────────────
    # Only meaningful when BOTH sides were captured (else the
    # patient simply didn't perform that direction in the run
    # and asymmetry would be a spurious flag).
    both_captured = primary_mag > 0 and secondary_mag > 0
    asymmetry_deg = (
        abs(primary_mag - secondary_mag) if both_captured else 0.0
    )
    asymmetry_flag = both_captured and asymmetry_deg > _NECK_LATERAL_ASYMMETRY_FLAG_DEG

    # ── Build response ───────────────────────────────────────
    p_lo, p_hi = _MERGED_NECKLATERAL_PRIMARY_TARGET
    s_lo, s_hi = _MERGED_NECKLATERAL_SECONDARY_TARGET
    p_target = p_hi
    p_pct = (primary_mag / p_target) * 100.0 if p_target > 0 else 0.0
    p_status = _classify_in_range(primary_mag, p_lo, p_hi)

    interpretation_primary = (
        f"Left lateral flexion measured {primary_mag:.1f}°, which is "
        f"{p_pct:.0f}% of the {p_lo:.0f}°–{p_hi:.0f}° normal range "
        f"— {p_status}."
    )
    if secondary_mag > 0:
        s_status = _classify_in_range(secondary_mag, s_lo, s_hi)
        interpretation_secondary = (
            f"Right lateral flexion measured {secondary_mag:.1f}°, "
            f"which is {(secondary_mag / s_hi) * 100.0:.0f}% of the "
            f"{s_lo:.0f}°–{s_hi:.0f}° normal range — {s_status}."
        )
    else:
        interpretation_secondary = (
            "Right lateral flexion direction was not detected in this "
            "recording."
        )
    interpretation = f"{interpretation_primary} {interpretation_secondary}"
    if asymmetry_flag:
        interpretation += (
            f" Notable left-right asymmetry detected "
            f"({asymmetry_deg:.1f}°). Clinical correlation recommended."
        )

    # Two key frames only — peak left + peak right. Matches the
    # other merged tests' layout (no neutral frame).
    key_frames: list[dict] = []
    if primary_peak_idx >= 0 and primary_mag > 0:
        kf = _grab_neck_key_frame(
            video_path, primary_peak_idx, raw,
            f"Left lateral flexion ({primary_mag:.1f}°)",
        )
        if kf:
            key_frames.append(kf)
    if secondary_peak_idx >= 0 and secondary_mag > 0:
        kf = _grab_neck_key_frame(
            video_path, secondary_peak_idx, raw,
            f"Right lateral flexion ({secondary_mag:.1f}°)",
        )
        if kf:
            key_frames.append(kf)

    return {
        "body_part": "neck",
        "movement": "lateral_flexion",
        "side": None,
        "peak_angle": (
            float(primary_peak_signed) if primary_peak_signed is not None else None
        ),
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
        "primary_label": "Left Lateral Flexion",
        "secondary_label": "Right Lateral Flexion",
    }


def _is_neck_rotation_neutral(
    nose_x: float, le_x: float, re_x: float,
) -> tuple[bool, float]:
    """Returns (is_facing_forward, ear_width_px). Ports
    isStableFacingForward from motionlens-web/lib/biomech/neck.ts.

    The patient is "facing forward" when the nose sits roughly
    between the two ears in image-x — specifically within
    _NECK_ROT_FACING_FORWARD_RATIO of half the ear-width of the
    ear midline. Also rejects the frame when the ears are too
    close in pixel space (patient too far from camera)."""
    ear_width = abs(le_x - re_x)
    if ear_width < _NECK_ROT_MIN_EAR_WIDTH_PX:
        return False, ear_width
    ear_mid_x = (le_x + re_x) / 2.0
    ratio = abs((nose_x - ear_mid_x) / (ear_width / 2.0))
    return ratio <= _NECK_ROT_FACING_FORWARD_RATIO, ear_width


def _analyze_neck_rotation(
    video_path: str,
    pose_options,
) -> dict:
    """Backend pipeline for the merged neck rotation test (left +
    right captured in one trial). Math + sign convention mirror
    motionlens-web/lib/biomech/neck.ts:computeNeckRotationFromBaseline
    verbatim so live + upload report the same metric.

    Two-phase flow (matches shoulder rotation):

      Phase 1 — lock a calibration baseline from the first
        _NECK_ROT_CALIBRATION_STABLE_FRAMES consecutive frames
        where the patient is facing the camera squarely. Baseline
        is locked ONCE and never relocked (a mid-trial relock
        would shift the magnitude scale).

      Phase 2 — per-frame ear-width foreshortening + nose-direction
        sign. Inside the deadband (|signed| < 5°) frames neither
        peak slot is updated. Frames with bad nose/ear visibility
        are skipped; running peaks hold their last good value
        (no 0° contamination at the extreme of rotation where
        one ear can briefly occlude).
    """
    from engines.gait_engine import build_time_series, extract_poses

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # Pre-flight: FRONTAL view required (same physical setup as
    # lateral_flexion). Re-use that helper — both tests share the
    # shoulder-width / ear-width frontal-view geometric check.
    lateral_msg = _lateral_profile_for_neck_lateral_flexion(ts)
    if lateral_msg:
        raise ValueError(lateral_msg)

    nose_entry = ts["nose"]
    le_entry   = ts["left_ear"]
    re_entry   = ts["right_ear"]
    ls_entry   = ts["left_shoulder"]
    rs_entry   = ts["right_shoulder"]

    nx  = nose_entry["x_px"];  nv  = nose_entry["vis"]
    lex = le_entry["x_px"];    lev = le_entry["vis"]
    rex = re_entry["x_px"];    rev = re_entry["vis"]
    lsv = ls_entry["vis"]
    rsv = rs_entry["vis"]

    n = int(min(len(nx), len(lex), len(rex), len(lsv), len(rsv)))

    # ── Phase 1: baseline lock ───────────────────────────────
    # Look for the first run of _NECK_ROT_CALIBRATION_STABLE_FRAMES
    # consecutive frames where the patient is facing forward
    # squarely AND all five anchor keypoints are confidently
    # visible. Once found, snapshot ear-width + nose ratio.
    baseline_ear_width: Optional[float] = None
    baseline_nose_ratio: Optional[float] = None
    baseline_locked_idx = -1
    stable_count = 0
    for i in range(n):
        if (nv[i]  < _NECK_VIS_THRESHOLD
                or lev[i] < _NECK_VIS_THRESHOLD
                or rev[i] < _NECK_VIS_THRESHOLD
                or lsv[i] < _NECK_VIS_THRESHOLD
                or rsv[i] < _NECK_VIS_THRESHOLD):
            stable_count = 0
            continue
        n_x  = float(nx[i])
        le_x = float(lex[i])
        re_x = float(rex[i])
        ok, ear_width = _is_neck_rotation_neutral(n_x, le_x, re_x)
        if not ok:
            stable_count = 0
            continue
        stable_count += 1
        if stable_count >= _NECK_ROT_CALIBRATION_STABLE_FRAMES:
            baseline_ear_width = ear_width
            ear_mid_x = (le_x + re_x) / 2.0
            baseline_nose_ratio = (n_x - ear_mid_x) / (ear_width / 2.0)
            baseline_locked_idx = i
            break

    if baseline_ear_width is None or baseline_nose_ratio is None:
        # Surfaced as HTTP 400 by api.analyze_neck.
        raise ValueError(
            "Neutral pose not detected. Please start the recording with "
            "the patient facing the camera squarely (head straight, "
            "both ears equally visible) for at least 1 second before "
            "rotating."
        )

    # ── Phase 2: per-frame magnitude + direction ─────────────
    primary_peak_signed:   Optional[float] = None  # left rotation (negative)
    primary_peak_idx   = -1
    secondary_peak_signed: Optional[float] = None  # right rotation (positive)
    secondary_peak_idx = -1
    valid_frames = 0
    for i in range(n):
        # Nose + both ears required for the foreshortening formula.
        # Shoulder visibility checked to ensure the pose pipeline
        # is tracking a coherent torso (drops in shoulder vis
        # usually mean the patient walked out of frame).
        if (nv[i]  < _NECK_VIS_THRESHOLD
                or lev[i] < _NECK_VIS_THRESHOLD
                or rev[i] < _NECK_VIS_THRESHOLD
                or lsv[i] < _NECK_VIS_THRESHOLD
                or rsv[i] < _NECK_VIS_THRESHOLD):
            continue
        n_x  = float(nx[i])
        le_x = float(lex[i])
        re_x = float(rex[i])
        current_ear_width = abs(le_x - re_x)
        # Clamp ratio to [0, 1]: it can technically exceed 1 if the
        # patient stepped closer to the camera since calibration
        # (apparent ear-width grows); clamping keeps acos defined
        # and under-reports rather than crashing.
        ratio = max(0.0, min(1.0, current_ear_width / baseline_ear_width))
        magnitude_deg = math.degrees(math.acos(ratio))

        # Sign from nose direction relative to ear midline, with
        # baseline ratio subtracted so anatomical asymmetry (off-
        # centre nose at calibration) cancels out.
        if current_ear_width > 1e-3:
            ear_mid_x = (le_x + re_x) / 2.0
            current_nose_ratio = (n_x - ear_mid_x) / (current_ear_width / 2.0)
        else:
            current_nose_ratio = 0.0
        delta = current_nose_ratio - baseline_nose_ratio
        sign = -1.0 if delta < 0 else 1.0
        signed = sign * magnitude_deg

        # Inside deadband → neither peak updated (neutral pose
        # noise can flip the sign at small magnitudes).
        if abs(signed) < _NECK_ROT_DEADBAND_DEG:
            valid_frames += 1
            continue
        # Anatomical clamp on REPORTED magnitudes (the math itself
        # naturally caps at ~90° but the spec asks for 80° as a
        # clinical reporting ceiling).
        clamped_mag = min(_NECK_ROT_ANATOMICAL_MAX_DEG, magnitude_deg)
        clamped_signed = sign * clamped_mag

        if clamped_signed < 0:
            if primary_peak_signed is None or clamped_signed < primary_peak_signed:
                primary_peak_signed = clamped_signed
                primary_peak_idx = i
        else:
            if secondary_peak_signed is None or clamped_signed > secondary_peak_signed:
                secondary_peak_signed = clamped_signed
                secondary_peak_idx = i
        valid_frames += 1

    if valid_frames < max(3, int(fps * 0.5)):
        raise ValueError("poor_visibility")

    primary_mag = (
        float(-primary_peak_signed) if primary_peak_signed is not None else 0.0
    )
    secondary_mag = (
        float(secondary_peak_signed) if secondary_peak_signed is not None else 0.0
    )

    # ── Asymmetry detection ──────────────────────────────────
    both_captured = primary_mag > 0 and secondary_mag > 0
    asymmetry_deg = (
        abs(primary_mag - secondary_mag) if both_captured else 0.0
    )
    asymmetry_flag = both_captured and asymmetry_deg > _NECK_ROT_ASYMMETRY_FLAG_DEG

    # ── Build response ───────────────────────────────────────
    p_lo, p_hi = _MERGED_NECKROT_PRIMARY_TARGET
    s_lo, s_hi = _MERGED_NECKROT_SECONDARY_TARGET
    p_target = p_hi
    p_pct = (primary_mag / p_target) * 100.0 if p_target > 0 else 0.0
    p_status = _classify_in_range(primary_mag, p_lo, p_hi)

    interpretation_primary = (
        f"Left rotation measured {primary_mag:.1f}°, which is "
        f"{p_pct:.0f}% of the {p_lo:.0f}°–{p_hi:.0f}° normal range "
        f"— {p_status}."
    )
    if secondary_mag > 0:
        s_status = _classify_in_range(secondary_mag, s_lo, s_hi)
        interpretation_secondary = (
            f"Right rotation measured {secondary_mag:.1f}°, which is "
            f"{(secondary_mag / s_hi) * 100.0:.0f}% of the "
            f"{s_lo:.0f}°–{s_hi:.0f}° normal range — {s_status}."
        )
    else:
        interpretation_secondary = (
            "Right rotation direction was not detected in this recording."
        )
    interpretation = f"{interpretation_primary} {interpretation_secondary}"
    if asymmetry_flag:
        interpretation += (
            f" Notable left-right asymmetry detected "
            f"({asymmetry_deg:.1f}°). Clinical correlation recommended."
        )
    # 2D approximation caveat — surfaced in the interpretation so
    # the clinician sees it on the report. Rotation around the
    # vertical axis is fundamentally a 3D motion projected to 2D;
    # the ear-width foreshortening cancels nose anatomy but is
    # still sensitive to head-tilt / chin-up combined motions.
    interpretation += (
        " Note: rotation is measured from a 2D pose estimate; "
        "results may be less precise than for in-plane neck movements."
    )

    # Two key frames only — peak left + peak right.
    key_frames: list[dict] = []
    if primary_peak_idx >= 0 and primary_mag > 0:
        kf = _grab_neck_key_frame(
            video_path, primary_peak_idx, raw,
            f"Left rotation ({primary_mag:.1f}°)",
        )
        if kf:
            key_frames.append(kf)
    if secondary_peak_idx >= 0 and secondary_mag > 0:
        kf = _grab_neck_key_frame(
            video_path, secondary_peak_idx, raw,
            f"Right rotation ({secondary_mag:.1f}°)",
        )
        if kf:
            key_frames.append(kf)

    return {
        "body_part": "neck",
        "movement": "rotation",
        "side": None,
        "peak_angle": (
            float(primary_peak_signed) if primary_peak_signed is not None else None
        ),
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
        "primary_label": "Left Rotation",
        "secondary_label": "Right Rotation",
    }


def analyze_neck(
    video_path: str,
    pose_options,
    movement: str,
) -> dict:
    """Run the BlazePose-Full backend pipeline on an uploaded
    neck-ROM clip.

    Args:
        video_path:   path to the uploaded (and optionally repaired)
                      video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        movement:     "flexion_extension" / "lateral_flexion" /
                      "rotation" — the three merged neck tests
                      now routed to backend.

    Returns:
        Dict matching the merged BiomechData Pydantic schema —
        secondary_peak_* + primary/secondary labels populated so
        the existing dual-row neck report renders without any
        frontend display changes.

    Raises:
        ValueError: input invalid, frontal/lateral-view rejection
                    (depending on movement), insufficient nose
                    visibility, or fewer than ~half a second of
                    usable frames. The endpoint maps these to
                    HTTP 400 with the original user-facing
                    message preserved.
    """
    # Lateral flexion + rotation each run through their own
    # pipelines (different math + view requirement + dual-side
    # labels). Branch early so the existing flex/ext code below
    # stays untouched.
    if movement == "lateral_flexion":
        return _analyze_neck_lateral_flexion(video_path, pose_options)
    if movement == "rotation":
        return _analyze_neck_rotation(video_path, pose_options)

    # Local imports keep the single-frame surface dependency-light.
    from engines.gait_engine import build_time_series, extract_poses

    if movement != "flexion_extension":
        raise ValueError(f"Unsupported neck movement: {movement!r}")

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # Pre-flight: reject pure-frontal recordings fast (the math
    # collapses there). Run before the heavy per-frame loop.
    frontal_msg = _frontal_view_for_neck(ts)
    if frontal_msg:
        raise ValueError(frontal_msg)

    # Pre-flight: nose must be visible in at least half the frames.
    # Without it, the ear→nose tilt is undefined.
    if _poor_nose_visibility(ts):
        raise ValueError("poor_visibility")

    nose_entry = ts["nose"]
    le_entry   = ts["left_ear"]
    re_entry   = ts["right_ear"]
    ls_entry   = ts["left_shoulder"]
    rs_entry   = ts["right_shoulder"]

    nx  = nose_entry["x_px"];  ny  = nose_entry["y_px"];  nv  = nose_entry["vis"]
    lex = le_entry["x_px"];    ley = le_entry["y_px"];    lev = le_entry["vis"]
    rex = re_entry["x_px"];    rey = re_entry["y_px"];    rev = re_entry["vis"]
    lsv = ls_entry["vis"]
    rsv = rs_entry["vis"]

    n = int(min(len(nx), len(lex), len(rex), len(lsv), len(rsv)))

    # Per-frame signed neck tilt. None for frames where any of
    # the five required landmarks dips below threshold — the
    # min/max tracker just skips those frames, so the running
    # peak holds its last good value (no false 0° contamination
    # at the extremes where an ear can briefly be occluded).
    signed_angles: list[Optional[float]] = []
    valid_frames = 0
    for i in range(n):
        if (nv[i]  < _NECK_VIS_THRESHOLD
                or lev[i] < _NECK_VIS_THRESHOLD
                or rev[i] < _NECK_VIS_THRESHOLD
                or lsv[i] < _NECK_VIS_THRESHOLD
                or rsv[i] < _NECK_VIS_THRESHOLD):
            signed_angles.append(None)
            continue
        ear_mid_x = (float(lex[i]) + float(rex[i])) / 2.0
        ear_mid_y = (float(ley[i]) + float(rey[i])) / 2.0
        fx = float(nx[i]) - ear_mid_x
        fy = float(ny[i]) - ear_mid_y
        if math.hypot(fx, fy) < 1e-4:
            signed_angles.append(None)
            continue
        tilt_deg = math.degrees(math.atan2(fy, abs(fx)))
        signed = tilt_deg - _NECK_NEUTRAL_TILT_BASELINE_DEG
        # Anatomical sanity: drop frames where the signed tilt
        # exceeds the physical limits in either direction —
        # almost certainly a measurement artefact (chin occluding
        # an ear, etc.).
        if signed > _NECK_FLEXION_ANATOMICAL_MAX_DEG + 15.0:
            signed_angles.append(None)
            continue
        if signed < -(_NECK_EXTENSION_ANATOMICAL_MAX_DEG + 15.0):
            signed_angles.append(None)
            continue
        signed_angles.append(signed)
        valid_frames += 1

    if valid_frames < max(3, int(fps * 0.5)):
        raise ValueError("poor_visibility")

    # ── Dual-peak tracking ───────────────────────────────────
    # No direction detection needed — the sign of the signed
    # tilt is the direction. Inside the deadband the frame is
    # neutral and updates neither peak slot.
    primary_peak_signed:   Optional[float] = None  # flexion (positive)
    primary_peak_idx   = -1
    secondary_peak_signed: Optional[float] = None  # extension (negative)
    secondary_peak_idx = -1
    for i, a in enumerate(signed_angles):
        if a is None:
            continue
        if a > _NECK_FLEXEXT_DEADBAND_DEG:
            if primary_peak_signed is None or a > primary_peak_signed:
                primary_peak_signed = a
                primary_peak_idx = i
        elif a < -_NECK_FLEXEXT_DEADBAND_DEG:
            if secondary_peak_signed is None or a < secondary_peak_signed:
                secondary_peak_signed = a
                secondary_peak_idx = i
        # Else: in deadband → neutral, neither peak updated.

    # Clamp the REPORTED magnitudes to the anatomical max so a
    # single artefact frame past the sanity ceiling can't lock an
    # implausible peak. We still keep the raw signed peak for the
    # `peak_angle` field (debug / re-analysis).
    primary_mag = (
        min(_NECK_FLEXION_ANATOMICAL_MAX_DEG, max(0.0, float(primary_peak_signed)))
        if primary_peak_signed is not None else 0.0
    )
    secondary_mag = (
        min(_NECK_EXTENSION_ANATOMICAL_MAX_DEG, max(0.0, float(-secondary_peak_signed)))
        if secondary_peak_signed is not None else 0.0
    )

    # ── Build response ───────────────────────────────────────
    p_lo, p_hi = _MERGED_NECKFLEXEXT_PRIMARY_TARGET
    s_lo, s_hi = _MERGED_NECKFLEXEXT_SECONDARY_TARGET
    p_target = p_hi
    p_pct = (primary_mag / p_target) * 100.0 if p_target > 0 else 0.0
    p_status = _classify_in_range(primary_mag, p_lo, p_hi)

    interpretation_primary = (
        f"Flexion measured {primary_mag:.1f}°, which is "
        f"{p_pct:.0f}% of the {p_lo:.0f}°–{p_hi:.0f}° normal range "
        f"— {p_status}."
    )
    if secondary_mag > 0:
        s_status = _classify_in_range(secondary_mag, s_lo, s_hi)
        interpretation_secondary = (
            f"Extension measured {secondary_mag:.1f}°, which is "
            f"{(secondary_mag / s_hi) * 100.0:.0f}% of the "
            f"{s_lo:.0f}°–{s_hi:.0f}° normal range — {s_status}."
        )
    else:
        interpretation_secondary = (
            "Extension direction was not detected in this recording."
        )
    interpretation = f"{interpretation_primary} {interpretation_secondary}"

    # Two key frames only (Flexion + Extension peaks). Matches the
    # merged knee + shoulder ab/ad layout — no neutral frame.
    key_frames: list[dict] = []
    if primary_peak_idx >= 0 and primary_mag > 0:
        kf = _grab_neck_key_frame(
            video_path, primary_peak_idx, raw,
            f"Flexion ({primary_mag:.1f}°)",
        )
        if kf:
            key_frames.append(kf)
    if secondary_peak_idx >= 0 and secondary_mag > 0:
        kf = _grab_neck_key_frame(
            video_path, secondary_peak_idx, raw,
            f"Extension ({secondary_mag:.1f}°)",
        )
        if kf:
            key_frames.append(kf)

    return {
        "body_part": "neck",
        "movement": "flexion_extension",
        # Neck flex/ext doesn't carry a side; field kept on the DTO
        # for sibling-endpoint parity (the report omits the side
        # column when this is None).
        "side": None,
        "peak_angle": (
            float(primary_peak_signed) if primary_peak_signed is not None else None
        ),
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
