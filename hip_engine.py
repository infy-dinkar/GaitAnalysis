"""hip_engine.py — clinical hip flexion ROM analysis on backend
MediaPipe (BlazePose Full, 33 keypoints).

Mirrors knee_engine.py / ankle_engine.py / the merged shoulder
branches architecturally:

  • Reuses the shared gait pose pipeline (extract_poses +
    build_time_series), inherits the _pose_rotation portrait-
    video correction set by extract_poses, ships its own
    side-aware key-frame screenshot helper.
  • Pre-flight wrong-side detection (left vs right hip
    visibility) fails fast with a clear actionable message
    rather than producing garbage values.

Math (matches motionlens-web/lib/biomech/hip.ts:computeHipAngle
for movement="flexion" verbatim, so live + upload report the
same metric):

    trunk = shoulder − hip   (vector pointing UP from hip)
    thigh = knee − hip       (vector pointing DOWN at neutral)
    interior = unsigned angle between trunk and thigh
             = atan2(|cross|, dot) × 180/π
    flexion  = 180° − interior

  - 0°    = leg fully extended at side (neutral standing)
  - ~90°  = thigh horizontal (leg lifted forward)
  - ~120° = clinical max for healthy adults (knee well above hip)

Lateral camera view is the standard clinical setup (patient
filmed from the side so the leg arc is visible in-plane), but
the math also works for frontal recordings — the per-frame
unsigned interior angle doesn't depend on facing direction. So
unlike neck flex/ext or shoulder rotation, no view rejection is
needed beyond the wrong-side visibility check.
"""
from __future__ import annotations

import base64
import math
from typing import Optional

import cv2
import numpy as np

from gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)


# Clinical normal ranges. Spec values used here; browser
# HIP_MOVEMENTS metadata has slightly different bounds (flexion
# [110,130], extension [10,30]), so the range column may differ
# on the report column until clinical guidance picks a single
# source. Using spec values consistently across both movements.
HIP_NORMAL_RANGES = {
    "flexion":   {"range": (100, 120), "target": 120.0},
    "extension": {"range": (20,  30),  "target": 30.0},
}

# Merged internal + external rotation. Seated heel-fixed test:
# patient sits upright on a chair with both feet flat on the
# ground, then rotates ONE leg at the hip while keeping the heel
# planted as the pivot. The toes (foot_index) swing laterally
# around the heel — outward = external rotation, inward =
# internal rotation. One recording captures both peaks. Math
# mirrors lib/biomech/hip-live.ts:computeHipAngle so live +
# upload report the same metric.
_MERGED_HIPROT_PRIMARY_TARGET:   tuple[float, float] = (30.0, 45.0)  # Internal
_MERGED_HIPROT_SECONDARY_TARGET: tuple[float, float] = (30.0, 45.0)  # External

# Anatomical reporting cap per side. Hip rotation ROM rarely
# exceeds 45° in healthy adults. Matches the live-mode
# HIP_ROTATION_MAX_DEG constant.
_HIP_ROT_MAX_DEG = 45.0

# Deadband on the signed foot-vector angle (degrees) before a
# direction is committed. Below this the foot is too close to
# pointing-at-camera neutral for reliable internal-vs-external
# discrimination — frames count toward valid_frames but don't
# update either peak slot. Mirrors the live constant
# HIP_ROT_DIRECTION_DEADBAND_DEG.
_HIP_ROT_DIRECTION_DEADBAND_DEG = 3.0

# Asymmetry-flag threshold for the internal-vs-external side-to-
# side difference. Same clinical convention as neck rotation.
_HIP_ROT_ASYMMETRY_FLAG_DEG = 10.0


# Visibility floor — looser than the per-frame default because
# the smoothed time-series tolerates brief dips better than
# raw frames.
_HIP_VIS_THRESHOLD = 0.4

# Anatomical sanity bounds on the hip-angle magnitudes.
# The flexion + extension formulas BOTH return |180° − interior|
# (unsigned deviation from standing straight), so a single
# frame's reading doesn't itself tell us which direction the
# patient was moving — the test movement selection does. Use
# different ceilings per movement to clamp reported peaks at
# the appropriate clinical max.
#
#   Flexion 140°  = knee well past chest; beyond this is a
#                   keypoint artefact (knee left frame, etc.).
#   Extension 30° = anatomical max for healthy adults; spec
#                   guidance. Hip extension is a small range
#                   (only 20-30° normal), so tight clamping
#                   suppresses spurious large readings if the
#                   test was performed incorrectly (e.g. patient
#                   accidentally did flexion).
_HIP_FLEXION_ANATOMICAL_MAX   = 140.0
_HIP_EXTENSION_ANATOMICAL_MAX = 30.0
_HIP_HYPEREXTENSION_LIMIT     = -5.0


# ─── Math primitive ─────────────────────────────────────────────
def _hip_interior_angle(
    shoulder: tuple[float, float],
    hip:      tuple[float, float],
    knee:     tuple[float, float],
) -> Optional[float]:
    """Unsigned interior angle (in degrees) at the hip vertex
    between the trunk vector (hip → shoulder) and the thigh
    vector (hip → knee). 180° = standing straight, smaller =
    more flexed. Returns None when either vector degenerates."""
    trunk_x = shoulder[0] - hip[0]
    trunk_y = shoulder[1] - hip[1]
    thigh_x = knee[0] - hip[0]
    thigh_y = knee[1] - hip[1]
    nt = math.hypot(trunk_x, trunk_y)
    nh = math.hypot(thigh_x, thigh_y)
    if nt < 1e-6 or nh < 1e-6:
        return None
    cross = abs(trunk_x * thigh_y - trunk_y * thigh_x)
    dot   = trunk_x * thigh_x + trunk_y * thigh_y
    return math.degrees(math.atan2(cross, dot))


# ─── Asymmetric ROM classifier ──────────────────────────────────
def _classify_in_range(value: float, lo: float, hi: float) -> str:
    """Same asymmetric range classifier as the other engines.
    Below range = fair / poor (mild / notable restriction);
    above range = good up to ~30%, then fair, then poor."""
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


# ─── Pre-flight: wrong-side detection ───────────────────────────
def _wrong_side_for_hip_video(ts: dict, side: str) -> Optional[str]:
    """Pre-flight check: if the requested side's hip has
    consistently lower visibility than the other side's, the
    user almost certainly picked the wrong side. Returns a
    user-facing error string, or None when selection is
    plausibly correct. Conservative thresholds so borderline
    lateral views (one side modestly occluded by the trunk in
    a strict-lateral recording) don't false-positive."""
    l_entry = ts.get("left_hip")  or {}
    r_entry = ts.get("right_hip") or {}
    l_v = l_entry.get("vis")
    r_v = r_entry.get("vis")
    if l_v is None or r_v is None:
        return None
    n = min(len(l_v), len(r_v))
    if n == 0:
        return None
    l_mean = float(np.mean(l_v[:n]))
    r_mean = float(np.mean(r_v[:n]))
    requested_mean = l_mean if side == "left" else r_mean
    other_mean = r_mean if side == "left" else l_mean
    other_side = "right" if side == "left" else "left"
    # Hip keypoints are extrapolated by MediaPipe even when one
    # side is occluded, so the gap has to be unambiguous before
    # we reject. (Same conservative pattern as knee + shoulder.)
    if (
        requested_mean < 0.4
        and other_mean > 0.6
        and (other_mean - requested_mean) > 0.25
    ):
        return (
            f"Requested side '{side}' has lower visibility than the other "
            f"side. Please re-record showing the correct leg clearly, or "
            f"switch to the '{other_side}' side."
        )
    return None


# ─── Rotation: signed foot-vector angle + direction helper ──────
def _hip_rotation_signed_foot_angle(
    heel_x: float, heel_y: float,
    foot_x: float, foot_y: float,
) -> float:
    """Signed angle (degrees) of the (heel → foot_index) vector
    from straight-down in the image. Mirrors the live JS helper
    `signedAngleBetween(0, 1, footX, footY)` so live + upload
    report the same metric.

    Mapping (un-mirrored MediaPipe frame, y-axis pointing down):
      • foot pointing forward at the camera → angle ≈ 0°
      • toes swung to image-right (footX > 0) → angle < 0
      • toes swung to image-left  (footX < 0) → angle > 0
    """
    fx = foot_x - heel_x
    fy = foot_y - heel_y
    return math.degrees(math.atan2(-fx, fy))


def _hip_rotation_outward_sign(side: str) -> float:
    """Per-side outward sign used to map raw foot-angle into the
    "internal vs external" labelling. Matches the live constant
    in lib/biomech/hip-live.ts:detectHipRotationDirection.

    In an un-mirrored MediaPipe frame the patient's LEFT side
    appears on image-right — so "outward" for the LEFT foot
    (toes swinging away from body midline = external rotation)
    lives in +x territory. The RIGHT foot's "outward" lives in
    −x. Multiplying the signed foot-angle by `outward_sign`
    normalises both sides to the same rule.

    The caller uses `signed_angle * outward_sign` to route into
    the correct peak slot, matching the live convention.
    """
    return 1.0 if side == "left" else -1.0


# ─── Key-frame helper ───────────────────────────────────────────
# Mirrors knee_engine._grab_knee_key_frame: seek to the source-
# video frame, apply the same pose-based rotation extract_poses
# applied, draw a skeleton overlay emphasising the test-side
# leg + trunk, return a JPEG data URL.
def _grab_hip_key_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    label: str,
    side: str,
) -> Optional[dict]:
    if frame_index < 0:
        return None
    # extract_poses stores the pose rotation it applied; re-apply
    # to the screenshot frame so the JPEG and the keypoint overlay
    # stay aligned.
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

    # Hip-relevant edges: trunk (shoulder ↔ hip on the test side),
    # plus both legs incl. foot lever (heel → foot_index) used by
    # the seated rotation test. Test side highlighted,
    # contralateral in grey for context.
    edges = [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
        ("left_hip",       "left_knee"),
        ("left_knee",      "left_ankle"),
        ("left_ankle",     "left_heel"),
        ("left_heel",      "left_foot_index"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
        ("right_ankle",    "right_heel"),
        ("right_heel",     "right_foot_index"),
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

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "label": label,
        "frame_index": int(frame_index),
        "image_data_url": f"data:image/jpeg;base64,{b64}",
    }


def _analyze_hip_rotation(
    video_path: str,
    pose_options,
    side: str,
) -> dict:
    """Backend pipeline for the merged hip rotation test —
    seated heel-fixed pivot variant (internal + external captured
    in one trial).

    Seated test setup: patient sits upright on a chair with both
    feet flat on the ground. Heel of the test leg stays planted
    as the pivot; the toes (foot_index) swing laterally as the
    hip rotates. One recording captures both peaks.

    Per-frame math (mirrors lib/biomech/hip-live.ts):

        foot_vec    = foot_index − heel        (image plane)
        signed_ang  = atan2(−foot_vec.x, foot_vec.y) × 180/π
        magnitude   = min(|signed_ang|, 45°)
        outward     = +1 (left) | −1 (right)
        direction   = signed_ang × outward     (live convention)

    The live mode maps `direction > 0 → external` and
    `direction < 0 → internal`. We mirror that here so live +
    upload route peaks into the same primary/secondary slots —
    primary = internal rotation, secondary = external rotation.

    Frames where |signed_ang| < _HIP_ROT_DIRECTION_DEADBAND_DEG
    count toward valid_frames but don't update either peak slot,
    suppressing neutral-pose jitter.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # Pre-flight wrong-side check (reused from flexion/extension).
    wrong_side_msg = _wrong_side_for_hip_video(ts, side)
    if wrong_side_msg:
        raise ValueError(wrong_side_msg)

    heel_key = f"{side}_heel"
    foot_key = f"{side}_foot_index"

    hex_ = ts[heel_key]["x_px"];  hey = ts[heel_key]["y_px"];  vhe = ts[heel_key]["vis"]
    fx_  = ts[foot_key]["x_px"];  fy_ = ts[foot_key]["y_px"];  vfi = ts[foot_key]["vis"]

    n = int(min(len(hex_), len(fx_)))

    outward_sign = _hip_rotation_outward_sign(side)

    # Live-convention sign of `signed_ang × outward`:
    #   > 0 → external rotation (toes swung outward)
    #   < 0 → internal rotation (toes swung inward)
    # Primary slot = internal, secondary slot = external —
    # so internal peaks are tracked on the NEGATIVE side and
    # external on the POSITIVE side.
    primary_peak_mag:   Optional[float] = None  # internal
    primary_peak_idx   = -1
    secondary_peak_mag: Optional[float] = None  # external
    secondary_peak_idx = -1
    valid_frames = 0
    direction_frames = 0  # frames that committed a direction

    for i in range(n):
        if vhe[i] < _HIP_VIS_THRESHOLD or vfi[i] < _HIP_VIS_THRESHOLD:
            continue
        valid_frames += 1

        signed_ang = _hip_rotation_signed_foot_angle(
            float(hex_[i]), float(hey[i]),
            float(fx_[i]),  float(fy_[i]),
        )
        # Deadband — below this the foot is too close to
        # pointing-at-camera neutral for reliable left/right
        # discrimination.
        if abs(signed_ang) < _HIP_ROT_DIRECTION_DEADBAND_DEG:
            continue

        magnitude = min(abs(signed_ang), _HIP_ROT_MAX_DEG)
        direction_signed = signed_ang * outward_sign
        direction_frames += 1

        if direction_signed > 0:
            # External rotation peak.
            if secondary_peak_mag is None or magnitude > secondary_peak_mag:
                secondary_peak_mag = magnitude
                secondary_peak_idx = i
        else:
            # Internal rotation peak.
            if primary_peak_mag is None or magnitude > primary_peak_mag:
                primary_peak_mag = magnitude
                primary_peak_idx = i

    if valid_frames < max(3, int(fps * 0.5)):
        raise ValueError("poor_visibility")

    if direction_frames == 0:
        # Heel + foot_index were visible but the foot never
        # rotated past the deadband — patient didn't perform
        # the test. Surfaced as HTTP 400 by api.analyze_hip.
        raise ValueError(
            "Hip rotation not detected. Please sit upright on a chair "
            "with both feet flat on the ground, then rotate the test "
            "leg outward and inward at the hip while keeping the heel "
            "planted as the pivot."
        )

    # Signed peak values for the response (preserve direction).
    primary_peak_signed = (
        -float(primary_peak_mag) if primary_peak_mag is not None else None
    )
    secondary_peak_signed = (
        float(secondary_peak_mag) if secondary_peak_mag is not None else None
    )

    primary_mag = (
        float(primary_peak_mag) if primary_peak_mag is not None else 0.0
    )
    secondary_mag = (
        float(secondary_peak_mag) if secondary_peak_mag is not None else 0.0
    )

    # ── Asymmetry detection ──────────────────────────────────
    both_captured = primary_mag > 0 and secondary_mag > 0
    asymmetry_deg = (
        abs(primary_mag - secondary_mag) if both_captured else 0.0
    )
    asymmetry_flag = both_captured and asymmetry_deg > _HIP_ROT_ASYMMETRY_FLAG_DEG

    # ── Build response ───────────────────────────────────────
    p_lo, p_hi = _MERGED_HIPROT_PRIMARY_TARGET
    s_lo, s_hi = _MERGED_HIPROT_SECONDARY_TARGET
    p_target = p_hi
    p_pct = (primary_mag / p_target) * 100.0 if p_target > 0 else 0.0
    p_status = _classify_in_range(primary_mag, p_lo, p_hi)

    interpretation_primary = (
        f"Internal rotation ({side.capitalize()}) measured "
        f"{primary_mag:.1f}°, which is {p_pct:.0f}% of the "
        f"{p_lo:.0f}°–{p_hi:.0f}° normal range — {p_status}."
    )
    if secondary_mag > 0:
        s_status = _classify_in_range(secondary_mag, s_lo, s_hi)
        interpretation_secondary = (
            f"External rotation ({side.capitalize()}) measured "
            f"{secondary_mag:.1f}°, which is "
            f"{(secondary_mag / s_hi) * 100.0:.0f}% of the "
            f"{s_lo:.0f}°–{s_hi:.0f}° normal range — {s_status}."
        )
    else:
        interpretation_secondary = (
            "External rotation direction was not detected in this recording."
        )
    interpretation = f"{interpretation_primary} {interpretation_secondary}"
    if asymmetry_flag:
        interpretation += (
            f" Notable internal-vs-external asymmetry detected "
            f"({asymmetry_deg:.1f}°). Clinical correlation recommended."
        )
    interpretation += (
        " Note: hip rotation is measured from a 2D pose estimate; "
        "patient must be seated upright with the heel of the test "
        "leg planted as a pivot for accurate results."
    )

    # Two key frames — internal + external peaks. Matches the
    # other merged tests' layout (no neutral frame).
    key_frames: list[dict] = []
    if primary_peak_idx >= 0 and primary_mag > 0:
        kf = _grab_hip_key_frame(
            video_path, primary_peak_idx, raw,
            f"Internal Rotation ({primary_mag:.1f}°)", side,
        )
        if kf:
            key_frames.append(kf)
    if secondary_peak_idx >= 0 and secondary_mag > 0:
        kf = _grab_hip_key_frame(
            video_path, secondary_peak_idx, raw,
            f"External Rotation ({secondary_mag:.1f}°)", side,
        )
        if kf:
            key_frames.append(kf)

    return {
        "body_part": "hip",
        "movement": "rotation",
        "side": side,
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
        "primary_label": "Internal Rotation",
        "secondary_label": "External Rotation",
    }


# ─── Main entry point ───────────────────────────────────────────
def analyze_hip(
    video_path: str,
    pose_options,
    movement: str,
    side: str,
) -> dict:
    """Run the BlazePose-Full hip pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded (and optionally repaired)
                      video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        movement:     "flexion", "extension", or "rotation" — the
                      three hip tests now routed to backend.
                      Flexion / extension are single-direction
                      tests; rotation is a merged (internal +
                      external) test using the seated heel-fixed
                      pivot pipeline.
        side:         "left" or "right".

    Returns:
        Dict matching the single-direction BiomechData Pydantic
        schema (no secondary_peak_* fields — neither hip test is
        merged). Caller wraps in BiomechResponse.

    Raises:
        ValueError: input invalid, wrong-side selection, or
                    fewer than ~half a second of usable frames.
                    The endpoint maps these to HTTP 400 with the
                    original user-facing message preserved.
    """
    # Merged rotation runs through its own seated heel-fixed
    # pipeline (heel→foot_index angle from vertical + dual-peak
    # tracking + 2D-approximation caveat). Branch early so the
    # existing flexion / extension code below stays untouched.
    if movement == "rotation":
        return _analyze_hip_rotation(video_path, pose_options, side)

    if movement not in ("flexion", "extension"):
        raise ValueError(f"Unsupported hip movement: {movement!r}")
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    # Movement-specific clamp + display label. Same per-frame
    # math (180° − interior angle between trunk and thigh) is
    # used for both — the formula returns the unsigned deviation
    # from standing-straight, and the test movement the user
    # selected determines clinical interpretation + reference
    # range. Tight clamp on extension (anatomical max only 30°)
    # also acts as a sanity guard: if the patient accidentally
    # performed flexion when extension was selected, the reading
    # gets capped at 30° rather than reporting an implausible
    # large value as extension.
    if movement == "flexion":
        anatomical_max = _HIP_FLEXION_ANATOMICAL_MAX
        movement_label = "Hip flexion"
        peak_caption_prefix = "Peak Flexion"
    else:  # extension
        anatomical_max = _HIP_EXTENSION_ANATOMICAL_MAX
        movement_label = "Hip extension"
        peak_caption_prefix = "Peak Extension"

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # Pre-flight wrong-side check — fail fast with a clear
    # message rather than running a full analysis on the
    # unmoving leg.
    wrong_side_msg = _wrong_side_for_hip_video(ts, side)
    if wrong_side_msg:
        raise ValueError(wrong_side_msg)

    shoulder_key = f"{side}_shoulder"
    hip_key      = f"{side}_hip"
    knee_key     = f"{side}_knee"

    sx = ts[shoulder_key]["x_px"]; sy = ts[shoulder_key]["y_px"]; vs = ts[shoulder_key]["vis"]
    hx = ts[hip_key]["x_px"];      hy = ts[hip_key]["y_px"];      vh = ts[hip_key]["vis"]
    kx = ts[knee_key]["x_px"];     ky = ts[knee_key]["y_px"];     vk = ts[knee_key]["vis"]

    n = int(min(len(sx), len(hx), len(kx)))

    # Per-frame hip flexion (180° − interior). Matches the
    # browser computeHipAngle("flexion") output so live + upload
    # report the same metric. None for frames with low
    # shoulder / hip / knee visibility — the min/max tracker
    # just skips those frames, so the running peak holds its
    # last good value (critical at peak hip flexion where the
    # knee can leave the top of the frame).
    angles: list[Optional[float]] = []
    valid_frames = 0
    for i in range(n):
        if (vs[i] < _HIP_VIS_THRESHOLD
                or vh[i] < _HIP_VIS_THRESHOLD
                or vk[i] < _HIP_VIS_THRESHOLD):
            angles.append(None)
            continue
        sh_pt = (float(sx[i]), float(sy[i]))
        hp_pt = (float(hx[i]), float(hy[i]))
        kn_pt = (float(kx[i]), float(ky[i]))
        interior = _hip_interior_angle(sh_pt, hp_pt, kn_pt)
        if interior is None:
            angles.append(None)
            continue
        deviation = 180.0 - interior
        # Anatomical sanity: drop frames where the computed
        # deviation is clearly an artefact (knee/hip keypoint
        # mis-tracking). Per-movement ceiling: large for
        # flexion (knee can reach the chest), small for
        # extension (~30° anatomical max). Frames past the
        # ceiling get dropped, then accepted values are
        # clamped to the ceiling so a single noise spike can't
        # lock an implausible peak.
        if (deviation < _HIP_HYPEREXTENSION_LIMIT
                or deviation > anatomical_max + 15.0):
            angles.append(None)
            continue
        deviation = max(_HIP_HYPEREXTENSION_LIMIT,
                        min(anatomical_max, deviation))
        angles.append(deviation)
        valid_frames += 1

    if valid_frames < max(3, int(fps * 0.5)):
        # Less than ~half a second of usable footage. Surfaced
        # as "hip and leg not clearly visible" rather than a
        # generic analysis failure.
        raise ValueError("poor_visibility")

    # ── Min/max tracking ───────────────────────────────────
    # Peak = max deviation from standing-straight (the formula
    # gives unsigned magnitude, so this works for both flexion
    # and extension — the test type chosen by the user gives
    # the clinical direction).
    # Neutral = frame with smallest |deviation| (closest to
    # standing-straight). Used for the report's neutral key
    # frame so the operator can verify the starting pose.
    peak_mag: float = -math.inf
    peak_idx: int = -1
    neutral_idx: int = -1
    neutral_abs: float = math.inf
    for i, a in enumerate(angles):
        if a is None:
            continue
        if a > peak_mag:
            peak_mag = a
            peak_idx = i
        if abs(a) < neutral_abs:
            neutral_abs = abs(a)
            neutral_idx = i

    # Guard rail — unreachable given the valid_frames check
    # above but keeps the type-checker happy.
    if peak_idx < 0:
        raise ValueError("poor_visibility")

    # ── Build response ─────────────────────────────────────
    normal = HIP_NORMAL_RANGES[movement]
    ref_low, ref_high = normal["range"]
    target = normal["target"]
    percentage = (peak_mag / target) * 100.0 if target > 0 else 0.0
    status = _classify_in_range(peak_mag, float(ref_low), float(ref_high))

    interpretation = (
        f"{movement_label} ({side.capitalize()}) measured "
        f"{peak_mag:.1f}°, which is {percentage:.0f}% of the "
        f"{ref_low:.0f}°–{ref_high:.0f}° normal range — {status}."
    )

    # Two key frames: neutral (starting pose) + peak. Matches
    # the single-direction shoulder layout — operator gets
    # both the starting reference and the maximum-ROM snapshot
    # to verify the test was performed correctly.
    key_frames: list[dict] = []
    if neutral_idx >= 0:
        kf = _grab_hip_key_frame(
            video_path, neutral_idx, raw, "Neutral — start", side,
        )
        if kf:
            key_frames.append(kf)
    if peak_idx >= 0 and peak_mag > 0:
        kf = _grab_hip_key_frame(
            video_path, peak_idx, raw,
            f"{peak_caption_prefix} ({peak_mag:.1f}°)", side,
        )
        if kf:
            key_frames.append(kf)

    return {
        "body_part": "hip",
        "movement": movement,
        "side": side,
        "peak_angle": float(peak_mag),
        "peak_magnitude": float(peak_mag),
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
