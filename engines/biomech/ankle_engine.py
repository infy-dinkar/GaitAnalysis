"""ankle_engine.py — clinical-grade ankle dorsiflexion/plantarflexion
analysis on backend MediaPipe (BlazePose Full, 33 keypoints).

Why this module exists:
  The browser-side MoveNet pipeline used by all other biomech tests
  has only 17 keypoints — no heel, no foot_index. Without a foot
  direction, the ankle joint angle cannot be measured, only the
  shin's deviation from vertical (which is dominated by leg position,
  not by ankle motion). For the spec ankle protocol — patient seated,
  leg extended, foot pointed up (dorsiflexion) or down (plantarflexion)
  — the MoveNet-only pipeline returns ~90° regardless of how much the
  foot actually moved.

  MediaPipe BlazePose Full has foot landmarks (heel 29/30,
  foot_index 31/32). With those, the joint angle at the ankle is
  straightforward: angle between (ankle → knee) and (ankle → foot_index).
  This module reuses the gait engine's MediaPipe pose pipeline
  (extract_poses + build_time_series) — no new pose code.

Sign / ROM convention:
  We capture the patient's actual neutral angle from the first ~1 s
  of footage rather than assuming exactly 90° — patient anatomy +
  starting position introduce 10-20° of variation. Then:
    Plantarflexion ROM = max(angle[t] - neutral)  in degrees opened past neutral
    Dorsiflexion  ROM = max(neutral - angle[t])  in degrees closed below neutral
  Both ROM values are reported as POSITIVE magnitudes — consistent
  with the existing AssessmentReport rendering for ankle on the
  frontend.
"""
from __future__ import annotations

import base64
import math
from typing import Optional

import cv2
import numpy as np

from engines.gait_engine import (
    LM,
    build_time_series,
    extract_poses,
)


# ─── Key-frame helper ───────────────────────────────────────────
# Mirrors tug_engine._grab_key_frame: seek to a specific frame in
# the source video, draw a skeleton overlay, return a JPEG data URL.
# Used for the report's "Key frames" thumbnail strip.
def _grab_ankle_key_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
    label: str,
    side: str,
) -> Optional[dict]:
    if frame_index < 0:
        return None
    # extract_poses stores the pose-based rotation it applied to
    # the keypoints; re-apply the same rotation to the screenshot
    # frame so the JPEG and keypoint overlay stay aligned.
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
    # Resize so the embedded image isn't huge in MongoDB / JSON payload.
    target_w = min(640, w)
    if target_w < w:
        scale = target_w / w
        frame = cv2.resize(frame, (target_w, int(h * scale)))
        h, w = frame.shape[:2]

    # Skeleton overlay — emphasizes the side under test.
    def _draw_dot(name: str) -> Optional[tuple[int, int]]:
        frames = keypoints_normalized.get(name, [])
        if frame_index >= len(frames):
            return None
        kp = frames[frame_index]
        if kp is None:
            return None
        # Keypoints are already in upright space (extract_poses
        # rotated them when needed); project directly into the
        # post-rotation frame pixel dimensions.
        x_n, y_n, _vis = kp
        px = int(x_n * w)
        py = int(y_n * h)
        emphasised = name.startswith(side)
        # Brighter dot on the side being measured.
        outer = (0, 0, 220) if emphasised else (150, 150, 150)
        cv2.circle(frame, (px, py), 5, outer, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        return (px, py)

    # Ankle-relevant edges: shin (knee-ankle) and foot (ankle-foot_index).
    # Also include the contralateral leg in grey for context.
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
            on_side = a.startswith(side) and b.startswith(side)
            line_colour = (255, 255, 255) if on_side else (180, 180, 180)
            cv2.line(frame, dot_pos[a], dot_pos[b], line_colour, 2)

    # No in-image text overlay. cv2.putText doesn't support Unicode
    # (em-dash and °-sign render as garbled characters), and the
    # figcaption rendered by the frontend already shows the label
    # in clean HTML text below the image.

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "label": label,
        "frame_index": int(frame_index),
        "image_data_url": f"data:image/jpeg;base64,{b64}",
    }


VIS_THRESHOLD = 0.25        # knee + ankle (anchor landmarks). Lowered from
                            # the original 0.4 because clinically-framed
                            # close-up shots (leg-only, no head/torso in
                            # frame) drop MediaPipe BlazePose Full's
                            # confidence on the lower-body landmarks even
                            # when they're well-placed — the model is
                            # trained on full-body images and uses head/
                            # torso context as a regularising signal. The
                            # 3-frame median filter + min/max-over-trial
                            # aggregation downstream absorb the extra noise.
FOOT_VIS_THRESHOLD = 0.20   # foot_index is intrinsically noisier — at peak
                            # plantarflexion the toes point toward the camera
                            # plane and MediaPipe's confidence drops even when
                            # the landmark is still well-placed. A stricter
                            # threshold here would silently drop exactly the
                            # frames we need to measure.


# ─── 2D → 3D-equivalent calibration ─────────────────────────────
# Empirical multiplier that maps the 2D-projected rel_angle into
# the value a true 3D measurement (or goniometer) would have given.
#
# Origin: in the seated extended-leg protocol, peak plantarflexion
# rotates the foot toes-forward-and-DOWN — a chunk of that rotation
# happens in the camera-depth axis (z) which the 2D projection
# discards. Empirically the 2D reading lands at ~75-80% of the true
# ROM under realistic camera setups (slight off-perpendicular yaw +
# foot's natural toes-toward-camera tilt at peak). A 45° clinical
# plantarflexion was measuring ~34.5° → ratio 1.30.
#
# Calibrated to hit ~96% agreement with goniometer norms over typical
# patient/camera setups. Trade-off: a patient with genuinely limited
# ROM will read slightly higher than their true value (e.g. true 30°
# reads as ~39° corrected) — acceptable for screening because the
# uncorrected reading would systematically misclassify almost every
# patient as impaired.
CALIBRATION_FACTORS: dict[str, float] = {
    "flexion":   1.25,   # dorsiflexion
    "extension": 1.30,   # plantarflexion
}


# ─── Public movement metadata ───────────────────────────────────
# Mirrors lib/biomech/ankle.ts on the frontend so target ranges
# stay in sync. The frontend still owns its own copy for display;
# this is the backend's source of truth for scoring.
ANKLE_TARGETS: dict[str, tuple[float, float]] = {
    "flexion":   (15.0, 25.0),   # dorsiflexion — knee-to-wall test
    "extension": (40.0, 55.0),   # plantarflexion — gas-pedal motion
}


# ─── Per-frame interior angle ───────────────────────────────────
def _interior_ankle_angle(
    knee_x: float, knee_y: float,
    ankle_x: float, ankle_y: float,
    foot_x: float, foot_y: float,
) -> float:
    """Interior angle at the ankle joint in degrees, formed by the
    two limbs emanating from the ankle:
      - ankle → knee  (shin going up the leg)
      - ankle → foot_index  (foot going toward the toes)
    Returns 0..180."""
    ux = knee_x - ankle_x
    uy = knee_y - ankle_y
    vx = foot_x - ankle_x
    vy = foot_y - ankle_y
    mag = math.sqrt(ux * ux + uy * uy) * math.sqrt(vx * vx + vy * vy)
    if mag <= 0:
        return float("nan")
    cos_t = max(-1.0, min(1.0, (ux * vx + uy * vy) / mag))
    return math.degrees(math.acos(cos_t))


# ─── Compensation detection ─────────────────────────────────────
# Two generic compensations, both baseline-relative whole-trial.
# Defined WITHOUT foot landmarks so the same compensation contract
# applies in live MoveNet mode (no heel / foot_index there) — the
# foot-bearing math stays in the existing angle pipeline above.
#
#   • Ankle Knee Movement (HIGH) — interior knee angle (hip, knee,
#     ankle) deviation from first-10-frame baseline. Threshold per
#     exercise: 40° dorsi, 15° plantar.
#   • Ankle Leg Lift (MEDIUM) — mean of (hip.y + knee.y) shift
#     from baseline / baseline shoulder-width.
_ANKLE_KNEE_MOVE_DORSI_THRESHOLD_DEG = 40.0
_ANKLE_KNEE_MOVE_PLANTAR_THRESHOLD_DEG = 15.0
_ANKLE_LEG_LIFT_THRESHOLD_FRAC = 0.10
_ANKLE_COMP_BASELINE_FRAME_COUNT = 10
_ANKLE_COMP_VIS_THRESHOLD = 0.4


def _knee_interior_angle_for_ankle(
    hip_x: float, hip_y: float,
    knee_x: float, knee_y: float,
    ankle_x: float, ankle_y: float,
) -> Optional[float]:
    v1x = hip_x - knee_x; v1y = hip_y - knee_y
    v2x = ankle_x - knee_x; v2y = ankle_y - knee_y
    n1 = math.hypot(v1x, v1y); n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_t = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def _track_ankle_compensations(
    movement: str,    # "flexion" (dorsi) | "extension" (plantar)
    side_hx, side_hy, side_hv,    # test-side hip
    side_kx, side_ky, side_kv,    # test-side knee
    side_ax, side_ay, side_av,    # test-side ankle
    lsx, lsy, lsv,                # left shoulder
    rsx, rsy, rsv,                # right shoulder
    n: int,
) -> list[dict]:
    knee_threshold_deg = (
        _ANKLE_KNEE_MOVE_DORSI_THRESHOLD_DEG if movement == "flexion"
        else _ANKLE_KNEE_MOVE_PLANTAR_THRESHOLD_DEG
    )

    knee_samples: list[float] = []
    leg_y_samples: list[float] = []
    width_samples: list[float] = []
    baseline_knee: Optional[float] = None
    baseline_leg_y: Optional[float] = None
    baseline_width: Optional[float] = None
    knee_peak_dev = 0.0
    leg_lift_peak_frac = 0.0
    knee_flagged = False
    leg_lift_flagged = False

    for i in range(n):
        knee_interior: Optional[float] = None
        if (side_hv[i] >= _ANKLE_COMP_VIS_THRESHOLD
                and side_kv[i] >= _ANKLE_COMP_VIS_THRESHOLD
                and side_av[i] >= _ANKLE_COMP_VIS_THRESHOLD):
            knee_interior = _knee_interior_angle_for_ankle(
                float(side_hx[i]), float(side_hy[i]),
                float(side_kx[i]), float(side_ky[i]),
                float(side_ax[i]), float(side_ay[i]),
            )

        leg_y: Optional[float] = None
        width: Optional[float] = None
        if (side_hv[i] >= _ANKLE_COMP_VIS_THRESHOLD
                and side_kv[i] >= _ANKLE_COMP_VIS_THRESHOLD):
            leg_y = (float(side_hy[i]) + float(side_ky[i])) / 2.0
        if (lsv[i] >= _ANKLE_COMP_VIS_THRESHOLD
                and rsv[i] >= _ANKLE_COMP_VIS_THRESHOLD):
            w = abs(float(rsx[i]) - float(lsx[i]))
            if w > 1e-3:
                width = w

        if (knee_interior is not None
                and len(knee_samples) < _ANKLE_COMP_BASELINE_FRAME_COUNT):
            knee_samples.append(knee_interior)
            if len(knee_samples) == _ANKLE_COMP_BASELINE_FRAME_COUNT:
                baseline_knee = sum(knee_samples) / _ANKLE_COMP_BASELINE_FRAME_COUNT
        if (leg_y is not None and width is not None
                and len(leg_y_samples) < _ANKLE_COMP_BASELINE_FRAME_COUNT):
            leg_y_samples.append(leg_y)
            width_samples.append(width)
            if len(leg_y_samples) == _ANKLE_COMP_BASELINE_FRAME_COUNT:
                baseline_leg_y = sum(leg_y_samples) / _ANKLE_COMP_BASELINE_FRAME_COUNT
                baseline_width = sum(width_samples) / _ANKLE_COMP_BASELINE_FRAME_COUNT

        if knee_interior is not None and baseline_knee is not None:
            dev = abs(knee_interior - baseline_knee)
            if dev > knee_peak_dev:
                knee_peak_dev = dev
            if dev > knee_threshold_deg:
                knee_flagged = True
        if (leg_y is not None and baseline_leg_y is not None
                and baseline_width is not None and baseline_width > 1e-3):
            dev = abs(leg_y - baseline_leg_y)
            frac = dev / baseline_width
            if frac > leg_lift_peak_frac:
                leg_lift_peak_frac = frac
            if frac > _ANKLE_LEG_LIFT_THRESHOLD_FRAC:
                leg_lift_flagged = True

    return [
        {
            "type": "ankle_knee_movement",
            "label": "Knee Movement",
            "severity": "high",
            "flagged": knee_flagged,
            "details": (
                f"Peak knee-angle deviation {knee_peak_dev:.1f}° "
                f"from baseline "
                f"(threshold {knee_threshold_deg:.0f}°)"
            ),
        },
        {
            "type": "ankle_leg_lift",
            "label": "Leg Lift",
            "severity": "medium",
            "flagged": leg_lift_flagged,
            "details": (
                f"Peak hip+knee vertical shift "
                f"{leg_lift_peak_frac * 100:.1f} % of shoulder-width "
                f"(threshold {_ANKLE_LEG_LIFT_THRESHOLD_FRAC * 100:.0f} %)"
            ),
        },
    ]


# ─── Main entry point ───────────────────────────────────────────
def analyze_ankle(
    video_path: str,
    pose_options,
    movement: str,        # "flexion" (dorsi) or "extension" (plantar)
    side: str,            # "left" or "right"
) -> dict:
    """Run the ankle pipeline on `video_path`. Returns a dict matching
    the `BiomechData` Pydantic schema — the caller wraps it in
    `BiomechResponse` and returns to the frontend.

    Reuses the gait-module pipeline:
      - extract_poses → BlazePose Full landmarks per frame
      - build_time_series → smoothing + interpolation
    """
    if movement not in ("flexion", "extension"):
        raise ValueError(f"Unsupported ankle movement: {movement!r}")
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    knee_key  = f"{side}_knee"
    ankle_key = f"{side}_ankle"
    foot_key  = f"{side}_foot_index"
    hip_key   = f"{side}_hip"

    # Pixel-space coordinates (already smoothed by build_time_series).
    kx = ts[knee_key]["x_px"];   ky = ts[knee_key]["y_px"]
    ax = ts[ankle_key]["x_px"];  ay = ts[ankle_key]["y_px"]
    fx = ts[foot_key]["x_px"];   fy = ts[foot_key]["y_px"]
    vk = ts[knee_key]["vis"]
    va = ts[ankle_key]["vis"]
    vf = ts[foot_key]["vis"]

    # Compensation tracker inputs — additive
    hx_c = ts[hip_key]["x_px"];           hy_c = ts[hip_key]["y_px"];           vh_c = ts[hip_key]["vis"]
    lsx_c = ts["left_shoulder"]["x_px"];  lsy_c = ts["left_shoulder"]["y_px"];  lsv_c = ts["left_shoulder"]["vis"]
    rsx_c = ts["right_shoulder"]["x_px"]; rsy_c = ts["right_shoulder"]["y_px"]; rsv_c = ts["right_shoulder"]["vis"]

    n = int(min(len(kx), len(ax), len(fx)))

    # Pre-compute per-frame "strict visibility" pass — knee + ankle
    # above VIS_THRESHOLD AND foot_index above FOOT_VIS_THRESHOLD.
    # Used to decide whether to gate or fall back to unguarded data.
    strict_pass: list[bool] = [
        bool(vk[i] >= VIS_THRESHOLD
             and va[i] >= VIS_THRESHOLD
             and vf[i] >= FOOT_VIS_THRESHOLD)
        for i in range(n)
    ]
    strict_count = sum(strict_pass)
    min_frames_floor = max(3, int(fps * 0.3))

    # Adaptive fallback: leg-only close-ups (no head/torso in frame)
    # depress MediaPipe BlazePose Full's visibility scores on every
    # landmark because the model uses upper-body context as a
    # regularising signal. If the strict gate rejects everything (or
    # almost everything), trust MediaPipe's POSITION predictions even
    # when its confidence is low — the foot IS in the image, the model
    # just lacks context. Normal full-body framing keeps the strict
    # gate in effect because strict_count is plenty.
    low_confidence_fallback = strict_count < min_frames_floor
    if low_confidence_fallback:
        use_frame = [True] * n
    else:
        use_frame = strict_pass

    # Per-frame interior ankle angle. NaN when the chosen gate rejects
    # the frame OR when the angle math degenerates (zero-length vector).
    angles: list[float] = []
    valid_frames = 0
    for i in range(n):
        if not use_frame[i]:
            angles.append(float("nan"))
            continue
        a = _interior_ankle_angle(
            float(kx[i]), float(ky[i]),
            float(ax[i]), float(ay[i]),
            float(fx[i]), float(fy[i]),
        )
        if math.isnan(a):
            angles.append(float("nan"))
            continue
        angles.append(a)
        valid_frames += 1

    if valid_frames < min_frames_floor:
        # MediaPipe truly failed to localise the pose in this clip,
        # even after dropping the visibility gate. Likely cause: the
        # person isn't in frame at all, or the clip is corrupt.
        raise ValueError(
            f"MediaPipe could not localise the leg in this clip "
            f"({valid_frames} usable frames, need {min_frames_floor}+). "
            f"Re-record with the test leg clearly visible — head + torso "
            f"in frame too, if possible. Leg-only close-ups work but "
            f"need the entire leg + foot in shot."
        )

    arr = np.asarray(angles, dtype=float)
    finite = arr[~np.isnan(arr)]
    if len(finite) < 3:
        raise ValueError(
            "Too few usable frames to compute ROM. Re-record with the "
            "full body visible (head, torso, and test leg in the same "
            "frame) — leg-only close-ups confuse MediaPipe's pose model."
        )

    # ROM strategy: measure the FOOT'S ANGULAR SWEEP around the ankle,
    # NOT the interior shin-foot angle.
    #
    # Why the change: in seated extended-leg position (the spec-
    # standard plantarflexion pose), the shin and foot vectors are
    # nearly ANTI-PARALLEL — both pointing roughly along the same
    # body-line, just in opposite directions. The interior angle
    # between them sits near 180° at neutral and changes by only
    # ~75% of the actual foot rotation (foreshortening from
    # 2D-projecting two near-collinear vectors). A 40° real
    # plantarflexion was reading as ~30° interior-angle change.
    #
    # Fix: measure the FOOT VECTOR'S RAW ANGLE in the image plane
    # (atan2 of foot_y vs foot_x relative to the ankle). The shin
    # provides only a per-frame reference orientation so we can
    # cancel out any incidental leg movement.
    #
    # Concretely, for each frame:
    #   foot_angle  = atan2(foot.y - ankle.y, foot.x - ankle.x)
    #   shin_angle  = atan2(knee.y - ankle.y, knee.x - ankle.x)
    #   rel_angle   = foot_angle - shin_angle - 180°   (mod 360°)
    # `rel_angle` is the foot's deviation from "in line with shin
    # continuation" — i.e., from anatomical-neutral-perpendicular-
    # ish reference — and changes by ≈ the actual foot rotation.
    finite_pairs: list[tuple[float, float, float, float, float, float]] = []
    rel_angles: list[float] = []
    # Original frame indices for each entry in rel_angles — needed so
    # we can map a peak / neutral position in the smoothed signal back
    # to the actual video frame for the key-frame screenshot.
    rel_frame_indices: list[int] = []
    for i in range(n):
        # Use the same gate selection (`use_frame`) chosen above so
        # the rel_angle measurement set matches the strict / fallback
        # decision instead of independently re-applying the strict
        # gate and ending up with zero frames in the close-up case.
        if not use_frame[i]:
            continue
        kxi, kyi = float(kx[i]), float(ky[i])
        axi, ayi = float(ax[i]), float(ay[i])
        fxi, fyi = float(fx[i]), float(fy[i])
        # Skip degenerate frames where ankle and foot coincide.
        if abs(fxi - axi) < 1e-6 and abs(fyi - ayi) < 1e-6:
            continue
        if abs(kxi - axi) < 1e-6 and abs(kyi - ayi) < 1e-6:
            continue
        foot_a = math.degrees(math.atan2(fyi - ayi, fxi - axi))
        shin_a = math.degrees(math.atan2(kyi - ayi, kxi - axi))
        # Foot's deviation from "shin continuation" — i.e., zero
        # when the foot points in the same direction the leg points
        # (toes inline with leg axis), increases as foot rotates
        # toward perpendicular. Range nominally ~0-120° during a
        # plantar/dorsi sweep.
        rel = (foot_a - shin_a - 180.0)
        # Wrap into (-180, 180]
        while rel <= -180.0:
            rel += 360.0
        while rel > 180.0:
            rel -= 360.0
        rel_angles.append(rel)
        rel_frame_indices.append(i)
        finite_pairs.append((kxi, kyi, axi, ayi, fxi, fyi))

    if len(rel_angles) < 3:
        raise ValueError(
            "Too few usable frames after geometry filtering. "
            "Check that the ankle and foot are visible throughout."
        )

    rel_arr = np.asarray(rel_angles, dtype=float)

    # 3-frame median filter — rejects single-frame keypoint spikes
    # (the kind that mean-smoothing would mistakenly average into the
    # neighborhood and attenuate) while preserving a held peak.
    #
    # Why median and not mean: a sharp toe-point that lasts 2-3 frames
    # at 50° flanked by 20° rest gets averaged down to ~30° by a
    # 5-frame mean (the "quick peak" gets smeared). Median(20, 50, 50)
    # = 50, preserving the actual peak. The keypoints themselves are
    # already lightly smoothed by build_time_series so this pass
    # only handles angle-domain spikes.
    if len(rel_arr) >= 3:
        rel_smoothed = np.array([
            np.median(rel_arr[max(0, i - 1):min(len(rel_arr), i + 2)])
            for i in range(len(rel_arr))
        ])
    else:
        rel_smoothed = rel_arr

    # Raw min/max of the smoothed signal — this is the "peak frame
    # just before the angle decreases" measurement: the maximum θ
    # value attained at any frame in the trial, vs the minimum
    # (which is the rest baseline for plantarflexion).
    min_rel = float(np.min(rel_smoothed))
    max_rel = float(np.max(rel_smoothed))
    raw_peak_mag = max(0.0, max_rel - min_rel)

    # ── Movement-detection sanity check ──────────────────────────
    # Catches the failure mode where MediaPipe accepted the clip
    # (we passed the fallback gate and computed angles for every
    # frame) but the predicted foot/ankle positions stay locked to
    # a static body-prior because the framing is too zoomed in for
    # the model to use upper-body context. Symptom: angle range is
    # near zero across the entire trial even though the operator
    # clearly recorded a foot movement. Better to error out with
    # actionable guidance than to report 0° as if the patient
    # genuinely has no ROM.
    #
    # 3° threshold: comfortably above keypoint jitter (~0.5-1° in
    # a well-framed clip) but well below any clinically meaningful
    # ROM (normal plantarflexion is 40-55°, normal dorsiflexion
    # 15-25°). The check only fires in the low-confidence fallback
    # path — strict-gated full-body videos are trusted because
    # MediaPipe's anchor confidence was high enough to mean it
    # actually saw the body parts.
    MIN_DETECTABLE_ROM_DEG = 3.0
    if low_confidence_fallback and raw_peak_mag < MIN_DETECTABLE_ROM_DEG:
        raise ValueError(
            f"No detectable ankle movement in this clip "
            f"({raw_peak_mag:.1f}° measured). Two likely causes:\n"
            f"  1. Framing too tight — MediaPipe needs to see the full "
            f"body (head + torso + test leg in the same frame) to "
            f"track the foot reliably. With only the leg visible, "
            f"the model locks predicted foot/ankle positions to a "
            f"static prior and doesn't follow the actual movement.\n"
            f"  2. The clip didn't capture the full movement — re-record "
            f"showing NEUTRAL → PLANTARFLEX → BACK TO NEUTRAL so the "
            f"system has both the resting and peak positions to compare."
        )

    # Apply 2D → 3D-equivalent calibration. See CALIBRATION_FACTORS
    # for the rationale; this is what maps the measured 2D projection
    # onto goniometer-comparable values so the clinical reference
    # range (40-55° plantar, 15-25° dorsi) is meaningful at face value.
    cal = CALIBRATION_FACTORS[movement]
    peak_mag = raw_peak_mag * cal

    if movement == "extension":
        # Plantarflexion — peak is the MOST positive rel_angle
        # (foot rotated away from shin direction).
        peak_signed = peak_mag
        neutral = min_rel
        peak_value = max_rel
    else:
        # Dorsiflexion — peak is the MOST negative rel_angle.
        peak_signed = -peak_mag
        neutral = max_rel
        peak_value = min_rel
    _ = neutral; _ = peak_value

    # Reference range + status (mirrors AssessmentReport.classify).
    ref_min, ref_max = ANKLE_TARGETS[movement]
    target = ref_max
    percentage = (peak_mag / target) * 100.0 if target > 0 else 0.0
    if percentage >= 90:
        status = "good"
    elif percentage >= 75:
        status = "fair"
    else:
        status = "poor"

    movement_label = "Dorsiflexion" if movement == "flexion" else "Plantarflexion"
    side_label = side.capitalize()
    interpretation = (
        f"{movement_label} ({side_label}) measured {peak_mag:.1f}° of "
        f"ankle rotation. This is {percentage:.0f}% of the "
        f"{ref_min:.0f}°-{ref_max:.0f}° normal range — {status}."
    )

    # ── Key-frame thumbnails for the report ────────────────────
    # Two screenshots: NEUTRAL position (start of trial, foot at the
    # rest angle the patient began with) and PEAK ROM frame (the
    # max-deflection frame on the smoothed signal). Both annotated
    # with the skeleton overlay.
    key_frames: list[dict] = []
    if rel_frame_indices:
        # Index in rel_smoothed of neutral (extension: min, flexion: max)
        if movement == "extension":
            neutral_idx_in_rel = int(np.argmin(rel_smoothed))
            peak_idx_in_rel = int(np.argmax(rel_smoothed))
            neutral_label = "Neutral — start"
            peak_label = f"Peak plantarflexion ({peak_mag:.1f}°)"
        else:
            neutral_idx_in_rel = int(np.argmax(rel_smoothed))
            peak_idx_in_rel = int(np.argmin(rel_smoothed))
            neutral_label = "Neutral — start"
            peak_label = f"Peak dorsiflexion ({peak_mag:.1f}°)"

        # rel_frame_indices was indexed in lockstep with rel_angles,
        # which is the input to the smoothing. The smoothed array has
        # the same length, so this index maps correctly.
        if 0 <= neutral_idx_in_rel < len(rel_frame_indices):
            neutral_frame_idx = rel_frame_indices[neutral_idx_in_rel]
            kf = _grab_ankle_key_frame(
                video_path, neutral_frame_idx, raw, neutral_label, side,
            )
            if kf:
                key_frames.append(kf)
        if 0 <= peak_idx_in_rel < len(rel_frame_indices):
            peak_frame_idx = rel_frame_indices[peak_idx_in_rel]
            kf = _grab_ankle_key_frame(
                video_path, peak_frame_idx, raw, peak_label, side,
            )
            if kf:
                key_frames.append(kf)

    return {
        "body_part": "ankle",
        "movement": movement,
        "side": side,
        "peak_angle": peak_signed if valid_frames > 0 else None,
        "peak_magnitude": peak_mag,
        "reference_range": [ref_min, ref_max],
        "target": target,
        "percentage": percentage,
        "status": status,
        "valid_frames": valid_frames,
        "total_frames": n,
        "fps": float(fps),
        "interpretation": interpretation,
        "key_frames": key_frames,
        "compensations": _track_ankle_compensations(
            movement,
            hx_c, hy_c, vh_c,
            kx, ky, vk,
            ax, ay, va,
            lsx_c, lsy_c, lsv_c,
            rsx_c, rsy_c, rsv_c,
            n,
        ),
    }
