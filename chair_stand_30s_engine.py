"""chair_stand_30s_engine.py — backend 30-Second Chair Stand test
(Test C3) on the MediaPipe BlazePose Full (33-keypoint) pipeline.

Mirrors the frontend live-mode math in
motionlens-web/lib/orthopedic/chairStand30s.ts. Differs from the
5xSTS engine (sit_to_stand_engine.py) in three ways:

  1. Timer-driven, NOT rep-driven — analyze the first 30 seconds
     of the clip (or the full clip if shorter); reps continue to
     accrue during that window.
  2. Primary outcome = rep COUNT (not total time).
  3. Fatigue is reported as a linear regression slope
     (seconds-per-rep added per rep-index) rather than the
     last/first ratio used by 5xSTS.

Classification uses CDC STEADI age + sex norm bands. Below-threshold
rep count = positive screen for fall risk.
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Optional

import cv2

from gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/chairStand30s.ts) ────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TRIAL_DURATION_SEC = 30.0
_STAND_DELTA_FRAC = 0.20
_SIT_DELTA_FRAC = 0.08
_ARM_UNCROSS_TORSO_FRAC = 0.50

# ─── CDC STEADI age + sex norm bands ────────────────────────────
# Mirrors getChairStand30sNorm() in lib/orthopedic/normsDatabase.ts.
# Below-this-many-reps = at risk per CDC STEADI.
_CHAIR_STAND_BANDS: list[dict] = [
    {"age_min": 60, "age_max": 64, "male": 14, "female": 12},
    {"age_min": 65, "age_max": 69, "male": 12, "female": 11},
    {"age_min": 70, "age_max": 74, "male": 12, "female": 10},
    {"age_min": 75, "age_max": 79, "male": 11, "female": 10},
    {"age_min": 80, "age_max": 84, "male": 10, "female": 9},
    {"age_min": 85, "age_max": 89, "male": 8,  "female": 8},
    {"age_min": 90, "age_max": 94, "male": 7,  "female": 4},
]
_CHAIR_STAND_GENERIC_MALE = 11
_CHAIR_STAND_GENERIC_FEMALE = 10


def _get_chair_stand_norm(
    age: Optional[int],
    sex: Optional[str],
) -> dict:
    """Returns {threshold, comparable, band_label}. Mirrors the JS
    getChairStand30sNorm helper."""
    if age is None or sex is None or sex == "other":
        return {
            "threshold": (
                _CHAIR_STAND_GENERIC_FEMALE if sex == "female"
                else _CHAIR_STAND_GENERIC_MALE
            ),
            "comparable": False,
            "band_label": "generic threshold (patient demographics incomplete)",
        }
    if age < 60:
        band = _CHAIR_STAND_BANDS[0]
        return {
            "threshold": int(band[sex]),
            "comparable": False,
            "band_label": (
                f"closest CDC band (60–64) — patient age {age} is below "
                f"the published range"
            ),
        }
    if age > 94:
        band = _CHAIR_STAND_BANDS[-1]
        return {
            "threshold": int(band[sex]),
            "comparable": False,
            "band_label": (
                f"closest CDC band (90–94) — patient age {age} is above "
                f"the published range"
            ),
        }
    for band in _CHAIR_STAND_BANDS:
        if band["age_min"] <= age <= band["age_max"]:
            sex_label = "men" if sex == "male" else "women"
            return {
                "threshold": int(band[sex]),
                "comparable": True,
                "band_label": (
                    f"CDC STEADI {band['age_min']}–{band['age_max']} {sex_label}"
                ),
            }
    return {
        "threshold": (
            _CHAIR_STAND_GENERIC_FEMALE if sex == "female"
            else _CHAIR_STAND_GENERIC_MALE
        ),
        "comparable": False,
        "band_label": "generic threshold (band lookup failed)",
    }


# ─── Per-frame helpers (identical to sit_to_stand_engine.py) ────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _compute_hip_mid_y(ts: dict, i: int) -> Optional[float]:
    l_vis = _visible(ts, "left_hip", i)
    r_vis = _visible(ts, "right_hip", i)
    if not l_vis and not r_vis:
        return None
    if l_vis and r_vis:
        return (float(ts["left_hip"]["y_px"][i]) + float(ts["right_hip"]["y_px"][i])) / 2.0
    return float(ts["left_hip"]["y_px"][i]) if l_vis else float(ts["right_hip"]["y_px"][i])


def _compute_shoulder_mid_y(ts: dict, i: int) -> Optional[float]:
    l_vis = _visible(ts, "left_shoulder", i)
    r_vis = _visible(ts, "right_shoulder", i)
    if not l_vis and not r_vis:
        return None
    if l_vis and r_vis:
        return (float(ts["left_shoulder"]["y_px"][i]) + float(ts["right_shoulder"]["y_px"][i])) / 2.0
    return float(ts["left_shoulder"]["y_px"][i]) if l_vis else float(ts["right_shoulder"]["y_px"][i])


def _compute_knee_angle(ts: dict, i: int) -> Optional[float]:
    best_angle: Optional[float] = None
    best_score = 0.0
    for hip_k, knee_k, ankle_k in [
        ("left_hip",  "left_knee",  "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ]:
        if not (_visible(ts, hip_k, i) and _visible(ts, knee_k, i) and _visible(ts, ankle_k, i)):
            continue
        min_score = min(
            float(ts[hip_k]["vis"][i]),
            float(ts[knee_k]["vis"][i]),
            float(ts[ankle_k]["vis"][i]),
        )
        if min_score <= best_score:
            continue
        hx = float(ts[hip_k]["x_px"][i]); hy = float(ts[hip_k]["y_px"][i])
        kx = float(ts[knee_k]["x_px"][i]); ky = float(ts[knee_k]["y_px"][i])
        ax = float(ts[ankle_k]["x_px"][i]); ay = float(ts[ankle_k]["y_px"][i])
        v1x, v1y = hx - kx, hy - ky
        v2x, v2y = ax - kx, ay - ky
        m1 = math.hypot(v1x, v1y)
        m2 = math.hypot(v2x, v2y)
        if m1 < 1e-6 or m2 < 1e-6:
            continue
        cos_theta = (v1x * v2x + v1y * v2y) / (m1 * m2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        best_angle = math.degrees(math.acos(cos_theta))
        best_score = min_score
    return best_angle


def _compute_leg_length_px(ts: dict, i: int) -> Optional[float]:
    best_len: Optional[float] = None
    best_score = 0.0
    for hip_k, ankle_k in [
        ("left_hip",  "left_ankle"),
        ("right_hip", "right_ankle"),
    ]:
        if not (_visible(ts, hip_k, i) and _visible(ts, ankle_k, i)):
            continue
        min_score = min(
            float(ts[hip_k]["vis"][i]),
            float(ts[ankle_k]["vis"][i]),
        )
        if min_score <= best_score:
            continue
        hx = float(ts[hip_k]["x_px"][i]); hy = float(ts[hip_k]["y_px"][i])
        ax = float(ts[ankle_k]["x_px"][i]); ay = float(ts[ankle_k]["y_px"][i])
        best_len = math.hypot(ax - hx, ay - hy)
        best_score = min_score
    return best_len


def _are_arms_crossed(ts: dict, i: int) -> bool:
    shoulder_y = _compute_shoulder_mid_y(ts, i)
    hip_y = _compute_hip_mid_y(ts, i)
    if shoulder_y is None or hip_y is None:
        return True
    torso_h = abs(hip_y - shoulder_y)
    if torso_h < 1.0:
        return True
    tolerance = torso_h * _ARM_UNCROSS_TORSO_FRAC
    if _visible(ts, "left_wrist", i):
        if float(ts["left_wrist"]["y_px"][i]) > shoulder_y + tolerance:
            return False
    if _visible(ts, "right_wrist", i):
        if float(ts["right_wrist"]["y_px"][i]) > shoulder_y + tolerance:
            return False
    return True


# ─── Fatigue regression + depth SD ──────────────────────────────
def _fatigue_slope_sec_per_rep(rep_durations: list[float]) -> float:
    """Linear regression slope of rep duration vs rep index. Positive
    = each successive rep is taking longer (fatigue). Returns 0 with
    fewer than 2 reps."""
    n = len(rep_durations)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(rep_durations) / n
    num = 0.0
    den = 0.0
    for i in range(n):
        num += (i - mean_x) * (rep_durations[i] - mean_y)
        den += (i - mean_x) ** 2
    return 0.0 if den == 0 else num / den


def _depth_sd_deg(min_knee_angles: list[float]) -> float:
    n = len(min_knee_angles)
    if n < 2:
        return 0.0
    mean = sum(min_knee_angles) / n
    variance = sum((v - mean) ** 2 for v in min_knee_angles) / n
    return float(math.sqrt(variance))


# ─── Deepest-knee screenshot ────────────────────────────────────
def _grab_deepest_knee_frame(
    video_path: str,
    frame_index: int,
    keypoints_normalized: dict,
) -> Optional[str]:
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

    def _kp(name: str) -> Optional[tuple[int, int]]:
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
        p = _kp(name)
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
        ("left_shoulder",  "left_elbow"),
        ("left_elbow",     "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow",    "right_wrist"),
    ]
    for a, b in edges:
        if a in dot_pos and b in dot_pos:
            cv2.line(frame, dot_pos[a], dot_pos[b], (255, 255, 255), 2)
    for _name, (px, py) in dot_pos.items():
        cv2.circle(frame, (px, py), 5, (0, 0, 220), -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ─── Main entry point ──────────────────────────────────────────
def analyze_chair_stand_30s(
    video_path: str,
    pose_options,
    patient_age: Optional[int] = None,
    patient_sex: Optional[str] = None,
) -> dict:
    """Run the 30-Second Chair Stand pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        patient_age:  patient age in years (for CDC STEADI norm).
        patient_sex:  'male' | 'female' | 'other' | None.

    Returns dict matching the frontend ChairStand30sResult shape so
    the existing ChairStand30sReport renders without translation.
    """
    raw, fps, _ = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_knee"]["y"]),
        len(ts["right_knee"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
    ))
    if n == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    # Visibility gate — same lateral-tolerant rule as STS.
    visible_frames = 0
    for i in range(n):
        trunk_ok = (
            (_visible(ts, "left_hip", i) or _visible(ts, "right_hip", i))
            and (_visible(ts, "left_shoulder", i) or _visible(ts, "right_shoulder", i))
        )
        leg_ok = (
            (_visible(ts, "left_hip", i) and _visible(ts, "left_knee", i) and _visible(ts, "left_ankle", i))
            or (_visible(ts, "right_hip", i) and _visible(ts, "right_knee", i) and _visible(ts, "right_ankle", i))
        )
        if trunk_ok and leg_ok:
            visible_frames += 1
    if visible_frames < max(3, int(n * 0.30)):
        raise ValueError("poor_visibility")

    # Baseline lock — first valid frame.
    baseline_y: Optional[float] = None
    leg_length_px: Optional[float] = None
    first_valid_idx = -1
    for i in range(n):
        if baseline_y is None:
            baseline_y = _compute_hip_mid_y(ts, i)
        if leg_length_px is None:
            leg_length_px = _compute_leg_length_px(ts, i)
        if baseline_y is not None and leg_length_px is not None:
            first_valid_idx = i
            break
    if baseline_y is None or leg_length_px is None or first_valid_idx < 0:
        raise ValueError("poor_visibility")

    # Rep state machine — same as sit_to_stand_engine.
    stand_threshold = baseline_y - leg_length_px * _STAND_DELTA_FRAC
    sit_threshold   = baseline_y - leg_length_px * _SIT_DELTA_FRAC
    timeout_frames = int(_TRIAL_DURATION_SEC * fps)

    current_state = "sitting"
    sit_events_ms: list[float] = [0.0]
    reps: list[dict] = []
    current_min_knee = 180.0
    arm_uncrossed_flag = False
    deepest_knee_overall = 180.0
    deepest_knee_frame = first_valid_idx
    last_valid_frame = first_valid_idx
    completed_within_timer = True

    for i in range(first_valid_idx, n):
        elapsed_frames = i - first_valid_idx
        if elapsed_frames >= timeout_frames:
            # 30 s window elapsed — STOP. Any partial rep beyond this
            # point doesn't count (PDF Test C3 spec).
            break

        hip_y = _compute_hip_mid_y(ts, i)
        if hip_y is None:
            continue
        last_valid_frame = i
        t_ms = (elapsed_frames / fps) * 1000.0

        knee = _compute_knee_angle(ts, i)
        if knee is not None and knee < current_min_knee:
            current_min_knee = knee
        if knee is not None and knee < deepest_knee_overall:
            deepest_knee_overall = knee
            deepest_knee_frame = i

        if not _are_arms_crossed(ts, i):
            arm_uncrossed_flag = True

        if current_state == "sitting" and hip_y < stand_threshold:
            current_state = "standing"
        elif current_state == "standing" and hip_y > sit_threshold:
            current_state = "sitting"
            sit_events_ms.append(t_ms)
            start_ms = sit_events_ms[-2]
            end_ms   = sit_events_ms[-1]
            duration_sec = max(0.0, (end_ms - start_ms) / 1000.0)
            reps.append({
                "rep_index": len(reps) + 1,
                "duration_seconds": float(duration_sec),
                "min_knee_angle_deg": float(current_min_knee),
            })
            current_min_knee = 180.0

    # If we didn't reach the 30s mark, the clip ran short.
    final_elapsed_frames = last_valid_frame - first_valid_idx
    if final_elapsed_frames < timeout_frames:
        completed_within_timer = False

    # Aggregates.
    rep_durations = [r["duration_seconds"] for r in reps]
    mean_dur = (sum(rep_durations) / len(rep_durations)) if rep_durations else 0.0
    fatigue_slope = _fatigue_slope_sec_per_rep(rep_durations)
    min_knee_angles = [r["min_knee_angle_deg"] for r in reps]
    depth_sd = _depth_sd_deg(min_knee_angles)

    rep_count = len(reps)

    # Norm lookup.
    norm = _get_chair_stand_norm(patient_age, patient_sex)
    threshold = int(norm["threshold"])
    classification = "above_norm" if rep_count >= threshold else "below_norm"
    norm_passed = rep_count >= threshold

    # Termination: live mode reports 'completed' when the 30s timer
    # expired and 'stopped' when the operator clicked stop. For
    # uploads we use 'completed' when the clip covered the full 30s
    # window and 'stopped' otherwise.
    termination = "completed" if completed_within_timer else "stopped"

    trial_duration_seconds = float(final_elapsed_frames / fps) if fps > 0 else 0.0

    # 10 Hz samples export.
    sample_step = max(1, int(round(fps / _SAMPLE_HZ)))
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    sample_end = min(last_valid_frame, first_valid_idx + timeout_frames - 1)
    for i in range(first_valid_idx, sample_end + 1, sample_step):
        t_ms = ((i - first_valid_idx) / fps) * 1000.0
        samples.append({
            "t_ms": float(t_ms),
            "hip_mid_y":      _maybe_float(_compute_hip_mid_y(ts, i)),
            "knee_angle_deg": _maybe_float(_compute_knee_angle(ts, i)),
            "arms_crossed":   bool(_are_arms_crossed(ts, i)),
        })
        kp_frame: list[dict] = [{"x": 0.0, "y": 0.0, "score": 0.0} for _ in range(33)]
        for name, idx_n in LM.items():
            frames = raw.get(name, [])
            if i < len(frames) and frames[i] is not None:
                x_n, y_n, vis = frames[i]
                kp_frame[idx_n] = {
                    "x": float(x_n) * float(raw.get("_frame_w", 1) or 1),
                    "y": float(y_n) * float(raw.get("_frame_h", 1) or 1),
                    "score": float(vis),
                }
        keypoints_export.append(kp_frame)

    last_rep_screenshot = _grab_deepest_knee_frame(video_path, deepest_knee_frame, raw)

    # Interpretation.
    parts: list[str] = []
    if rep_count == 0:
        interpretation = "No completed reps captured — re-run the trial."
    else:
        if classification == "above_norm":
            parts.append(
                f"Completed {rep_count} reps in 30 s — at or above the "
                f"{norm['band_label']} cutoff of {threshold} reps. "
                f"Lower-extremity strength within the expected range."
            )
        else:
            parts.append(
                f"Completed {rep_count} reps in 30 s — below the "
                f"{norm['band_label']} cutoff of {threshold} reps. "
                f"Positive screen for fall risk per CDC STEADI."
            )
        if not norm["comparable"]:
            parts.append(
                "Norm comparison is approximate — patient demographics "
                "were missing or outside the published 60–94 age range."
            )
        if fatigue_slope > 0.05 and rep_count >= 3:
            parts.append(
                f"Per-rep duration trended upward (~{fatigue_slope:.2f} s "
                f"added per rep) — suggests fatigue across the trial."
            )
        if arm_uncrossed_flag:
            parts.append(
                "Arms uncrossed at one or more points during the trial — "
                "strength assessment may be inflated."
            )
        interpretation = " ".join(parts)

    return {
        "rep_count": int(rep_count),
        "reps": reps,
        "rep_durations": rep_durations,
        "mean_rep_duration_sec": float(mean_dur),
        "depth_sd_deg": float(depth_sd),
        "fatigue_slope_sec_per_rep": float(fatigue_slope),
        "norm_threshold": threshold,
        "norm_band_label": norm["band_label"],
        "norm_comparable": bool(norm["comparable"]),
        "classification": classification,
        "arm_uncrossed_flag": bool(arm_uncrossed_flag),
        "termination": termination,
        "trial_duration_seconds": float(trial_duration_seconds),
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "samples": samples,
        "keypoints": keypoints_export,
        "last_rep_screenshot_data_url": last_rep_screenshot,
        # Extras
        "fps": float(fps),
        "total_frames": int(n),
        "valid_frames": int(visible_frames),
        "interpretation": interpretation,
        "norm_passed": bool(norm_passed),
    }


def _maybe_float(v: Optional[float]) -> Optional[float]:
    return float(v) if v is not None else None
