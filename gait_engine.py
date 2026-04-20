"""
gait_engine.py
Core gait analysis engine.
Handles: Pose extraction (MediaPipe), preprocessing, 10-feature computation, insights.
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import math
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# LANDMARK INDICES (MediaPipe Pose)
# ──────────────────────────────────────────────
LM = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_heel":      29,
    "right_heel":     30,
    "left_foot_index":  31,
    "right_foot_index": 32,
}

VISIBILITY_THRESHOLD = 0.4


# ──────────────────────────────────────────────
# STAGE 1: POSE EXTRACTION
# ──────────────────────────────────────────────
def extract_poses(video_path: str, pose_model, progress_callback=None):
    """
    Extract landmark coordinates from every frame of the video.
    Returns a dict: { landmark_name: [(x, y, visibility)] | None per frame }
    Also returns fps and total_frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw = {name: [] for name in LM}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_model.process(rgb)

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            for name, idx in LM.items():
                lm = lms[idx]
                if lm.visibility >= VISIBILITY_THRESHOLD:
                    raw[name].append((lm.x, lm.y, lm.visibility))
                else:
                    raw[name].append(None)
        else:
            for name in LM:
                raw[name].append(None)

        frame_idx += 1
        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return raw, fps, total_frames


# ──────────────────────────────────────────────
# STAGE 2: PREPROCESSING
# ──────────────────────────────────────────────
def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values."""
    nans = np.isnan(arr)
    if nans.all():
        return arr
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def _remove_outliers(arr: np.ndarray, z_thresh=3.5) -> np.ndarray:
    """Replace points > z_thresh std-devs from mean with NaN, then re-interpolate."""
    if np.all(np.isnan(arr)):
        return arr
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0:
        return arr
    z = np.abs(arr - mean) / std
    arr[z > z_thresh] = np.nan
    return _interp_nans(arr)


def _smooth(arr: np.ndarray, window=11, poly=3) -> np.ndarray:
    """Apply Savitzky-Golay filter, adapting window to array length."""
    n = len(arr)
    if n < 5:
        return arr
    win = min(window, n if n % 2 != 0 else n - 1)
    if win < 5:
        win = 5
    if win % 2 == 0:
        win -= 1
    poly = min(poly, win - 1)
    try:
        return savgol_filter(arr, win, poly)
    except Exception:
        return arr


def build_time_series(raw: dict) -> dict:
    """
    Convert raw frame-by-frame list to smoothed numpy arrays per coordinate.
    Returns: { landmark_name: { 'x': np.array, 'y': np.array } }
    """
    ts = {}
    for name, frames in raw.items():
        xs = np.array([f[0] if f is not None else np.nan for f in frames])
        ys = np.array([f[1] if f is not None else np.nan for f in frames])

        xs = _interp_nans(xs)
        ys = _interp_nans(ys)

        xs = _remove_outliers(xs)
        ys = _remove_outliers(ys)

        xs = _smooth(xs)
        ys = _smooth(ys)

        ts[name] = {"x": xs, "y": ys}
    return ts


# ──────────────────────────────────────────────
# HELPER: angle between 3 points
# ──────────────────────────────────────────────
def _angle3(A, B, C):
    """Angle at B formed by A-B-C. All points are (x, y) tuples/arrays."""
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cos_a = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


# ──────────────────────────────────────────────
# STAGE 3: FEATURE EXTRACTION (10 FEATURES)
# ──────────────────────────────────────────────

def compute_step_count(ts: dict, fps: float) -> dict:
    """F1: Detect steps using heel X-coordinate peaks."""
    results = {}
    for side in ("left", "right"):
        heel_x = ts[f"{side}_heel"]["x"]
        # Distance between peaks: at least 0.3 seconds apart
        distance = max(int(0.3 * fps), 3)
        peaks, _ = find_peaks(heel_x, distance=distance, prominence=0.01)
        troughs, _ = find_peaks(-heel_x, distance=distance, prominence=0.01)
        step_indices = np.sort(np.concatenate([peaks, troughs]))
        results[side] = {"count": len(step_indices), "indices": step_indices}

    total = (results["left"]["count"] + results["right"]["count"]) // 2
    combined_indices = np.sort(np.concatenate([
        results["left"]["indices"], results["right"]["indices"]
    ]))
    return {
        "total_steps": max(total, len(combined_indices) // 2),
        "left_count": results["left"]["count"],
        "right_count": results["right"]["count"],
        "left_indices": results["left"]["indices"],
        "right_indices": results["right"]["indices"],
        "combined_indices": combined_indices,
    }


def compute_cadence(step_data: dict, total_frames: int, fps: float) -> float:
    """F2: Steps per minute."""
    duration = total_frames / fps
    if duration <= 0:
        return 0.0
    return round((step_data["total_steps"] / duration) * 60, 1)


def compute_knee_angles(ts: dict) -> dict:
    """F3: Frame-wise knee angles for both legs."""
    angles = {}
    for side in ("left", "right"):
        hip_x = ts[f"{side}_hip"]["x"]
        hip_y = ts[f"{side}_hip"]["y"]
        knee_x = ts[f"{side}_knee"]["x"]
        knee_y = ts[f"{side}_knee"]["y"]
        ankle_x = ts[f"{side}_ankle"]["x"]
        ankle_y = ts[f"{side}_ankle"]["y"]

        n = len(knee_x)
        ang = np.zeros(n)
        for i in range(n):
            A = (hip_x[i], hip_y[i])
            B = (knee_x[i], knee_y[i])
            C = (ankle_x[i], ankle_y[i])
            ang[i] = _angle3(A, B, C)
        angles[side] = ang
    angles["left_mean"] = float(np.nanmean(angles["left"]))
    angles["right_mean"] = float(np.nanmean(angles["right"]))
    angles["overall_mean"] = float(np.nanmean([angles["left_mean"], angles["right_mean"]]))
    return angles


def compute_step_timing(step_data: dict, fps: float) -> dict:
    """F4: Time between consecutive steps."""
    timing = {}
    for side in ("left", "right"):
        indices = step_data[f"{side}_indices"]
        if len(indices) >= 2:
            diffs = np.diff(indices) / fps
            timing[side] = diffs
        else:
            timing[side] = np.array([])

    all_diffs = []
    for side in ("left", "right"):
        if len(timing[side]) > 0:
            all_diffs.extend(timing[side].tolist())
    timing["mean_step_time"] = float(np.mean(all_diffs)) if all_diffs else 0.0
    return timing


def compute_symmetry(ts: dict) -> float:
    """F5: Left/right step symmetry (1.0 = perfect)."""
    l_x = ts["left_ankle"]["x"]
    r_x = ts["right_ankle"]["x"]
    L_mean = float(np.nanmean(np.abs(np.diff(l_x))))
    R_mean = float(np.nanmean(np.abs(np.diff(r_x))))
    denom = (L_mean + R_mean) / 2.0
    if denom < 1e-9:
        return 1.0
    sym = 1.0 - abs(L_mean - R_mean) / denom
    return round(float(np.clip(sym, 0.0, 1.0)), 3)


def compute_walking_direction(ts: dict) -> str:
    """F6: Determine walking direction (left→right or right→left)."""
    hip_x = (ts["left_hip"]["x"] + ts["right_hip"]["x"]) / 2.0
    n = len(hip_x)
    seg = max(n // 6, 5)
    start_med = np.nanmedian(hip_x[:seg])
    end_med = np.nanmedian(hip_x[-seg:])
    if end_med > start_med:
        return "Left → Right"
    elif end_med < start_med:
        return "Right → Left"
    else:
        return "Stationary / Unknown"


def compute_step_length(ts: dict, step_data: dict) -> dict:
    """F7: Normalized step length using heel separation at strike."""
    l_heel = ts["left_heel"]["x"]
    r_heel = ts["right_heel"]["x"]
    l_hip = ts["left_hip"]["x"]
    r_hip = ts["right_hip"]["x"]

    hip_width = np.nanmean(np.abs(l_hip - r_hip))
    if hip_width < 1e-9:
        hip_width = 1.0

    step_lengths = []
    left_idx = step_data["left_indices"]
    right_idx = step_data["right_indices"]

    for li in left_idx:
        if li < len(l_heel) and li < len(r_heel):
            sl = abs(l_heel[li] - r_heel[li]) / hip_width
            step_lengths.append(sl)

    for ri in right_idx:
        if ri < len(l_heel) and ri < len(r_heel):
            sl = abs(l_heel[ri] - r_heel[ri]) / hip_width
            step_lengths.append(sl)

    if not step_lengths:
        return {"values": np.array([]), "mean": 0.0}

    vals = np.array(step_lengths)
    return {"values": vals, "mean": float(np.nanmean(vals))}


def compute_stride_consistency(step_timing: dict) -> float:
    """F8: Coefficient of Variation of stride times (lower = more consistent)."""
    all_times = []
    for side in ("left", "right"):
        if len(step_timing[side]) > 0:
            all_times.extend(step_timing[side].tolist())
    if len(all_times) < 2:
        return 0.0
    arr = np.array(all_times)
    mean = np.mean(arr)
    if mean < 1e-9:
        return 0.0
    cv = float(np.std(arr) / mean)
    return round(cv, 3)


def compute_ankle_trajectory(ts: dict) -> dict:
    """F9: Ankle X trajectories over time."""
    return {
        "left_x": ts["left_ankle"]["x"],
        "right_x": ts["right_ankle"]["x"],
        "left_y": ts["left_ankle"]["y"],
        "right_y": ts["right_ankle"]["y"],
    }


def compute_torso_lean(ts: dict) -> dict:
    """F10: Frame-wise torso lean angle."""
    sh_x = (ts["left_shoulder"]["x"] + ts["right_shoulder"]["x"]) / 2.0
    sh_y = (ts["left_shoulder"]["y"] + ts["right_shoulder"]["y"]) / 2.0
    hp_x = (ts["left_hip"]["x"] + ts["right_hip"]["x"]) / 2.0
    hp_y = (ts["left_hip"]["y"] + ts["right_hip"]["y"]) / 2.0

    dx = sh_x - hp_x
    dy = -(sh_y - hp_y)  # invert Y (image coords)
    angles = np.degrees(np.arctan2(dx, dy))

    return {
        "angles": angles,
        "mean": float(np.nanmean(angles)),
        "std": float(np.nanstd(angles)),
    }


# ──────────────────────────────────────────────
# MASTER COMPUTE FUNCTION
# ──────────────────────────────────────────────
def compute_all_features(ts: dict, fps: float, total_frames: int) -> dict:
    """Run all 10 feature extractions and return a unified dict."""
    step_data = compute_step_count(ts, fps)
    cadence = compute_cadence(step_data, total_frames, fps)
    knee_angles = compute_knee_angles(ts)
    step_timing = compute_step_timing(step_data, fps)
    symmetry = compute_symmetry(ts)
    direction = compute_walking_direction(ts)
    step_length = compute_step_length(ts, step_data)
    stride_cv = compute_stride_consistency(step_timing)
    ankle_traj = compute_ankle_trajectory(ts)
    torso_lean = compute_torso_lean(ts)

    return {
        "step_data": step_data,
        "cadence": cadence,
        "knee_angles": knee_angles,
        "step_timing": step_timing,
        "symmetry": symmetry,
        "direction": direction,
        "step_length": step_length,
        "stride_cv": stride_cv,
        "ankle_trajectory": ankle_traj,
        "torso_lean": torso_lean,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps if fps > 0 else 0,
    }


# ──────────────────────────────────────────────
# STAGE 5 (PART): INSIGHTS & SUGGESTIONS
# ──────────────────────────────────────────────
def interpret(features: dict) -> dict:
    """
    Rule-based threshold analysis.
    Returns: { "observations": [...], "suggestions": [...] }
    """
    obs = []
    sug = []

    cadence = features["cadence"]
    symmetry = features["symmetry"]
    knee_mean = features["knee_angles"]["overall_mean"]
    stride_cv = features["stride_cv"]
    step_length_mean = features["step_length"]["mean"]
    torso_mean = features["torso_lean"]["mean"]
    direction = features["direction"]
    step_count = features["step_data"]["total_steps"]
    duration = features["duration_sec"]

    # ── Cadence ──────────────────────────────
    if cadence == 0:
        obs.append("⚠️ No steps detected — the subject may not be walking clearly in this video.")
        sug.append("Ensure a clear lateral (side-view) video showing a full walking cycle.")
    elif cadence < 80:
        obs.append(f"🐢 Cadence is low ({cadence} steps/min). Normal range is 100–120 steps/min.")
        sug.append("Focus on increasing walking pace with shorter, quicker steps.")
    elif cadence > 140:
        obs.append(f"🏃 Cadence is very high ({cadence} steps/min). May indicate running or rapid movement.")
        sug.append("If walking, ensure video captures a full stride cycle clearly.")
    else:
        obs.append(f"✅ Cadence is within normal range ({cadence} steps/min).")

    # ── Symmetry ─────────────────────────────
    if symmetry < 0.85:
        obs.append(f"⚠️ Gait asymmetry detected (score: {symmetry:.2f}). Left/right movement differs significantly.")
        sug.append("Consult a physical therapist to evaluate potential limb-length discrepancy or muscle imbalance.")
    elif symmetry < 0.93:
        obs.append(f"🟡 Mild asymmetry detected (score: {symmetry:.2f}). Slight left/right variation.")
        sug.append("Add single-leg balance exercises to improve bilateral coordination.")
    else:
        obs.append(f"✅ Gait symmetry is excellent (score: {symmetry:.2f}).")

    # ── Knee Angle ───────────────────────────
    if knee_mean < 150:
        obs.append(f"🦵 Average knee angle is {knee_mean:.1f}°. Knees flex well during stance phase.")
        if knee_mean < 130:
            sug.append("Excessive knee flexion detected — consider quadriceps strengthening exercises.")
        else:
            sug.append("Good knee flexion. Maintain with regular stretching.")
    else:
        obs.append(f"🦵 Average knee angle is {knee_mean:.1f}°. Minimal knee flexion — possible stiff-legged walking.")
        sug.append("Work on hip flexor and hamstring flexibility to improve knee bend during walking.")

    # ── Stride Consistency ───────────────────
    if stride_cv > 0.25:
        obs.append(f"📉 High stride variability (CV: {stride_cv:.2f}). Inconsistent step rhythm detected.")
        sug.append("Practice metronome-paced walking to improve stride regularity.")
    elif stride_cv > 0.15:
        obs.append(f"🟡 Moderate stride variability (CV: {stride_cv:.2f}).")
        sug.append("Focus on rhythmic walking patterns to reduce stride timing variation.")
    else:
        obs.append(f"✅ Stride consistency is good (CV: {stride_cv:.2f}).")

    # ── Torso Lean ───────────────────────────
    abs_lean = abs(torso_mean)
    if abs_lean > 15:
        direction_str = "forward" if torso_mean > 0 else "backward"
        obs.append(f"🏋️ Significant torso lean {direction_str} ({torso_mean:.1f}°).")
        sug.append("Practice upright posture walking — engage core and keep gaze forward.")
    elif abs_lean > 7:
        obs.append(f"🟡 Mild torso lean detected ({torso_mean:.1f}°).")
        sug.append("Strengthen core muscles (planks, bird-dogs) to improve trunk stability.")
    else:
        obs.append(f"✅ Torso is well-aligned ({torso_mean:.1f}° lean).")

    # ── Step Length ──────────────────────────
    if step_length_mean > 0:
        if step_length_mean < 0.3:
            obs.append(f"👣 Step length is relatively short ({step_length_mean:.2f} normalized).")
            sug.append("Focus on lengthening stride with hip extension exercises.")
        elif step_length_mean > 1.0:
            obs.append(f"👣 Step length is long ({step_length_mean:.2f} normalized). May indicate overstriding.")
            sug.append("Reduce overstride to lower heel-strike impact forces.")
        else:
            obs.append(f"✅ Step length appears normal ({step_length_mean:.2f} normalized).")

    # ── Walking Direction ─────────────────────
    obs.append(f"🧭 Walking direction detected: {direction}.")

    # ── Duration ─────────────────────────────
    obs.append(f"⏱️ Video duration analyzed: {duration:.1f} seconds, {step_count} steps detected.")

    return {"observations": obs, "suggestions": sug}
