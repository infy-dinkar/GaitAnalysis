"""
gait_engine.py
Core gait analysis engine.

Pipeline:
  extract_poses  → MediaPipe landmarks (also captures W/H, fps).
  build_time_series → interpolate / outlier-reject / Savitzky-Golay smooth,
                      keep BOTH normalized and pixel coords.
  segment_passes → standalone helper, splits the video into stable-direction
                   passes and identifies steady-state ("core") frame ranges.
  compute_meters_per_pixel → standalone helper, anatomical scale from height.
  compute_metrics → standalone helper, computes the FULL metric dict for an
                    arbitrary subset of frame indices.  Called twice by
                    compute_all_features:  once with all frames (Total) and
                    once with steady-state frames (Clean).
  compute_all_features → produces { total_metrics, clean_metrics, ... } and
                         spreads `clean_metrics` to the top level so that the
                         existing app.py / gait_plots.py keys still resolve.
  interpret → reads ONLY from features['clean_metrics'].

Metric math fixes vs the original engine:
  • Knee flexion uses PIXEL coordinates (anisotropic-normalized coords were
    biasing the angle low; peak now reaches ~60-70° for normal swing).
  • Heel strikes are detected from heel-Y peaks (foot lowest in the image
    when planted) instead of heel-X peaks-and-troughs (which double-counted
    each gait cycle and broke step length / cadence).
  • Stride CV is reported as a PERCENTAGE.
  • Symmetry is step-time-based: 1 - |mean(L_stride) - mean(R_stride)| / mean.
  • Torso lean sign is flipped per pass so + always means leaning FORWARD in
    the walking direction, regardless of L→R or R→L pass.
  • Cadence and step length live ONLY inside validated passes (turning,
    acceleration, and deceleration frames are excluded by frame mask).
"""

import math
import warnings

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# LANDMARK INDICES
# ──────────────────────────────────────────────
LM = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
    "left_heel":      29, "right_heel":     30,
    "left_foot_index":  31, "right_foot_index": 32,
}

VISIBILITY_THRESHOLD = 0.4
DEFAULT_HEIGHT_CM    = 170
LEG_HEIGHT_RATIO     = 0.53


# ══════════════════════════════════════════════
# STAGE 1 — POSE EXTRACTION
# ══════════════════════════════════════════════
def extract_poses(video_path: str, pose_options, progress_callback=None):
    pose_model = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1

    raw = {name: [] for name in LM}
    raw["_frame_w"] = frame_w
    raw["_frame_h"] = frame_h

    frame_idx = 0
    last_ts_ms = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int((frame_idx * 1000) / fps)
        if ts_ms <= last_ts_ms:
            ts_ms = last_ts_ms + 1
        last_ts_ms = ts_ms

        result = pose_model.detect_for_video(mp_image, ts_ms)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]
            for name, idx in LM.items():
                lm = lms[idx]
                if lm.visibility and lm.visibility >= VISIBILITY_THRESHOLD:
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
    pose_model.close()
    return raw, fps, total_frames


# ══════════════════════════════════════════════
# STAGE 2 — PREPROCESSING
# ══════════════════════════════════════════════
def _interp_nans(arr: np.ndarray) -> np.ndarray:
    nans = np.isnan(arr)
    if nans.all():
        return arr
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def _remove_outliers(arr: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    if np.all(np.isnan(arr)):
        return arr
    m, s = np.nanmean(arr), np.nanstd(arr)
    if s == 0:
        return arr
    arr[np.abs(arr - m) / s > z_thresh] = np.nan
    return _interp_nans(arr)


def _smooth(arr: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
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
    fw = raw.get("_frame_w", 1) or 1
    fh = raw.get("_frame_h", 1) or 1

    ts: dict = {}
    for name in LM:
        frames = raw[name]
        xs = np.array([f[0] if f is not None else np.nan for f in frames], dtype=float)
        ys = np.array([f[1] if f is not None else np.nan for f in frames], dtype=float)

        xs = _interp_nans(xs); ys = _interp_nans(ys)
        xs = _remove_outliers(xs); ys = _remove_outliers(ys)
        xs = _smooth(xs); ys = _smooth(ys)

        ts[name] = {
            "x":    xs,
            "y":    ys,
            "x_px": xs * fw,
            "y_px": ys * fh,
        }
    ts["_frame_w"] = fw
    ts["_frame_h"] = fh
    return ts


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def _angle3(A, B, C) -> float:
    """Interior angle (deg) at B for points A-B-C (any units, just must be consistent)."""
    BA = np.asarray(A, dtype=float) - np.asarray(B, dtype=float)
    BC = np.asarray(C, dtype=float) - np.asarray(B, dtype=float)
    denom = (np.linalg.norm(BA) * np.linalg.norm(BC)) + 1e-9
    cos_a = np.clip(np.dot(BA, BC) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


# ══════════════════════════════════════════════
# DIRECTION SEGMENTATION  (standalone)
# ══════════════════════════════════════════════
def segment_passes(
    hip_x: np.ndarray,
    fps: float,
    min_pass_sec: float = 1.0,
    vel_stability: float = 0.15,
    turn_thresh_ratio: float = 0.20,
):
    """
    Segment a 1-D hip-midpoint x trajectory (pixels) into stable-direction passes.

      1. Smooth x with Savitzky-Golay (window ≈ fps, polyorder=3).
      2. Velocity = dx/dt.  Suppress sign during TURNING:
         |v| < turn_thresh_ratio × median|v|.
      3. Group contiguous same-sign frames; reject runs < min_pass_sec.
      4. Trim acceleration & deceleration via velocity-stability test
         (keep frames within ±vel_stability of run's median |v|).
         Fallback: trim first/last 18% of the run.

    Returns: list[ {start, end, core_start, core_end, direction} ].
    """
    n = len(hip_x)
    if n < int(fps * min_pass_sec):
        return []

    win = max(int(round(fps)) | 1, 5)
    if win >= n:
        win = (n - 1) if (n - 1) % 2 == 1 else (n - 2)
    xs = savgol_filter(hip_x, win, 3) if win >= 5 else hip_x.copy()

    velocity     = np.gradient(xs)
    abs_v        = np.abs(velocity)
    median_abs_v = np.median(abs_v) + 1e-9
    sign         = np.sign(velocity).astype(int)
    sign[abs_v < turn_thresh_ratio * median_abs_v] = 0

    passes     = []
    min_frames = max(int(fps * min_pass_sec), 5)

    i = 0
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j < n and sign[j] == sign[i]:
            j += 1
        run_len = j - i

        if run_len >= min_frames:
            seg_v       = abs_v[i:j]
            med_v       = np.median(seg_v) + 1e-9
            stable_mask = np.abs(seg_v - med_v) <= vel_stability * med_v
            stable_idx  = np.where(stable_mask)[0]

            if len(stable_idx) >= max(min_frames // 2, 5):
                core_start = i + int(stable_idx[0])
                core_end   = i + int(stable_idx[-1]) + 1
            else:
                trim       = max(int(run_len * 0.18), 1)
                core_start = i + trim
                core_end   = j - trim

            if core_end - core_start >= max(min_frames // 2, 5):
                passes.append({
                    "start":      i,
                    "end":        j,
                    "core_start": core_start,
                    "core_end":   core_end,
                    "direction":  int(sign[i]),
                })
        i = j if j > i else i + 1

    return passes


# ══════════════════════════════════════════════
# SCALE CALIBRATION  (standalone)
# ══════════════════════════════════════════════
def _stance_mask_per_leg(ankle_y_px: np.ndarray) -> np.ndarray:
    if len(ankle_y_px) < 3:
        return np.zeros_like(ankle_y_px, dtype=bool)
    vel = np.abs(np.gradient(ankle_y_px))
    return vel <= np.percentile(vel, 30)


def compute_meters_per_pixel(ts: dict, user_height_cm: float,
                             pass_segments=None, stance_frames: dict = None):
    """
    leg_length_m   = (height_cm / 100) * 0.53
    leg_length_px  = median over stance frames of euclidean(hip_px, ankle_px)
    Returns meters_per_pixel, or None if calibration could not be made.
    """
    leg_length_m = (float(user_height_cm) / 100.0) * LEG_HEIGHT_RATIO
    n = len(ts["left_hip"]["x_px"])

    if pass_segments:
        pass_mask = np.zeros(n, dtype=bool)
        for p in pass_segments:
            pass_mask[p["core_start"]:p["core_end"]] = True
    else:
        pass_mask = np.ones(n, dtype=bool)

    distances = []
    for side in ("left", "right"):
        hip_x = ts[f"{side}_hip"]["x_px"];  hip_y = ts[f"{side}_hip"]["y_px"]
        ank_x = ts[f"{side}_ankle"]["x_px"]; ank_y = ts[f"{side}_ankle"]["y_px"]
        stance = (
            stance_frames[side]
            if (stance_frames is not None and side in stance_frames)
            else _stance_mask_per_leg(ank_y)
        )
        d = np.hypot(hip_x - ank_x, hip_y - ank_y)
        sel = d[stance & pass_mask]
        if len(sel) > 5:
            distances.extend(sel.tolist())

    if not distances:
        return None
    leg_px = float(np.median(distances))
    if leg_px < 1e-6:
        return None
    return leg_length_m / leg_px


# ══════════════════════════════════════════════
# CORE PER-FRAME COMPUTATIONS
# (used by compute_metrics; computed once per video)
# ══════════════════════════════════════════════
def _detect_strikes(ts: dict, fps: float) -> dict:
    """
    Heel strikes per leg = local MAXIMA of heel y in image coords
    (largest y = lowest point on screen = foot planted).

    distance = int(fps * 0.4)  ⇒ at least 0.4s between same-leg strikes
                                  (cadence ≤ ~150 steps/min cap).
    """
    distance = max(int(fps * 0.4), 3)
    strikes  = {}
    for side in ("left", "right"):
        heel_y = ts[f"{side}_heel"]["y"]
        if len(heel_y) < distance * 2:
            strikes[side] = np.array([], dtype=int)
            continue
        rng  = float(np.ptp(heel_y))
        prom = max(rng * 0.10, 0.003)        # adaptive prominence, floor at 0.3% image-height
        peaks, _ = find_peaks(heel_y, distance=distance, prominence=prom)
        strikes[side] = peaks.astype(int)
    return strikes


def _knee_angles_px(ts: dict) -> dict:
    """
    Per-frame knee flexion (deg) using PIXEL coords.
    0° = fully extended, peak ~60-70° during swing.

    Why pixel coords: with normalized [0,1] coords the x and y axes have
    different scale (divided by W vs H), so the dot-product angle is biased.
    Using pixel coords (square pixels assumption) gives the true planar angle.
    """
    angles = {}
    for side in ("left", "right"):
        hx = ts[f"{side}_hip"]["x_px"];   hy = ts[f"{side}_hip"]["y_px"]
        kx = ts[f"{side}_knee"]["x_px"];  ky = ts[f"{side}_knee"]["y_px"]
        ax = ts[f"{side}_ankle"]["x_px"]; ay = ts[f"{side}_ankle"]["y_px"]
        n = len(kx)
        ang = np.zeros(n)
        for i in range(n):
            interior = _angle3((hx[i], hy[i]), (kx[i], ky[i]), (ax[i], ay[i]))
            ang[i] = 180.0 - interior
        angles[side] = ang
    return angles


def _torso_lean_arr(ts: dict, pass_segments=None) -> np.ndarray:
    """
    Per-frame torso lean (deg). Positive = leaning FORWARD in walking direction.

    The raw atan2 gives + when the shoulder is to the right of the hip in image
    coords. For an L→R walker that's "forward"; for R→L it's "backward". So we
    flip the sign inside R→L passes when pass_segments is provided.
    """
    sh_x = (ts["left_shoulder"]["x"] + ts["right_shoulder"]["x"]) / 2.0
    sh_y = (ts["left_shoulder"]["y"] + ts["right_shoulder"]["y"]) / 2.0
    hp_x = (ts["left_hip"]["x"]      + ts["right_hip"]["x"])      / 2.0
    hp_y = (ts["left_hip"]["y"]      + ts["right_hip"]["y"])      / 2.0
    dx = sh_x - hp_x
    dy = -(sh_y - hp_y)
    angles = np.degrees(np.arctan2(dx, dy))

    if pass_segments:
        for p in pass_segments:
            if p["direction"] < 0:
                angles[p["start"]:p["end"]] = -angles[p["start"]:p["end"]]
    return angles


# ══════════════════════════════════════════════
# STANDALONE — METRIC COMPUTATION OVER A FRAME SUBSET
# ══════════════════════════════════════════════
def compute_metrics(ts: dict, frame_indices, mpp, fps: float,
                    pass_segments=None, strikes=None, knee_full=None,
                    torso_full=None) -> dict:
    """
    Compute the full gait-metric dict over ONLY the given frame_indices.

    Args:
        ts             : time-series dict from build_time_series().
        frame_indices  : list/array of integer frame numbers to include
                         (use np.arange(total_frames) for "Total" mode,
                          union of pass cores for "Clean" mode).
        mpp            : meters_per_pixel (None → step length stays in pixels).
        fps            : video frame rate.
        pass_segments  : optional; needed for direction-aware torso-lean sign flip.
        strikes/knee_full/torso_full : optional pre-computed, to avoid
                         recomputing per-frame arrays for the second call.

    Returns: complete metric dict (same shape consumed by app.py / plots).
    """
    n_total = len(ts["left_heel"]["y"])

    mask = np.zeros(n_total, dtype=bool)
    if frame_indices is not None and len(frame_indices) > 0:
        idx = np.asarray(frame_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_total)]
        mask[idx] = True

    n_mask  = int(mask.sum())
    dur_sec = n_mask / fps if fps > 0 else 0.0

    # --- per-frame caches (compute once, reuse for Total + Clean) ---
    if strikes    is None: strikes    = _detect_strikes(ts, fps)
    if knee_full  is None: knee_full  = _knee_angles_px(ts)
    if torso_full is None: torso_full = _torso_lean_arr(ts, pass_segments)

    # --- step indices restricted to mask ---
    def _filter(idx):
        if len(idx) == 0:
            return idx
        return idx[mask[idx]]

    L_idx    = _filter(strikes["left"])
    R_idx    = _filter(strikes["right"])
    combined = np.sort(np.concatenate([L_idx, R_idx])) if (len(L_idx) or len(R_idx)) else np.array([], dtype=int)

    # Each detected heel strike = 1 step
    total_steps = len(combined)
    step_data = {
        "total_steps":      total_steps,
        "left_count":       int(len(L_idx)),
        "right_count":      int(len(R_idx)),
        "left_indices":     L_idx,
        "right_indices":    R_idx,
        "combined_indices": combined,
    }

    # --- cadence ---
    cadence = round((total_steps / dur_sec) * 60.0, 1) if dur_sec > 0 else 0.0

    # --- knee angles (mean / peak / min computed only over masked frames) ---
    knee_angles = {}
    for side in ("left", "right"):
        full = knee_full[side]
        m_arr = full[mask] if n_mask > 0 else np.array([])
        knee_angles[side]            = full
        knee_angles[f"{side}_mean"]  = float(np.nanmean(m_arr)) if len(m_arr) else 0.0
        knee_angles[f"peak_{side}"]  = float(np.nanmax(m_arr))  if len(m_arr) else 0.0
        knee_angles[f"min_{side}"]   = float(np.nanmin(m_arr))  if len(m_arr) else 0.0
    knee_angles["overall_mean"] = float(np.nanmean([knee_angles["left_mean"],  knee_angles["right_mean"]]))
    knee_angles["overall_peak"] = float(max(knee_angles["peak_left"],         knee_angles["peak_right"]))
    knee_angles["overall_min"]  = float(min(knee_angles["min_left"],          knee_angles["min_right"]))

    # --- step / stride times (per leg = same-leg strike intervals) ---
    # Important: when multiple passes are involved we compute intervals
    # WITHIN each pass core only — concatenating strikes across passes would
    # treat the inter-pass turning gap as a stride and inflate stride-CV.
    step_timing = {"left": np.array([]), "right": np.array([])}
    all_st = []
    if pass_segments and n_mask < n_total:
        for p in pass_segments:
            a, b = p["core_start"], p["core_end"]
            for side in ("left", "right"):
                in_pass = strikes[side][(strikes[side] >= a) & (strikes[side] < b)]
                if len(in_pass) >= 2:
                    ints = np.diff(in_pass) / fps
                    step_timing[side] = np.concatenate([step_timing[side], ints])
                    all_st.extend(ints.tolist())
    else:
        for side in ("left", "right"):
            idx = step_data[f"{side}_indices"]
            if len(idx) >= 2:
                ints = np.diff(idx) / fps
                step_timing[side] = ints
                all_st.extend(ints.tolist())
    step_timing["mean_step_time"] = float(np.mean(all_st)) if all_st else 0.0

    # --- symmetry (step-time means) ---
    L_t, R_t = step_timing["left"], step_timing["right"]
    if len(L_t) and len(R_t):
        Lm, Rm = float(np.mean(L_t)), float(np.mean(R_t))
        denom  = (Lm + Rm) / 2.0
        sym    = 1.0 - abs(Lm - Rm) / denom if denom > 1e-9 else 1.0
        sym    = float(np.clip(sym, 0.0, 1.0))
    else:
        sym = 1.0

    # --- stride CV (PERCENT) ---
    if len(all_st) >= 2 and np.mean(all_st) > 1e-9:
        cv_pct = round(float(np.std(all_st) / np.mean(all_st)) * 100.0, 2)
    else:
        cv_pct = 0.0

    # --- step length (distance the body advances per step) ---
    # Biomechanical definition: step length = horizontal distance between two
    # CONSECUTIVE heel strikes (one foot, then the other).  Measuring
    # |L_heel - R_heel| AT a single strike is wrong — the swinging leg is
    # mid-flight, not at its full back-position, so the separation is roughly
    # half of the true step length.
    l_heel_px = ts["left_heel"]["x_px"]
    r_heel_px = ts["right_heel"]["x_px"]
    # Build chronological list of (frame, leg_label, x_at_planted_heel).
    events = [(int(f), "L", float(l_heel_px[f])) for f in L_idx] + \
             [(int(f), "R", float(r_heel_px[f])) for f in R_idx]
    events.sort(key=lambda e: e[0])
    sl_px_list = []
    for i in range(1, len(events)):
        f0, leg0, x0 = events[i - 1]
        f1, leg1, x1 = events[i]
        if leg0 == leg1:
            continue                              # same-leg duplicate; skip
        # Reject pathologically long gaps (e.g. across a turn that escaped pass trim).
        # Adult step time is ~0.5s; >2× is almost certainly a pass-boundary artifact.
        if (f1 - f0) / fps > 1.2:
            continue
        sl_px_list.append(abs(x1 - x0))
    sep_px = np.array(sl_px_list, dtype=float)
    if mpp and mpp > 0:
        sl_vals, sl_unit = sep_px * mpp, "m"
    else:
        sl_vals, sl_unit = sep_px, "px"
    step_length = {
        "values": sl_vals,
        "mean":   float(np.nanmean(sl_vals)) if len(sl_vals) else 0.0,
        "std":    float(np.nanstd(sl_vals))  if len(sl_vals) else 0.0,
        "unit":   sl_unit,
    }

    # --- torso lean (mean / std over masked frames) ---
    t_arr = torso_full[mask] if n_mask > 0 else torso_full
    torso_lean = {
        "angles": torso_full,
        "mean":   float(np.nanmean(t_arr)) if len(t_arr) else 0.0,
        "std":    float(np.nanstd(t_arr))  if len(t_arr) else 0.0,
    }

    # --- ankle trajectory (full series; for plotting only) ---
    ankle_traj = {
        "left_x":  ts["left_ankle"]["x"],
        "right_x": ts["right_ankle"]["x"],
        "left_y":  ts["left_ankle"]["y"],
        "right_y": ts["right_ankle"]["y"],
    }

    return {
        "step_data":        step_data,
        "cadence":          cadence,
        "knee_angles":      knee_angles,
        "step_timing":      step_timing,
        "symmetry":         round(sym, 3),
        "step_length":      step_length,
        "stride_cv":        cv_pct,
        "ankle_trajectory": ankle_traj,
        "torso_lean":       torso_lean,
        "duration_sec":     dur_sec,
        "n_frames":         n_mask,
    }


# ══════════════════════════════════════════════
# DIRECTION SUMMARY STRING
# ══════════════════════════════════════════════
def compute_walking_direction(pass_segments) -> str:
    if not pass_segments:
        return "Unknown"
    n_lr = sum(1 for p in pass_segments if p["direction"] > 0)
    n_rl = sum(1 for p in pass_segments if p["direction"] < 0)
    if n_rl == 0:
        return "Left → Right"
    if n_lr == 0:
        return "Right → Left"
    return f"Bidirectional ({n_lr} L→R, {n_rl} R→L)"


# ══════════════════════════════════════════════
# MASTER COMPUTE  (Total + Clean)
# ══════════════════════════════════════════════
def compute_all_features(ts: dict, fps: float, total_frames: int,
                         user_height_cm: float = DEFAULT_HEIGHT_CM) -> dict:
    """
    Run the full pipeline: detect passes, calibrate scale, compute
    Total metrics (whole video, informational) AND Clean metrics
    (steady-state only, used by the rule engine).

    Top-level keys mirror clean_metrics so existing app.py / gait_plots.py
    code keeps working without changes.
    """
    n = len(ts["left_heel"]["y"])

    # 1. Pass segmentation from hip-midpoint (NOT shoulders/nose, which oscillate).
    hip_x_px      = (ts["left_hip"]["x_px"] + ts["right_hip"]["x_px"]) / 2.0
    pass_segments = segment_passes(hip_x_px, fps)

    # 2. Anatomical scale.
    mpp = compute_meters_per_pixel(ts, user_height_cm, pass_segments=pass_segments)

    # 3. Pre-compute per-frame caches (shared by Total & Clean).
    strikes    = _detect_strikes(ts, fps)
    knee_full  = _knee_angles_px(ts)
    torso_full = _torso_lean_arr(ts, pass_segments)

    # 4. TOTAL metrics — every frame.
    total_indices = np.arange(n)
    total_metrics = compute_metrics(
        ts, total_indices, mpp, fps,
        pass_segments=pass_segments,
        strikes=strikes, knee_full=knee_full, torso_full=torso_full,
    )

    # 5. CLEAN metrics — only frames inside any pass core (steady state).
    if pass_segments:
        clean_mask = np.zeros(n, dtype=bool)
        for p in pass_segments:
            clean_mask[p["core_start"]:p["core_end"]] = True
        clean_indices = np.where(clean_mask)[0]
    else:
        clean_indices = total_indices       # graceful fallback

    clean_metrics = compute_metrics(
        ts, clean_indices, mpp, fps,
        pass_segments=pass_segments,
        strikes=strikes, knee_full=knee_full, torso_full=torso_full,
    )

    direction = compute_walking_direction(pass_segments)

    # Top-level = clean (drives plots and inference).
    return {
        # ── original top-level keys (= clean) ─────────────
        **clean_metrics,
        "direction":              direction,
        "fps":                    fps,
        "total_frames":           total_frames,
        "duration_sec":           total_frames / fps if fps > 0 else 0,

        # ── new dual + audit keys ─────────────────────────
        "total_metrics":          total_metrics,
        "clean_metrics":          clean_metrics,
        "pass_segments":          pass_segments,
        "num_passes":             len(pass_segments),
        "frames_used":            clean_metrics["n_frames"],
        "steady_state_duration_s": clean_metrics["duration_sec"],
        "meters_per_pixel":       mpp,
        "user_height_cm":         user_height_cm,
    }


# ══════════════════════════════════════════════
# RULE-BASED INFERENCE  (consumes ONLY clean_metrics)
# ══════════════════════════════════════════════
def interpret(features: dict) -> dict:
    """All thresholds compare against clean_metrics — never total."""
    clean = features.get("clean_metrics", features)

    cadence       = clean["cadence"]
    symmetry      = clean["symmetry"]
    knee_peak     = clean["knee_angles"].get("overall_peak", 0.0)
    knee_min      = clean["knee_angles"].get("overall_min", 0.0)
    knee_overall  = clean["knee_angles"]["overall_mean"]
    stride_cv_pct = clean["stride_cv"]
    step_len_mean = clean["step_length"]["mean"]
    step_len_unit = clean["step_length"].get("unit", "m")
    torso_mean    = clean["torso_lean"]["mean"]
    direction     = features.get("direction", "Unknown")
    num_passes    = features.get("num_passes", 0)
    frames_used   = features.get("frames_used", 0)
    total_dur     = features.get("duration_sec", 0)
    clean_dur     = features.get("steady_state_duration_s", 0)
    step_count    = clean["step_data"]["total_steps"]

    obs, sug = [], []

    # ── Cadence (target 100–120) ─────────────────────────
    if cadence == 0:
        obs.append(" No clean walking detected — no validated passes available.")
        sug.append("Record a longer side-view clip with at least one continuous direction-stable walk of ≥1 s.")
    elif cadence < 100:
        obs.append(f" Cadence is low ({cadence} steps/min). Adult normal range is 100–120.")
        sug.append("Focus on quicker, shorter steps to bring cadence into the normal range.")
    elif cadence > 120:
        obs.append(f" Cadence is high ({cadence} steps/min). Normal range is 100–120; may indicate brisk walking or jogging.")
    else:
        obs.append(f" Cadence is in the adult normal range ({cadence} steps/min).")

    # ── Step length (target 0.6–0.8 m) ───────────────────
    if step_len_mean > 0:
        if step_len_unit == "m":
            if step_len_mean < 0.55:
                obs.append(f" Step length is short ({step_len_mean:.2f} m). Adult normal is 0.6–0.8 m.")
                sug.append("Lengthen stride with hip-extensor strengthening and walking drills.")
            elif step_len_mean > 0.85:
                obs.append(f" Step length is long ({step_len_mean:.2f} m). Possible overstriding.")
                sug.append("Reduce overstride to lower heel-strike impact.")
            else:
                obs.append(f" Step length is in adult normal range ({step_len_mean:.2f} m).")
        else:
            obs.append(f" Step length: {step_len_mean:.1f} px (real-world scale unavailable).")

    # ── Symmetry (target > 95%) ──────────────────────────
    sym_pct = symmetry * 100
    if sym_pct < 85:
        obs.append(f" Significant gait asymmetry ({sym_pct:.1f}%). Left and right step rhythm differ markedly.")
        sug.append("Consult a physical therapist to evaluate possible limb-length discrepancy or muscle imbalance.")
    elif sym_pct < 95:
        obs.append(f" Mild gait asymmetry ({sym_pct:.1f}%). Adult target is > 95%.")
        sug.append("Add single-leg balance work to improve bilateral coordination.")
    else:
        obs.append(f" Gait symmetry is excellent ({sym_pct:.1f}%).")

    # ── Stride CV (target < 3 %) ─────────────────────────
    if stride_cv_pct > 6:
        obs.append(f" High stride-time variability (CV = {stride_cv_pct:.1f} %). Adult target is < 3 %.")
        sug.append("Practice metronome-paced walking to improve rhythm consistency.")
    elif stride_cv_pct > 3:
        obs.append(f" Moderate stride-time variability (CV = {stride_cv_pct:.1f} %).")
        sug.append("Focus on rhythmic walking to reduce stride-to-stride variation.")
    else:
        obs.append(f" Stride consistency is good (CV = {stride_cv_pct:.1f} %).")

    # ── Knee flexion (target peak 60-70°) ────────────────
    if knee_min > 12:
        obs.append(f" Knee lacks full extension at stance (min flexion {knee_min:.1f}°). Normal is ~0°.")
        sug.append("Hamstring and calf stretching to allow full knee extension.")
    if knee_peak < 50:
        obs.append(f" Restricted swing-phase knee flexion (peak {knee_peak:.1f}°). Normal is 60–70°.")
        sug.append("Knee mobility drills or recumbent biking to expand flexion range.")
    elif knee_peak > 80:
        obs.append(f" Unusually high knee flexion peak ({knee_peak:.1f}°) — could indicate jogging.")
    else:
        obs.append(f" Swing-phase knee flexion peak is in normal range ({knee_peak:.1f}°).")

    # ── Torso lean (target |lean| < 5°) ──────────────────
    abs_lean = abs(torso_mean)
    if abs_lean > 10:
        d = "forward" if torso_mean > 0 else "backward"
        obs.append(f" Significant torso lean {d} ({torso_mean:.1f}°).")
        sug.append("Practice upright posture walking — engage core, keep gaze forward.")
    elif abs_lean > 5:
        obs.append(f" Mild torso lean ({torso_mean:.1f}°).")
        sug.append("Strengthen core (planks, bird-dogs) to improve trunk stability.")
    else:
        obs.append(f" Torso is well-aligned ({torso_mean:.1f}° lean).")

    # ── Coverage / direction ─────────────────────────────
    obs.append(
        f" Walking direction: {direction} — "
        f"{num_passes} clean pass{'es' if num_passes != 1 else ''} "
        f"({clean_dur:.1f} s steady-state of {total_dur:.1f} s total)."
    )
    obs.append(f"⏱ {step_count} clean strikes used for assessment.")

    return {"observations": obs, "suggestions": sug}
