"""tandem_walk_engine.py — backend Tandem Walk (E1) pipeline on the
MediaPipe BlazePose Full (33-keypoint) stack.

Mirrors the frontend math in
motionlens-web/lib/orthopedic/tandemWalk.ts so live + upload return
identical numbers.

Pipeline:
  1. Reuse gait_engine.extract_poses() + build_time_series() for the
     smoothed 33-kp landmarks per frame.
  2. Visibility gate — ≥ 30 % of frames must have both shoulders +
     both hips + both ankles tracked.
  3. Sub-sample to SAMPLE_HZ (matches the frontend live cadence) so
     velocity thresholds expressed in px/sample are comparable.
  4. Per-foot velocity-threshold state machine fires footstrike events
     on swinging → planted transitions. Same algorithm as the
     frontend `detectFootstrike` — central-difference velocity over
     3 sub-sampled frames, MIN_PLANTED_FRAMES consecutive low-vel
     frames to confirm, MIN_FRAMES_BETWEEN_STRIKES gate to debounce.
  5. Least-squares fit of x = a·y + b through the hip-midpoint
     samples → the walking-line reference.
  6. For each footstrike: perpendicular pixel distance to the line,
     normalised by shoulder width × ASSUMED_SHOULDER_WIDTH_CM → cm.
  7. Aggregate: misstep count, mean / max deviation, step-time CV,
     arm-grab count, trunk-sway range.
  8. Classification mirrors classifyTandemWalk().
"""
from __future__ import annotations

import base64
import logging
import math
from typing import Optional

import cv2

from engines.gait_engine import (
    LM,
    apply_rotation as _apply_rot,
    build_time_series,
    extract_poses,
)

log = logging.getLogger(__name__)


# ─── Spec constants (mirror lib/orthopedic/tandemWalk.ts) ───────
_VIS_THRESHOLD = 0.15
_SAMPLE_HZ = 10
_TARGET_STEP_COUNT = 10
# Step-detector tuning — see lib/orthopedic/tandemWalk.ts for the
# rationale. Threshold sits well above MediaPipe per-frame jitter so
# noise can't oscillate the state machine; debounce + minimum swing
# displacement filter the few jitter spikes that get through.
_PLANTED_VELOCITY_PX_PER_SEC = 100.0
_MIN_PLANTED_FRAMES = 2
_MIN_FRAMES_BETWEEN_STRIKES = 10
_MIN_SWING_DISPLACEMENT_RATIO = 0.15
_ASSUMED_SHOULDER_WIDTH_CM = 40.0
# Effective deviation = max(0, raw - tolerance). 5 cm ≈ 12 % of the
# 40 cm shoulder-width scale — covers MediaPipe jitter + natural hip
# sway + ankle-keypoint offset above the floor contact point.
_DEVIATION_TOLERANCE_CM = 5.0
_MISSTEP_DEVIATION_CM = 6.0
_POSITIVE_SCREEN_MISSTEP_COUNT = 2
_ABNORMAL_MEAN_DEVIATION_CM = 3.0
_BORDERLINE_MEAN_DEVIATION_CM = 1.5
_ARM_ABDUCTION_DEG = 45.0
_ARM_GRAB_DEDUPE_FRAMES = 5


# ─── Per-frame helpers ──────────────────────────────────────────
def _visible(ts: dict, key: str, i: int) -> bool:
    return float(ts[key]["vis"][i]) >= _VIS_THRESHOLD


def _all_visible(ts: dict, keys: tuple[str, ...], i: int) -> bool:
    return all(_visible(ts, k, i) for k in keys)


def _hip_mid_x(ts: dict, i: int) -> Optional[float]:
    if not _all_visible(ts, ("left_hip", "right_hip"), i):
        return None
    return (
        float(ts["left_hip"]["x_px"][i])
        + float(ts["right_hip"]["x_px"][i])
    ) / 2.0


def _hip_mid_y(ts: dict, i: int) -> Optional[float]:
    if not _all_visible(ts, ("left_hip", "right_hip"), i):
        return None
    return (
        float(ts["left_hip"]["y_px"][i])
        + float(ts["right_hip"]["y_px"][i])
    ) / 2.0


def _shoulder_mid_x(ts: dict, i: int) -> Optional[float]:
    if not _all_visible(ts, ("left_shoulder", "right_shoulder"), i):
        return None
    return (
        float(ts["left_shoulder"]["x_px"][i])
        + float(ts["right_shoulder"]["x_px"][i])
    ) / 2.0


def _shoulder_width(ts: dict, i: int) -> Optional[float]:
    if not _all_visible(ts, ("left_shoulder", "right_shoulder"), i):
        return None
    lx = float(ts["left_shoulder"]["x_px"][i])
    ly = float(ts["left_shoulder"]["y_px"][i])
    rx = float(ts["right_shoulder"]["x_px"][i])
    ry = float(ts["right_shoulder"]["y_px"][i])
    return math.hypot(lx - rx, ly - ry)


def _foot_pos(ts: dict, i: int, side: str) -> Optional[tuple[float, float]]:
    """Test-side ankle, falling back to foot_index when the ankle is
    occluded (often happens when the back foot's heel is hidden by the
    front foot's shin in tight tandem stance)."""
    ankle_key = f"{side}_ankle"
    foot_key = f"{side}_foot_index"
    if _visible(ts, ankle_key, i):
        return (
            float(ts[ankle_key]["x_px"][i]),
            float(ts[ankle_key]["y_px"][i]),
        )
    if _visible(ts, foot_key, i):
        return (
            float(ts[foot_key]["x_px"][i]),
            float(ts[foot_key]["y_px"][i]),
        )
    return None


def _arm_abduction_deg(ts: dict, i: int, side: str) -> Optional[float]:
    sh_key = f"{side}_shoulder"
    wr_key = f"{side}_wrist"
    hp_key = f"{side}_hip"
    if not _all_visible(ts, (sh_key, wr_key, hp_key), i):
        return None
    sx = float(ts[sh_key]["x_px"][i]); sy = float(ts[sh_key]["y_px"][i])
    wx = float(ts[wr_key]["x_px"][i]); wy = float(ts[wr_key]["y_px"][i])
    hx = float(ts[hp_key]["x_px"][i]); hy = float(ts[hp_key]["y_px"][i])
    ax = wx - sx; ay = wy - sy
    bx = hx - sx; by = hy - sy
    ma = math.hypot(ax, ay); mb = math.hypot(bx, by)
    if ma < 1e-6 or mb < 1e-6:
        return None
    cos = (ax * bx + ay * by) / (ma * mb)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


# ─── Per-foot step detector ─────────────────────────────────────
class _FootState:
    __slots__ = (
        "prev_y", "prev_prev_y", "prev_prev_prev_y",
        "low_vel_frames", "state", "last_strike_frame",
        "swing_start_y", "swing_max_displacement_px",
    )

    def __init__(self) -> None:
        self.prev_y: Optional[float] = None
        self.prev_prev_y: Optional[float] = None
        self.prev_prev_prev_y: Optional[float] = None
        self.low_vel_frames = 0
        self.state = "init"
        self.last_strike_frame = -_MIN_FRAMES_BETWEEN_STRIKES
        self.swing_start_y: Optional[float] = None
        self.swing_max_displacement_px = 0.0


def _detect_footstrike(
    state: _FootState,
    foot_y: Optional[float],
    frame_idx: int,
    velocity_threshold_per_frame: float,
    min_swing_displacement_px: float,
) -> bool:
    if foot_y is None:
        state.prev_prev_prev_y = state.prev_prev_y
        state.prev_prev_y = state.prev_y
        state.prev_y = None
        state.low_vel_frames = 0
        return False

    vel_abs = math.inf
    if state.prev_prev_y is not None and state.prev_prev_prev_y is not None:
        vel_abs = abs((foot_y - state.prev_prev_prev_y) / 3.0)

    strike = False
    moving = vel_abs > velocity_threshold_per_frame
    if moving:
        state.low_vel_frames = 0
        if state.state != "swinging":
            # Entered swing — anchor the start position so we can
            # require a real forward sweep before the next plant
            # counts as a footstrike.
            state.state = "swinging"
            state.swing_start_y = foot_y
            state.swing_max_displacement_px = 0.0
        elif state.swing_start_y is not None:
            disp = abs(foot_y - state.swing_start_y)
            if disp > state.swing_max_displacement_px:
                state.swing_max_displacement_px = disp
    else:
        state.low_vel_frames += 1
        if (
            state.state == "swinging"
            and state.low_vel_frames >= _MIN_PLANTED_FRAMES
            and frame_idx - state.last_strike_frame >= _MIN_FRAMES_BETWEEN_STRIKES
            and state.swing_max_displacement_px >= min_swing_displacement_px
        ):
            state.state = "planted"
            state.last_strike_frame = frame_idx
            state.swing_start_y = None
            state.swing_max_displacement_px = 0.0
            strike = True

    state.prev_prev_prev_y = state.prev_prev_y
    state.prev_prev_y = state.prev_y
    state.prev_y = foot_y
    return strike


# ─── Line fit + perpendicular distance ──────────────────────────
def _fit_walking_line(
    points: list[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    """Least-squares fit  x = a·y + b. Returns (a, b) or None."""
    if len(points) < 3:
        return None
    n = len(points)
    sum_y = sum(p[1] for p in points)
    sum_x = sum(p[0] for p in points)
    sum_yy = sum(p[1] * p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    denom = n * sum_yy - sum_y * sum_y
    if abs(denom) < 1e-9:
        return None
    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_x - a * sum_y) / n
    return (a, b)


def _perp_distance_px(
    px: float, py: float, a: float, b: float,
) -> float:
    return abs(px - (a * py + b)) / math.sqrt(1.0 + a * a)


# ─── Classification ─────────────────────────────────────────────
def _classify(misstep_count: int, mean_deviation_cm: float) -> str:
    if (
        misstep_count >= _POSITIVE_SCREEN_MISSTEP_COUNT
        or mean_deviation_cm > _ABNORMAL_MEAN_DEVIATION_CM
    ):
        return "positive_screen"
    if misstep_count == 1 or mean_deviation_cm > _BORDERLINE_MEAN_DEVIATION_CM:
        return "borderline"
    return "normal"


def _build_interpretation(result: dict) -> str:
    rep_count = len(result["steps"])
    target = _TARGET_STEP_COUNT
    incomplete = rep_count < target
    summary = (
        f"{rep_count} of {target} steps captured"
        if incomplete else
        f"{rep_count} steps captured"
    )
    parts: list[str] = []
    parts.append(
        f"{summary}. Mean lateral deviation "
        f"{result['mean_deviation_cm']:.1f} cm, worst step "
        f"{result['max_deviation_cm']:.1f} cm."
    )
    misstep = int(result["misstep_count"])
    if misstep >= _POSITIVE_SCREEN_MISSTEP_COUNT:
        parts.append(
            f"{misstep} missteps detected "
            f"(>= {_POSITIVE_SCREEN_MISSTEP_COUNT} = positive screen for "
            f"cerebellar / vestibular dysfunction)."
        )
    elif misstep == 1:
        parts.append("1 misstep detected — borderline.")
    else:
        parts.append("No missteps detected.")
    if result["mean_deviation_cm"] > _ABNORMAL_MEAN_DEVIATION_CM:
        parts.append(
            f"Mean foot placement was abnormal "
            f"(> {_ABNORMAL_MEAN_DEVIATION_CM:.0f} cm from the walking line)."
        )
    if int(result["arm_grab_count"]) > 0:
        parts.append(
            f"{int(result['arm_grab_count'])} arm-grab event(s) — "
            f"patient threw an arm out (> {_ARM_ABDUCTION_DEG:.0f} deg "
            f"abduction) for balance."
        )
    if result["step_time_cv"] > 0.25:
        parts.append(
            f"Step-time variability is high (CV = {result['step_time_cv']:.2f}) "
            f"— irregular cadence, often seen with cerebellar involvement."
        )
    if result["trunk_sway_range_cm"] > 8:
        parts.append(
            f"Trunk sway range {result['trunk_sway_range_cm']:.1f} cm — "
            f"broad lateral excursion suggests proximal instability."
        )
    return " ".join(parts)


# ─── Screenshot ────────────────────────────────────────────────
def _grab_capture_frame(
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

    def _kp_pixel(name: str) -> Optional[tuple[int, int]]:
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
        p = _kp_pixel(name)
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
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


# ─── Main entry point ──────────────────────────────────────────
def analyze_tandem_walk(
    video_path: str,
    pose_options,
    patient_age: Optional[int] = None,
) -> dict:
    """Run the E1 Tandem Walk pipeline on an uploaded clip.

    Args:
        video_path:   path to the uploaded video file on disk.
        pose_options: PoseLandmarkerOptions built by
                      api._build_gait_pose_options().
        patient_age:  optional context metadata, recorded verbatim
                      in the result.

    Returns:
        Dict matching the frontend TandemWalkResult shape plus a few
        diagnostic extras. The frontend ignores unknown keys.

    Raises:
        ValueError: 'poor_visibility' when the patient isn't clearly
                    visible in enough frames, or 'no_steps' when no
                    footstrikes are detected.
    """
    raw, fps, _cv_total_frames = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    n_full = int(min(
        len(ts["left_hip"]["y"]),
        len(ts["right_hip"]["y"]),
        len(ts["left_ankle"]["y"]),
        len(ts["right_ankle"]["y"]),
        len(ts["left_shoulder"]["y"]),
        len(ts["right_shoulder"]["y"]),
        len(ts["left_knee"]["y"]),
        len(ts["right_knee"]["y"]),
    ))
    if n_full == 0 or fps <= 0:
        raise ValueError("poor_visibility")

    # ── Visibility gate ─────────────────────────────────────
    required_keys = (
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_ankle", "right_ankle",
    )
    visible_frames = sum(
        1 for i in range(n_full) if _all_visible(ts, required_keys, i)
    )
    if visible_frames < max(3, int(n_full * 0.30)):
        raise ValueError("poor_visibility")

    # ── Sub-sample to SAMPLE_HZ ─────────────────────────────
    step = max(1, int(round(fps / _SAMPLE_HZ)))
    sampled_full_indices = list(range(0, n_full, step))
    sampled_count = len(sampled_full_indices)

    # Per-frame metrics on the sampled grid.
    samples: list[dict] = []
    keypoints_export: list[list[dict]] = []
    for full_i in sampled_full_indices:
        t_ms = (full_i / fps) * 1000.0 if fps > 0 else 0.0
        l = _foot_pos(ts, full_i, "left")
        r = _foot_pos(ts, full_i, "right")
        samples.append({
            "t_ms": float(t_ms),
            "hip_mid_x":       _hip_mid_x(ts, full_i),
            "hip_mid_y":       _hip_mid_y(ts, full_i),
            "shoulder_mid_x":  _shoulder_mid_x(ts, full_i),
            "shoulder_width_px": _shoulder_width(ts, full_i),
            "left_foot_x":     l[0] if l is not None else None,
            "left_foot_y":     l[1] if l is not None else None,
            "right_foot_x":    r[0] if r is not None else None,
            "right_foot_y":    r[1] if r is not None else None,
            "left_arm_abduction_deg":  _arm_abduction_deg(ts, full_i, "left"),
            "right_arm_abduction_deg": _arm_abduction_deg(ts, full_i, "right"),
        })
        kp_frame: list[dict] = [
            {"x": 0.0, "y": 0.0, "score": 0.0} for _ in range(33)
        ]
        for name, idx_n in LM.items():
            frames = raw.get(name, [])
            if full_i < len(frames) and frames[full_i] is not None:
                x_n, y_n, vis = frames[full_i]
                kp_frame[idx_n] = {
                    "x": float(x_n) * float(raw.get("_frame_w", 1) or 1),
                    "y": float(y_n) * float(raw.get("_frame_h", 1) or 1),
                    "score": float(vis),
                }
        keypoints_export.append(kp_frame)

    # ── Per-foot step detection on the sampled grid ─────────
    # Velocity threshold + minimum swing displacement together
    # filter out MediaPipe per-frame jitter; without the displacement
    # check, a one-frame velocity spike on a stationary foot would
    # still register as a swing→plant cycle.
    velocity_thresh_per_frame = _PLANTED_VELOCITY_PX_PER_SEC / _SAMPLE_HZ
    left_state = _FootState()
    right_state = _FootState()
    raw_strikes: list[dict] = []
    for sample_idx, sm in enumerate(samples):
        sw = sm["shoulder_width_px"]
        min_swing_disp_px = (
            float(sw) * _MIN_SWING_DISPLACEMENT_RATIO
            if sw is not None and sw > 0 else 20.0
        )
        lf_strike = _detect_footstrike(
            left_state,
            sm["left_foot_y"],
            sample_idx,
            velocity_thresh_per_frame,
            min_swing_disp_px,
        )
        if lf_strike:
            raw_strikes.append({"side": "left", "sample_index": sample_idx})
        rf_strike = _detect_footstrike(
            right_state,
            sm["right_foot_y"],
            sample_idx,
            velocity_thresh_per_frame,
            min_swing_disp_px,
        )
        if rf_strike:
            raw_strikes.append({"side": "right", "sample_index": sample_idx})
        if len(raw_strikes) >= _TARGET_STEP_COUNT:
            break

    if not raw_strikes:
        raise ValueError(
            "no_steps: no footstrikes detected. Please re-record the patient "
            "performing a 10-step heel-to-toe walk toward the camera."
        )

    # ── Fit walking line through hip-midpoint samples ───────
    hip_mid_points: list[tuple[float, float]] = []
    for sm in samples:
        if sm["hip_mid_x"] is not None and sm["hip_mid_y"] is not None:
            hip_mid_points.append((float(sm["hip_mid_x"]), float(sm["hip_mid_y"])))
    line = _fit_walking_line(hip_mid_points)

    # ── Mean shoulder width across the trial ────────────────
    sw_vals = [
        float(sm["shoulder_width_px"])
        for sm in samples
        if sm["shoulder_width_px"] is not None and sm["shoulder_width_px"] > 0
    ]
    mean_shoulder_px = sum(sw_vals) / len(sw_vals) if sw_vals else 0.0

    # ── Build per-step records with deviation in px and cm ─
    steps: list[dict] = []
    for i, rs in enumerate(raw_strikes[:_TARGET_STEP_COUNT]):
        sm = samples[rs["sample_index"]]
        fx = sm["left_foot_x"] if rs["side"] == "left" else sm["right_foot_x"]
        fy = sm["left_foot_y"] if rs["side"] == "left" else sm["right_foot_y"]
        if fx is None or fy is None:
            continue
        dev_px = _perp_distance_px(float(fx), float(fy), line[0], line[1]) if line else None
        sw = sm["shoulder_width_px"]
        raw_dev_cm = (
            (dev_px / sw) * _ASSUMED_SHOULDER_WIDTH_CM
            if (dev_px is not None and sw is not None and sw > 0)
            else None
        )
        # Subtract the deviation tolerance so natural wobble + pose
        # noise does not register as deviation. Effective value is
        # what is_misstep + classification + aggregates see.
        effective_dev_cm = (
            max(0.0, raw_dev_cm - _DEVIATION_TOLERANCE_CM)
            if raw_dev_cm is not None else None
        )
        is_misstep = (
            effective_dev_cm is not None
            and effective_dev_cm > _MISSTEP_DEVIATION_CM
        )
        steps.append({
            "step_index": i + 1,
            "side": rs["side"],
            "t_ms": float(sm["t_ms"]),
            "foot_x": float(fx),
            "foot_y": float(fy),
            "deviation_px": float(dev_px) if dev_px is not None else None,
            "raw_deviation_cm":
                float(raw_dev_cm) if raw_dev_cm is not None else None,
            "deviation_cm":
                float(effective_dev_cm) if effective_dev_cm is not None else None,
            "is_misstep": bool(is_misstep),
            "shoulder_width_px": float(sw) if sw is not None else None,
        })

    # ── Step-time statistics ────────────────────────────────
    step_times = [
        steps[i]["t_ms"] - steps[i - 1]["t_ms"]
        for i in range(1, len(steps))
    ]
    if step_times:
        st_mean = sum(step_times) / len(step_times)
        st_var = sum((x - st_mean) ** 2 for x in step_times) / len(step_times)
        st_std = math.sqrt(st_var)
        st_cv = st_std / st_mean if st_mean > 0 else 0.0
    else:
        st_mean = st_std = st_cv = 0.0

    # ── Arm-grab events (discrete, debounced) ──────────────
    arm_grab_count = 0
    in_segment = False
    last_grab_idx = -_ARM_GRAB_DEDUPE_FRAMES
    for i, sm in enumerate(samples):
        l = sm["left_arm_abduction_deg"]
        r = sm["right_arm_abduction_deg"]
        abducted = (
            (l is not None and l > _ARM_ABDUCTION_DEG)
            or (r is not None and r > _ARM_ABDUCTION_DEG)
        )
        if abducted:
            if not in_segment and i - last_grab_idx >= _ARM_GRAB_DEDUPE_FRAMES:
                arm_grab_count += 1
                in_segment = True
                last_grab_idx = i
            elif in_segment:
                last_grab_idx = i
        else:
            in_segment = False

    # ── Trunk sway (residual of shoulder-mid x from line) ──
    residuals: list[float] = []
    for sm in samples:
        if sm["shoulder_mid_x"] is None or sm["hip_mid_y"] is None:
            continue
        if line is not None:
            expected = line[0] * float(sm["hip_mid_y"]) + line[1]
            residuals.append(float(sm["shoulder_mid_x"]) - expected)
        else:
            residuals.append(float(sm["shoulder_mid_x"]))
    if line is None and residuals:
        m = sum(residuals) / len(residuals)
        residuals = [v - m for v in residuals]
    sway_px = (max(residuals) - min(residuals)) if residuals else 0.0
    sway_cm = (
        (sway_px / mean_shoulder_px) * _ASSUMED_SHOULDER_WIDTH_CM
        if mean_shoulder_px > 0 else 0.0
    )

    # ── Deviation aggregates ────────────────────────────────
    dev_cms = [
        float(s["deviation_cm"]) for s in steps
        if s["deviation_cm"] is not None
    ]
    mean_dev_cm = (sum(dev_cms) / len(dev_cms)) if dev_cms else 0.0
    max_dev_cm = max(dev_cms) if dev_cms else 0.0
    misstep_count = sum(1 for s in steps if s["is_misstep"])

    incomplete = len(steps) < _TARGET_STEP_COUNT
    duration_seconds = float(n_full / fps) if fps > 0 else 0.0
    termination = "timeout" if incomplete else "completed"
    classification = _classify(misstep_count, mean_dev_cm)

    # Screenshot: the frame of the first detected footstrike.
    screenshot: Optional[str] = None
    if raw_strikes:
        full_frame_idx = sampled_full_indices[raw_strikes[0]["sample_index"]]
        screenshot = _grab_capture_frame(video_path, full_frame_idx, raw)

    result = {
        "steps": steps,
        "misstep_count": int(misstep_count),
        "arm_grab_count": int(arm_grab_count),
        "mean_deviation_cm": float(mean_dev_cm),
        "max_deviation_cm": float(max_dev_cm),
        "step_time_mean_ms": float(st_mean),
        "step_time_stddev_ms": float(st_std),
        "step_time_cv": float(st_cv),
        "trunk_sway_range_px": float(sway_px),
        "trunk_sway_range_cm": float(sway_cm),
        "classification": classification,
        "duration_seconds": duration_seconds,
        "termination": termination,
        "incomplete": bool(incomplete),
        "walking_line":
            {"a": float(line[0]), "b": float(line[1])} if line else None,
        "mean_shoulder_width_px": float(mean_shoulder_px),
        "samples": samples,
        "keypoints": keypoints_export,
        "capture_screenshot_data_url": screenshot,
        "patient_age": int(patient_age) if patient_age is not None else None,
        # Diagnostic extras
        "fps": float(fps),
        "total_frames": int(n_full),
        "valid_frames": int(visible_frames),
    }
    result["interpretation"] = _build_interpretation(result)
    return result
