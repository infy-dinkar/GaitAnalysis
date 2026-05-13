"""sppb_balance_engine.py — SPPB Component 1 (Balance) analysis.

Why this module exists separately from the frontend C4 (4-Stage Balance)
math: SPPB Balance needs to detect three stage positions (side-by-side,
semi-tandem, tandem) from a single uploaded clip with no live operator
intervention. The frontend C4 path is interactive — the operator
explicitly progresses between stages and the live MoveNet pose model
runs in the browser. Here we do everything offline on the server.

Why MediaPipe instead of reusing C4 math: MoveNet has 17 keypoints,
none of which are foot-specific (no heel, no foot_index). For the
semi-tandem and tandem stages especially, "heel of one foot beside
the big toe of the other" is the spec — and that geometry literally
requires the foot landmarks. MediaPipe BlazePose Full (33 keypoints,
indices 29-32 for heels + foot indices) is the right tool.

Output: per-stage results in a shape directly consumable by the
existing `buildBalanceComponent` builder on the frontend — i.e.,
`{1?: StageResult, 2?: StageResult, 3?: StageResult}` — so the
composite SPPB scoring (Guralnik 1994 cutoffs) keeps working
unchanged.

This module does NOT touch:
  - gait_engine.py (only imports its pose pipeline)
  - tug_engine.py (only imports the WebM-repair helper)
  - the frontend C4 math (lib/orthopedic/fourStageBalance.ts)
  - the C2 5xSTS math
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from gait_engine import (
    LM,
    build_time_series,
    extract_poses,
)


# ─── Constants ──────────────────────────────────────────────────
# Match the frontend C4 constants where applicable so the two paths
# produce comparable results. Anything stage-detection-specific
# here is foot-landmark-based and uses different thresholds.

VIS_THRESHOLD       = 0.4   # anchor landmarks
FOOT_VIS_THRESHOLD  = 0.3   # heel + foot_index can be slightly noisier

STAGE_HOLD_SEC      = 10.0  # SPPB spec — same as C4
ARM_GRAB_DEG        = 45.0  # wrist abducts past 45° from straight-down

# Body-height-relative thresholds for stage detection. Working in
# RELATIVE units (fraction of body height in px) lets the same
# numbers work across camera distances.
#
# Detection strategy — DY-PRIMARY:
# Earlier versions of this classifier relied heavily on a
# `dx_tandem` measurement (heel-of-front-foot to big-toe-of-back-
# foot, X distance) to discriminate stages 2 / 3 from stage 1.
# That depended on (a) correctly identifying which foot is "front",
# and (b) the patient being perfectly square to the camera. In
# real-patient testing both assumptions broke — the patient stood
# at a slight angle, and the front/back foot detection (which used
# foot_index Y) flipped frame-to-frame, producing huge dx_tandem
# values that never matched the tight thresholds.
#
# What DOES discriminate reliably across camera angles: the
# vertical separation between the two heels in the image (dy_heel).
# In a frontal-ish view:
#   side-by-side  → both feet at floor at the same depth → dy_heel small
#   semi-tandem   → one foot half-step ahead in depth → dy_heel medium
#   tandem        → one foot fully ahead → dy_heel large
# The dx_heel measurement still serves as a sanity gate: stances
# wider than ~50 % of body height aren't SPPB-valid regardless of
# dy_heel.
DY_STAGE1_MAX     = 0.12   # ≤ this → likely Stage 1 (side-by-side)
DY_STAGE2_MAX     = 0.28   # (DY_STAGE1_MAX, this] → likely Stage 2
DY_STAGE3_MAX     = 0.55   # (DY_STAGE2_MAX, this] → likely Stage 3
DX_HEEL_VALID_MAX = 0.55   # heels wider than this → not an SPPB stance

# Frame-grouping / hysteresis. We want a contiguous run of frames
# where the patient is unambiguously in stage K — but real video
# has 1-2 frame jitter at the boundary. The MIN_RUN gate rejects
# spurious flickers; the SMOOTH window applies a majority vote over
# a small neighbourhood.
MIN_RUN_SEC          = 0.3   # ignore stage-runs shorter than this
SMOOTH_WINDOW        = 3     # majority-vote over 3 frames
TRANSITION_GRACE_SEC = 0.7   # tolerated gaps within a hold
# A run can absorb up to TRANSITION_GRACE_SEC of non-matching frames
# without breaking. 0.7 s is the sweet spot empirically:
#   - long enough to absorb brief mid-hold MediaPipe misclassifications
#     (typically 0.1-0.3 s of stray non-stage frames)
#   - short enough that an actual stage transition (~2-3 s of the
#     patient adjusting feet) breaks the run cleanly
#   - prevents Stage 1's run from absorbing Stage 2's hold when
#     boundary-noise Stage 1 classifications appear inside the
#     Stage 2 window (which was making Stage 1 run for 20+ s on
#     production with the old 1.5 s grace).


# ─── Per-frame stage classifier ─────────────────────────────────
def _classify_stage(
    heel_lx: float, heel_ly: float,
    heel_rx: float, heel_ry: float,
    fi_lx: float,   fi_ly: float,
    fi_rx: float,   fi_ry: float,
    body_h_px: float,
) -> Optional[int]:
    """Classify a single frame's foot geometry into one of {1, 2, 3}
    or None for invalid / non-SPPB stance.

    Strategy: dy_heel (vertical separation between heels in the
    image) is the primary discriminator because it's robust to
    patient body angle and works regardless of which foot is in
    front. dx_heel acts as a sanity gate — stances wider than ~50 %
    body height are not SPPB stances.

    Stage 1 (side-by-side):  dy_heel <= 0.10  (heels at same depth)
    Stage 2 (semi-tandem):   0.10 < dy_heel <= 0.20
    Stage 3 (tandem):        0.20 < dy_heel <= 0.45

    Foot landmarks (fi_lx etc.) are accepted but ignored — kept in
    the signature for backward compatibility with the existing
    call site. Future refinements may bring them back as secondary
    confirmation.
    """
    if body_h_px <= 0:
        return None

    dx_heel = abs(heel_lx - heel_rx)
    dy_heel = abs(heel_ly - heel_ry)

    dx_heel_n = dx_heel / body_h_px
    dy_heel_n = dy_heel / body_h_px

    # Wide-stance sanity gate. Not an SPPB stance.
    if dx_heel_n > DX_HEEL_VALID_MAX:
        return None

    if dy_heel_n <= DY_STAGE1_MAX:
        return 1
    if dy_heel_n <= DY_STAGE2_MAX:
        return 2
    if dy_heel_n <= DY_STAGE3_MAX:
        return 3
    return None


def _body_height_px(ts: dict, i: int, vis_threshold: float) -> Optional[float]:
    """HIP-midpoint to ANKLE-midpoint vertical distance in pixels.

    Why pinned to hip-to-ankle (not shoulder-to-ankle):
    MediaPipe's shoulder detection reliability varies across
    platforms (different cv2 / ffmpeg / MediaPipe versions on
    Linux production vs Windows dev). On one platform it's stable;
    on the other it sometimes misses or misplaces shoulders. That
    made the previous "max visible span" reference produce
    dramatically different body-height values for the same video
    on different platforms — same dy_heel_px ends up as different
    dy_heel_n ratios → different stage classifications.

    Hips and ankles are detected consistently when the lower body
    is in frame (which is a hard requirement for SPPB Balance
    anyway). Pinning to hip-to-ankle gives a cross-platform-stable
    scale. The stage thresholds (DY_STAGE1_MAX etc.) were tuned
    against this scale and don't need re-tuning.

    Falls back to heel/foot_index if ankles aren't both visible.
    """
    # Hip midpoint
    hip_ys: list[float] = []
    for k in ("left_hip", "right_hip"):
        if ts[k]["vis"][i] >= vis_threshold:
            hip_ys.append(float(ts[k]["y_px"][i]))
    if not hip_ys:
        return None
    hip_y = sum(hip_ys) / len(hip_ys)

    # Ankle midpoint (with foot-landmark fallback so we still have
    # a reference when ankles are partially occluded).
    foot_ys: list[float] = []
    for k in ("left_ankle", "right_ankle"):
        if ts[k]["vis"][i] >= vis_threshold:
            foot_ys.append(float(ts[k]["y_px"][i]))
    if not foot_ys:
        for k in ("left_heel", "right_heel", "left_foot_index", "right_foot_index"):
            if k in ts and ts[k]["vis"][i] >= vis_threshold:
                foot_ys.append(float(ts[k]["y_px"][i]))
    if not foot_ys:
        return None
    foot_y = sum(foot_ys) / len(foot_ys)

    h = abs(foot_y - hip_y)
    return h if h > 0 else None


def _is_arm_grab(ts: dict, i: int) -> bool:
    """Wrist abducts >45° from straight-down on either side — mirrors
    isArmGrab() in lib/orthopedic/fourStageBalance.ts."""
    # MediaPipe BlazePose: left_wrist=15, right_wrist=16.
    # gait_engine.LM doesn't include wrists today; use raw indices via
    # ts dict only if the keys exist. To stay self-contained we map
    # by name when present, else skip — patient can still pass with
    # no arm-grab detection (just one of several termination triggers).
    sides = [
        ("left_shoulder", "left_wrist"),
        ("right_shoulder", "right_wrist"),
    ]
    for sh_key, wr_key in sides:
        if sh_key not in ts or wr_key not in ts:
            continue
        if ts[sh_key]["vis"][i] < VIS_THRESHOLD: continue
        if ts[wr_key]["vis"][i] < VIS_THRESHOLD: continue
        sh_x = ts[sh_key]["x_px"][i]; sh_y = ts[sh_key]["y_px"][i]
        wr_x = ts[wr_key]["x_px"][i]; wr_y = ts[wr_key]["y_px"][i]
        vx = wr_x - sh_x
        vy = wr_y - sh_y
        if math.hypot(vx, vy) == 0: continue
        angle = abs(math.degrees(math.atan2(vx, vy)))
        if angle > ARM_GRAB_DEG:
            return True
    return False


# ─── Run-finding ────────────────────────────────────────────────
def _find_longest_run(
    labels: list[Optional[int]],
    stage: int,
    fps: float,
    grace_frames: int,
) -> Optional[tuple[int, int]]:
    """Find the longest contiguous span of frames where `labels[i] == stage`,
    tolerating up to `grace_frames` interleaved non-stage frames per
    contiguous run.

    Returns (start_idx, end_idx) inclusive, or None. `end_idx` includes
    the trailing grace-period frames after the last in-stage frame —
    semantically the patient is still IN the stance during those
    short gaps (we just briefly lost MediaPipe detection), so they
    count toward the hold duration.
    """
    n = len(labels)
    best: Optional[tuple[int, int]] = None
    best_len = -1

    i = 0
    while i < n:
        if labels[i] != stage:
            i += 1
            continue
        # Start a run.
        start = i
        last_in_stage = i
        j = i + 1
        while j < n:
            if labels[j] == stage:
                last_in_stage = j
            elif j - last_in_stage > grace_frames:
                break
            j += 1
        # Include the trailing grace window as part of the hold. The
        # patient was in the stance during those frames; the classifier
        # just couldn't see them clearly. Bounded by the actual frame
        # that broke the run so we don't extend into frames where the
        # patient clearly transitioned to the next stage.
        end_with_grace = min(j - 1, n - 1)
        run_len = end_with_grace - start + 1
        if run_len > best_len:
            best_len = run_len
            best = (start, end_with_grace)
        i = j
    if best is None or best_len < int(fps * MIN_RUN_SEC):
        return None
    return best


def _smooth_labels(labels: list[Optional[int]]) -> list[Optional[int]]:
    """Majority vote over a SMOOTH_WINDOW neighbourhood. Replaces
    spurious single-frame stage flickers with the surrounding stage."""
    n = len(labels)
    out: list[Optional[int]] = [None] * n
    half = SMOOTH_WINDOW // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        votes: dict[Optional[int], int] = {}
        for k in range(lo, hi):
            votes[labels[k]] = votes.get(labels[k], 0) + 1
        # Pick the highest-vote label, breaking ties by preferring
        # the centre frame's original label.
        best = labels[i]
        best_count = votes.get(best, 0)
        for k, v in votes.items():
            if v > best_count:
                best = k
                best_count = v
        out[i] = best
    return out


# ─── Sway helpers ───────────────────────────────────────────────
def _sway_path_length(path: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def _sway_ellipse_95(path: list[tuple[float, float]]) -> float:
    n = len(path)
    if n < 3:
        return 0.0
    mx = sum(p[0] for p in path) / n
    my = sum(p[1] for p in path) / n
    sxx = syy = sxy = 0.0
    for p in path:
        dx = p[0] - mx; dy = p[1] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n; syy /= n; sxy /= n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, tr * tr / 4 - det))
    l1 = tr / 2 + disc
    l2 = max(0.0, tr / 2 - disc)
    return math.pi * l1 * l2 * 5.991


# ─── Public entry point ─────────────────────────────────────────
def analyze_sppb_balance(
    video_path: str,
    pose_options,
) -> dict:
    """Run SPPB Balance analysis on `video_path`.

    Returns a dict with one entry per stage attempted (keyed by
    string "1", "2", "3" so it serialises cleanly through JSON). Each
    entry matches the frontend StageResult shape (subset of the C4
    StageResult — samples / keypoints are omitted because the live
    UI uses them for sticky-overlay replay, and we don't need that
    server-side).

    A stage is considered "attempted" when it has at least one frame
    classified into it during the trial. A stage is "passed" when the
    longest contiguous hold reaches STAGE_HOLD_SEC seconds. SPPB
    progression: if Stage K fails, stages K+1 and onward are reported
    as `not_attempted` even if foot positions matched briefly during
    setup adjustments — this keeps the buildBalanceComponent scoring
    consistent with the spec.
    """
    raw, fps, _ = extract_poses(video_path, pose_options)
    ts = build_time_series(raw)

    # All landmark series have the same length (= number of decoded
    # frames) since build_time_series preserves length and only
    # interpolates NaNs.
    n = len(ts["left_heel"]["x_px"])

    if n == 0:
        raise ValueError(
            "No frames could be decoded from the video. The file may be "
            "corrupted or in an unsupported format."
        )
    if n < int(fps * 1.0):
        raise ValueError(
            f"Recording is too short ({n / fps:.1f} s of decoded video). "
            f"Please record at least 30 seconds covering all three stages."
        )

    # Foot-visibility pre-flight. Without heel + foot_index visible
    # in most frames, stage detection is impossible regardless of how
    # long the video is. This is the most common failure mode (camera
    # framed at head/torso level only) and the original "video too
    # short" message obscured it. Check up-front so the operator gets
    # an actionable error before the engine runs the full pipeline.
    l_heel_v_arr  = ts["left_heel"]["vis"]
    r_heel_v_arr  = ts["right_heel"]["vis"]
    l_fi_v_arr    = ts["left_foot_index"]["vis"]
    r_fi_v_arr    = ts["right_foot_index"]["vis"]

    visible_foot_frames = 0
    for i in range(n):
        if (
            l_heel_v_arr[i] >= FOOT_VIS_THRESHOLD
            and r_heel_v_arr[i] >= FOOT_VIS_THRESHOLD
            and l_fi_v_arr[i] >= FOOT_VIS_THRESHOLD
            and r_fi_v_arr[i] >= FOOT_VIS_THRESHOLD
        ):
            visible_foot_frames += 1

    visible_ratio = visible_foot_frames / n if n > 0 else 0.0
    if visible_foot_frames < int(fps * 1.0) or visible_ratio < 0.20:
        raise ValueError(
            "Could not see the patient's feet in the recording. "
            "Make sure the camera frames the patient's FULL BODY — "
            "head to toes — so heel and toe positions are visible "
            "throughout the test. "
            f"(Feet were detected in only {visible_ratio * 100:.0f}% "
            f"of frames; need ≥ 20%.)"
        )

    # Per-frame stage classification.
    labels: list[Optional[int]] = []
    arm_grabs: list[bool] = []
    hip_path_full: list[Optional[tuple[float, float]]] = []

    l_heel_xs = ts["left_heel"]["x_px"];  l_heel_ys = ts["left_heel"]["y_px"]
    r_heel_xs = ts["right_heel"]["x_px"]; r_heel_ys = ts["right_heel"]["y_px"]
    l_fi_xs   = ts["left_foot_index"]["x_px"];  l_fi_ys = ts["left_foot_index"]["y_px"]
    r_fi_xs   = ts["right_foot_index"]["x_px"]; r_fi_ys = ts["right_foot_index"]["y_px"]
    l_heel_v  = ts["left_heel"]["vis"];   r_heel_v  = ts["right_heel"]["vis"]
    l_fi_v    = ts["left_foot_index"]["vis"]; r_fi_v = ts["right_foot_index"]["vis"]
    l_hip_xs  = ts["left_hip"]["x_px"];   l_hip_ys  = ts["left_hip"]["y_px"]
    r_hip_xs  = ts["right_hip"]["x_px"];  r_hip_ys  = ts["right_hip"]["y_px"]
    l_hip_v   = ts["left_hip"]["vis"];    r_hip_v   = ts["right_hip"]["vis"]

    # First pass — collect per-frame body_h to determine a stable
    # reference. Per-frame body_h varied 153-285 px in real testing
    # (because MediaPipe's "top" landmark detection flickers between
    # shoulder and hip across frames), which made the same physical
    # heel-Y-gap produce different normalized values frame-to-frame.
    # That broke contiguous-run detection. Locking the reference to
    # the video-wide median gives every frame the same yardstick.
    body_h_failed_count = 0
    body_h_samples: list[float] = []
    for i in range(n):
        if (
            l_heel_v[i] < FOOT_VIS_THRESHOLD or r_heel_v[i] < FOOT_VIS_THRESHOLD
            or l_fi_v[i] < FOOT_VIS_THRESHOLD or r_fi_v[i] < FOOT_VIS_THRESHOLD
        ):
            continue
        body_h = _body_height_px(ts, i, VIS_THRESHOLD)
        if body_h is None or body_h <= 0:
            body_h_failed_count += 1
            continue
        body_h_samples.append(body_h)

    stable_body_h: Optional[float] = None
    if body_h_samples:
        sorted_h = sorted(body_h_samples)
        stable_body_h = sorted_h[len(sorted_h) // 2]  # median

    if stable_body_h is None or stable_body_h <= 0:
        raise ValueError(
            "Could not determine a reliable body-height reference from "
            "the video. Make sure the patient's upper body (shoulders or "
            "hips) is visible alongside the feet."
        )

    # Second pass — classify using the locked reference.
    geometry_unmatched_count = 0
    dx_heel_n_samples: list[float] = []
    dy_heel_n_samples: list[float] = []
    dx_tandem_n_samples: list[float] = []

    for i in range(n):
        # Hip midpoint for sway path.
        if l_hip_v[i] >= VIS_THRESHOLD and r_hip_v[i] >= VIS_THRESHOLD:
            hip_path_full.append((
                (float(l_hip_xs[i]) + float(r_hip_xs[i])) / 2.0,
                (float(l_hip_ys[i]) + float(r_hip_ys[i])) / 2.0,
            ))
        else:
            hip_path_full.append(None)

        # Arm grab.
        arm_grabs.append(_is_arm_grab(ts, i))

        # Foot landmark visibility gate.
        if (
            l_heel_v[i] < FOOT_VIS_THRESHOLD or r_heel_v[i] < FOOT_VIS_THRESHOLD
            or l_fi_v[i] < FOOT_VIS_THRESHOLD or r_fi_v[i] < FOOT_VIS_THRESHOLD
        ):
            labels.append(None)
            continue

        # Compute the body-relative measurements inline so we can
        # record them for diagnostics. Duplicates the math in
        # _classify_stage; keeping the classifier untouched.
        kx_l, ky_l = float(l_heel_xs[i]), float(l_heel_ys[i])
        kx_r, ky_r = float(r_heel_xs[i]), float(r_heel_ys[i])
        fx_l, fy_l = float(l_fi_xs[i]), float(l_fi_ys[i])
        fx_r, fy_r = float(r_fi_xs[i]), float(r_fi_ys[i])
        dx_heel_n_samples.append(abs(kx_l - kx_r) / stable_body_h)
        dy_heel_n_samples.append(abs(ky_l - ky_r) / stable_body_h)
        if fy_l > fy_r:
            dx_tandem_n_samples.append(abs(kx_l - fx_r) / stable_body_h)
        else:
            dx_tandem_n_samples.append(abs(kx_r - fx_l) / stable_body_h)

        stage = _classify_stage(
            kx_l, ky_l, kx_r, ky_r,
            fx_l, fy_l, fx_r, fy_r,
            stable_body_h,
        )
        if stage is None:
            geometry_unmatched_count += 1
        labels.append(stage)

    smoothed = _smooth_labels(labels)
    grace_frames = max(1, int(fps * TRANSITION_GRACE_SEC))

    # Pre-compute the longest run for EVERY stage regardless of
    # progression. We report these in diagnostics so the operator
    # can see whether stages 2/3 had runs even when Stage 1 fails
    # (and thus blocks progression).
    longest_runs_per_stage: dict[str, dict] = {}
    for s_idx in (1, 2, 3):
        r = _find_longest_run(smoothed, s_idx, fps, grace_frames)
        if r is None:
            longest_runs_per_stage[str(s_idx)] = {"frames": 0, "seconds": 0.0}
        else:
            frames = r[1] - r[0] + 1
            longest_runs_per_stage[str(s_idx)] = {
                "frames": frames,
                "seconds": round(frames / fps, 2),
            }

    # Per-stage runs.
    stages_out: dict[str, dict] = {}
    progression_blocked = False

    for stage_idx in (1, 2, 3):
        if progression_blocked:
            stages_out[str(stage_idx)] = {
                "stage": stage_idx,
                "outcome": "not_attempted",
                "hold_seconds": 0.0,
                "duration_seconds": 0.0,
                "failure_mode": None,
                "sway_path_px": 0.0,
                "sway_95_ellipse_px2": 0.0,
                "hip_path": [],
                "samples": [],
                "keypoints": [],
                "screenshot_data_url": None,
            }
            continue

        run = _find_longest_run(smoothed, stage_idx, fps, grace_frames)
        if run is None:
            # Stage not attempted at all (or run too short).
            stages_out[str(stage_idx)] = {
                "stage": stage_idx,
                "outcome": "not_attempted",
                "hold_seconds": 0.0,
                "duration_seconds": 0.0,
                "failure_mode": None,
                "sway_path_px": 0.0,
                "sway_95_ellipse_px2": 0.0,
                "hip_path": [],
                "samples": [],
                "keypoints": [],
                "screenshot_data_url": None,
            }
            progression_blocked = True
            continue

        start_i, end_i = run
        duration_sec = (end_i - start_i + 1) / fps

        # Failure detection within the run window.
        failure_mode: Optional[str] = None
        if any(arm_grabs[start_i:end_i + 1]):
            failure_mode = "arm_grab"

        # Outcome.
        if duration_sec >= STAGE_HOLD_SEC and failure_mode is None:
            outcome = "pass"
            hold_sec = STAGE_HOLD_SEC
        else:
            outcome = "fail"
            hold_sec = min(duration_sec, STAGE_HOLD_SEC)
            if failure_mode is None:
                failure_mode = "position_lost"
            progression_blocked = True

        # Hip-path within the run for sway analytics.
        path_pts: list[tuple[float, float]] = []
        for k in range(start_i, end_i + 1):
            p = hip_path_full[k]
            if p is not None:
                path_pts.append(p)

        stages_out[str(stage_idx)] = {
            "stage": stage_idx,
            "outcome": outcome,
            "hold_seconds": float(hold_sec),
            "duration_seconds": float(duration_sec),
            "failure_mode": failure_mode,
            "sway_path_px": float(_sway_path_length(path_pts)),
            "sway_95_ellipse_px2": float(_sway_ellipse_95(path_pts)),
            "hip_path": [{"x": p[0], "y": p[1]} for p in path_pts],
            "samples": [],
            "keypoints": [],
            "screenshot_data_url": None,
        }

    # Diagnostic counts — useful for debugging when stages aren't
    # being detected as expected. Counted on the SMOOTHED labels so
    # the numbers reflect what the run-finder actually saw.
    stage_frame_counts = {
        "stage_1": sum(1 for v in smoothed if v == 1),
        "stage_2": sum(1 for v in smoothed if v == 2),
        "stage_3": sum(1 for v in smoothed if v == 3),
        "unclassified": sum(1 for v in smoothed if v is None),
    }

    # Measurement statistics — what the classifier ACTUALLY saw frame
    # by frame. When stages aren't detecting, these tell us whether the
    # patient's geometry sits inside or outside the threshold windows.
    def _quartile_stats(samples: list[float]) -> dict:
        if not samples:
            return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
        arr = sorted(samples)
        n_s = len(arr)
        def at(p: float) -> float:
            idx = max(0, min(n_s - 1, int(p * (n_s - 1))))
            return round(arr[idx], 4)
        return {
            "count": n_s,
            "min": round(arr[0], 4),
            "p25": at(0.25),
            "median": at(0.5),
            "p75": at(0.75),
            "max": round(arr[-1], 4),
        }

    return {
        "fps": float(fps),
        "total_frames": n,
        "stages": stages_out,
        "diagnostics": {
            "frame_classification_counts": stage_frame_counts,
            "longest_runs_per_stage": longest_runs_per_stage,
            "visible_foot_frames": visible_foot_frames,
            "visible_foot_ratio": round(visible_ratio, 3),
            "min_run_frames": int(fps * MIN_RUN_SEC),
            "smooth_window": SMOOTH_WINDOW,
            "body_h_failed_count": body_h_failed_count,
            "geometry_unmatched_count": geometry_unmatched_count,
            "body_h_px": _quartile_stats(body_h_samples),
            "stable_body_h_px": round(stable_body_h, 1),
            "dx_heel_n": _quartile_stats(dx_heel_n_samples),
            "dy_heel_n": _quartile_stats(dy_heel_n_samples),
            "dx_tandem_n": _quartile_stats(dx_tandem_n_samples),
            "frame_width": int(ts.get("_frame_w", 0)),
            "frame_height": int(ts.get("_frame_h", 0)),
            "thresholds": {
                # Kept the old keys for frontend compatibility even
                # though the classifier no longer reads them.
                "stage1_x_min": 0.01,
                "stage1_x_max": DX_HEEL_VALID_MAX,
                "stage_dy_tight": DY_STAGE1_MAX,
                "stage_dy_med_min": DY_STAGE1_MAX,
                "stage_dy_med_max": DY_STAGE2_MAX,
                "stage_dy_large": DY_STAGE2_MAX,
                "tandem_x": 0.0,
                "semi_x": 0.0,
                # New canonical thresholds — the frontend will start
                # using these once the diagnostics panel learns about
                # the dy-primary classifier.
                "dy_stage1_max": DY_STAGE1_MAX,
                "dy_stage2_max": DY_STAGE2_MAX,
                "dy_stage3_max": DY_STAGE3_MAX,
                "dx_heel_valid_max": DX_HEEL_VALID_MAX,
            },
        },
    }
