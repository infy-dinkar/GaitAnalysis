"""posture_engine.py — clinical posture analysis on backend
MediaPipe (BlazePose Full, 33 keypoints, IMAGE running mode).

Replaces the browser MoveNet pipeline at
motionlens-web/lib/posture/analyzer.ts (which is being removed
from the posture module by this PR). Same per-view measurement
math + same sign conventions (Phase A) + same keypoint
persistence layout (Phase B) so existing saved reports keep
rendering and the frontend display components don't need to
change.

Why this exists separately from gait/biomech engines:
  • Posture analyses STATIC PHOTOS, not video — requires
    RunningMode.IMAGE for the MediaPipe landmarker (the
    gait/biomech pipelines all use RunningMode.VIDEO).
  • Phone photos carry EXIF rotation metadata that
    cv2.imread()-style decoders don't honour; we use Pillow's
    ImageOps.exif_transpose() to apply the rotation BEFORE
    handing pixels to MediaPipe.
  • Output keypoints are emitted in the 17-element MoveNet
    layout (indexed via lib/pose/landmarks.ts:LM) — the
    overlay components and saved-report viewer read keypoints
    by those indices, so feeding raw BlazePose 33-element
    arrays would offset every dot/line in the saved overlay.

Math (matches motionlens-web/lib/posture/measurements.ts exactly):

  FRONT VIEW
    - headTilt          : lineTiltFromHorizontal(L-ear,  R-ear)
                          → positive when right ear is LOWER
    - shoulderTilt      : lineTiltFromHorizontal(L-sh,   R-sh)
                          → positive when right shoulder is LOWER
    - hipTilt           : lineTiltFromHorizontal(R-hip,  L-hip)
                          → positive when LEFT hip is LOWER
                          (Phase A spec convention)
    - leftKneeAlignment : interior angle at the L-knee
    - rightKneeAlignment: interior angle at the R-knee
    - frontalTrunkLean  : -lineTiltFromVertical(sh-mid, hip-mid)
                          → positive = lean toward patient's right
                          (Phase A spec convention)

  SIDE VIEW (per-side blocks; left + right reported separately)
    Per side `left` / `right`:
      - forwardHeadPct    = ((ear.x   - ankle.x) / bodyH) * 100
      - shoulderShiftPct  = ((sh.x    - ankle.x) / bodyH) * 100
      - hipShiftPct       = ((hip.x   - ankle.x) / bodyH) * 100
      - kneeShiftPct      = ((knee.x  - ankle.x) / bodyH) * 100
      - trunkLeanDeg      = bilateral midpoint, sign anchored
                            by pickedSide (positive = forward)
    bodyH = |ankle-mid.y − shoulder-mid.y| in pixels.
    `pickedSide` = side with higher min(score) across its 5 anchor
    keypoints.

  RELATIVE UNITS NOTE (Phase B): all side-view shifts are reported
  as % of body height. NOT calibrated to cm. The response carries
  `"relative_units": true` so the frontend can render the
  RelativeUnitsCaveat banner unchanged.

Findings (severity buckets match measurements.ts:
  TILT_OK=1.5°, TILT_MILD=3.5°, SHIFT_OK=3%, SHIFT_MILD=7%)
are generated server-side and returned alongside the metrics.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Optional

import mediapipe as mp
import numpy as np

from engines.posture_silhouette import (
    build_silhouette_logged,
    plumb_reference_x,
)

# Pillow imported lazily inside _load_image_rgb so a missing PIL
# dependency only breaks the posture endpoint, not the whole
# FastAPI app on module import.

log = logging.getLogger("motionlens.posture")


# ─── Visibility floor for posture math ─────────────────────────
# Matches the VIS_THRESHOLD in motionlens-web/lib/posture/
# measurements.ts (0.2). Looser than the per-frame default
# elsewhere because posture is a single still photo — we can't
# rely on time-series smoothing to recover from low-confidence
# keypoints, so we accept slightly-noisy landmarks rather than
# refuse to score the photo.
_POSTURE_VIS_THRESHOLD = 0.2


# ─── MoveNet ←→ BlazePose keypoint index mapping ───────────────
# The frontend's lib/pose/landmarks.ts:LM enum uses MoveNet's
# 17-keypoint layout (LEFT_EAR=3, LEFT_SHOULDER=5, etc.). The
# saved-report overlay viewer + the (now legacy) measurements.ts
# module both index keypoints by these positions. To keep the
# saved-report rendering and the frontend `LM` accessors working
# unchanged, we emit a 17-element keypoints array indexed
# MoveNet-style, with values pulled from the corresponding
# BlazePose 33-element landmark output.
_KP_NAMES_MOVENET = (
    "nose",
    "left_eye",       # BlazePose "left eye center" (idx 2)
    "right_eye",      # BlazePose "right eye center" (idx 5)
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# BlazePose 33-landmark index that maps to each MoveNet position.
# (None of these need the BlazePose foot/hand/mouth sub-points
# that MoveNet doesn't have.)
_MOVENET_TO_BLAZEPOSE_IDX = (
    0,   # nose
    2,   # left eye center
    5,   # right eye center
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
)


# ─── Math helpers (verbatim ports of measurements.ts) ──────────
def _line_tilt_from_horizontal(
    ax: float, ay: float, bx: float, by: float,
) -> float:
    """Tilt angle (degrees) of the line through (a,b) from
    horizontal, signed in (-90°, 90°]. Positive = `b` is LOWER
    in the image than `a`. Direction-agnostic via |Δx|."""
    dx = abs(bx - ax)
    dy = by - ay
    if dx < 1e-6:
        return 90.0 if dy > 0 else (-90.0 if dy < 0 else 0.0)
    return math.degrees(math.atan2(dy, dx))


def _line_tilt_from_vertical(
    top_x: float, top_y: float, bot_x: float, bot_y: float,
) -> float:
    """Tilt angle (degrees) from vertical, signed in (-90°, 90°].
    Positive = `top` is to the RIGHT of `bottom` in the image."""
    dx = top_x - bot_x
    dy_abs = abs(top_y - bot_y)
    if dy_abs < 1e-6:
        return 90.0 if dx > 0 else (-90.0 if dx < 0 else 0.0)
    return math.degrees(math.atan2(dx, dy_abs))


# Tilt / shift bucket thresholds — match measurements.ts. Phase
# A's clinical guidance used these values; the user's new spec
# proposed different cutoffs but the user also explicitly asked
# us to PRESERVE Phase A/B behaviour, so we keep these.
_TILT_OK_DEG = 1.5
_TILT_MILD_DEG = 3.5
_SHIFT_OK_PCT = 3.0
_SHIFT_MILD_PCT = 7.0
# Knee alignment is graded on its own (5° / 10° per measurements.ts).
_KNEE_OK_DEG = 5.0
_KNEE_MILD_DEG = 10.0


# ─── Pose model (IMAGE running mode) ───────────────────────────
def _build_image_pose_options():
    """PoseLandmarker options for RunningMode.IMAGE — posture
    analyses still photos, not video, so VIDEO/LIVE_STREAM modes
    would over-constrain the input.

    The model file is whatever `_ensure_pose_model_file()` defaults
    to, which is the BlazePose HEAVY asset — every offline/upload
    path takes the accurate model and only the realtime stream pool
    asks for "full". (This docstring said "Full" until the default
    flipped; it did not, and the code never did.)

    Segmentation is ON. The mask is used for OVERLAY GEOMETRY ONLY
    (engines/posture_silhouette.py) — no posture metric, grading or
    finding reads it, so every number is unchanged. Measured cost on
    Heavy at 1280x720 is ~+5 ms per photo against ~115 ms of
    inference, i.e. under 5%.
    """
    # Imported here so the constant is resolved at first call
    # (avoids touching the model path during module import — same
    # pattern api._build_gait_pose_options uses).
    from engines.biomech_flow import _ensure_pose_model_file
    model_path = _ensure_pose_model_file()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=True,
    )


# ─── Landmarker singleton ──────────────────────────────────────
# Constructing a PoseLandmarker costs ~700 ms (it loads and warms a
# 29 MB Heavy .task). This used to happen once PER PHOTO, so a
# five-view multi-view request spent ~3.5 s building models against
# ~0.6 s of actual inference. One process-wide instance removes all
# but the first.
#
# The lock guards construction AND detect(). MediaPipe Tasks does not
# document a landmarker as safe for concurrent detect() calls, and the
# cost of being wrong is a corrupted graph rather than a slow response,
# so inference is serialised. That is not a regression: the endpoint
# (api.analyze_posture) is an async def that calls this engine
# synchronously, so it already blocks its worker for the whole
# request.
_landmarker = None
_landmarker_lock = threading.Lock()


def _get_landmarker():
    """Process-wide PoseLandmarker, built on first use."""
    global _landmarker
    if _landmarker is None:
        with _landmarker_lock:
            if _landmarker is None:      # re-check inside the lock
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                _landmarker = PoseLandmarker.create_from_options(
                    _build_image_pose_options(),
                )
    return _landmarker


# ─── Image loading: EXIF-correct + RGB numpy ───────────────────
def _load_image_rgb(image_path: str) -> tuple[np.ndarray, int, int]:
    """Load image from disk, apply EXIF rotation correction (BUG
    10 in the spec — phone portrait photos otherwise get analysed
    sideways), return RGB numpy array + (width, height).

    Raises ValueError on invalid / unreadable images so the API
    endpoint can map to HTTP 400 invalid_image."""
    # Lazy import — keeps the whole FastAPI app up even if the
    # Pillow dep isn't installed on a particular environment;
    # only the posture endpoint would fail with a useful error.
    from PIL import Image, ImageOps
    try:
        with Image.open(image_path) as raw:
            corrected = ImageOps.exif_transpose(raw)
            if corrected.mode != "RGB":
                corrected = corrected.convert("RGB")
            arr = np.array(corrected, dtype=np.uint8)
            return arr, corrected.width, corrected.height
    except Exception as exc:
        raise ValueError("invalid_image") from exc


# ─── Pose extraction ──────────────────────────────────────────
def _extract_posture_keypoints(
    rgb_array: np.ndarray, image_width: int, image_height: int,
) -> tuple[list[dict], Optional[np.ndarray]]:
    """Run the BlazePose IMAGE-mode landmarker on a single
    RGB array and return `(keypoints, mask)`.

    `mask` is the person-segmentation confidence map: float32,
    shape (H, W) matching the input exactly, values in [0, 1].
    It is None whenever no person was found or the model returned
    no mask — a missing mask is NEVER an error, it just means the
    overlay falls back to landmark-anchored reference lines.

    `keypoints` is a 17-element list of keypoint dicts indexed in
    the MoveNet layout (so the frontend `LM` accessors keep
    working unchanged).

    The list ALWAYS has 17 entries with the same shape
      {"x": float (px), "y": float (px), "score": float,
       "name": "left_ear" | ... }
    matching MoveNet's "always emit all 17 keypoints" contract.
    Low-visibility / undetected landmarks get score=0 (and
    placeholder coordinates) — the downstream `_is_visible`
    check filters them by score, same as the browser path used
    to do. This is what the saved-report viewer + KeypointDTO
    serialiser expect (the ReportCreatePayload schema declares
    KeypointDTO[] non-nullable per element)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
    landmarker = _get_landmarker()
    with _landmarker_lock:
        result = landmarker.detect(mp_image)

    # numpy_view() is a VIEW onto memory the Image owns, and the
    # singleton reuses its buffers between calls — so copy before the
    # next detect() can overwrite it.
    mask: Optional[np.ndarray] = None
    masks = getattr(result, "segmentation_masks", None)
    if masks:
        try:
            mask = np.array(masks[0].numpy_view(), dtype=np.float32, copy=True)
            # SHAPE NORMALISATION. mediapipe 0.10.x hands back (H, W);
            # 1.0.0 hands back (H, W, 1) — a trailing channel axis. Every
            # consumer downstream indexes rows as mask[y] and expects a
            # 1-D row, so the 3-D form silently produced no geometry at
            # all: build_silhouette's `ndim != 2` guard returned None,
            # the attach is conditional, and the key simply vanished from
            # every view. That shipped to prod as a total feature outage
            # that looked exactly like "no mask available", because the
            # version was floating (requirements pinned only
            # mediapipe>=0.10.10) and the local env was still on 0.10.x.
            #
            # Collapse it here, once, so shape handling lives at the
            # single point the mask enters the codebase rather than in
            # each consumer.
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        except Exception:  # pragma: no cover — never fail a view on this
            log.warning("posture: segmentation mask unreadable", exc_info=True)
            mask = None

    def _empty_kp(movenet_idx: int) -> dict:
        # Placeholder for an undetected / low-visibility keypoint.
        # Frontend reads `score >= 0.2` to gate visibility, so a
        # score-0 entry is skipped naturally. Position (0, 0) so
        # the array can serialise as JSON without surprising
        # NaN values.
        return {
            "x": 0.0,
            "y": 0.0,
            "score": 0.0,
            "name": _KP_NAMES_MOVENET[movenet_idx],
        }

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        # No person detected — engine raises poor_visibility
        # downstream once the trunk-anchor gate runs. Return a
        # full 17-element placeholder array so callers that only
        # touch keypoints don't blow up.
        return [_empty_kp(i) for i in range(17)], mask

    lms = result.pose_landmarks[0]
    out: list[dict] = []
    for movenet_idx, blazepose_idx in enumerate(_MOVENET_TO_BLAZEPOSE_IDX):
        lm = lms[blazepose_idx]
        vis = float(lm.visibility) if lm.visibility is not None else 0.0
        # Keypoints are returned in image PIXEL coordinates (matches
        # MoveNet output convention; the frontend overlay scales
        # them when persisting to the compressed display image).
        if vis < _POSTURE_VIS_THRESHOLD:
            # Below the visibility floor — emit the placeholder so
            # the array stays 17-aligned and the saved-report
            # schema (KeypointDTO[] non-nullable per element)
            # validates cleanly.
            out.append(_empty_kp(movenet_idx))
        else:
            out.append({
                "x": float(lm.x) * image_width,
                "y": float(lm.y) * image_height,
                "score": vis,
                "name": _KP_NAMES_MOVENET[movenet_idx],
            })
    return out, mask


def _is_visible(kp: dict) -> bool:
    return (kp.get("score") or 0.0) >= _POSTURE_VIS_THRESHOLD


# ─── Body-height proxy for normalisation ──────────────────────
def _body_height_px(kps: list[dict]) -> Optional[float]:
    """Shoulder-midpoint to ankle-midpoint vertical distance, used
    to normalise side-view horizontal shifts (so "10% forward
    head" reads consistently across image sizes)."""
    ls = kps[5]; rs = kps[6]
    la = kps[15]; ra = kps[16]
    if not all(_is_visible(k) for k in (ls, rs, la, ra)):
        return None
    sh_mid_y = (ls["y"] + rs["y"]) / 2.0
    an_mid_y = (la["y"] + ra["y"]) / 2.0
    return abs(an_mid_y - sh_mid_y)


# ─── Front-view measurements ───────────────────────────────────
def _compute_front_measurements(kps: list[dict]) -> dict:
    out: dict = {
        "headTilt": None,
        "shoulderTilt": None,
        "hipTilt": None,
        "leftKneeAlignment": None,
        "rightKneeAlignment": None,
        "frontalTrunkLean": None,
    }
    l_ear = kps[3]; r_ear = kps[4]
    if _is_visible(l_ear) and _is_visible(r_ear):
        out["headTilt"] = _line_tilt_from_horizontal(
            l_ear["x"], l_ear["y"], r_ear["x"], r_ear["y"],
        )
    l_sh = kps[5]; r_sh = kps[6]
    if _is_visible(l_sh) and _is_visible(r_sh):
        out["shoulderTilt"] = _line_tilt_from_horizontal(
            l_sh["x"], l_sh["y"], r_sh["x"], r_sh["y"],
        )
    l_hip = kps[11]; r_hip = kps[12]
    if _is_visible(l_hip) and _is_visible(r_hip):
        # Phase A spec convention: positive = LEFT hip lower.
        # Pass (right, left) so the helper's "second arg lower
        # → positive" maps to "left lower → positive".
        out["hipTilt"] = _line_tilt_from_horizontal(
            r_hip["x"], r_hip["y"], l_hip["x"], l_hip["y"],
        )

    def _interior_at_knee(
        hip: dict, knee: dict, ank: dict,
    ) -> Optional[float]:
        up_x = hip["x"] - knee["x"]
        up_y = hip["y"] - knee["y"]
        down_x = ank["x"] - knee["x"]
        down_y = ank["y"] - knee["y"]
        dot = up_x * down_x + up_y * down_y
        mag = math.hypot(up_x, up_y) * math.hypot(down_x, down_y)
        if mag <= 0:
            return None
        cos_t = max(-1.0, min(1.0, dot / mag))
        return math.degrees(math.acos(cos_t))

    l_knee = kps[13]; l_ank = kps[15]
    if _is_visible(l_hip) and _is_visible(l_knee) and _is_visible(l_ank):
        out["leftKneeAlignment"] = _interior_at_knee(l_hip, l_knee, l_ank)
    r_knee = kps[14]; r_ank = kps[16]
    if _is_visible(r_hip) and _is_visible(r_knee) and _is_visible(r_ank):
        out["rightKneeAlignment"] = _interior_at_knee(r_hip, r_knee, r_ank)

    if (_is_visible(l_sh) and _is_visible(r_sh)
            and _is_visible(l_hip) and _is_visible(r_hip)):
        hip_mid_x = (l_hip["x"] + r_hip["x"]) / 2.0
        hip_mid_y = (l_hip["y"] + r_hip["y"]) / 2.0
        sh_mid_x = (l_sh["x"] + r_sh["x"]) / 2.0
        sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
        # Phase A spec convention: positive = lean toward patient's
        # right (camera's left in mirror frontal view).
        raw_tilt = _line_tilt_from_vertical(
            sh_mid_x, sh_mid_y, hip_mid_x, hip_mid_y,
        )
        out["frontalTrunkLean"] = -raw_tilt
    return out


# ─── Side-view measurements (per-side blocks; never averaged) ──
_SIDE_INDICES = {
    "left":  {"ear": 3, "sh": 5, "hip": 11, "knee": 13, "ank": 15},
    "right": {"ear": 4, "sh": 6, "hip": 12, "knee": 14, "ank": 16},
}


def _compute_one_side(
    kps: list[dict],
    idx: dict,
    body_h: Optional[float],
    trunk_lean: Optional[float],
    plumb_x: Optional[float] = None,
) -> Optional[dict]:
    ear   = kps[idx["ear"]]
    sh    = kps[idx["sh"]]
    hip   = kps[idx["hip"]]
    knee  = kps[idx["knee"]]
    ankle = kps[idx["ank"]]
    if not all(_is_visible(k) for k in (ear, sh, hip, knee, ankle)):
        return None
    out: dict = {
        "forwardHeadPct": None,
        "shoulderShiftPct": None,
        "hipShiftPct": None,
        "kneeShiftPct": None,
        "trunkLeanDeg": trunk_lean,
    }
    # REFERENCE X. The plumb the overlay draws, when the segmentation
    # mask gave one; the same-side ankle landmark otherwise (the
    # historical behaviour, and still what a maskless photo gets).
    #
    # They are not the same point: in profile the ankle landmark sits
    # INSIDE the leg while the foot extends forward of it, so the drawn
    # line hung roughly half a foot length ahead of the x the numbers
    # were measured from. A clinician dropping a physical plumb line
    # uses the foot, which is what plumb_x is.
    ref_x = plumb_x if plumb_x is not None else ankle["x"]
    out["shiftReference"] = "silhouette" if plumb_x is not None else "ankle"

    if body_h is not None and body_h > 0:
        out["forwardHeadPct"]   = ((ear["x"]   - ref_x) / body_h) * 100.0
        out["shoulderShiftPct"] = ((sh["x"]    - ref_x) / body_h) * 100.0
        out["hipShiftPct"]      = ((hip["x"]   - ref_x) / body_h) * 100.0
        out["kneeShiftPct"]     = ((knee["x"]  - ref_x) / body_h) * 100.0
        # TEMPORARY, for the review comparison: the same four measured
        # the old way, so the shift can be quantified per photo. API +
        # debug only — nothing renders these. Remove once signed off.
        out["forwardHeadPct_ankleRef"]   = ((ear["x"]  - ankle["x"]) / body_h) * 100.0
        out["shoulderShiftPct_ankleRef"] = ((sh["x"]   - ankle["x"]) / body_h) * 100.0
        out["hipShiftPct_ankleRef"]      = ((hip["x"]  - ankle["x"]) / body_h) * 100.0
        out["kneeShiftPct_ankleRef"]     = ((knee["x"] - ankle["x"]) / body_h) * 100.0
    return out


def _compute_side_measurements(
    kps: list[dict], plumb_x: Optional[float] = None,
) -> dict:
    # pickedSide = side with the highest min-confidence across
    # its 5 anchor keypoints. Used to anchor the bilateral
    # trunk-lean sign so positive consistently means "anatomical
    # forward" regardless of which side faces the camera.
    picked_side: Optional[str] = None
    best_min_score = 0.0
    for name, idx in _SIDE_INDICES.items():
        pts = [kps[idx["ear"]], kps[idx["sh"]], kps[idx["hip"]],
               kps[idx["knee"]], kps[idx["ank"]]]
        if not all(_is_visible(k) for k in pts):
            continue
        min_score = min((p["score"] or 0.0) for p in pts)
        if min_score > best_min_score:
            best_min_score = min_score
            picked_side = name

    # Phase A spec convention: sagittal trunk lean = angle of
    # hip-midpoint → shoulder-midpoint line vs vertical. Sign
    # anchored via pickedSide ("forward" depends on which side
    # faces the camera).
    trunk_lean: Optional[float] = None
    l_hip = kps[11]; r_hip = kps[12]
    l_sh  = kps[5];  r_sh  = kps[6]
    if all(_is_visible(k) for k in (l_hip, r_hip, l_sh, r_sh)):
        hip_mid_x = (l_hip["x"] + r_hip["x"]) / 2.0
        hip_mid_y = (l_hip["y"] + r_hip["y"]) / 2.0
        sh_mid_x = (l_sh["x"] + r_sh["x"]) / 2.0
        sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
        raw_tilt = _line_tilt_from_vertical(
            sh_mid_x, sh_mid_y, hip_mid_x, hip_mid_y,
        )
        if picked_side == "left":
            trunk_lean = raw_tilt
        elif picked_side == "right":
            trunk_lean = -raw_tilt

    # Near/far confidence telemetry. A profile camera sees one ear and
    # one hip; BlazePose still emits the occluded pair, often above the
    # 0.2 visibility floor, which is why the far-side block survives the
    # gate at all. One line per side-view analysis (not per request —
    # the engine has no request context) so Batch 2 can size a proper
    # near/far threshold from real captures instead of guessing one.
    log.info(
        "posture side: ear L/R=%.2f/%.2f hip L/R=%.2f/%.2f picked=%s",
        kps[3].get("score") or 0.0, kps[4].get("score") or 0.0,
        kps[11].get("score") or 0.0, kps[12].get("score") or 0.0,
        picked_side,
    )

    body_h = _body_height_px(kps)

    # How far the drawn plumb sits from the ankle landmark the numbers
    # used to reference — i.e. exactly how much this moved the metrics,
    # in pixels and as a fraction of body height.
    if plumb_x is not None and picked_side is not None:
        ank = kps[_SIDE_INDICES[picked_side]["ank"]]
        delta = plumb_x - ank["x"]
        log.info(
            "posture side: shift reference = silhouette plumb_x=%.1f "
            "ankle_x=%.1f delta=%+.1f px (%+.2f%% of body height)",
            plumb_x, ank["x"], delta,
            (delta / body_h * 100.0) if body_h else 0.0,
        )
    else:
        log.info("posture side: shift reference = ankle landmark "
                 "(no silhouette plumb available)")

    return {
        "pickedSide": picked_side,
        "left":  _compute_one_side(
            kps, _SIDE_INDICES["left"], body_h, trunk_lean, plumb_x),
        "right": _compute_one_side(
            kps, _SIDE_INDICES["right"], body_h, trunk_lean, plumb_x),
    }


# ─── Findings (server-side equivalent of measurements.ts:
#               buildFrontFindings + buildSideFindings) ─────────
def _grade_tilt(value: float, label: str, dir_pair: tuple[str, str]) -> dict:
    abs_v = abs(value)
    direction = dir_pair[1] if value >= 0 else dir_pair[0]
    if abs_v < _TILT_OK_DEG:
        return {
            "label": label,
            "value": f"{value:.1f}°",
            "severity": "ok",
            "detail": f"{label} is well aligned.",
        }
    if abs_v < _TILT_MILD_DEG:
        return {
            "label": label,
            "value": f"{value:.1f}°",
            "severity": "mild",
            "detail": f"Mild {label.lower()} tilt toward the {direction}.",
        }
    return {
        "label": label,
        "value": f"{value:.1f}°",
        "severity": "notable",
        "detail": f"Notable {label.lower()} tilt toward the {direction} — worth noting.",
    }


def _grade_shift(value: float, label: str) -> dict:
    abs_v = abs(value)
    direction = "forward" if value >= 0 else "backward"
    if abs_v < _SHIFT_OK_PCT:
        return {
            "label": label,
            "value": f"{value:.1f}%",
            "severity": "ok",
            "detail": f"{label} is aligned with the plumb line.",
        }
    if abs_v < _SHIFT_MILD_PCT:
        return {
            "label": label,
            "value": f"{value:.1f}%",
            "severity": "mild",
            "detail": f"Mild {direction} shift of {label.lower()}.",
        }
    return {
        "label": label,
        "value": f"{value:.1f}%",
        "severity": "notable",
        "detail": f"Notable {direction} shift of {label.lower()}.",
    }


def _build_front_findings(m: dict) -> list[dict]:
    out: list[dict] = []
    if m.get("headTilt") is not None:
        out.append(_grade_tilt(m["headTilt"], "Head tilt", ("left", "right")))
    if m.get("shoulderTilt") is not None:
        out.append(_grade_tilt(m["shoulderTilt"], "Shoulder tilt", ("left", "right")))
    if m.get("hipTilt") is not None:
        out.append(_grade_tilt(m["hipTilt"], "Hip tilt", ("left", "right")))
    for side_label, key in (("Left knee alignment", "leftKneeAlignment"),
                            ("Right knee alignment", "rightKneeAlignment")):
        val = m.get(key)
        if val is None:
            continue
        dev = 180.0 - val
        abs_dev = abs(dev)
        if abs_dev < _KNEE_OK_DEG:
            severity = "ok"
            detail = f"{side_label.split(' ')[0]} knee is well aligned in the frontal plane."
        elif abs_dev < _KNEE_MILD_DEG:
            severity = "mild"
            detail = f"{side_label} deviates {dev:.1f}° from a straight hip-knee-ankle line."
        else:
            severity = "notable"
            detail = f"{side_label} deviates {dev:.1f}° from a straight hip-knee-ankle line."
        out.append({
            "label": side_label,
            "value": f"{dev:.1f}°",
            "severity": severity,
            "detail": detail,
        })
    if m.get("frontalTrunkLean") is not None:
        v = m["frontalTrunkLean"]
        abs_v = abs(v)
        direction = "right" if v >= 0 else "left"
        if abs_v < _TILT_OK_DEG:
            severity = "ok"
            detail = "Trunk is upright in the frontal plane (positive = lean to patient's right)."
        elif abs_v < _TILT_MILD_DEG:
            severity = "mild"
            detail = f"Mild trunk lean toward the patient's {direction} (positive = lean to patient's right)."
        else:
            severity = "notable"
            detail = f"Notable trunk lean toward the patient's {direction} (positive = lean to patient's right)."
        out.append({
            "label": "Frontal trunk lean",
            "value": f"{v:.1f}°",
            "severity": severity,
            "detail": detail,
        })
    return out


def _build_side_findings(m: dict) -> list[dict]:
    out: list[dict] = []
    for view in ("left", "right"):
        block = m.get(view)
        if block is None:
            continue
        tag = f"({view} side view)"
        for key, label in (("forwardHeadPct",   "Head"),
                           ("shoulderShiftPct", "Shoulders"),
                           ("hipShiftPct",      "Hips"),
                           ("kneeShiftPct",     "Knees")):
            val = block.get(key)
            if val is None:
                continue
            finding = _grade_shift(val, label)
            finding["label"] = f"{finding['label']} {tag}"
            out.append(finding)
    # Trunk lean is bilateral — emit once.
    left_block = m.get("left") or {}
    right_block = m.get("right") or {}
    trunk = left_block.get("trunkLeanDeg")
    if trunk is None:
        trunk = right_block.get("trunkLeanDeg")
    if trunk is not None:
        abs_t = abs(trunk)
        if abs_t < 2.0:
            severity = "ok"
            detail = "Trunk is upright."
        elif abs_t < 5.0:
            severity = "mild"
            direction = "forward" if trunk > 0 else "backward"
            detail = f"Trunk leans {direction} {abs_t:.1f}° from vertical."
        else:
            severity = "notable"
            direction = "forward" if trunk > 0 else "backward"
            detail = f"Trunk leans {direction} {abs_t:.1f}° from vertical."
        out.append({
            "label": "Trunk lean",
            "value": f"{trunk:.1f}°",
            "severity": severity,
            "detail": detail,
        })
    return out


# ─── Main entry point ──────────────────────────────────────────
def analyze_posture_image(image_path: str, view: str) -> dict:
    """Run the BlazePose Full IMAGE-mode pipeline on a single
    posture photo.

    Args:
        image_path: path to the uploaded photo on disk.
        view:       "front" or "side". Determines which measurement
                    block is computed; the keypoints array is the
                    same shape regardless of view.

    Returns:
        Dict with:
          {
            "view": "front" | "side",
            "imageWidth":  int,
            "imageHeight": int,
            "keypoints":   [17 items, MoveNet-indexed, dicts or null],
            "front":       {...} | None,     # populated when view=="front"
            "side":        {...} | None,     # populated when view=="side"
            "findings":    [...],            # severity-graded findings
          }

    Raises:
        ValueError("invalid_image")   when PIL can't read the file.
        ValueError("poor_visibility") when no person is detected
                                       in the image.
    """
    if view not in ("front", "side"):
        raise ValueError(f"Unsupported posture view: {view!r}")

    rgb, img_w, img_h = _load_image_rgb(image_path)
    kps, mask = _extract_posture_keypoints(rgb, img_w, img_h)

    # Need at least the trunk anchors (both shoulders + both hips)
    # for either view to be usable. Below that, the photo isn't
    # going to produce meaningful posture metrics.
    trunk_anchors = [kps[5], kps[6], kps[11], kps[12]]
    if not all(_is_visible(k) for k in trunk_anchors):
        raise ValueError("poor_visibility")

    # Built BEFORE the metrics: the side shifts are referenced to the
    # same plumb the overlay draws, so the geometry has to exist first.
    silhouette = build_silhouette_logged(mask, kps, view)

    front: Optional[dict] = None
    side: Optional[dict] = None
    findings: list[dict] = []

    if view == "front":
        front = _compute_front_measurements(kps)
        findings = _build_front_findings(front)
    else:
        side = _compute_side_measurements(kps, plumb_reference_x(silhouette))
        findings = _build_side_findings(side)

    out = {
        "view": view,
        "imageWidth": img_w,
        "imageHeight": img_h,
        "keypoints": kps,
        "front": front,
        "side": side,
        "findings": findings,
    }

    # OVERLAY-ONLY addition. Attached only when the mask yielded usable
    # geometry, so a response without the key is byte-identical to what
    # this function returned before — which is also exactly what every
    # already-saved report looks like, and what the frontend's fallback
    # path expects. Never let overlay geometry fail an analysis.
    if silhouette is not None:
        out["silhouette"] = silhouette
    return out


def analyze_posture_combined(
    front_image_path: str,
    side_image_path: str,
) -> dict:
    """Two-view posture analysis. Calls analyze_posture_image on
    each photo and returns a single combined response — the
    shape the new /api/analyze-posture endpoint returns to the
    frontend.

    Returns:
        Dict with `front` + `side` blocks, each matching the
        existing PostureAnalysisResult shape on the frontend so
        the analyzer.ts adapter can destructure into two
        PostureAnalysisResult objects without remapping.
        `relative_units: true` flag preserved (Phase B) so the
        RelativeUnitsCaveat component keeps rendering.
    """
    front_result = analyze_posture_image(front_image_path, "front")
    side_result  = analyze_posture_image(side_image_path,  "side")
    return {
        "front": front_result,
        "side":  side_result,
        # Phase B: % shifts are relative to body height in pixels;
        # NOT calibrated to cm. The frontend's
        # RelativeUnitsCaveat banner reads this flag.
        "relative_units": True,
    }
