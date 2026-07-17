"""posture_engine_multi_view.py — additive multi-view posture wrappers.

Adds `back`, `left_side`, `right_side` view analyses to the existing
posture pipeline WITHOUT touching `posture_engine.analyze_posture_image`,
`_compute_front_measurements`, `_compute_side_measurements`, or the
current `/api/analyze-posture` request/response contract for `front` +
`side`. All new behaviour = new wrapper code; existing math is imported
and CALLED, never modified.

Public entry points
  • analyze_posture_view_multi(image_path, view) → dict
      view ∈ {"front", "side", "back", "left_side", "right_side"}
      "front" + "side" simply delegate to the existing entry so this
      function is a strict superset — safe to call from anywhere the
      old function was used.

  • analyze_posture_combined_multi(front_path, side_path,
                                   back_path, left_side_path,
                                   right_side_path) → dict
      Fields are OPTIONAL. Return dict always includes the existing
      "front" + "side" + "relative_units" keys (populated when those
      images were provided or omitted when they were not). New views
      appear ONLY when their image was provided.

Back view — HONEST 2D LIMITS
  Measurable: shoulder level (tilt°), pelvic level (tilt°), lateral
  trunk shift (% of body height), left/right knee alignment (interior
  angle).
  Explicitly `not_assessed`: forward head, thoracic kyphosis, lumbar
  lordosis, scoliosis / Cobb angle — a single flat 2D back-view frame
  cannot capture spine curvature.

Back view — L/R SWAP GUARD
  BlazePose from BEHIND is not verified in this repo (no back-view
  test photo). The claim "MediaPipe mirrors L/R from behind" is
  UNVERIFIABLE without a real image, so we ship the swap behind a
  clearly-named constant defaulted to OFF. If the swap is later
  confirmed necessary via a real photo, flip `_BACK_VIEW_SWAP_LR_LABELS`
  to True (or add an env-var override) — the metrics + keypoint
  labels will then flip together, keeping the report consistent.

Left / right explicit sides
  Reuses `posture_engine._compute_side_measurements` (which auto-picks
  the higher-confidence side) then OVERRIDES `pickedSide` to the
  declared side and re-anchors the trunk-lean sign so positive
  consistently means "anatomical forward" for the declared side.
  The auto-pick behaviour of the existing `side` key is untouched.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

# Every symbol below is IMPORTED from the existing engine. No
# original function is modified.
from engines.posture_engine import (
    _build_side_findings,
    _compute_front_measurements,
    _compute_side_measurements,
    _extract_posture_keypoints,
    _is_visible,
    _line_tilt_from_horizontal,
    _line_tilt_from_vertical,
    _load_image_rgb,
    _KP_NAMES_MOVENET,
    _POSTURE_VIS_THRESHOLD,
    _SHIFT_MILD_PCT,
    _SHIFT_OK_PCT,
    _TILT_MILD_DEG,
    _TILT_OK_DEG,
    _grade_shift,
    _grade_tilt,
    analyze_posture_image,
)

log = logging.getLogger("motionlens.posture.multi_view")


# ─── Back-view swap guard ──────────────────────────────────────
# UNVERIFIABLE without a real back-view photo (no test asset in the
# repo — see the audit note in this module's docstring). Default OFF
# — MediaPipe's raw left/right labelling is used as-is. Flip to True
# only after a real back photo confirms MediaPipe returns
# SCREEN-mirrored labels from behind.
#
# Environment override for ops: set `POSTURE_BACK_VIEW_SWAP_LR=1` to
# enable the swap at runtime without a code change (useful when a
# clinician verifies the behaviour on their setup).
_BACK_VIEW_SWAP_LR_LABELS: bool = (
    os.environ.get("POSTURE_BACK_VIEW_SWAP_LR", "0").strip().lower()
    in ("1", "true", "yes", "on")
)


# ─── View allowlist ────────────────────────────────────────────
_SUPPORTED_VIEWS = ("front", "side", "back", "left_side", "right_side")


# ─── Back-view keypoint L/R swap (label-only) ──────────────────
# BlazePose emits 17 MoveNet-indexed keypoints. When we treat the
# raw "left_*" landmarks as if they were "right_*" (and vice versa),
# we need to swap the pair indices so both the metrics AND the
# rendered overlay agree.
_MOVENET_LR_PAIRS = (
    (1, 2),    # left_eye ↔ right_eye
    (3, 4),    # left_ear ↔ right_ear
    (5, 6),    # left_shoulder ↔ right_shoulder
    (7, 8),    # left_elbow ↔ right_elbow
    (9, 10),   # left_wrist ↔ right_wrist
    (11, 12),  # left_hip ↔ right_hip
    (13, 14),  # left_knee ↔ right_knee
    (15, 16),  # left_ankle ↔ right_ankle
)


def _swap_keypoints_lr(kps: list[dict]) -> list[dict]:
    """Return a NEW list with L/R keypoint pairs swapped. Each entry
    keeps its coord + score; only the pair-index and (for label
    consistency) the `name` field are swapped. Non-paired keypoints
    (nose) pass through untouched.
    """
    swapped = list(kps)  # shallow copy first
    for i, j in _MOVENET_LR_PAIRS:
        if i < len(swapped) and j < len(swapped):
            a = dict(swapped[i])
            b = dict(swapped[j])
            # Swap the CONTENT but retain the target position's name
            # so the array remains MoveNet-indexed for the overlay.
            a_kept_name = a.get("name")
            b_kept_name = b.get("name")
            swapped[i] = {**b, "name": a_kept_name or _KP_NAMES_MOVENET[i]}
            swapped[j] = {**a, "name": b_kept_name or _KP_NAMES_MOVENET[j]}
    return swapped


# ─── Back-view analysis ────────────────────────────────────────
# Metrics: shoulder tilt, hip tilt, lateral trunk shift (% body H),
# left/right knee alignment. Everything else honestly not_assessed.

_BACK_NOT_ASSESSED = [
    {
        "label": "Forward head",
        "reason": (
            "Sagittal-plane measurement — not visible from a back-view "
            "camera. Capture a side view for this metric."
        ),
    },
    {
        "label": "Thoracic kyphosis",
        "reason": (
            "Spine curvature is not resolvable from a single flat 2D "
            "back view. Requires a sagittal camera or a 3D scan."
        ),
    },
    {
        "label": "Lumbar lordosis",
        "reason": (
            "Spine curvature is not resolvable from a single flat 2D "
            "back view. Requires a sagittal camera or a 3D scan."
        ),
    },
    {
        "label": "Scoliosis / Cobb angle",
        "reason": (
            "Requires a full radiograph or a 3D surface topography "
            "system — not derivable from a 2D pose skeleton. Lateral "
            "trunk shift is provided as a coarse frontal-plane proxy."
        ),
    },
]


def _compute_back_measurements(kps: list[dict]) -> dict:
    """Compute the honestly-computable back-view metrics. Reuses the
    existing math helpers from posture_engine by CALL only. Does NOT
    modify `_compute_front_measurements` — its shape includes several
    front-only fields (headTilt, frontalTrunkLean) that don't apply
    from behind, so we build a distinct back-view result shape.
    """
    out: dict = {
        "shoulderTilt": None,   # deg, +right-side-down (same sign as front)
        "hipTilt": None,        # deg, +left-side-down (same sign as front)
        "lateralTrunkShiftPct": None,  # (shoulder-mid.x − hip-mid.x) / bodyH
        "leftKneeAlignment": None,     # interior angle at L knee
        "rightKneeAlignment": None,
    }

    l_sh = kps[5]
    r_sh = kps[6]
    if _is_visible(l_sh) and _is_visible(r_sh):
        out["shoulderTilt"] = _line_tilt_from_horizontal(
            l_sh["x"], l_sh["y"], r_sh["x"], r_sh["y"],
        )

    l_hip = kps[11]
    r_hip = kps[12]
    if _is_visible(l_hip) and _is_visible(r_hip):
        out["hipTilt"] = _line_tilt_from_horizontal(
            r_hip["x"], r_hip["y"], l_hip["x"], l_hip["y"],
        )

    # Lateral trunk shift — horizontal offset between shoulder-mid
    # and hip-mid, normalised by body height (shoulder-mid.y to
    # ankle-mid.y). Same "% of body height" convention used by the
    # side-view forwardHeadPct / shoulderShiftPct family — keeps
    # units consistent across views.
    l_ank = kps[15]
    r_ank = kps[16]
    if (
        _is_visible(l_sh) and _is_visible(r_sh)
        and _is_visible(l_hip) and _is_visible(r_hip)
        and _is_visible(l_ank) and _is_visible(r_ank)
    ):
        sh_mid_x = (l_sh["x"] + r_sh["x"]) / 2.0
        sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
        hip_mid_x = (l_hip["x"] + r_hip["x"]) / 2.0
        an_mid_y = (l_ank["y"] + r_ank["y"]) / 2.0
        body_h = abs(an_mid_y - sh_mid_y)
        if body_h > 0:
            out["lateralTrunkShiftPct"] = (
                (sh_mid_x - hip_mid_x) / body_h * 100.0
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

    l_knee = kps[13]
    if _is_visible(l_hip) and _is_visible(l_knee) and _is_visible(l_ank):
        out["leftKneeAlignment"] = _interior_at_knee(l_hip, l_knee, l_ank)
    r_knee = kps[14]
    if _is_visible(r_hip) and _is_visible(r_knee) and _is_visible(r_ank):
        out["rightKneeAlignment"] = _interior_at_knee(r_hip, r_knee, r_ank)

    return out


def _build_back_findings(m: dict) -> list[dict]:
    """Emit the same severity-graded findings shape as the existing
    front / side findings, restricted to items back view can honestly
    score. Uses the shared _grade_tilt / _grade_shift helpers verbatim.
    """
    out: list[dict] = []
    if m.get("shoulderTilt") is not None:
        out.append(_grade_tilt(m["shoulderTilt"], "Shoulder tilt", ("left", "right")))
    if m.get("hipTilt") is not None:
        out.append(_grade_tilt(m["hipTilt"], "Hip tilt", ("left", "right")))
    if m.get("lateralTrunkShiftPct") is not None:
        # Same 3% / 7% thresholds as side-view shifts. "Positive" here
        # = shoulder-mid to the RIGHT of hip-mid in image coords; for
        # a back view without the swap flag, that maps to patient's
        # LEFT side (mirror). Grading uses magnitude only.
        v = m["lateralTrunkShiftPct"]
        abs_v = abs(v)
        # Match the direction wording used for side-view shifts, but
        # rename to laterality for the back view.
        direction = "right" if v >= 0 else "left"
        if abs_v < _SHIFT_OK_PCT:
            severity = "ok"
            detail = "Trunk is centred over the pelvis in the frontal plane."
        elif abs_v < _SHIFT_MILD_PCT:
            severity = "mild"
            detail = f"Mild lateral trunk shift toward the {direction}."
        else:
            severity = "notable"
            detail = f"Notable lateral trunk shift toward the {direction}."
        out.append({
            "label": "Lateral trunk shift",
            "value": f"{v:.1f}%",
            "severity": severity,
            "detail": detail,
        })

    # Knee alignments — grade same as front-view: dev = 180 − interior.
    for side_label, key in (
        ("Left knee alignment", "leftKneeAlignment"),
        ("Right knee alignment", "rightKneeAlignment"),
    ):
        val = m.get(key)
        if val is None:
            continue
        dev = 180.0 - val
        abs_dev = abs(dev)
        if abs_dev < 5.0:
            severity = "ok"
            detail = f"{side_label.split(' ')[0]} knee is well aligned."
        elif abs_dev < 10.0:
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

    return out


def analyze_posture_back(image_path: str) -> dict:
    """Back-view analysis wrapper. Loads the image via the shared
    loader, extracts keypoints via the shared extractor, applies the
    optional L/R swap, then scores back-view-honest metrics.

    Same failure semantics as analyze_posture_image:
      raises ValueError("invalid_image") when PIL can't read the file
      raises ValueError("poor_visibility") when trunk anchors missing.
    """
    rgb, img_w, img_h = _load_image_rgb(image_path)
    raw_kps = _extract_posture_keypoints(rgb, img_w, img_h)

    trunk_anchors = [raw_kps[5], raw_kps[6], raw_kps[11], raw_kps[12]]
    if not all(_is_visible(k) for k in trunk_anchors):
        raise ValueError("poor_visibility")

    # L/R swap gate — OFF by default (UNVERIFIABLE without a real
    # back-view photo). See module docstring + the constant comment.
    kps = _swap_keypoints_lr(raw_kps) if _BACK_VIEW_SWAP_LR_LABELS else raw_kps

    back = _compute_back_measurements(kps)
    findings = _build_back_findings(back)

    return {
        "view": "back",
        "imageWidth": img_w,
        "imageHeight": img_h,
        "keypoints": kps,
        "back": back,
        "not_assessed": _BACK_NOT_ASSESSED,
        "findings": findings,
        # Honest disclosure of whether the swap ran on this response.
        "lr_swap_applied": bool(_BACK_VIEW_SWAP_LR_LABELS),
    }


# ─── Explicit-side (left_side / right_side) analysis ───────────
def _compute_side_measurements_forced(
    kps: list[dict], forced_side: str,
) -> dict:
    """Wrap _compute_side_measurements and override pickedSide to
    the declared side. Recomputes trunk_lean sign against the forced
    side so positive still means "anatomical forward".

    The existing auto-pick function isn't modified; we CALL it,
    then override the two fields that depend on which side won the
    confidence race.
    """
    if forced_side not in ("left", "right"):
        raise ValueError(f"Unsupported forced side: {forced_side!r}")

    base = _compute_side_measurements(kps)

    # Recompute trunk_lean sign for the forced side. Uses the same
    # helpers + convention as posture_engine._compute_side_measurements.
    l_hip = kps[11]
    r_hip = kps[12]
    l_sh = kps[5]
    r_sh = kps[6]
    trunk_lean: Optional[float] = None
    if all(_is_visible(k) for k in (l_hip, r_hip, l_sh, r_sh)):
        hip_mid_x = (l_hip["x"] + r_hip["x"]) / 2.0
        hip_mid_y = (l_hip["y"] + r_hip["y"]) / 2.0
        sh_mid_x = (l_sh["x"] + r_sh["x"]) / 2.0
        sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
        raw_tilt = _line_tilt_from_vertical(
            sh_mid_x, sh_mid_y, hip_mid_x, hip_mid_y,
        )
        trunk_lean = raw_tilt if forced_side == "left" else -raw_tilt

    # Overwrite pickedSide + the trunk_lean field on the FORCED side's
    # block. The other side's block is untouched (may still be present
    # if it had visible keypoints — the report can choose to render
    # only the forced side).
    out = dict(base)
    out["pickedSide"] = forced_side
    forced_block = out.get(forced_side)
    if isinstance(forced_block, dict):
        forced_block = dict(forced_block)
        forced_block["trunkLeanDeg"] = trunk_lean
        out[forced_side] = forced_block
    # Also update the OTHER side's trunk_lean so both blocks agree on
    # the anchor (mirrors what the existing function does).
    other = "right" if forced_side == "left" else "left"
    other_block = out.get(other)
    if isinstance(other_block, dict):
        other_block = dict(other_block)
        other_block["trunkLeanDeg"] = trunk_lean
        out[other] = other_block
    return out


# ─── Facing detection + sagittal sign correction (explicit views) ──
# AUDIT FIX B1: the four shift percentages coming out of
# _compute_side_measurements are raw `landmark_x − ankle_x` — positive
# means "toward image-right", NOT "anatomically forward". Whether
# image-right IS forward depends on which way the patient faces on
# screen, and real capture photos proved the declared side is NOT a
# reliable proxy for that (mirrored live captures + free-form phone
# photos both invert the assumed mapping). So facing is derived from
# the keypoints themselves.
#
# Screen-x ↔ anterior mapping used everywhere below:
#   facing "right" (anterior = image +x) → raw positive already means
#                                          forward → keep the sign.
#   facing "left"  (anterior = image −x) → raw positive means backward
#                                          → negate all four shifts.
# Post-correction convention: FORWARD = POSITIVE, backward = negative.

# Minimum head↔ankle horizontal separation for a confident facing
# call, as a fraction of body height (shoulder→ankle vertical
# distance). In any normal standing profile the head sits well
# anterior of the ankle plumb line (typically 5–15% of body height),
# so 2% is far below every legitimate profile and only rejects
# genuinely ambiguous poses (true frontal photo, tracking noise).
_FACING_MIN_SEP_FRAC: float = 0.02

# The four sagittal shift fields the correction applies to.
_EXPLICIT_SHIFT_KEYS = (
    "forwardHeadPct",
    "shoulderShiftPct",
    "hipShiftPct",
    "kneeShiftPct",
)


def _derive_facing(kps: list[dict], declared_side: str) -> Optional[str]:
    """Derive which way the patient faces on screen from keypoint
    geometry: "left" = anterior toward image −x, "right" = anterior
    toward image +x, None = cannot tell (don't guess).

    Signal: sign(head_x − ankle_x) on the declared side. The head
    (nose preferred, declared-side ear as fallback) is always anterior
    of the ankle plumb line in a standing profile — even a patient
    with a backward-shifted trunk keeps the nose forward of the
    ankle — so its horizontal offset from the ankle gives the facing
    direction. Below the _FACING_MIN_SEP_FRAC threshold (or with the
    needed landmarks missing) return None; callers then mark the
    shifts as insufficient data instead of guessing a sign.
    """
    idx = (
        {"ear": 3, "sh": 5, "ank": 15}
        if declared_side == "left"
        else {"ear": 4, "sh": 6, "ank": 16}
    )
    sh = kps[idx["sh"]]
    ank = kps[idx["ank"]]
    if not (_is_visible(sh) and _is_visible(ank)):
        return None
    nose = kps[0]
    ear = kps[idx["ear"]]
    head = nose if _is_visible(nose) else (ear if _is_visible(ear) else None)
    if head is None:
        return None
    body_h = abs(ank["y"] - sh["y"])
    if body_h <= 0:
        return None
    dx = head["x"] - ank["x"]
    if abs(dx) < _FACING_MIN_SEP_FRAC * body_h:
        return None
    return "right" if dx > 0 else "left"


def _correct_shifts_for_facing(block: dict, facing: str) -> dict:
    """Return a copy of a per-side metric block with the four shift
    percentages re-signed so FORWARD = POSITIVE. See the mapping
    comment above: facing right keeps raw signs, facing left negates.
    trunkLeanDeg is NOT touched here — it keeps the pre-existing
    declared-side anchoring from _compute_side_measurements_forced.
    """
    out = dict(block)
    if facing == "left":
        for key in _EXPLICIT_SHIFT_KEYS:
            val = out.get(key)
            if val is not None:
                out[key] = -val
    return out


def _build_explicit_side_findings(
    side_block: dict, declared_side: str, shifts_available: bool,
) -> list[dict]:
    """AUDIT FIX B2: findings for an explicit view come from the
    DECLARED side's block ONLY — the far-side block is occluded from
    a profile camera and its (hallucinated) values previously leaked
    contradictory rows into the table. Row tags use the VIEW name so
    a right-side view never shows "(left side view)" rows.

    Severity + direction reuse `_grade_shift` BY CALL — its internal
    `direction = "forward" if value >= 0` branch is correct because it
    receives the already-corrected value (forward = positive). Only
    the label tag is overridden. The legacy `_build_side_findings` is
    not modified or called here.
    """
    out: list[dict] = []
    tag = f"({declared_side} side view)"
    block = side_block.get(declared_side)
    if isinstance(block, dict) and shifts_available:
        for key, label in (
            ("forwardHeadPct", "Head"),
            ("shoulderShiftPct", "Shoulders"),
            ("hipShiftPct", "Hips"),
            ("kneeShiftPct", "Knees"),
        ):
            val = block.get(key)
            if val is None:
                continue
            finding = _grade_shift(val, label)
            finding["label"] = f"{label} {tag}"
            out.append(finding)
    elif not shifts_available:
        out.append({
            "label": f"Sagittal shifts {tag}",
            "value": "—",
            "severity": "mild",
            "detail": (
                "Facing direction could not be determined from the "
                "keypoints, so forward/backward signs are unreliable "
                "on this photo. Retake with a clear full-profile view."
            ),
        })

    # Trunk lean — bilateral, emit once. Same 2°/5° buckets as the
    # legacy side findings (replicated here because the legacy builder
    # also emits far-side shift rows we must exclude).
    other = "right" if declared_side == "left" else "left"
    trunk = None
    if isinstance(block, dict):
        trunk = block.get("trunkLeanDeg")
    if trunk is None:
        other_block = side_block.get(other)
        if isinstance(other_block, dict):
            trunk = other_block.get("trunkLeanDeg")
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


def _analyze_explicit_from_kps(
    kps: list[dict], side: str, img_w: int, img_h: int,
) -> dict:
    """Everything after keypoint extraction for an explicit side view.
    Split out of analyze_posture_side_explicit so it can be exercised
    with synthetic keypoints (no image / model needed).
    """
    trunk_anchors = [kps[5], kps[6], kps[11], kps[12]]
    if not all(_is_visible(k) for k in trunk_anchors):
        raise ValueError("poor_visibility")

    side_block = _compute_side_measurements_forced(kps, side)

    # B1 — facing-derived sign correction on the DECLARED block only.
    facing = _derive_facing(kps, side)
    shifts_available = facing is not None
    forced_block = side_block.get(side)
    if isinstance(forced_block, dict):
        if shifts_available:
            forced_block = _correct_shifts_for_facing(forced_block, facing)
        else:
            # Can't determine the sign — publish no signed shifts
            # rather than guessing (overlay then draws no badges).
            forced_block = dict(forced_block)
            for key in _EXPLICIT_SHIFT_KEYS:
                forced_block[key] = None
        side_block = dict(side_block)
        side_block[side] = forced_block

    # Non-mirrored profile geometry: the patient's LEFT side toward
    # the camera means the patient faces image-LEFT (and vice versa).
    # (An older comment claimed the opposite; real capture photos
    # contradicted it — see audit.) A mismatch usually means the
    # capture was mirrored or the operator declared the wrong side —
    # keypoints win, but the caveat surfaces it.
    view_key = "left_side" if side == "left" else "right_side"
    facing_caveat: Optional[str] = None
    if facing is not None and facing != side:
        facing_caveat = (
            f"Declared {view_key.replace('_', '-')} but the image "
            f"appears {facing}-facing — check the capture (mirroring "
            "or side mix-up). Shift signs follow the keypoints."
        )

    # B2 — findings from the declared block only, view-name tags.
    findings = _build_explicit_side_findings(side_block, side, shifts_available)

    return {
        "view": view_key,
        "imageWidth": img_w,
        "imageHeight": img_h,
        "keypoints": kps,
        "side": side_block,
        "findings": findings,
        "explicit_side": side,
        # Additive disclosure fields (frontend may ignore them).
        "facing": facing,
        "facing_caveat": facing_caveat,
        "shifts_insufficient_data": not shifts_available,
    }


def analyze_posture_side_explicit(image_path: str, side: str) -> dict:
    """Left/right explicit-side wrapper. Same shape as
    analyze_posture_image(view="side") but pickedSide is FORCED to
    the declared side instead of being auto-picked, sagittal shift
    signs are corrected to keypoint-derived facing (forward =
    positive), and findings come from the declared block only.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported explicit side: {side!r}")

    rgb, img_w, img_h = _load_image_rgb(image_path)
    kps = _extract_posture_keypoints(rgb, img_w, img_h)
    return _analyze_explicit_from_kps(kps, side, img_w, img_h)


def apply_side_facing_correction(side_result: dict) -> dict:
    """Post-process the LEGACY auto-pick `side` view result (the
    dict `analyze_posture_image(path, "side")` returns) so its
    pickedSide block + findings use the same facing-corrected
    FORWARD = POSITIVE convention as the explicit views.

    Additive by design: posture_engine.py stays byte-identical — the
    endpoint calls this on the already-built result. The auto-picked
    side plays the role the declared side plays for explicit views;
    facing itself still comes from keypoint geometry (_derive_facing),
    so mirrored captures and either facing direction are handled.

    Never raises; if the result is missing the pieces we need
    (per-view error dict, no pickedSide, no keypoints) it is
    returned unchanged.
    """
    if not isinstance(side_result, dict) or "error" in side_result:
        return side_result
    kps = side_result.get("keypoints")
    side_block = side_result.get("side")
    if not isinstance(kps, list) or not isinstance(side_block, dict):
        return side_result
    picked = side_block.get("pickedSide")
    if picked not in ("left", "right"):
        return side_result

    facing = _derive_facing(kps, picked)
    shifts_available = facing is not None
    picked_block = side_block.get(picked)

    new_side_block = dict(side_block)
    if isinstance(picked_block, dict):
        if shifts_available:
            picked_block = _correct_shifts_for_facing(picked_block, facing)
        else:
            picked_block = dict(picked_block)
            for key in _EXPLICIT_SHIFT_KEYS:
                picked_block[key] = None
        new_side_block[picked] = picked_block

    out = dict(side_result)
    out["side"] = new_side_block
    out["findings"] = _build_explicit_side_findings(
        new_side_block, picked, shifts_available,
    )
    out["facing"] = facing
    out["shifts_insufficient_data"] = not shifts_available
    return out


# ─── Multi-view combined entry ─────────────────────────────────
def analyze_posture_combined_multi(
    front_image_path: Optional[str] = None,
    side_image_path: Optional[str] = None,
    back_image_path: Optional[str] = None,
    left_side_image_path: Optional[str] = None,
    right_side_image_path: Optional[str] = None,
) -> dict:
    """Combined multi-view response. Populates only the views whose
    image path was provided. `relative_units: true` preserved.

    Each new-view analysis is wrapped so that a failure on one view
    (e.g. poor_visibility) does NOT break the others — the failing
    view lands in the response as
      {"view": "<key>", "error": "<code>"}
    while the successful views keep their full shape.
    """
    out: dict = {"relative_units": True}
    if front_image_path:
        out["front"] = analyze_posture_image(front_image_path, "front")
    if side_image_path:
        out["side"] = analyze_posture_image(side_image_path, "side")
    if back_image_path:
        out["back"] = _safe_view("back", analyze_posture_back, back_image_path)
    if left_side_image_path:
        out["left_side"] = _safe_view(
            "left_side", analyze_posture_side_explicit,
            left_side_image_path, "left",
        )
    if right_side_image_path:
        out["right_side"] = _safe_view(
            "right_side", analyze_posture_side_explicit,
            right_side_image_path, "right",
        )
    return out


def _safe_view(view_key: str, fn, *args) -> dict:
    """Run `fn(*args)`; on ValueError, return a per-view error dict
    so that one failing view doesn't wipe out the others in the
    combined response. Unexpected exceptions are logged and surface
    the same way — never re-raised into the endpoint (front/side keep
    their historical exception path for backward compatibility).
    """
    try:
        return fn(*args)
    except ValueError as e:
        return {"view": view_key, "error": str(e)}
    except Exception as e:  # pragma: no cover — defensive
        log.exception("posture multi-view %s failed: %s", view_key, e)
        return {"view": view_key, "error": "analysis_failed"}


__all__ = (
    "analyze_posture_back",
    "analyze_posture_side_explicit",
    "analyze_posture_combined_multi",
    "apply_side_facing_correction",
    "_BACK_VIEW_SWAP_LR_LABELS",
    "_SUPPORTED_VIEWS",
)
