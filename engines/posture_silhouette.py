"""Body-relative reference geometry from the BlazePose segmentation mask.

OVERLAY ONLY. Nothing in this module feeds a posture metric, a grading
threshold, or a finding. `posture_engine` attaches the output under a
new optional `silhouette` key per view; every existing number is
computed from landmarks exactly as before and is bit-identical whether
or not a mask was available.

Why a mask at all: the drawn reference lines used to be anchored to
landmarks and spanned the whole frame -- the front plumb sat on the
shoulder midpoint (a point no front metric is actually measured
against) and the horizontals ran edge to edge. Both read as "the grid",
not "this patient". The silhouette gives the real body centre and the
real body width at each landmark height, so the lines sit on the person.

Coordinate space: image pixels, identical to the keypoint array and to
the frontend canvas (which is sized to the source image). Everything
returned is plain float/int/list so it serialises straight into the
report `metrics` blob.

Degradation is always silent and always to None: a missing mask, a
person half out of frame, a mask that disagrees with the landmarks --
each returns None for that piece of geometry, and the overlay falls
back to its previous landmark-anchored line.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("motionlens.posture.silhouette")

# MediaPipe Tasks hands back a float32 confidence map in [0, 1] -- NOT
# the 0/255 RED-channel bitmap the browser Solutions API produces. The
# two are not interchangeable, so this threshold is not portable to the
# frontend.
MASK_THRESHOLD: float = 0.5

# Every Nth scanned row is kept for the drawn centreline. Over ~600 rows
# of body that is ~100 points: enough to read the curve, small enough
# that five views of it do not bloat a saved report.
CENTERLINE_ROW_STRIDE: int = 6

# A row's body run may be at most this multiple of the landmark-derived
# width before it is treated as contaminated (arms merging into the
# torso silhouette) and clamped around the seed. Same slack the Rehab
# spine work settled on.
RUN_WIDTH_SLACK: float = 1.7

# If more than half the scanned rows are unusable, the mask and the
# landmarks disagree badly enough that the midline is not trustworthy.
MIN_ACCEPTED_ROW_FRAC: float = 0.5

# MoveNet-layout indices -- the posture keypoint array's own convention.
_L_EAR, _R_EAR = 3, 4
_L_SH, _R_SH = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANK, _R_ANK = 15, 16

# Matches posture_engine._POSTURE_VIS_THRESHOLD. Duplicated rather than
# imported so this module stays free of a circular import back into the
# engine that calls it.
_VIS_FLOOR = 0.2


# --- Small helpers ---------------------------------------------
def _vis(kp: Optional[dict]) -> bool:
    return bool(kp) and (kp.get("score") or 0.0) >= _VIS_FLOOR


def _mid(a: dict, b: dict) -> tuple[float, float]:
    return ((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp01(t: float) -> float:
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def _binary_row(mask: np.ndarray, y: int) -> Optional[np.ndarray]:
    if y < 0 or y >= mask.shape[0]:
        return None
    return mask[y] >= MASK_THRESHOLD


def _run_bounds(row: np.ndarray, seed_x: int) -> Optional[tuple[int, int]]:
    """Inclusive [xL, xR] of the body run containing `seed_x`.

    Walks outward from the seed to that run's own edges. Vectorised via
    the transition indices rather than a Python while-loop, because a
    720-row scan across a 1280-wide mask is ~400k steps otherwise.
    """
    w = row.shape[0]
    if seed_x < 0 or seed_x >= w or not bool(row[seed_x]):
        return None
    # diff[i] is non-zero exactly where row[i] != row[i + 1], so a run
    # starting at s shows up as a change at s - 1 and one ending at e
    # shows up as a change at e.
    changes = np.flatnonzero(np.diff(row.astype(np.int8)))
    before = changes[changes < seed_x]
    after = changes[changes >= seed_x]
    x_left = int(before[-1]) + 1 if before.size else 0
    x_right = int(after[0]) if after.size else w - 1
    return x_left, x_right


# --- (b) Public: one row's body extent -------------------------
def row_extent(
    mask: np.ndarray, y: float, seed_x: float,
) -> Optional[list[float]]:
    """[x_left, x_right] of the body run crossing `seed_x` at row `y`.

    Deliberately UNCAPPED -- this sets how long a drawn horizontal is,
    not what anything measures, so an arm merged into the torso making
    the row wider is the honest answer for a line length.
    """
    row = _binary_row(mask, int(round(y)))
    if row is None:
        return None
    bounds = _run_bounds(row, int(round(seed_x)))
    if bounds is None:
        return None
    return [float(bounds[0]), float(bounds[1])]


# --- (a) Public: body midline ----------------------------------
def body_midline_x(
    mask: np.ndarray, kps: list[dict],
) -> Optional[tuple[float, list[list[float]]]]:
    """(midline_x, centerline) from the silhouette, or None.

    Scans every row from shoulder-mid height to ankle-mid height. Each
    row is seeded at the landmark-interpolated body axis and the run
    CONTAINING that seed is taken -- never the row's outer edges. An
    edge-to-edge scan averages `arm -> torso -> arm` and collapses onto
    the frame's own centre, which looks plausible and is wrong;
    anchoring to the axis is what makes the result mean "centre of the
    body".

    midline_x is the MEDIAN of the accepted row centres, so a handful of
    rows where a hand overlaps the hip cannot drag the line the way a
    mean would allow.
    """
    need = (kps[_L_SH], kps[_R_SH], kps[_L_HIP], kps[_R_HIP],
            kps[_L_ANK], kps[_R_ANK])
    if not all(_vis(k) for k in need):
        return None

    sh_x, sh_y = _mid(kps[_L_SH], kps[_R_SH])
    hip_x, hip_y = _mid(kps[_L_HIP], kps[_R_HIP])
    ank_x, ank_y = _mid(kps[_L_ANK], kps[_R_ANK])

    h, w = mask.shape[0], mask.shape[1]
    y_top = max(0, int(round(min(sh_y, ank_y))))
    y_bot = min(h - 1, int(round(max(sh_y, ank_y))))
    if y_bot - y_top < 8:
        return None

    shoulder_w = abs(kps[_L_SH]["x"] - kps[_R_SH]["x"])
    hip_w = abs(kps[_L_HIP]["x"] - kps[_R_HIP]["x"])

    l_ank_x = float(kps[_L_ANK]["x"])
    r_ank_x = float(kps[_R_ANK]["x"])

    centres: list[float] = []          # midline voters (primary seed only)
    centerline: list[list[float]] = []  # everything, for display
    scanned = 0
    accepted = 0
    last_centre: Optional[float] = None

    for y in range(y_top, y_bot + 1):
        scanned += 1
        # Seed on the landmark axis, piecewise: shoulder -> hip through
        # the trunk, hip -> ankle down the legs. A single shoulder ->
        # ankle lerp drifts off the body through the pelvis whenever the
        # patient leans.
        if hip_y > sh_y and y <= hip_y:
            t = _clamp01((y - sh_y) / (hip_y - sh_y))
            seed = _lerp(sh_x, hip_x, t)
            w_ref = _lerp(shoulder_w, hip_w, t)
        else:
            denom = (ank_y - hip_y) or 1.0
            t = _clamp01((y - hip_y) / denom)
            seed = _lerp(hip_x, ank_x, t)
            w_ref = hip_w

        row = _binary_row(mask, y)
        if row is None:
            continue
        bounds = _run_bounds(row, int(round(seed)))

        # FEET APART. Below the hips the seed follows the ankle MIDPOINT,
        # which lands in the gap between the legs as soon as the patient
        # stands with any stance width -- every such row was skipped, and
        # enough of them tripped the >50% guard and killed the midline
        # outright. Retry once at the nearer ankle: "nearer" is measured
        # against the last accepted centre so the scan keeps following
        # the same leg down instead of alternating between them.
        via_retry = False
        if bounds is None and y > hip_y:
            ref = last_centre if last_centre is not None else l_ank_x
            near_ank_x = min(
                (l_ank_x, r_ank_x), key=lambda ax: abs(ax - ref),
            )
            t_leg = _clamp01((y - hip_y) / ((ank_y - hip_y) or 1.0))
            retry_seed = _lerp(hip_x, near_ank_x, t_leg)
            bounds = _run_bounds(row, int(round(retry_seed)))
            if bounds is not None:
                seed = retry_seed         # the cap below re-centres here
                via_retry = True

        if bounds is None:
            continue                      # seed sits off the silhouette
        x_left, x_right = float(bounds[0]), float(bounds[1])

        # Contaminated row (an arm merged into the torso run): keep the
        # measurement but clamp it symmetrically around the seed, so one
        # wide arm cannot pull the centre sideways.
        cap = max(4.0, w_ref * RUN_WIDTH_SLACK)
        if (x_right - x_left) > cap:
            x_left = max(x_left, seed - cap / 2.0)
            x_right = min(x_right, seed + cap / 2.0)
            if x_right <= x_left:
                continue

        cx = (x_left + x_right) / 2.0
        accepted += 1
        last_centre = cx

        # A retry row measured ONE LEG, not the body -- its centre is
        # half a stance width off the trunk axis. It counts as accepted
        # (the scan is alive, the guard should not fire) and it is drawn
        # on the centreline, but it must not vote on midline_x: with
        # legs typically two thirds of the scanned height, letting them
        # vote drags the median onto whichever leg the scan followed.
        # Measured on synthetic stances, feeding retry rows into the
        # median put the midline 23-59 px off centre as stance widened
        # -- a confidently wrong plumb, worse than the None it replaced.
        if not via_retry:
            centres.append(cx)
        if (y - y_top) % CENTERLINE_ROW_STRIDE == 0:
            centerline.append([round(cx, 2), float(y)])

    if not centres or scanned == 0:
        return None
    if accepted < MIN_ACCEPTED_ROW_FRAC * scanned:
        return None                       # mask and landmarks disagree
    midline = float(np.median(np.asarray(centres, dtype=np.float64)))
    if not (0.0 <= midline < w):
        return None
    return midline, centerline


# --- (c) Public: side-view plumb -------------------------------
def plumb_x_side(mask: np.ndarray, kps: list[dict]) -> Optional[float]:
    """Centre of the foot silhouette at ankle-mid height, or None.

    In profile the ankle LANDMARK sits inside the leg while the foot
    extends forward of it, so a plumb dropped on the landmark hangs off
    the front of the foot. Taking the run at that row puts the line
    between the posterior and anterior edges of the foot, which is where
    a clinician would drop a physical plumb line.

    The side-view percentage metrics keep using the landmark ankle-mid
    (posture_engine._compute_one_side) -- this moves the DRAWN line only.
    """
    if _vis(kps[_L_ANK]) and _vis(kps[_R_ANK]):
        ank_x, ank_y = _mid(kps[_L_ANK], kps[_R_ANK])
    elif _vis(kps[_L_ANK]):
        ank_x, ank_y = kps[_L_ANK]["x"], kps[_L_ANK]["y"]
    elif _vis(kps[_R_ANK]):
        ank_x, ank_y = kps[_R_ANK]["x"], kps[_R_ANK]["y"]
    else:
        return None
    ext = row_extent(mask, ank_y, ank_x)
    if ext is None:
        return None
    return float((ext[0] + ext[1]) / 2.0)


# --- Assembly --------------------------------------------------
_EXTENT_PAIRS: tuple[tuple[str, int, int], ...] = (
    ("ear", _L_EAR, _R_EAR),
    ("shoulder", _L_SH, _R_SH),
    ("hip", _L_HIP, _R_HIP),
    ("knee", _L_KNEE, _R_KNEE),
    ("ankle", _L_ANK, _R_ANK),
)


def build_silhouette(
    mask: Optional[np.ndarray], kps: list[dict],
) -> Optional[dict]:
    """The whole `silhouette` block for one view, or None.

    None whenever the mask is absent or unusable -- callers attach the
    key only when this returns a dict, so a report saved without one is
    indistinguishable from a report saved before this existed, and the
    overlay's fallback path handles both identically.
    """
    if mask is None or getattr(mask, "ndim", 0) != 2:
        return None

    out: dict = {
        "midline_x": None,
        "centerline": [],
        "extents": {},
        "plumb_x": None,
    }

    mid = body_midline_x(mask, kps)
    if mid is not None:
        out["midline_x"] = round(mid[0], 2)
        out["centerline"] = mid[1]

    for name, li, ri in _EXTENT_PAIRS:
        a, b = kps[li], kps[ri]
        if _vis(a) and _vis(b):
            seed_x, row_y = _mid(a, b)
        elif _vis(a):
            seed_x, row_y = a["x"], a["y"]
        elif _vis(b):
            seed_x, row_y = b["x"], b["y"]
        else:
            continue
        ext = row_extent(mask, row_y, seed_x)
        if ext is not None:
            out["extents"][name] = [round(ext[0], 2), round(ext[1], 2)]

    plumb = plumb_x_side(mask, kps)
    if plumb is not None:
        out["plumb_x"] = round(plumb, 2)

    # Nothing usable came back -- say so with None rather than shipping
    # an empty block the frontend would have to special-case.
    if (out["midline_x"] is None and not out["extents"]
            and out["plumb_x"] is None):
        return None
    return out


def build_silhouette_logged(
    mask: Optional[np.ndarray], kps: list[dict], view: str,
) -> Optional[dict]:
    """build_silhouette + the one-line diagnostic, returning the block.

    Split out from attach_silhouette because the MEASUREMENTS now need
    plumb_x, so the silhouette has to exist before the metric block is
    computed rather than being bolted onto the finished result.
    """
    try:
        sil = build_silhouette(mask, kps)
    except Exception:  # pragma: no cover — defensive
        log.warning("posture: silhouette build failed for %s",
                    view, exc_info=True)
        return None
    if sil is None:
        log.info(
            "posture: silhouette None for %s — mask=%s", view,
            "absent" if mask is None
            else f"shape={mask.shape} ndim={mask.ndim}",
        )
    return sil


def plumb_reference_x(silhouette: Optional[dict]) -> Optional[float]:
    """The x the side overlay DRAWS its plumb at, or None.

    The single definition of "the reference", so the measurement and the
    drawn line cannot diverge — they used to: the line sat on the foot
    silhouette while the numbers measured from the ankle landmark buried
    inside the leg, roughly half a foot length apart.
    """
    if not silhouette:
        return None
    px = silhouette.get("plumb_x")
    return float(px) if px is not None else None


def attach_silhouette(
    out: dict, mask: Optional[np.ndarray], kps: list[dict], view: str,
) -> None:
    """Attach the silhouette block to a view result, in place, or log
    why it could not.

    The single attach point for every view. Best-effort by contract: a
    missing mask, geometry the mask and the landmarks disagree on, or an
    outright exception all leave the key absent — which is what a
    response looked like before the silhouette existed, and what the
    frontend's fallback path already handles. A view is never failed
    over overlay decoration.

    ⚠️ That silence is why a total outage went unnoticed in prod: on
    mediapipe 1.0.0 every view returned None and the response was simply
    missing a key, indistinguishable from "this photo had no usable
    mask". ONE line per view (never per row) now records the mask's
    actual shape whenever the block comes back empty, which turns that
    class of bug into a log read instead of a container autopsy.
    """
    try:
        sil = build_silhouette(mask, kps)
    except Exception:  # pragma: no cover — defensive
        log.warning("posture: silhouette build failed for %s",
                    view, exc_info=True)
        return
    if sil is None:
        log.info(
            "posture: silhouette None for %s — mask=%s", view,
            "absent" if mask is None
            else f"shape={mask.shape} ndim={mask.ndim}",
        )
        return
    out["silhouette"] = sil


__all__ = (
    "MASK_THRESHOLD",
    "attach_silhouette",
    "body_midline_x",
    "build_silhouette_logged",
    "plumb_reference_x",
    "build_silhouette",
    "plumb_x_side",
    "row_extent",
)
