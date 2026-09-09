"""Landmark-quality weights for video-derived metrics.

Every metric in the gait report is an aggregate over frames or over
strides. Until now each frame counted 1.0 regardless of how well the
joints behind it were tracked -- including frames whose landmark had
dropped below the extraction floor and been INTERPOLATED back in by
_interp_nans (visibility stored as 0.0, position invented). This module
turns per-frame visibility into a weight so an aggregate can discount
poorly-seen frames instead of trusting them equally.

    q(frame) = min visibility over the joints the metric reads
    w        = 0                       q <  QUALITY_FLOOR   (interpolated / lost)
             = (q - floor)/(full - floor)   floor <= q < full  (seen, unreliably)
             = 1                       q >= QUALITY_FULL    (seen clearly)

Design rules the callers follow:
  * MEAN-type metrics use weighted_mean (np.average).
  * PEAK / MIN / ROM use weighted_extreme -- taken ONLY over frames with
    w == 1.0. A maximum is a single frame; averaging cannot rescue it, so
    the only honest option is to refuse frames that were not clearly
    seen. None when no such frame exists; the caller decides whether to
    degrade to w > 0 and flag it.
  * EVENT-based metrics (heel strikes) weight each stride by the mean
    w across it, and by 0 if the strike frame itself was interpolated --
    that event was manufactured by the interpolation, not observed.
  * COUNTS are never weighted: a fractional step count is meaningless.

The thresholds are the same 0.4 / 0.7 used by the reliability tiers.
They have NOT been validated against a real far-side leg -- run the
scratchpad gait_vis_distribution.py on real clips before trusting them.

Pure numpy; no MediaPipe, no engine imports, so both gait and biomech
can call it against the same {joint: {..., "vis": array}} time-series.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

QUALITY_FLOOR: float = 0.4   # below: weight 0 (extraction floor / interpolated)
QUALITY_FULL: float = 0.7    # at/above: weight 1 (clearly seen)


def frame_weights(
    vis: dict[str, np.ndarray],
    joints: list[str],
    floor: float = QUALITY_FLOOR,
    full: float = QUALITY_FULL,
) -> np.ndarray:
    """Per-frame weight for a metric that reads `joints`.

    `vis` is a mapping joint-name -> per-frame visibility array (the
    engine's ts[name]["vis"]). The frame's quality is the MIN over the
    metric's joints -- one lost landmark makes the whole angle unreliable
    -- then mapped linearly onto [0, 1] between `floor` and `full`.
    Interpolated frames carry visibility 0.0 and therefore weight 0.
    """
    if not joints:
        raise ValueError("frame_weights: joints must be non-empty")
    arrs = [np.asarray(vis[j], dtype=float) for j in joints]
    n = min(a.shape[0] for a in arrs)
    q = np.min(np.vstack([a[:n] for a in arrs]), axis=0)
    span = float(full - floor)
    if span <= 0:
        return (q >= full).astype(float)
    return np.clip((q - floor) / span, 0.0, 1.0)


def stride_weights(w: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """One weight per consecutive strike pair (hs[k], hs[k+1]).

    Mean of w over the stride's frames, except 0 when the strike frame
    itself has w == 0: a heel-strike detected on an interpolated heel is
    an artefact of the interpolation, not an observation. Length is
    len(strikes) - 1, aligned with np.diff(strikes).
    """
    w = np.asarray(w, dtype=float)
    hs = np.asarray(strikes, dtype=int)
    if hs.size < 2:
        return np.zeros(0, dtype=float)
    out = np.zeros(hs.size - 1, dtype=float)
    n = w.shape[0]
    for k in range(hs.size - 1):
        a, b = int(hs[k]), int(hs[k + 1])
        if a < 0 or a >= n or b <= a:
            continue
        if w[a] <= 0.0:
            continue                      # manufactured strike -> 0
        seg = w[a:min(b, n)]
        out[k] = float(np.mean(seg)) if seg.size else 0.0
    return out


def weighted_extreme(
    x: np.ndarray, w: np.ndarray, fn: Callable = np.nanmax,
) -> Optional[float]:
    """`fn` over the samples with w == 1.0 (NaNs dropped); None if none."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    n = min(x.shape[0], w.shape[0])
    if n == 0:
        return None
    sel = x[:n][w[:n] >= 1.0]
    sel = sel[~np.isnan(sel)]
    if sel.size == 0:
        return None
    return float(fn(sel))


def weighted_mean(x: np.ndarray, w: np.ndarray) -> Optional[float]:
    """np.average over non-NaN samples; None when the weights sum to 0."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    n = min(x.shape[0], w.shape[0])
    if n == 0:
        return None
    x, w = x[:n], w[:n]
    ok = ~np.isnan(x)
    if not ok.any():
        return None
    ws = float(w[ok].sum())
    if ws <= 0.0:
        return None
    return float(np.average(x[ok], weights=w[ok]))


def weighted_std(x: np.ndarray, w: np.ndarray) -> Optional[float]:
    """Weighted population standard deviation
    (var = sum w (x-m)^2 / sum w). With all weights equal this is exactly
    np.std (ddof=0), which is what the un-weighted metrics used, so a
    fully-visible clip reproduces the old numbers. None when undefined."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    n = min(x.shape[0], w.shape[0])
    if n == 0:
        return None
    x, w = x[:n], w[:n]
    ok = ~np.isnan(x) & (w > 0)
    if not ok.any():
        return None
    x, w = x[ok], w[ok]
    v1 = float(w.sum())
    if v1 <= 0.0:
        return None
    m = float(np.average(x, weights=w))
    return float(np.sqrt(float((w * (x - m) ** 2).sum()) / v1))


__all__ = (
    "QUALITY_FLOOR",
    "QUALITY_FULL",
    "frame_weights",
    "stride_weights",
    "weighted_extreme",
    "weighted_mean",
    "weighted_std",
)
