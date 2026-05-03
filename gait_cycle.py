"""
gait_cycle.py
Pure (no Streamlit) helpers for cycle-normalized gait analysis.

Pipeline:
    detect_heel_strikes  → integer frame indices of foot-down events
    extract_cycles       → resample each consecutive HS pair to N points
                           (optional clean-mask filter so cycles that cross a
                           turning / accel / decel region are dropped)
    stride_durations     → per-cycle frame count, K-aligned with extract_cycles
    filter_cycles        → drop duration outliers (MAD) and cycles whose
                           NaN-fraction exceeds max_nan_fraction
    ensemble_statistics  → mean ± SD curves across kept cycles, plus K count
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ──────────────────────────────────────────────
# 1. HEEL-STRIKE DETECTION
# ──────────────────────────────────────────────
def _amplitude_filter(smoothed: np.ndarray, hs: np.ndarray, fps: float,
                      min_amp_ratio: float) -> tuple[np.ndarray, int]:
    """
    Reject strikes whose peak-to-trough amplitude is below
    `min_amp_ratio × median amplitude` of all detected strikes.

    Self-calibrating per leg: a healthy walker with uniform clearance keeps
    every strike (all amplitudes near the median, ratio ≈ 1.0). A dragging /
    shuffling leg with intermittent flat heel-y signals will have a few
    strikes with much smaller amplitudes — those get dropped.

    For each strike i, amplitude is computed as
        smoothed[strike_i] − min(smoothed[strike_i − lookback : strike_i])
    where lookback = int(fps * 0.3) frames.
    """
    if hs is None or len(hs) == 0:
        return hs, 0

    lookback = max(int(fps * 0.3), 3)
    amps = np.zeros(len(hs), dtype=float)
    for i, s in enumerate(hs):
        s = int(s)
        if s <= 0:
            amps[i] = np.nan
            continue
        lo = max(0, s - lookback)
        preceding = smoothed[lo:s]
        if len(preceding) == 0 or np.all(np.isnan(preceding)):
            amps[i] = np.nan
            continue
        trough = float(np.nanmin(preceding))
        peak   = float(smoothed[s])
        amps[i] = peak - trough

    valid = amps[~np.isnan(amps)]
    if len(valid) == 0:
        return hs, 0
    median_amp = float(np.median(valid))
    if median_amp <= 0:
        return hs, 0
    keep = amps >= (min_amp_ratio * median_amp)
    n_rejected = int((~keep).sum())
    return hs[keep], n_rejected


def detect_heel_strikes(heel_y_signal: np.ndarray, fps: float,
                        expected_cadence_spm: float = 110,
                        min_amp_ratio: float = 0.35,
                        return_metadata: bool = False):
    """
    Detect heel-strike frame indices from a per-frame heel vertical position
    signal (image coordinates: larger y = lower = foot planted).

    Pipeline:
      1. Savitzky-Golay smooth (window ≈ fps/3, polyorder=3).
      2. find_peaks with
            distance     = 0.7 × expected_stride_period (frames)
            height       = median(smoothed)         # rejects swing dips
            prominence   = 15                       # rejects jitter (pixels)
      3. POST-DETECTION AMPLITUDE FILTER (new):
            Reject strikes whose peak-to-trough amplitude < min_amp_ratio
            × median amplitude across the leg's strikes. Self-calibrating
            per leg — drops shuffling/foot-drag events whose heel-y barely
            rises above the surrounding signal.

    Args:
        heel_y_signal:        per-frame heel y (pixel coords, y-down).
        fps:                  video frame rate.
        expected_cadence_spm: assumed total cadence; sets min stride distance.
        min_amp_ratio:        amplitude floor as a ratio of the leg's
                              median amplitude. 0.5 keeps strikes that
                              are at least half as crisp as the median.
        return_metadata:      if True, returns (hs, dict) where dict has
                              accepted / rejected / initial / min_amp_ratio.

    Returns:
        np.ndarray of frame indices (default), OR (np.ndarray, dict)
        when return_metadata=True.
    """
    def _ret(arr, n_initial=0, n_rejected=0):
        if not return_metadata:
            return arr
        return arr, {
            "accepted":      int(len(arr)),
            "rejected":      int(n_rejected),
            "initial":       int(n_initial),
            "min_amp_ratio": float(min_amp_ratio),
        }

    n = len(heel_y_signal)
    if n < 5:
        return _ret(np.array([], dtype=int))

    sg_win = int(fps // 3)
    if sg_win % 2 == 0:
        sg_win += 1
    if sg_win < 5:
        sg_win = 5
    if sg_win > n:
        sg_win = n if n % 2 == 1 else n - 1
    if sg_win < 5:
        return _ret(np.array([], dtype=int))

    smoothed = savgol_filter(heel_y_signal, sg_win, polyorder=3)

    # Expected per-leg stride period in frames.
    # cadence_spm is total steps/min (both legs); per leg = cadence_spm / 2.
    expected_stride_fr = (60.0 / max(expected_cadence_spm / 2.0, 1.0)) * fps
    min_dist = max(int(expected_stride_fr * 0.7), 3)

    height_thresh = float(np.nanmedian(smoothed))

    hs, _ = find_peaks(
        smoothed,
        distance=min_dist,
        height=height_thresh,
        prominence=15,
    )
    hs = hs.astype(int)
    n_initial = int(len(hs))

    # NEW: amplitude filter on top of (distance, height, prominence).
    hs, n_rejected = _amplitude_filter(smoothed, hs, fps, min_amp_ratio)

    return _ret(hs, n_initial=n_initial, n_rejected=n_rejected)


# ──────────────────────────────────────────────
# 2. PAIR-LEVEL FILTER (internal helper, used by extract_cycles + stride_durations)
# ──────────────────────────────────────────────
def _valid_pair_indices(heel_strikes: np.ndarray, n_signal: int,
                        clean_mask: np.ndarray | None = None,
                        min_len: int = 5) -> list[int]:
    """Return list of k such that pair (hs[k], hs[k+1]) is a valid cycle."""
    valid = []
    if heel_strikes is None or len(heel_strikes) < 2:
        return valid
    for k in range(len(heel_strikes) - 1):
        a, b = int(heel_strikes[k]), int(heel_strikes[k + 1])
        if a < 0 or b > n_signal:
            continue
        if b - a < min_len:
            continue
        if clean_mask is not None and not clean_mask[a:b].all():
            continue
        valid.append(k)
    return valid


# ──────────────────────────────────────────────
# 3. EXTRACT CYCLES (resample to N points)
# ──────────────────────────────────────────────
def extract_cycles(signal: np.ndarray, heel_strikes: np.ndarray,
                   n_points: int = 101,
                   clean_mask: np.ndarray | None = None,
                   max_dur_ratio: float = 1.5,
                   min_dur_ratio: float = 0.7,
                   return_metadata: bool = False):
    """
    For each consecutive heel-strike pair (hs[k], hs[k+1]), slice the signal
    and resample to exactly n_points via linear interpolation.

    Skips:
      • pairs shorter than 5 frames
      • pairs whose [hs[k], hs[k+1]) range is not fully inside `clean_mask`
        (only applied when clean_mask is provided)
      • DURATION VALIDATION (additive on top of clean_mask):
          duration > max_dur_ratio × median(diff(heel_strikes))   → too long
          duration < min_dur_ratio × median(diff(heel_strikes))   → too short
        Median is taken across ALL adjacent heel-strike intervals (robust
        to a few cross-turn-gap outliers). Catches multi-stride spans that
        slip through clean_mask when intermediate strikes were missed, and
        catches double-detection events too close together.

    Returns:
        np.ndarray of shape (K, n_points), or empty (0, n_points).

        If return_metadata=True, returns (cycles, dict) with:
            total            — number of clean_mask-passing pairs considered
            kept             — number of cycles in returned array
            rejected_long    — duration > max_dur_ratio × median
            rejected_short   — duration < min_dur_ratio × median
            median_duration  — frames; np.nan if not computable
            kept_mask        — bool ndarray of shape (total,); True where
                               the corresponding pair survived. Same K-order
                               as `stride_durations(...)`, so callers can
                               slice `sd[kept_mask]` to stay K-aligned.
    """
    def _ret(arr, total=0, kept=0, long=0, short=0,
             median_dur=float("nan"), kept_mask=None):
        if not return_metadata:
            return arr
        if kept_mask is None:
            kept_mask = np.zeros(0, dtype=bool)
        return arr, {
            "total":           int(total),
            "kept":            int(kept),
            "rejected_long":   int(long),
            "rejected_short":  int(short),
            "median_duration": float(median_dur),
            "kept_mask":       kept_mask,
        }

    if signal is None or len(signal) == 0 or heel_strikes is None or len(heel_strikes) < 2:
        return _ret(np.empty((0, n_points)))

    n_signal = len(signal)
    valid_ks = _valid_pair_indices(heel_strikes, n_signal, clean_mask=clean_mask)
    total_pairs = len(valid_ks)
    if not valid_ks:
        return _ret(np.empty((0, n_points)), total=0)

    # Median stride duration across ALL adjacent intervals — robust enough
    # that a handful of cross-turn-gap outliers won't poison the threshold.
    diffs = np.diff(np.asarray(heel_strikes, dtype=int))
    median_dur = float(np.median(diffs)) if len(diffs) > 0 else float("nan")

    tau_star = np.linspace(0, 1, n_points)
    cycles: list[np.ndarray] = []
    kept_mask = np.zeros(total_pairs, dtype=bool)
    rejected_long = 0
    rejected_short = 0

    for i, k in enumerate(valid_ks):
        a, b = int(heel_strikes[k]), int(heel_strikes[k + 1])
        duration = b - a

        if not np.isnan(median_dur) and median_dur > 0:
            if duration > max_dur_ratio * median_dur:
                rejected_long += 1
                continue
            if duration < min_dur_ratio * median_dur:
                rejected_short += 1
                continue

        seg = np.asarray(signal[a:b], dtype=float)
        valid = ~np.isnan(seg)
        if valid.sum() < 2:
            cycles.append(np.full(n_points, np.nan))
            kept_mask[i] = True
            continue
        tau = np.linspace(0, 1, len(seg))
        if valid.all():
            cycles.append(np.interp(tau_star, tau, seg))
        else:
            # NaN-aware interpolation: skip NaN samples; np.interp clamps at edges.
            cycles.append(np.interp(tau_star, tau[valid], seg[valid]))
        kept_mask[i] = True

    arr = np.array(cycles) if cycles else np.empty((0, n_points))
    return _ret(
        arr,
        total=total_pairs,
        kept=len(cycles),
        long=rejected_long,
        short=rejected_short,
        median_dur=median_dur,
        kept_mask=kept_mask,
    )


# ──────────────────────────────────────────────
# 4. STRIDE DURATIONS (K-aligned with extract_cycles output)
# ──────────────────────────────────────────────
def stride_durations(heel_strikes: np.ndarray,
                     clean_mask: np.ndarray | None = None,
                     signal_length: int | None = None,
                     min_len: int = 5) -> np.ndarray:
    """
    Return stride durations (in frames) corresponding 1-to-1 to the cycles
    that extract_cycles would emit for the same `heel_strikes` and
    `clean_mask`.  Pass the same clean_mask & length here as you used there.
    """
    if heel_strikes is None or len(heel_strikes) < 2:
        return np.array([], dtype=int)

    if signal_length is None:
        signal_length = int(heel_strikes[-1] + 1)

    valid_ks = _valid_pair_indices(
        heel_strikes, signal_length, clean_mask=clean_mask, min_len=min_len,
    )
    if not valid_ks:
        return np.array([], dtype=int)
    return np.array(
        [int(heel_strikes[k + 1] - heel_strikes[k]) for k in valid_ks],
        dtype=int,
    )


# ──────────────────────────────────────────────
# 5. FILTER CYCLES (duration MAD + NaN-fraction)
# ──────────────────────────────────────────────
def filter_cycles(cycles: np.ndarray,
                  stride_durations_frames: np.ndarray,
                  mad_multiplier: float = 3.0,
                  max_nan_fraction: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop cycles whose stride duration is a MAD-based outlier
    (|T − median(T)| > mad_multiplier × MAD), AND cycles whose
    fraction of NaN samples exceeds max_nan_fraction.

    Returns: (filtered_cycles, kept_indices_into_input).
    """
    if cycles is None or cycles.ndim != 2 or cycles.shape[0] == 0:
        return np.empty((0, cycles.shape[1] if cycles is not None and cycles.ndim == 2 else 101)), \
               np.array([], dtype=int)

    K, N = cycles.shape

    # NaN-fraction filter
    nan_frac = np.isnan(cycles).sum(axis=1) / N
    nan_keep = nan_frac <= max_nan_fraction

    # Duration MAD filter
    sd = np.asarray(stride_durations_frames, dtype=float) if stride_durations_frames is not None else np.array([])
    if len(sd) == K and K >= 2:
        T_med = np.median(sd)
        mad = np.median(np.abs(sd - T_med))
        if mad > 0:
            dur_keep = np.abs(sd - T_med) <= mad_multiplier * mad
        else:
            dur_keep = np.ones(K, dtype=bool)
    else:
        dur_keep = np.ones(K, dtype=bool)

    keep = nan_keep & dur_keep
    return cycles[keep], np.where(keep)[0]


# ──────────────────────────────────────────────
# 6. ENSEMBLE STATISTICS
# ──────────────────────────────────────────────
def ensemble_statistics(cycles: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Return (mean_curve, std_curve, K) where
        mean / std are per-percent arrays of length n_points
        K          = number of cycles averaged

    If K == 0: returns (None, None, 0) — does NOT raise.
    Uses ddof=1 (Bessel's correction) when K >= 2; std = zeros when K == 1.
    Uses np.nanmean / np.nanstd to tolerate any residual NaN.
    """
    if cycles is None or cycles.ndim != 2 or cycles.shape[0] == 0:
        return None, None, 0

    K = int(cycles.shape[0])
    mean_curve = np.nanmean(cycles, axis=0)
    if K >= 2:
        std_curve = np.nanstd(cycles, axis=0, ddof=1)
    else:
        std_curve = np.zeros_like(mean_curve)
    return mean_curve, std_curve, K
