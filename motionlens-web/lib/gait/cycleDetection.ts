/**
 * Detect heel-strike events from the time series of an ankle's vertical position.
 * Heel strikes manifest as local maxima of y (ankle at lowest point in image-space,
 * which has y growing downward). Robust to NaN gaps via forward/back-fill.
 */

export interface HeelStrike {
  frame: number;
  time: number; // seconds
}

export interface DetectionDiagnostics {
  validFraction: number; // 0..1 — fraction of input frames with non-NaN values
  smoothedRange: number; // peak-to-peak range of smoothed signal (px)
}

export function detectHeelStrikes(
  ankleY: number[],
  fps: number,
  options: { minSeparationSec?: number; smoothing?: number } = {},
): { strikes: HeelStrike[]; diag: DetectionDiagnostics } {
  const { minSeparationSec = 0.4, smoothing = 5 } = options;
  if (ankleY.length < 3) {
    return { strikes: [], diag: { validFraction: 0, smoothedRange: 0 } };
  }

  // 1. forward + back fill NaN gaps so the smoothing window doesn't propagate them
  const filled = fillGaps(ankleY);
  const validCount = ankleY.filter((v) => Number.isFinite(v)).length;
  const validFraction = validCount / ankleY.length;

  // 2. moving-average smoothing
  const smoothed = movingAverage(filled, smoothing);

  // 3. peak amplitude check — if signal is essentially flat, abort
  const min = Math.min(...smoothed);
  const max = Math.max(...smoothed);
  const range = max - min;
  if (range < 4) {
    // fewer than ~4 px peak-to-peak: pose noise, not gait
    return { strikes: [], diag: { validFraction, smoothedRange: range } };
  }

  // 4. peak finder — maxima of y, separated by at least minSep frames
  const minSep = Math.max(1, Math.round(minSeparationSec * fps));
  const threshold = min + range * 0.35; // ignore tiny ripples in trough region
  const strikes: HeelStrike[] = [];

  for (let i = 1; i < smoothed.length - 1; i++) {
    const v = smoothed[i];
    const isPeak = v > smoothed[i - 1] && v >= smoothed[i + 1] && v > threshold;
    if (!isPeak) continue;
    const last = strikes[strikes.length - 1];
    if (last && i - last.frame < minSep) {
      if (v > smoothed[last.frame]) {
        last.frame = i;
        last.time = i / fps;
      }
      continue;
    }
    strikes.push({ frame: i, time: i / fps });
  }

  return { strikes, diag: { validFraction, smoothedRange: range } };
}

/** Forward-fill then back-fill to remove NaN gaps. Returns a copy. */
function fillGaps(arr: number[]): number[] {
  const out = arr.slice();
  // forward fill
  let last: number | null = null;
  for (let i = 0; i < out.length; i++) {
    if (Number.isFinite(out[i])) last = out[i];
    else if (last !== null) out[i] = last;
  }
  // back fill leading NaNs
  let next: number | null = null;
  for (let i = out.length - 1; i >= 0; i--) {
    if (Number.isFinite(out[i])) next = out[i];
    else if (next !== null) out[i] = next;
    else out[i] = 0; // entirely empty input
  }
  return out;
}

function movingAverage(arr: number[], window: number): number[] {
  if (window <= 1) return arr;
  const half = Math.floor(window / 2);
  const out: number[] = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(arr.length - 1, i + half); j++) {
      sum += arr[j];
      count++;
    }
    out[i] = sum / count;
  }
  return out;
}
