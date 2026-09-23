// What a Fruit Harvest round measures.
//
// Two producers write into one mutable object, the same arrangement
// HandState uses:
//
//   • the React pose loop  — shoulder angle, every pose frame
//   • the Phaser scene     — collects, misses, zones, timings
//
// SHOULDER ANGLE. The formula is the biomech module's, not a new one:
// lib/biomech/shoulder-live.ts:131 shoulderAbductionAdduction() is the
// angle between the trunk axis (shoulder -> hip) and the upper arm
// (shoulder -> elbow). Direction comes from that module's
// detectShoulderAbAdDirection(). Reusing both is the only way the
// game's degrees can be compared with a biomech assessment at all.
//
// These are still a GAME estimate. The patient is reaching for fruit,
// not holding a measured pose side-on to the camera, and the result
// screen says so.

// The SAME keypoint type the biomech module takes, so its formulas
// apply to this array without a cast.
import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";
import {
  computeShoulderAngle,
  detectShoulderAbAdDirection,
} from "@/lib/biomech/shoulder-live";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
import { VIS_FLOOR, type Hand } from "@/lib/games/handTracker";

/** A peak must be held this long to count. A single frame at 140 deg
 *  is a tracking spike, not range of motion. */
export const PEAK_HOLD_MS = 150;

export interface ZoneHits {
  abduction: number;
  adduction: number;
}

export interface CalibrationSummary {
  /** Reach in each direction, in ARM LENGTHS — scale-free, so it can
   *  be compared between patients and between sessions. */
  up: number;
  side: number;
  across: number;
  headroom_ratio: number;
}

export interface RoundMetrics {
  startedAtMs: number;
  // ── written by the pose loop
  maxAbductionDeg: number;
  maxAdductionDeg: number;
  /** True when any peak was taken without a visible hip, so the trunk
   *  axis fell back to screen vertical. */
  abductionLowConfidence: boolean;
  angleFrames: number;
  // ── written by the scene
  harvested: number;
  missed: number;
  collectSecs: number[];
  zoneHits: ZoneHits;
  firstHalf: { hit: number; total: number };
  secondHalf: { hit: number; total: number };
}

export function createRoundMetrics(startedAtMs: number): RoundMetrics {
  return {
    startedAtMs,
    maxAbductionDeg: 0,
    maxAdductionDeg: 0,
    abductionLowConfidence: false,
    angleFrames: 0,
    harvested: 0,
    missed: 0,
    collectSecs: [],
    zoneHits: { abduction: 0, adduction: 0 },
    firstHalf: { hit: 0, total: 0 },
    secondHalf: { hit: 0, total: 0 },
  };
}

interface Sample {
  t: number;
  v: number;
}

/**
 * Largest value that was HELD for at least PEAK_HOLD_MS.
 *
 * Implemented as the running maximum of a trailing-window minimum: if
 * the smallest angle seen over the last 150 ms is X, then the angle was
 * at or above X for that whole window. Taking the max of that over the
 * round gives a peak that cannot come from one spiking frame, without
 * needing to store the whole trace.
 */
class HeldPeak {
  private win: Sample[] = [];
  peak = 0;

  push(t: number, v: number): void {
    this.win.push({ t, v });
    const cutoff = t - PEAK_HOLD_MS;
    while (this.win.length > 0 && this.win[0].t < cutoff) this.win.shift();
    // Only a full window can certify a hold.
    if (this.win.length < 2 || t - this.win[0].t < PEAK_HOLD_MS) return;
    let lo = Infinity;
    for (const s of this.win) if (s.v < lo) lo = s.v;
    if (lo > this.peak) this.peak = lo;
  }

  /** The arm left this direction — the next stretch is a fresh hold. */
  reset(): void {
    this.win = [];
  }
}

export class MetricsRecorder {
  readonly m: RoundMetrics;
  private hand: Hand;
  private abPeak = new HeldPeak();
  private adPeak = new HeldPeak();

  constructor(hand: Hand, startedAtMs: number) {
    this.hand = hand;
    this.m = createRoundMetrics(startedAtMs);
  }

  /**
   * Feed one pose frame. `kp` is the RAW keypoint array in video pixel
   * space — the same array the biomech module expects, un-mirrored, so
   * its formulas apply unchanged.
   */
  sample(kp: Keypoint[] | null, nowMs: number): void {
    if (!kp) return;
    const idx = this.hand === "left"
      ? { s: LM_LIVE.LEFT_SHOULDER, e: LM_LIVE.LEFT_ELBOW, h: LM_LIVE.LEFT_HIP }
      : { s: LM_LIVE.RIGHT_SHOULDER, e: LM_LIVE.RIGHT_ELBOW, h: LM_LIVE.RIGHT_HIP };

    const sh = kp[idx.s];
    const el = kp[idx.e];
    const hip = kp[idx.h];
    // The game's own floor, stricter than the biomech module's 0.15 —
    // a reaching patient is not holding a measured pose, so a weak
    // landmark is more likely to be wrong here.
    const ok = (p: Keypoint | undefined) => !!p && (p.score ?? 0) >= VIS_FLOOR;
    if (!ok(sh) || !ok(el)) {
      this.abPeak.reset();
      this.adPeak.reset();
      return;
    }

    let deg: number | null;
    if (ok(hip)) {
      // "abduction" and "adduction" share one formula in that module
      // (shoulder-live.ts:172); the direction is detected separately.
      deg = computeShoulderAngle("abduction", kp, this.hand);
    } else {
      // No hip: substitute screen vertical for the trunk axis and flag
      // the round. A leaning patient will read a few degrees off, which
      // is worth saying out loud rather than hiding.
      deg = angleFromVertical(sh!, el!);
      this.m.abductionLowConfidence = true;
    }
    if (deg === null || !Number.isFinite(deg)) return;

    this.m.angleFrames += 1;
    const dir = detectShoulderAbAdDirection(kp, this.hand);
    if (dir === "abduction") {
      this.abPeak.push(nowMs, deg);
      this.adPeak.reset();
      this.m.maxAbductionDeg = Math.round(this.abPeak.peak);
    } else if (dir === "adduction") {
      this.adPeak.push(nowMs, deg);
      this.abPeak.reset();
      this.m.maxAdductionDeg = Math.round(this.adPeak.peak);
    } else {
      // Inside the direction deadband — commit to neither.
      this.abPeak.reset();
      this.adPeak.reset();
    }
  }

  /** A fruit was collected. `ageMs` is spawn -> touch. */
  onCollect(ageMs: number, zone: "abduction" | "adduction", half: 1 | 2): void {
    this.m.harvested += 1;
    this.m.collectSecs.push(ageMs / 1000);
    this.m.zoneHits[zone] += 1;
    const h = half === 1 ? this.m.firstHalf : this.m.secondHalf;
    h.hit += 1;
    h.total += 1;
  }

  onMiss(half: 1 | 2): void {
    this.m.missed += 1;
    const h = half === 1 ? this.m.firstHalf : this.m.secondHalf;
    h.total += 1;
  }
}

/** Angle of the upper arm from screen vertical, in degrees. The
 *  fallback trunk axis when the hip is not visible. */
function angleFromVertical(s: Keypoint, e: Keypoint): number {
  // Image y grows downward, so "down the body" is +y.
  const ax = e.x - s.x;
  const ay = e.y - s.y;
  const dot = ay;                    // dot with (0, 1)
  const cross = ax;                  // cross with (0, 1)
  return Math.abs((Math.atan2(cross, dot) * 180) / Math.PI);
}

export function pct(hit: number, total: number): number {
  return total > 0 ? Math.round((hit / total) * 100) : 0;
}

export function meanCollectSec(secs: number[]): number | null {
  if (secs.length === 0) return null;
  return Math.round((secs.reduce((a, b) => a + b, 0) / secs.length) * 100) / 100;
}
