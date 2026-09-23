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
// ADDUCTION DEGREES ARE DELIBERATELY NOT MEASURED. The across-body
// reaches this game produces are HORIZONTAL adduction (transverse
// plane). The biomech module's adduction figure is frontal-plane
// adduction, whose reference range is 30-50 deg. They share one
// formula and are not the same measurement, so reporting the game's
// number as "adduction" would put a ~90 deg transverse value against a
// 30-50 deg frontal range. A frontal 2-D camera cannot separate the
// planes at all. The across-body reach COUNT (zone_hits.adduction) is
// what the game can honestly report, and it is what it reports.
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

/** Live readout for ?gamedebug=1 — why the direction came out the way
 *  it did, in the same units the biomech rule uses. */
export interface AngleDebug {
  angleDeg: number | null;
  dir: string;
  /** Which branch of detectShoulderAbAdDirection decided it. */
  decidedBy: string;
  /** Elbow, in shoulder widths: + = outward from the shoulder. */
  elbowDx: number;
  /** Elbow height above the shoulder, in shoulder widths. */
  elbowDy: number;
  /** Wrist, in shoulder widths: + = outward from the shoulder. */
  wristDx: number;
  /** Elbow and wrist relative to the MID-SHOULDER line: + = outward,
   *  negative = across the body. The biomech rule never looks at
   *  these; they are here to show what it is missing. */
  elbowFromMid: number;
  wristFromMid: number;
  /** Did this frame contribute to the abduction peak, and if not why. */
  counted: string;
}

export interface RoundMetrics {
  startedAtMs: number;
  /** Diagnostics only — never saved to a report. */
  dbg: AngleDebug;
  // ── written by the pose loop
  maxAbductionDeg: number;
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
    dbg: {
      angleDeg: null,
      dir: "-",
      decidedBy: "-",
      elbowDx: 0,
      elbowDy: 0,
      wristDx: 0,
      elbowFromMid: 0,
      wristFromMid: 0,
      counted: "-",
    },
    maxAbductionDeg: 0,
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
      this.m.dbg.counted = "no — shoulder or elbow not visible";
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

    // Mirror the biomech rule's own arithmetic so the overlay can show
    // WHICH branch decided the label. Read-only: lib/biomech is not
    // touched, this only re-derives the same ratios for display.
    const d = this.m.dbg;
    d.angleDeg = Math.round(deg);
    d.dir = dir ?? "none (deadband)";
    let wristOwnSide: boolean | null = null;
    const ls = kp[LM_LIVE.LEFT_SHOULDER];
    const rs = kp[LM_LIVE.RIGHT_SHOULDER];
    const wr = kp[this.hand === "left" ? LM_LIVE.LEFT_WRIST : LM_LIVE.RIGHT_WRIST];
    if (ls && rs) {
      const width = Math.abs(rs.x - ls.x) || 1;
      const midX = (ls.x + rs.x) / 2;
      const outward = Math.sign(sh!.x - midX) || 1;
      const dy = (sh!.y - el!.y) / width;
      d.elbowDx = round2(((el!.x - sh!.x) * outward) / width);
      d.elbowDy = round2(dy);
      d.elbowFromMid = round2(((el!.x - midX) * outward) / width);
      if (wr && (wr.score ?? 0) >= VIS_FLOOR) {
        d.wristDx = round2(((wr.x - sh!.x) * outward) / width);
        d.wristFromMid = round2(((wr.x - midX) * outward) / width);
        // Positive = still on the patient's own side of the midline.
        wristOwnSide = d.wristFromMid > 0;
      }
      // The rule checks the y override first, then the x deadband.
      d.decidedBy = dy > 0.20
        ? "elbow ABOVE shoulder (y-override -> abduction)"
        : Math.abs(d.elbowDx) < 0.03
          ? "elbow inside x deadband (no direction)"
          : "elbow x vs shoulder";
    }
    // ABDUCTION GUARD.
    //
    // The biomech direction rule reads the ELBOW only, and returns
    // "abduction" whenever the elbow is raised more than 0.20 shoulder
    // widths — even when the hand is reaching ACROSS the body. That is
    // how an across-and-up reach was being scored as 167 deg of
    // abduction. A frame only counts now when the chosen WRIST is still
    // on its own side of the mid-shoulder line.
    //
    // Excluding resets the window rather than skipping it, so the
    // 150 ms hold must be 150 ms of COUNTED frames — a hold cannot be
    // certified across a gap where the arm was across the body.
    if (dir !== "abduction") {
      this.abPeak.reset();
      this.m.dbg.counted = `no — direction ${dir ?? "undecided"}`;
      return;
    }
    if (wristOwnSide !== true) {
      this.abPeak.reset();
      this.m.dbg.counted = wristOwnSide === null
        ? "no — wrist not visible"
        : "no — wrist has crossed the midline";
      return;
    }
    this.abPeak.push(nowMs, deg);
    this.m.maxAbductionDeg = Math.round(this.abPeak.peak);
    this.m.dbg.counted = "yes";
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

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}
