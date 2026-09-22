// Three-hold reach calibration.
//
// The patient holds the chosen hand still in three poses; each hold
// fills a 5 s ring, and the ring RESETS the moment the wrist drifts.
// The recorded point is the median of the last second of samples, so a
// single bad frame cannot move it.
//
// Samples that fail the visibility floor or fall outside the frame are
// dropped at source. An earlier prototype skipped that and recorded
// MediaPipe's extrapolated guesses for a hand that was off-camera,
// which produced a "reach" extending well past the frame edge.

import type { Hand } from "@/lib/games/handTracker";

export const HOLD_MS = 5000;
/** Wrist drift (normalised units) that resets the ring. ~4% of frame. */
export const STILL_TOLERANCE = 0.04;
/** Window whose median becomes the recorded point. */
export const MEDIAN_WINDOW_MS = 1000;

export type HoldId = "up" | "side" | "across";

export interface HoldDef {
  id: HoldId;
  title: string;
  instruction: string;
}

export const HOLDS: HoldDef[] = [
  {
    id: "up",
    title: "Arm straight up",
    instruction: "Reach as high as you comfortably can and hold still.",
  },
  {
    id: "side",
    title: "Arm out to the side",
    instruction: "Reach out sideways, same side as your playing hand.",
  },
  {
    id: "across",
    title: "Arm across your body",
    instruction: "Reach across toward your other shoulder and hold still.",
  },
];

export interface Point {
  nx: number;
  ny: number;
}

interface Sample extends Point {
  t: number;
}

/**
 * Drives one hold. Feed it every frame; it reports ring progress and
 * hands back a point once the hold completes.
 */
export class HoldTracker {
  private samples: Sample[] = [];
  private anchor: Point | null = null;
  private startedAt = 0;

  /** 0..1 ring fill. */
  progress = 0;
  /** True while the wrist is inside tolerance and the ring is filling. */
  holding = false;

  reset(): void {
    this.samples = [];
    this.anchor = null;
    this.startedAt = 0;
    this.progress = 0;
    this.holding = false;
  }

  /**
   * @returns the recorded point when the hold completes, else null.
   */
  feed(nx: number, ny: number, usable: boolean, nowMs: number): Point | null {
    // Rejected sample — visibility floor or out of frame. Treat exactly
    // like a break in the hold rather than guessing a position.
    if (!usable) {
      this.anchor = null;
      this.samples = [];
      this.startedAt = 0;
      this.progress = 0;
      this.holding = false;
      return null;
    }

    if (!this.anchor) {
      this.anchor = { nx, ny };
      this.startedAt = nowMs;
      this.samples = [{ nx, ny, t: nowMs }];
      this.progress = 0;
      this.holding = true;
      return null;
    }

    const drift = Math.hypot(nx - this.anchor.nx, ny - this.anchor.ny);
    if (drift > STILL_TOLERANCE) {
      // Moved too far — restart the ring from the new position.
      this.anchor = { nx, ny };
      this.startedAt = nowMs;
      this.samples = [{ nx, ny, t: nowMs }];
      this.progress = 0;
      this.holding = true;
      return null;
    }

    this.samples.push({ nx, ny, t: nowMs });
    this.holding = true;
    const elapsed = nowMs - this.startedAt;
    this.progress = Math.min(1, elapsed / HOLD_MS);

    if (elapsed < HOLD_MS) return null;

    // Median of the last MEDIAN_WINDOW_MS of samples.
    const cutoff = nowMs - MEDIAN_WINDOW_MS;
    const tail = this.samples.filter((s) => s.t >= cutoff);
    const use = tail.length > 0 ? tail : this.samples;
    const point = { nx: median(use.map((s) => s.nx)), ny: median(use.map((s) => s.ny)) };
    this.reset();
    return point;
  }
}

function median(values: number[]): number {
  const a = [...values].sort((p, q) => p - q);
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

export interface ReachBox {
  xLo: number;
  xHi: number;
  yLo: number;
  yHi: number;
  /** Body midline in mirrored normalised x. */
  midX: number;
  hand: Hand;
}

/** Keep fruit off the very edge of what the patient can reach. */
const EDGE_INSET = 0.06;
/** How far below the lowest hold the box extends, as a fraction of its
 *  own height — the three holds all sit at or above shoulder height, so
 *  without this every fruit would spawn in the top half of the frame. */
const BOTTOM_EXTEND = 0.3;
/** A box narrower than this in either axis is unusable. */
const MIN_SPAN = 0.1;

/**
 * Build the spawn box from the three recorded holds.
 *
 * The across-midline hold is what pulls `xLo`/`xHi` past the body
 * centre, and fruit placed there is what exercises adduction — so the
 * box is deliberately NOT clamped to the playing side.
 */
export function buildReachBox(
  up: Point,
  side: Point,
  across: Point,
  midX: number,
  hand: Hand,
): ReachBox {
  const xs = [up.nx, side.nx, across.nx];
  const ys = [up.ny, side.ny, across.ny];

  let xLo = Math.min(...xs);
  let xHi = Math.max(...xs);
  let yLo = Math.min(...ys);
  let yHi = Math.max(...ys);

  yHi = yHi + (yHi - yLo) * BOTTOM_EXTEND;

  const insetX = (xHi - xLo) * EDGE_INSET;
  const insetY = (yHi - yLo) * EDGE_INSET;
  xLo += insetX;
  xHi -= insetX;
  yLo += insetY;
  yHi -= insetY;

  // Guard degenerate boxes (a patient who barely moved) by widening
  // around the centre rather than letting spawns collapse to a point.
  if (xHi - xLo < MIN_SPAN) {
    const c = (xHi + xLo) / 2;
    xLo = c - MIN_SPAN / 2;
    xHi = c + MIN_SPAN / 2;
  }
  if (yHi - yLo < MIN_SPAN) {
    const c = (yHi + yLo) / 2;
    yLo = c - MIN_SPAN / 2;
    yHi = c + MIN_SPAN / 2;
  }

  return {
    xLo: clamp01(xLo),
    xHi: clamp01(xHi),
    yLo: clamp01(yLo),
    yHi: clamp01(yHi),
    midX: clamp01(midX),
    hand,
  };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Pick a spawn point inside the box, on the requested side of the
 * midline.
 *
 * Mirrored space: the patient's right hand is drawn on the RIGHT of the
 * screen (higher nx), so for a right-handed session "same side" is the
 * high-x half and "across" is the low-x half. Reversed for the left.
 */
export function spawnPoint(
  box: ReachBox,
  region: "same" | "across",
  rand: () => number,
): Point {
  const sameIsHighX = box.hand === "right";
  const wantHighX = region === "same" ? sameIsHighX : !sameIsHighX;

  let lo: number;
  let hi: number;
  if (wantHighX) {
    lo = Math.max(box.xLo, box.midX);
    hi = box.xHi;
  } else {
    lo = box.xLo;
    hi = Math.min(box.xHi, box.midX);
  }
  // The midline can fall outside the measured box (a patient who never
  // crossed it). Fall back to the whole box rather than an empty range.
  if (hi - lo < 0.02) {
    lo = box.xLo;
    hi = box.xHi;
  }

  return {
    nx: lo + rand() * (hi - lo),
    ny: box.yLo + rand() * (box.yHi - box.yLo),
  };
}
