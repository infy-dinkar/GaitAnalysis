// The wind corridor's shape.
//
// Pure geometry, no Phaser. The scene draws from it and the metrics
// measure against it, and they must agree exactly — a corridor that is
// drawn one place and scored another is worse than no corridor at all,
// so there is one function and both callers use it.
//
// COORDINATES. Heights are in cover-normalised y (`ny`), the same space
// the reach box uses, so the corridor is defined in terms of the
// patient's body rather than the screen. Horizontal position is
// `worldX`, measured in CANVAS WIDTHS travelled: the ribbon scrolls by
// advancing worldX, and a curve frequency of 1.0 means one full wave
// per screen width whatever the screen is.

import type { ReachBox } from "@/lib/games/calibration";
import type { KiteLevel } from "@/lib/games/kiteLevels";

export interface Corridor {
  /** Centre of the patient's vertical reach, in ny. */
  midNy: number;
  /** How far the centreline may swing either side of `midNy`. Already
   *  has the corridor's own half-width subtracted, so the ribbon's
   *  EDGES stay inside the reach, not just its middle. */
  ampNy: number;
  /** Half the corridor's height, in ny. */
  halfNy: number;
  /** Cycles per canvas width. */
  freq: number;
  /** Phase offsets, so two rounds do not present the same wave. */
  p1: number;
  p2: number;
}

/**
 * Build the corridor for one round.
 *
 * The reach box's vertical span is the whole budget: the corridor's
 * half-width is taken out of it first, and whatever is left is how far
 * the centreline is allowed to wander. If a patient's calibrated
 * vertical reach is too shallow for the level's corridor, the corridor
 * is narrowed to fit rather than being allowed to leave the reach —
 * an unreachable ribbon would measure nothing but frustration.
 */
export function makeCorridor(
  box: ReachBox,
  level: KiteLevel,
  rand: () => number = Math.random,
): Corridor {
  const span = Math.max(1e-4, box.yHi - box.yLo);
  const midNy = (box.yLo + box.yHi) / 2;

  // Never let the ribbon fill the whole reach: leave at least a
  // quarter of the span for the centreline to move in, or the game
  // becomes "hold still" rather than "follow".
  let halfNy = (span * level.corridorFraction) / 2;
  const maxHalf = span * 0.375;
  if (halfNy > maxHalf) halfNy = maxHalf;

  const ampNy = Math.max(0, (span / 2 - halfNy) * level.curveAmp);

  return {
    midNy,
    ampNy,
    halfNy,
    freq: level.curveFreq,
    p1: rand() * Math.PI * 2,
    p2: rand() * Math.PI * 2,
  };
}

/**
 * Centreline height at a point along the ribbon.
 *
 * Two sine waves whose weights sum to 1, so the result is always within
 * ±1 and the centreline can never leave `midNy ± ampNy`. The second is
 * at 1.7x the first's frequency — an irrational-ish ratio, so the shape
 * does not visibly repeat within a 60 s round.
 */
export function centreNy(c: Corridor, worldX: number): number {
  const a = Math.sin(2 * Math.PI * c.freq * worldX + c.p1);
  const b = Math.sin(2 * Math.PI * c.freq * 1.7 * worldX + c.p2);
  return c.midNy + c.ampNy * (0.62 * a + 0.38 * b);
}

/**
 * How far off the centreline a point is, as a multiple of the corridor's
 * half-width: 0 is dead centre, 1 is exactly on the edge, above 1 is
 * outside.
 *
 * Reported rather than a bare distance because a wide corridor and a
 * narrow one are different tasks, and "40% of the way to the edge"
 * compares across them while "0.06 of a frame" does not.
 */
export function deviationRatio(
  c: Corridor,
  worldX: number,
  ny: number,
): number {
  if (c.halfNy <= 0) return 0;
  return Math.abs(ny - centreNy(c, worldX)) / c.halfNy;
}
