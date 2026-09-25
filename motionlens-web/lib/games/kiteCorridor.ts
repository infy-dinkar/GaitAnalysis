// The wind corridor's shape.
//
// Pure geometry, no Phaser. The scene draws from it and the metrics
// measure against it, and they must agree exactly — a corridor drawn in
// one place and scored in another is worse than no corridor at all, so
// there is one set of functions and both callers use them.
//
// COORDINATES. Heights are in cover-normalised y (`ny`), the same space
// the reach box uses, so the corridor is defined in terms of the
// patient's body rather than the screen. Horizontal position is
// `worldX`, measured in CANVAS WIDTHS travelled: the ribbon scrolls by
// advancing worldX, and a wave count of 2 means two full waves visible
// across the screen whatever the screen is.
//
// NOTHING HERE TAKES THE DISPLAY SCALE. The kite's size and the
// corridor's width come from the level and the patient's reach, and
// that is the whole point — see the note at the top of kiteLevels.ts.
//
// ── THREE GUARANTEES ──
//
//  1. The corridor's EDGES stay inside the calibrated vertical reach,
//     not just its centreline. Every part of the ribbon is somewhere
//     the hand can actually go.
//  2. The corridor's width is always the level's factor times the kite
//     height, so "the anchor is inside" and "the kite looks inside" are
//     the same judgement.
//  3. The steepest slope never demands more vertical hand speed than
//     the level allows. This one is ENFORCED by reducing the amplitude,
//     not assumed — see makeCorridor.

import type { ReachBox } from "@/lib/games/calibration";
import { WIDTH_WAVE_FREQ, type KiteLevel } from "@/lib/games/kiteLevels";

/** However tight the anti-cheat cap gets, one pass may not shrink the
 *  lane and kite below this fraction of their previous size — a kite
 *  reduced to a speck would be unplayable, and an unplayable game is
 *  not an improvement on a cheatable one. */
const MIN_KITE_SCALE = 0.35;

export interface Corridor {
  /** Centre of the patient's vertical reach, and its bounds. */
  midNy: number;
  loNy: number;
  hiNy: number;
  /** Kite height, in ny. The corridor is a multiple of this. */
  kiteNy: number;
  /** Half-width bounds, in ny. */
  halfMinNy: number;
  halfMaxNy: number;
  /** Amplitude actually used, after both bounds were applied. */
  ampNy: number;
  /** Which bound decided it. "speed" means the level's followability
   *  ceiling was the binding constraint, not the reach. */
  ampLimitedBy: "reach" | "speed";
  /** What the target amplitude would have been on the reach bound
   *  alone, for the debug readout. */
  ampNominalNy: number;
  /** Centreline waves, in cycles per canvas width, and their weights. */
  f1: number;
  f2: number;
  mainShare: number;
  secondShare: number;
  p1: number;
  p2: number;
  /** Width wave. */
  widthPhase: number;
  /** Factor at the midpoint of its range — the split between a "narrow"
   *  and a "wide" stretch. */
  medianFactor: number;
  factorMin: number;
  factorMax: number;
  /** Peak vertical hand speed the finished path demands, arm lengths
   *  per second. At or below the level's ceiling by construction. */
  peakHandSpeedArmPerSec: number;
  /** Arm lengths per unit of ny, as measured when the round began. */
  nyToArm: number;
}

/** Kite height in ny, from the level and the patient's reach. */
export function kiteHeightNy(box: ReachBox, level: KiteLevel): number {
  const span = Math.max(1e-4, box.yHi - box.yLo);
  return span * level.kiteFraction;
}

/**
 * Build the corridor for one round.
 *
 * @param nyToArm arm lengths per unit of ny — `cover.dispH / armLenPx`.
 *                Pass 0 when it cannot be measured, and the speed
 *                ceiling is simply not applied.
 */
export function makeCorridor(
  box: ReachBox,
  level: KiteLevel,
  nyToArm: number,
  rand: () => number = Math.random,
): Corridor {
  const span = Math.max(1e-4, box.yHi - box.yLo);
  const midNy = (box.yLo + box.yHi) / 2;

  let kiteNy = kiteHeightNy(box, level);
  let halfMinNy = (kiteNy * level.widthFactorMin) / 2;
  let halfMaxNy = (kiteNy * level.widthFactorMax) / 2;

  // Guarantee 1, the hard case: if the level's widest corridor would
  // not fit inside the reach with room to move, shrink both bounds
  // together so the ratio between them — and the ratio to the kite —
  // is kept. A corridor that left the reach would be unreachable; one
  // that filled it would make the game "hold still".
  const maxAllowedHalf = span * 0.375;
  if (halfMaxNy > maxAllowedHalf) {
    const k = maxAllowedHalf / halfMaxNy;
    halfMaxNy *= k;
    halfMinNy *= k;
  }

  const f1 = level.waves;
  const f2 = f1 * level.secondRatio;
  const secondShare = level.secondShare;
  const mainShare = 1 - secondShare;

  // Guarantee 3. The centreline is
  //   amp * (m*sin(2*pi*f1*x + p1) + s*sin(2*pi*f2*x + p2))
  // so its steepest slope per canvas width is at most
  //   amp * 2*pi * (m*f1 + s*f2)
  // and the vertical speed that demands is that times the scroll speed,
  // converted from ny into arm lengths.
  const slopeShape = 2 * Math.PI * (mainShare * f1 + secondShare * f2);
  const perAmp = slopeShape * level.scrollPerSec * nyToArm;

  // ── Amplitude and lane width are solved together.
  //
  // They depend on each other: the lane's width comes out of the reach
  // before the centreline gets its room, and the lane is then capped as
  // a share of the amplitude that leaves. Three passes settle it — each
  // narrowing of the lane frees room, which raises the amplitude, which
  // raises the cap, and it converges from below.
  let ampNy = 0;
  let ampNominalNy = 0;
  let ampLimitedBy: "reach" | "speed" = "reach";
  for (let pass = 0; pass < 4; pass++) {
    const room = Math.max(0, span / 2 - halfMaxNy);
    ampNominalNy = room * level.curveAmp;
    ampNy = ampNominalNy;
    ampLimitedBy = "reach";
    if (perAmp > 0) {
      const ampFromSpeed = level.maxHandSpeedArmPerSec / perAmp;
      if (ampFromSpeed < ampNy) {
        ampNy = ampFromSpeed;
        ampLimitedBy = "speed";
      }
    }
    const capHalf = level.maxHalfOverAmp * ampNy;
    if (halfMaxNy <= capHalf + 1e-12) break;
    // Scale the lane AND the kite by the same factor, so the kite still
    // fits its lane exactly as the level specifies.
    const k = Math.max(MIN_KITE_SCALE, capHalf / halfMaxNy);
    halfMaxNy *= k;
    halfMinNy *= k;
    kiteNy *= k;
  }

  return {
    midNy,
    loNy: box.yLo,
    hiNy: box.yHi,
    kiteNy,
    halfMinNy,
    halfMaxNy,
    ampNy,
    ampLimitedBy,
    ampNominalNy,
    f1,
    f2,
    mainShare,
    secondShare,
    p1: rand() * Math.PI * 2,
    p2: rand() * Math.PI * 2,
    widthPhase: rand() * Math.PI * 2,
    medianFactor: (level.widthFactorMin + level.widthFactorMax) / 2,
    factorMin: level.widthFactorMin,
    factorMax: level.widthFactorMax,
    peakHandSpeedArmPerSec: ampNy * perAmp,
    nyToArm,
  };
}

/** Width factor at a point along the ribbon: a slow sine between the
 *  level's two bounds, so the squeeze is gradual. */
export function factorAt(c: Corridor, worldX: number): number {
  const mid = (c.factorMin + c.factorMax) / 2;
  const amp = (c.factorMax - c.factorMin) / 2;
  return mid + amp * Math.sin(2 * Math.PI * WIDTH_WAVE_FREQ * worldX + c.widthPhase);
}

/** Half the corridor's height at a point, in ny. */
export function halfNyAt(c: Corridor, worldX: number): number {
  const mid = (c.halfMinNy + c.halfMaxNy) / 2;
  const amp = (c.halfMaxNy - c.halfMinNy) / 2;
  return mid + amp * Math.sin(2 * Math.PI * WIDTH_WAVE_FREQ * worldX + c.widthPhase);
}

/** Is this stretch narrower than the level's median? */
export function isNarrowAt(c: Corridor, worldX: number): boolean {
  return factorAt(c, worldX) < c.medianFactor;
}

/**
 * Centreline height at a point along the ribbon.
 *
 * Two sines whose weights sum to 1, so the result is always within
 * ±ampNy. The second is at an irrational-ish multiple of the first's
 * frequency, so the shape does not visibly repeat within a 60 s round.
 *
 * The clamp at the end is guarantee 1's belt and braces: the amplitude
 * was already sized against the WIDEST part of the corridor, so it
 * should never bite — but if a reach box ever arrived that made it,
 * the corridor stays inside the reach rather than the guarantee being
 * merely documented.
 */
export function centreNy(c: Corridor, worldX: number): number {
  const a = Math.sin(2 * Math.PI * c.f1 * worldX + c.p1);
  const b = Math.sin(2 * Math.PI * c.f2 * worldX + c.p2);
  const raw = c.midNy + c.ampNy * (c.mainShare * a + c.secondShare * b);
  const half = halfNyAt(c, worldX);
  return Math.min(c.hiNy - half, Math.max(c.loNy + half, raw));
}

/**
 * How far off the centreline a point is, as a multiple of the LOCAL
 * half-width: 0 is dead centre, 1 is exactly on the edge, above 1 is
 * outside.
 *
 * Local, not global — where the lane narrows, the same absolute
 * distance is a bigger error, and the score should say so.
 */
export function deviationRatio(
  c: Corridor,
  worldX: number,
  ny: number,
): number {
  const half = halfNyAt(c, worldX);
  if (half <= 0) return 0;
  return Math.abs(ny - centreNy(c, worldX)) / half;
}

/**
 * Vertical hand speed the path demands right here, in arm lengths per
 * second. Diagnostics only — the ceiling is enforced at build time.
 */
export function requiredHandSpeed(
  c: Corridor,
  worldX: number,
  scrollPerSec: number,
): number {
  const da = 2 * Math.PI * c.f1
    * Math.cos(2 * Math.PI * c.f1 * worldX + c.p1) * c.mainShare;
  const db = 2 * Math.PI * c.f2
    * Math.cos(2 * Math.PI * c.f2 * worldX + c.p2) * c.secondShare;
  return Math.abs(c.ampNy * (da + db)) * scrollPerSec * c.nyToArm;
}
