// Kite Flying difficulty levels.
//
// Same arrangement as the other two games: every tunable the scene
// reads lives here and the scene hard-codes none of them.
//
// ── SIZE COMES FROM THE LEVEL, NEVER FROM THE DISPLAY SETTING ──
//
// The kite's height is a fraction of the patient's calibrated VERTICAL
// REACH, and the corridor's width is a multiple of the kite. The
// Normal/Large control on the setup screen scales text and HUD only.
//
// It used to scale the kite too, and that quietly broke the game: on
// Large the kite grew, the corridor did not, and a big kite inside a
// fixed lane was "in the wind" almost all the time. Tying both to the
// body and to each other means the task is the same difficulty however
// the display is set.

import type { BaseLevel, LevelId } from "@/lib/games/gameDefinition";

export type { LevelId };

/** Round length. 60 s, the same as Fruit Harvest and Cloudburst, so the
 *  half-by-half numbers mean the same thing across all three. */
export const ROUND_MS = 60_000;

/**
 * How quickly the corridor's width drifts, in cycles per canvas width.
 *
 * Low on purpose: at 0.37 the width takes nearly three screen widths to
 * go from its narrowest to its widest and back, so it is felt as a
 * gradual squeeze rather than as a step.
 */
export const WIDTH_WAVE_FREQ = 0.37;

/** How the centreline's two sines are weighted, and how much faster the
 *  second one is. The second is there to stop the path repeating
 *  visibly; it is not meant to be a second set of turns, so it stays
 *  small. Both are on the same amplitude budget. */
export const WAVE_MAIN_SHARE = 0.75;
export const WAVE_SECOND_SHARE = 0.25;
export const WAVE_SECOND_RATIO = 1.7;

/** How long the palm may sit outside the corridor before the kite
 *  tumbles. Long enough to survive a wobble, short enough that drifting
 *  out is not free. */
export const FALL_AFTER_MS = 1500;
/** Tumble down, then this long before the kite reappears at the hand. */
export const TUMBLE_MS = 900;
export const RESPAWN_MS = 1000;

export interface KiteLevel extends BaseLevel {
  /**
   * Kite HEIGHT as a fraction of the calibrated vertical reach. The
   * only thing that sets the kite's size.
   */
  kiteFraction: number;
  /**
   * Corridor WIDTH = kite height x a factor, drifting smoothly between
   * these two along the path. Because it is a fixed multiple of the
   * kite, "the anchor is inside" and "the kite looks inside" stay the
   * same judgement at every level and on every screen.
   */
  widthFactorMin: number;
  widthFactorMax: number;
  /** Full waves of the centreline visible across the screen at once. */
  waves: number;
  /**
   * Target amplitude, as a share of the room left over once the
   * corridor's own half-width is taken out of the vertical reach.
   *
   * THAT ROOM IS A HARD CEILING. The centreline cannot swing further
   * than `span/2 - halfMax` without the corridor's edge leaving the
   * patient's reach, so the deepest possible curve is 27.9% of the
   * reach at level 1 and 39.2% at level 2 — the wider the lane, the
   * less room is left to move it in. No scroll speed changes that.
   *
   * Below that ceiling the speed ceiling can still win; makeCorridor()
   * reports which bound actually applied.
   */
  curveAmp: number;
  /**
   * Ceiling on the vertical hand speed the path demands, in ARM LENGTHS
   * per second, at the steepest point.
   *
   * This is the followability guarantee and it is enforced, not hoped
   * for: the amplitude is reduced until the peak slope meets it. Scale
   * free, so it means the same for a tall patient and a short one at
   * any distance from the camera.
   */
  maxHandSpeedArmPerSec: number;
  /** Scroll speed, in canvas widths per second. */
  scrollPerSec: number;
}

export const LEVELS: Record<LevelId, KiteLevel> = {
  1: {
    id: 1,
    label: "Level 1",
    // A bigger kite in a lane about three times its height, drifting
    // past slowly over two deep waves.
    kiteFraction: 0.13,
    widthFactorMin: 2.6,
    widthFactorMax: 3.4,
    waves: 2,
    // 90% of the room, leaving a tenth as margin rather than letting
    // the corridor's edge kiss the measured limit of the reach at every
    // peak — the reach box is an estimate from three held points, not a
    // survey.
    curveAmp: 0.9,
    maxHandSpeedArmPerSec: 0.6,
    // Slow enough that the reach ceiling, not the hand-speed ceiling,
    // is what limits the curve. At the old 0.10 the speed cap flattened
    // the path to 7.5% of the reach; here it is about 25%.
    scrollPerSec: 0.045,
  },
  2: {
    id: 2,
    label: "Level 2",
    // A smaller kite in a lane about twice its height — so the margin
    // for error is roughly half — and a deeper curve, which its higher
    // hand-speed ceiling is what pays for.
    kiteFraction: 0.09,
    widthFactorMin: 1.8,
    widthFactorMax: 2.4,
    // Two waves, not three. A third wave costs amplitude twice over —
    // the slope scales with frequency, so the speed ceiling pushes the
    // curve down further, and a shallower curve was the complaint.
    waves: 2,
    curveAmp: 0.9,
    maxHandSpeedArmPerSec: 0.9,
    // 0.07 was asked for and was not enough: at that speed the hand
    // speed ceiling still bound and the curve stopped at 26% of the
    // reach. 0.05 lets it reach about 35%, which is close to the 39.2%
    // its lane width allows at all.
    scrollPerSec: 0.05,
  },
};

export const DEFAULT_LEVEL: LevelId = 1;

export function levelById(id: number): KiteLevel {
  return LEVELS[(id === 2 ? 2 : 1) as LevelId];
}
