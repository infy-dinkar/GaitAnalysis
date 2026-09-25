// Kite Flying difficulty levels.
//
// Same arrangement as the other two games: every tunable the scene
// reads lives here and the scene hard-codes none of them.
//
// A level changes HOW WIDE the corridor is, HOW FAST it scrolls and HOW
// SHARP its curves are. It does not change the round length, the fall
// rule or the metrics — those are the game, not the difficulty.

import type { BaseLevel, LevelId } from "@/lib/games/gameDefinition";

export type { LevelId };

/** Round length. 60 s, the same as Fruit Harvest and Cloudburst, so the
 *  half-by-half numbers mean the same thing across all three. */
export const ROUND_MS = 60_000;

/**
 * Kite size as a fraction of the canvas unit.
 *
 * At least as large as a Fruit Harvest fruit at its biggest — that
 * game's level 1 draws fruit at 0.13 * 1.3 = 0.169 — because on camera
 * the kite at 0.11 was simply too small to find from 2 m. This is a
 * drawing size, not a level value: nothing in LEVELS below changes with
 * it and the difficulty is unaffected.
 */
export const KITE_FRACTION = 0.17;

/** How long the palm may sit outside the corridor before the kite
 *  tumbles. Long enough to survive a wobble, short enough that drifting
 *  out is not free. */
export const FALL_AFTER_MS = 1500;
/** Tumble down, then this long before the kite reappears at the hand. */
export const TUMBLE_MS = 900;
export const RESPAWN_MS = 1000;

export interface KiteLevel extends BaseLevel {
  /**
   * Corridor height as a fraction of the patient's calibrated VERTICAL
   * reach. The centreline is then kept inside that reach with the
   * corridor's own half-width subtracted, so every part of the ribbon
   * is somewhere the hand can actually go.
   */
  corridorFraction: number;
  /** Scroll speed, in canvas widths per second. */
  scrollPerSec: number;
  /** How much of the available height the centreline uses, 0..1. */
  curveAmp: number;
  /** Spatial frequency of the centreline, in cycles per canvas width. */
  curveFreq: number;
}

export const LEVELS: Record<LevelId, KiteLevel> = {
  1: {
    id: 1,
    label: "Level 1",
    // A wide lane, drifting past slowly, with long lazy curves: the
    // patient can hold a steady arm and still stay in.
    corridorFraction: 0.42,
    scrollPerSec: 0.1,
    curveAmp: 0.55,
    curveFreq: 1.0,
  },
  2: {
    id: 2,
    label: "Level 2",
    // 65% of level 1's width, 1.5x the scroll, and curves that are both
    // taller and more frequent — so corrections have to be quicker and
    // more accurate at once.
    corridorFraction: 0.273, // 0.42 * 0.65
    scrollPerSec: 0.15,
    curveAmp: 0.75,
    curveFreq: 1.6,
  },
};

export const DEFAULT_LEVEL: LevelId = 1;

export function levelById(id: number): KiteLevel {
  return LEVELS[(id === 2 ? 2 : 1) as LevelId];
}
