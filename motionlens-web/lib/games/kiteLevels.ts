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

// The centreline's second sine moved into the level config below: it
// turned out to be the lever that stops a still hand passing, and the
// two levels want different amounts of it.

// ── The opening of a round.
//
// The corridor arrives straight and level at mid-reach, so the patient
// can find the kite, see where the lane is and get into it before the
// task begins. Then the waves fade in.
//
// It is also why the scored window is shorter than the round: a flat
// band is not a tracking task, and letting it into the averages would
// flatter every patient by the same arbitrary amount.

/** Dead straight for this long. */
export const LEAD_IN_MS = 5000;
/** Then the amplitude eases from 0 to full over this. */
export const RAMP_MS = 3000;
/** Nothing before this is scored. */
export const SCORED_FROM_MS = LEAD_IN_MS + RAMP_MS;

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
   * The centreline is a main sine plus a second one at
   * `waves * secondRatio`, weighted `1 - secondShare` and `secondShare`.
   *
   * The second sine started as decoration — something to stop the path
   * repeating visibly. It is load-bearing now. A single sine spends
   * most of its time near its own extremes, which is exactly where a
   * patient can park a still hand and be covered by the passing lane;
   * the second sine wiggles through those turnarounds and breaks the
   * dwell up. Raising either number shortens the dwell and costs
   * amplitude, because slope scales with frequency and the hand-speed
   * ceiling is enforced.
   */
  secondShare: number;
  secondRatio: number;
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
   * THE ANTI-CHEAT INVARIANT: the lane's half-width may not exceed this
   * share of the wave's amplitude.
   *
   * A patient can pass this game by holding their hand still whenever
   * the lane is wide enough to keep covering one fixed height as it
   * sweeps past — and that is decided by half-width OVER amplitude,
   * nothing else. Sizing the lane from the kite alone is not enough,
   * because the hand-speed ceiling shrinks the amplitude for some
   * patients (a short arm on a tall canvas) and the cheat reopens for
   * exactly them.
   *
   * So the lane is capped here too, and when the cap bites the KITE is
   * scaled down with it, which keeps `width = factor x kite height`
   * true and the kite visibly fitting its lane.
   */
  maxHalfOverAmp: number;
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
    // An UPPER BOUND now, not the answer: maxHalfOverAmp below almost
    // always scales the lane — and the kite with it — down from here.
    kiteFraction: 0.13,
    // Back to the floor the brief allows. The lane's width is fixed by
    // the anti-cheat cap below, NOT by this factor, so lowering it does
    // not narrow the lane — it enlarges the kite inside the same lane.
    // 1.6 is as big as the kite can be made without the lane ceasing to
    // read as a lane around it.
    widthFactorMin: 1.6,
    widthFactorMax: 2.1,
    waves: 2,
    secondShare: 0.25,
    secondRatio: 1.7,
    // 0.16 -> 0.165. Measured, not derived: this is the value at which
    // the lane comes out about a fifth wider across the range of reach
    // geometries, with the width factors above raised in step so the
    // kite keeps its size.
    maxHalfOverAmp: 0.2475,
    // 90% of the room, leaving a tenth as margin rather than letting
    // the corridor's edge kiss the measured limit of the reach at every
    // peak — the reach box is an estimate from three held points, not a
    // survey.
    curveAmp: 0.9,
    maxHandSpeedArmPerSec: 0.6,
    // Slower again. A deeper wave is what stops a still hand passing —
    // the lane is capped as a share of the amplitude, so a bigger
    // amplitude buys a bigger lane and a bigger kite at the same
    // cheat-resistance — and depth is bought with scroll speed.
    scrollPerSec: 0.04,
  },
  2: {
    id: 2,
    label: "Level 2",
    // A smaller kite in a lane about twice its height — so the margin
    // for error is roughly half — and a deeper curve, which its higher
    // hand-speed ceiling is what pays for.
    kiteFraction: 0.09,
    // 1.6-2.1 x 1.19, in step with maxHalfOverAmp below — see level 1.
    widthFactorMin: 1.9,
    widthFactorMax: 2.5,
    secondShare: 0.25,
    // Livelier than level 1's: it shortens the turnaround a still hand
    // can sit in, at the cost of some amplitude.
    secondRatio: 2.6,
    // Two waves, not three. A third wave costs amplitude twice over —
    // the slope scales with frequency, so the speed ceiling pushes the
    // curve down further, and a shallower curve was the complaint.
    waves: 2,
    curveAmp: 0.95,
    // 0.14 x 1.2. Still tighter than level 1's: this ratio is the only
    // thing that decides whether a still hand can pass, and level 2's
    // bar is 25% against level 1's 35%.
    // 0.14 -> 0.158, measured the same way. Still tighter than level
    // 1's: this ratio is the only thing that decides whether a still
    // hand can pass, and level 2's bar is 25% against level 1's 35%.
    // 0.158 x ~1.62. The sizes are the requirement here and this
    // ratio is the only thing that sets them, so it is chosen to
    // deliver a 1.5x lane and left there — the still-hand score is
    // reported rather than used as a brake.
    maxHalfOverAmp: 0.256,
    maxHandSpeedArmPerSec: 0.9,
    // Slower than level 1's, which looks wrong for a harder level and
    // is not: level 2's difficulty is its narrower lane and its deeper
    // wave, and the deeper wave is only affordable under its hand-speed
    // ceiling at this scroll. It still demands 0.90 arm/s of vertical
    // hand speed against level 1's 0.60.
    scrollPerSec: 0.035,
  },
};

export const DEFAULT_LEVEL: LevelId = 1;

export function levelById(id: number): KiteLevel {
  return LEVELS[(id === 2 ? 2 : 1) as LevelId];
}
