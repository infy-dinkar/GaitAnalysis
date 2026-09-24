// Cloudburst difficulty levels.
//
// Same arrangement as Fruit Harvest's levels.ts: every tunable the
// scene reads lives here and the scene hard-codes none of them, so a
// third level is a new entry and nothing else.
//
// Levels change SIZE, FALL TIME, SPAWN RATE and HOW MUCH LIGHTNING.
// They do not change the round length, the on-screen cap or the speed
// ramp — those are the game, not the difficulty.

import type { BaseLevel, LevelId } from "@/lib/games/gameDefinition";

export type { LevelId };

/**
 * Round length. 60 s, deliberately the same as Fruit Harvest so the two
 * games' half-by-half accuracy numbers mean the same thing.
 *
 * Declared here rather than shared, because round length is a property
 * of a game (GameDefinition.roundMs), not of the platform.
 */
export const ROUND_MS = 60_000;

/** Item size and hit radius as fractions of the canvas unit, at
 *  itemScale 1. Slightly smaller than a fruit: a falling item is moving
 *  and does not need to be as large to be read. */
export const BASE_ITEM_FRACTION = 0.12;
export const BASE_HIT_FRACTION = 0.07;

/** Most items on screen at once, any type. */
export const MAX_ITEMS = 3;

/**
 * Speed multiplier over the round: 1.0 at the start, this at the end,
 * interpolated linearly on PLAYED time (held time is already excluded).
 *
 * It applies to items already falling as well as new ones, so the ramp
 * is felt continuously rather than in steps.
 */
export const SPEED_RAMP_TO = 1.6;

/** Cover-space height the item is given ABOVE the top of the reach box,
 *  so it is visible and trackable before it becomes catchable. */
export const SPAWN_LEAD = 0.1;

/**
 * Share of lightning AIMED at the lane the patient's hand is already
 * in, rather than placed at random.
 *
 * This is what turns avoiding from a decision into a movement: park the
 * hand in one lane and wait, and about half the bolts come to you. The
 * other half stays random, so the whole width still has to be watched.
 */
export const AIMED_LIGHTNING_FRACTION = 0.5;

/** A new bolt must clear any drop still near the spawn height by this
 *  many item widths in x, so a bolt is never laid on top of a drop
 *  where the two read as one object. */
export const MIN_LANE_SEPARATION = 1.15;

// ── Big centre strike.
//
// A whole-band hazard on a timer, aimed at the middle of the reach. It
// asks for something the falling items do not: commit to a direction
// and move the body out of the way.

/** Width of the struck band, as a fraction of the calibrated reach
 *  width, centred on the middle of that reach. */
export const STRIKE_BAND_FRACTION = 0.4;

/** Warning before the strike lands. Long enough to see it, read it and
 *  move a whole arm; short enough to still be a hurry. */
export const STRIKE_WARN_MS = 1500;

/**
 * The strike itself: ONE flash, never a strobe.
 *
 * PHOTOSENSITIVITY. Nothing in this game flashes faster than 3 Hz. The
 * warning glow pulses at ~0.67 Hz, the strike is a single 150 ms event
 * once every 9-15 s, the error tint is rate-limited (SAFE_FLASH_GAP_MS
 * in the scene), and no flash is full-screen white: the strike is
 * band-limited and warm, the error tint is partial red.
 */
export const STRIKE_FLASH_MS = 150;

export interface CloudburstLevel extends BaseLevel {
  /** Multiplier on the drawn size AND on the hit radius, so a bigger
   *  item stays proportionally as easy to touch. Same rule as Fruit
   *  Harvest — which means bigger lightning is also easier to touch by
   *  accident. That is intended: the halo the patient sees is the
   *  target, and a graze that is not penalised would teach the wrong
   *  thing. */
  itemScale: number;
  /** Time from spawn to the miss line at speed 1.0, in ms. The scene
   *  derives a velocity from this and the actual reach height, so the
   *  number means the same on any screen and any body. */
  fallMs: number;
  /** Minimum gap between spawns. */
  spawnGapMs: number;
  /** Share of spawns that are lightning rather than water. */
  lightningFraction: number;
  /** Gap between big centre strikes, sampled uniformly in this range.
   *  A range rather than a fixed period so the patient cannot learn to
   *  count the beat instead of watching for the warning. */
  strikeGapMinMs: number;
  strikeGapMaxMs: number;
}

export const LEVELS: Record<LevelId, CloudburstLevel> = {
  1: {
    id: 1,
    label: "Level 1",
    // Bigger and slower: the patient has time to see the item, decide
    // what it is, and move.
    itemScale: 1.3,
    fallMs: 3000,
    spawnGapMs: 900,
    // A third of everything falling is now a hazard. Below about this
    // the patient can catch on reflex and never has to decide.
    lightningFraction: 0.35,
    strikeGapMinMs: 12_000,
    strikeGapMaxMs: 15_000,
  },
  2: {
    id: 2,
    label: "Level 2",
    itemScale: 1.0,
    // ~73% of level 1's fall time, and a gap ~78% as long. At the end
    // of the ramp this is a 1375 ms crossing — fast, but still over a
    // second of warning.
    fallMs: 2200,
    spawnGapMs: 700,
    // Close to half: more hazard than reward, so catching has to be
    // chosen rather than done to everything that falls.
    lightningFraction: 0.45,
    strikeGapMinMs: 9_000,
    strikeGapMaxMs: 11_000,
  },
};

export const DEFAULT_LEVEL: LevelId = 1;

export function levelById(id: number): CloudburstLevel {
  return LEVELS[(id === 2 ? 2 : 1) as LevelId];
}
