// Fruit Harvest difficulty levels.
//
// Every tunable the scene reads lives here. The scene hard-codes none
// of them, so a new level is a new entry in LEVELS and nothing else.
//
// The BASE_* values are the game exactly as it shipped: level 1 keeps
// that timing and only enlarges the fruit, level 2 keeps that size and
// only quickens the timing. One axis at a time, so a patient moving
// between them meets one new demand rather than two.

import type { BaseLevel, LevelId } from "@/lib/games/gameDefinition";

export type { LevelId };

/**
 * Round length. Fixed at 60 s for every level — levels change what
 * happens during the round, never how long it lasts.
 *
 * It lives here rather than in the scene because the React shell needs
 * it too (for `duration_sec` on the report), and the scene file imports
 * Phaser. Importing it from there dragged 1.3 MB into the shell's
 * chunk and made the scene's dynamic import pointless.
 */
export const ROUND_MS = 60_000;

export interface LevelConfig extends BaseLevel {
  id: LevelId;
  /** Shown on the HUD and the result screen. */
  label: string;
  /**
   * Multiplier on the fruit's on-screen size AND on the hit radius, so
   * a bigger fruit stays proportionally as easy to touch — enlarging
   * the art without the radius would make level 1 look easier while
   * playing exactly as hard.
   */
  fruitScale: number;
  /** How long a fruit waits to be picked before it falls away. */
  ttlMs: number;
  /** Minimum gap between spawns. */
  spawnGapMs: number;
  /** Most fruit on screen at once. */
  maxFruit: number;
}

/** The game as it shipped — level 1's timing and level 2's size. */
export const BASE_TTL_MS = 5200;
export const BASE_SPAWN_GAP_MS = 600;
export const BASE_MAX_FRUIT = 3;

/** Fruit and hit sizes as fractions of the canvas unit, at scale 1. */
export const BASE_FRUIT_FRACTION = 0.13;
export const BASE_HIT_FRACTION = 0.075;

export const LEVELS: Record<LevelId, LevelConfig> = {
  1: {
    id: 1,
    label: "Level 1",
    // 1.3x — easier to see and to reach for, same rhythm as before.
    fruitScale: 1.3,
    ttlMs: BASE_TTL_MS,             // 5200
    spawnGapMs: BASE_SPAWN_GAP_MS,  // 600
    maxFruit: BASE_MAX_FRUIT,       // 3
  },
  2: {
    id: 2,
    label: "Level 2",
    fruitScale: 1.0,
    // 65% of the base: 3380 ms. Still comfortably longer than the
    // 300 ms wobble warning, so the warning keeps its meaning.
    ttlMs: Math.round(BASE_TTL_MS * 0.65),          // 3380
    // 60% of the base: 360 ms.
    spawnGapMs: Math.round(BASE_SPAWN_GAP_MS * 0.6), // 360
    maxFruit: BASE_MAX_FRUIT,       // 3 — unchanged
  },
};

export const DEFAULT_LEVEL: LevelId = 1;

export function levelById(id: number): LevelConfig {
  return LEVELS[(id === 2 ? 2 : 1) as LevelId];
}
