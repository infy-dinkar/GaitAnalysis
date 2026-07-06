// Progression ladders — 5-step difficulty ladder per exercise.
//
// Each ladder is an ordered array of config variants. Level 3 = the
// current default (matches what's hardcoded in every page today), so
// existing sessions and public-flow renders behave identically.
// Level 0 = easiest, level 4 = hardest.
//
// Consumers:
//   • useProgressionLevel(patientId, slug, ladder) resolves the
//     patient's current level index from session history, returns
//     the corresponding config.
//   • Pages import the appropriate ladder + call the hook OR fall
//     back to defaultLevelConfig(ladder) for public-flow rendering.

/** @typedef {{level: number, [k: string]: any}} LadderEntry */

// ─── Knee ──────────────────────────────────────────────────────
export const SQUAT_LADDER = [
  { level: 0, topThreshold: 170, depthThreshold: 155, minAmplitude: 15, targetReps: 5,  pointsPerRep: 6 },
  { level: 1, topThreshold: 165, depthThreshold: 140, minAmplitude: 30, targetReps: 8,  pointsPerRep: 8 },
  { level: 2, topThreshold: 165, depthThreshold: 125, minAmplitude: 40, targetReps: 10, pointsPerRep: 9 },
  { level: 3, topThreshold: 160, depthThreshold: 110, minAmplitude: 50, targetReps: 10, pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 160, depthThreshold: 100, minAmplitude: 55, targetReps: 12, pointsPerRep: 12 },
];

export const MINI_SQUAT_LADDER = [
  { level: 0, topThreshold: 170, depthThreshold: 160, minAmplitude: 10, targetReps: 8,  pointsPerRep: 5 },
  { level: 1, topThreshold: 170, depthThreshold: 150, minAmplitude: 18, targetReps: 10, pointsPerRep: 6 },
  { level: 2, topThreshold: 165, depthThreshold: 145, minAmplitude: 22, targetReps: 12, pointsPerRep: 7 },
  { level: 3, topThreshold: 165, depthThreshold: 140, minAmplitude: 25, targetReps: 12, pointsPerRep: 8 }, // = default
  { level: 4, topThreshold: 165, depthThreshold: 130, minAmplitude: 30, targetReps: 15, pointsPerRep: 10 },
];

export const KNEE_EXTENSION_LADDER = [
  { level: 0, hitRadiusMultiplier: 1.8, pointsPerHit: 5,  pointsPerMiss: -1 },
  { level: 1, hitRadiusMultiplier: 1.5, pointsPerHit: 8,  pointsPerMiss: -1 },
  { level: 2, hitRadiusMultiplier: 1.4, pointsPerHit: 9,  pointsPerMiss: -2 },
  { level: 3, hitRadiusMultiplier: 1.3, pointsPerHit: 10, pointsPerMiss: -2 }, // = default
  { level: 4, hitRadiusMultiplier: 1.1, pointsPerHit: 12, pointsPerMiss: -3 },
];

export const STEP_UP_LADDER = [
  { level: 0, topThreshold: 170, depthThreshold: 145, minAmplitude: 20, targetReps: 6,  pointsPerRep: 6 },
  { level: 1, topThreshold: 168, depthThreshold: 135, minAmplitude: 30, targetReps: 8,  pointsPerRep: 8 },
  { level: 2, topThreshold: 165, depthThreshold: 128, minAmplitude: 35, targetReps: 10, pointsPerRep: 9 },
  { level: 3, topThreshold: 165, depthThreshold: 120, minAmplitude: 40, targetReps: 10, pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 165, depthThreshold: 110, minAmplitude: 50, targetReps: 12, pointsPerRep: 12 },
];

export const WALL_SIT_LADDER = [
  { level: 0, min: 100, max: 130, targetHoldMs: 10_000, hysteresis: 5 },
  { level: 1, min:  90, max: 115, targetHoldMs: 15_000, hysteresis: 4 },
  { level: 2, min:  85, max: 105, targetHoldMs: 20_000, hysteresis: 3 },
  { level: 3, min:  80, max: 100, targetHoldMs: 30_000, hysteresis: 3 }, // = default
  { level: 4, min:  75, max:  95, targetHoldMs: 45_000, hysteresis: 3 },
];

export const SINGLE_LEG_SQUAT_LADDER = [
  { level: 0, topThreshold: 170, depthThreshold: 150, minAmplitude: 15, targetReps: 4,  pointsPerRep: 8 },
  { level: 1, topThreshold: 165, depthThreshold: 140, minAmplitude: 20, targetReps: 6,  pointsPerRep: 10 },
  { level: 2, topThreshold: 162, depthThreshold: 130, minAmplitude: 28, targetReps: 8,  pointsPerRep: 11 },
  { level: 3, topThreshold: 160, depthThreshold: 120, minAmplitude: 35, targetReps: 8,  pointsPerRep: 12 }, // = default
  { level: 4, topThreshold: 160, depthThreshold: 110, minAmplitude: 45, targetReps: 10, pointsPerRep: 14 },
];

// ─── Hip ───────────────────────────────────────────────────────
export const PELVIC_HOLD_LADDER = [
  { level: 0, min: -10, max: 10, targetHoldMs: 10_000, hysteresis: 2 },
  { level: 1, min:  -8, max:  8, targetHoldMs: 15_000, hysteresis: 2 },
  { level: 2, min:  -6, max:  6, targetHoldMs: 20_000, hysteresis: 1.5 },
  { level: 3, min:  -5, max:  5, targetHoldMs: 25_000, hysteresis: 1.5 }, // = default
  { level: 4, min:  -4, max:  4, targetHoldMs: 35_000, hysteresis: 1 },
];

export const HIP_ABDUCTION_LADDER = [
  { level: 0, hitRadiusMultiplier: 1.7, pointsPerHit: 6,  pointsPerMiss: -1 },
  { level: 1, hitRadiusMultiplier: 1.5, pointsPerHit: 8,  pointsPerMiss: -1 },
  { level: 2, hitRadiusMultiplier: 1.35, pointsPerHit: 9, pointsPerMiss: -2 },
  { level: 3, hitRadiusMultiplier: 1.25, pointsPerHit: 10, pointsPerMiss: -2 }, // = default
  { level: 4, hitRadiusMultiplier: 1.1, pointsPerHit: 12, pointsPerMiss: -3 },
];

export const WEIGHT_SHIFT_LADDER = [
  { level: 0, halfWidth: 0.25, dwellMs: 2500 },
  { level: 1, halfWidth: 0.20, dwellMs: 2000 },
  { level: 2, halfWidth: 0.17, dwellMs: 1750 },
  { level: 3, halfWidth: 0.15, dwellMs: 1500 }, // = default
  { level: 4, halfWidth: 0.12, dwellMs: 1250 },
];

export const BRIDGE_LADDER = [
  { level: 0, topThreshold: 130, depthThreshold: 100, minAmplitude: 25, targetReps: 6,  pointsPerRep: 6 },
  { level: 1, topThreshold: 140, depthThreshold: 108, minAmplitude: 35, targetReps: 8,  pointsPerRep: 8 },
  { level: 2, topThreshold: 145, depthThreshold: 112, minAmplitude: 45, targetReps: 10, pointsPerRep: 9 },
  { level: 3, topThreshold: 150, depthThreshold: 115, minAmplitude: 50, targetReps: 10, pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 155, depthThreshold: 118, minAmplitude: 55, targetReps: 12, pointsPerRep: 12 },
];

export const MARCHING_LADDER = [
  { level: 0, bpm:  60, perfectWindowMs: 250, goodWindowMs: 500, pointsPerfect: 5, pointsGood: 3 },
  { level: 1, bpm:  75, perfectWindowMs: 200, goodWindowMs: 400, pointsPerfect: 7, pointsGood: 4 },
  { level: 2, bpm:  85, perfectWindowMs: 175, goodWindowMs: 375, pointsPerfect: 8, pointsGood: 4 },
  { level: 3, bpm:  90, perfectWindowMs: 150, goodWindowMs: 350, pointsPerfect: 10, pointsGood: 5 }, // = default
  { level: 4, bpm: 100, perfectWindowMs: 120, goodWindowMs: 300, pointsPerfect: 12, pointsGood: 6 },
];

export const LATERAL_STEP_LADDER = [
  { level: 0, topThreshold: 168, depthThreshold: 148, minAmplitude: 15, targetReps: 6,  pointsPerRep: 7 },
  { level: 1, topThreshold: 165, depthThreshold: 138, minAmplitude: 20, targetReps: 8,  pointsPerRep: 8 },
  { level: 2, topThreshold: 162, depthThreshold: 130, minAmplitude: 25, targetReps: 10, pointsPerRep: 9 },
  { level: 3, topThreshold: 160, depthThreshold: 125, minAmplitude: 30, targetReps: 10, pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 160, depthThreshold: 115, minAmplitude: 35, targetReps: 12, pointsPerRep: 12 },
];

// ─── Back ──────────────────────────────────────────────────────
export const POSTURE_HOLD_LADDER = [
  { level: 0, min: 0, max: 18, targetHoldMs: 10_000, hysteresis: 3 },
  { level: 1, min: 0, max: 15, targetHoldMs: 15_000, hysteresis: 2.5 },
  { level: 2, min: 0, max: 13, targetHoldMs: 18_000, hysteresis: 2 },
  { level: 3, min: 0, max: 12, targetHoldMs: 20_000, hysteresis: 2 }, // = default
  { level: 4, min: 0, max: 10, targetHoldMs: 25_000, hysteresis: 1.5 },
];

export const BACK_EXTENSION_LADDER = [
  { level: 0, topThreshold:  8, depthThreshold: 2, minAmplitude: 4, targetReps: 5,  pointsPerRep: 6 },
  { level: 1, topThreshold: 10, depthThreshold: 4, minAmplitude: 5, targetReps: 6,  pointsPerRep: 8 },
  { level: 2, topThreshold: 11, depthThreshold: 4, minAmplitude: 6, targetReps: 7,  pointsPerRep: 9 },
  { level: 3, topThreshold: 12, depthThreshold: 5, minAmplitude: 7, targetReps: 8,  pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 15, depthThreshold: 5, minAmplitude: 9, targetReps: 10, pointsPerRep: 12 },
];

export const SIDE_BEND_LADDER = [
  { level: 0, hitRadiusMultiplier: 1.7, pointsPerHit: 6,  pointsPerMiss: -1 },
  { level: 1, hitRadiusMultiplier: 1.5, pointsPerHit: 8,  pointsPerMiss: -1 },
  { level: 2, hitRadiusMultiplier: 1.4, pointsPerHit: 9,  pointsPerMiss: -2 },
  { level: 3, hitRadiusMultiplier: 1.3, pointsPerHit: 10, pointsPerMiss: -2 }, // = default
  { level: 4, hitRadiusMultiplier: 1.1, pointsPerHit: 12, pointsPerMiss: -3 },
];

export const BIRD_DOG_LADDER = [
  { level: 0, achievedThresholdPct: 55, requiredHoldMs: 2000, tolerance: 30 },
  { level: 1, achievedThresholdPct: 60, requiredHoldMs: 3000, tolerance: 25 },
  { level: 2, achievedThresholdPct: 65, requiredHoldMs: 3500, tolerance: 22 },
  { level: 3, achievedThresholdPct: 70, requiredHoldMs: 4000, tolerance: 20 }, // = default
  { level: 4, achievedThresholdPct: 80, requiredHoldMs: 5000, tolerance: 15 },
];

export const HIP_HINGE_LADDER = [
  { level: 0, topThreshold: 20, depthThreshold:  6, minAmplitude: 12, targetReps: 6,  pointsPerRep: 6 },
  { level: 1, topThreshold: 25, depthThreshold:  8, minAmplitude: 16, targetReps: 8,  pointsPerRep: 8 },
  { level: 2, topThreshold: 28, depthThreshold: 10, minAmplitude: 18, targetReps: 9,  pointsPerRep: 9 },
  { level: 3, topThreshold: 30, depthThreshold: 10, minAmplitude: 20, targetReps: 10, pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 35, depthThreshold: 12, minAmplitude: 25, targetReps: 12, pointsPerRep: 12 },
];

export const CAT_COW_LADDER = [
  { level: 0, accuracyTolerance: 0.16, smoothnessTolerance: 0.0015, pointsPerSample: 1 },
  { level: 1, accuracyTolerance: 0.13, smoothnessTolerance: 0.0013, pointsPerSample: 1 },
  { level: 2, accuracyTolerance: 0.11, smoothnessTolerance: 0.0011, pointsPerSample: 1 },
  { level: 3, accuracyTolerance: 0.10, smoothnessTolerance: 0.001,  pointsPerSample: 1 }, // = default
  { level: 4, accuracyTolerance: 0.07, smoothnessTolerance: 0.0008, pointsPerSample: 1 },
];

// ─── Shoulder ──────────────────────────────────────────────────
export const SHOULDER_RAISE_LADDER = [
  { level: 0, hitRadiusMultiplier: 1.6, pointsPerHit: 6,  pointsPerMiss: -1 },
  { level: 1, hitRadiusMultiplier: 1.4, pointsPerHit: 8,  pointsPerMiss: -1 },
  { level: 2, hitRadiusMultiplier: 1.3, pointsPerHit: 9,  pointsPerMiss: -2 },
  { level: 3, hitRadiusMultiplier: 1.2, pointsPerHit: 10, pointsPerMiss: -2 }, // = default
  { level: 4, hitRadiusMultiplier: 1.0, pointsPerHit: 12, pointsPerMiss: -3 },
];

export const WALL_CLOCK_LADDER = [
  { level: 0, hitRadiusMultiplier: 1.7, pointsPerHit: 6,  pointsPerMiss: -1 },
  { level: 1, hitRadiusMultiplier: 1.5, pointsPerHit: 8,  pointsPerMiss: -1 },
  { level: 2, hitRadiusMultiplier: 1.35, pointsPerHit: 9, pointsPerMiss: -2 },
  { level: 3, hitRadiusMultiplier: 1.25, pointsPerHit: 10, pointsPerMiss: -2 }, // = default
  { level: 4, hitRadiusMultiplier: 1.05, pointsPerHit: 12, pointsPerMiss: -3 },
];

export const PENDULUM_LADDER = [
  { level: 0, accuracyTolerance: 0.10, smoothnessTolerance: 0.0015, pointsPerSample: 1 },
  { level: 1, accuracyTolerance: 0.08, smoothnessTolerance: 0.0013, pointsPerSample: 1 },
  { level: 2, accuracyTolerance: 0.07, smoothnessTolerance: 0.0011, pointsPerSample: 1 },
  { level: 3, accuracyTolerance: 0.06, smoothnessTolerance: 0.001,  pointsPerSample: 1 }, // = default
  { level: 4, accuracyTolerance: 0.04, smoothnessTolerance: 0.0008, pointsPerSample: 1 },
];

export const WALL_SLIDE_LADDER = [
  { level: 0, min: 120, max: 150, targetHoldMs: 8000,  hysteresis: 4 },
  { level: 1, min: 130, max: 155, targetHoldMs: 12000, hysteresis: 4 },
  { level: 2, min: 135, max: 158, targetHoldMs: 16000, hysteresis: 3 },
  { level: 3, min: 140, max: 160, targetHoldMs: 20000, hysteresis: 3 }, // = default
  { level: 4, min: 145, max: 165, targetHoldMs: 30000, hysteresis: 2 },
];

export const EXTERNAL_ROTATION_LADDER = [
  { level: 0, topThreshold: 80, depthThreshold: 65, minAmplitude: 12, targetReps: 6,  pointsPerRep: 7 },
  { level: 1, topThreshold: 77, depthThreshold: 60, minAmplitude: 18, targetReps: 8,  pointsPerRep: 8 },
  { level: 2, topThreshold: 76, depthThreshold: 55, minAmplitude: 22, targetReps: 9,  pointsPerRep: 9 },
  { level: 3, topThreshold: 75, depthThreshold: 50, minAmplitude: 25, targetReps: 10, pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 72, depthThreshold: 45, minAmplitude: 30, targetReps: 12, pointsPerRep: 12 },
];

export const SCAPULAR_SET_LADDER = [
  { level: 0, topThreshold: 3, depthThreshold: 1, minAmplitude: 2, targetReps: 6,  pointsPerRep: 7 },
  { level: 1, topThreshold: 4, depthThreshold: 2, minAmplitude: 2, targetReps: 8,  pointsPerRep: 8 },
  { level: 2, topThreshold: 4.5, depthThreshold: 2, minAmplitude: 3, targetReps: 9, pointsPerRep: 9 },
  { level: 3, topThreshold: 5, depthThreshold: 2, minAmplitude: 3, targetReps: 10, pointsPerRep: 10 }, // = default
  { level: 4, topThreshold: 6, depthThreshold: 3, minAmplitude: 4, targetReps: 12, pointsPerRep: 12 },
];

/** Slug → ladder lookup. */
export const LADDERS_BY_SLUG = {
  "squat":              SQUAT_LADDER,
  "mini-squat":         MINI_SQUAT_LADDER,
  "knee-extension":     KNEE_EXTENSION_LADDER,
  "step-up":            STEP_UP_LADDER,
  "wall-sit":           WALL_SIT_LADDER,
  "single-leg-squat":   SINGLE_LEG_SQUAT_LADDER,
  "pelvic-hold":        PELVIC_HOLD_LADDER,
  "hip-abduction":      HIP_ABDUCTION_LADDER,
  "weight-shift":       WEIGHT_SHIFT_LADDER,
  "bridge":             BRIDGE_LADDER,
  "marching":           MARCHING_LADDER,
  "lateral-step":       LATERAL_STEP_LADDER,
  "posture-hold":       POSTURE_HOLD_LADDER,
  "back-extension":     BACK_EXTENSION_LADDER,
  "side-bend":          SIDE_BEND_LADDER,
  "bird-dog":           BIRD_DOG_LADDER,
  "hip-hinge":          HIP_HINGE_LADDER,
  "cat-cow":            CAT_COW_LADDER,
  "shoulder-raise":     SHOULDER_RAISE_LADDER,
  "wall-clock":         WALL_CLOCK_LADDER,
  "pendulum":           PENDULUM_LADDER,
  "wall-slide":         WALL_SLIDE_LADDER,
  "external-rotation":  EXTERNAL_ROTATION_LADDER,
  "scapular-set":       SCAPULAR_SET_LADDER,
};

/**
 * Level index of the "default" rung — the one that matches what the
 * page currently hardcodes. Public flow uses this when no history
 * exists. Locked to 3 across every ladder above (empirically the
 * "current default" position we designed the ladders around).
 */
export const DEFAULT_LEVEL_INDEX = 3;

/**
 * Look up a ladder by slug. Returns null when the slug is unknown.
 * @param {string} slug
 */
export function ladderOf(slug) {
  return LADDERS_BY_SLUG[slug] ?? null;
}

/**
 * Public-flow / patient-not-supplied config: the "default" level entry.
 * @param {LadderEntry[] | null} ladder
 */
export function defaultLevelConfig(ladder) {
  if (!ladder || ladder.length === 0) return null;
  const idx = Math.min(DEFAULT_LEVEL_INDEX, ladder.length - 1);
  return ladder[idx];
}
