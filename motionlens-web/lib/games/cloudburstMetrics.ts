// What a Cloudburst round measures, and what gets saved.
//
// The SHOULDER ANGLE half is not here: Cloudburst uses the shared
// MetricsRecorder unchanged (lib/games/gameMetrics.ts), whose sample()
// reads pose, not gameplay, and works for any game. Everything below is
// the gameplay half, and it travels on the RESULT object the scene
// builds at the end of the round rather than in the recorder — which is
// why no shared file had to change to add a second game.
//
// TWO KINDS OF CORRECTNESS. Catching water is a commission task:
// accuracy is hits over attempts. Avoiding lightning is an INHIBITION
// task: the score is for the things the patient did NOT do. They are
// reported separately and never averaged together, because a patient
// who catches everything and grabs every bolt is not "50% accurate" —
// they have one intact ability and one impaired one.

import type { GameResult, MetricsContext } from "@/lib/games/gameDefinition";

export type CloudburstZone = "same_side" | "across";

export interface CloudburstResult extends GameResult {
  caught: number;
  dropsMissed: number;
  lightningTouched: number;
  lightningAvoided: number;
  /** One entry per CAUGHT drop that had a measurable reaction. */
  reactionSecs: number[];
  firstHalf: { hit: number; total: number };
  secondHalf: { hit: number; total: number };
  zoneHits: Record<CloudburstZone, number>;
}

export function blankCloudburstResult(level: CloudburstResult["level"]): CloudburstResult {
  return {
    level,
    caught: 0,
    dropsMissed: 0,
    lightningTouched: 0,
    lightningAvoided: 0,
    reactionSecs: [],
    firstHalf: { hit: 0, total: 0 },
    secondHalf: { hit: 0, total: 0 },
    zoneHits: { same_side: 0, across: 0 },
  };
}

export function pct(hit: number, total: number): number {
  return total > 0 ? Math.round((hit / total) * 100) : 0;
}

export function meanSec(secs: number[]): number | null {
  if (secs.length === 0) return null;
  return Math.round((secs.reduce((a, b) => a + b, 0) / secs.length) * 100) / 100;
}

export function buildCloudburstMetrics({
  m,
  result,
  roundMs,
  calibration,
}: MetricsContext<CloudburstResult>): Record<string, unknown> {
  const dropsSeen = result.caught + result.dropsMissed;
  const boltsSeen = result.lightningTouched + result.lightningAvoided;
  return {
    game: "cloudburst",
    level: result.level,
    duration_sec: Math.round(roundMs / 1000),

    // ── Catching (commission)
    drops_caught: result.caught,
    drops_missed: result.dropsMissed,
    catch_accuracy_pct: pct(result.caught, dropsSeen),

    // ── Avoiding (inhibition). Kept apart from the numbers above.
    lightning_touched: result.lightningTouched,
    lightning_avoided: result.lightningAvoided,
    avoidance_pct: pct(result.lightningAvoided, boltsSeen),

    // Caught drops only — a drop that was never reached for has no
    // reaction time, and averaging in the misses would flatter a
    // patient who ignored the hard ones.
    avg_reaction_sec: meanSec(result.reactionSecs),

    // Where rising fall speed starts to cost accuracy.
    first_half_catch_accuracy_pct: pct(result.firstHalf.hit, result.firstHalf.total),
    second_half_catch_accuracy_pct: pct(result.secondHalf.hit, result.secondHalf.total),

    zone_hits: result.zoneHits,

    // Recorded and saved, never shown — exactly as Fruit Harvest does
    // it. A frontal 2-D camera during a reaching game cannot produce a
    // degree figure fit to sit beside the Biomechanics module's.
    max_abduction_deg: m.maxAbductionDeg,
    abduction_low_confidence: m.abductionLowConfidence,

    calibration,
    started_at_ms: m.startedAtMs,
  };
}
