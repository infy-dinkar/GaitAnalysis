// What a Kite Flying round measures, and what gets saved.
//
// Two different things are being measured and they come from two
// different sources on purpose:
//
//   • WHERE the hand was — corridor time and deviation — is taken from
//     the RAW palm, like everything else here.
//   • WHAT THE PATIENT SAW — the kite wobbling, the kite falling — is
//     driven by the DRAWN cursor, because feedback has to match what is
//     on screen or the game lies to them.
//
// So `kite_falls` is a count of what happened to the kite, while
// `time_in_corridor_pct` is a measure of the hand. They will differ
// slightly, and that is correct.
//
// The smoothness half of the round lives in lib/games/kiteMovement.ts.

import type { GameResult, MetricsContext } from "@/lib/games/gameDefinition";
import type { Hand } from "@/lib/games/handTracker";
import { smoothnessScore } from "@/lib/games/kiteMovement";

export interface KiteResult extends GameResult {
  hand: Hand;
  /** Time the raw palm was inside the corridor, and total time it was
   *  measurable at all — the hand out of frame and the fall/respawn
   *  window are in neither. */
  insideMs: number;
  measuredMs: number;
  falls: number;
  /** Running sum and count of the deviation ratio, so the mean can be
   *  taken without keeping the whole trace. */
  devSum: number;
  devCount: number;
  firstHalf: { inside: number; total: number };
  secondHalf: { inside: number; total: number };
  /** From MovementAnalyser. Null when the patient barely moved. */
  peaksPerSec: number | null;
  peakCount: number;
  movingSec: number;
}

export function blankKiteResult(
  level: KiteResult["level"],
  hand: Hand,
): KiteResult {
  return {
    level,
    hand,
    insideMs: 0,
    measuredMs: 0,
    falls: 0,
    devSum: 0,
    devCount: 0,
    firstHalf: { inside: 0, total: 0 },
    secondHalf: { inside: 0, total: 0 },
    peaksPerSec: null,
    peakCount: 0,
    movingSec: 0,
  };
}

export function pct(part: number, whole: number): number {
  return whole > 0 ? Math.round((part / whole) * 100) : 0;
}

export function buildKiteMetrics({
  m,
  result,
  roundMs,
  calibration,
}: MetricsContext<KiteResult>): Record<string, unknown> {
  const pps = result.peaksPerSec;
  return {
    game: "kite_flying",
    level: result.level,
    hand: result.hand,
    duration_sec: Math.round(roundMs / 1000),

    // ── Holding the line
    time_in_corridor_pct: pct(result.insideMs, result.measuredMs),
    // Mean distance from the centreline as a percentage of the
    // corridor's HALF-WIDTH: 0 is dead centre, 100 is the edge, over
    // 100 means the average position was outside. A ratio rather than a
    // distance, so a wide corridor and a narrow one compare.
    mean_deviation_pct: result.devCount > 0
      ? Math.round((result.devSum / result.devCount) * 100)
      : null,
    kite_falls: result.falls,

    // ── How the hand moved. Null when the patient barely moved — see
    //    MIN_MOVING_SEC in kiteMovement.ts.
    velocity_peaks_per_sec: pps === null ? null : Math.round(pps * 100) / 100,
    smoothness_score: pps === null ? null : smoothnessScore(pps),

    first_half_time_in_corridor_pct: pct(
      result.firstHalf.inside,
      result.firstHalf.total,
    ),
    second_half_time_in_corridor_pct: pct(
      result.secondHalf.inside,
      result.secondHalf.total,
    ),

    // Recorded and saved, never shown — exactly as the other two games
    // do it. A frontal 2-D camera during a moving task cannot produce a
    // degree figure fit to sit beside the Biomechanics module's.
    max_abduction_deg: m.maxAbductionDeg,
    abduction_low_confidence: m.abductionLowConfidence,

    calibration,
    started_at_ms: m.startedAtMs,
  };
}
