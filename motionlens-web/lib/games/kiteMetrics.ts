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
import { SCORED_FROM_MS } from "@/lib/games/kiteLevels";
import type { KiteTotals } from "@/lib/games/kiteSession";

/**
 * The round's totals (from KiteSession) plus the movement analysis.
 *
 * Time inside is split by how WIDE the corridor was at the moment, not
 * only by when: the lane breathes between the level's two width
 * factors, and holding the line through a narrow stretch is a harder
 * thing than holding it through a wide one. Averaging the two hides
 * exactly the part worth seeing.
 */
export interface KiteResult extends GameResult, KiteTotals {
  hand: Hand;
  /** From MovementAnalyser. Null when the patient barely moved. */
  peaksPerSec: number | null;
  peakCount: number;
  movingSec: number;
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
    // The round is 60 s but the opening is a straight lead-in and a
    // fade-in, which is not a tracking task — nothing below is measured
    // over it. Recorded so a clinician comparing two rounds can see
    // what window the percentages are percentages OF.
    scored_sec: Math.round((roundMs - SCORED_FROM_MS) / 1000),

    // ── Holding the line
    time_in_corridor_pct: pct(result.insideMs, result.measuredMs),
    // Mean distance from the centreline as a percentage of the LOCAL
    // half-width: 0 is dead centre, 100 is the edge, over 100 means the
    // average position was outside. Local, because where the lane
    // narrows the same absolute error is a bigger one.
    mean_deviation_pct: result.devCount > 0
      ? Math.round((result.devSum / result.devCount) * 100)
      : null,
    kite_falls: result.falls,

    // Split at the level's median width factor. A patient who holds
    // the wide stretches and loses the narrow ones has a precision
    // problem; one who loses both has something else.
    narrow_time_in_corridor_pct: result.narrowMs > 0
      ? pct(result.narrowInsideMs, result.narrowMs)
      : null,
    wide_time_in_corridor_pct: result.wideMs > 0
      ? pct(result.wideInsideMs, result.wideMs)
      : null,

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
