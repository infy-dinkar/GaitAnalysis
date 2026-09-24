// Fruit Harvest's half of the report payload.
//
// Lifted out of FruitHarvestGame.tsx unchanged — same keys, same order,
// same arithmetic. It is a plain function in a plain module so it can
// be exercised directly, without a browser, a camera or React: the
// shape of this object is what a clinician reads months later, and a
// renamed key fails silently (GamesBody just prints "—").
//
// The shell supplies `calibration` already converted to arm lengths,
// because the three holds are the same for every game.

import { meanCollectSec, pct } from "@/lib/games/gameMetrics";
import type { GameResult, MetricsContext } from "@/lib/games/gameDefinition";

export interface FruitHarvestResult extends GameResult {
  harvested: number;
  missed: number;
}

export function buildFruitHarvestMetrics({
  m,
  result,
  roundMs,
  calibration,
}: MetricsContext<FruitHarvestResult>): Record<string, unknown> {
  const total = m.harvested + m.missed;
  return {
    game: "fruit_harvest",
    level: result.level,
    duration_sec: Math.round(roundMs / 1000),
    harvested: m.harvested,
    missed: m.missed,
    accuracy_pct: pct(m.harvested, total),
    avg_collect_sec: meanCollectSec(m.collectSecs),
    max_abduction_deg: m.maxAbductionDeg,
    // No max_adduction_deg: see lib/games/gameMetrics.ts. The
    // across-body reach count below is what the game can honestly
    // report for adduction.
    abduction_low_confidence: m.abductionLowConfidence,
    zone_hits: m.zoneHits,
    first_half_accuracy_pct: pct(m.firstHalf.hit, m.firstHalf.total),
    second_half_accuracy_pct: pct(m.secondHalf.hit, m.secondHalf.total),
    calibration,
    started_at_ms: m.startedAtMs,
  };
}
