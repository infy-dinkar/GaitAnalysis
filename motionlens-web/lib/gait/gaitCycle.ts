// getGaitCycleBlock — pure presentational derivation for the
// "Gait Cycle %" visual block.
//
// Reads the 5 additive keys the backend `_gait_cycle_percentages`
// helper writes into `metrics_clean` / `metrics_total`, plus the
// existing `stride_cv` (variability) and `cadence`. Everything here
// is a plain function — no data fetching, no side effects.
//
// Cadence's "expected" upper-bound comes from the shared TS
// `NORMAL_RANGES.cadence` (defined in lib/gait/metrics.ts) — the
// engine has a different hard-coded 100-120 range for its
// text-observation obs strings, but we deliberately reuse the TS
// constant so the on-screen "[expected]" pill is a single source
// for display. The engine's obs text is left untouched.

import type { MetricsBlockDTO } from "@/lib/api";
import { NORMAL_RANGES } from "@/lib/gait/metrics";

export interface GaitCycleSide {
  stancePct: number | null;
  swingPct: number | null;
}

export interface GaitCycleBlockData {
  left: GaitCycleSide;
  right: GaitCycleSide;
  doubleSupport: number | null;
  variabilityPct: number | null;
  optimumDeviationPct: number | null;
  cadence: number | null;
  cadenceExpected: number | null;
}

function n(v: number | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Derive the block from a MetricsBlockDTO (metrics_clean preferred).
 *  Returns nulls gracefully when the additive keys are absent, so
 *  old saved reports render nothing rather than crashing. */
export function getGaitCycleBlock(metrics: MetricsBlockDTO | null | undefined): GaitCycleBlockData {
  const m = metrics ?? ({} as Partial<MetricsBlockDTO>);
  const stanceL = n(m.stance_pct_left);
  const stanceR = n(m.stance_pct_right);
  const swingL = n(m.swing_pct_left);
  const swingR = n(m.swing_pct_right);

  // Optimum-deviation = how far the mean stance % is from the
  // clinical target of 60 % (typical adult single-limb stance
  // portion of one gait cycle). Uses whichever sides are present.
  const availableStances = [stanceL, stanceR].filter(
    (v): v is number => v !== null,
  );
  const meanStance =
    availableStances.length > 0
      ? availableStances.reduce((a, b) => a + b, 0) / availableStances.length
      : null;
  const optimumDeviationPct =
    meanStance !== null
      ? Math.round(Math.abs(meanStance - 60) * 10) / 10
      : null;

  // Cadence — pass through from the same metrics block. Expected =
  // upper bound of the shared TS NORMAL_RANGES.cadence constant, so
  // the report card shows something like "[130]".
  const cadence = n(m.cadence);
  const cadenceExpected = NORMAL_RANGES.cadence[1];

  return {
    left: { stancePct: stanceL, swingPct: swingL },
    right: { stancePct: stanceR, swingPct: swingR },
    doubleSupport: n(m.double_support_pct),
    // Variability = existing stride_cv (percent). Do NOT recompute.
    variabilityPct: n(m.stride_cv),
    optimumDeviationPct,
    cadence,
    cadenceExpected,
  };
}

/** True when the block has enough data to be worth rendering. */
export function gaitCycleBlockHasData(b: GaitCycleBlockData): boolean {
  return (
    b.left.stancePct !== null
    || b.right.stancePct !== null
    || b.doubleSupport !== null
    || b.variabilityPct !== null
    || b.cadence !== null
  );
}
