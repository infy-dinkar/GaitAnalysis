// getGaitCycleBlock — pure presentational derivation for the
// "Gait Cycle %" visual block.
//
// Reads the 5 additive keys the backend `_gait_cycle_percentages`
// helper writes into `metrics_clean` / `metrics_total`, plus the
// existing `stride_cv` (variability) and `cadence`. Everything here
// is a plain function — no data fetching, no side effects.
//
// Cadence's "expected" marker is DISPLAY-ONLY and lives here — set
// to 120 to match the doctor-validated reference system that this
// block was aligned against. `NORMAL_RANGES.cadence` in
// lib/gait/metrics.ts and the engine's obs-string range remain
// untouched (both still used by their own callers for OK / low /
// high grading of individual sessions).
import type { MetricsBlockDTO } from "@/lib/api";

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

/** Derive the block from up-to-two MetricsBlockDTOs. Prefers the
 *  primary block's phase keys (stance/swing/DS); falls back to the
 *  secondary block per key when the primary's is null. Cadence +
 *  variability follow the same per-key fallback. This is the
 *  fix for the recorded-video path where the pass-validation
 *  gates zero out `metrics_clean` but `metrics_total` still has
 *  a usable signal — old behaviour would pick clean-only and
 *  show "not computed"; now the block shows whichever numbers
 *  are actually present.
 *
 *  Old callers passing a single block still work: undefined
 *  secondary → per-key fallback is a no-op. */
export function getGaitCycleBlock(
  primary: MetricsBlockDTO | null | undefined,
  secondary?: MetricsBlockDTO | null,
): GaitCycleBlockData {
  const p = primary ?? ({} as Partial<MetricsBlockDTO>);
  const s = secondary ?? ({} as Partial<MetricsBlockDTO>);
  const pick = <K extends keyof MetricsBlockDTO>(
    k: K,
  ): number | null => n((p[k] ?? s[k]) as number | null | undefined);

  const stanceL = pick("stance_pct_left");
  const stanceR = pick("stance_pct_right");
  const swingL = pick("swing_pct_left");
  const swingR = pick("swing_pct_right");

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

  // Cadence — same per-key fallback as the phase keys. Expected =
  // display-only marker matching the doctor-validated reference
  // system. NOT read from NORMAL_RANGES (which is a session-grading
  // range, not a single reference value).
  const cadence = pick("cadence");
  const cadenceExpected = 120;

  return {
    left: { stancePct: stanceL, swingPct: swingL },
    right: { stancePct: stanceR, swingPct: swingR },
    doubleSupport: pick("double_support_pct"),
    // Variability = existing stride_cv (percent). Do NOT recompute.
    variabilityPct: pick("stride_cv"),
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
