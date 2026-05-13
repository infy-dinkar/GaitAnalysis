// SPPB (Short Physical Performance Battery) — composite test
// orchestrator types + 0-12 scoring + classification + interpretation.
//
// SPPB combines three sub-tests into a single assessment session:
//   1. Balance     — 3 stages (side-by-side, semi-tandem, tandem)
//   2. Gait Speed  — 4 m usual-pace walk, 2 trials (better of 2)
//   3. Chair Stand — 5x sit-to-stand (reuses C2 timing rules)
//
// Each component scores 0-4, summed into a 0-12 total. The cutoffs
// below are from the original Guralnik 1994 instrument and the EUGMS
// standardised SPPB protocol — DO NOT change without spec update.

import type { StageResult } from "@/lib/orthopedic/fourStageBalance";
import type { SitToStandResult } from "@/lib/orthopedic/sitToStand";

// ─── Per-component result types ────────────────────────────────

/** SPPB only uses the first 3 stages of the 4-stage balance protocol. */
export type BalanceStageKey = 1 | 2 | 3;

export interface BalanceComponentResult {
  stages: {
    1?: StageResult;
    2?: StageResult;
    3?: StageResult;
  };
  /** Highest stage held for the full 10 s. 0 if Stage 1 failed. */
  final_stage_completed: 0 | 1 | 2 | 3;
  /** Stage 3 (tandem) hold duration in seconds, used by SPPB scoring
   *  to distinguish 2 / 3 / 4-point outcomes. */
  stage3_hold_seconds: number;
  score: 0 | 1 | 2 | 3 | 4;
}

export interface GaitSpeedTrial {
  duration_sec: number;
  /** True when the trial was completed cleanly; false if the operator
   *  manually stopped or the patient didn't reach the marker. */
  completed: boolean;
  /** Wall-clock timestamp at start (ms) — kept for debug only. */
  started_at_ms: number;
}

export interface GaitSpeedComponentResult {
  trial1: GaitSpeedTrial | null;
  trial2: GaitSpeedTrial | null;
  /** Faster of the two completed trials, in seconds. Null when neither
   *  trial completed. */
  best_time_sec: number | null;
  /** 4.0 / best_time_sec, in m/s. Null when no completed trial. */
  speed_mps: number | null;
  score: 0 | 1 | 2 | 3 | 4;
}

export interface ChairStandComponentResult {
  trial: SitToStandResult;
  score: 0 | 1 | 2 | 3 | 4;
}

export type SPPBClassification =
  | "minimal_mild_limitation"      // 10-12
  | "moderate_limitation"          // 7-9
  | "severe_limitation";           // 0-6

export interface SPPBResult {
  patient_age: number | null;
  // ── Components ─────────────────────────────────────────────
  balance: BalanceComponentResult;
  gait_speed: GaitSpeedComponentResult;
  chair_stand: ChairStandComponentResult;
  // ── Aggregate ──────────────────────────────────────────────
  total_score: number;             // 0-12
  classification: SPPBClassification;
  interpretation: string;          // plain-language paragraph
  recommendation: string;          // clinical recommendation
}

// ─── Scoring functions (cutoffs from Guralnik 1994) ────────────

/** Balance score from stage-3 hold time + which stage was reached. */
export function scoreBalance(
  final_stage: 0 | 1 | 2 | 3,
  stage3_hold_seconds: number,
): 0 | 1 | 2 | 3 | 4 {
  if (final_stage === 0) return 0;                  // failed Stage 1
  if (final_stage === 1) return 1;                  // passed S1, failed S2
  if (final_stage === 2) {
    // Passed S2 but tandem (S3) held < 3 s
    return 2;
  }
  // final_stage === 3 → patient completed all three. Tandem hold
  // duration determines 2 / 3 / 4 points.
  if (stage3_hold_seconds < 3) return 2;
  if (stage3_hold_seconds < 10) return 3;
  return 4;
}

/** Gait speed score from walking speed in m/s. */
export function scoreGaitSpeed(speed_mps: number | null): 0 | 1 | 2 | 3 | 4 {
  if (speed_mps === null || speed_mps <= 0) return 0;
  if (speed_mps < 0.43) return 1;
  if (speed_mps < 0.60) return 2;
  if (speed_mps < 0.77) return 3;
  return 4;
}

/** Chair stand score from 5-rep total time in seconds.
 *  `completed = false` (couldn't do 5 reps in 60 s) → 0. */
export function scoreChairStand(
  total_seconds: number,
  completed: boolean,
): 0 | 1 | 2 | 3 | 4 {
  if (!completed) return 0;
  if (total_seconds > 60) return 0;
  if (total_seconds > 16.7) return 1;
  if (total_seconds > 13.7) return 2;
  if (total_seconds > 11.2) return 3;
  return 4;
}

export function classifyTotal(total: number): SPPBClassification {
  if (total >= 10) return "minimal_mild_limitation";
  if (total >= 7) return "moderate_limitation";
  return "severe_limitation";
}

export const SPPB_CLASSIFICATION_LABEL: Record<SPPBClassification, string> = {
  minimal_mild_limitation: "Minimal to mild limitation",
  moderate_limitation:     "Moderate limitation",
  severe_limitation:       "Severe lower-extremity limitation",
};

export const SPPB_CLASSIFICATION_TONE: Record<SPPBClassification, string> = {
  minimal_mild_limitation: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  moderate_limitation:     "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  severe_limitation:       "bg-red-500/10 text-red-700 dark:text-red-400",
};

// ─── Recommendation + interpretation ───────────────────────────

export function buildRecommendation(total: number): string {
  if (total <= 6) {
    return (
      "RECOMMENDATION: Comprehensive geriatric assessment referral. " +
      "High-priority fall-prevention intervention indicated."
    );
  }
  if (total <= 9) {
    return (
      "RECOMMENDATION: Physical therapy evaluation. Targeted " +
      "exercise program addressing the weakest sub-domain."
    );
  }
  return (
    "RECOMMENDATION: Continue maintenance activities. " +
    "Re-assess in 6-12 months."
  );
}

export function buildInterpretation(
  balance: BalanceComponentResult,
  gait: GaitSpeedComponentResult,
  chair: ChairStandComponentResult,
  total: number,
): string {
  const classification = classifyTotal(total);

  // Identify the weakest sub-domain for moderate-score commentary.
  const weakest = [
    { name: "balance",     score: balance.score },
    { name: "gait speed",  score: gait.score },
    { name: "chair stand", score: chair.score },
  ].sort((a, b) => a.score - b.score)[0];

  if (classification === "minimal_mild_limitation") {
    return (
      `Patient demonstrated excellent physical performance with a ` +
      `composite SPPB score of ${total}/12, indicating minimal to mild ` +
      `functional limitation. All three sub-domains scored within ` +
      `community-ambulatory range.`
    );
  }
  if (classification === "moderate_limitation") {
    return (
      `Patient achieved a composite SPPB score of ${total}/12, ` +
      `indicating moderate functional limitation. Weakness identified ` +
      `in ${weakest.name} (${weakest.score}/4). Targeted intervention ` +
      `in this sub-domain may yield the largest functional gain.`
    );
  }
  return (
    `Patient scored ${total}/12, indicating severe lower-extremity ` +
    `limitation. This score is associated with elevated risk of falls, ` +
    `hospitalisation, and disability over the next 12-24 months. ` +
    `Comprehensive geriatric assessment recommended.`
  );
}

// ─── Component-result builders (used by the orchestrator) ──────

/** Build a BalanceComponentResult from raw StageResult entries
 *  produced by the C4 math library (capped at stages 1-3). */
export function buildBalanceComponent(stages: {
  1?: StageResult;
  2?: StageResult;
  3?: StageResult;
}): BalanceComponentResult {
  let final_stage: 0 | 1 | 2 | 3 = 0;
  if (stages[1]?.outcome === "pass") final_stage = 1;
  if (stages[1]?.outcome === "pass" && stages[2]?.outcome === "pass") final_stage = 2;
  if (stages[1]?.outcome === "pass" && stages[2]?.outcome === "pass" && stages[3]?.outcome === "pass") final_stage = 3;
  const stage3_hold = stages[3]?.duration_seconds ?? 0;
  return {
    stages,
    final_stage_completed: final_stage,
    stage3_hold_seconds: stage3_hold,
    score: scoreBalance(final_stage, stage3_hold),
  };
}

export function buildGaitSpeedComponent(
  trial1: GaitSpeedTrial | null,
  trial2: GaitSpeedTrial | null,
): GaitSpeedComponentResult {
  const completed = [trial1, trial2].filter(
    (t): t is GaitSpeedTrial => !!t && t.completed,
  );
  const best = completed.length
    ? completed.reduce((a, b) => (a.duration_sec < b.duration_sec ? a : b))
    : null;
  const best_time = best?.duration_sec ?? null;
  const speed = best_time && best_time > 0 ? 4.0 / best_time : null;
  return {
    trial1,
    trial2,
    best_time_sec: best_time,
    speed_mps: speed !== null ? Math.round(speed * 100) / 100 : null,
    score: scoreGaitSpeed(speed),
  };
}

export function buildChairStandComponent(
  trial: SitToStandResult,
): ChairStandComponentResult {
  const completed = trial.termination === "completed";
  return {
    trial,
    score: scoreChairStand(trial.total_time_seconds, completed),
  };
}

export function buildSPPBResult(
  balance: BalanceComponentResult,
  gait: GaitSpeedComponentResult,
  chair: ChairStandComponentResult,
  patientAge: number | null,
): SPPBResult {
  const total = balance.score + gait.score + chair.score;
  return {
    patient_age: patientAge,
    balance,
    gait_speed: gait,
    chair_stand: chair,
    total_score: total,
    classification: classifyTotal(total),
    interpretation: buildInterpretation(balance, gait, chair, total),
    recommendation: buildRecommendation(total),
  };
}
