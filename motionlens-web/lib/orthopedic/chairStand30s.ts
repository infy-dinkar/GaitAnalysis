// 30-Second Chair Stand test — math, classification, fatigue
// regression, depth consistency, interpretation.
//
// Spec: MotionLens Test Battery v1.0, Test C3 (pp. 14-15).
// Pose model: MoveNet 17-keypoint (LM indices in @/lib/pose/landmarks).
//
// Single-trial test (no L/R split). Patient performs as many full
// sit-to-stand cycles as possible in 30 seconds. Classification is
// against CDC STEADI age/sex norms — looked up from the centralized
// normsDatabase module.
//
// Reuses the per-frame math from the 5xSTS implementation but
// inverts the termination condition: 30 s timer is the only stop,
// and the primary outcome is rep COUNT rather than total time.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";
import { getChairStand30sNorm, type Sex } from "@/lib/orthopedic/normsDatabase";

const VIS_THRESHOLD = 0.3;

export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// PDF Test C3 spec values.
export const TRIAL_DURATION_SEC = 30;

// Sit / stand detection — same thresholds as 5xSTS.
export const STAND_DELTA_FRAC = 0.20;
export const SIT_DELTA_FRAC   = 0.08;

// Arm-uncrossing detection — wrist may sit at chest level (between
// shoulder and mid-torso) before being treated as "uncrossed".
export const ARM_UNCROSS_TORSO_FRAC = 0.50;

export type Classification = "above_norm" | "below_norm";
export type Termination = "completed" | "stopped";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

// ─── Per-frame computations ─────────────────────────────────────

export function computeHipMidY(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!visible(lHip) && !visible(rHip)) return null;
  if (visible(lHip) && visible(rHip)) return (lHip.y + rHip.y) / 2;
  return visible(lHip) ? lHip.y : rHip.y;
}

export function computeShoulderMidY(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!visible(lSh) && !visible(rSh)) return null;
  if (visible(lSh) && visible(rSh)) return (lSh.y + rSh.y) / 2;
  return visible(lSh) ? lSh.y : rSh.y;
}

export function computeLegLengthPx(keypoints: Keypoint[]): number | null {
  const sides: Array<[number, number]> = [
    [LM.LEFT_HIP,  LM.LEFT_ANKLE],
    [LM.RIGHT_HIP, LM.RIGHT_ANKLE],
  ];
  let best: number | null = null;
  let bestScore = 0;
  for (const [hipIdx, ankleIdx] of sides) {
    const hip = keypoints[hipIdx];
    const ankle = keypoints[ankleIdx];
    if (!visible(hip) || !visible(ankle)) continue;
    const score = Math.min(hip.score ?? 0, ankle.score ?? 0);
    if (score > bestScore) {
      bestScore = score;
      best = Math.hypot(ankle.x - hip.x, ankle.y - hip.y);
    }
  }
  return best;
}

export function computeKneeAngle(keypoints: Keypoint[]): number | null {
  const sides: Array<[number, number, number]> = [
    [LM.LEFT_HIP,  LM.LEFT_KNEE,  LM.LEFT_ANKLE],
    [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
  ];
  let bestAngle: number | null = null;
  let bestScore = 0;
  for (const [hipIdx, kneeIdx, ankleIdx] of sides) {
    const hip = keypoints[hipIdx];
    const knee = keypoints[kneeIdx];
    const ankle = keypoints[ankleIdx];
    if (!visible(hip) || !visible(knee) || !visible(ankle)) continue;
    const minScore = Math.min(hip.score ?? 0, knee.score ?? 0, ankle.score ?? 0);
    if (minScore <= bestScore) continue;
    const v1x = hip.x - knee.x; const v1y = hip.y - knee.y;
    const v2x = ankle.x - knee.x; const v2y = ankle.y - knee.y;
    const m1 = Math.hypot(v1x, v1y);
    const m2 = Math.hypot(v2x, v2y);
    if (m1 === 0 || m2 === 0) continue;
    const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / (m1 * m2)));
    bestAngle = (Math.acos(cos) * 180) / Math.PI;
    bestScore = minScore;
  }
  return bestAngle;
}

export function areArmsCrossed(keypoints: Keypoint[]): boolean {
  const shoulderY = computeShoulderMidY(keypoints);
  const hipY      = computeHipMidY(keypoints);
  if (shoulderY === null || hipY === null) return true;
  const torsoH = Math.abs(hipY - shoulderY);
  if (torsoH < 1) return true;
  const tolerance = torsoH * ARM_UNCROSS_TORSO_FRAC;
  const lW = keypoints[LM.LEFT_WRIST];
  const rW = keypoints[LM.RIGHT_WRIST];
  if (visible(lW) && lW.y > shoulderY + tolerance) return false;
  if (visible(rW) && rW.y > shoulderY + tolerance) return false;
  return true;
}

// ─── Rep state machine (sit ↔ stand) ───────────────────────────

export type SitState = "sitting" | "standing";

export interface RepDetectorState {
  baselineY: number | null;
  legLengthPx: number | null;
  current: SitState;
  /** Sit-event timestamps (ms relative to start). The first one is
   *  t = 0 (timer start = patient seated). Each subsequent entry
   *  closes a rep. */
  sitEvents: number[];
  currentMinKneeAngle: number;
}

export function newRepDetector(): RepDetectorState {
  return {
    baselineY: null,
    legLengthPx: null,
    current: "sitting",
    sitEvents: [0],
    currentMinKneeAngle: 180,
  };
}

export interface RepDetectionResult {
  completedRep: boolean;
  state: SitState;
}

export function stepRepDetector(
  s: RepDetectorState,
  hipMidY: number | null,
  kneeAngle: number | null,
  tMs: number,
): RepDetectionResult {
  if (hipMidY === null) return { completedRep: false, state: s.current };
  if (s.baselineY === null) {
    s.baselineY = hipMidY;
    return { completedRep: false, state: s.current };
  }
  if (kneeAngle !== null && kneeAngle < s.currentMinKneeAngle) {
    s.currentMinKneeAngle = kneeAngle;
  }
  const ref = s.legLengthPx ?? 200;
  const standThreshold = s.baselineY - ref * STAND_DELTA_FRAC;
  const sitThreshold   = s.baselineY - ref * SIT_DELTA_FRAC;

  if (s.current === "sitting" && hipMidY < standThreshold) {
    s.current = "standing";
  } else if (s.current === "standing" && hipMidY > sitThreshold) {
    s.current = "sitting";
    s.sitEvents.push(tMs);
    return { completedRep: true, state: "sitting" };
  }
  return { completedRep: false, state: s.current };
}

// ─── Aggregates ─────────────────────────────────────────────────

export interface RepMetrics {
  rep_index: number;
  duration_seconds: number;
  min_knee_angle_deg: number;
}

export interface FrameSample {
  t_ms: number;
  hip_mid_y: number | null;
  knee_angle_deg: number | null;
  arms_crossed: boolean;
}

// Linear regression: returns the slope (seconds-per-rep added per
// rep-index increment). Positive = each successive rep is taking
// longer than the previous one (fatigue). Returns 0 with too few
// points.
export function fatigueSlopeSecPerRep(repDurations: number[]): number {
  const n = repDurations.length;
  if (n < 2) return 0;
  const meanX = (n - 1) / 2; // x = 0..n-1
  const meanY = repDurations.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - meanX) * (repDurations[i] - meanY);
    den += (i - meanX) * (i - meanX);
  }
  return den === 0 ? 0 : num / den;
}

export function depthSdDeg(reps: RepMetrics[]): number {
  if (reps.length < 2) return 0;
  const xs = reps.map((r) => r.min_knee_angle_deg);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const variance = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / xs.length;
  return Math.sqrt(variance);
}

export interface ChairStand30sResult {
  rep_count: number;
  reps: RepMetrics[];
  rep_durations: number[];
  mean_rep_duration_sec: number;
  depth_sd_deg: number;
  fatigue_slope_sec_per_rep: number;
  norm_threshold: number;
  norm_band_label: string;
  norm_comparable: boolean;
  classification: Classification;
  arm_uncrossed_flag: boolean;
  termination: Termination;
  trial_duration_seconds: number;
  /** Patient demographics snapshot at the time of the trial. */
  patient_age: number | null;
  patient_sex: Sex | "other" | null;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  last_rep_screenshot_data_url: string | null;
}

export function summarizeTrial(
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  reps: RepMetrics[],
  armUncrossed: boolean,
  samples: FrameSample[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  lastRepScreenshotDataUrl: string | null,
  patientAge: number | null,
  patientSex: Sex | "other" | null,
): ChairStand30sResult {
  const trialDuration = Math.max(0, (endedAtMs - startedAtMs) / 1000);
  const repDurations = reps.map((r) => r.duration_seconds);
  const meanDur =
    repDurations.length === 0 ? 0 : repDurations.reduce((a, b) => a + b, 0) / repDurations.length;

  const norm = getChairStand30sNorm(patientAge, patientSex);
  const repCount = reps.length;
  const classification: Classification =
    repCount >= norm.belowAverageThreshold ? "above_norm" : "below_norm";

  return {
    rep_count: repCount,
    reps,
    rep_durations: repDurations,
    mean_rep_duration_sec: meanDur,
    depth_sd_deg: depthSdDeg(reps),
    fatigue_slope_sec_per_rep: fatigueSlopeSecPerRep(repDurations),
    norm_threshold: norm.belowAverageThreshold,
    norm_band_label: norm.bandLabel,
    norm_comparable: norm.comparable,
    classification,
    arm_uncrossed_flag: armUncrossed,
    termination,
    trial_duration_seconds: trialDuration,
    patient_age: patientAge,
    patient_sex: patientSex,
    samples,
    keypoints,
    last_rep_screenshot_data_url: lastRepScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: ChairStand30sResult): string {
  const parts: string[] = [];
  const reps = result.rep_count;
  const threshold = result.norm_threshold;

  if (reps === 0) {
    return "No completed reps captured — re-run the trial.";
  }

  if (result.classification === "above_norm") {
    parts.push(
      `Completed ${reps} reps in 30 s — at or above the ${result.norm_band_label} ` +
      `cutoff of ${threshold} reps. Lower-extremity strength within the expected range.`,
    );
  } else {
    parts.push(
      `Completed ${reps} reps in 30 s — below the ${result.norm_band_label} ` +
      `cutoff of ${threshold} reps. Positive screen for fall risk per CDC STEADI.`,
    );
  }

  if (!result.norm_comparable) {
    parts.push(
      "Norm comparison is approximate — patient demographics were missing or " +
      "outside the published 60–94 age range.",
    );
  }

  if (result.fatigue_slope_sec_per_rep > 0.05 && reps >= 3) {
    parts.push(
      `Per-rep duration trended upward (~${result.fatigue_slope_sec_per_rep.toFixed(2)} s ` +
      `added per rep) — suggests fatigue across the trial.`,
    );
  }

  if (result.arm_uncrossed_flag) {
    parts.push(
      "Arms uncrossed at one or more points during the trial — strength assessment may be inflated.",
    );
  }

  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const CLASSIFICATION_LABEL: Record<Classification, string> = {
  above_norm: "Above norm",
  below_norm: "Below norm",
};

export const CLASSIFICATION_TONE: Record<Classification, string> = {
  above_norm: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  below_norm: "bg-red-500/10 text-red-700 dark:text-red-400",
};
