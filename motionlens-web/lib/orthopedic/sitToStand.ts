// 5x Sit-to-Stand test — math, conventions, rep detector,
// classification, fatigue/arm-uncross flags, interpretation.
//
// Spec: MotionLens Test Battery v1.0, Test C2 (p. 14).
// Pose model: MoveNet 17-keypoint (LM indices in @/lib/pose/landmarks).
//
// This is a SINGLE-TRIAL test — patient performs 5 sit-to-stand
// cycles using both legs. No L/R split.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

const VIS_THRESHOLD = 0.3;

export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// PDF Test C2 spec values — DO NOT change without spec update.
export const TARGET_REP_COUNT = 5;
export const TRIAL_TIMEOUT_SEC = 30;

// Time-band classification (PDF Test C2).
export const NORMAL_MAX_SEC = 12;
export const BORDERLINE_MAX_SEC = 15;

// Fatigue flag: last rep > first rep × 1.6 → significant fatigue.
export const FATIGUE_RATIO = 1.6;

// Sit / stand detection. baseline = hip-mid Y when seated. Standing
// is detected when hip-mid Y rises above baseline by at least
// STAND_DELTA_FRAC × leg-length pixels. Sit-back is detected when
// hip-mid Y returns within SIT_DELTA_FRAC × leg-length of baseline.
// These ratios are tight enough to ignore micro-shifts but wide
// enough to fire on a real stand even when the camera is set high.
export const STAND_DELTA_FRAC = 0.20;
export const SIT_DELTA_FRAC   = 0.08;

// Arm-uncrossing detection: tolerance (in shoulder→hip torso-height
// fraction) for how far below the shoulder line the wrist is allowed
// to drop before we flag the arms as uncrossed. 0.50 ≈ "wrist may
// sit at chest level (between shoulder and mid-torso) — below that
// is uncrossed". Treated as a sustained-event flag (set once per
// trial; not per frame).
export const ARM_UNCROSS_TORSO_FRAC = 0.50;

export type Classification = "normal" | "borderline" | "weakness";
export type Termination = "completed" | "timeout" | "stopped";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

// ─── Per-frame computations ─────────────────────────────────────

export function computeHipMidY(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!visible(lHip) && !visible(rHip)) return null;
  // Lateral view often loses the far hip — fall back to whichever
  // hip is visible.
  if (visible(lHip) && visible(rHip)) return (lHip.y + rHip.y) / 2;
  return (visible(lHip) ? lHip.y : rHip.y);
}

export function computeShoulderMidY(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!visible(lSh) && !visible(rSh)) return null;
  if (visible(lSh) && visible(rSh)) return (lSh.y + rSh.y) / 2;
  return (visible(lSh) ? lSh.y : rSh.y);
}

// Compute the camera-facing leg's leg-length proxy in pixels
// (hip→ankle). Returns null when neither leg's hip+ankle are both
// visible. Used to scale the sit/stand delta thresholds so they
// adapt to patient size and camera distance.
export function computeLegLengthPx(keypoints: Keypoint[]): number | null {
  const sides: Array<["left" | "right", number, number]> = [
    ["left",  LM.LEFT_HIP,  LM.LEFT_ANKLE],
    ["right", LM.RIGHT_HIP, LM.RIGHT_ANKLE],
  ];
  let best: number | null = null;
  let bestScore = 0;
  for (const [, hipIdx, ankleIdx] of sides) {
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

// Knee angle of the camera-facing leg. Returns the interior angle at
// the knee (180° = fully extended / standing, smaller = bent /
// sitting). Uses whichever side has the highest min-score across
// hip + knee + ankle.
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

// Arms-crossed check. Returns false if EITHER wrist drops more than
// ARM_UNCROSS_TORSO_FRAC × torso-height below the shoulder line.
// In a properly crossed position both wrists sit near the opposite
// shoulder, so they are at most ~chest-level (slightly below
// shoulder Y in image coords). Returns true when both wrists are
// either un-detectable (occluded but trial hasn't seen a clear
// drop) or above the threshold.
export function areArmsCrossed(keypoints: Keypoint[]): boolean {
  const shoulderY = computeShoulderMidY(keypoints);
  const hipY      = computeHipMidY(keypoints);
  if (shoulderY === null || hipY === null) return true; // can't tell — assume OK
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
  baselineY: number | null;       // hip-mid Y when seated (auto-captured)
  legLengthPx: number | null;     // captured at first complete frame
  current: SitState;
  // Tracks all sit-down events (timestamps in ms, relative to
  // start). The first sit event is t=0 (initial position); each
  // subsequent sit event closes a rep.
  sitEvents: number[];
  // Running min knee angle within the current cycle (clears each sit).
  currentMinKneeAngle: number;
}

export function newRepDetector(): RepDetectorState {
  return {
    baselineY: null,
    legLengthPx: null,
    current: "sitting",
    sitEvents: [0],          // seed with t=0 (timer start = patient seated)
    currentMinKneeAngle: 180,
  };
}

export interface RepDetectionResult {
  /** True when this frame transitioned standing → sitting and a rep
   *  was committed by the caller. */
  completedRep: boolean;
  /** Most recent state. */
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

  // Threshold scaling — fall back to a sensible pixel default if
  // we couldn't get leg length yet.
  const ref = s.legLengthPx ?? 200;
  const standThreshold = s.baselineY - ref * STAND_DELTA_FRAC;
  const sitThreshold   = s.baselineY - ref * SIT_DELTA_FRAC;

  // Image-y-down: standing = SMALLER y (hip rises in the image),
  // sitting = larger y (back to baseline).
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
  rep_index: number;          // 1..N
  /** Wall-clock duration of this rep in seconds. */
  duration_seconds: number;
  /** Smallest knee angle reached during this rep (degrees). */
  min_knee_angle_deg: number;
}

export interface FrameSample {
  t_ms: number;
  hip_mid_y: number | null;
  knee_angle_deg: number | null;
  arms_crossed: boolean;
}

export function classifyTotalTime(totalSec: number): Classification {
  if (totalSec < NORMAL_MAX_SEC) return "normal";
  if (totalSec <= BORDERLINE_MAX_SEC) return "borderline";
  return "weakness";
}

export function computeCvPercent(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return 0;
  const variance =
    values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / values.length;
  return Math.sqrt(variance) / mean * 100;
}

export interface SitToStandResult {
  total_time_seconds: number;
  reps: RepMetrics[];
  rep_durations: number[];
  cv_percent: number;
  classification: Classification;
  fatigue_flag: boolean;
  arm_uncrossed_flag: boolean;
  termination: Termination;
  incomplete: boolean;
  trial_duration_seconds: number;
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
): SitToStandResult {
  const trialDuration = Math.max(0, (endedAtMs - startedAtMs) / 1000);
  const incomplete = reps.length < TARGET_REP_COUNT;
  const repDurations = reps.map((r) => r.duration_seconds);
  const totalTime = repDurations.reduce((a, b) => a + b, 0);
  const cv = computeCvPercent(repDurations);
  const fatigueFlag =
    reps.length >= 2 &&
    repDurations[repDurations.length - 1] > repDurations[0] * FATIGUE_RATIO;

  return {
    total_time_seconds: totalTime,
    reps,
    rep_durations: repDurations,
    cv_percent: cv,
    classification: classifyTotalTime(totalTime),
    fatigue_flag: fatigueFlag,
    arm_uncrossed_flag: armUncrossed,
    termination,
    incomplete,
    trial_duration_seconds: trialDuration,
    samples,
    keypoints,
    last_rep_screenshot_data_url: lastRepScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: SitToStandResult): string {
  const parts: string[] = [];
  const repCount = result.reps.length;
  const repsPhrase = result.incomplete
    ? `${repCount} of ${TARGET_REP_COUNT} reps`
    : `${TARGET_REP_COUNT} reps`;

  if (repCount === 0) {
    return "No completed reps captured — re-run the trial.";
  }

  const totalStr = result.total_time_seconds.toFixed(1);
  if (result.classification === "normal") {
    parts.push(
      `Completed ${repsPhrase} in ${totalStr} s (< ${NORMAL_MAX_SEC} s) — ` +
      `normal lower-extremity strength for the 5x sit-to-stand benchmark.`,
    );
  } else if (result.classification === "borderline") {
    parts.push(
      `Completed ${repsPhrase} in ${totalStr} s (${NORMAL_MAX_SEC}–${BORDERLINE_MAX_SEC} s) — ` +
      `borderline performance; consider re-test or further evaluation.`,
    );
  } else {
    parts.push(
      `Completed ${repsPhrase} in ${totalStr} s (> ${BORDERLINE_MAX_SEC} s) — ` +
      `lower-extremity weakness / elevated fall risk.`,
    );
  }

  if (result.fatigue_flag && result.reps.length >= 2) {
    parts.push(
      `Last rep was ${result.rep_durations[result.rep_durations.length - 1].toFixed(1)} s ` +
      `vs first rep ${result.rep_durations[0].toFixed(1)} s ` +
      `(> ${Math.round((FATIGUE_RATIO - 1) * 100)}% slowdown) — significant fatigue.`,
    );
  }

  if (result.arm_uncrossed_flag) {
    parts.push(
      "Arms uncrossed at one or more points during the trial — strength assessment may be inflated.",
    );
  }

  if (result.incomplete) {
    parts.push(
      `Trial ended before ${TARGET_REP_COUNT} reps were captured ` +
      `(${result.termination === "timeout" ? "30 s timeout" : "stopped early"}).`,
    );
  }

  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const CLASSIFICATION_LABEL: Record<Classification, string> = {
  normal:     "Normal",
  borderline: "Borderline",
  weakness:   "Weakness",
};

export const CLASSIFICATION_TONE: Record<Classification, string> = {
  normal:     "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  borderline: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  weakness:   "bg-red-500/10 text-red-700 dark:text-red-400",
};
