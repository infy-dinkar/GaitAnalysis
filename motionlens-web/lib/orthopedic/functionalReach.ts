// Functional Reach Test (C6) — math, validity gates, classification,
// upload-mode API client.
//
// Setup:
//   Patient stands SIDE-ON to the camera (one shoulder toward the
//   lens, near arm raised to ~90° shoulder flexion, fist closed).
//   The patient reaches forward as far as possible WITHOUT stepping,
//   lifting the heels, or losing balance, briefly holds the peak,
//   then returns. Three trials in a single ~30 s recording window.
//
// Scale calibration:
//   `CalibrationResult.pixels_per_cm` is the only thing this file
//   cares about. The provider lives elsewhere (Step 2 hooks up a
//   shared height-based calibration). When `calibration` is null
//   the test still runs but reach is reported in relative pixel
//   units only — no fall-risk classification.
//
// Math:
//   • Baseline (point A) = median wrist x over the first
//     BASELINE_HOLD_SEC seconds AFTER the test-side wrist is stable
//     at shoulder height. Locks in once the arm has been at ~90° for
//     that whole window.
//   • Per-frame "reach" = |wrist_x − baseline_wrist_x| (px).
//   • Peak detection: local maxima of the absolute displacement
//     trace with minimum prominence + minimum inter-peak distance.
//     Top NUM_TRIALS peaks (by magnitude) are kept.
//   • Each peak owns a TRIAL WINDOW spanning the half-way points
//     toward the adjacent peaks (or the recording bounds for the
//     first / last peak).
//   • Validity, per trial:
//       heel_rise  → heel y drifted > HEEL_RISE_THRESHOLD_CM from
//                    baseline anywhere in the window
//       step       → ankle x drifted > STEP_THRESHOLD_CM from
//                    baseline anywhere in the window
//       no_motion  → peak magnitude below MIN_REACH_FOR_VALID
//     Anything else = valid.
//   • Best valid trial = highest reach across valid trials.
//
// Backend mirrors this math in
// engines/orthopedic/functional_reach_engine.py.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";
import {
  pxToCm,
  type CalibrationResult,
} from "@/lib/calibration/types";

export type { CalibrationResult } from "@/lib/calibration/types";
export { pxToCm } from "@/lib/calibration/types";

// ─── Tuning constants ──────────────────────────────────────────
export const VIS_THRESHOLD = 0.3;
export const SAMPLE_HZ = 15;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

/** Total length of the recording window — long enough for 3 reaches
 *  at a comfortable pace. */
export const RECORDING_DURATION_SEC = 30;

/** How long the arm must be at ~90° shoulder height before the
 *  baseline locks in. */
export const BASELINE_HOLD_SEC = 1;
export const BASELINE_HOLD_SAMPLES = Math.round(BASELINE_HOLD_SEC * SAMPLE_HZ);

/** Wrist y within this many px of shoulder y = "at shoulder height". */
export const SHOULDER_HEIGHT_TOLERANCE_PX = 40;

export const NUM_TRIALS = 3;

/** Smoothed heel-rise signal (foot_index_y − heel_y, delta from
 *  baseline) must exceed this for a trial to be voided as a heel-rise.
 *  Raised from 2 cm to 3.5 cm: a true clinical heel-off is ≥ 3 cm,
 *  and 2 cm sat inside the residual jitter on the smoothed signal
 *  during forward reach (BlazePose foot-model re-fit). */
export const HEEL_RISE_THRESHOLD_CM = 3.5;

/** Smoothed ankle-x deviation from baseline must exceed this to be
 *  flagged as a step. Raised from 5 cm to 8 cm: forward body
 *  translation during reach moves the planted-foot ankle 3-5 cm in
 *  image space — that's not a step. A real foot repositioning is
 *  ≥ 8 cm. */
export const STEP_THRESHOLD_CM = 8.0;

/** Reaches below this magnitude are noise / not a true reach attempt. */
export const MIN_REACH_FOR_VALID_CM = 3.0;

/** Fall-risk cutoffs (cm). Only applied when a calibration is
 *  available — uncalibrated reports show relative units. */
export const LOW_FALL_RISK_MIN_CM = 25.0;
export const MODERATE_FALL_RISK_MIN_CM = 15.0;
export const VERY_HIGH_FALL_RISK_MAX_CM = 10.0;

/** Uncalibrated fall-backs (expressed as a fraction of the patient's
 *  LEG PIXEL LENGTH — hip y to ankle y in the baseline standing
 *  window — so the threshold scales with how close the patient is to
 *  the camera AND with the anatomical region the heel/ankle actually
 *  inhabit). For an 80 cm leg ≈ 320 px on a typical webcam, the heel
 *  fallback resolves to ~13 px and the step fallback to ~32 px — much
 *  more realistic than the previous 4 px / 11 px torso-based values. */
export const HEEL_RISE_FALLBACK_FRACTION_OF_LEG = 0.04;
export const STEP_FALLBACK_FRACTION_OF_LEG = 0.10;
export const MIN_REACH_FALLBACK_FRACTION_OF_LEG = 0.06;

/** Validity event must persist for at least this many consecutive
 *  smoothed samples to count. At SAMPLE_HZ = 15, 5 frames is ~333 ms.
 *  A momentary spike that rides through the rolling median can no
 *  longer void a trial — the event has to be genuinely sustained. */
export const SUSTAINED_VIOLATION_FRAMES = 5;

/** Peak-detection: minimum gap between peaks in samples + minimum
 *  prominence as a fraction of the largest peak. */
export const PEAK_MIN_DISTANCE_SAMPLES = Math.round(1.5 * SAMPLE_HZ);
export const PEAK_MIN_PROMINENCE_FRACTION = 0.25;

/** Rolling-median window applied to heel y and ankle x before the
 *  per-trial validity scan. Without smoothing, a SINGLE noisy
 *  MediaPipe heel sample can fabricate a 10-20 cm "heel rise" /
 *  "step" that voids the trial. A ~470 ms window kills isolated
 *  outliers but still preserves a real heel-rise of a few hundred
 *  milliseconds. */
export const HEEL_ANKLE_SMOOTH_WINDOW = 7;

/** How far around each detected peak the validity scan looks. ±2 s
 *  comfortably covers the forward reach + brief hold + return phase
 *  of a single trial while excluding "just standing" time where
 *  MediaPipe heel jitter has nothing to do with the test movement. */
export const TRIAL_VALIDITY_HALF_WINDOW_SEC = 2.0;
export const TRIAL_VALIDITY_HALF_WINDOW_SAMPLES =
  Math.round(TRIAL_VALIDITY_HALF_WINDOW_SEC * SAMPLE_HZ);

export type Side = "left" | "right";

export type FallRiskClass = "low" | "moderate" | "high" | "very_high";

export type TrialValidity = "valid" | "heel_rise" | "step" | "no_motion";

export type Termination = "completed" | "stopped";

// ─── Per-frame helpers ──────────────────────────────────────────

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

export const SIDE_INDICES = {
  left: {
    wrist: LM.LEFT_WRIST,
    elbow: LM.LEFT_ELBOW,
    shoulder: LM.LEFT_SHOULDER,
    hip: LM.LEFT_HIP,
    ankle: LM.LEFT_ANKLE,
    heel: LM.LEFT_HEEL,
    foot: LM.LEFT_FOOT_INDEX,
  },
  right: {
    wrist: LM.RIGHT_WRIST,
    elbow: LM.RIGHT_ELBOW,
    shoulder: LM.RIGHT_SHOULDER,
    hip: LM.RIGHT_HIP,
    ankle: LM.RIGHT_ANKLE,
    heel: LM.RIGHT_HEEL,
    foot: LM.RIGHT_FOOT_INDEX,
  },
} as const;

export function isTestSideTrackable(kp: Keypoint[], side: Side): boolean {
  const idx = SIDE_INDICES[side];
  return (
    visible(kp[idx.wrist]) &&
    visible(kp[idx.shoulder]) &&
    visible(kp[idx.hip]) &&
    visible(kp[idx.ankle]) &&
    visible(kp[idx.heel])
  );
}

export function getWristX(kp: Keypoint[], side: Side): number | null {
  const w = kp[SIDE_INDICES[side].wrist];
  return visible(w) ? w.x : null;
}

export function getWristY(kp: Keypoint[], side: Side): number | null {
  const w = kp[SIDE_INDICES[side].wrist];
  return visible(w) ? w.y : null;
}

export function getShoulderY(kp: Keypoint[], side: Side): number | null {
  const s = kp[SIDE_INDICES[side].shoulder];
  return visible(s) ? s.y : null;
}

export function getHipY(kp: Keypoint[], side: Side): number | null {
  const h = kp[SIDE_INDICES[side].hip];
  return visible(h) ? h.y : null;
}

export function getAnkleX(kp: Keypoint[], side: Side): number | null {
  const a = kp[SIDE_INDICES[side].ankle];
  return visible(a) ? a.x : null;
}

export function getHeelY(kp: Keypoint[], side: Side): number | null {
  const h = kp[SIDE_INDICES[side].heel];
  return visible(h) ? h.y : null;
}

export function getFootIndexY(kp: Keypoint[], side: Side): number | null {
  const f = kp[SIDE_INDICES[side].foot];
  return visible(f) ? f.y : null;
}

/** Is the wrist at ~shoulder height (i.e. the patient has raised the
 *  test-side arm to 90°)? Used to gate baseline acquisition. */
export function isArmRaisedToShoulder(kp: Keypoint[], side: Side): boolean {
  const w = kp[SIDE_INDICES[side].wrist];
  const s = kp[SIDE_INDICES[side].shoulder];
  if (!visible(w) || !visible(s)) return false;
  return Math.abs(w.y - s.y) <= SHOULDER_HEIGHT_TOLERANCE_PX;
}

/** Trunk forward-lean angle (degrees) — angle of the hip-to-shoulder
 *  line from vertical. 0° = upright, larger = leaning more. Image
 *  coordinates are y-down; "up" vertical is (0, -1). */
export function computeTrunkAngleDeg(kp: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const h = kp[idx.hip];
  const s = kp[idx.shoulder];
  if (!visible(h) || !visible(s)) return null;
  const vx = s.x - h.x;
  const vy = s.y - h.y;
  const len = Math.hypot(vx, vy);
  if (len < 1e-6) return null;
  const cos = (-vy) / len; // dot product with (0,-1) / |v|
  const a = Math.acos(Math.max(-1, Math.min(1, cos)));
  return (a * 180) / Math.PI;
}

// ─── Per-frame sample shape ─────────────────────────────────────

export interface FrameSample {
  t_ms: number;
  wrist_x_px: number | null;
  wrist_y_px: number | null;
  shoulder_y_px: number | null;
  ankle_x_px: number | null;
  heel_y_px: number | null;
  /** Toe / first metatarsal y in pixels. Used together with heel_y
   *  for a LOCAL heel-rise signal — see summarizeTrial. */
  foot_index_y_px: number | null;
  trunk_angle_deg: number | null;
  arm_raised: boolean;
}

export function buildSample(
  tMs: number,
  kp: Keypoint[],
  side: Side,
): FrameSample {
  return {
    t_ms: tMs,
    wrist_x_px: getWristX(kp, side),
    wrist_y_px: getWristY(kp, side),
    shoulder_y_px: getShoulderY(kp, side),
    ankle_x_px: getAnkleX(kp, side),
    heel_y_px: getHeelY(kp, side),
    foot_index_y_px: getFootIndexY(kp, side),
    trunk_angle_deg: computeTrunkAngleDeg(kp, side),
    arm_raised: isArmRaisedToShoulder(kp, side),
  };
}

// ─── Result shapes ──────────────────────────────────────────────

export interface Trial {
  trial_index: number;
  /** Index into `samples[]` at which the peak displacement occurred. */
  peak_sample_index: number;
  peak_t_ms: number;
  /** Signed displacement (wrist_x − baseline_wrist_x) at the peak. */
  signed_displacement_px: number;
  /** |signed_displacement_px|. */
  reach_px: number;
  reach_cm: number | null;
  trunk_angle_at_peak_deg: number | null;
  validity: TrialValidity;
  invalidity_detail: string | null;
  /** Index range owned by this trial (for the heel/step gate scans). */
  window_start_index: number;
  window_end_index: number;
  /** Max UPWARD heel motion observed in the smoothed trial window
   *  (positive = heel moved up from baseline). Directional: downward
   *  noise glitches do not count. Field name kept for backward
   *  compatibility with saved reports. */
  max_heel_drift_px: number;
  max_heel_drift_cm: number | null;
  /** Max ankle-x deviation observed in the smoothed trial window
   *  (bi-directional — a step forward OR backward voids the trial). */
  max_ankle_drift_px: number;
  max_ankle_drift_cm: number | null;
}

export interface FunctionalReachResult {
  side_tested: Side;

  // Baseline (point A) — locked once the arm has been at shoulder
  // height for BASELINE_HOLD_SEC seconds.
  baseline_locked: boolean;
  baseline_locked_at_index: number | null;
  baseline_wrist_x_px: number | null;
  baseline_ankle_x_px: number | null;
  baseline_heel_y_px: number | null;

  // Per-trial breakdown + best.
  trials: Trial[];
  best_valid_trial_index: number | null;
  best_valid_reach_px: number | null;
  best_valid_reach_cm: number | null;

  /** Only populated when calibrated. */
  classification: FallRiskClass | null;

  /** Calibration that was applied. Null = relative-units mode. */
  calibration: CalibrationResult | null;

  duration_seconds: number;
  termination: Termination;

  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  peak_screenshot_data_url: string | null;
}

// ─── Peak detection ─────────────────────────────────────────────

interface PeakCandidate {
  index: number;
  value: number;
}

/** Find local-maxima indices in `values`, gated by minimum prominence
 *  and minimum inter-peak distance. Nulls are treated as gaps. */
function findPeaks(
  values: Array<number | null>,
  minDistance: number,
  minProminenceAbs: number,
): PeakCandidate[] {
  const n = values.length;
  const peaks: PeakCandidate[] = [];
  for (let i = 1; i < n - 1; i++) {
    const v = values[i];
    if (v === null) continue;
    const prev = values[i - 1];
    const next = values[i + 1];
    if (prev === null || next === null) continue;
    if (!(v >= prev && v >= next)) continue;
    if (v === prev && v === next) continue;

    // Prominence: how far this peak stands above the nearest valley
    // on either side, within a search window.
    const lookBack = Math.min(i, Math.max(minDistance, 30));
    const lookFwd = Math.min(n - 1 - i, Math.max(minDistance, 30));
    let leftMin = v;
    for (let j = i - 1; j >= i - lookBack; j--) {
      const x = values[j];
      if (x === null) continue;
      if (x < leftMin) leftMin = x;
    }
    let rightMin = v;
    for (let j = i + 1; j <= i + lookFwd; j++) {
      const x = values[j];
      if (x === null) continue;
      if (x < rightMin) rightMin = x;
    }
    const prom = v - Math.max(leftMin, rightMin);
    if (prom < minProminenceAbs) continue;

    peaks.push({ index: i, value: v });
  }

  // Enforce minimum inter-peak distance by keeping the higher of any
  // overlapping pair.
  peaks.sort((a, b) => b.value - a.value);
  const kept: PeakCandidate[] = [];
  for (const p of peaks) {
    let conflict = false;
    for (const k of kept) {
      if (Math.abs(p.index - k.index) < minDistance) {
        conflict = true;
        break;
      }
    }
    if (!conflict) kept.push(p);
  }
  return kept;
}

// ─── Aggregation ────────────────────────────────────────────────

export interface SummarizeInput {
  side: Side;
  startedAtMs: number;
  endedAtMs: number;
  termination: Termination;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  calibration: CalibrationResult | null;
  peakScreenshotDataUrl: string | null;
}

export function summarizeTrial(input: SummarizeInput): FunctionalReachResult {
  const {
    side, startedAtMs, endedAtMs, termination, samples, keypoints,
    calibration, peakScreenshotDataUrl,
  } = input;
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);

  const empty: FunctionalReachResult = {
    side_tested: side,
    baseline_locked: false,
    baseline_locked_at_index: null,
    baseline_wrist_x_px: null,
    baseline_ankle_x_px: null,
    baseline_heel_y_px: null,
    trials: [],
    best_valid_trial_index: null,
    best_valid_reach_px: null,
    best_valid_reach_cm: null,
    classification: null,
    calibration,
    duration_seconds: duration,
    termination,
    samples,
    keypoints,
    peak_screenshot_data_url: peakScreenshotDataUrl,
  };

  if (samples.length === 0) return empty;

  // 1) Find the baseline lock-in window: the first run of
  //    BASELINE_HOLD_SAMPLES consecutive samples with arm_raised AND
  //    all required signals visible.
  let baselineEndIdx: number | null = null;
  let consecutive = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const allOK =
      s.arm_raised &&
      s.wrist_x_px !== null &&
      s.ankle_x_px !== null &&
      s.heel_y_px !== null;
    if (allOK) {
      consecutive += 1;
      if (consecutive >= BASELINE_HOLD_SAMPLES) {
        baselineEndIdx = i;
        break;
      }
    } else {
      consecutive = 0;
    }
  }

  if (baselineEndIdx === null) {
    // Couldn't lock a baseline — return an empty result with samples
    // intact so the operator can see what happened.
    return empty;
  }

  const baselineStartIdx = baselineEndIdx - BASELINE_HOLD_SAMPLES + 1;
  const baselineWrists: number[] = [];
  const baselineAnkles: number[] = [];
  const baselineHeels: number[] = [];
  // Local heel-rise signal = foot_index_y − heel_y. At rest the foot
  // is flat → delta ≈ 0. When the heel rotates up around the ball of
  // the foot, heel_y decreases (heel moves up) but foot_index_y stays
  // (toe planted), so the delta grows. This signal is INVARIANT to
  // body lean, camera shake, and patient depth — all of which move
  // the foot landmarks together.
  const baselineFootLift: number[] = [];
  for (let i = baselineStartIdx; i <= baselineEndIdx; i++) {
    const s = samples[i];
    if (s.wrist_x_px !== null) baselineWrists.push(s.wrist_x_px);
    if (s.ankle_x_px !== null) baselineAnkles.push(s.ankle_x_px);
    if (s.heel_y_px !== null) baselineHeels.push(s.heel_y_px);
    if (s.heel_y_px !== null && s.foot_index_y_px !== null) {
      baselineFootLift.push(s.foot_index_y_px - s.heel_y_px);
    }
  }
  if (baselineWrists.length === 0) return empty;

  const baselineWristX = median(baselineWrists);
  const baselineAnkleX = baselineAnkles.length ? median(baselineAnkles) : 0;
  const baselineHeelY = baselineHeels.length ? median(baselineHeels) : 0;
  const baselineFootLiftPx =
    baselineFootLift.length ? median(baselineFootLift) : 0;

  // 2) Pull LEG length (px) for relative-mode validity thresholds.
  //    Leg = hip y to ankle y in the baseline window. Foot landmarks
  //    sit at the bottom of this anatomy, so scaling the heel/ankle
  //    gates against leg length keeps the thresholds proportional to
  //    the relevant body region — not to the torso, which has a
  //    different absolute scale and a different noise profile.
  let legLengthPx = 0;
  for (let i = baselineStartIdx; i <= baselineEndIdx; i++) {
    const hipKp = keypoints[i]?.[SIDE_INDICES[side].hip];
    const ankleY = samples[i].ankle_x_px !== null
      ? keypoints[i]?.[SIDE_INDICES[side].ankle]?.y
      : undefined;
    if (hipKp && hipKp.y !== undefined && ankleY !== undefined) {
      legLengthPx = Math.max(legLengthPx, Math.abs(ankleY - hipKp.y));
    }
  }
  if (legLengthPx === 0) legLengthPx = 300; // last-resort default

  // Validity thresholds in PX. Use calibration when available;
  // otherwise fall back to a LEG-LENGTH fraction so the gates scale
  // with the patient's anatomy AND with camera distance.
  const heelRiseThreshPx =
    calibration
      ? HEEL_RISE_THRESHOLD_CM * calibration.pixels_per_cm
      : HEEL_RISE_FALLBACK_FRACTION_OF_LEG * legLengthPx;
  const stepThreshPx =
    calibration
      ? STEP_THRESHOLD_CM * calibration.pixels_per_cm
      : STEP_FALLBACK_FRACTION_OF_LEG * legLengthPx;
  const minReachThreshPx =
    calibration
      ? MIN_REACH_FOR_VALID_CM * calibration.pixels_per_cm
      : MIN_REACH_FALLBACK_FRACTION_OF_LEG * legLengthPx;

  // 3) Build the absolute-displacement trace from the end of the
  //    baseline window onward.
  const trace: Array<number | null> = [];
  for (let i = 0; i < samples.length; i++) {
    if (i <= baselineEndIdx) {
      trace.push(0);
      continue;
    }
    const wx = samples[i].wrist_x_px;
    trace.push(wx === null ? null : Math.abs(wx - baselineWristX));
  }

  // 4) Peak detection — keep up to NUM_TRIALS, pick the top by reach.
  let maxAbsObserved = 0;
  for (const v of trace) {
    if (v !== null && v > maxAbsObserved) maxAbsObserved = v;
  }
  const peakProminencePx = Math.max(
    minReachThreshPx,
    PEAK_MIN_PROMINENCE_FRACTION * maxAbsObserved,
  );
  const peaksRaw = findPeaks(trace, PEAK_MIN_DISTANCE_SAMPLES, peakProminencePx);
  // Highest NUM_TRIALS peaks, then sort by time order for display.
  const peaks = peaksRaw.slice(0, NUM_TRIALS).sort((a, b) => a.index - b.index);

  // Pre-smooth the LOCAL foot-lift signal + ankle x. Comparing
  // absolute heel_y to a baseline far back in time is fragile —
  // MediaPipe systematically shifts the heel landmark upward when
  // the trunk leans forward (body-model re-fit), so heel-rise fires
  // on every trial even when the heel is glued to the floor. The
  // local signal (foot_index_y − heel_y) cancels that bias because
  // both landmarks move together under body-lean / camera shake /
  // depth change. It only grows when the foot actually rotates around
  // the ball — i.e. a real heel rise.
  const footLiftRaw: Array<number | null> = samples.map((s) =>
    s.heel_y_px !== null && s.foot_index_y_px !== null
      ? s.foot_index_y_px - s.heel_y_px
      : null,
  );
  const ankleXRaw = samples.map((s) => s.ankle_x_px);
  const footLiftSmoothed = rollingMedianFilter(footLiftRaw, HEEL_ANKLE_SMOOTH_WINDOW);
  const ankleXSmoothed = rollingMedianFilter(ankleXRaw, HEEL_ANKLE_SMOOTH_WINDOW);

  // 5) Compute per-trial windows + validity.
  const trials: Trial[] = peaks.map((pk, i) => {
    // Trial-ownership window (still kept for the API field so the
    // chart layer can colour-band the trace) — half-way to adjacent
    // peaks, just like before.
    const prevMid =
      i === 0
        ? baselineEndIdx + 1
        : Math.floor((peaks[i - 1].index + pk.index) / 2);
    const nextMid =
      i === peaks.length - 1
        ? samples.length - 1
        : Math.floor((pk.index + peaks[i + 1].index) / 2);

    // Validity-scan window: a narrow ±2 s slice around the peak. Wider
    // ownership window can include 5+ seconds of "just standing"
    // frames where MediaPipe heel/ankle jitter has nothing to do with
    // the reach itself.
    const scanStart = Math.max(prevMid, pk.index - TRIAL_VALIDITY_HALF_WINDOW_SAMPLES);
    const scanEnd = Math.min(nextMid, pk.index + TRIAL_VALIDITY_HALF_WINDOW_SAMPLES);

    // Walk the smoothed signal twice:
    //   (a) record the MAX rise / drift across the scan window for the
    //       report — this is the magnitude the report displays.
    //   (b) record the MAX-SUSTAINED rise / drift: the largest value
    //       that held for at least SUSTAINED_VIOLATION_FRAMES
    //       consecutive smoothed samples above its threshold. Only
    //       (b) is allowed to flip the trial to invalid.
    //
    // Heel rise = positive growth of (foot_index_y − heel_y) above
    // its baseline value. Downward "rise" is not physical — clamp
    // at zero. Ankle x stays bidirectional (a step can go either way).
    let maxHeelRisePx = 0;
    let maxAnkleDriftPx = 0;
    let heelRunFrames = 0;
    let heelRunMin = Infinity;
    let sustainedHeelRisePx = 0;
    let ankleRunFrames = 0;
    let ankleRunMin = Infinity;
    let sustainedAnkleDriftPx = 0;
    for (let j = scanStart; j <= scanEnd; j++) {
      const fl = footLiftSmoothed[j];
      const ax = ankleXSmoothed[j];

      if (fl !== null) {
        const rise = Math.max(0, fl - baselineFootLiftPx);
        if (rise > maxHeelRisePx) maxHeelRisePx = rise;
        if (rise >= heelRiseThreshPx) {
          heelRunFrames += 1;
          if (rise < heelRunMin) heelRunMin = rise;
          if (heelRunFrames >= SUSTAINED_VIOLATION_FRAMES && heelRunMin > sustainedHeelRisePx) {
            sustainedHeelRisePx = heelRunMin;
          }
        } else {
          heelRunFrames = 0;
          heelRunMin = Infinity;
        }
      } else {
        // Treat missing samples as a run break (don't accumulate).
        heelRunFrames = 0;
        heelRunMin = Infinity;
      }

      if (ax !== null) {
        const d = Math.abs(ax - baselineAnkleX);
        if (d > maxAnkleDriftPx) maxAnkleDriftPx = d;
        if (d >= stepThreshPx) {
          ankleRunFrames += 1;
          if (d < ankleRunMin) ankleRunMin = d;
          if (ankleRunFrames >= SUSTAINED_VIOLATION_FRAMES && ankleRunMin > sustainedAnkleDriftPx) {
            sustainedAnkleDriftPx = ankleRunMin;
          }
        } else {
          ankleRunFrames = 0;
          ankleRunMin = Infinity;
        }
      } else {
        ankleRunFrames = 0;
        ankleRunMin = Infinity;
      }
    }
    const maxHeelDriftPx = maxHeelRisePx; // back-compat field alias

    const reachPx = pk.value;
    const reachCm = pxToCm(reachPx, calibration);
    const heelDriftCm = pxToCm(maxHeelDriftPx, calibration);
    const ankleDriftCm = pxToCm(maxAnkleDriftPx, calibration);

    // Validity is gated on the SUSTAINED value, not the bare maximum.
    // A brief excursion above threshold no longer voids a trial.
    let validity: TrialValidity = "valid";
    let invalidityDetail: string | null = null;
    if (reachPx < minReachThreshPx) {
      validity = "no_motion";
      invalidityDetail = "Reach below minimum threshold — likely a noisy detection.";
    } else if (sustainedHeelRisePx > 0) {
      validity = "heel_rise";
      const sustainedCm = pxToCm(sustainedHeelRisePx, calibration);
      const cmText = sustainedCm !== null
        ? `${sustainedCm.toFixed(1)} cm`
        : `${sustainedHeelRisePx.toFixed(0)} px`;
      const thresholdText = calibration
        ? `${HEEL_RISE_THRESHOLD_CM.toFixed(1)} cm`
        : `${heelRiseThreshPx.toFixed(0)} px`;
      invalidityDetail = `Heel lifted ${cmText} for ≥ ${SUSTAINED_VIOLATION_FRAMES} frames (threshold ${thresholdText}).`;
    } else if (sustainedAnkleDriftPx > 0) {
      validity = "step";
      const sustainedCm = pxToCm(sustainedAnkleDriftPx, calibration);
      const cmText = sustainedCm !== null
        ? `${sustainedCm.toFixed(1)} cm`
        : `${sustainedAnkleDriftPx.toFixed(0)} px`;
      const thresholdText = calibration
        ? `${STEP_THRESHOLD_CM.toFixed(1)} cm`
        : `${stepThreshPx.toFixed(0)} px`;
      invalidityDetail = `Foot shifted ${cmText} for ≥ ${SUSTAINED_VIOLATION_FRAMES} frames (threshold ${thresholdText}).`;
    }
    // Acknowledge that maxHeelDriftPx + maxAnkleDriftPx are now
    // purely informational (reported in the trial log).
    void heelDriftCm; void ankleDriftCm;

    // Signed displacement at peak (for direction visualisation).
    const peakWristX = samples[pk.index].wrist_x_px;
    const signedDisp =
      peakWristX === null ? 0 : peakWristX - baselineWristX;

    return {
      trial_index: i,
      peak_sample_index: pk.index,
      peak_t_ms: samples[pk.index].t_ms,
      signed_displacement_px: signedDisp,
      reach_px: reachPx,
      reach_cm: reachCm,
      trunk_angle_at_peak_deg: samples[pk.index].trunk_angle_deg,
      validity,
      invalidity_detail: invalidityDetail,
      window_start_index: prevMid,
      window_end_index: nextMid,
      max_heel_drift_px: maxHeelDriftPx,
      max_heel_drift_cm: heelDriftCm,
      max_ankle_drift_px: maxAnkleDriftPx,
      max_ankle_drift_cm: ankleDriftCm,
    };
  });

  // 6) Best valid trial — highest reach among valid ones.
  let bestIdx: number | null = null;
  let bestReachPx: number | null = null;
  for (const t of trials) {
    if (t.validity !== "valid") continue;
    if (bestReachPx === null || t.reach_px > bestReachPx) {
      bestReachPx = t.reach_px;
      bestIdx = t.trial_index;
    }
  }
  const bestReachCm = bestReachPx !== null ? pxToCm(bestReachPx, calibration) : null;

  const classification: FallRiskClass | null =
    bestReachCm !== null ? classifyFallRisk(bestReachCm) : null;

  return {
    side_tested: side,
    baseline_locked: true,
    baseline_locked_at_index: baselineEndIdx,
    baseline_wrist_x_px: baselineWristX,
    baseline_ankle_x_px: baselineAnkleX,
    baseline_heel_y_px: baselineHeelY,
    trials,
    best_valid_trial_index: bestIdx,
    best_valid_reach_px: bestReachPx,
    best_valid_reach_cm: bestReachCm,
    classification,
    calibration,
    duration_seconds: duration,
    termination,
    samples,
    keypoints,
    peak_screenshot_data_url: peakScreenshotDataUrl,
  };
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return 0;
  if (n % 2 === 1) return sorted[(n - 1) >> 1];
  return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

/** Centered rolling-median filter. Single-frame outliers (MediaPipe
 *  glitches) are rejected; sustained motion across ≥ ⌈window/2⌉
 *  samples is preserved. Nulls inside the window are simply omitted
 *  from that window's median rather than poisoning it. */
function rollingMedianFilter(
  values: Array<number | null>,
  window: number,
): Array<number | null> {
  const out: Array<number | null> = new Array(values.length).fill(null);
  const half = Math.floor(window / 2);
  const buf: number[] = [];
  for (let i = 0; i < values.length; i++) {
    buf.length = 0;
    const lo = Math.max(0, i - half);
    const hi = Math.min(values.length - 1, i + half);
    for (let j = lo; j <= hi; j++) {
      const v = values[j];
      if (v !== null) buf.push(v);
    }
    if (buf.length === 0) continue;
    buf.sort((a, b) => a - b);
    out[i] = buf[(buf.length - 1) >> 1];
  }
  return out;
}

// ─── Classification ─────────────────────────────────────────────

export function classifyFallRisk(reachCm: number): FallRiskClass {
  if (reachCm < VERY_HIGH_FALL_RISK_MAX_CM) return "very_high";
  if (reachCm < MODERATE_FALL_RISK_MIN_CM) return "high";
  if (reachCm < LOW_FALL_RISK_MIN_CM) return "moderate";
  return "low";
}

export const FALL_RISK_LABEL: Record<FallRiskClass, string> = {
  low: "Low fall risk",
  moderate: "Moderate fall risk",
  high: "High fall risk",
  very_high: "Very high fall risk",
};

export const FALL_RISK_TONE: Record<FallRiskClass, string> = {
  low: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  moderate: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  high: "bg-orange-500/10 text-orange-700 dark:text-orange-400",
  very_high: "bg-red-500/10 text-red-700 dark:text-red-400",
};

export const TRIAL_VALIDITY_LABEL: Record<TrialValidity, string> = {
  valid: "Valid",
  heel_rise: "Heel rise",
  step: "Step",
  no_motion: "No motion",
};

export const TRIAL_VALIDITY_TONE: Record<TrialValidity, string> = {
  valid: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  heel_rise: "bg-red-500/10 text-red-700 dark:text-red-400",
  step: "bg-red-500/10 text-red-700 dark:text-red-400",
  no_motion: "bg-zinc-500/10 text-zinc-700 dark:text-zinc-400",
};

// ─── Interpretation ─────────────────────────────────────────────

export function buildInterpretation(result: FunctionalReachResult): string {
  if (!result.baseline_locked) {
    return (
      "Baseline could not be locked — the patient's test-side arm wasn't held " +
      "stably at shoulder height for the required interval. Re-record with the " +
      "near arm raised to ~90° for at least one second before reaching."
    );
  }
  const validTrials = result.trials.filter((t) => t.validity === "valid");
  if (result.trials.length === 0) {
    return "No reach attempts were detected within the recording window.";
  }
  if (validTrials.length === 0) {
    const heelRise = result.trials.some((t) => t.validity === "heel_rise");
    const step = result.trials.some((t) => t.validity === "step");
    const reasons: string[] = [];
    if (heelRise) reasons.push("heel rise");
    if (step) reasons.push("stepping");
    return (
      `All ${result.trials.length} trials were voided` +
      (reasons.length ? ` (${reasons.join(" + ")})` : "") +
      ". Re-record asking the patient to keep both heels down and the feet planted."
    );
  }

  if (result.best_valid_reach_cm === null) {
    // Uncalibrated — only relative units to talk about.
    const px = result.best_valid_reach_px ?? 0;
    return (
      `Best valid reach was ${px.toFixed(0)} px (relative units — no scale ` +
      `calibration was applied). Fall-risk classification requires ` +
      `height-based calibration for absolute distance.`
    );
  }

  const reach = result.best_valid_reach_cm.toFixed(1);
  const cls = result.classification;
  if (cls === "low") {
    return (
      `Best valid reach: ${reach} cm — low fall risk (≥ ${LOW_FALL_RISK_MIN_CM} cm). ` +
      `Normal functional reach for community-dwelling adults.`
    );
  }
  if (cls === "moderate") {
    return (
      `Best valid reach: ${reach} cm — moderate fall risk ` +
      `(${MODERATE_FALL_RISK_MIN_CM}–${LOW_FALL_RISK_MIN_CM} cm). ` +
      `Consider balance training and review medications affecting balance.`
    );
  }
  if (cls === "high") {
    return (
      `Best valid reach: ${reach} cm — high fall risk ` +
      `(${VERY_HIGH_FALL_RISK_MAX_CM}–${MODERATE_FALL_RISK_MIN_CM} cm). ` +
      `Recommend a comprehensive fall-prevention programme.`
    );
  }
  return (
    `Best valid reach: ${reach} cm — very high fall risk ` +
    `(< ${VERY_HIGH_FALL_RISK_MAX_CM} cm). Significant balance impairment; ` +
    `urgent fall-prevention assessment indicated.`
  );
}

// ─── Upload-mode API client ─────────────────────────────────────

interface FunctionalReachResponseDTO {
  success: boolean;
  data: FunctionalReachResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

function humanizeUploadError(raw: string | null): string {
  if (!raw) return "Analysis failed. Please try again.";
  const s = raw.toLowerCase();
  if (s.includes("poor_visibility")) {
    return (
      "Patient's body is not clearly visible. Re-record from the side with " +
      "the full body (head to feet) and the raised arm in frame."
    );
  }
  if (s.includes("no_baseline")) {
    return (
      "Couldn't establish a baseline — the patient's arm was never stably " +
      "held at shoulder height before reaching. Re-record with the arm at " +
      "~90° for ~1 s before the first reach."
    );
  }
  if (s.includes("no_reach")) {
    return (
      "No reach attempts detected. Re-record with the patient reaching " +
      "forward as far as comfortable, three times."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video quality too low. Please record at 24 fps or higher.";
  }
  if (s.includes("too short")) {
    return "Video is too short. Please record at least 10 seconds.";
  }
  if (s.includes("too long")) {
    return "Video is too long.";
  }
  if (s.includes("file too large")) {
    return "File too large.";
  }
  return raw;
}

/** Upload a Functional Reach clip + optional calibration payload to
 *  the backend.
 *
 *  Two ways to convey scale to the backend:
 *    1. `calibration` — a complete CalibrationResult (frontend already
 *       locked one in via HeightCalibrationStep in live mode).
 *    2. `patientHeightCm` — the patient's standing height. Backend
 *       measures body pixel height from the early frames of the clip
 *       and derives pixels_per_cm itself.
 *
 *  Either / both are optional; if neither is provided the test runs
 *  in relative-units mode. */
export async function analyzeFunctionalReachUpload(
  file: File,
  side: Side,
  calibration: CalibrationResult | null,
  patientHeightCm: number | null,
  onProgress?: (pct: number) => void,
): Promise<FunctionalReachResult> {
  const form = new FormData();
  form.append("video", file, file.name || "functional_reach.mp4");
  form.append("side", side);
  if (calibration) {
    form.append("calibration", JSON.stringify(calibration));
  }
  if (patientHeightCm !== null && Number.isFinite(patientHeightCm) && patientHeightCm > 0) {
    form.append("patient_height_cm", String(patientHeightCm));
  }

  onProgress?.(5);
  let pulseHandle: ReturnType<typeof setTimeout> | null = null;
  if (onProgress) {
    let pct = 5;
    const pulse = () => {
      pct = Math.min(90, pct + 5);
      onProgress(pct);
      pulseHandle = setTimeout(pulse, 1500);
    };
    pulseHandle = setTimeout(pulse, 1500);
  }

  try {
    const res = await authedFetch("/api/analyze-functional-reach", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `Functional Reach analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as FunctionalReachResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
