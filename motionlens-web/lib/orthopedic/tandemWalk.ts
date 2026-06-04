// Tandem Walk (E1) — heel-to-toe gait screen for cerebellar /
// vestibular dysfunction.
//
// Setup:
//   A straight line is marked on the floor with tape. Patient walks
//   heel-to-toe along it, taking 10 steps TOWARD the camera with eyes
//   open. Frontal view, full body in frame.
//
// Why this module is SEPARATE from existing modules:
//   • Frontal camera (not lateral like SLR/AKE/MTT/FL/STSQ).
//   • Multi-step (10 footstrikes), not multi-rep — each step is a
//     discrete planting event, not a repeated up/down cycle. So this
//     module CANNOT reuse the SLS/FL `PeakState` detector; we use a
//     per-foot velocity-threshold state machine instead.
//   • Single trial, no L/R split.
//
// Reference line (no operator calibration):
//   The patient's hip-midpoint trajectory is fit by least squares to
//   the form  x = a·y + b. That line is the "intended walking path".
//   For each footstrike at (fx, fy) the perpendicular deviation from
//   the line is  |fx − (a·fy + b)| / √(1 + a²)  in pixels.
//
// Pixel-to-cm normalisation:
//   At each footstrike frame we use the patient's shoulder width in
//   pixels as a scale ruler. With  ASSUMED_SHOULDER_WIDTH_CM = 40
//   (adult average), an approximate cm deviation is
//     dev_cm = (dev_px / shoulder_width_px) × 40
//   This keeps the metric scale-stable as the patient walks toward
//   the camera (shoulder width grows in proportion).

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";

const VIS_THRESHOLD = 0.3;

// Sample rate for the per-frame time-series stored on the result.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// 10 steps + a few seconds of setup; auto-finishes on the 10th.
export const TARGET_STEP_COUNT = 10;
export const TRIAL_TIMEOUT_SEC = 30;

// Foot velocity threshold for "planted" state. Above this magnitude
// the foot is in swing; below for at least MIN_PLANTED_FRAMES → the
// foot has struck the ground. 100 px/sec sits well above MediaPipe's
// natural per-frame keypoint jitter (typically 5-30 px/sec on a
// stationary subject) so noise can't oscillate the state machine.
export const PLANTED_VELOCITY_PX_PER_SEC = 100;
export const MIN_PLANTED_FRAMES = 2;
// Minimum frames between consecutive footstrikes per foot (debounce).
// Real tandem steps take 1-2 s; this 1-second floor prevents rapid-fire
// false strikes from any residual jitter that gets past the velocity
// gate.
export const MIN_FRAMES_BETWEEN_STRIKES = 10;
// Minimum swing distance (as a fraction of shoulder width in pixels)
// the foot must travel during a swing before the next plant counts as
// a real footstrike. Even with the higher velocity gate, a brief
// jitter spike could mark a 1-2 frame "swing" without the foot
// actually moving — this filter rejects those.
export const MIN_SWING_DISPLACEMENT_RATIO = 0.15;

// Scale assumption — adult average shoulder-to-shoulder width.
// 40 cm is a reasonable single-number proxy; the metric is the RATIO
// (deviation / shoulder width), this constant just converts back to
// a clinically readable cm number.
export const ASSUMED_SHOULDER_WIDTH_CM = 40;

// Deviation tolerance ("noise floor") subtracted from every raw
// footstrike deviation before the value is checked against the
// misstep / mean cutoffs. Even a patient walking perfectly straight
// along the line shows a few cm of computed deviation because of:
//   • MediaPipe pose-detection jitter at the moment of strike,
//   • natural hip-midpoint sway during gait (the reference line is
//     fit to the hip-midpoint trail and absorbs only the long-term
//     average — short-term wobble still leaks into per-step deviation),
//   • the ankle keypoint sits ABOVE the foot's contact point with
//     the floor, so even a perfectly-on-line foot reads a few cm
//     off when the patient's body sways even slightly,
//   • approximate cm conversion via ASSUMED_SHOULDER_WIDTH_CM (the
//     patient's real shoulder width may differ by ~10 %).
// 5 cm ≈ 12 % of the 40 cm shoulder-width scale. Sits above the
// 5-10 % clinical tolerance band typically cited because the
// hip-midpoint-fit line is a noisier reference than a pre-calibrated
// floor marker would be.
export const DEVIATION_TOLERANCE_CM = 5;

// Misstep — a foot landing visibly off the walking line. Threshold
// applied to the EFFECTIVE deviation (raw minus the tolerance above),
// so a real sidestep ≥ MISSTEP_DEVIATION_CM + DEVIATION_TOLERANCE_CM
// raw cm registers, but normal heel-to-toe wobble does not.
export const MISSTEP_DEVIATION_CM = 6;

// Classification cutoffs (user spec):
//   ≥ 2 missteps in 10 steps  → positive screen
//   mean deviation > 3 cm     → abnormal foot placement
export const POSITIVE_SCREEN_MISSTEP_COUNT = 2;
export const ABNORMAL_MEAN_DEVIATION_CM = 3;
export const BORDERLINE_MEAN_DEVIATION_CM = 1.5;

// Arm-grab — wrist abducted more than this many degrees from the
// body axis means the patient is throwing the arm out for balance.
export const ARM_ABDUCTION_DEG = 45;
// Frames before a continuous arm-out segment counts as a new "event".
// Without this debounce one sustained abduction counts as N events.
export const ARM_GRAB_DEDUPE_FRAMES = 5;

export type TandemClassification = "normal" | "borderline" | "positive_screen";
export type Termination = "completed" | "timeout" | "stopped";
export type FootSide = "left" | "right";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

// ─── Per-frame computations ─────────────────────────────────────

/** Midpoint x of the two hips at this frame. null when both hips
 *  aren't visible. The trail of these points is what we fit the
 *  walking line to. */
export function computeHipMidX(keypoints: Keypoint[]): number | null {
  const lh = keypoints[LM.LEFT_HIP];
  const rh = keypoints[LM.RIGHT_HIP];
  if (!visible(lh) || !visible(rh)) return null;
  return (lh.x + rh.x) / 2;
}

/** Midpoint y of the two hips at this frame. */
export function computeHipMidY(keypoints: Keypoint[]): number | null {
  const lh = keypoints[LM.LEFT_HIP];
  const rh = keypoints[LM.RIGHT_HIP];
  if (!visible(lh) || !visible(rh)) return null;
  return (lh.y + rh.y) / 2;
}

/** Midpoint x of the two shoulders — used for trunk sway. */
export function computeShoulderMidX(keypoints: Keypoint[]): number | null {
  const ls = keypoints[LM.LEFT_SHOULDER];
  const rs = keypoints[LM.RIGHT_SHOULDER];
  if (!visible(ls) || !visible(rs)) return null;
  return (ls.x + rs.x) / 2;
}

/** Shoulder-to-shoulder pixel distance — used to normalise pixel
 *  measurements to approximate cm. */
export function computeShoulderWidthPx(keypoints: Keypoint[]): number | null {
  const ls = keypoints[LM.LEFT_SHOULDER];
  const rs = keypoints[LM.RIGHT_SHOULDER];
  if (!visible(ls) || !visible(rs)) return null;
  return Math.hypot(ls.x - rs.x, ls.y - rs.y);
}

/** Foot position for a given side. Prefer ankle; fall back to the
 *  foot_index keypoint when the ankle isn't visible (often happens
 *  when the back foot's heel is occluded by the front foot's shin in
 *  tight tandem stance). */
export function computeFootPos(
  keypoints: Keypoint[], side: FootSide,
): { x: number; y: number } | null {
  const ankleIdx = side === "left" ? LM.LEFT_ANKLE : LM.RIGHT_ANKLE;
  const footIdx = side === "left" ? LM.LEFT_FOOT_INDEX : LM.RIGHT_FOOT_INDEX;
  const ankle = keypoints[ankleIdx];
  if (visible(ankle)) return { x: ankle.x, y: ankle.y };
  const foot = keypoints[footIdx];
  if (visible(foot)) return { x: foot.x, y: foot.y };
  return null;
}

/** Arm abduction angle (deg) — angle between the shoulder→wrist
 *  vector and the shoulder→hip vector. ~0° = arm hangs along the
 *  body; > ARM_ABDUCTION_DEG = arm thrown out laterally. */
export function computeArmAbductionDeg(
  keypoints: Keypoint[], side: FootSide,
): number | null {
  const shIdx = side === "left" ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER;
  const wrIdx = side === "left" ? LM.LEFT_WRIST : LM.RIGHT_WRIST;
  const hipIdx = side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP;
  const sh = keypoints[shIdx];
  const wr = keypoints[wrIdx];
  const hp = keypoints[hipIdx];
  if (!visible(sh) || !visible(wr) || !visible(hp)) return null;
  const ax = wr.x - sh.x; const ay = wr.y - sh.y;
  const bx = hp.x - sh.x; const by = hp.y - sh.y;
  const ma = Math.hypot(ax, ay); const mb = Math.hypot(bx, by);
  if (ma === 0 || mb === 0) return null;
  const cos = Math.max(-1, Math.min(1, (ax * bx + ay * by) / (ma * mb)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/** Pre-record gate: both ankles + hips + shoulders visible. */
export function isPatientTrackable(keypoints: Keypoint[]): boolean {
  return (
    visible(keypoints[LM.LEFT_SHOULDER]) &&
    visible(keypoints[LM.RIGHT_SHOULDER]) &&
    visible(keypoints[LM.LEFT_HIP]) &&
    visible(keypoints[LM.RIGHT_HIP]) &&
    visible(keypoints[LM.LEFT_ANKLE]) &&
    visible(keypoints[LM.RIGHT_ANKLE])
  );
}

// ─── Per-foot step detector (live mode) ─────────────────────────
//
// State machine per foot:
//   "init"      no Y history yet
//   "swinging"  velocity > threshold — foot is moving
//   "planted"   velocity < threshold for ≥ MIN_PLANTED_FRAMES
//
// A footstrike fires on the swinging → planted transition. The
// detector tracks each foot independently; the live capture loop
// runs it twice per frame (once per side).

export interface FootState {
  prevY: number | null;
  prevPrevY: number | null;
  prevPrevPrevY: number | null;
  lowVelFrames: number;
  state: "init" | "swinging" | "planted";
  lastStrikeFrame: number;
  /** Foot Y at the moment the current swing began (state went
   *  init/planted → swinging). Used to measure how far the foot
   *  travelled before the next plant. */
  swingStartY: number | null;
  /** Maximum |footY − swingStartY| observed during the current swing,
   *  in pixels. A real footstrike requires this to be at least
   *  `minSwingDisplacementPx` (passed in by the caller) before the
   *  swing → plant transition counts. */
  swingMaxDisplacementPx: number;
}

export function newFootState(): FootState {
  return {
    prevY: null,
    prevPrevY: null,
    prevPrevPrevY: null,
    lowVelFrames: 0,
    state: "init",
    lastStrikeFrame: -MIN_FRAMES_BETWEEN_STRIKES,
    swingStartY: null,
    swingMaxDisplacementPx: 0,
  };
}

/** Advances the foot's state machine by one frame. Returns true iff
 *  this frame is the moment a swinging → planted transition was
 *  CONFIRMED — i.e. a real footstrike with verified swing
 *  displacement. The caller decides what to record. */
export function detectFootstrike(
  state: FootState,
  footY: number | null,
  frameIdx: number,
  /** Velocity threshold in PX per FRAME (not per second) at the
   *  caller's sampling rate. The capture loop converts the per-sec
   *  constant by dividing by SAMPLE_HZ. */
  velocityThresholdPxPerFrame: number,
  /** Minimum swing distance (pixels) the foot must traverse before
   *  the next plant counts as a real footstrike. The caller computes
   *  this from the current shoulder-width pixel measurement times
   *  MIN_SWING_DISPLACEMENT_RATIO. */
  minSwingDisplacementPx: number,
): boolean {
  if (footY === null) {
    state.prevPrevPrevY = state.prevPrevY;
    state.prevPrevY = state.prevY;
    state.prevY = null;
    state.lowVelFrames = 0;
    return false;
  }

  // 3-frame central-difference velocity for noise robustness.
  let velAbs = Infinity;
  if (state.prevPrevY !== null && state.prevPrevPrevY !== null) {
    velAbs = Math.abs((footY - state.prevPrevPrevY) / 3);
  }

  let strike = false;
  const moving = velAbs > velocityThresholdPxPerFrame;
  if (moving) {
    state.lowVelFrames = 0;
    if (state.state !== "swinging") {
      // Entered swing — record the start position so we can verify
      // the foot actually traveled before the next plant.
      state.state = "swinging";
      state.swingStartY = footY;
      state.swingMaxDisplacementPx = 0;
    } else if (state.swingStartY !== null) {
      const disp = Math.abs(footY - state.swingStartY);
      if (disp > state.swingMaxDisplacementPx) {
        state.swingMaxDisplacementPx = disp;
      }
    }
  } else {
    state.lowVelFrames += 1;
    if (
      state.state === "swinging" &&
      state.lowVelFrames >= MIN_PLANTED_FRAMES &&
      frameIdx - state.lastStrikeFrame >= MIN_FRAMES_BETWEEN_STRIKES &&
      // NEW: foot must have actually swept forward by at least the
      // minimum displacement. A brief velocity spike from jitter
      // would otherwise still register as a swing→plant cycle.
      state.swingMaxDisplacementPx >= minSwingDisplacementPx
    ) {
      state.state = "planted";
      state.lastStrikeFrame = frameIdx;
      state.swingStartY = null;
      state.swingMaxDisplacementPx = 0;
      strike = true;
    }
  }

  state.prevPrevPrevY = state.prevPrevY;
  state.prevPrevY = state.prevY;
  state.prevY = footY;
  return strike;
}

// ─── Walking-line fit + perpendicular deviation ─────────────────

/** Least-squares fit of x = a·y + b through the hip-midpoint samples.
 *  Returns null when fewer than 3 valid hip-mid points are available. */
export function fitWalkingLine(
  hipMidPoints: Array<{ x: number; y: number }>,
): { a: number; b: number } | null {
  if (hipMidPoints.length < 3) return null;
  const n = hipMidPoints.length;
  let sumY = 0, sumX = 0, sumYY = 0, sumXY = 0;
  for (const p of hipMidPoints) {
    sumY += p.y; sumX += p.x;
    sumYY += p.y * p.y;
    sumXY += p.x * p.y;
  }
  const denom = n * sumYY - sumY * sumY;
  if (Math.abs(denom) < 1e-9) return null;
  const a = (n * sumXY - sumX * sumY) / denom;
  const b = (sumX - a * sumY) / n;
  return { a, b };
}

/** Perpendicular pixel distance from a point to a line x = a·y + b. */
export function perpDistancePx(
  px: number, py: number,
  line: { a: number; b: number },
): number {
  const num = Math.abs(px - (line.a * py + line.b));
  const den = Math.sqrt(1 + line.a * line.a);
  return num / den;
}

// ─── Per-step + per-trial aggregation ───────────────────────────

export interface FrameSample {
  t_ms: number;
  hip_mid_x: number | null;
  hip_mid_y: number | null;
  shoulder_mid_x: number | null;
  shoulder_width_px: number | null;
  left_foot_x: number | null;
  left_foot_y: number | null;
  right_foot_x: number | null;
  right_foot_y: number | null;
  left_arm_abduction_deg: number | null;
  right_arm_abduction_deg: number | null;
}

export interface StepEvent {
  step_index: number;            // 1..10
  side: FootSide;
  t_ms: number;
  foot_x: number;
  foot_y: number;
  /** Perpendicular pixel distance from the walking-line fit. Computed
   *  POST-HOC after the line is known — null until then. */
  deviation_px: number | null;
  /** Raw cm deviation before any tolerance is subtracted — normalised
   *  by shoulder width at this step's frame × ASSUMED_SHOULDER_WIDTH_CM.
   *  Kept on the step for transparency / debugging. */
  raw_deviation_cm: number | null;
  /** Effective cm deviation = max(0, raw - DEVIATION_TOLERANCE_CM).
   *  This is the value used for misstep flagging, mean / max
   *  aggregation, and classification. */
  deviation_cm: number | null;
  /** True iff deviation_cm (the effective value) > MISSTEP_DEVIATION_CM. */
  is_misstep: boolean;
  /** Shoulder width in pixels at this step (the scale ruler). */
  shoulder_width_px: number | null;
}

export interface TandemWalkResult {
  steps: StepEvent[];
  // Aggregate metrics
  misstep_count: number;
  arm_grab_count: number;
  mean_deviation_cm: number;
  max_deviation_cm: number;
  step_time_mean_ms: number;
  step_time_stddev_ms: number;
  step_time_cv: number;                  // stddev / mean
  trunk_sway_range_px: number;           // raw range of shoulder-mid x residuals
  trunk_sway_range_cm: number;           // converted via mean shoulder width
  classification: TandemClassification;
  duration_seconds: number;
  termination: Termination;
  incomplete: boolean;
  // Walking line + scale (kept on result so the report can draw)
  walking_line: { a: number; b: number } | null;
  mean_shoulder_width_px: number;
  // Time series
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  capture_screenshot_data_url: string | null;
  // Patient-context input (optional, recorded verbatim)
  patient_age: number | null;
}

// ─── Helpers ────────────────────────────────────────────────────

function stddev(values: number[]): number {
  if (values.length < 2) return 0;
  const m = values.reduce((s, v) => s + v, 0) / values.length;
  return Math.sqrt(
    values.reduce((s, v) => s + (v - m) * (v - m), 0) / values.length,
  );
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

// ─── Classification ─────────────────────────────────────────────

export function classifyTandemWalk(
  misstepCount: number,
  meanDeviationCm: number,
): TandemClassification {
  if (
    misstepCount >= POSITIVE_SCREEN_MISSTEP_COUNT ||
    meanDeviationCm > ABNORMAL_MEAN_DEVIATION_CM
  ) {
    return "positive_screen";
  }
  if (misstepCount === 1 || meanDeviationCm > BORDERLINE_MEAN_DEVIATION_CM) {
    return "borderline";
  }
  return "normal";
}

// ─── Per-trial aggregator ──────────────────────────────────────

/**
 * Walks the samples + the list of detected footstrike events and
 * produces a fully-populated TandemWalkResult. The same algorithm
 * runs in both modes (live calls this at trial end, upload calls
 * the Python mirror).
 */
export function summarizeTrial(
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  samples: FrameSample[],
  rawStrikes: Array<{ side: FootSide; sample_index: number }>,
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  captureScreenshotDataUrl: string | null,
  patientAge: number | null,
): TandemWalkResult {
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);

  // 1) Fit the walking line through hip-midpoint samples.
  const hipMidPoints: Array<{ x: number; y: number }> = [];
  for (const s of samples) {
    if (s.hip_mid_x !== null && s.hip_mid_y !== null) {
      hipMidPoints.push({ x: s.hip_mid_x, y: s.hip_mid_y });
    }
  }
  const line = fitWalkingLine(hipMidPoints);

  // 2) Mean shoulder width across the trial — used for trunk-sway cm.
  const shoulderWidthVals: number[] = samples
    .map((s) => s.shoulder_width_px)
    .filter((v): v is number => v !== null && v > 0);
  const meanShoulderWidthPx = mean(shoulderWidthVals);

  // 3) Build StepEvent objects with deviation in px AND cm.
  const steps: StepEvent[] = [];
  for (let i = 0; i < Math.min(rawStrikes.length, TARGET_STEP_COUNT); i++) {
    const r = rawStrikes[i];
    const s = samples[r.sample_index];
    if (!s) continue;
    const foot = r.side === "left"
      ? { x: s.left_foot_x, y: s.left_foot_y }
      : { x: s.right_foot_x, y: s.right_foot_y };
    if (foot.x === null || foot.y === null) continue;

    const devPx = line !== null ? perpDistancePx(foot.x, foot.y, line) : null;
    const sw = s.shoulder_width_px;
    const rawDevCm = devPx !== null && sw !== null && sw > 0
      ? (devPx / sw) * ASSUMED_SHOULDER_WIDTH_CM
      : null;
    // Subtract the deviation tolerance noise floor so natural wobble
    // (a few cm of unavoidable hip-midpoint sway + pose-detection
    // jitter) does not register as deviation. Effective value is what
    // the misstep + classification logic sees.
    const effectiveDevCm = rawDevCm !== null
      ? Math.max(0, rawDevCm - DEVIATION_TOLERANCE_CM)
      : null;
    const isMisstep =
      effectiveDevCm !== null && effectiveDevCm > MISSTEP_DEVIATION_CM;

    steps.push({
      step_index: i + 1,
      side: r.side,
      t_ms: s.t_ms,
      foot_x: foot.x,
      foot_y: foot.y,
      deviation_px: devPx,
      raw_deviation_cm: rawDevCm,
      deviation_cm: effectiveDevCm,
      is_misstep: isMisstep,
      shoulder_width_px: sw,
    });
  }

  // 4) Step-time statistics across consecutive footstrikes.
  const stepTimes: number[] = [];
  for (let i = 1; i < steps.length; i++) {
    stepTimes.push(steps[i].t_ms - steps[i - 1].t_ms);
  }
  const stepTimeMean = mean(stepTimes);
  const stepTimeStd = stddev(stepTimes);
  const stepTimeCV = stepTimeMean > 0 ? stepTimeStd / stepTimeMean : 0;

  // 5) Arm-grab count — discrete events from continuous abductions.
  let armGrabCount = 0;
  let inGrabSegment = false;
  let lastGrabSampleIdx = -ARM_GRAB_DEDUPE_FRAMES;
  for (let i = 0; i < samples.length; i++) {
    const sm = samples[i];
    const l = sm.left_arm_abduction_deg;
    const r = sm.right_arm_abduction_deg;
    const abducted =
      (l !== null && l > ARM_ABDUCTION_DEG) ||
      (r !== null && r > ARM_ABDUCTION_DEG);
    if (abducted) {
      if (!inGrabSegment && i - lastGrabSampleIdx >= ARM_GRAB_DEDUPE_FRAMES) {
        armGrabCount += 1;
        inGrabSegment = true;
        lastGrabSampleIdx = i;
      } else if (inGrabSegment) {
        lastGrabSampleIdx = i;
      }
    } else {
      inGrabSegment = false;
    }
  }

  // 6) Trunk sway — residuals of shoulder-mid x against the walking
  //    line (or against its mean if no line was fittable).
  let trunkResiduals: number[] = [];
  for (const sm of samples) {
    if (sm.shoulder_mid_x === null || sm.hip_mid_y === null) continue;
    const expectedX = line !== null
      ? line.a * sm.hip_mid_y + line.b
      : null;
    if (expectedX === null) {
      trunkResiduals.push(sm.shoulder_mid_x);
    } else {
      trunkResiduals.push(sm.shoulder_mid_x - expectedX);
    }
  }
  if (line === null && trunkResiduals.length > 0) {
    // Without a line, just detrend by the mean.
    const m = mean(trunkResiduals);
    trunkResiduals = trunkResiduals.map((v) => v - m);
  }
  const swayPx = trunkResiduals.length === 0
    ? 0
    : Math.max(...trunkResiduals) - Math.min(...trunkResiduals);
  const swayCm = meanShoulderWidthPx > 0
    ? (swayPx / meanShoulderWidthPx) * ASSUMED_SHOULDER_WIDTH_CM
    : 0;

  // 7) Deviation aggregates.
  const devCms = steps
    .map((st) => st.deviation_cm)
    .filter((v): v is number => v !== null);
  const meanDevCm = mean(devCms);
  const maxDevCm = devCms.length > 0 ? Math.max(...devCms) : 0;
  const misstepCount = steps.filter((s) => s.is_misstep).length;

  const incomplete = steps.length < TARGET_STEP_COUNT;
  const classification = classifyTandemWalk(misstepCount, meanDevCm);

  return {
    steps,
    misstep_count: misstepCount,
    arm_grab_count: armGrabCount,
    mean_deviation_cm: meanDevCm,
    max_deviation_cm: maxDevCm,
    step_time_mean_ms: stepTimeMean,
    step_time_stddev_ms: stepTimeStd,
    step_time_cv: stepTimeCV,
    trunk_sway_range_px: swayPx,
    trunk_sway_range_cm: swayCm,
    classification,
    duration_seconds: duration,
    termination,
    incomplete,
    walking_line: line,
    mean_shoulder_width_px: meanShoulderWidthPx,
    samples,
    keypoints,
    capture_screenshot_data_url: captureScreenshotDataUrl,
    patient_age: patientAge,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: TandemWalkResult | null): string {
  if (!result) return "No completed tandem-walk trial to interpret.";

  const summary = result.incomplete
    ? `${result.steps.length} of ${TARGET_STEP_COUNT} steps captured`
    : `${result.steps.length} steps captured`;

  const parts: string[] = [];
  parts.push(
    `${summary}. Mean lateral deviation ${result.mean_deviation_cm.toFixed(1)} cm, ` +
    `worst step ${result.max_deviation_cm.toFixed(1)} cm.`,
  );

  if (result.misstep_count >= POSITIVE_SCREEN_MISSTEP_COUNT) {
    parts.push(
      `${result.misstep_count} missteps detected ` +
      `(≥ ${POSITIVE_SCREEN_MISSTEP_COUNT} = positive screen for ` +
      `cerebellar / vestibular dysfunction).`,
    );
  } else if (result.misstep_count === 1) {
    parts.push(`1 misstep detected — borderline.`);
  } else {
    parts.push(`No missteps detected.`);
  }

  if (result.mean_deviation_cm > ABNORMAL_MEAN_DEVIATION_CM) {
    parts.push(
      `Mean foot placement was abnormal ` +
      `(> ${ABNORMAL_MEAN_DEVIATION_CM} cm from the walking line).`,
    );
  }

  if (result.arm_grab_count > 0) {
    parts.push(
      `${result.arm_grab_count} arm-grab event(s) — patient threw an arm ` +
      `out (> ${ARM_ABDUCTION_DEG}° abduction) for balance.`,
    );
  }

  if (result.step_time_cv > 0.25) {
    parts.push(
      `Step-time variability is high (CV = ${result.step_time_cv.toFixed(2)}) — ` +
      `irregular cadence, often seen with cerebellar involvement.`,
    );
  }

  if (result.trunk_sway_range_cm > 8) {
    parts.push(
      `Trunk sway range ${result.trunk_sway_range_cm.toFixed(1)} cm — broad ` +
      `lateral excursion suggests proximal instability.`,
    );
  }

  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const TANDEM_CLASSIFICATION_LABEL: Record<TandemClassification, string> = {
  normal:          "Normal",
  borderline:      "Borderline",
  positive_screen: "Positive screen",
};

export const TANDEM_CLASSIFICATION_TONE: Record<TandemClassification, string> = {
  normal:          "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  borderline:      "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  positive_screen: "bg-red-500/10 text-red-700 dark:text-red-400",
};

// ─── Upload-mode API client ─────────────────────────────────────

interface TandemWalkResponseDTO {
  success: boolean;
  data: TandemWalkResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

function humanizeUploadError(raw: string | null): string {
  if (!raw) return "Analysis failed. Please try again.";
  const s = raw.toLowerCase();
  if (s.includes("poor_visibility")) {
    return (
      "Patient is not clearly visible in the recording. Re-record with the " +
      "full body in frame, frontal view, patient walking toward the camera."
    );
  }
  if (s.includes("no_steps") || s.includes("steps not detected")) {
    return (
      "No footstrikes detected. Please re-record the patient performing a " +
      "10-step heel-to-toe walk toward the camera."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video quality too low. Please record at 24 fps or higher.";
  }
  if (s.includes("too short")) {
    return "Video is too short. Please record at least 5 seconds.";
  }
  if (s.includes("too long")) {
    return "Video is too long. Maximum 60 seconds.";
  }
  if (s.includes("file too large")) {
    return "File too large. Maximum 100 MB.";
  }
  return raw;
}

export async function analyzeTandemWalkUpload(
  file: File,
  patientAge: number | null,
  onProgress?: (pct: number) => void,
): Promise<TandemWalkResult> {
  const form = new FormData();
  form.append("video", file, file.name || "tandem_walk.mp4");
  if (patientAge !== null && Number.isFinite(patientAge)) {
    form.append("patient_age", String(patientAge));
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
    const res = await authedFetch("/api/analyze-tandem-walk", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `Tandem-walk analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as TandemWalkResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
