// Sit-to-Stand QUALITY (B4) — movement-quality functional test.
//
// This module is COMPLETELY SEPARATE from the existing 5x Sit-to-Stand
// (C2, see lib/orthopedic/sitToStand.ts). C2 measures SPEED — how
// quickly the patient completes 5 reps. B4 measures QUALITY — how the
// patient stands and sits, used post-TKR / post-THR and in geriatric
// rehab. They share no code; do not cross-import.
//
// Setup:
//   Standard chair (~45 cm seat height, no armrests). Patient seated,
//   feet flat, arms crossed at the chest. Stands up at a self-selected
//   (comfortable) pace, then sits back down with control. THREE reps.
//   Camera is LATERAL (side view), patient seen in profile, full body
//   visible. The operator chooses which side faces the camera before
//   starting; that side's hip, knee, ankle, shoulder, and wrist drive
//   all measurements.
//
// What we measure per rep:
//   • Phase timing — sit_to_stand_ms, pause_ms, stand_to_sit_ms.
//     Phase boundaries detected from the test-side hip Y velocity
//     trace (image y-down): rising = velocity negative; pause =
//     velocity ≈ 0 at top; descending = velocity positive.
//   • Trunk forward lean at seat-off — angle of test-side hip→shoulder
//     vs vertical, captured at the seat_off frame.
//   • Knee flexion at seat-off — inner angle at the knee
//     (knee→hip, knee→ankle), captured at the seat_off frame.
//   • Smoothness score — 1 / (1 + stddev(acc) / mean(|vel|)) over the
//     rising phase. 1.0 = constant velocity (perfectly smooth);
//     lower = jerkier.
//   • Hand-use flag — true if the test-side wrist dropped more than
//     0.10 × leg-length below the test-side shoulder during the rising
//     phase. Indicates the patient unfolded the arms to push off (a
//     compensation for lower-extremity weakness).
//
// Reported numbers are MEDIANS across the 3 reps.
//
// Rep detection re-uses the SLS / Forward-Lunge `PeakState` /
// `detectPeak` pattern, fed `-hipY` so the peak-detector fires at the
// VALLEY in actual hipY — i.e. the moment the patient is standing
// tallest. (In image y-down: standing = small y = "lowest" hipY = the
// valley in the raw trace.)

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";

const VIS_THRESHOLD = 0.3;

// Sample rate for the per-frame time-series stored on the result.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// 3 reps at ~5 s each + settle = ~25 s headroom.
export const TARGET_REP_COUNT = 3;
export const TRIAL_TIMEOUT_SEC = 45;

// Rep-detection tuning — fed `-hipY` so this becomes "patient must
// rise at least 50 px above the seated baseline" and "at least 30
// sampled frames between consecutive standing moments". Same numbers
// SLS / FL use because the underlying detector is identical.
export const PEAK_MIN_DEPTH_PX = 50;
export const PEAK_MIN_SEPARATION_FRAMES = 30;

// Stable-velocity threshold used for phase boundary detection.
// Expressed as |Δhip-Y per sample| ≤ this many pixels = "not moving".
// Below this we call the rising or descending segments terminated.
export const PHASE_STABLE_VELOCITY_PX = 2.0;

// Hand-use detection: the camera-side wrist must drop by at least
// this fraction of leg length BELOW the camera-side shoulder line
// during the rising phase to flag the rep.
export const HAND_USE_WRIST_DROP_RATIO = 0.10;

// Classification cutoffs (cross-checked with the user spec).
//   Trunk lean at seat-off:
//     30–45° = efficient pattern (clinical sweet spot)
//     > 55°  = momentum-dependent (hip/quad weakness)
export const TRUNK_LEAN_EFFICIENT_MIN_DEG = 30;
export const TRUNK_LEAN_EFFICIENT_MAX_DEG = 45;
export const TRUNK_LEAN_MOMENTUM_DEG = 55;

// Smoothness gate for the "smooth" verdict.
export const SMOOTHNESS_SMOOTH_MIN = 0.7;

export type Side = "left" | "right";
export type STSQualityClassification = "smooth" | "hesitant" | "compensated";
export type Termination = "completed" | "timeout" | "stopped";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

const SIDE_INDICES = {
  left: {
    shoulder: LM.LEFT_SHOULDER,
    wrist: LM.LEFT_WRIST,
    hip: LM.LEFT_HIP,
    knee: LM.LEFT_KNEE,
    ankle: LM.LEFT_ANKLE,
  },
  right: {
    shoulder: LM.RIGHT_SHOULDER,
    wrist: LM.RIGHT_WRIST,
    hip: LM.RIGHT_HIP,
    knee: LM.RIGHT_KNEE,
    ankle: LM.RIGHT_ANKLE,
  },
} as const;

// ─── Per-frame computations ─────────────────────────────────────

function vectorAngleDeg(
  ax: number, ay: number,
  bx: number, by: number,
): number | null {
  const ma = Math.hypot(ax, ay);
  const mb = Math.hypot(bx, by);
  if (ma === 0 || mb === 0) return null;
  const cos = Math.max(-1, Math.min(1, (ax * bx + ay * by) / (ma * mb)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/** Inner knee angle (180° = straight, 90° = bent square). Test-side
 *  hip → knee → ankle. Same formula as Forward Lunge. */
export function computeKneeAngle(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  if (!visible(hip) || !visible(knee) || !visible(ankle)) return null;
  return vectorAngleDeg(
    hip.x - knee.x, hip.y - knee.y,
    ankle.x - knee.x, ankle.y - knee.y,
  );
}

/** Trunk lean from vertical (0° = body upright, 90° = horizontal).
 *  Test-side hip → test-side shoulder. Same formula as Forward Lunge. */
export function computeTrunkLeanDeg(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const sh = keypoints[idx.shoulder];
  if (!visible(hip) || !visible(sh)) return null;
  const dx = Math.abs(sh.x - hip.x);
  const dy = Math.abs(sh.y - hip.y);
  if (dx === 0 && dy === 0) return null;
  return (Math.atan2(dx, dy) * 180) / Math.PI;
}

/** Test-side hip Y in pixels — drives rep detection (fed negated) and
 *  phase segmentation. */
export function computeHipY(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  if (!visible(hip)) return null;
  return hip.y;
}

/** Test-side wrist Y in pixels — used to detect hand-on-thigh push-off. */
export function computeWristY(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const wr = keypoints[idx.wrist];
  if (!visible(wr)) return null;
  return wr.y;
}

/** Test-side shoulder Y in pixels — reference line for hand-use detection. */
export function computeShoulderY(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const sh = keypoints[idx.shoulder];
  if (!visible(sh)) return null;
  return sh.y;
}

/** Test-side hip→ankle pixel distance — used to normalise the
 *  hand-use drop threshold by leg length. */
export function computeLegLengthPx(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const ankle = keypoints[idx.ankle];
  if (!visible(hip) || !visible(ankle)) return null;
  return Math.hypot(ankle.x - hip.x, ankle.y - hip.y);
}

/** Pre-record gate: the test-side keypoints we need throughout the
 *  trial must all be trackable. The contralateral side is occluded by
 *  the chair / body and is intentionally not required. */
export function isTestSideTrackable(keypoints: Keypoint[], side: Side): boolean {
  const idx = SIDE_INDICES[side];
  return (
    visible(keypoints[idx.shoulder]) &&
    visible(keypoints[idx.hip]) &&
    visible(keypoints[idx.knee]) &&
    visible(keypoints[idx.ankle])
  );
}

// ─── Rep detector (FL pattern, fed `-hipY` so peaks = standing) ─

interface PeakState {
  baselineY: number | null;
  prevY: number | null;
  prevPrevY: number | null;
  prevFrameIdx: number;
  lastPeakFrame: number;
  recentMaxY: number;
}

export function newPeakState(): PeakState {
  return {
    baselineY: null,
    prevY: null,
    prevPrevY: null,
    prevFrameIdx: 0,
    lastPeakFrame: -PEAK_MIN_SEPARATION_FRAMES,
    recentMaxY: -Infinity,
  };
}

/** Identical to lib/orthopedic/forwardLunge.ts:detectPeak — kept
 *  local so this module has no cross-dep on FL. The CALLER feeds
 *  `-hipY` so the detected "peak" is the patient at standing height
 *  (the valley in actual hipY in image y-down). */
export function detectPeak(
  state: PeakState,
  signal: number | null,
  frameIdx: number,
): boolean {
  if (signal === null) {
    state.prevPrevY = state.prevY;
    state.prevY = null;
    state.prevFrameIdx = frameIdx;
    return false;
  }
  if (state.baselineY === null) {
    state.baselineY = signal;
    state.prevY = signal;
    state.prevFrameIdx = frameIdx;
    return false;
  }

  let peakDetected = false;
  if (state.prevPrevY !== null && state.prevY !== null) {
    const ascending = state.prevY > state.prevPrevY;
    const turning   = signal < state.prevY;
    const sinceLast = frameIdx - state.lastPeakFrame;
    const depth     = state.prevY - state.baselineY;

    if (
      ascending &&
      turning &&
      sinceLast >= PEAK_MIN_SEPARATION_FRAMES &&
      depth >= PEAK_MIN_DEPTH_PX
    ) {
      state.lastPeakFrame = frameIdx - 1;
      peakDetected = true;
    }
  }

  state.prevPrevY = state.prevY;
  state.prevY = signal;
  state.prevFrameIdx = frameIdx;
  return peakDetected;
}

// ─── Phase boundary detection ───────────────────────────────────

/** Given the hipY trace, the seated-baseline value, and the index of
 *  a confirmed top-of-stand frame, walk outward to find the seat-off,
 *  start-of-descent, and re-seated frames. Returns null for any
 *  boundary the trace doesn't support (e.g. trial cut off mid-rep). */
export function detectPhaseBoundaries(
  hipYSeries: (number | null)[],
  baselineY: number,
  topOfStandIndex: number,
): {
  seat_off: number | null;
  top_of_stand: number;
  start_of_descent: number | null;
  re_seated: number | null;
} {
  const v = topOfStandIndex;

  // Walk BACKWARDS from v: the seat-off frame is where the rising
  // segment STARTED — the first frame walking back where hipY is
  // very close to baseline OR velocity dropped to ≈ 0.
  let seatOff: number | null = null;
  for (let i = v - 1; i >= 0; i--) {
    const cur = hipYSeries[i];
    const next = hipYSeries[i + 1];
    if (cur === null || next === null) continue;
    const vel = cur - next; // pos = was higher (in image y-down), i.e. moving down before this point
    // We're walking backwards through the RISING segment. In a rising
    // segment vel should be positive (cur was further from top than next).
    // Stop when vel approaches zero — that's where rising started.
    if (Math.abs(vel) < PHASE_STABLE_VELOCITY_PX) {
      seatOff = i;
      break;
    }
    // Also stop if hipY is back at (or below) baseline — we've reached
    // the seated platform.
    if (cur >= baselineY - PHASE_STABLE_VELOCITY_PX) {
      seatOff = i;
      break;
    }
  }
  if (seatOff === null) seatOff = 0;

  // Walk FORWARDS from v: start-of-descent is the first frame after
  // the pause where hipY meaningfully increases again.
  let startOfDescent: number | null = null;
  for (let i = v + 1; i < hipYSeries.length; i++) {
    const cur = hipYSeries[i];
    const prev = hipYSeries[i - 1];
    if (cur === null || prev === null) continue;
    const vel = cur - prev; // pos = hipY increasing = patient descending
    if (vel > PHASE_STABLE_VELOCITY_PX) {
      startOfDescent = i;
      break;
    }
  }

  // Continue walking forwards to find the re-seated frame — first
  // frame where hipY is back at (or above) baseline AND velocity has
  // returned to ≈ 0.
  let reSeated: number | null = null;
  if (startOfDescent !== null) {
    for (let i = startOfDescent + 1; i < hipYSeries.length; i++) {
      const cur = hipYSeries[i];
      const prev = hipYSeries[i - 1];
      if (cur === null || prev === null) continue;
      const atBaseline = cur >= baselineY - PHASE_STABLE_VELOCITY_PX;
      const vel = cur - prev;
      if (atBaseline && Math.abs(vel) < PHASE_STABLE_VELOCITY_PX) {
        reSeated = i;
        break;
      }
    }
  }

  return {
    seat_off: seatOff,
    top_of_stand: v,
    start_of_descent: startOfDescent,
    re_seated: reSeated,
  };
}

// ─── Smoothness score (rising-phase jerk proxy) ─────────────────

/** Returns a smoothness score in 0..1, where 1.0 = constant-velocity
 *  rise (no jerk). Computed as 1 / (1 + stddev(acceleration) /
 *  mean(|velocity|)) over the (seat_off → top_of_stand) window. */
export function computeSmoothnessScore(
  hipYSeries: (number | null)[],
  startIdx: number,
  endIdx: number,
): number | null {
  if (endIdx <= startIdx + 3) return null;
  const segment: number[] = [];
  for (let i = startIdx; i <= endIdx; i++) {
    const v = hipYSeries[i];
    if (v === null) return null;
    segment.push(v);
  }
  // Central-difference velocity per frame.
  const vel: number[] = [];
  for (let i = 1; i < segment.length - 1; i++) {
    vel.push((segment[i + 1] - segment[i - 1]) / 2);
  }
  if (vel.length < 3) return null;
  // Central-difference acceleration per frame.
  const acc: number[] = [];
  for (let i = 1; i < vel.length - 1; i++) {
    acc.push((vel[i + 1] - vel[i - 1]) / 2);
  }
  if (acc.length === 0) return null;
  const meanAbsV = vel.reduce((s, v) => s + Math.abs(v), 0) / vel.length;
  if (meanAbsV === 0) return null;
  const meanA = acc.reduce((s, v) => s + v, 0) / acc.length;
  const variance = acc.reduce((s, v) => s + (v - meanA) * (v - meanA), 0) / acc.length;
  const stdA = Math.sqrt(variance);
  const jerkProxy = stdA / meanAbsV;
  return 1 / (1 + jerkProxy);
}

// ─── Hand-use detection during rising phase ─────────────────────

/** Returns true iff at any frame in [startIdx, endIdx] the test-side
 *  wrist dropped more than HAND_USE_WRIST_DROP_RATIO × leg-length
 *  below the test-side shoulder. Both wrist and shoulder must be
 *  visible at that frame for it to count. */
export function detectHandUseInPhase(
  wristYSeries: (number | null)[],
  shoulderYSeries: (number | null)[],
  legLengthPxSeries: (number | null)[],
  startIdx: number,
  endIdx: number,
): boolean {
  for (let i = startIdx; i <= endIdx; i++) {
    const wr = wristYSeries[i];
    const sh = shoulderYSeries[i];
    const leg = legLengthPxSeries[i];
    if (wr === null || sh === null || leg === null || leg <= 0) continue;
    const dropPx = wr - sh; // image y-down: wrist below shoulder = wr > sh
    if (dropPx > leg * HAND_USE_WRIST_DROP_RATIO) {
      return true;
    }
  }
  return false;
}

// ─── Per-rep + per-trial aggregation ────────────────────────────

export interface FrameSample {
  t_ms: number;
  hip_y: number | null;
  knee_angle_deg: number | null;
  trunk_lean_deg: number | null;
  wrist_y: number | null;
  shoulder_y: number | null;
  leg_length_px: number | null;
}

export interface RepMetrics {
  rep_index: number;                    // 1..3
  seat_off_t_ms: number | null;
  top_of_stand_t_ms: number;
  start_of_descent_t_ms: number | null;
  re_seated_t_ms: number | null;
  sit_to_stand_ms: number | null;
  pause_ms: number | null;
  stand_to_sit_ms: number | null;
  trunk_lean_at_seat_off_deg: number | null;
  knee_angle_at_seat_off_deg: number | null;
  smoothness_score: number | null;      // 0..1, higher = smoother
  hand_use_detected: boolean;
}

export interface STSQualityResult {
  camera_side: Side;
  chair_seat_height_cm: number | null;
  reps: RepMetrics[];
  // Medians across captured reps. 0 / null when no rep contributed.
  median_sit_to_stand_ms: number | null;
  median_pause_ms: number | null;
  median_stand_to_sit_ms: number | null;
  median_trunk_lean_deg: number | null;
  median_knee_angle_deg: number | null;
  median_smoothness_score: number | null;
  hand_use_count: number;               // 0..3
  any_hand_use: boolean;
  classification: STSQualityClassification;
  duration_seconds: number;
  termination: Termination;
  incomplete: boolean;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  worst_rep_screenshot_data_url: string | null;
}

// ─── Median utility ─────────────────────────────────────────────

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

// ─── Classification ─────────────────────────────────────────────

export function classifySTSQuality(
  medianSmoothness: number | null,
  medianTrunkDeg: number | null,
  handUseCount: number,
): STSQualityClassification {
  // Compensated takes priority — any clear lower-extremity weakness sign.
  if (handUseCount >= 2) return "compensated";
  if (medianTrunkDeg !== null && medianTrunkDeg > TRUNK_LEAN_MOMENTUM_DEG) {
    return "compensated";
  }
  // Smooth requires both clean kinematics and no hand-use even once.
  if (
    handUseCount === 0 &&
    medianSmoothness !== null &&
    medianSmoothness >= SMOOTHNESS_SMOOTH_MIN
  ) {
    return "smooth";
  }
  return "hesitant";
}

// ─── Per-trial aggregator ──────────────────────────────────────

/** Walk the samples + the list of detected top-of-stand frame indices
 *  and produce a fully-populated STSQualityResult. `samples` is the
 *  per-frame time-series captured during the trial. */
export function summarizeTrial(
  cameraSide: Side,
  chairSeatHeightCm: number | null,
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  samples: FrameSample[],
  topOfStandSampleIndices: number[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  worstRepScreenshotDataUrl: string | null,
): STSQualityResult {
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);
  const hipYSeries: (number | null)[] = samples.map((s) => s.hip_y);
  const wristYSeries: (number | null)[] = samples.map((s) => s.wrist_y);
  const shoulderYSeries: (number | null)[] = samples.map((s) => s.shoulder_y);
  const legSeries: (number | null)[] = samples.map((s) => s.leg_length_px);

  // Seated baseline: use the median of the first 1 s of samples (or
  // the first 5 samples if the trial is shorter).
  const baselineWindow = Math.min(samples.length, Math.max(5, SAMPLE_HZ));
  const baselineCandidates: number[] = [];
  for (let i = 0; i < baselineWindow; i++) {
    const v = hipYSeries[i];
    if (v !== null) baselineCandidates.push(v);
  }
  const baselineY = median(baselineCandidates) ?? 0;

  const reps: RepMetrics[] = [];
  for (let r = 0; r < topOfStandSampleIndices.length && r < TARGET_REP_COUNT; r++) {
    const topIdx = topOfStandSampleIndices[r];
    const phase = detectPhaseBoundaries(hipYSeries, baselineY, topIdx);
    const tAt = (idx: number | null): number | null =>
      idx === null || idx < 0 || idx >= samples.length
        ? null
        : samples[idx].t_ms;

    const seatOffT  = tAt(phase.seat_off);
    const topT      = samples[topIdx].t_ms;
    const sodT      = tAt(phase.start_of_descent);
    const reseatedT = tAt(phase.re_seated);

    const sitToStand = seatOffT !== null
      ? Math.max(0, topT - seatOffT)
      : null;
    const pause = sodT !== null
      ? Math.max(0, sodT - topT)
      : null;
    const standToSit = sodT !== null && reseatedT !== null
      ? Math.max(0, reseatedT - sodT)
      : null;

    // Trunk + knee captured AT the seat_off moment (or the top frame
    // as a fallback if seat_off detection failed).
    const seatOffSample = phase.seat_off !== null
      ? samples[phase.seat_off]
      : samples[topIdx];

    // Smoothness over the rising window (seat_off → top_of_stand).
    const smoothness = phase.seat_off !== null
      ? computeSmoothnessScore(hipYSeries, phase.seat_off, topIdx)
      : null;

    // Hand-use scan over the rising phase only.
    const handUse = phase.seat_off !== null
      ? detectHandUseInPhase(
          wristYSeries, shoulderYSeries, legSeries,
          phase.seat_off, topIdx,
        )
      : false;

    reps.push({
      rep_index: r + 1,
      seat_off_t_ms: seatOffT,
      top_of_stand_t_ms: topT,
      start_of_descent_t_ms: sodT,
      re_seated_t_ms: reseatedT,
      sit_to_stand_ms: sitToStand,
      pause_ms: pause,
      stand_to_sit_ms: standToSit,
      trunk_lean_at_seat_off_deg: seatOffSample.trunk_lean_deg,
      knee_angle_at_seat_off_deg: seatOffSample.knee_angle_deg,
      smoothness_score: smoothness,
      hand_use_detected: handUse,
    });
  }

  const incomplete = reps.length < TARGET_REP_COUNT;
  const handUseCount = reps.filter((r) => r.hand_use_detected).length;

  const sitToStandVals  = reps.map((r) => r.sit_to_stand_ms).filter((v): v is number => v !== null);
  const pauseVals       = reps.map((r) => r.pause_ms).filter((v): v is number => v !== null);
  const standToSitVals  = reps.map((r) => r.stand_to_sit_ms).filter((v): v is number => v !== null);
  const trunkVals       = reps.map((r) => r.trunk_lean_at_seat_off_deg).filter((v): v is number => v !== null);
  const kneeVals        = reps.map((r) => r.knee_angle_at_seat_off_deg).filter((v): v is number => v !== null);
  const smoothnessVals  = reps.map((r) => r.smoothness_score).filter((v): v is number => v !== null);

  const medSit  = median(sitToStandVals);
  const medPau  = median(pauseVals);
  const medSit2 = median(standToSitVals);
  const medTrunk = median(trunkVals);
  const medKnee  = median(kneeVals);
  const medSm    = median(smoothnessVals);

  const classification = classifySTSQuality(medSm, medTrunk, handUseCount);

  return {
    camera_side: cameraSide,
    chair_seat_height_cm: chairSeatHeightCm,
    reps,
    median_sit_to_stand_ms: medSit,
    median_pause_ms: medPau,
    median_stand_to_sit_ms: medSit2,
    median_trunk_lean_deg: medTrunk,
    median_knee_angle_deg: medKnee,
    median_smoothness_score: medSm,
    hand_use_count: handUseCount,
    any_hand_use: handUseCount > 0,
    classification,
    duration_seconds: duration,
    termination,
    incomplete,
    samples,
    keypoints,
    worst_rep_screenshot_data_url: worstRepScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: STSQualityResult | null): string {
  if (!result) return "No completed STS-quality trial to interpret.";

  const repSummary = result.incomplete
    ? `${result.reps.length} of ${TARGET_REP_COUNT} reps captured`
    : `${result.reps.length} reps captured`;

  const issues: string[] = [];
  if (result.any_hand_use) {
    issues.push(
      `wrist pushed off below the shoulder on ${result.hand_use_count} of ` +
      `${result.reps.length} reps — significant lower-extremity weakness`,
    );
  }
  if (result.median_trunk_lean_deg !== null) {
    if (result.median_trunk_lean_deg > TRUNK_LEAN_MOMENTUM_DEG) {
      issues.push(
        `median trunk lean ${result.median_trunk_lean_deg.toFixed(1)}° at seat-off ` +
        `(> ${TRUNK_LEAN_MOMENTUM_DEG}°) — momentum-dependent strategy, hip/quad weakness`,
      );
    } else if (
      result.median_trunk_lean_deg >= TRUNK_LEAN_EFFICIENT_MIN_DEG &&
      result.median_trunk_lean_deg <= TRUNK_LEAN_EFFICIENT_MAX_DEG
    ) {
      // No issue — efficient pattern.
    } else if (result.median_trunk_lean_deg < TRUNK_LEAN_EFFICIENT_MIN_DEG) {
      issues.push(
        `median trunk lean ${result.median_trunk_lean_deg.toFixed(1)}° at seat-off ` +
        `(< ${TRUNK_LEAN_EFFICIENT_MIN_DEG}°) — limited forward weight transfer`,
      );
    }
  }
  if (
    result.median_smoothness_score !== null &&
    result.median_smoothness_score < SMOOTHNESS_SMOOTH_MIN
  ) {
    issues.push(
      `median smoothness ${result.median_smoothness_score.toFixed(2)} ` +
      `(< ${SMOOTHNESS_SMOOTH_MIN}) — hesitant / jerky rise`,
    );
  }

  const cls = result.classification;
  if (cls === "smooth" && issues.length === 0) {
    const t = result.median_trunk_lean_deg;
    const k = result.median_knee_angle_deg;
    const trunkStr = t !== null ? `${t.toFixed(0)}°` : "—";
    const kneeStr  = k !== null ? `${k.toFixed(0)}°` : "—";
    return (
      `${repSummary}: smooth movement quality. Median trunk lean ${trunkStr} ` +
      `at seat-off, knee ${kneeStr} at seat-off, no hand-use compensation detected.`
    );
  }
  return `${repSummary}: ${cls}. ${issues.join("; ")}.`;
}

// ─── Display helpers ────────────────────────────────────────────

export const STS_QUALITY_CLASSIFICATION_LABEL: Record<STSQualityClassification, string> = {
  smooth:      "Smooth",
  hesitant:    "Hesitant",
  compensated: "Compensated",
};

export const STS_QUALITY_CLASSIFICATION_TONE: Record<STSQualityClassification, string> = {
  smooth:      "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  hesitant:    "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  compensated: "bg-red-500/10 text-red-700 dark:text-red-400",
};

// ─── Upload-mode API client ─────────────────────────────────────
//
// POST /api/analyze-sts-quality accepts ONE clip + the camera-facing
// side + an optional chair seat height. Single trial, no L/R split —
// unlike SLR / AKE / MTT / FL which post one clip per side.

interface STSQualityResponseDTO {
  success: boolean;
  data: STSQualityResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

function humanizeUploadError(raw: string | null): string {
  if (!raw) return "Analysis failed. Please try again.";
  const s = raw.toLowerCase();
  if (s.includes("poor_visibility")) {
    return (
      "Patient is not clearly visible in the recording. Re-record from " +
      "the side with the full body in frame — shoulder, hip, knee, and " +
      "ankle on the camera side."
    );
  }
  if (s.includes("no_reps") || s.includes("reps not detected")) {
    return (
      "No sit-to-stand reps detected. Please re-record the patient performing " +
      "3 sit-to-stand cycles from a standard chair."
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

export async function analyzeSTSQualityUpload(
  file: File,
  side: Side,
  chairSeatHeightCm: number | null,
  onProgress?: (pct: number) => void,
): Promise<STSQualityResult> {
  const form = new FormData();
  form.append("video", file, file.name || "sts_quality.mp4");
  form.append("side", side);
  if (chairSeatHeightCm !== null && Number.isFinite(chairSeatHeightCm)) {
    form.append("chair_seat_height_cm", String(chairSeatHeightCm));
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
    const res = await authedFetch("/api/analyze-sts-quality", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `STS-quality analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as STSQualityResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
