// Single-Leg Squat / Step-Down test — math, conventions, rep detector,
// classification, composite risk score.
//
// Spec: MotionLens Test Battery v1.0, Test B1 (pp. 9-10).
// Pose model: MoveNet 17-keypoint (LM indices in @/lib/pose/landmarks).
//
// Sign conventions (PDF Appendix B):
//   pelvic tilt:  positive = LEFT side down (line of right→left hip
//                 vs horizontal).
//   trunk lean:   frontal view, positive = lean to patient's RIGHT
//                 (hip-midpoint → shoulder-midpoint vs vertical).
//   KFPPA:        magnitude (always >= 0). Sign convention is implicit
//                 ("positive = valgus") and applied at classification.
//
// Per-rep metrics are computed at the bottom of each squat (peak hip-
// midpoint Y in image-y-down coords). Five reps are detected with a
// peak-and-valley rule; if fewer reps complete within 30 s, the trial
// ends with whatever was captured (marked incomplete).

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

const VIS_THRESHOLD = 0.3;

// Sample rate for the per-frame time-series.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// Trial duration ceiling — if 5 reps aren't detected in this window,
// we stop with whatever reps did complete.
export const TARGET_REP_COUNT = 5;
export const TRIAL_TIMEOUT_SEC = 30;

// Rep-detection tuning (see findPeak() below). All values are in
// pixel space at the camera's native resolution (typically 720p).
export const PEAK_MIN_DEPTH_PX = 50;
export const PEAK_MIN_SEPARATION_FRAMES = 30;

// Camera-squareness validation: the line between the patient's two
// shoulders must be within this many degrees of horizontal before we
// allow the trial to start. (PDF: even 15° rotation distorts KFPPA.)
export const SQUARENESS_TOLERANCE_DEG = 5;

// PDF Test B1 cutoffs — DO NOT change without spec update.
export const KFPPA_GOOD_MAX_DEG = 10;        // <10° = good knee tracking
export const KFPPA_BORDERLINE_MAX_DEG = 15;  // 10-15° = borderline; >15° = valgus
export const PELVIC_DROP_THRESHOLD_DEG = 5;  // >5° = hip abductor insufficiency
export const TRUNK_LEAN_THRESHOLD_DEG = 7;   // >7° = compensatory lateral lean
export const ASYMMETRY_THRESHOLD_DEG = 5;    // L-R KFPPA delta >5° = targeted

export type Side = "left" | "right";
export type KneeClassification = "good" | "borderline" | "valgus";
export type RiskScore = "low" | "moderate" | "high";
export type Termination = "completed" | "timeout" | "stopped" | "rotated_out";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

const SIDE_INDICES = {
  left:  { hip: LM.LEFT_HIP,  knee: LM.LEFT_KNEE,  ankle: LM.LEFT_ANKLE  },
  right: { hip: LM.RIGHT_HIP, knee: LM.RIGHT_KNEE, ankle: LM.RIGHT_ANKLE },
} as const;

// ─── Per-frame computations ─────────────────────────────────────

// KFPPA = angle between the (hip→knee) vector and the (knee→ankle)
// vector, computed in 2D image space. 0° = perfectly straight leg in
// the frontal plane; magnitude grows as the knee falls medially or
// laterally relative to the hip-ankle line. Returned as a magnitude
// (always >= 0) — the spec's "positive = valgus" sign convention is
// applied implicitly because clinical KFPPA classification compares
// magnitude against the 10° / 15° cutoffs.
export function computeKFPPA(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  if (![hip, knee, ankle].every(visible)) return null;

  const v1x = knee.x - hip.x;
  const v1y = knee.y - hip.y;
  const v2x = ankle.x - knee.x;
  const v2y = ankle.y - knee.y;
  const m1 = Math.hypot(v1x, v1y);
  const m2 = Math.hypot(v2x, v2y);
  if (m1 === 0 || m2 === 0) return null;

  const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / (m1 * m2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

// Pelvic tilt — same convention as Trendelenburg / posture (Appendix
// B, positive = left side down). Vector points right→left so the
// arithmetic is left.y - right.y.
export function computePelvicTilt(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!visible(lHip) || !visible(rHip)) return null;
  return (Math.atan2(lHip.y - rHip.y, lHip.x - rHip.x) * 180) / Math.PI;
}

// Trunk lean (frontal): hip-midpoint → shoulder-midpoint line vs
// vertical. Positive = lean to patient's RIGHT (camera's left in
// mirror frontal view).
export function computeTrunkLean(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  const lSh  = keypoints[LM.LEFT_SHOULDER];
  const rSh  = keypoints[LM.RIGHT_SHOULDER];
  if (![lHip, rHip, lSh, rSh].every(visible)) return null;
  const hipMidX = (lHip.x + rHip.x) / 2;
  const hipMidY = (lHip.y + rHip.y) / 2;
  const shMidX  = (lSh.x  + rSh.x ) / 2;
  const shMidY  = (lSh.y  + rSh.y ) / 2;
  const vx = shMidX - hipMidX;
  const vy = shMidY - hipMidY;
  const mag = Math.abs((Math.atan2(vx, vy) * 180) / Math.PI);
  return (vx < 0 ? 1 : -1) * mag;
}

// Hip-midpoint Y — used by the rep detector. Returns null when hip
// keypoints aren't both visible.
export function computeHipMidY(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!visible(lHip) || !visible(rHip)) return null;
  return (lHip.y + rHip.y) / 2;
}

// Stance-side leg length in pixels (hip → ankle). Used to normalise
// squat depth so the metric is comparable across patient heights.
export function computeLegLengthPx(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const ankle = keypoints[idx.ankle];
  if (!visible(hip) || !visible(ankle)) return null;
  return Math.hypot(ankle.x - hip.x, ankle.y - hip.y);
}

// Camera-squareness check. Returns the LINE angle (orientation-
// agnostic) of the shoulder line from horizontal in degrees, in the
// range [-90°, 90°]. 0° = perfectly level; magnitudes near 0 are
// good. We normalise because MoveNet's LEFT_SHOULDER lands on the
// image-right side of an un-mirrored selfie feed, so the directed
// L→R vector points along negative-X — its raw atan2 angle is ≈180°
// for a level patient. Treating the shoulders as a LINE (no
// direction) makes the value match the spec convention.
export function computeShoulderHorizontalDeg(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!visible(lSh) || !visible(rSh)) return null;
  let deg = (Math.atan2(rSh.y - lSh.y, rSh.x - lSh.x) * 180) / Math.PI;
  if (deg > 90) deg -= 180;
  if (deg < -90) deg += 180;
  return deg;
}

export function isCameraSquare(angleDeg: number | null): boolean {
  if (angleDeg === null) return false;
  return Math.abs(angleDeg) <= SQUARENESS_TOLERANCE_DEG;
}

// ─── Classification ──────────────────────────────────────────────

export function classifyKFPPA(maxDeg: number): KneeClassification {
  if (maxDeg < KFPPA_GOOD_MAX_DEG) return "good";
  if (maxDeg <= KFPPA_BORDERLINE_MAX_DEG) return "borderline";
  return "valgus";
}

// Composite injury-risk score per side. PDF defines the cutoffs but
// not the combination rule — we use a simple flag-counting scheme:
//   high     ← KFPPA >15° (always elevated risk on its own) OR ≥2 flags
//   moderate ← exactly 1 flag (KFPPA 10-15°, drop >5°, lean >7°)
//   low      ← all metrics within the normal band
export function compositeRisk(
  worstKFPPADeg: number,
  meanPelvicDropDeg: number,
  meanTrunkLeanDeg: number,
): RiskScore {
  if (worstKFPPADeg > KFPPA_BORDERLINE_MAX_DEG) return "high";
  let flags = 0;
  if (worstKFPPADeg >= KFPPA_GOOD_MAX_DEG) flags += 1;
  if (Math.abs(meanPelvicDropDeg) > PELVIC_DROP_THRESHOLD_DEG) flags += 1;
  if (Math.abs(meanTrunkLeanDeg)  > TRUNK_LEAN_THRESHOLD_DEG)  flags += 1;
  if (flags >= 2) return "high";
  if (flags === 1) return "moderate";
  return "low";
}

// ─── Per-rep + per-side aggregation ─────────────────────────────

export interface RepMetrics {
  rep_index: number;        // 1..5
  t_ms: number;             // when the bottom of the squat was reached
  kfppa_deg: number | null; // peak frontal-plane projection at bottom
  pelvic_drop_deg: number | null;
  trunk_lean_deg: number | null;
  depth_pct: number | null; // hip-mid drop / leg length, % at bottom
}

export interface FrameSample {
  t_ms: number;
  hip_mid_y: number | null;
  kfppa_deg: number | null;
  pelvic_drop_deg: number | null;
  trunk_lean_deg: number | null;
}

export interface SingleLegSquatSideResult {
  side_tested: Side;
  reps: RepMetrics[];
  // Aggregates across the captured reps (worst = highest KFPPA).
  worst_rep_index: number | null;
  worst_kfppa_deg: number;
  mean_pelvic_drop_deg: number;
  mean_trunk_lean_deg: number;
  mean_depth_pct: number;
  classification: KneeClassification;
  risk_score: RiskScore;
  duration_seconds: number;
  termination: Termination;
  /** True if fewer than TARGET_REP_COUNT reps were captured. */
  incomplete: boolean;
  /** Per-frame time-series. */
  samples: FrameSample[];
  /** Per-frame keypoints (PDF Section 2 (a) compliance). */
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  /** JPEG data-URL of the worst-rep frame (skeleton-overlaid). */
  worst_rep_screenshot_data_url: string | null;
}

export interface SingleLegSquatFullResult {
  left:  SingleLegSquatSideResult | null;
  right: SingleLegSquatSideResult | null;
}

// ─── Live rep detector (peak-and-valley on hip-midpoint Y) ───────

interface PeakState {
  baselineY: number | null;
  prevY: number | null;
  prevPrevY: number | null;
  prevFrameIdx: number;
  lastPeakFrame: number;
  recentMaxY: number;       // running max since last "going up" turn
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

// Returns true if the *previous* frame was a confirmed squat-bottom
// peak — i.e. we just saw the patient transition from descending
// (Y increasing) to ascending (Y decreasing). The caller uses this
// to commit a RepMetrics row using the previous frame's data.
export function detectPeak(
  state: PeakState,
  hipMidY: number | null,
  frameIdx: number,
): boolean {
  if (hipMidY === null) {
    state.prevPrevY = state.prevY;
    state.prevY = null;
    state.prevFrameIdx = frameIdx;
    return false;
  }

  // First valid sample becomes the baseline.
  if (state.baselineY === null) {
    state.baselineY = hipMidY;
    state.prevY = hipMidY;
    state.prevFrameIdx = frameIdx;
    return false;
  }

  let peakDetected = false;
  // Need three consecutive valid samples to evaluate direction change.
  if (state.prevPrevY !== null && state.prevY !== null) {
    const ascending = state.prevY > state.prevPrevY;       // Y went up (going down in space)
    const turning   = hipMidY < state.prevY;               // Y now going down (going up in space)
    const sinceLast = frameIdx - state.lastPeakFrame;
    const depth     = state.prevY - state.baselineY;       // Y above baseline

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
  state.prevY = hipMidY;
  state.prevFrameIdx = frameIdx;
  return peakDetected;
}

// ─── Per-side aggregator ────────────────────────────────────────

export function summarizeSide(
  side: Side,
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  reps: RepMetrics[],
  samples: FrameSample[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  worstRepScreenshotDataUrl: string | null,
): SingleLegSquatSideResult {
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);
  const incomplete = reps.length < TARGET_REP_COUNT;

  let worstIdx: number | null = null;
  let worstKFPPA = 0;
  for (let i = 0; i < reps.length; i++) {
    const k = reps[i].kfppa_deg;
    if (k !== null && k > worstKFPPA) {
      worstKFPPA = k;
      worstIdx = i;
    }
  }

  const mean = (xs: number[]) =>
    xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;

  const drops = reps.map((r) => r.pelvic_drop_deg).filter((v): v is number => v !== null);
  const leans = reps.map((r) => r.trunk_lean_deg).filter((v): v is number => v !== null);
  const depths = reps.map((r) => r.depth_pct).filter((v): v is number => v !== null);

  const meanDrop = mean(drops);
  const meanLean = mean(leans);

  return {
    side_tested: side,
    reps,
    worst_rep_index: worstIdx,
    worst_kfppa_deg: worstKFPPA,
    mean_pelvic_drop_deg: meanDrop,
    mean_trunk_lean_deg: meanLean,
    mean_depth_pct: mean(depths),
    classification: classifyKFPPA(worstKFPPA),
    risk_score: compositeRisk(worstKFPPA, meanDrop, meanLean),
    duration_seconds: duration,
    termination,
    incomplete,
    samples,
    keypoints,
    worst_rep_screenshot_data_url: worstRepScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: SingleLegSquatFullResult): string {
  const parts: string[] = [];

  for (const side of ["left", "right"] as const) {
    const r = side === "left" ? result.left : result.right;
    if (!r) continue;
    const sideLabel = side === "left" ? "Left-leg squat" : "Right-leg squat";
    const repSummary = r.incomplete
      ? `${r.reps.length} of ${TARGET_REP_COUNT} reps completed`
      : `${r.reps.length} reps completed`;

    if (r.classification === "good") {
      parts.push(
        `${sideLabel} (${repSummary}): good knee tracking — worst KFPPA ` +
        `${r.worst_kfppa_deg.toFixed(1)}° (< ${KFPPA_GOOD_MAX_DEG}°).`,
      );
    } else if (r.classification === "borderline") {
      parts.push(
        `${sideLabel} (${repSummary}): borderline knee tracking — worst KFPPA ` +
        `${r.worst_kfppa_deg.toFixed(1)}° (${KFPPA_GOOD_MAX_DEG}–${KFPPA_BORDERLINE_MAX_DEG}°). ` +
        `Monitor and reassess.`,
      );
    } else {
      parts.push(
        `${sideLabel} (${repSummary}): dynamic valgus — worst KFPPA ` +
        `${r.worst_kfppa_deg.toFixed(1)}° (> ${KFPPA_BORDERLINE_MAX_DEG}°), ` +
        `elevated ACL/PFP risk.`,
      );
    }

    if (Math.abs(r.mean_pelvic_drop_deg) > PELVIC_DROP_THRESHOLD_DEG) {
      parts.push(
        `${sideLabel}: mean pelvic drop ${Math.abs(r.mean_pelvic_drop_deg).toFixed(1)}° ` +
        `(> ${PELVIC_DROP_THRESHOLD_DEG}°) — hip abductor insufficiency on the stance side.`,
      );
    }
    if (Math.abs(r.mean_trunk_lean_deg) > TRUNK_LEAN_THRESHOLD_DEG) {
      parts.push(
        `${sideLabel}: mean trunk lateral lean ${Math.abs(r.mean_trunk_lean_deg).toFixed(1)}° ` +
        `(> ${TRUNK_LEAN_THRESHOLD_DEG}°) — compensatory lean.`,
      );
    }
  }

  // L–R asymmetry callout (PDF: >5° KFPPA delta = targeted intervention).
  if (result.left && result.right) {
    const delta = Math.abs(result.left.worst_kfppa_deg - result.right.worst_kfppa_deg);
    if (delta > ASYMMETRY_THRESHOLD_DEG) {
      const worse = result.left.worst_kfppa_deg > result.right.worst_kfppa_deg ? "left" : "right";
      parts.push(
        `L–R asymmetry of ${delta.toFixed(1)}° in worst KFPPA (> ${ASYMMETRY_THRESHOLD_DEG}°) — ` +
        `targeted intervention indicated on the ${worse} side.`,
      );
    }
  }

  if (parts.length === 0) {
    return "No completed squat trials to interpret.";
  }
  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const KNEE_CLASSIFICATION_LABEL: Record<KneeClassification, string> = {
  good:       "Good",
  borderline: "Borderline",
  valgus:     "Valgus",
};

export const KNEE_CLASSIFICATION_TONE: Record<KneeClassification, string> = {
  good:       "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  borderline: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  valgus:     "bg-red-500/10 text-red-700 dark:text-red-400",
};

export const RISK_LABEL: Record<RiskScore, string> = {
  low:      "Low risk",
  moderate: "Moderate risk",
  high:     "High risk",
};

export const RISK_TONE: Record<RiskScore, string> = {
  low:      "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  moderate: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  high:     "bg-red-500/10 text-red-700 dark:text-red-400",
};
