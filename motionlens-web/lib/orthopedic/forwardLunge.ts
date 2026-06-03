// Forward Lunge (B3) — functional-movement test math, per-rep
// detection (mirrored from SLS), per-rep / per-side aggregation,
// classification, upload-mode API client.
//
// Setup:
//   Patient stands tall, hands on hips or at sides. Steps forward
//   into a lunge with the TEST leg, lowering until the back knee
//   approaches the floor (front knee ≈ 90°), holds ~1 s, then pushes
//   back to standing. 5 reps per side. Camera is LATERAL on the side
//   of the front (test) leg, full body in frame.
//
// What we measure on the FRONT (test) leg per rep:
//   • Knee flexion at bottom — inner angle at the knee between
//     (knee→hip) and (knee→ankle). Target 85–95°. We track the actual
//     inner angle (180° straight, 90° bent). The "flexion" the doctor
//     thinks about is 180° − inner_angle, but we keep the inner-angle
//     convention internally to match SLS / AKE.
//   • Knee-over-toe excursion — horizontal distance between the front
//     knee and the front ankle, signed in the body-forward direction,
//     normalised by hip-to-ankle leg length. The flag fires when the
//     ratio exceeds 0.06 (~5 cm forward on an 85 cm leg).
//   • Trunk forward lean — angle of the test-side hip→shoulder vector
//     from vertical. The flag fires above 20°.
//   • Depth variation — max − min of knee_angle_at_bottom across the
//     captured reps. Large variation flags fatigue / inconsistency.
//
// Rep detection (live + upload):
//   The patient's TEST-side hip Y drops during each descent and
//   rebounds during ascent. We treat that trajectory exactly the way
//   SLS treats the hip-midpoint Y: a peak-and-valley detector with
//   PEAK_MIN_SEPARATION_FRAMES + PEAK_MIN_DEPTH_PX gates. The pattern
//   below is intentionally bit-identical to lib/orthopedic/singleLegSquat.ts
//   so behaviour stays predictable across the multi-rep modules.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";

const VIS_THRESHOLD = 0.3;

// Sample rate for the per-frame time-series stored on the result.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// Trial budget. Five lunges at ~5-7 s each plus settle-in / standing
// time = ~45 s headroom; the trial auto-finishes on the 5th rep.
export const TARGET_REP_COUNT = 5;
export const TRIAL_TIMEOUT_SEC = 45;

// Rep-detection tuning — bit-identical to SLS so the same find_peaks
// gates land in the backend mirror.
export const PEAK_MIN_DEPTH_PX = 50;
export const PEAK_MIN_SEPARATION_FRAMES = 30;

// Knee-flexion target band at the bottom of the lunge. The inner knee
// angle (180° = straight, 90° = bent square) should land 85–95°.
export const KNEE_TARGET_MIN_DEG = 85;
export const KNEE_TARGET_MAX_DEG = 95;
// Outside this wider band, the rep is flagged "out of range".
export const KNEE_DEPTH_HARD_MIN_DEG = 70;
export const KNEE_DEPTH_HARD_MAX_DEG = 110;

// Knee-over-toe ratio (signed, forward-positive, fraction of leg
// length). Above this = knee passes meaningfully forward of the foot.
// ~0.06 ≈ 5 cm forward on an 85 cm leg.
export const KOT_FLAG_RATIO = 0.06;

// Trunk forward-lean ceiling (degrees from vertical) before the flag
// fires. 20° is the user-facing cutoff; below 15° is the ideal.
export const TRUNK_LEAN_FLAG_DEG = 20;
export const TRUNK_LEAN_IDEAL_DEG = 15;

// Depth-variation ceiling (degrees) across the 5 reps. Above this =
// fatigue / inconsistency flag.
export const DEPTH_VARIATION_FLAG_DEG = 15;

export type Side = "left" | "right";
export type LungeClassification = "good" | "borderline" | "poor";
export type Termination = "completed" | "timeout" | "stopped";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

const SIDE_INDICES = {
  left: {
    shoulder: LM.LEFT_SHOULDER,
    hip: LM.LEFT_HIP,
    knee: LM.LEFT_KNEE,
    ankle: LM.LEFT_ANKLE,
    heel: LM.LEFT_HEEL,
    foot_index: LM.LEFT_FOOT_INDEX,
  },
  right: {
    shoulder: LM.RIGHT_SHOULDER,
    hip: LM.RIGHT_HIP,
    knee: LM.RIGHT_KNEE,
    ankle: LM.RIGHT_ANKLE,
    heel: LM.RIGHT_HEEL,
    foot_index: LM.RIGHT_FOOT_INDEX,
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

// Inner angle at the knee — 180° = perfectly straight, 90° = bent
// square. We report this raw value; the doctor mentally converts to
// "flexion" (180° − inner) when reading the report.
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

/**
 * Patient body-forward direction in image x. Returns +1 if the toes
 * point in the +x direction (foot_index.x > heel.x), −1 the other
 * way. Used to give the KOT ratio a meaningful "knee forward of foot"
 * sign regardless of which way the patient happens to be facing.
 */
export function detectBodyDirection(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const heel = keypoints[idx.heel];
  const foot = keypoints[idx.foot_index];
  if (!visible(heel) || !visible(foot)) {
    // Fall back to ankle → foot_index, then ankle as the foot anchor.
    const ankle = keypoints[idx.ankle];
    if (!visible(ankle) || !visible(foot)) return null;
    const dx = foot.x - ankle.x;
    if (Math.abs(dx) < 1e-3) return null;
    return dx > 0 ? 1 : -1;
  }
  const dx = foot.x - heel.x;
  if (Math.abs(dx) < 1e-3) return null;
  return dx > 0 ? 1 : -1;
}

/**
 * Signed knee-over-toe ratio: positive = knee is forward of the foot
 * in the patient's forward direction. Normalised by hip-to-ankle leg
 * length so the value is comparable across patients without explicit
 * cm calibration. The classifier flags `> KOT_FLAG_RATIO`.
 */
export function computeKneeOverToeRatio(
  keypoints: Keypoint[], side: Side,
): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  if (!visible(hip) || !visible(knee) || !visible(ankle)) return null;

  const direction = detectBodyDirection(keypoints, side);
  if (direction === null) return null;

  const legLengthPx = Math.hypot(ankle.x - hip.x, ankle.y - hip.y);
  if (legLengthPx <= 0) return null;

  const kneeForwardPx = (knee.x - ankle.x) * direction;
  return kneeForwardPx / legLengthPx;
}

/**
 * Trunk lean from vertical, using the TEST-side hip → TEST-side
 * shoulder vector. Test-side keypoints sit nearest the camera in a
 * lateral view and are the most reliably tracked. 0° = body perfectly
 * vertical; 90° = body horizontal.
 */
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

/** Test-side hip Y in pixels — drives the rep-detector signal. */
export function computeHipY(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  if (!visible(hip)) return null;
  return hip.y;
}

/** Pre-record gate: the whole test-side leg chain must be trackable
 *  plus the test-side shoulder (for trunk lean). */
export function isTestSideTrackable(keypoints: Keypoint[], side: Side): boolean {
  const idx = SIDE_INDICES[side];
  return (
    visible(keypoints[idx.shoulder]) &&
    visible(keypoints[idx.hip]) &&
    visible(keypoints[idx.knee]) &&
    visible(keypoints[idx.ankle])
  );
}

// ─── Rep detector (peak-and-valley on TEST-side hip Y) ──────────
// Bit-identical pattern to lib/orthopedic/singleLegSquat.ts so the
// SLS reviewer's mental model carries over. Only the input signal is
// the TEST-side hip Y instead of the hip-midpoint Y (in lateral view
// the contralateral hip is occluded, so single-side is the reliable
// channel).

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

/** Returns true iff the PREVIOUS frame was a confirmed lunge-bottom
 *  peak — i.e. the patient just transitioned from descending
 *  (Y increasing) to ascending (Y decreasing). */
export function detectPeak(
  state: PeakState,
  hipY: number | null,
  frameIdx: number,
): boolean {
  if (hipY === null) {
    state.prevPrevY = state.prevY;
    state.prevY = null;
    state.prevFrameIdx = frameIdx;
    return false;
  }
  if (state.baselineY === null) {
    state.baselineY = hipY;
    state.prevY = hipY;
    state.prevFrameIdx = frameIdx;
    return false;
  }

  let peakDetected = false;
  if (state.prevPrevY !== null && state.prevY !== null) {
    const ascending = state.prevY > state.prevPrevY;       // Y went up (going down in space)
    const turning   = hipY < state.prevY;                  // Y now going down (going up in space)
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
  state.prevY = hipY;
  state.prevFrameIdx = frameIdx;
  return peakDetected;
}

// ─── Per-rep + per-side aggregation ─────────────────────────────

export interface RepMetrics {
  rep_index: number;                  // 1..5
  t_ms: number;                       // when the bottom of the lunge was reached
  knee_angle_at_bottom_deg: number | null;  // inner angle (90° target, 180° straight)
  knee_over_toe_ratio: number | null;       // signed, forward-positive, /leg-length
  trunk_lean_deg: number | null;            // 0 = vertical, larger = more lean
}

export interface FrameSample {
  t_ms: number;
  hip_y: number | null;
  knee_angle_deg: number | null;
  knee_over_toe_ratio: number | null;
  trunk_lean_deg: number | null;
}

export interface ForwardLungeSideResult {
  side_tested: Side;
  reps: RepMetrics[];
  /** Index in `reps` of the worst rep (largest composite deviation). */
  worst_rep_index: number | null;
  /** Inner knee angle at the worst rep's bottom — closest summary of
   *  "did the patient hit ~90°". */
  worst_rep_knee_angle_deg: number;
  /** Max knee-over-toe ratio across captured reps. > KOT_FLAG_RATIO
   *  triggers the ankle/quad-dominance flag. */
  worst_rep_kot_ratio: number;
  /** Max trunk lean across captured reps. > TRUNK_LEAN_FLAG_DEG
   *  triggers the posterior-chain compensation flag. */
  worst_rep_trunk_lean_deg: number;
  /** Mean knee_angle_at_bottom across captured reps. */
  mean_knee_angle_deg: number;
  /** max − min of knee_angle_at_bottom across captured reps. >
   *  DEPTH_VARIATION_FLAG_DEG triggers the fatigue flag. */
  depth_variation_deg: number;
  /** Per-flag breakdown so the report can show why a side scored
   *  "borderline" or "poor". */
  depth_out_of_band: boolean;   // any rep outside KNEE_DEPTH_HARD_*
  kot_flagged: boolean;
  trunk_lean_flagged: boolean;
  fatigue_flagged: boolean;
  classification: LungeClassification;
  duration_seconds: number;
  termination: Termination;
  /** True if fewer than TARGET_REP_COUNT reps were captured. */
  incomplete: boolean;
  /** 10 Hz per-frame time-series. */
  samples: FrameSample[];
  /** Per-frame keypoints (PDF Section 2 (a) compliance). */
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  /** JPEG data-URL of the worst-rep frame (skeleton-overlaid). */
  worst_rep_screenshot_data_url: string | null;
}

export interface ForwardLungeFullResult {
  left:  ForwardLungeSideResult | null;
  right: ForwardLungeSideResult | null;
}

// ─── Worst-rep composite scoring ────────────────────────────────

/**
 * How far this rep is from "ideal". Higher = worse. Weights tuned so
 * each metric contributes roughly equally on its threshold value:
 *   • 1° off the knee target ≈ 1 point
 *   • 0.005 KOT ratio        ≈ 1 point  (so 0.06 threshold ≈ 12 pts)
 *   • 1° of trunk lean       ≈ 1 point  (20° threshold ≈ 20 pts)
 */
export function repCompositeScore(r: RepMetrics): number {
  let s = 0;
  if (r.knee_angle_at_bottom_deg !== null) {
    s += Math.abs(r.knee_angle_at_bottom_deg - 90);
  }
  if (r.knee_over_toe_ratio !== null) {
    s += Math.max(0, r.knee_over_toe_ratio) * 200;
  }
  if (r.trunk_lean_deg !== null) {
    s += r.trunk_lean_deg;
  }
  return s;
}

// ─── Classification ─────────────────────────────────────────────

export function classifyLunge(
  worstKneeDeg: number,
  worstKotRatio: number,
  worstTrunkDeg: number,
  depthVariationDeg: number,
  incomplete: boolean,
): LungeClassification {
  const depthOOB =
    worstKneeDeg < KNEE_DEPTH_HARD_MIN_DEG ||
    worstKneeDeg > KNEE_DEPTH_HARD_MAX_DEG;
  const kotFlag   = worstKotRatio  > KOT_FLAG_RATIO;
  const trunkFlag = worstTrunkDeg  > TRUNK_LEAN_FLAG_DEG;
  const fatigue   = depthVariationDeg > DEPTH_VARIATION_FLAG_DEG;
  const flagCount =
    (depthOOB  ? 1 : 0) +
    (kotFlag   ? 1 : 0) +
    (trunkFlag ? 1 : 0) +
    (fatigue   ? 1 : 0);

  // Severe single flag → "poor" on its own.
  if (worstTrunkDeg > 30) return "poor";
  if (worstKotRatio > 0.12) return "poor";

  if (incomplete) return "borderline";
  if (flagCount === 0) return "good";
  if (flagCount >= 3)  return "poor";
  return "borderline";
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
): ForwardLungeSideResult {
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);
  const incomplete = reps.length < TARGET_REP_COUNT;

  // Pick the rep with the highest composite deviation.
  let worstIdx: number | null = null;
  let worstScore = -Infinity;
  for (let i = 0; i < reps.length; i++) {
    const s = repCompositeScore(reps[i]);
    if (s > worstScore) {
      worstScore = s;
      worstIdx = i;
    }
  }

  const kneeAt = (i: number | null): number => {
    if (i === null) return 0;
    const v = reps[i].knee_angle_at_bottom_deg;
    return v === null ? 0 : v;
  };

  const kneeVals  = reps.map((r) => r.knee_angle_at_bottom_deg).filter((v): v is number => v !== null);
  const kotVals   = reps.map((r) => r.knee_over_toe_ratio).filter((v): v is number => v !== null);
  const trunkVals = reps.map((r) => r.trunk_lean_deg).filter((v): v is number => v !== null);

  const meanKnee = kneeVals.length === 0
    ? 0
    : kneeVals.reduce((a, b) => a + b, 0) / kneeVals.length;
  const depthVariation = kneeVals.length === 0
    ? 0
    : Math.max(...kneeVals) - Math.min(...kneeVals);
  const worstKnee  = kneeAt(worstIdx);
  const worstKot   = kotVals.length   === 0 ? 0 : Math.max(...kotVals);
  const worstTrunk = trunkVals.length === 0 ? 0 : Math.max(...trunkVals);

  const depthOOB =
    kneeVals.length === 0
      ? true
      : kneeVals.some(
          (v) =>
            v < KNEE_DEPTH_HARD_MIN_DEG ||
            v > KNEE_DEPTH_HARD_MAX_DEG,
        );
  const kotFlag   = worstKot  > KOT_FLAG_RATIO;
  const trunkFlag = worstTrunk > TRUNK_LEAN_FLAG_DEG;
  const fatigue   = depthVariation > DEPTH_VARIATION_FLAG_DEG;

  const classification = classifyLunge(
    kneeVals.length === 0 ? 0 : worstKnee || meanKnee,
    worstKot,
    worstTrunk,
    depthVariation,
    incomplete,
  );

  return {
    side_tested: side,
    reps,
    worst_rep_index: worstIdx,
    worst_rep_knee_angle_deg: worstKnee,
    worst_rep_kot_ratio: worstKot,
    worst_rep_trunk_lean_deg: worstTrunk,
    mean_knee_angle_deg: meanKnee,
    depth_variation_deg: depthVariation,
    depth_out_of_band: depthOOB,
    kot_flagged: kotFlag,
    trunk_lean_flagged: trunkFlag,
    fatigue_flagged: fatigue,
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

export function buildInterpretation(result: ForwardLungeFullResult): string {
  const parts: string[] = [];

  for (const side of ["left", "right"] as const) {
    const r = side === "left" ? result.left : result.right;
    if (!r) continue;
    const sideLabel = side === "left" ? "Left-leg lunge" : "Right-leg lunge";
    const repSummary = r.incomplete
      ? `${r.reps.length} of ${TARGET_REP_COUNT} reps captured`
      : `${r.reps.length} reps captured`;

    const issues: string[] = [];
    if (r.depth_out_of_band) {
      issues.push(
        `depth out of band (worst rep ${r.worst_rep_knee_angle_deg.toFixed(0)}°, ` +
        `target ${KNEE_TARGET_MIN_DEG}–${KNEE_TARGET_MAX_DEG}°)`,
      );
    }
    if (r.kot_flagged) {
      issues.push(
        `knee passed forward of the foot — worst rep ${(r.worst_rep_kot_ratio * 100).toFixed(1)}% ` +
        `of leg length (flag at ${(KOT_FLAG_RATIO * 100).toFixed(0)}%) — ankle/quadriceps dominance`,
      );
    }
    if (r.trunk_lean_flagged) {
      issues.push(
        `trunk forward lean ${r.worst_rep_trunk_lean_deg.toFixed(1)}° ` +
        `(flag at ${TRUNK_LEAN_FLAG_DEG}°) — posterior-chain compensation`,
      );
    }
    if (r.fatigue_flagged) {
      issues.push(
        `depth varied ${r.depth_variation_deg.toFixed(1)}° across reps ` +
        `(flag at ${DEPTH_VARIATION_FLAG_DEG}°) — possible fatigue / inconsistency`,
      );
    }

    if (issues.length === 0) {
      parts.push(
        `${sideLabel} (${repSummary}): good. Mean knee depth ${r.mean_knee_angle_deg.toFixed(0)}°, ` +
        `knee never passed forward of the foot, trunk stayed under ${TRUNK_LEAN_FLAG_DEG}° throughout.`,
      );
    } else {
      const cls = r.classification === "poor" ? "poor" : "borderline";
      parts.push(
        `${sideLabel} (${repSummary}): ${cls}. ${issues.join("; ")}.`,
      );
    }
  }

  if (parts.length === 0) {
    return "No completed forward-lunge trials to interpret.";
  }
  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const LUNGE_CLASSIFICATION_LABEL: Record<LungeClassification, string> = {
  good:       "Good",
  borderline: "Borderline",
  poor:       "Poor",
};

export const LUNGE_CLASSIFICATION_TONE: Record<LungeClassification, string> = {
  good:       "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  borderline: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  poor:       "bg-red-500/10 text-red-700 dark:text-red-400",
};

// ─── Upload-mode API client ─────────────────────────────────────
//
// POST /api/analyze-forward-lunge accepts ONE side per call. The
// frontend uploads left + right clips sequentially (not in parallel —
// same cold-worker 502 mitigation as SLR / AKE / MTT) and assembles
// ForwardLungeFullResult { left, right } client-side. Backend math
// + classification cutoffs mirror this file exactly, so the returned
// ForwardLungeSideResult slots straight into ForwardLungeReport
// without translation.

interface FLResponseDTO {
  success: boolean;
  data: ForwardLungeSideResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

function humanizeUploadError(raw: string | null): string {
  if (!raw) return "Analysis failed. Please try again.";
  const s = raw.toLowerCase();
  if (s.includes("poor_visibility")) {
    return (
      "Patient's leg is not clearly visible in the recording. Re-record from " +
      "the side with the full body in frame, perpendicular to the camera."
    );
  }
  if (s.includes("no_reps") || s.includes("reps not detected")) {
    return (
      "No forward-lunge reps detected. Please re-record the patient performing " +
      "5 lunges on the test leg from the side."
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

export async function analyzeForwardLungeUpload(
  file: File,
  side: Side,
  onProgress?: (pct: number) => void,
): Promise<ForwardLungeSideResult> {
  const form = new FormData();
  form.append("video", file, file.name || "forward_lunge.mp4");
  form.append("side", side);

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
    const res = await authedFetch("/api/analyze-forward-lunge", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `Forward-lunge analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as FLResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
