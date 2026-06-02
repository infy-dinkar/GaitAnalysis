// Active Knee Extension (AKE) test — math, conventions, validity gate,
// per-side aggregation, classification, upload-mode API client.
//
// Setup:
//   Patient supine (face up). Camera on the side (lateral view). The
//   patient raises the test-side thigh to ~90° from the bed (thigh
//   vertical) and HOLDS that position; the hip stays fixed at 90° for
//   the whole trial. From that hold, the patient slowly extends the
//   knee — straightening the leg as far as the hamstrings allow.
//   Tight hamstrings stop the extension before a fully-straight (180°)
//   knee; the maximum knee angle reached is the result.
//
// Sign convention (image space, y-down):
//   • body baseline vector  = hip-midpoint → shoulder-midpoint
//                             (points toward the patient's head along
//                             the bed when they're supine)
//   • thigh vector          = test-side hip → test-side knee
//   • shin vector           = test-side knee → test-side ankle
//
//   • hip_flex_angle_deg    = 180° − inner_angle(body_baseline, thigh)
//                             — 0°  = leg flat alongside the body
//                             — 90° = thigh vertical (correct AKE hold)
//                             — 180° = thigh pulled onto the chest
//   • knee_angle_deg        = inner_angle(knee→hip, knee→ankle)
//                             — 90°  = knee bent (start position)
//                             — 180° = knee fully straight
//
// A frame counts as a valid extension measurement only when the thigh
// is held at ~90°, i.e. hip_flex_angle_deg is within
// [THIGH_HELD_MIN_DEG, THIGH_HELD_MAX_DEG]. Frames outside this band
// are excluded from the max-knee calculation.
//
// The backend mirrors this math exactly in
// engines/orthopedic/ake_engine.py so live and upload return identical
// numbers.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";

// ─── Tuning constants ───────────────────────────────────────────
const VIS_THRESHOLD = 0.3;

// Sample rate for the per-frame time-series stored on the result.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// One continuous trial per side — no rep counting. The patient holds
// the thigh up and slowly extends the knee within this window; we
// keep the maximum valid knee angle reached across the whole trial.
export const TRIAL_DURATION_SEC = 15;

// Thigh-stability window (degrees of hip flexion). Valid AKE
// measurement frames require the thigh to sit within this band of
// ~90°. ±15° tolerates pose-noise wobble + the natural slight drift
// patients show holding the leg up. Frames outside this band are
// excluded from the max-knee calculation.
export const THIGH_HELD_MIN_DEG = 75;
export const THIGH_HELD_MAX_DEG = 105;

// Lower bound for "we actually saw an extension attempt". The starting
// knee bend is typically ~90°; if the captured max never exceeds this
// the patient never tried to extend (or the engine never tracked any
// valid frames).
export const MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG = 95;

// Classification cutoffs on the extension DEFICIT (180° − max knee).
//   ≤ 10°  = normal hamstring flexibility
//   11–20° = mild tightness
//   21–35° = moderate tightness
//   >  35° = severe tightness
export const NORMAL_MAX_DEFICIT_DEG   = 10;
export const MILD_MAX_DEFICIT_DEG     = 20;
export const MODERATE_MAX_DEFICIT_DEG = 35;

export type Side = "left" | "right";
export type AKEClassification = "normal" | "mild" | "moderate" | "severe";
export type Termination = "completed" | "timeout" | "stopped";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

const SIDE_INDICES = {
  left:  { hip: LM.LEFT_HIP,  knee: LM.LEFT_KNEE,  ankle: LM.LEFT_ANKLE  },
  right: { hip: LM.RIGHT_HIP, knee: LM.RIGHT_KNEE, ankle: LM.RIGHT_ANKLE },
} as const;

// ─── Per-frame computations ─────────────────────────────────────

// Angle (0..180°) between two 2D vectors.
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

// hip_flex_angle = 180° − angle(torso, thigh)
// 0° at rest (leg flat in line with body), 90° thigh vertical, >90°
// thigh past vertical (toward chest).
export function computeHipFlexAngle(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  const lSh  = keypoints[LM.LEFT_SHOULDER];
  const rSh  = keypoints[LM.RIGHT_SHOULDER];
  if (
    !visible(hip) || !visible(knee) ||
    !visible(lHip) || !visible(rHip) ||
    !visible(lSh) || !visible(rSh)
  ) return null;

  const hipMidX = (lHip.x + rHip.x) / 2;
  const hipMidY = (lHip.y + rHip.y) / 2;
  const shMidX  = (lSh.x  + rSh.x ) / 2;
  const shMidY  = (lSh.y  + rSh.y ) / 2;

  const torsoVx = shMidX - hipMidX;
  const torsoVy = shMidY - hipMidY;
  const thighVx = knee.x - hip.x;
  const thighVy = knee.y - hip.y;

  const inner = vectorAngleDeg(torsoVx, torsoVy, thighVx, thighVy);
  if (inner === null) return null;
  return 180 - inner;
}

// Inner angle at the knee — 180° = perfectly straight, 90° = bent.
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

// Thigh held at ~90° from the bed.
export function isThighHeld(hipFlexAngleDeg: number | null): boolean {
  if (hipFlexAngleDeg === null) return false;
  return (
    hipFlexAngleDeg >= THIGH_HELD_MIN_DEG &&
    hipFlexAngleDeg <= THIGH_HELD_MAX_DEG
  );
}

// Pre-record gate: all test-side leg landmarks (hip + knee + ankle)
// AND both hips + both shoulders (for the body baseline) visible.
export function isTestSideTrackable(keypoints: Keypoint[], side: Side): boolean {
  const idx = SIDE_INDICES[side];
  return (
    visible(keypoints[idx.hip])  &&
    visible(keypoints[idx.knee]) &&
    visible(keypoints[idx.ankle]) &&
    visible(keypoints[LM.LEFT_HIP])  &&
    visible(keypoints[LM.RIGHT_HIP]) &&
    visible(keypoints[LM.LEFT_SHOULDER])  &&
    visible(keypoints[LM.RIGHT_SHOULDER])
  );
}

// ─── Classification ──────────────────────────────────────────────

export function classifyAKE(deficitDeg: number): AKEClassification {
  if (deficitDeg <= NORMAL_MAX_DEFICIT_DEG)   return "normal";
  if (deficitDeg <= MILD_MAX_DEFICIT_DEG)     return "mild";
  if (deficitDeg <= MODERATE_MAX_DEFICIT_DEG) return "moderate";
  return "severe";
}

// ─── Per-trial aggregation ──────────────────────────────────────

export interface FrameSample {
  t_ms: number;
  knee_angle_deg: number | null;
  hip_flex_angle_deg: number | null;
  thigh_held: boolean;
}

export interface AKESideResult {
  side_tested: Side;
  /** Highest knee angle reached while the thigh was held at ~90°.
   *  0 if no frame qualified. 180° = perfectly straight. */
  max_knee_angle_deg: number;
  /** Extension deficit = 180° − max_knee_angle_deg. Bigger = tighter
   *  hamstrings. 0 = perfectly straight. */
  deficit_deg: number;
  /** Frame index (within `samples`) at which max_knee_angle_deg was
   *  reached. Null when no valid frame qualified. */
  max_knee_sample_index: number | null;
  /** Hip flexion angle (thigh-from-body) at the moment of peak knee
   *  extension. Surfaced so the report can show the operator how
   *  stable the thigh was at the captured peak. */
  hip_flex_angle_at_peak_deg: number | null;
  classification: AKEClassification;
  duration_seconds: number;
  termination: Termination;
  /** Fraction of trial frames where the thigh passed the held-at-90°
   *  gate. Low values mean the patient couldn't hold the hip stable. */
  thigh_held_fraction: number;
  /** Per-frame time-series. */
  samples: FrameSample[];
  /** Per-frame keypoints (PDF Section 2 (a) compliance). */
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  /** JPEG data-URL of the peak-extension frame (skeleton-overlaid). */
  peak_screenshot_data_url: string | null;
}

export interface AKEFullResult {
  left:  AKESideResult | null;
  right: AKESideResult | null;
}

export function summarizeSide(
  side: Side,
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  samples: FrameSample[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  peakScreenshotDataUrl: string | null,
): AKESideResult {
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);

  let maxKnee = 0;
  let maxIdx: number | null = null;
  let hipAtPeak: number | null = null;
  let heldCount = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.thigh_held) heldCount += 1;
    if (
      s.thigh_held &&
      s.knee_angle_deg !== null &&
      s.knee_angle_deg > maxKnee
    ) {
      maxKnee = s.knee_angle_deg;
      maxIdx = i;
      hipAtPeak = s.hip_flex_angle_deg;
    }
  }
  const thighHeldFraction =
    samples.length === 0 ? 0 : heldCount / samples.length;

  const deficit = maxKnee > 0 ? Math.max(0, 180 - maxKnee) : 180;

  return {
    side_tested: side,
    max_knee_angle_deg: maxKnee,
    deficit_deg: deficit,
    max_knee_sample_index: maxIdx,
    hip_flex_angle_at_peak_deg: hipAtPeak,
    classification: classifyAKE(deficit),
    duration_seconds: duration,
    termination,
    thigh_held_fraction: thighHeldFraction,
    samples,
    keypoints,
    peak_screenshot_data_url: peakScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: AKEFullResult): string {
  const parts: string[] = [];

  for (const side of ["left", "right"] as const) {
    const r = side === "left" ? result.left : result.right;
    if (!r) continue;
    const sideLabel = side === "left" ? "Left AKE" : "Right AKE";
    const knee = r.max_knee_angle_deg.toFixed(1);
    const deficit = r.deficit_deg.toFixed(1);

    if (r.max_knee_angle_deg < MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG) {
      parts.push(
        `${sideLabel}: no extension detected (peak knee ${knee}°). Re-record with ` +
        `the thigh held vertical and the patient slowly straightening the knee.`,
      );
      continue;
    }

    if (r.thigh_held_fraction < 0.3) {
      parts.push(
        `${sideLabel}: thigh did not stay at ~90° for most of the trial ` +
        `(${(r.thigh_held_fraction * 100).toFixed(0)}% of frames passed the ` +
        `${THIGH_HELD_MIN_DEG}–${THIGH_HELD_MAX_DEG}° hip-flexion gate). ` +
        `Result reflects the best qualifying moment (max knee ${knee}°, ` +
        `deficit ${deficit}°) — consider a fresh attempt with the hip held stable.`,
      );
      continue;
    }

    if (r.classification === "normal") {
      parts.push(
        `${sideLabel}: max knee ${knee}°, deficit ${deficit}° — ` +
        `normal hamstring flexibility (deficit ≤ ${NORMAL_MAX_DEFICIT_DEG}°).`,
      );
    } else if (r.classification === "mild") {
      parts.push(
        `${sideLabel}: max knee ${knee}°, deficit ${deficit}° — ` +
        `mild hamstring tightness ` +
        `(${NORMAL_MAX_DEFICIT_DEG + 1}–${MILD_MAX_DEFICIT_DEG}°).`,
      );
    } else if (r.classification === "moderate") {
      parts.push(
        `${sideLabel}: max knee ${knee}°, deficit ${deficit}° — ` +
        `moderate hamstring tightness ` +
        `(${MILD_MAX_DEFICIT_DEG + 1}–${MODERATE_MAX_DEFICIT_DEG}°). ` +
        `Stretching program indicated.`,
      );
    } else {
      parts.push(
        `${sideLabel}: max knee ${knee}°, deficit ${deficit}° — ` +
        `severe hamstring tightness (> ${MODERATE_MAX_DEFICIT_DEG}°). ` +
        `Targeted hamstring lengthening recommended.`,
      );
    }
  }

  // L–R asymmetry callout (>10° deficit delta = clinically meaningful side
  // difference for hamstring length).
  if (
    result.left  && result.right &&
    result.left.max_knee_angle_deg  >= MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG &&
    result.right.max_knee_angle_deg >= MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG
  ) {
    const delta = Math.abs(result.left.deficit_deg - result.right.deficit_deg);
    if (delta > 10) {
      const tighter =
        result.left.deficit_deg > result.right.deficit_deg ? "left" : "right";
      parts.push(
        `L–R asymmetry of ${delta.toFixed(1)}° in extension deficit — the ` +
        `${tighter} hamstring is the tighter side.`,
      );
    }
  }

  if (parts.length === 0) {
    return "No completed AKE trials to interpret.";
  }
  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const AKE_CLASSIFICATION_LABEL: Record<AKEClassification, string> = {
  normal:   "Normal",
  mild:     "Mild tightness",
  moderate: "Moderate tightness",
  severe:   "Severe tightness",
};

export const AKE_CLASSIFICATION_TONE: Record<AKEClassification, string> = {
  normal:   "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  mild:     "bg-sky-500/10 text-sky-700 dark:text-sky-400",
  moderate: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  severe:   "bg-red-500/10 text-red-700 dark:text-red-400",
};

// ─── Upload-mode API client ─────────────────────────────────────
//
// POST /api/analyze-ake accepts ONE side per call. The frontend
// uploads left + right clips sequentially (not in parallel — same
// cold-worker 502 mitigation as SLR) and assembles AKEFullResult
// { left, right } client-side. Backend math + classification cutoffs
// mirror this file exactly, so the returned AKESideResult slots
// straight into AKEReport without translation.

interface AKEResponseDTO {
  success: boolean;
  data: AKESideResult | null;
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
      "the side with the patient's full body (head to feet) in frame."
    );
  }
  if (s.includes("thigh_not_held")) {
    return (
      "Thigh did not stay at ~90° during the trial. Re-record with the patient " +
      "holding the hip flexed at 90° throughout the knee extension."
    );
  }
  if (s.includes("no_extension")) {
    return (
      "No knee extension detected. Please re-record the patient slowly " +
      "straightening the knee while keeping the thigh vertical."
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

export async function analyzeAKEUpload(
  file: File,
  side: Side,
  onProgress?: (pct: number) => void,
): Promise<AKESideResult> {
  const form = new FormData();
  form.append("video", file, file.name || "ake.mp4");
  form.append("side", side);

  // Indeterminate-style progress: the fetch API doesn't expose real
  // upload progress without XHR/streams, and the bulk of the elapsed
  // time is server-side analysis anyway. Pulse 5% → ~90% → 100%.
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
    const res = await authedFetch("/api/analyze-ake", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `AKE analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as AKEResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
