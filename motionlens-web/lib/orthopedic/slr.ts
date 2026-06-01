// Straight Leg Raise (SLR) test — math, conventions, validity gate,
// per-side aggregation, classification, upload-mode API client.
//
// Setup:
//   Patient supine (face up). Camera on the side (lateral view). Test
//   one leg at a time; the camera should sit on the same side as the
//   leg being raised so the test-side hip → knee → ankle chain is
//   clearly visible. Patient raises the leg with the knee held
//   straight; the maximum angle from flat is the result.
//
// Sign convention (image space, y-down):
//   • body baseline vector  = hip-midpoint → shoulder-midpoint
//   • leg vector            = test-side hip → test-side ankle
//   • inner_angle           = angle between body baseline and leg (0..180°)
//                             — 180° when leg + torso are collinear
//                               (patient lying flat, feet pointing away
//                               from head)
//   • raise_angle           = 180° − inner_angle
//                             — 0°  = leg flat alongside the body
//                             — 90° = leg vertical
//                             — >90° = past vertical (hypermobile)
//   This is independent of head orientation in the frame (left or
//   right side of the image) and of camera placement, because both
//   vectors flip together when the patient flips.
//
// Knee straightness:
//   inner angle at the knee between (knee → hip) and (knee → ankle).
//   180° = perfectly straight; the rep is "valid" only while
//   knee_angle_deg ≥ STRAIGHT_THRESHOLD_DEG.
//
// Result per side = max(raise_angle) across frames where the knee was
// straight AND the test-side leg landmarks were all visible. The
// backend mirrors this math exactly in
// engines/orthopedic/slr_engine.py so live and upload return identical
// numbers.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";

// ─── Tuning constants ───────────────────────────────────────────
const VIS_THRESHOLD = 0.3;

// Sample rate for the per-frame time-series stored on the result.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// One continuous trial per side — no rep counting. The patient raises
// the leg once (or a few practice attempts) within the window; we
// keep the maximum valid raise across the whole trial.
export const TRIAL_DURATION_SEC = 15;

// Knee inner-angle threshold for "straight enough" (degrees). 180° is
// perfectly straight; we allow 20° of bend to tolerate 2D pose noise
// and slight anatomical bend, but reject obvious flexion. Frames
// below this threshold are excluded from the max-raise calculation.
export const STRAIGHT_THRESHOLD_DEG = 160;

// Lower bound for "the patient actually raised the leg". If no valid
// frame reaches this raise angle the trial is flagged as
// no_raise_detected.
export const MIN_RAISE_FOR_VALID_TRIAL_DEG = 5;

// PDF Interpretation cutoffs.
export const SEVERELY_LIMITED_MAX_DEG = 30;   // <30° = severely limited
export const POSITIVE_MAX_DEG          = 70;  // 30–70° = positive
export const NORMAL_MAX_DEG            = 90;  // 70–90° = normal; >90° = hypermobile

export type Side = "left" | "right";
export type SLRClassification =
  | "severely_limited"
  | "positive"
  | "normal"
  | "hypermobile";
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

// raise_angle = 180° − angle(torso, test-leg)
// 0° at rest (leg flat in line with body), 90° leg vertical, >90°
// past vertical (hypermobile).
export function computeRaiseAngle(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const ankle = keypoints[idx.ankle];
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  const lSh  = keypoints[LM.LEFT_SHOULDER];
  const rSh  = keypoints[LM.RIGHT_SHOULDER];
  if (
    !visible(hip) || !visible(ankle) ||
    !visible(lHip) || !visible(rHip) ||
    !visible(lSh) || !visible(rSh)
  ) return null;

  const hipMidX = (lHip.x + rHip.x) / 2;
  const hipMidY = (lHip.y + rHip.y) / 2;
  const shMidX  = (lSh.x  + rSh.x ) / 2;
  const shMidY  = (lSh.y  + rSh.y ) / 2;

  const torsoVx = shMidX - hipMidX;
  const torsoVy = shMidY - hipMidY;
  const legVx   = ankle.x - hip.x;
  const legVy   = ankle.y - hip.y;

  const inner = vectorAngleDeg(torsoVx, torsoVy, legVx, legVy);
  if (inner === null) return null;
  return 180 - inner;
}

// Inner angle at the knee — 180° = perfectly straight.
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

export function classifySLR(maxRaiseDeg: number): SLRClassification {
  if (maxRaiseDeg < SEVERELY_LIMITED_MAX_DEG) return "severely_limited";
  if (maxRaiseDeg < POSITIVE_MAX_DEG)          return "positive";
  if (maxRaiseDeg <= NORMAL_MAX_DEG)           return "normal";
  return "hypermobile";
}

// ─── Per-trial aggregation ──────────────────────────────────────

export interface FrameSample {
  t_ms: number;
  raise_angle_deg: number | null;
  knee_angle_deg: number | null;
  knee_straight: boolean;
}

export interface SLRSideResult {
  side_tested: Side;
  /** Highest raise angle reached while the knee was straight. 0 if
   *  no frame qualified. Not capped at 90°. */
  max_raise_angle_deg: number;
  /** Frame index (within `samples`) at which max_raise_angle_deg was
   *  reached. Null when no valid frame qualified. */
  max_raise_sample_index: number | null;
  /** Best knee straightness (closest to 180°) observed at the moment
   *  of peak raise. Surfaced so the report can show the operator how
   *  straight the knee actually was at the captured peak. */
  knee_angle_at_peak_deg: number | null;
  classification: SLRClassification;
  duration_seconds: number;
  termination: Termination;
  /** Fraction of trial frames where the knee passed the straightness
   *  gate. Low values mean the patient bent the knee throughout. */
  knee_straight_fraction: number;
  /** Per-frame time-series. */
  samples: FrameSample[];
  /** Per-frame keypoints (PDF Section 2 (a) compliance). */
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  /** JPEG data-URL of the peak-raise frame (skeleton-overlaid). */
  peak_screenshot_data_url: string | null;
}

export interface SLRFullResult {
  left:  SLRSideResult | null;
  right: SLRSideResult | null;
}

export function summarizeSide(
  side: Side,
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  samples: FrameSample[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  peakScreenshotDataUrl: string | null,
): SLRSideResult {
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);

  let maxRaise = 0;
  let maxIdx: number | null = null;
  let kneeAtPeak: number | null = null;
  let straightCount = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.knee_straight) straightCount += 1;
    if (
      s.knee_straight &&
      s.raise_angle_deg !== null &&
      s.raise_angle_deg > maxRaise
    ) {
      maxRaise = s.raise_angle_deg;
      maxIdx = i;
      kneeAtPeak = s.knee_angle_deg;
    }
  }
  const kneeStraightFraction =
    samples.length === 0 ? 0 : straightCount / samples.length;

  return {
    side_tested: side,
    max_raise_angle_deg: maxRaise,
    max_raise_sample_index: maxIdx,
    knee_angle_at_peak_deg: kneeAtPeak,
    classification: classifySLR(maxRaise),
    duration_seconds: duration,
    termination,
    knee_straight_fraction: kneeStraightFraction,
    samples,
    keypoints,
    peak_screenshot_data_url: peakScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: SLRFullResult): string {
  const parts: string[] = [];

  for (const side of ["left", "right"] as const) {
    const r = side === "left" ? result.left : result.right;
    if (!r) continue;
    const sideLabel = side === "left" ? "Left SLR" : "Right SLR";
    const angle = r.max_raise_angle_deg.toFixed(1);

    if (r.max_raise_angle_deg < MIN_RAISE_FOR_VALID_TRIAL_DEG) {
      parts.push(
        `${sideLabel}: no raise detected (${angle}°). Re-record with the leg lifted ` +
        `from flat to as high as the patient can comfortably reach while keeping ` +
        `the knee straight.`,
      );
      continue;
    }

    if (r.knee_straight_fraction < 0.3) {
      parts.push(
        `${sideLabel}: knee did not stay straight for most of the trial ` +
        `(${(r.knee_straight_fraction * 100).toFixed(0)}% of frames passed the ` +
        `${STRAIGHT_THRESHOLD_DEG}° straightness gate). Result reflects the best ` +
        `qualifying moment (${angle}°) — consider a fresh attempt with the knee held straight.`,
      );
      continue;
    }

    if (r.classification === "severely_limited") {
      parts.push(
        `${sideLabel}: ${angle}° — severely limited (< ${SEVERELY_LIMITED_MAX_DEG}°). ` +
        `Marked restriction; correlate with hamstring length, lumbar pathology, ` +
        `and pain provocation.`,
      );
    } else if (r.classification === "positive") {
      parts.push(
        `${sideLabel}: ${angle}° — positive SLR (${SEVERELY_LIMITED_MAX_DEG}–${POSITIVE_MAX_DEG}°). ` +
        `Range suggests possible neural tension; consider lumbar nerve-root involvement and ` +
        `complementary tests (Bragard's, slump).`,
      );
    } else if (r.classification === "normal") {
      parts.push(
        `${sideLabel}: ${angle}° — within normal range (${POSITIVE_MAX_DEG}–${NORMAL_MAX_DEG}°).`,
      );
    } else {
      parts.push(
        `${sideLabel}: ${angle}° — hypermobile range (> ${NORMAL_MAX_DEG}°). ` +
        `Note overall joint hypermobility if observed elsewhere.`,
      );
    }
  }

  // L–R asymmetry callout (>15° delta = clinically meaningful side difference).
  if (
    result.left  && result.right &&
    result.left.max_raise_angle_deg  >= MIN_RAISE_FOR_VALID_TRIAL_DEG &&
    result.right.max_raise_angle_deg >= MIN_RAISE_FOR_VALID_TRIAL_DEG
  ) {
    const delta = Math.abs(
      result.left.max_raise_angle_deg - result.right.max_raise_angle_deg,
    );
    if (delta > 15) {
      const worse =
        result.left.max_raise_angle_deg < result.right.max_raise_angle_deg
          ? "left"
          : "right";
      parts.push(
        `L–R asymmetry of ${delta.toFixed(1)}° — the ${worse} side is the more ` +
        `restricted leg.`,
      );
    }
  }

  if (parts.length === 0) {
    return "No completed SLR trials to interpret.";
  }
  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const SLR_CLASSIFICATION_LABEL: Record<SLRClassification, string> = {
  severely_limited: "Severely limited",
  positive:         "Positive SLR",
  normal:           "Normal",
  hypermobile:      "Hypermobile",
};

export const SLR_CLASSIFICATION_TONE: Record<SLRClassification, string> = {
  severely_limited: "bg-red-500/10 text-red-700 dark:text-red-400",
  positive:         "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  normal:           "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  hypermobile:      "bg-sky-500/10 text-sky-700 dark:text-sky-400",
};

// ─── Upload-mode API client ─────────────────────────────────────
//
// POST /api/analyze-slr accepts ONE side per call. The frontend
// uploads left + right clips in parallel (Promise.allSettled) and
// assembles SLRFullResult { left, right } client-side. Backend math
// + classification cutoffs mirror this file exactly, so the returned
// SLRSideResult slots straight into SLRReport without translation.

interface SLRResponseDTO {
  success: boolean;
  data: SLRSideResult | null;
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
  if (s.includes("knee_not_straight")) {
    return (
      "Knee did not stay straight during the raise. Re-record with the patient " +
      "keeping the knee fully extended throughout the lift."
    );
  }
  if (s.includes("no_raise") || s.includes("no raise detected")) {
    return (
      "No leg raise detected. Please re-record the patient raising the leg from " +
      "flat to as high as they can comfortably reach."
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

export async function analyzeSLRUpload(
  file: File,
  side: Side,
  onProgress?: (pct: number) => void,
): Promise<SLRSideResult> {
  const form = new FormData();
  form.append("video", file, file.name || "slr.mp4");
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
    const res = await authedFetch("/api/analyze-slr", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `SLR analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as SLRResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
