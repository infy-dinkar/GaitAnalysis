// Modified Thomas Test (MTT) — math, stability detection, per-side
// aggregation, classification, upload-mode API client.
//
// Setup:
//   Patient sits on the edge of a table, then lies back so the upper
//   body (waist-up) is supported and both legs hang off the edge below
//   the waist. The patient pulls ONE knee up to the chest (this knee
//   stabilises the pelvis — it is NOT the leg being measured). The
//   OTHER leg is left to relax and hang naturally off the edge — that
//   hanging leg is the TEST leg. Camera is placed on the same side as
//   the hanging leg (lateral view), framed tall so the whole body from
//   shoulder down to the hanging ankle is visible.
//
// Static-hold, not motion:
//   Unlike SLR / AKE where we track a peak during a slow movement,
//   MTT measures a SETTLED position. Wait until the pose stops moving
//   (hip + knee angles steady within ±STABILITY_JITTER_MAX_DEG for
//   STABILITY_WINDOW_FRAMES consecutive samples — ~1.5 s at 10 Hz)
//   and capture the MEDIAN angles over that window. The engine
//   mirrors this on the backend by finding the longest stable window
//   in the uploaded clip and returning the same medians, so live and
//   upload report identical numbers.
//
// Two angles per side (both on the test-side leg only):
//   • Hip angle   — inner angle at the hip between hip→shoulder and
//                   hip→knee. Thigh hanging in line with the body (down
//                   along the body axis) ≈ 180°. A tight hip flexor
//                   keeps the thigh from dropping, pulling the angle
//                   below 180°.
//   • Knee angle  — inner angle at the knee between knee→hip and
//                   knee→ankle. A naturally bent hanging shin sits at
//                   ~80–90°. A tight rectus femoris extends the knee
//                   toward straight, pushing the angle above ~100°.
//
// Classification cutoffs:
//   Hip (hip flexor):
//     ≥ 170°       → "normal"
//     155°–170°    → "mild" tightness
//     < 155°       → "significant" tightness
//   Knee (rectus femoris):
//     ≤ 100°       → "normal"
//     > 100°       → "tight" (rectus femoris)

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";

// ─── Tuning constants ───────────────────────────────────────────
const VIS_THRESHOLD = 0.3;

export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// Trial window — generous because the patient needs to physically
// settle into the hanging-leg position. Auto-capture short-circuits
// this once stability is detected.
export const TRIAL_DURATION_SEC = 30;

// Stability window: how many consecutive samples must all sit within
// the jitter band to qualify as "settled". 1.5 s at 10 Hz = 15 frames.
export const STABILITY_WINDOW_FRAMES = 15;

// Maximum within-window stddev (degrees) allowed for both the hip and
// the knee angle for the window to count as stable. 2.5° tolerates
// the natural MediaPipe pose-noise floor while still rejecting active
// motion.
export const STABILITY_JITTER_MAX_DEG = 2.5;

// Classification cutoffs (hip flexor — angle at the hip).
export const HIP_NORMAL_MIN_DEG = 170;
export const HIP_MILD_MIN_DEG = 155;

// Classification cutoffs (rectus femoris — angle at the knee).
// Knee inner angle ≤ this = normal; > this = rectus tightness.
export const KNEE_NORMAL_MAX_DEG = 100;

export type Side = "left" | "right";
export type HipClassification = "normal" | "mild" | "significant";
export type KneeClassification = "normal" | "tight";
export type Termination = "captured" | "timeout" | "stopped";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

const SIDE_INDICES = {
  left: {
    shoulder: LM.LEFT_SHOULDER,
    hip: LM.LEFT_HIP,
    knee: LM.LEFT_KNEE,
    ankle: LM.LEFT_ANKLE,
  },
  right: {
    shoulder: LM.RIGHT_SHOULDER,
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

// Hip angle = inner angle at the hip between (hip→shoulder) and
// (hip→knee). 180° = thigh in line with the body (test leg hanging
// straight down along the body axis). < 180° = thigh can't drop fully
// = tight hip flexor.
export function computeHipAngle(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const shoulder = keypoints[idx.shoulder];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  if (!visible(shoulder) || !visible(hip) || !visible(knee)) return null;
  return vectorAngleDeg(
    shoulder.x - hip.x, shoulder.y - hip.y,
    knee.x     - hip.x, knee.y     - hip.y,
  );
}

// Knee angle = inner angle at the knee between (knee→hip) and
// (knee→ankle). ~90° = shin hanging naturally bent. > 100° = knee
// straightening = rectus femoris tightness.
export function computeKneeAngle(keypoints: Keypoint[], side: Side): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  if (!visible(hip) || !visible(knee) || !visible(ankle)) return null;
  return vectorAngleDeg(
    hip.x   - knee.x, hip.y   - knee.y,
    ankle.x - knee.x, ankle.y - knee.y,
  );
}

// Pre-record gate: test-side shoulder + hip + knee + ankle all
// visible. The contralateral leg is intentionally NOT required —
// it's pulled to the chest and frequently occluded or out of frame.
export function isTestSideTrackable(keypoints: Keypoint[], side: Side): boolean {
  const idx = SIDE_INDICES[side];
  return (
    visible(keypoints[idx.shoulder]) &&
    visible(keypoints[idx.hip]) &&
    visible(keypoints[idx.knee]) &&
    visible(keypoints[idx.ankle])
  );
}

// ─── Stability + reduction ──────────────────────────────────────

/**
 * Population standard deviation of a number list. Returns null when
 * the list has fewer than 2 elements.
 */
export function stddevDeg(values: number[]): number | null {
  if (values.length < 2) return null;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / values.length;
  return Math.sqrt(variance);
}

/** Median of a number list. Returns null when the list is empty. */
export function medianDeg(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export interface StableWindow {
  hip_angle_deg: number;
  knee_angle_deg: number;
  hip_angle_stddev_deg: number;
  knee_angle_stddev_deg: number;
  /** Index (within the samples array) of the LAST frame in the window
   *  — i.e. the frame at which stability was first confirmed. */
  end_index: number;
}

/**
 * Inspect the tail of `samples`. If the last STABILITY_WINDOW_FRAMES
 * entries all have non-null angles AND both stddevs are within the
 * jitter threshold, return a StableWindow built from the medians.
 * Otherwise return null. Use this in the live frame callback to fire
 * the auto-capture trigger.
 */
export function detectStableTail(samples: FrameSample[]): StableWindow | null {
  if (samples.length < STABILITY_WINDOW_FRAMES) return null;
  const tail = samples.slice(samples.length - STABILITY_WINDOW_FRAMES);
  const hips: number[] = [];
  const knees: number[] = [];
  for (const s of tail) {
    if (s.hip_angle_deg === null || s.knee_angle_deg === null) return null;
    hips.push(s.hip_angle_deg);
    knees.push(s.knee_angle_deg);
  }
  const hipStd = stddevDeg(hips);
  const kneeStd = stddevDeg(knees);
  if (hipStd === null || kneeStd === null) return null;
  if (hipStd > STABILITY_JITTER_MAX_DEG) return null;
  if (kneeStd > STABILITY_JITTER_MAX_DEG) return null;
  const hipMed = medianDeg(hips);
  const kneeMed = medianDeg(knees);
  if (hipMed === null || kneeMed === null) return null;
  return {
    hip_angle_deg: hipMed,
    knee_angle_deg: kneeMed,
    hip_angle_stddev_deg: hipStd,
    knee_angle_stddev_deg: kneeStd,
    end_index: samples.length - 1,
  };
}

// ─── Classification ─────────────────────────────────────────────

export function classifyHip(hipAngleDeg: number): HipClassification {
  if (hipAngleDeg >= HIP_NORMAL_MIN_DEG) return "normal";
  if (hipAngleDeg >= HIP_MILD_MIN_DEG)   return "mild";
  return "significant";
}

export function classifyKnee(kneeAngleDeg: number): KneeClassification {
  return kneeAngleDeg > KNEE_NORMAL_MAX_DEG ? "tight" : "normal";
}

// ─── Per-trial aggregation ──────────────────────────────────────

export interface FrameSample {
  t_ms: number;
  hip_angle_deg: number | null;
  knee_angle_deg: number | null;
  /** True if the rolling window ending at this frame met the
   *  stability gate. Set on every frame so the report chart can
   *  highlight the stable region. */
  stable: boolean;
}

export interface ModifiedThomasSideResult {
  side_tested: Side;
  /** Settled hip angle — median over the captured stable window.
   *  Larger = thigh hangs further down. 180° ≈ neutral. */
  hip_angle_deg: number;
  /** Settled knee angle — median over the captured stable window.
   *  ~80–90° = relaxed bent knee. > 100° = rectus femoris tightness. */
  knee_angle_deg: number;
  hip_classification: HipClassification;
  knee_classification: KneeClassification;
  /** Within-window stddev at the moment of capture. Surfaced so the
   *  report can show the operator how steady the pose was. */
  hip_angle_stddev_deg: number;
  knee_angle_stddev_deg: number;
  /** True when the trial timed out without auto-capture firing —
   *  the engine returned the median of the last available window,
   *  but it didn't meet the jitter gate. The doctor should note this
   *  when interpreting borderline values. */
  low_confidence: boolean;
  /** Frame index (within `samples`) where the capture window ended. */
  capture_sample_index: number | null;
  duration_seconds: number;
  termination: Termination;
  /** Per-frame time-series. */
  samples: FrameSample[];
  /** Per-frame keypoints (PDF Section 2 (a) compliance). */
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  /** JPEG data-URL of the capture-moment frame (skeleton-overlaid). */
  capture_screenshot_data_url: string | null;
}

export interface ModifiedThomasFullResult {
  left:  ModifiedThomasSideResult | null;
  right: ModifiedThomasSideResult | null;
}

export function summarizeSide(
  side: Side,
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  samples: FrameSample[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  capture: StableWindow | null,
  lowConfidence: boolean,
  captureScreenshotDataUrl: string | null,
): ModifiedThomasSideResult {
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);

  // capture is null only when we couldn't even build a fallback window
  // (e.g. trial stopped before STABILITY_WINDOW_FRAMES samples). In
  // that case report 0/0 with low_confidence=true — the saved row
  // still renders, but the doctor sees a sane "not measured" view.
  const hipDeg  = capture?.hip_angle_deg  ?? 0;
  const kneeDeg = capture?.knee_angle_deg ?? 0;
  const hipStd  = capture?.hip_angle_stddev_deg  ?? 0;
  const kneeStd = capture?.knee_angle_stddev_deg ?? 0;
  const captureIdx = capture?.end_index ?? null;

  return {
    side_tested: side,
    hip_angle_deg: hipDeg,
    knee_angle_deg: kneeDeg,
    hip_classification: classifyHip(hipDeg),
    knee_classification: classifyKnee(kneeDeg),
    hip_angle_stddev_deg: hipStd,
    knee_angle_stddev_deg: kneeStd,
    low_confidence: lowConfidence || capture === null,
    capture_sample_index: captureIdx,
    duration_seconds: duration,
    termination,
    samples,
    keypoints,
    capture_screenshot_data_url: captureScreenshotDataUrl,
  };
}

/**
 * Fallback reduction used when auto-capture didn't fire (trial timeout
 * or stopped early). Looks at the LAST STABILITY_WINDOW_FRAMES samples
 * with non-null angles; if at least 5 valid samples are present,
 * returns a StableWindow built from their medians (flagged
 * low_confidence by the caller). Otherwise returns null.
 */
export function fallbackReduction(samples: FrameSample[]): StableWindow | null {
  const tail = samples.slice(-STABILITY_WINDOW_FRAMES);
  const hips: number[] = [];
  const knees: number[] = [];
  for (const s of tail) {
    if (s.hip_angle_deg !== null) hips.push(s.hip_angle_deg);
    if (s.knee_angle_deg !== null) knees.push(s.knee_angle_deg);
  }
  if (hips.length < 5 || knees.length < 5) return null;
  const hipMed = medianDeg(hips);
  const kneeMed = medianDeg(knees);
  const hipStd = stddevDeg(hips) ?? 0;
  const kneeStd = stddevDeg(knees) ?? 0;
  if (hipMed === null || kneeMed === null) return null;
  return {
    hip_angle_deg: hipMed,
    knee_angle_deg: kneeMed,
    hip_angle_stddev_deg: hipStd,
    knee_angle_stddev_deg: kneeStd,
    end_index: samples.length - 1,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: ModifiedThomasFullResult): string {
  const parts: string[] = [];

  for (const side of ["left", "right"] as const) {
    const r = side === "left" ? result.left : result.right;
    if (!r) continue;
    const sideLabel = side === "left" ? "Left MTT" : "Right MTT";
    const hip = r.hip_angle_deg.toFixed(1);
    const knee = r.knee_angle_deg.toFixed(1);

    if (r.hip_angle_deg <= 0 || r.knee_angle_deg <= 0) {
      parts.push(
        `${sideLabel}: no settled position captured. Re-record with the patient ` +
        `lying still in the Modified Thomas position for at least 2 seconds.`,
      );
      continue;
    }

    let hipText: string;
    if (r.hip_classification === "normal") {
      hipText = `hip ${hip}° — normal hip-flexor length (≥ ${HIP_NORMAL_MIN_DEG}°)`;
    } else if (r.hip_classification === "mild") {
      hipText = (
        `hip ${hip}° — mild hip-flexor tightness ` +
        `(${HIP_MILD_MIN_DEG}–${HIP_NORMAL_MIN_DEG}°)`
      );
    } else {
      hipText = (
        `hip ${hip}° — significant hip-flexor tightness ` +
        `(< ${HIP_MILD_MIN_DEG}°)`
      );
    }

    const kneeText =
      r.knee_classification === "normal"
        ? `knee ${knee}° — relaxed (≤ ${KNEE_NORMAL_MAX_DEG}°), no rectus femoris tightness`
        : `knee ${knee}° — extended (> ${KNEE_NORMAL_MAX_DEG}°), rectus femoris tightness present`;

    const confidence = r.low_confidence
      ? " (low-confidence capture — pose didn't fully settle, consider re-recording)"
      : "";

    parts.push(`${sideLabel}: ${hipText}; ${kneeText}.${confidence}`);
  }

  // L–R asymmetry callouts.
  if (
    result.left  && result.right &&
    result.left.hip_angle_deg  > 0 && result.right.hip_angle_deg  > 0
  ) {
    const hipDelta = Math.abs(result.left.hip_angle_deg - result.right.hip_angle_deg);
    if (hipDelta > 10) {
      const tighter =
        result.left.hip_angle_deg < result.right.hip_angle_deg ? "left" : "right";
      parts.push(
        `L–R hip asymmetry of ${hipDelta.toFixed(1)}° — the ${tighter} hip flexor ` +
        `is the tighter side.`,
      );
    }
  }

  if (parts.length === 0) {
    return "No completed MTT trials to interpret.";
  }
  return parts.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const HIP_CLASSIFICATION_LABEL: Record<HipClassification, string> = {
  normal:      "Normal",
  mild:        "Mild tightness",
  significant: "Significant tightness",
};

export const HIP_CLASSIFICATION_TONE: Record<HipClassification, string> = {
  normal:      "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  mild:        "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  significant: "bg-red-500/10 text-red-700 dark:text-red-400",
};

export const KNEE_CLASSIFICATION_LABEL: Record<KneeClassification, string> = {
  normal: "Relaxed",
  tight:  "Rectus tightness",
};

export const KNEE_CLASSIFICATION_TONE: Record<KneeClassification, string> = {
  normal: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  tight:  "bg-amber-500/10 text-amber-700 dark:text-amber-400",
};

// ─── Upload-mode API client ─────────────────────────────────────
//
// POST /api/analyze-modified-thomas accepts ONE side per call. The
// frontend uploads left + right clips sequentially (not in parallel —
// same cold-worker 502 mitigation as SLR / AKE) and assembles
// ModifiedThomasFullResult { left, right } client-side. Backend math
// + classification cutoffs mirror this file exactly, so the returned
// ModifiedThomasSideResult slots straight into ModifiedThomasReport
// without translation.

interface MTTResponseDTO {
  success: boolean;
  data: ModifiedThomasSideResult | null;
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
      "the side with the patient's whole body — shoulder down to the hanging " +
      "ankle — in the same tall frame."
    );
  }
  if (s.includes("no_stable_window")) {
    return (
      "The patient never held a stable position. Re-record with the patient " +
      "settled into the Modified Thomas position and held still for at least " +
      "2 seconds."
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

export async function analyzeModifiedThomasUpload(
  file: File,
  side: Side,
  onProgress?: (pct: number) => void,
): Promise<ModifiedThomasSideResult> {
  const form = new FormData();
  form.append("video", file, file.name || "modified_thomas.mp4");
  form.append("side", side);

  // Indeterminate-style progress: pulse 5% → ~90% → 100%.
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
    const res = await authedFetch("/api/analyze-modified-thomas", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `Modified Thomas analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as MTTResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
