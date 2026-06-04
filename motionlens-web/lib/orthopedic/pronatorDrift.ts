// Pronator Drift (E2) — 2D vertical-drift screen for subtle upper-
// motor-neuron weakness.
//
// Setup:
//   Patient stands (or sits) facing the camera, both arms extended
//   forward at shoulder height (90° shoulder flexion), elbows
//   straight, palms up. Eyes closed for the hold. Camera is FRONTAL,
//   chest height, far enough that both extended arms are fully in
//   frame. ~20 s hold; audio cues at start + end (no visual cue —
//   eyes are closed).
//
// What we measure:
//   • Per-arm vertical drift (px and cm) — final wrist Y minus the
//     baseline wrist Y. Positive value = wrist dropped in real space
//     (image y-down: drop = increasing y).
//   • Per-arm drift velocity (cm/sec) — least-squares slope of
//     drift-vs-time across the hold (after the baseline window).
//   • Left–right asymmetry — ratio of the two arms' positive (downward)
//     drift; classic pronator drift is a strongly asymmetric pattern.
//
// CRITICAL 2D LIMITATION (surfaced in the report):
//   True clinical pronator drift involves the forearm rotating /
//   pronating as it drops. A monocular 2D system CANNOT measure
//   rotation — we only see the vertical drop component. Clinical
//   judgement remains required; the report explicitly flags this.
//
// Pixel-to-cm normalisation:
//   Same scheme as Tandem Walk — at each frame the patient's
//   shoulder-to-shoulder width in pixels IS the scale ruler, and
//   ASSUMED_SHOULDER_WIDTH_CM converts back to a clinically readable
//   number. Self-calibrating across patient distance and zoom.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { authedFetch, AuthError } from "@/lib/auth";

const VIS_THRESHOLD = 0.3;

// Sampling cadence for the per-frame time-series saved on the result.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// Hold timing (seconds). 20 s sits at the lower end of the clinical
// 20–30 s range; gives a clean signal without exhausting patients.
export const TARGET_HOLD_DURATION_SEC = 20;
// Skip this many ms of arm-settle motion at the very start before
// the baseline window opens.
export const SETTLE_DELAY_MS = 500;
// Baseline = median wrist Y across this many seconds AFTER the
// settle delay. 1 s × 10 Hz = 10 samples; the median rejects single-
// frame keypoint outliers.
export const BASELINE_WINDOW_SEC = 1;
// Final position = median wrist Y across this many seconds at the
// END of the hold.
export const FINAL_WINDOW_SEC = 1;
// 3-second visual+audio countdown before the hold actually begins.
export const COUNTDOWN_SEC = 3;

// Calibration assumption — shoulder-to-shoulder width for an adult.
// drift_cm = (drift_px / shoulder_width_px) × this constant.
export const ASSUMED_SHOULDER_WIDTH_CM = 40;

// Classification cutoffs (cross-checked against the user spec):
//   • Asymmetric drop pattern (classic pronator drift):
//       drop on one side > POSITIVE_DRIFT_CM   (5 cm)
//       AND opposite side stable (< STABLE_THRESHOLD_CM = 2 cm)
//   • Asymmetry ratio gate:
//       max_drop / min_drop  > POSITIVE_ASYMMETRY_RATIO (3:1)
//       AND max_drop > BORDERLINE_DRIFT_CM (2 cm) so two
//       essentially-stable arms don't trip the ratio.
export const POSITIVE_DRIFT_CM = 5;
export const STABLE_THRESHOLD_CM = 2;
export const POSITIVE_ASYMMETRY_RATIO = 3;
export const BORDERLINE_DRIFT_CM = 2;

export type PronatorDriftClassification = "normal" | "borderline" | "positive_screen";
export type Termination = "completed" | "stopped" | "timeout";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

// ─── Per-frame computations ─────────────────────────────────────

export function computeWristY(keypoints: Keypoint[], side: "left" | "right"): number | null {
  const idx = side === "left" ? LM.LEFT_WRIST : LM.RIGHT_WRIST;
  const wr = keypoints[idx];
  if (!visible(wr)) return null;
  return wr.y;
}

export function computeShoulderY(keypoints: Keypoint[], side: "left" | "right"): number | null {
  const idx = side === "left" ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER;
  const sh = keypoints[idx];
  if (!visible(sh)) return null;
  return sh.y;
}

export function computeShoulderWidthPx(keypoints: Keypoint[]): number | null {
  const ls = keypoints[LM.LEFT_SHOULDER];
  const rs = keypoints[LM.RIGHT_SHOULDER];
  if (!visible(ls) || !visible(rs)) return null;
  return Math.hypot(ls.x - rs.x, ls.y - rs.y);
}

/** Test setup gate: both wrists + both shoulders visible. The
 *  patient should hold both arms extended; if either wrist isn't
 *  tracked we can't measure drift on that side. */
export function isBothArmsTrackable(keypoints: Keypoint[]): boolean {
  return (
    visible(keypoints[LM.LEFT_SHOULDER]) &&
    visible(keypoints[LM.RIGHT_SHOULDER]) &&
    visible(keypoints[LM.LEFT_WRIST]) &&
    visible(keypoints[LM.RIGHT_WRIST])
  );
}

// ─── Median + linear-regression helpers ─────────────────────────

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/** Least-squares slope (y = m·x + c, returns m) for the given pairs.
 *  Returns 0 if fewer than 2 valid pairs or the variance is zero. */
function linearSlope(xs: number[], ys: number[]): number {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return 0;
  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i]; sumY += ys[i];
    sumXX += xs[i] * xs[i];
    sumXY += xs[i] * ys[i];
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-9) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

// ─── Result types ───────────────────────────────────────────────

export interface FrameSample {
  t_ms: number;
  left_wrist_y: number | null;
  right_wrist_y: number | null;
  left_shoulder_y: number | null;
  right_shoulder_y: number | null;
  shoulder_width_px: number | null;
}

export interface ArmDriftSummary {
  /** Median baseline wrist Y in pixels, across the BASELINE window. */
  baseline_wrist_y_px: number | null;
  /** Median final wrist Y in pixels, across the FINAL window. */
  final_wrist_y_px: number | null;
  /** Signed pixel drift = final − baseline. Positive = wrist dropped
   *  in image-y-down space. Null when either median is null. */
  drift_px: number | null;
  /** Signed cm drift, normalised by the mean shoulder width across
   *  the trial × ASSUMED_SHOULDER_WIDTH_CM. */
  drift_cm: number | null;
  /** Least-squares slope of (wrist_y − baseline) vs time, expressed
   *  in cm per second. Same sign convention as drift_cm. */
  drift_velocity_cm_per_sec: number | null;
  /** Time-series of per-frame drift in cm (since baseline_t to now).
   *  Used by the report chart. Null entries reflect frames where the
   *  wrist wasn't visible. */
  drift_cm_series: (number | null)[];
}

export interface PronatorDriftResult {
  hold_duration_seconds: number;
  /** Mean shoulder width across the captured samples — used to
   *  convert pixel drift to approximate cm. */
  mean_shoulder_width_px: number;
  left: ArmDriftSummary;
  right: ArmDriftSummary;
  /** Time axis (seconds since the hold began) for the drift series. */
  t_seconds_series: number[];
  /** max(left, right) downward drift in cm. Negative drifts (wrist
   *  moving up) clamped to 0 for screening purposes. */
  max_downward_drift_cm: number;
  /** min(left, right) downward drift. Same clamp. */
  min_downward_drift_cm: number;
  /** max_downward / min_downward. Reported as 999 when the smaller
   *  arm's drift is essentially zero (avoids divide-by-zero noise in
   *  the report). */
  asymmetry_ratio: number;
  /** |left_drift_cm − right_drift_cm|. Pairs naturally with the
   *  ratio for the asymmetry callout. */
  asymmetry_absolute_cm: number;
  classification: PronatorDriftClassification;
  termination: Termination;
  /** True when the hold ended before TARGET_HOLD_DURATION_SEC. */
  incomplete: boolean;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  capture_screenshot_data_url: string | null;
}

// ─── Classification ─────────────────────────────────────────────

export function classifyPronatorDrift(
  leftCm: number | null,
  rightCm: number | null,
): PronatorDriftClassification {
  // Treat upward drift as zero — the screen is about DOWNWARD drop.
  const lDown = leftCm  !== null ? Math.max(0, leftCm)  : 0;
  const rDown = rightCm !== null ? Math.max(0, rightCm) : 0;
  const maxDown = Math.max(lDown, rDown);
  const minDown = Math.min(lDown, rDown);

  // Classic asymmetric pattern: one side drops > 5 cm with the other
  // essentially stable (< 2 cm).
  if (maxDown > POSITIVE_DRIFT_CM && minDown < STABLE_THRESHOLD_CM) {
    return "positive_screen";
  }
  // Asymmetry ratio: only meaningful when the larger arm has actually
  // dropped (> borderline). Else two near-zero values would trip the
  // ratio on noise alone.
  if (maxDown > BORDERLINE_DRIFT_CM && minDown > 0.1) {
    const ratio = maxDown / minDown;
    if (ratio > POSITIVE_ASYMMETRY_RATIO) {
      return "positive_screen";
    }
  }
  if (maxDown > BORDERLINE_DRIFT_CM) {
    return "borderline";
  }
  return "normal";
}

// ─── Per-arm extraction ─────────────────────────────────────────

function extractArmSummary(
  samples: FrameSample[],
  side: "left" | "right",
  meanShoulderWidthPx: number,
  baselineStartIdx: number,
  baselineEndIdx: number,
  finalStartIdx: number,
  finalEndIdx: number,
): ArmDriftSummary {
  const key: "left_wrist_y" | "right_wrist_y" =
    side === "left" ? "left_wrist_y" : "right_wrist_y";

  const baselineVals: number[] = [];
  for (let i = baselineStartIdx; i <= baselineEndIdx && i < samples.length; i++) {
    const v = samples[i][key];
    if (v !== null) baselineVals.push(v);
  }
  const baselineY = median(baselineVals);

  const finalVals: number[] = [];
  for (let i = finalStartIdx; i <= finalEndIdx && i < samples.length; i++) {
    const v = samples[i][key];
    if (v !== null) finalVals.push(v);
  }
  const finalY = median(finalVals);

  const driftPx = baselineY !== null && finalY !== null
    ? finalY - baselineY
    : null;
  const driftCm = driftPx !== null && meanShoulderWidthPx > 0
    ? (driftPx / meanShoulderWidthPx) * ASSUMED_SHOULDER_WIDTH_CM
    : null;

  // Per-frame drift series (cm) — used by the report chart. Build it
  // for all samples, including those before the baseline window
  // (they'll just hover near zero) and after the final window.
  const driftCmSeries: (number | null)[] = samples.map((s) => {
    const v = s[key];
    if (v === null || baselineY === null || meanShoulderWidthPx <= 0) return null;
    return ((v - baselineY) / meanShoulderWidthPx) * ASSUMED_SHOULDER_WIDTH_CM;
  });

  // Drift velocity — slope of cm drift vs time IN SECONDS, computed
  // over the post-baseline portion of the hold only.
  const regXs: number[] = [];
  const regYs: number[] = [];
  for (let i = baselineEndIdx + 1; i < samples.length; i++) {
    const cm = driftCmSeries[i];
    if (cm === null) continue;
    regXs.push(samples[i].t_ms / 1000);
    regYs.push(cm);
  }
  const driftVelocityCmPerSec = regXs.length >= 2
    ? linearSlope(regXs, regYs)
    : null;

  return {
    baseline_wrist_y_px: baselineY,
    final_wrist_y_px: finalY,
    drift_px: driftPx,
    drift_cm: driftCm,
    drift_velocity_cm_per_sec: driftVelocityCmPerSec,
    drift_cm_series: driftCmSeries,
  };
}

// ─── Per-trial aggregator ──────────────────────────────────────

export function summarizeTrial(
  startedAtMs: number,
  endedAtMs: number,
  termination: Termination,
  samples: FrameSample[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  captureScreenshotDataUrl: string | null,
): PronatorDriftResult {
  const holdDuration = Math.max(0, (endedAtMs - startedAtMs) / 1000);

  // Determine baseline + final sample-index windows from t_ms.
  // Baseline runs [SETTLE_DELAY_MS, SETTLE_DELAY_MS + BASELINE_WINDOW_SEC * 1000].
  // Final runs the LAST FINAL_WINDOW_SEC of captured samples.
  const baselineStartMs = SETTLE_DELAY_MS;
  const baselineEndMs = baselineStartMs + BASELINE_WINDOW_SEC * 1000;
  let baselineStartIdx = 0;
  let baselineEndIdx = -1;
  for (let i = 0; i < samples.length; i++) {
    if (samples[i].t_ms >= baselineStartMs && baselineEndIdx < 0) {
      baselineStartIdx = i;
    }
    if (samples[i].t_ms <= baselineEndMs) {
      baselineEndIdx = i;
    }
  }
  if (baselineEndIdx < baselineStartIdx) baselineEndIdx = baselineStartIdx;

  const lastT = samples.length > 0
    ? samples[samples.length - 1].t_ms
    : 0;
  const finalStartMs = Math.max(baselineEndMs + 100, lastT - FINAL_WINDOW_SEC * 1000);
  let finalStartIdx = baselineEndIdx + 1;
  for (let i = baselineEndIdx + 1; i < samples.length; i++) {
    if (samples[i].t_ms >= finalStartMs) {
      finalStartIdx = i;
      break;
    }
  }
  const finalEndIdx = samples.length - 1;

  // Mean shoulder width across the captured samples — the cm ruler.
  const swVals: number[] = [];
  for (const s of samples) {
    if (s.shoulder_width_px !== null && s.shoulder_width_px > 0) {
      swVals.push(s.shoulder_width_px);
    }
  }
  const meanShoulderWidthPx = swVals.length === 0
    ? 0
    : swVals.reduce((a, b) => a + b, 0) / swVals.length;

  const left  = extractArmSummary(
    samples, "left",  meanShoulderWidthPx,
    baselineStartIdx, baselineEndIdx, finalStartIdx, finalEndIdx,
  );
  const right = extractArmSummary(
    samples, "right", meanShoulderWidthPx,
    baselineStartIdx, baselineEndIdx, finalStartIdx, finalEndIdx,
  );

  const lDown = left.drift_cm  !== null ? Math.max(0, left.drift_cm)  : 0;
  const rDown = right.drift_cm !== null ? Math.max(0, right.drift_cm) : 0;
  const maxDown = Math.max(lDown, rDown);
  const minDown = Math.min(lDown, rDown);
  const asymRatio = minDown > 0.1 ? maxDown / minDown : 999;
  const asymAbs = left.drift_cm !== null && right.drift_cm !== null
    ? Math.abs(left.drift_cm - right.drift_cm)
    : 0;

  const classification = classifyPronatorDrift(left.drift_cm, right.drift_cm);
  const incomplete = holdDuration < TARGET_HOLD_DURATION_SEC - 1;

  // Time axis for the chart, in seconds since hold start.
  const tSeries = samples.map((s) => s.t_ms / 1000);

  return {
    hold_duration_seconds: holdDuration,
    mean_shoulder_width_px: meanShoulderWidthPx,
    left,
    right,
    t_seconds_series: tSeries,
    max_downward_drift_cm: maxDown,
    min_downward_drift_cm: minDown,
    asymmetry_ratio: asymRatio,
    asymmetry_absolute_cm: asymAbs,
    classification,
    termination,
    incomplete,
    samples,
    keypoints,
    capture_screenshot_data_url: captureScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(result: PronatorDriftResult | null): string {
  if (!result) return "No completed pronator-drift trial to interpret.";

  const parts: string[] = [];
  parts.push(
    `${result.hold_duration_seconds.toFixed(1)} s hold. ` +
    `Left arm: ${fmtDriftDir(result.left.drift_cm)}. ` +
    `Right arm: ${fmtDriftDir(result.right.drift_cm)}.`,
  );

  if (result.classification === "positive_screen") {
    if (result.max_downward_drift_cm > POSITIVE_DRIFT_CM &&
        result.min_downward_drift_cm < STABLE_THRESHOLD_CM) {
      parts.push(
        `Asymmetric drop pattern: one arm dropped ` +
        `${result.max_downward_drift_cm.toFixed(1)} cm while the other ` +
        `stayed within ${STABLE_THRESHOLD_CM} cm — positive screen.`,
      );
    } else if (result.asymmetry_ratio > POSITIVE_ASYMMETRY_RATIO) {
      parts.push(
        `Asymmetry ratio ${result.asymmetry_ratio.toFixed(1)} : 1 ` +
        `(threshold ${POSITIVE_ASYMMETRY_RATIO} : 1) — positive screen for ` +
        `subtle upper-motor-neuron weakness on the affected side.`,
      );
    } else {
      parts.push("Positive screen — see report for details.");
    }
  } else if (result.classification === "borderline") {
    parts.push(
      `Borderline drop on at least one arm (${result.max_downward_drift_cm.toFixed(1)} cm). ` +
      `Repeat with a longer hold or correlate with strength testing.`,
    );
  } else {
    parts.push(
      `No significant drift — both arms stayed within the ` +
      `${BORDERLINE_DRIFT_CM} cm tolerance.`,
    );
  }

  parts.push(
    "Note: this 2D measurement captures the vertical drop only. True " +
    "clinical pronator drift also involves forearm rotation/pronation " +
    "as the arm drops — that rotation is NOT assessed here. Clinical " +
    "judgement required.",
  );

  return parts.join(" ");
}

function fmtDriftDir(cm: number | null): string {
  if (cm === null) return "not tracked";
  if (cm > 0.5) return `dropped ${cm.toFixed(1)} cm`;
  if (cm < -0.5) return `rose ${Math.abs(cm).toFixed(1)} cm`;
  return "held position";
}

// ─── Display helpers ────────────────────────────────────────────

export const PRONATOR_CLASSIFICATION_LABEL: Record<PronatorDriftClassification, string> = {
  normal:          "Normal",
  borderline:      "Borderline",
  positive_screen: "Positive screen",
};

export const PRONATOR_CLASSIFICATION_TONE: Record<PronatorDriftClassification, string> = {
  normal:          "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  borderline:      "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  positive_screen: "bg-red-500/10 text-red-700 dark:text-red-400",
};

// ─── Audio cues (Web Audio API) ─────────────────────────────────
//
// The hold is done with eyes closed — the patient cannot see any
// visual cue. We beep at start + end. Browsers gate audio behind a
// prior user gesture, so the first beep MUST happen after the
// operator's click on the Start button (which it does — Capture
// triggers `playStartBeep()` inside the click handler).

let _audioCtx: AudioContext | null = null;
function ctx(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (!_audioCtx) {
    const Ctor = window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    _audioCtx = new Ctor();
  }
  return _audioCtx;
}

function beep(freq: number, durationMs: number): void {
  const c = ctx();
  if (!c) return;
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.001, c.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.3, c.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + durationMs / 1000);
  osc.connect(gain);
  gain.connect(c.destination);
  osc.start();
  osc.stop(c.currentTime + durationMs / 1000 + 0.05);
}

/** Short low-pitched countdown tick. Used for the 3-2-1 prep. */
export function playCountdownBeep(): void { beep(440, 120); }
/** Distinct higher beep marking the START of the hold — patient
 *  should now be eyes-closed and holding the position. */
export function playStartBeep(): void { beep(880, 250); }
/** Lower, longer beep marking the END of the hold — patient may
 *  now open their eyes and lower their arms. */
export function playEndBeep(): void { beep(523, 500); }

// ─── Upload-mode API client ─────────────────────────────────────

interface PronatorDriftResponseDTO {
  success: boolean;
  data: PronatorDriftResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

function humanizeUploadError(raw: string | null): string {
  if (!raw) return "Analysis failed. Please try again.";
  const s = raw.toLowerCase();
  if (s.includes("poor_visibility")) {
    return (
      "Both arms aren't clearly visible in the recording. Re-record with " +
      "the patient facing the camera and both extended arms in frame."
    );
  }
  if (s.includes("too_short") || s.includes("too short")) {
    return (
      "Recording is too short. Please capture at least 5 seconds of the hold."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video quality too low. Please record at 24 fps or higher.";
  }
  if (s.includes("too long")) {
    return "Video is too long. Maximum 60 seconds.";
  }
  if (s.includes("file too large")) {
    return "File too large. Maximum 100 MB.";
  }
  return raw;
}

export async function analyzePronatorDriftUpload(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<PronatorDriftResult> {
  const form = new FormData();
  form.append("video", file, file.name || "pronator_drift.mp4");

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
    const res = await authedFetch("/api/analyze-pronator-drift", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail = typeof body.detail === "string"
        ? body.detail
        : `Pronator-drift analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as PronatorDriftResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
