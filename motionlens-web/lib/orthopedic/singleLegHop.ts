// D3 Single-Leg Hop — math, validity gates, classification, upload-
// mode API client.
//
// Test setup:
//   Patient stands SIDE-ON to the camera on the test leg, holds for
//   ~0.5 s (baseline lock), then hops FORWARD as far as comfortable
//   and lands on the SAME leg. Up to 3 trials in one recording per
//   leg. Patient repeats with the other leg in a second recording.
//
// Scale calibration:
//   Reuses CalibrationResult from lib/calibration/types — same
//   provider that drives Functional Reach. When null, the test runs
//   in relative-units mode (pixel distances only — no cm value, no
//   LSI classification).
//
// Per-leg analysis (one backend call per leg):
//   1. Standing baseline lock — first STANDING_HOLD_SAMPLES frames
//      where the test-side ankle.y is stable (foot grounded).
//   2. Takeoff = ankle.y rises >AIRBORNE_LIFT_FRAC_OF_LEG above
//      baseline (foot leaves the ground in image space).
//   3. Landing = ankle.y returns to within LANDED_BAND_FRAC_OF_LEG
//      of baseline AND has been airborne for >=MIN_AIRBORNE_FRAMES.
//   4. Hop distance = |heel.x_landing − heel.x_takeoff|, with
//      foot_index.x fallback when heel was occluded at either end.
//   5. Validity: contralateral ankle.y must NOT enter the grounded
//      band during the airborne window (single-leg gate). Distance
//      must exceed MIN_HOP_FOR_VALID.
//
// LSI (across legs) is computed CLIENT-SIDE after both per-leg calls
// return — see computeLSI() / classifyLSI() below.

import { authedFetch, AuthError } from "@/lib/auth";
import { type CalibrationResult } from "@/lib/calibration/types";

export type { CalibrationResult } from "@/lib/calibration/types";
export { pxToCm } from "@/lib/calibration/types";

// ─── Tuning constants (mirror the backend) ─────────────────────
export const VIS_THRESHOLD = 0.3;
export const SAMPLE_HZ = 30;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;
export const RECORDING_DURATION_SEC = 20;
export const STANDING_HOLD_SEC = 0.5;
export const STANDING_HOLD_SAMPLES = Math.round(
  STANDING_HOLD_SEC * SAMPLE_HZ,
);
export const MAX_TRIALS = 3;

/** Ankle.y must rise this fraction of leg-length above baseline to
 *  count as having left the ground. */
export const AIRBORNE_LIFT_FRAC_OF_LEG = 0.06;
/** Ankle.y must return within this fraction of leg-length of
 *  baseline to be confirmed landed. */
export const LANDED_BAND_FRAC_OF_LEG = 0.03;
/** Minimum airborne duration (sec) for a takeoff/landing pair to
 *  count as a real hop. */
export const MIN_AIRBORNE_SEC = 0.1;
export const MIN_AIRBORNE_SAMPLES = Math.round(MIN_AIRBORNE_SEC * SAMPLE_HZ);
/** Required quiet gap between successive trials (sec). */
export const MIN_TRIAL_GAP_SEC = 0.5;

/** Minimum calibrated hop distance (cm) for a trial to be valid. */
export const MIN_HOP_FOR_VALID_CM = 10.0;
/** Uncalibrated fallback (fraction of leg-length). */
export const MIN_HOP_FALLBACK_FRACTION_OF_LEG = 0.3;

/** Single-leg validity: contralateral foot must not touch ground
 *  for more than this many consecutive samples while airborne. */
export const CONTRALATERAL_TOUCH_GRACE_SAMPLES = 2;

// ─── LSI classification ────────────────────────────────────────
/** Standard ACL-clearance cutoffs. */
export const LSI_CLEARED_PCT = 90.0;
export const LSI_WARNING_PCT = 80.0;

export type Side = "left" | "right";
export type LSIClass = "cleared" | "warning" | "deficit" | "incomplete";

/** (weaker / stronger) × 100. Caller passes the SMALLER of the two
 *  best-valid distances as `weaker` and the larger as `stronger`.
 *  Returns null when either is missing or `stronger` is zero. */
export function computeLSI(
  weakerCm: number | null,
  strongerCm: number | null,
): number | null {
  if (
    weakerCm === null ||
    strongerCm === null ||
    !Number.isFinite(weakerCm) ||
    !Number.isFinite(strongerCm) ||
    strongerCm <= 0
  ) {
    return null;
  }
  return (weakerCm / strongerCm) * 100;
}

export function classifyLSI(lsiPct: number | null): LSIClass {
  if (lsiPct === null) return "incomplete";
  if (lsiPct >= LSI_CLEARED_PCT) return "cleared";
  if (lsiPct >= LSI_WARNING_PCT) return "warning";
  return "deficit";
}

// ─── DTO shape (match backend SingleLegHopResultDTO + SLHTrialDTO) ─

export interface Trial {
  trial_index: number;
  takeoff_frame_index: number;
  landing_frame_index: number;
  takeoff_t_ms: number;
  landing_t_ms: number;
  hop_distance_px: number;
  hop_distance_cm: number | null;
  valid: boolean;
  invalidation_reason: string | null;
}

export interface SingleLegHopResult {
  side_tested: Side;
  patient_height_cm: number | null;
  calibration: CalibrationResult | null;
  baseline_ankle_y_px: number;
  leg_length_px: number;
  trials: Trial[];
  best_valid_trial_index: number | null;
  best_valid_hop_px: number | null;
  best_valid_hop_cm: number | null;
  peak_screenshot_data_url: string | null;
  duration_seconds: number;
  termination: string;
  fps: number | null;
  total_frames: number | null;
  valid_frames: number | null;
  interpretation: string | null;
}

interface SingleLegHopResponseDTO {
  success: boolean;
  data: SingleLegHopResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

// ─── Combined two-leg result (client-side construction) ─────────

export interface SingleLegHopCombinedResult {
  left: SingleLegHopResult | null;
  right: SingleLegHopResult | null;
  /** Smaller of the two best_valid_hop_cm values, or null if either
   *  leg is missing a valid hop. */
  weaker_cm: number | null;
  /** Larger of the two best_valid_hop_cm values, or null if either
   *  leg is missing a valid hop. */
  stronger_cm: number | null;
  /** Which side was weaker — informational. */
  weaker_side: Side | null;
  lsi_pct: number | null;
  lsi_class: LSIClass;
  /** Calibration carried from whichever leg's response had one
   *  (they should be identical — same patient, same recording
   *  session — but if only one was calibrated we surface that). */
  calibration: CalibrationResult | null;
}

export function buildCombinedResult(
  left: SingleLegHopResult | null,
  right: SingleLegHopResult | null,
): SingleLegHopCombinedResult {
  const lBest =
    left && left.best_valid_hop_cm !== null ? left.best_valid_hop_cm : null;
  const rBest =
    right && right.best_valid_hop_cm !== null ? right.best_valid_hop_cm : null;
  let weakerCm: number | null = null;
  let strongerCm: number | null = null;
  let weakerSide: Side | null = null;
  if (lBest !== null && rBest !== null) {
    if (lBest <= rBest) {
      weakerCm = lBest;
      strongerCm = rBest;
      weakerSide = "left";
    } else {
      weakerCm = rBest;
      strongerCm = lBest;
      weakerSide = "right";
    }
  }
  const lsiPct = computeLSI(weakerCm, strongerCm);
  const lsiClass = classifyLSI(lsiPct);
  const calibration =
    (left && left.calibration) || (right && right.calibration) || null;
  return {
    left,
    right,
    weaker_cm: weakerCm,
    stronger_cm: strongerCm,
    weaker_side: weakerSide,
    lsi_pct: lsiPct,
    lsi_class: lsiClass,
    calibration,
  };
}

// ─── Upload-mode API client ────────────────────────────────────

function humanizeUploadError(raw: string | null | undefined): string {
  const s = (raw || "").toString().toLowerCase();
  if (!s) return "Single-Leg Hop analysis failed.";
  if (s.includes("poor_visibility")) {
    return (
      "MediaPipe could not see the test leg clearly. Re-record with the " +
      "full body in frame (head + torso + both feet)."
    );
  }
  if (s.includes("no_baseline")) {
    return (
      "Patient did not stand still on the test leg before hopping. " +
      "Re-record with a ~1 s static stance on the test leg before the " +
      "first hop."
    );
  }
  if (s.includes("no_hops_detected")) {
    return (
      "No hops were detected. Re-record with the patient hopping forward " +
      "on the test leg and landing on the same leg."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video frame rate too low — record at 24+ FPS.";
  }
  if (s.includes("too short")) return "Video is too short.";
  if (s.includes("too long")) return "Video is too long.";
  if (s.includes("file too large")) return "File too large.";
  return raw || "Single-Leg Hop analysis failed.";
}

/** Upload a Single-Leg Hop clip (one leg per call). Backend returns
 *  per-leg analysis; LSI is computed CLIENT-side once both per-leg
 *  results are in. */
export async function analyzeSingleLegHopUpload(
  file: File,
  side: Side,
  calibration: CalibrationResult | null,
  patientHeightCm: number | null,
  onProgress?: (pct: number) => void,
): Promise<SingleLegHopResult> {
  const form = new FormData();
  form.append("video", file, file.name || "single_leg_hop.mp4");
  form.append("side", side);
  if (calibration) {
    form.append("calibration", JSON.stringify(calibration));
  }
  if (
    patientHeightCm !== null &&
    Number.isFinite(patientHeightCm) &&
    patientHeightCm > 0
  ) {
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
    const res = await authedFetch("/api/analyze-single-leg-hop", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = await res
        .json()
        .catch(() => ({ detail: `HTTP ${res.status}` }));
      const detail =
        typeof body.detail === "string"
          ? body.detail
          : `Single-Leg Hop analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as SingleLegHopResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
