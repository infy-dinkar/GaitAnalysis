// D4 Counter-Movement Jump — types, constants, upload-mode API client.
//
// Test setup:
//   Patient stands upright, side-on to the camera. After a brief
//   static stance the patient dips into a quick squat (the
//   counter-movement) then jumps STRAIGHT UP as high as possible
//   and lands on BOTH feet. Up to 3 jumps in a single recording.
//
// Architecture mirrors D3 Single-Leg Hop's frontend client but
// without the per-leg / LSI structure:
//   • Single recording (no side parameter).
//   • Primary outcome = jump height (cm via height calibration).
//   • Secondary = flight time (s) + a gravity-based physics cross-
//     check (h = g·t²/8) — independent of pose calibration so it
//     stays useful even when the cm conversion fails.
//
// Scale calibration:
//   Reuses CalibrationResult from lib/calibration/types — same
//   provider that drives Functional Reach + Single-Leg Hop. When
//   null, jump heights are pixel-only.

import { authedFetch, AuthError } from "@/lib/auth";
import { type CalibrationResult } from "@/lib/calibration/types";

export type { CalibrationResult } from "@/lib/calibration/types";
export { pxToCm } from "@/lib/calibration/types";

// ─── Tuning constants (mirror the backend) ─────────────────────
export const VIS_THRESHOLD = 0.3;
export const SAMPLE_HZ = 30;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;
export const RECORDING_DURATION_SEC = 25;
export const STANDING_HOLD_SEC = 0.3;
export const STANDING_HOLD_SAMPLES = Math.round(
  STANDING_HOLD_SEC * SAMPLE_HZ,
);
export const MAX_TRIALS = 3;

export const AIRBORNE_LIFT_FRAC_OF_LEG = 0.06;
export const LANDED_BAND_FRAC_OF_LEG = 0.03;
export const MIN_AIRBORNE_SEC = 0.1;
export const MIN_AIRBORNE_SAMPLES = Math.round(MIN_AIRBORNE_SEC * SAMPLE_HZ);
export const MIN_TRIAL_GAP_SEC = 0.5;

export const MIN_JUMP_FOR_VALID_CM = 5.0;
export const MIN_JUMP_FALLBACK_FRACTION_OF_LEG = 0.05;

export const GRAVITY_M_PER_S2 = 9.81;

/** Projectile-motion height (cm) from flight time alone. */
export function physicsJumpHeightCm(flightTimeSec: number): number {
  if (!Number.isFinite(flightTimeSec) || flightTimeSec <= 0) return 0;
  const h_m = (GRAVITY_M_PER_S2 * flightTimeSec * flightTimeSec) / 8;
  return h_m * 100;
}

// ─── DTO shape (matches backend CMJResultDTO + CMJTrialDTO) ─────

export interface CMJTrial {
  trial_index: number;
  takeoff_frame_index: number;
  apex_frame_index: number;
  landing_frame_index: number;
  takeoff_t_ms: number;
  landing_t_ms: number;
  flight_time_sec: number;
  jump_height_px: number;
  jump_height_cm: number | null;
  physics_height_cm: number;
  valid: boolean;
  invalidation_reason: string | null;
}

export interface CMJResult {
  patient_height_cm: number | null;
  calibration: CalibrationResult | null;
  baseline_hip_y_px: number;
  baseline_left_ankle_y_px: number;
  baseline_right_ankle_y_px: number;
  leg_length_px: number;
  trials: CMJTrial[];
  best_valid_trial_index: number | null;
  best_valid_jump_px: number | null;
  best_valid_jump_cm: number | null;
  best_valid_flight_sec: number | null;
  mean_valid_jump_cm: number | null;
  mean_valid_flight_sec: number | null;
  peak_screenshot_data_url: string | null;
  duration_seconds: number;
  termination: string;
  fps: number | null;
  total_frames: number | null;
  valid_frames: number | null;
  interpretation: string | null;
}

interface CMJResponseDTO {
  success: boolean;
  data: CMJResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

// ─── Upload-mode API client ────────────────────────────────────

function humanizeUploadError(raw: string | null | undefined): string {
  const s = (raw || "").toString().toLowerCase();
  if (!s) return "CMJ analysis failed.";
  if (s.includes("poor_visibility")) {
    return (
      "MediaPipe could not see the patient clearly. Re-record with the " +
      "full body in frame (head, torso, both feet)."
    );
  }
  if (s.includes("no_baseline")) {
    return (
      "Patient did not stand still before jumping. Re-record with a " +
      "~1 s static standing pose before the first counter-movement jump."
    );
  }
  if (s.includes("no_jumps_detected")) {
    return (
      "No jumps were detected. Re-record with the patient performing a " +
      "clear counter-movement jump — both feet leave the ground together, " +
      "both feet land together."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video frame rate too low — record at 24+ FPS.";
  }
  if (s.includes("too short")) return "Video is too short.";
  if (s.includes("too long")) return "Video is too long.";
  if (s.includes("file too large")) return "File too large.";
  return raw || "CMJ analysis failed.";
}

/** Upload a CMJ clip. Both legs in one recording — no side param. */
export async function analyzeCounterMovementJumpUpload(
  file: File,
  calibration: CalibrationResult | null,
  patientHeightCm: number | null,
  onProgress?: (pct: number) => void,
): Promise<CMJResult> {
  const form = new FormData();
  form.append("video", file, file.name || "counter_movement_jump.mp4");
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
    const res = await authedFetch("/api/analyze-counter-movement-jump", {
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
          : `CMJ analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as CMJResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
