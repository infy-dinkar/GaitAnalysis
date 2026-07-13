// D2 Tuck Jump (Myer's Tuck Jump Assessment) — types, constants,
// upload-mode API client.
//
// Test setup:
//   Patient stands upright, FRONTAL to the camera (not side-on — the
//   frontal view is required for Myer's valgus / symmetry / footprint
//   items). Patient performs ~10 s of continuous tuck jumps —
//   knees pulled up to the chest at apex, land on BOTH feet,
//   immediately re-jump. Session is scored against Myer's 10-item
//   checklist (see tuck_jump_engine.py).
//
// Architecture mirrors D4 Counter-Movement Jump — same multipart
// upload path, same optional CalibrationResult + patient_height_cm,
// same "physics" flight-time cross-check helper.
//
// Reuse note (per project convention):
//   • `physicsJumpHeightCm` is IMPORTED from counterMovementJump.ts
//     rather than re-declared, so the two tests share a single source
//     of truth for the h = g·t²/8 constant.

import { authedFetch, AuthError } from "@/lib/auth";
import { type CalibrationResult } from "@/lib/calibration/types";
import {
  physicsJumpHeightCm as cmjPhysicsJumpHeightCm,
  GRAVITY_M_PER_S2 as CMJ_GRAVITY_M_PER_S2,
} from "@/lib/orthopedic/counterMovementJump";

export type { CalibrationResult } from "@/lib/calibration/types";
export { pxToCm } from "@/lib/calibration/types";

// ─── Tuning constants (mirror the backend) ─────────────────────
export const VIS_THRESHOLD = 0.15;
export const SAMPLE_HZ = 30;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;
/** Recorded window for the live-mode continuous tuck-jump session. */
export const RECORDING_DURATION_SEC = 15;
/** Myer's target session length. */
export const TARGET_SESSION_SEC = 10;
export const STANDING_HOLD_SEC = 0.3;
export const STANDING_HOLD_SAMPLES = Math.round(
  STANDING_HOLD_SEC * SAMPLE_HZ,
);

export const AIRBORNE_LIFT_FRAC_OF_LEG = 0.06;
export const LANDED_BAND_FRAC_OF_LEG = 0.03;
export const MIN_AIRBORNE_SEC = 0.08;
export const MIN_TRIAL_GAP_SEC = 0.15;

/** Any grounded gap longer than this fails Myer item 8 (pause). */
export const PAUSE_GAP_SEC = 0.6;

// Re-export the physics helper from CMJ so both tests share one
// implementation. This keeps counterMovementJump.ts byte-identical
// while still letting Tuck Jump's report import via
// `@/lib/orthopedic/tuckJump`.
export const GRAVITY_M_PER_S2 = CMJ_GRAVITY_M_PER_S2;

/** Projectile-motion height (cm) from flight time alone. */
export function physicsJumpHeightCm(flightTimeSec: number): number {
  return cmjPhysicsJumpHeightCm(flightTimeSec);
}

// ─── DTO shape (matches backend TuckJumpResultDTO) ─────────────
export type TuckJumpChecklistStatus = "pass" | "fail" | "not_assessed";

export interface TuckJumpChecklistItem {
  index: number;
  label: string;
  status: TuckJumpChecklistStatus;
  detail: string | null;
}

export interface TuckJumpJump {
  jump_index: number;
  takeoff_frame_index: number;
  apex_frame_index: number;
  landing_frame_index: number;
  takeoff_t_ms: number;
  landing_t_ms: number;
  flight_time_sec: number;
  jump_height_px: number;
  jump_height_cm: number | null;
  physics_height_cm: number;
  landing_kfppa_left_deg: number;
  landing_kfppa_right_deg: number;
  landing_kfppa_worse_deg: number;
  landing_ank_spread_px: number;
  landing_ank_spread_ratio: number;
  apex_l_thigh_rise_px: number;
  apex_r_thigh_rise_px: number;
  takeoff_side_delta_ms: number;
  landing_side_delta_ms: number;
  grounded_since_prev_ms: number | null;
  landing_ank_left_x_px: number;
  landing_ank_right_x_px: number;
}

export type TuckJumpClassification = "good" | "moderate" | "poor";

export interface TuckJumpResult {
  patient_height_cm: number | null;
  calibration: CalibrationResult | null;
  baseline_hip_y_px: number;
  baseline_left_ankle_y_px: number;
  baseline_right_ankle_y_px: number;
  baseline_ank_spread_px: number;
  baseline_shoulder_hip_span_px: number;
  leg_length_px: number;
  jumps: TuckJumpJump[];
  jump_count: number;
  mean_jump_height_px: number;
  mean_jump_height_cm: number | null;
  mean_valgus_worse_deg: number;
  max_valgus_worse_deg: number;
  height_fade_frac: number;
  valgus_growth_deg: number;
  footprint_drift_frac: number;
  pause_gap_max_ms: number;
  duration_seconds: number;
  checklist: TuckJumpChecklistItem[];
  measurable_fails: number;
  classification: TuckJumpClassification;
  peak_screenshot_data_url: string | null;
  fps: number | null;
  total_frames: number | null;
  valid_frames: number | null;
  interpretation: string | null;
}

interface TuckJumpResponseDTO {
  success: boolean;
  data: TuckJumpResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

// ─── Upload-mode API client ────────────────────────────────────
function humanizeUploadError(raw: string | null | undefined): string {
  const s = (raw || "").toString().toLowerCase();
  if (!s) return "Tuck Jump analysis failed.";
  if (s.includes("poor_visibility")) {
    return (
      "MediaPipe could not see the patient clearly. Re-record with the " +
      "full body FRONTAL to the camera (head, torso, both feet in frame)."
    );
  }
  if (s.includes("no_baseline")) {
    return (
      "Patient did not stand still before starting. Re-record with a " +
      "~1 s static standing pose FRONTAL to the camera before the first " +
      "tuck jump."
    );
  }
  if (s.includes("no_jumps_detected")) {
    return (
      "No tuck jumps were detected. Re-record with continuous tuck " +
      "jumps — both feet leave the ground together, knees pull to chest, " +
      "both feet land together."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video frame rate too low — record at 24+ FPS.";
  }
  if (s.includes("too short")) return "Video is too short.";
  if (s.includes("too long")) return "Video is too long.";
  if (s.includes("file too large")) return "File too large.";
  return raw || "Tuck Jump analysis failed.";
}

/** Upload a Tuck Jump clip. Continuous ~10 s session, both legs. */
export async function analyzeTuckJumpUpload(
  file: File,
  calibration: CalibrationResult | null,
  patientHeightCm: number | null,
  onProgress?: (pct: number) => void,
): Promise<TuckJumpResult> {
  const form = new FormData();
  form.append("video", file, file.name || "tuck_jump.mp4");
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
    const res = await authedFetch("/api/analyze-tuck-jump", {
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
          : `Tuck Jump analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as TuckJumpResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
