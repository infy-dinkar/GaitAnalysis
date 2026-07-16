// Squat (Lateral) — sagittal-plane physio squat: HTTP client + types.
//
// Follows the TJA / Overhead-Squat recipe: thin frontend, all math
// lives in the Python engine `squat_lateral_engine.py`. Live and
// upload modes POST video (multipart) to the same endpoint:
// `/api/analyze-squat-lateral`.

import { authedFetch, AuthError } from "@/lib/auth";
import { type CalibrationResult } from "@/lib/calibration/types";

export type { CalibrationResult } from "@/lib/calibration/types";
export { pxToCm } from "@/lib/calibration/types";

// ─── Tuning constants (mirror the backend) ─────────────────────
export const VIS_THRESHOLD = 0.15;
export const SAMPLE_HZ = 30;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;
/** Recorded window for the live-mode squat session (~5 slow reps). */
export const RECORDING_DURATION_SEC = 28;
/** Suggested session length shown to the operator. */
export const TARGET_SESSION_SEC = 20;
export const STANDING_HOLD_SEC = 0.3;
export const STANDING_HOLD_SAMPLES = Math.round(
  STANDING_HOLD_SEC * SAMPLE_HZ,
);

// ─── DTO shape (matches backend SquatLateralResultDTO) ─────────
export type SquatLateralSide = "left" | "right";
export type SquatLateralClassification =
  | "good"
  | "moderate"
  | "poor"
  | "insufficient_data";

export interface SquatLateralValgusNote {
  status: "not_assessed";
  reason: string;
}

export interface SquatLateralCaveat {
  code: string;
  label: string;
  detail: string | null;
}

export interface SquatLateralRep {
  rep_index: number;
  bottom_frame_index: number;
  bottom_t_ms: number;
  peak_knee_flexion_deg: number | null;
  peak_hip_flexion_deg: number | null;
  trunk_lean_deg: number | null;
  hip_knee_ratio: number | null;
  heel_rise: boolean;
  heel_rise_px: number;
}

export interface SquatLateralAngleTrace {
  t_ms: number[];
  knee: (number | null)[];
  hip: (number | null)[];
  bottom_frame_indices: number[];
  bottom_t_ms: number[];
}

export interface SquatLateralResult {
  side: SquatLateralSide;
  patient_height_cm: number | null;
  calibration: CalibrationResult | null;
  baseline_hip_y_px: number;
  baseline_heel_y_px: number;
  leg_length_px: number;
  heel_rise_threshold_px: number;
  reps: SquatLateralRep[];
  rep_count: number;
  peak_knee_flexion_deg: number | null;
  peak_hip_flexion_deg: number | null;
  trunk_lean_deg: number | null;
  hip_knee_ratio: number | null;
  heel_rise: boolean;
  any_heel_rise: boolean;
  deepest_rep_index: number;
  mean_peak_knee_flexion_deg: number | null;
  mean_trunk_lean_deg: number | null;
  classification: SquatLateralClassification;
  guard_reason?: string | null;
  valgus: SquatLateralValgusNote;
  caveats: SquatLateralCaveat[];
  peak_screenshot_data_url: string | null;
  angle_trace: SquatLateralAngleTrace;
  duration_seconds: number;
  fps: number | null;
  total_frames: number | null;
  valid_frames: number | null;
  interpretation: string | null;
}

interface SquatLateralResponseDTO {
  success: boolean;
  data: SquatLateralResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

// ─── Upload-mode API client ────────────────────────────────────
function humanizeUploadError(raw: string | null | undefined): string {
  const s = (raw || "").toString().toLowerCase();
  if (!s) return "Squat analysis failed.";
  if (s.includes("poor_visibility")) {
    return (
      "MediaPipe could not see the near-side leg clearly. Re-record "
      + "side-on to the camera with the FULL body in frame."
    );
  }
  if (s.includes("no_baseline")) {
    return (
      "Patient did not stand still before the first rep. Re-record "
      + "with ~1 s of static upright stance before starting the squats."
    );
  }
  if (s.includes("no_reps_detected") || s.includes("no_valid_reps")) {
    return (
      "No squat reps were detected. Re-record with the patient "
      + "performing 3-6 slow squats, returning fully to standing "
      + "between reps."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video frame rate too low — record at 24+ FPS.";
  }
  if (s.includes("too short")) return "Video is too short.";
  if (s.includes("too long")) return "Video is too long.";
  if (s.includes("file too large")) return "File too large.";
  return raw || "Squat analysis failed.";
}

/** Upload a lateral-squat clip. */
export async function analyzeSquatLateralUpload(
  file: File,
  side: SquatLateralSide,
  calibration: CalibrationResult | null,
  patientHeightCm: number | null,
  onProgress?: (pct: number) => void,
): Promise<SquatLateralResult> {
  const form = new FormData();
  form.append("video", file, file.name || "squat_lateral.mp4");
  form.append("side", side);
  if (calibration) {
    form.append("calibration", JSON.stringify(calibration));
  }
  if (
    patientHeightCm !== null
    && Number.isFinite(patientHeightCm)
    && patientHeightCm > 0
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
    const res = await authedFetch("/api/analyze-squat-lateral", {
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
          : `Squat analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as SquatLateralResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
