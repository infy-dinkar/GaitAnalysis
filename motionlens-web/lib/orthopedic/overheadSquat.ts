// B2 Overhead Squat (NASM / FMS-style) — types, constants,
// upload-mode API client.
//
// Test setup:
//   Patient stands FRONTAL to the camera, feet shoulder-width apart,
//   arms held straight overhead (biceps by ears). Performs 3-5 slow
//   overhead squats to about parallel depth, returning fully to
//   standing between reps. Session is scored against a 7-item
//   frontal-plane checklist (5 measurable + 2 not_assessed — torso
//   lean and heel rise need sagittal / feet close-up views).
//
// Architecture mirrors D2 Tuck Jump — thin HTTP client + types,
// NO client-side math. Both live + upload POST video to the same
// `/api/analyze-overhead-squat` endpoint.

import { authedFetch, AuthError } from "@/lib/auth";
import { type CalibrationResult } from "@/lib/calibration/types";

export type { CalibrationResult } from "@/lib/calibration/types";
export { pxToCm } from "@/lib/calibration/types";

// ─── Tuning constants (mirror the backend) ─────────────────────
export const VIS_THRESHOLD = 0.15;
export const SAMPLE_HZ = 30;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;
/** Recorded window for the live overhead-squat session. */
export const RECORDING_DURATION_SEC = 20;
/** Target session length — patient performs 3-5 slow reps. */
export const TARGET_SESSION_SEC = 15;
export const STANDING_HOLD_SEC = 0.3;
export const STANDING_HOLD_SAMPLES = Math.round(
  STANDING_HOLD_SEC * SAMPLE_HZ,
);

export const DESCENT_MIN_FRAC_OF_LEG = 0.05;
export const DEPTH_TARGET_FRAC_OF_LEG = 0.20;
export const VALGUS_KFPPA_FAIL_DEG = 12.0;
export const PELVIC_TILT_FAIL_FRAC_OF_HIPSPAN = 0.10;
export const ARM_OVERHEAD_MIN_FRAC_OF_TRUNK = 0.30;

// ─── DTO shape (matches backend OverheadSquatResultDTO) ────────
export type OverheadSquatChecklistStatus = "pass" | "fail" | "not_assessed";

export interface OverheadSquatChecklistItem {
  index: number;
  label: string;
  status: OverheadSquatChecklistStatus;
  detail: string | null;
}

export interface OverheadSquatRep {
  rep_index: number;
  descent_start_frame_index: number;
  bottom_frame_index: number;
  return_frame_index: number;
  descent_start_t_ms: number;
  bottom_t_ms: number;
  return_t_ms: number;
  depth_px: number;
  depth_cm: number | null;
  depth_frac_of_leg: number;
  bottom_kfppa_left_deg: number;
  bottom_kfppa_right_deg: number;
  bottom_kfppa_worse_deg: number;
  bottom_pelvic_tilt_px: number;
  bottom_pelvic_tilt_frac: number;
  bottom_ank_spread_px: number;
  bottom_ank_spread_ratio: number;
  bottom_l_arm_overhead_frac: number | null;
  bottom_r_arm_overhead_frac: number | null;
  bottom_worst_arm_overhead_frac: number | null;
}

export type OverheadSquatClassification = "good" | "moderate" | "poor";

export interface OverheadSquatResult {
  patient_height_cm: number | null;
  calibration: CalibrationResult | null;
  baseline_hip_y_px: number;
  baseline_shoulder_y_px: number;
  baseline_wrist_y_px: number | null;
  baseline_ank_spread_px: number;
  baseline_hip_span_px: number;
  leg_length_px: number;
  trunk_length_px: number;
  reps: OverheadSquatRep[];
  rep_count: number;
  mean_depth_frac: number;
  max_depth_frac: number;
  max_depth_cm: number | null;
  mean_valgus_worse_deg: number;
  max_valgus_worse_deg: number;
  max_pelvic_tilt_frac: number;
  mean_ank_spread_ratio: number;
  min_arm_overhead_frac: number | null;
  duration_seconds: number;
  checklist: OverheadSquatChecklistItem[];
  measurable_fails: number;
  classification: OverheadSquatClassification;
  peak_screenshot_data_url: string | null;
  fps: number | null;
  total_frames: number | null;
  valid_frames: number | null;
  interpretation: string | null;
}

interface OverheadSquatResponseDTO {
  success: boolean;
  data: OverheadSquatResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

// ─── Upload-mode API client ────────────────────────────────────
function humanizeUploadError(raw: string | null | undefined): string {
  const s = (raw || "").toString().toLowerCase();
  if (!s) return "Overhead Squat analysis failed.";
  if (s.includes("poor_visibility")) {
    return (
      "MediaPipe could not see the patient clearly. Re-record with the " +
      "full body FRONTAL to the camera (head, arms overhead, torso, both feet)."
    );
  }
  if (s.includes("no_baseline")) {
    return (
      "Patient did not stand still with arms overhead before the first " +
      "squat. Re-record with a ~1 s static stance holding arms straight " +
      "overhead before the first rep."
    );
  }
  if (s.includes("no_reps_detected")) {
    return (
      "No squat reps were detected. Re-record with the patient performing " +
      "3-5 slow overhead squats to about parallel depth, returning fully " +
      "to standing between reps."
    );
  }
  if (s.includes("frame rate too low") || s.includes("fps")) {
    return "Video frame rate too low — record at 24+ FPS.";
  }
  if (s.includes("too short")) return "Video is too short.";
  if (s.includes("too long")) return "Video is too long.";
  if (s.includes("file too large")) return "File too large.";
  return raw || "Overhead Squat analysis failed.";
}

/** Upload an overhead-squat clip. 3-5 reps, arms overhead. */
export async function analyzeOverheadSquatUpload(
  file: File,
  calibration: CalibrationResult | null,
  patientHeightCm: number | null,
  onProgress?: (pct: number) => void,
): Promise<OverheadSquatResult> {
  const form = new FormData();
  form.append("video", file, file.name || "overhead_squat.mp4");
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
    const res = await authedFetch("/api/analyze-overhead-squat", {
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
          : `Overhead Squat analysis failed (${res.status})`;
      throw new AuthError(humanizeUploadError(detail), res.status);
    }
    const payload = (await res.json()) as OverheadSquatResponseDTO;
    if (!payload.success || !payload.data) {
      throw new AuthError(humanizeUploadError(payload.error), 500);
    }
    return payload.data;
  } finally {
    if (pulseHandle !== null) clearTimeout(pulseHandle);
    onProgress?.(100);
  }
}
