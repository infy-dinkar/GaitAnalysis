// TUG (Timed Up and Go) DTOs — mirror tug_models.py.
//
// Talks to POST /api/analyze-tug, which accepts a recorded video
// + optional patient_age and returns the full 5-phase analysis.
//
// Unlike the live-detection orthopedic tests, TUG processing happens
// on the backend (MediaPipe BlazePose Full, 33 keypoints) so we
// only need result-type definitions on the frontend.

import { authedFetch, AuthError } from "@/lib/auth";

// ─── Result types ────────────────────────────────────────────────
export type TUGClassification =
  | "normal"
  | "mild_fall_risk"
  | "elevated_fall_risk"
  | "significant_fall_risk";

export type TUGPhaseName =
  | "sit_to_stand"
  | "walk_out"
  | "turn"
  | "walk_back"
  | "stand_to_sit";

export interface TUGPhase {
  phase: TUGPhaseName;
  duration_sec: number;
  start_frame: number;
  end_frame: number;
  step_count: number | null;
  step_length_l_px: number | null;
  step_length_r_px: number | null;
  cadence_steps_per_min: number | null;
  walking_speed_mps: number | null;
  truncated: boolean;
}

export interface TUGKeyFrame {
  label:
    | "test_start"
    | "end_of_sit_to_stand"
    | "start_of_turn"
    | "end_of_turn"
    | "test_end";
  frame_index: number;
  image_data_url: string;
}

export interface TUGFlag {
  code:
    | "turn_time_excessive"
    | "turn_steps_excessive"
    | "phase_truncated"
    | "turn_undetected"
    | "no_strikes_detected"
    | "test_too_fast";
  severity: "info" | "warning" | "concern";
  message: string;
}

export interface TUGResult {
  total_time_sec: number;
  classification: TUGClassification;
  sit_to_stand: TUGPhase;
  walk_out: TUGPhase;
  turn: TUGPhase;
  walk_back: TUGPhase;
  stand_to_sit: TUGPhase;
  flags: TUGFlag[];
  patient_age: number | null;
  age_norm_threshold_sec: number | null;
  age_norm_passed: boolean | null;
  interpretation: string;
  key_frames: TUGKeyFrame[];
  fps: number;
  total_frames: number;
}

export interface TUGResponseDTO {
  success: boolean;
  data: TUGResult | null;
  error: string | null;
  fps_warning: string | null;
  duration_warning: string | null;
}

// ─── Display labels (single source of truth) ────────────────────
export const TUG_CLASSIFICATION_LABEL: Record<TUGClassification, string> = {
  normal: "Normal mobility",
  mild_fall_risk: "Mild fall risk",
  elevated_fall_risk: "Elevated fall risk",
  significant_fall_risk: "Significant fall risk / impaired mobility",
};

// Tone classes pick up the design tokens used elsewhere on the
// dashboard. Green / amber / orange / red ladder mirrors the spec.
export const TUG_CLASSIFICATION_TONE: Record<TUGClassification, string> = {
  normal: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  mild_fall_risk: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  elevated_fall_risk: "bg-orange-500/10 text-orange-700 dark:text-orange-400",
  significant_fall_risk: "bg-red-500/10 text-red-700 dark:text-red-400",
};

export const TUG_PHASE_LABEL: Record<TUGPhaseName, string> = {
  sit_to_stand: "Sit-to-stand",
  walk_out: "Walk-out",
  turn: "Turn",
  walk_back: "Walk-back",
  stand_to_sit: "Stand-to-sit",
};

export const TUG_PHASE_COLOR: Record<TUGPhaseName, string> = {
  sit_to_stand: "#0ea5e9",  // sky
  walk_out: "#10b981",       // emerald
  turn: "#f59e0b",           // amber
  walk_back: "#8b5cf6",      // violet
  stand_to_sit: "#ef4444",   // red
};

export const TUG_KEY_FRAME_LABEL: Record<TUGKeyFrame["label"], string> = {
  test_start: "Test start (seated)",
  end_of_sit_to_stand: "Standing — end of sit-to-stand",
  start_of_turn: "At marker — start of turn",
  end_of_turn: "Facing back — end of turn",
  test_end: "Test end (seated)",
};

// FastAPI returns Pydantic validation errors as `{detail: [{loc, msg,
// type, ...}, ...]}`. Calling String() on that array yields the
// useless `"[object Object]"` — flatten it into a readable message.
function formatErrorDetail(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.length > 0) return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        if (!d || typeof d !== "object") return String(d);
        const obj = d as Record<string, unknown>;
        const loc = Array.isArray(obj.loc) ? obj.loc.join(".") : "";
        const msg = typeof obj.msg === "string" ? obj.msg : JSON.stringify(obj);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch {
      // fall through
    }
  }
  return `TUG analysis failed (${status})`;
}

// ─── API call ────────────────────────────────────────────────────
export async function analyzeTUG(
  videoBlob: Blob,
  patientAge: number | null,
  recordingDurationMs: number | null,
): Promise<TUGResponseDTO> {
  const form = new FormData();
  form.append("video", videoBlob, "tug.webm");
  if (patientAge !== null) {
    form.append("patient_age", String(patientAge));
  }
  // Recording-mode WebM containers from MediaRecorder often have
  // broken / missing duration metadata in their headers — sending
  // the client-measured wall-clock duration lets the backend compute
  // FPS as a fallback when cv2's CAP_PROP_FPS probe returns 0.
  if (recordingDurationMs !== null) {
    form.append("recording_duration_ms", String(Math.round(recordingDurationMs)));
  }
  const res = await authedFetch("/api/analyze-tug", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new AuthError(formatErrorDetail(body.detail, res.status), res.status);
  }
  return (await res.json()) as TUGResponseDTO;
}
