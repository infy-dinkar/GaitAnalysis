// SPPB Component 1 (Balance) — backend MediaPipe API client.
//
// POSTs a single recorded clip (all 3 stages performed sequentially)
// to /api/sppb/balance and returns the same per-stage shape that
// the existing buildBalanceComponent() consumes. SPPB scoring math
// downstream is unchanged.
//
// This is the ONLY SPPB sub-component that uses backend pose
// detection. Components 2 (gait speed) and 3 (chair stand) keep
// running the existing MoveNet live path on the frontend.
//
// MediaRecorder produces WebM clips whose containers often lack a
// duration header — the same problem TUG had. The backend reuses
// tug_engine._ensure_decodable_video to repair those before the
// cv2 FPS probe, so the client sends `recording_duration_ms`
// alongside the file.
import { authedFetch, AuthError } from "@/lib/auth";
import type { StageResult } from "@/lib/orthopedic/fourStageBalance";

export interface SPPBBalanceStagesDTO {
  "1"?: StageResult;
  "2"?: StageResult;
  "3"?: StageResult;
}

export interface QuartileStats {
  count: number;
  min: number | null;
  p25: number | null;
  median: number | null;
  p75: number | null;
  max: number | null;
}

export interface SPPBBalanceDiagnostics {
  frame_classification_counts: {
    stage_1: number;
    stage_2: number;
    stage_3: number;
    unclassified: number;
  };
  visible_foot_frames: number;
  visible_foot_ratio: number;
  min_run_frames: number;
  smooth_window: number;
  body_h_failed_count?: number;
  geometry_unmatched_count?: number;
  body_h_px?: QuartileStats;
  stable_body_h_px?: number;
  dx_heel_n?: QuartileStats;
  dy_heel_n?: QuartileStats;
  dx_tandem_n?: QuartileStats;
  frame_width?: number;
  frame_height?: number;
  longest_runs_per_stage?: {
    "1": { frames: number; seconds: number };
    "2": { frames: number; seconds: number };
    "3": { frames: number; seconds: number };
  };
  thresholds?: {
    stage1_x_min: number;
    stage1_x_max: number;
    stage_dy_tight: number;
    stage_dy_med_min: number;
    stage_dy_med_max: number;
    stage_dy_large: number;
    tandem_x: number;
    semi_x: number;
  };
}

export interface SPPBBalanceAnalysisData {
  fps: number;
  total_frames: number;
  stages: SPPBBalanceStagesDTO;
  diagnostics?: SPPBBalanceDiagnostics;
}

export interface SPPBBalanceResponseDTO {
  success: boolean;
  data: SPPBBalanceAnalysisData | null;
  error: string | null;
}

/** Convert the backend's string-keyed stages map into the numeric-
 *  keyed shape that buildBalanceComponent() (existing SPPB scorer)
 *  expects. */
export function normaliseBalanceStages(
  raw: SPPBBalanceStagesDTO,
): { 1?: StageResult; 2?: StageResult; 3?: StageResult } {
  return {
    1: raw["1"],
    2: raw["2"],
    3: raw["3"],
  };
}

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
  return `SPPB balance analysis failed (${status})`;
}

export async function analyzeSPPBBalance(
  videoBlob: Blob,
  recordingDurationMs: number | null,
): Promise<SPPBBalanceResponseDTO> {
  const form = new FormData();
  const filename = videoBlob instanceof File ? videoBlob.name : "sppb-balance.webm";
  form.append("video", videoBlob, filename);
  if (recordingDurationMs !== null && recordingDurationMs > 0) {
    form.append("recording_duration_ms", String(Math.round(recordingDurationMs)));
  }
  const res = await authedFetch("/api/sppb/balance", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new AuthError(formatErrorDetail(body.detail, res.status), res.status);
  }
  return (await res.json()) as SPPBBalanceResponseDTO;
}
