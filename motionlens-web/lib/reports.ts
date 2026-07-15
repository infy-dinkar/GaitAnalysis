// Authenticated API client for /api/reports + /api/patients/{id}/reports.

import { authedFetch, AuthError } from "@/lib/auth";

// ─── DTOs (mirror report_models.py) ──────────────────────────────
export interface ReportSummaryDTO {
  id: string;
  patient_id: string;
  doctor_id: string;
  module:
    | "gait"
    | "biomech"
    | "posture"
    | "trendelenburg"
    | "single_leg_squat"
    | "sit_to_stand"
    | "chair_stand_30s"
    | "single_leg_stance"
    | "four_stage_balance"
    | "tug"
    | "sppb"
    | "slr"
    | "ake"
    | "modified_thomas"
    | "forward_lunge"
    | "sts_quality"
    | "tandem_walk"
    | "pronator_drift"
    | "functional_reach"
    | "single_leg_hop"
    | "counter_movement_jump"
    | "tuck_jump"
    | "overhead_squat"
    | "squat_lateral"
    | "rehab";
  body_part: string | null;
  movement: string | null;
  side: string | null;
  created_at: string;
}

// Per-keypoint shape stored on posture sessions. Matches the
// `Keypoint` type emitted by @tensorflow-models/pose-detection
// (TF.js MoveNet); only x/y are required, score/name optional.
export interface KeypointDTO {
  x: number;
  y: number;
  score?: number;
  name?: string;
}

export interface ReportDTO extends ReportSummaryDTO {
  metrics: Record<string, unknown>;
  figures: Record<string, unknown>[];
  observations: Record<string, unknown>;
  video_filename: string | null;
  video_size_bytes: number | null;
  // Spec Section 2 (a): raw landmark stream as JSON. Posture saves
  // a single front + side snapshot; null-valued for legacy reports
  // saved before this field existed.
  keypoints?: Record<string, KeypointDTO[] | null> | null;
}

export interface ReportCreatePayload {
  module:
    | "gait"
    | "biomech"
    | "posture"
    | "trendelenburg"
    | "single_leg_squat"
    | "sit_to_stand"
    | "chair_stand_30s"
    | "single_leg_stance"
    | "four_stage_balance"
    | "tug"
    | "sppb"
    | "slr"
    | "ake"
    | "modified_thomas"
    | "forward_lunge"
    | "sts_quality"
    | "tandem_walk"
    | "pronator_drift"
    | "functional_reach"
    | "single_leg_hop"
    | "counter_movement_jump"
    | "tuck_jump"
    | "overhead_squat"
    | "squat_lateral"
    | "rehab";
  body_part?: "shoulder" | "neck" | "knee" | "hip" | "ankle";
  movement?: string;
  side?: "left" | "right";
  metrics?: Record<string, unknown>;
  figures?: Record<string, unknown>[];
  observations?: Record<string, unknown>;
  video_filename?: string;
  video_size_bytes?: number;
  keypoints?: Record<string, KeypointDTO[] | null>;
}

export interface ReportListResponse {
  success: boolean;
  data: ReportSummaryDTO[];
  total: number;
}

// ─── HTTP helpers ─────────────────────────────────────────────────

/** Normalise FastAPI's error body into a human-readable string.
 *
 * FastAPI returns:
 *   • string `detail` on hand-raised HTTPException → use as-is
 *   • ARRAY of Pydantic error objects on 422 validation failure →
 *     each element like { type, loc, msg, input }. Naively passing
 *     the array into `new Error(...)` produced `Error.message` of
 *     "[object Object]" (or comma-joined variant) which then leaked
 *     into the UI as literal "[object Object]" text.
 *   • object detail (rare) → fall back to JSON.stringify
 */
function normaliseErrorDetail(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.length > 0) return detail;
  if (Array.isArray(detail)) {
    // Pydantic 422 — pick the human-readable `.msg` fields and
    // annotate with the loc path so callers know which field failed.
    const parts = detail
      .map((item) => {
        if (item && typeof item === "object") {
          const rec = item as Record<string, unknown>;
          const msg = typeof rec.msg === "string" ? rec.msg : null;
          const loc = Array.isArray(rec.loc)
            ? (rec.loc as unknown[]).filter((s) => typeof s === "string").join(".")
            : null;
          if (msg && loc) return `${loc}: ${msg}`;
          if (msg) return msg;
        }
        return null;
      })
      .filter((s): s is string => Boolean(s));
    if (parts.length > 0) return parts.join("; ");
  }
  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch {
      // fall through
    }
  }
  return `Request failed (${status})`;
}

async function asJSON<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new AuthError(normaliseErrorDetail(body.detail, res.status), res.status);
  }
  return (await res.json()) as T;
}

// ─── Report endpoints ─────────────────────────────────────────────
export async function listPatientReports(patientId: string): Promise<ReportListResponse> {
  const res = await authedFetch(`/api/patients/${patientId}/reports`);
  return asJSON<ReportListResponse>(res);
}

export async function createReport(
  patientId: string,
  payload: ReportCreatePayload,
): Promise<ReportDTO> {
  const res = await authedFetch(`/api/patients/${patientId}/reports`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return asJSON<ReportDTO>(res);
}

export async function getReport(id: string): Promise<ReportDTO> {
  const res = await authedFetch(`/api/reports/${id}`);
  return asJSON<ReportDTO>(res);
}

export async function deleteReport(id: string): Promise<{ success: boolean }> {
  const res = await authedFetch(`/api/reports/${id}`, { method: "DELETE" });
  return asJSON<{ success: boolean }>(res);
}
