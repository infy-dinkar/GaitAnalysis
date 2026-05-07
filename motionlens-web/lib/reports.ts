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
    | "single_leg_stance";
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
    | "single_leg_stance";
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
async function asJSON<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new AuthError(body.detail || `Request failed (${res.status})`, res.status);
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
