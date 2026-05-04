// Authenticated API client for /api/reports + /api/patients/{id}/reports.

import { authedFetch, AuthError } from "@/lib/auth";

// ─── DTOs (mirror report_models.py) ──────────────────────────────
export interface ReportSummaryDTO {
  id: string;
  patient_id: string;
  doctor_id: string;
  module: "gait" | "biomech" | "posture";
  body_part: string | null;
  movement: string | null;
  side: string | null;
  created_at: string;
}

export interface ReportDTO extends ReportSummaryDTO {
  metrics: Record<string, unknown>;
  figures: Record<string, unknown>[];
  observations: Record<string, unknown>;
  video_filename: string | null;
  video_size_bytes: number | null;
}

export interface ReportCreatePayload {
  module: "gait" | "biomech" | "posture";
  body_part?: "shoulder" | "neck" | "knee" | "hip" | "ankle";
  movement?: string;
  side?: "left" | "right";
  metrics?: Record<string, unknown>;
  figures?: Record<string, unknown>[];
  observations?: Record<string, unknown>;
  video_filename?: string;
  video_size_bytes?: number;
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
