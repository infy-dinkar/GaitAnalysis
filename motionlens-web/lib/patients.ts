// Authenticated API client for patient + report endpoints.

import { authedFetch, AuthError } from "@/lib/auth";

// ─── DTOs (mirror patient_models.py + report_models.py) ──────────
export interface PatientDTO {
  id: string;
  doctor_id: string;
  name: string;
  age: number;
  gender: "male" | "female" | "other";
  height_cm: number;
  weight_kg: number | null;
  contact: string | null;
  medical_notes: string | null;
  created_at: string;
  updated_at: string;
  report_count: number;
}

export interface PatientCreatePayload {
  name: string;
  age: number;
  gender: "male" | "female" | "other";
  height_cm: number;
  weight_kg?: number | null;
  contact?: string | null;
  medical_notes?: string | null;
}

export interface PatientUpdatePayload {
  name?: string;
  age?: number;
  gender?: "male" | "female" | "other";
  height_cm?: number;
  weight_kg?: number | null;
  contact?: string | null;
  medical_notes?: string | null;
}

export interface PatientListResponse {
  success: boolean;
  data: PatientDTO[];
  total: number;
}

export interface PatientDeleteResponse {
  success: boolean;
  deleted_patient_id: string;
  deleted_reports_count: number;
}

// ─── HTTP helpers ─────────────────────────────────────────────────
async function asJSON<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new AuthError(body.detail || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

// ─── Patient CRUD ─────────────────────────────────────────────────
export async function listPatients(): Promise<PatientListResponse> {
  const res = await authedFetch("/api/patients");
  return asJSON<PatientListResponse>(res);
}

export async function getPatient(id: string): Promise<PatientDTO> {
  const res = await authedFetch(`/api/patients/${id}`);
  return asJSON<PatientDTO>(res);
}

export async function createPatient(payload: PatientCreatePayload): Promise<PatientDTO> {
  const res = await authedFetch("/api/patients", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return asJSON<PatientDTO>(res);
}

export async function updatePatient(
  id: string,
  payload: PatientUpdatePayload,
): Promise<PatientDTO> {
  const res = await authedFetch(`/api/patients/${id}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  return asJSON<PatientDTO>(res);
}

export async function deletePatient(id: string): Promise<PatientDeleteResponse> {
  const res = await authedFetch(`/api/patients/${id}`, { method: "DELETE" });
  return asJSON<PatientDeleteResponse>(res);
}
