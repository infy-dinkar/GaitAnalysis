// Authenticated API client for /api/patients/{id}/prescription.
//
// Prescriptions are the doctor-authored override of the auto
// recommendation. Only one prescription lives per (doctor, patient)
// pair — PUT is idempotent upsert.

import { authedFetch } from "@/lib/auth";

export interface PrescriptionDTO {
  id: string;
  patient_id: string;
  doctor_id: string;
  slugs: string[];
  notes: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface PrescriptionResponse {
  success: boolean;
  data: PrescriptionDTO | null;
}

async function asJSON<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail =
      typeof (body as { detail?: unknown }).detail === "string"
        ? (body as { detail: string }).detail
        : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

/** GET the doctor prescription for a patient. Returns null when the
 *  doctor has not saved one yet — callers fall back to the auto
 *  recommender in that case. */
export async function loadPrescription(
  patientId: string,
): Promise<PrescriptionDTO | null> {
  const res = await authedFetch(`/api/patients/${patientId}/prescription`);
  const body = await asJSON<PrescriptionResponse>(res);
  return body.data ?? null;
}

/** Upsert the prescription. Replaces the slug list wholesale. */
export async function savePrescription(
  patientId: string,
  slugs: string[],
  notes: Record<string, unknown> = {},
): Promise<PrescriptionDTO> {
  const res = await authedFetch(`/api/patients/${patientId}/prescription`, {
    method: "PUT",
    body: JSON.stringify({ slugs, notes }),
  });
  return asJSON<PrescriptionDTO>(res);
}

/** Delete the doctor prescription so the auto recommender wins. */
export async function clearPrescription(patientId: string): Promise<void> {
  const res = await authedFetch(`/api/patients/${patientId}/prescription`, {
    method: "DELETE",
  });
  await asJSON<PrescriptionResponse>(res);
}
