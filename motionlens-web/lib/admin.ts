// Admin user-management API client.
//
// Every call hits /api/admin/*, which is gated server-side by
// require_admin. A non-admin gets a 403 here — that response, not any
// client-side check, is the actual authorisation boundary.

import { authedFetch, type DoctorPublicDTO } from "@/lib/auth";

export interface AdminUserListResponse {
  success: boolean;
  data: DoctorPublicDTO[];
  total: number;
}

export interface AdminCreateUserPayload {
  name: string;
  email: string;
  password: string;
  role: string;
  specialization?: string | null;
  license_number?: string | null;
}

export interface AdminUpdateUserPayload {
  role?: string;
  is_active?: boolean;
}

/** Surface the backend's `detail` string when present — the admin
 *  guards (last-active-admin, self-deactivation) return actionable
 *  messages that are worth showing verbatim. */
async function asJSON<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (typeof body?.error === "string") detail = body.error;
    } catch {
      // non-JSON error body — keep the status-code message
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function adminListUsers(): Promise<AdminUserListResponse> {
  const res = await authedFetch("/api/admin/users");
  return asJSON<AdminUserListResponse>(res);
}

export async function adminCreateUser(
  payload: AdminCreateUserPayload,
): Promise<DoctorPublicDTO> {
  const res = await authedFetch("/api/admin/users", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return asJSON<DoctorPublicDTO>(res);
}

export async function adminUpdateUser(
  id: string,
  payload: AdminUpdateUserPayload,
): Promise<DoctorPublicDTO> {
  const res = await authedFetch(`/api/admin/users/${id}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  return asJSON<DoctorPublicDTO>(res);
}

/** Returns 204 with no body, so there is nothing to parse. */
export async function adminResetPassword(
  id: string,
  newPassword: string,
): Promise<void> {
  const res = await authedFetch(`/api/admin/users/${id}/password`, {
    method: "POST",
    body: JSON.stringify({ new_password: newPassword }),
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // no body
    }
    throw new Error(detail);
  }
}
