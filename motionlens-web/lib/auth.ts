// Auth API client + JWT token management.
//
// Tokens are stored in localStorage so login persists across browser
// reloads. On every API request that requires auth, we attach the
// Authorization: Bearer <token> header.

import { API_BASE_URL } from "@/lib/api";

const TOKEN_KEY = "motionlens.auth_token";
const DOCTOR_KEY = "motionlens.doctor";

// ─── DTOs (mirror auth_models.py / patient_models.py) ─────────────
export interface DoctorPublicDTO {
  id: string;
  email: string;
  name: string;
  specialization: string | null;
  license_number: string | null;
  created_at: string;
}

export interface AuthTokenResponse {
  success: boolean;
  token: string;
  token_type: string;
  expires_in: number;
  doctor: DoctorPublicDTO;
}

export interface SignupPayload {
  email: string;
  password: string;
  name: string;
  specialization?: string;
  license_number?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

// ─── Token storage ────────────────────────────────────────────────
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(DOCTOR_KEY);
}

export function getCachedDoctor(): DoctorPublicDTO | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(DOCTOR_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as DoctorPublicDTO;
  } catch {
    return null;
  }
}

export function setCachedDoctor(doctor: DoctorPublicDTO): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(DOCTOR_KEY, JSON.stringify(doctor));
}

// ─── HTTP helpers ─────────────────────────────────────────────────
async function postJSON<T>(endpoint: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new AuthError(errBody.detail || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

async function getJSON<T>(endpoint: string, token?: string): Promise<T> {
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE_URL}${endpoint}`, { headers });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new AuthError(errBody.detail || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

export class AuthError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "AuthError";
  }
}

// ─── Public API ───────────────────────────────────────────────────
export async function signup(payload: SignupPayload): Promise<AuthTokenResponse> {
  const data = await postJSON<AuthTokenResponse>("/api/auth/signup", payload);
  setToken(data.token);
  setCachedDoctor(data.doctor);
  return data;
}

export async function login(payload: LoginPayload): Promise<AuthTokenResponse> {
  const data = await postJSON<AuthTokenResponse>("/api/auth/login", payload);
  setToken(data.token);
  setCachedDoctor(data.doctor);
  return data;
}

export async function fetchCurrentDoctor(): Promise<DoctorPublicDTO | null> {
  const token = getToken();
  if (!token) return null;
  try {
    const doctor = await getJSON<DoctorPublicDTO>("/api/auth/me", token);
    setCachedDoctor(doctor);
    return doctor;
  } catch (e) {
    if (e instanceof AuthError && e.status === 401) {
      // Token invalid / expired — clear it
      clearToken();
    }
    return null;
  }
}

export function logout(): void {
  clearToken();
}

// ─── Authenticated fetch helper for other API calls ───────────────
export async function authedFetch(
  endpoint: string,
  init: RequestInit = {},
): Promise<Response> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return fetch(`${API_BASE_URL}${endpoint}`, { ...init, headers });
}
