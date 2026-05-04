"use client";
// Auth state provider — exposes the currently-logged-in doctor across
// the app. Hydrates from localStorage on mount and re-validates the
// JWT against /api/auth/me to ensure it's still active.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  type DoctorPublicDTO,
  type LoginPayload,
  type SignupPayload,
  fetchCurrentDoctor,
  getCachedDoctor,
  login as apiLogin,
  logout as apiLogout,
  signup as apiSignup,
} from "@/lib/auth";

interface AuthContextValue {
  doctor: DoctorPublicDTO | null;
  loading: boolean;            // true during initial auth check
  signIn: (payload: LoginPayload) => Promise<void>;
  signUp: (payload: SignupPayload) => Promise<void>;
  signOut: () => void;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  // Hydrate immediately from localStorage so SSR'd pages don't flash
  // "logged out" before the network roundtrip completes.
  const [doctor, setDoctor] = useState<DoctorPublicDTO | null>(() =>
    typeof window === "undefined" ? null : getCachedDoctor(),
  );
  const [loading, setLoading] = useState<boolean>(true);

  // On mount: re-validate token by hitting /api/auth/me. If the token
  // is expired or invalid the helper clears localStorage automatically.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const fresh = await fetchCurrentDoctor();
      if (!cancelled) {
        setDoctor(fresh);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(async (payload: LoginPayload) => {
    const res = await apiLogin(payload);
    setDoctor(res.doctor);
  }, []);

  const signUp = useCallback(async (payload: SignupPayload) => {
    const res = await apiSignup(payload);
    setDoctor(res.doctor);
  }, []);

  const signOut = useCallback(() => {
    apiLogout();
    setDoctor(null);
  }, []);

  const refresh = useCallback(async () => {
    const fresh = await fetchCurrentDoctor();
    setDoctor(fresh);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ doctor, loading, signIn, signUp, signOut, refresh }),
    [doctor, loading, signIn, signUp, signOut, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (ctx === null) {
    throw new Error("useAuth must be used within <AuthProvider>");
  }
  return ctx;
}
