"use client";
// Wrap any page/component that requires a logged-in doctor.
// Redirects to /auth/signin if no auth.
//
//   <AuthGuard>
//     <PatientDashboard />
//   </AuthGuard>

import { useEffect, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

interface Props {
  children: ReactNode;
  redirectTo?: string;
}

export function AuthGuard({ children, redirectTo = "/auth/signin" }: Props) {
  const { doctor, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && doctor === null) {
      router.replace(redirectTo);
    }
  }, [loading, doctor, router, redirectTo]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-muted">
          <Loader2 className="h-6 w-6 animate-spin text-accent" />
          <p className="text-sm">Verifying your session…</p>
        </div>
      </div>
    );
  }

  if (doctor === null) {
    // Redirect already triggered; render nothing to avoid flash
    return null;
  }

  return <>{children}</>;
}
