"use client";
// All /biomech/* routes are doctor-only. Public visitors get redirected
// to /auth/signin; logged-in doctors see the page normally.

import type { ReactNode } from "react";
import { AuthGuard } from "@/components/auth/AuthGuard";

export default function BiomechLayout({ children }: { children: ReactNode }) {
  return <AuthGuard>{children}</AuthGuard>;
}
