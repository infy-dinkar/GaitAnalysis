"use client";
// All /gait/* routes are doctor-only.

import type { ReactNode } from "react";
import { AuthGuard } from "@/components/auth/AuthGuard";

export default function GaitLayout({ children }: { children: ReactNode }) {
  return <AuthGuard>{children}</AuthGuard>;
}
