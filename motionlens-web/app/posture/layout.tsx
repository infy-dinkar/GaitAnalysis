"use client";
// /posture is doctor-only.

import type { ReactNode } from "react";
import { AuthGuard } from "@/components/auth/AuthGuard";

export default function PostureLayout({ children }: { children: ReactNode }) {
  return <AuthGuard>{children}</AuthGuard>;
}
