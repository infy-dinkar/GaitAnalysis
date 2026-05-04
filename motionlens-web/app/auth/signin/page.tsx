"use client";
// Sign-in page — two-column layout: medical animations | form
//
// Mobile: animations collapse to a top hero, form below.

import { Suspense } from "react";
import Link from "next/link";
import { MedicalAnimations } from "@/components/auth/MedicalAnimations";
import { SignInForm } from "@/components/auth/SignInForm";

export default function SignInPage() {
  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      {/* Left: animated medical visual (collapses to top on mobile) */}
      <div className="relative h-[40vh] w-full lg:h-screen lg:w-1/2">
        <MedicalAnimations />
      </div>

      {/* Right: form card */}
      <div className="flex flex-1 items-center justify-center bg-background px-6 py-10 lg:px-12">
        <div className="w-full max-w-md">
          {/* Logo / brand */}
          <Link href="/" className="mb-10 inline-flex items-center gap-0.5">
            <span className="text-xl font-semibold tracking-tight">MotionLens</span>
            <span className="text-xl font-semibold text-accent">.</span>
          </Link>

          <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
            <SignInForm />
          </Suspense>
        </div>
      </div>
    </div>
  );
}
