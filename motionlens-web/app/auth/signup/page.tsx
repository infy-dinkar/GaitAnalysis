"use client";
// Sign-up page — same two-column layout as sign-in.

import Link from "next/link";
import { MedicalAnimations } from "@/components/auth/MedicalAnimations";
import { SignUpForm } from "@/components/auth/SignUpForm";

export default function SignUpPage() {
  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      {/* Left: animated medical visual */}
      <div className="relative h-[40vh] w-full lg:h-screen lg:w-1/2">
        <MedicalAnimations />
      </div>

      {/* Right: form */}
      <div className="flex flex-1 items-center justify-center bg-background px-6 py-10 lg:px-12">
        <div className="w-full max-w-md">
          <Link href="/" className="mb-10 inline-flex items-center gap-0.5">
            <span className="text-xl font-semibold tracking-tight">MotionLens</span>
            <span className="text-xl font-semibold text-accent">.</span>
          </Link>

          <SignUpForm />
        </div>
      </div>
    </div>
  );
}
