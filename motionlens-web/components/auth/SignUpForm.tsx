"use client";
// Sign-up form for new doctors.

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Mail,
  Lock,
  User,
  Stethoscope,
  IdCard,
  Loader2,
  AlertCircle,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/contexts/AuthContext";

export function SignUpForm() {
  const router = useRouter();
  const { signUp } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [specialization, setSpecialization] = useState("");
  const [licenseNumber, setLicenseNumber] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await signUp({
        name: name.trim(),
        email: email.trim().toLowerCase(),
        password,
        specialization: specialization.trim() || undefined,
        license_number: licenseNumber.trim() || undefined,
      });
      router.replace("/dashboard");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Sign up failed";
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  const passwordTooShort = password.length > 0 && password.length < 8;

  return (
    <form onSubmit={onSubmit} className="w-full max-w-md space-y-5">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight text-foreground">
          Create your account
        </h1>
        <p className="text-sm text-muted">
          Join MotionLens. Add patients, run assessments, store reports — all in one place.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}

      {/* Full name */}
      <div className="space-y-1.5">
        <label htmlFor="name" className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Full name
        </label>
        <div className="relative">
          <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input
            id="name"
            type="text"
            autoComplete="name"
            required
            minLength={2}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Dr. Jane Doe"
            className="w-full rounded-card border border-border bg-surface px-10 py-3 text-sm text-foreground transition focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </div>
      </div>

      {/* Email */}
      <div className="space-y-1.5">
        <label htmlFor="email" className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Email
        </label>
        <div className="relative">
          <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="doctor@clinic.com"
            className="w-full rounded-card border border-border bg-surface px-10 py-3 text-sm text-foreground transition focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </div>
      </div>

      {/* Password */}
      <div className="space-y-1.5">
        <label htmlFor="password" className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Password
        </label>
        <div className="relative">
          <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input
            id="password"
            type="password"
            autoComplete="new-password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="At least 8 characters"
            className={`w-full rounded-card border bg-surface px-10 py-3 text-sm text-foreground transition focus:outline-none focus:ring-2 ${
              passwordTooShort
                ? "border-warning/50 focus:border-warning focus:ring-warning/20"
                : "border-border focus:border-accent focus:ring-accent/20"
            }`}
          />
        </div>
        {passwordTooShort && (
          <p className="text-xs text-warning">Password must be at least 8 characters</p>
        )}
      </div>

      {/* Specialization (optional) */}
      <div className="space-y-1.5">
        <label htmlFor="specialization" className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Specialization <span className="text-subtle/60 normal-case">(optional)</span>
        </label>
        <div className="relative">
          <Stethoscope className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input
            id="specialization"
            type="text"
            value={specialization}
            onChange={(e) => setSpecialization(e.target.value)}
            placeholder="e.g. Sports Physiotherapy"
            className="w-full rounded-card border border-border bg-surface px-10 py-3 text-sm text-foreground transition focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </div>
      </div>

      {/* License (optional) */}
      <div className="space-y-1.5">
        <label htmlFor="license" className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          License number <span className="text-subtle/60 normal-case">(optional)</span>
        </label>
        <div className="relative">
          <IdCard className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input
            id="license"
            type="text"
            value={licenseNumber}
            onChange={(e) => setLicenseNumber(e.target.value)}
            placeholder="e.g. PT-12345"
            className="w-full rounded-card border border-border bg-surface px-10 py-3 text-sm text-foreground transition focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </div>
      </div>

      {/* Submit */}
      <Button
        type="submit"
        disabled={busy || !name || !email || password.length < 8}
        className="w-full"
      >
        {busy ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Creating account…
          </>
        ) : (
          <>
            Create account
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </Button>

      {/* Footer */}
      <div className="border-t border-border pt-5 text-center">
        <p className="text-sm text-muted">
          Already have an account?{" "}
          <Link href="/auth/signin" className="font-medium text-accent hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </form>
  );
}
