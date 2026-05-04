"use client";
// Renders an inline banner across analysis pages telling the doctor
// whether the latest report was saved to the active patient.

import Link from "next/link";
import { CheckCircle2, AlertCircle, UserRound } from "lucide-react";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patient: PatientDTO | null;
  saveStatus: { ok: boolean; message: string } | null;
}

export function SaveStatusBanner({ patient, saveStatus }: Props) {
  if (!patient && !saveStatus) return null;

  // Pre-save banner — shown while patient is selected but report hasn't
  // been saved yet. Lets the doctor verify they're working on the right
  // patient before running the assessment.
  if (patient && !saveStatus) {
    return (
      <div className="mb-6 flex items-center gap-3 rounded-card border border-accent/30 bg-accent/5 px-4 py-3 text-sm">
        <UserRound className="h-4 w-4 text-accent" />
        <span className="text-foreground">
          Saving to{" "}
          <Link
            href={`/dashboard/patients/${patient.id}`}
            className="font-semibold text-accent hover:underline"
          >
            {patient.name}
          </Link>
          &apos;s record once the assessment completes.
        </span>
      </div>
    );
  }

  // Post-save outcome banner
  if (saveStatus?.ok) {
    return (
      <div className="mb-6 flex items-center gap-3 rounded-card border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-sm">
        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
        <span className="text-foreground">{saveStatus.message}</span>
        {patient && (
          <Link
            href={`/dashboard/patients/${patient.id}`}
            className="ml-auto text-xs font-medium text-emerald-700 hover:underline"
          >
            View patient →
          </Link>
        )}
      </div>
    );
  }

  if (saveStatus && !saveStatus.ok && saveStatus.message) {
    return (
      <div className="mb-6 flex items-center gap-3 rounded-card border border-error/30 bg-error/5 px-4 py-3 text-sm">
        <AlertCircle className="h-4 w-4 text-error" />
        <span className="text-foreground">Could not save report: {saveStatus.message}</span>
      </div>
    );
  }

  return null;
}
