"use client";
// Single patient card — used in /dashboard and /dashboard/patients lists.

import Link from "next/link";
import { ArrowUpRight, FileText, Mars, Venus } from "lucide-react";
import type { PatientDTO } from "@/lib/patients";

export function PatientCard({ patient }: { patient: PatientDTO }) {
  const GenderIcon =
    patient.gender === "male" ? Mars : patient.gender === "female" ? Venus : null;

  return (
    <Link
      href={`/dashboard/patients/${patient.id}`}
      className="group block rounded-card border border-border bg-surface p-5 transition hover:border-accent hover:shadow-glow-sm"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          {/* Name */}
          <h3 className="truncate text-base font-semibold tracking-tight text-foreground">
            {patient.name}
          </h3>

          {/* Meta row */}
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
            <span className="flex items-center gap-1">
              {GenderIcon && <GenderIcon className="h-3.5 w-3.5" />}
              {patient.gender.charAt(0).toUpperCase() + patient.gender.slice(1)}
            </span>
            <span>·</span>
            <span>{patient.age} yrs</span>
            <span>·</span>
            <span>{patient.height_cm} cm</span>
            {patient.weight_kg ? (
              <>
                <span>·</span>
                <span>{patient.weight_kg} kg</span>
              </>
            ) : null}
          </div>
        </div>

        <ArrowUpRight className="h-4 w-4 text-muted transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-accent" />
      </div>

      {/* Report count */}
      <div className="mt-4 flex items-center justify-between text-xs">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-elevated px-2.5 py-1 text-muted">
          <FileText className="h-3 w-3" />
          {patient.report_count} {patient.report_count === 1 ? "report" : "reports"}
        </span>
        <span className="tabular text-subtle">
          Added {formatDate(patient.created_at)}
        </span>
      </div>
    </Link>
  );
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return "";
  }
}
