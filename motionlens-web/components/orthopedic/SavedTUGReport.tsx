"use client";
// Saved-report-viewer adapter for the Timed Up and Go (TUG) test.

import { TUGReport } from "@/components/orthopedic/TUGReport";
import type { TUGResult } from "@/lib/orthopedic/tug";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedTUGReport({ patientName, patient, metrics }: Props) {
  const result = metrics.result as TUGResult | null | undefined;
  if (!result) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
        Saved TUG report is missing analysis data.
      </div>
    );
  }
  return (
    <TUGReport
      patientName={patientName}
      patient={patient}
      result={result}
    />
  );
}
