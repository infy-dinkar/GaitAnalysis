"use client";
// Saved-report adapter for the Pronator Drift (E2) test.

import { PronatorDriftReport } from "@/components/orthopedic/PronatorDriftReport";
import {
  buildInterpretation,
  type PronatorDriftResult,
} from "@/lib/orthopedic/pronatorDrift";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedPronatorDriftReport({ patientName, patient, metrics, observations }: Props) {
  const stored = metrics.result as PronatorDriftResult | undefined;
  if (!stored) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm text-foreground">
        Saved Pronator Drift report is missing its `result` payload. Re-record to refresh.
      </div>
    );
  }
  const cached = observations.interpretation;
  const interpretation =
    typeof cached === "string" && cached.length > 0
      ? cached
      : buildInterpretation(stored);

  return (
    <PronatorDriftReport
      patientName={patientName}
      patient={patient}
      result={stored}
      interpretation={interpretation}
    />
  );
}
