"use client";
// Saved-report adapter for the Tandem Walk (E1) test.

import { TandemWalkReport } from "@/components/orthopedic/TandemWalkReport";
import {
  buildInterpretation,
  type TandemWalkResult,
} from "@/lib/orthopedic/tandemWalk";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedTandemWalkReport({ patientName, patient, metrics, observations }: Props) {
  const stored = metrics.result as TandemWalkResult | undefined;
  if (!stored) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm text-foreground">
        Saved Tandem Walk report is missing its `result` payload. Re-record to refresh.
      </div>
    );
  }
  const cached = observations.interpretation;
  const interpretation =
    typeof cached === "string" && cached.length > 0
      ? cached
      : buildInterpretation(stored);

  return (
    <TandemWalkReport
      patientName={patientName}
      patient={patient}
      result={stored}
      interpretation={interpretation}
    />
  );
}
