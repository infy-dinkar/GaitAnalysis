"use client";
// Saved-report-viewer adapter for the B2 Overhead Squat test.
// Re-hydrates the OverheadSquatResult blob from the persisted
// metrics and hands it to the live OverheadSquatReport component.

import {
  OverheadSquatReport,
  buildOverheadSquatInterpretation,
} from "@/components/orthopedic/OverheadSquatReport";
import { type OverheadSquatResult } from "@/lib/orthopedic/overheadSquat";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedOverheadSquatReport({
  patientName,
  patient,
  metrics,
  observations,
}: Props) {
  const result = (metrics.result as OverheadSquatResult | undefined) ?? null;
  if (!result) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm">
        <p className="font-medium">
          This saved Overhead Squat report is missing its result blob.
        </p>
      </div>
    );
  }

  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildOverheadSquatInterpretation(result);

  return (
    <OverheadSquatReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
