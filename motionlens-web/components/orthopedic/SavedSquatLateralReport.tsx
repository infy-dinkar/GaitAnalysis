"use client";
// Saved-report-viewer adapter for Squat (Lateral). Re-hydrates the
// SquatLateralResult blob from persisted metrics and hands it to the
// live SquatLateralReport component.

import {
  SquatLateralReport,
  buildSquatLateralInterpretation,
} from "@/components/orthopedic/SquatLateralReport";
import { type SquatLateralResult } from "@/lib/orthopedic/squatLateral";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedSquatLateralReport({
  patientName,
  patient,
  metrics,
  observations,
}: Props) {
  const result = (metrics.result as SquatLateralResult | undefined) ?? null;
  if (!result) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm">
        <p className="font-medium">
          This saved Squat (Lateral) report is missing its result blob.
        </p>
      </div>
    );
  }

  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildSquatLateralInterpretation(result);

  return (
    <SquatLateralReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
