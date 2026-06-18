"use client";
// Saved-report-viewer adapter for the D4 Counter-Movement Jump
// test. Re-hydrates the CMJResult blob from the persisted metrics
// and hands it to the live CMJReport component.

import {
  CMJReport,
  buildCMJInterpretation,
} from "@/components/orthopedic/CMJReport";
import { type CMJResult } from "@/lib/orthopedic/counterMovementJump";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedCMJReport({
  patientName,
  patient,
  metrics,
  observations,
}: Props) {
  const result = (metrics.result as CMJResult | undefined) ?? null;
  if (!result) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm">
        <p className="font-medium">
          This saved Counter-Movement Jump report is missing its result blob.
        </p>
      </div>
    );
  }

  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildCMJInterpretation(result);

  return (
    <CMJReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
