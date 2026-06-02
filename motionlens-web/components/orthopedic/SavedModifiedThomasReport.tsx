"use client";
// Saved-report-viewer adapter for the Modified Thomas Test. Pulls the
// per-side blobs out of metrics + the cached interpretation out of
// observations and hands them to the live ModifiedThomasReport
// component.

import { ModifiedThomasReport } from "@/components/orthopedic/ModifiedThomasReport";
import {
  buildInterpretation,
  type ModifiedThomasFullResult,
  type ModifiedThomasSideResult,
} from "@/lib/orthopedic/modifiedThomas";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedModifiedThomasReport({ patientName, patient, metrics, observations }: Props) {
  const result: ModifiedThomasFullResult = {
    left:  (metrics.left  as ModifiedThomasSideResult | null | undefined) ?? null,
    right: (metrics.right as ModifiedThomasSideResult | null | undefined) ?? null,
  };
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(result);

  return (
    <ModifiedThomasReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
