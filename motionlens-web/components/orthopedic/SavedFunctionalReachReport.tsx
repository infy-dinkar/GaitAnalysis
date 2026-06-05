"use client";
// Saved-report-viewer adapter for the C6 Functional Reach test.
// Pulls the blob out of metrics + the cached interpretation out of
// observations and hands them to the live FunctionalReachReport
// component.

import { FunctionalReachReport } from "@/components/orthopedic/FunctionalReachReport";
import {
  buildInterpretation,
  type FunctionalReachResult,
} from "@/lib/orthopedic/functionalReach";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedFunctionalReachReport({
  patientName,
  patient,
  metrics,
  observations,
}: Props) {
  const result = (metrics.result as FunctionalReachResult | undefined) ?? null;
  if (!result) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm">
        <p className="font-medium">This saved Functional Reach report is missing its result blob.</p>
      </div>
    );
  }

  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(result);

  return (
    <FunctionalReachReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
