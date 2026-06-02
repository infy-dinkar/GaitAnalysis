"use client";
// Saved-report-viewer adapter for the Active Knee Extension test.
// Pulls the per-side blobs out of metrics + the cached interpretation
// out of observations and hands them to the live AKEReport component.

import { AKEReport } from "@/components/orthopedic/AKEReport";
import {
  buildInterpretation,
  type AKEFullResult,
  type AKESideResult,
} from "@/lib/orthopedic/ake";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedAKEReport({ patientName, patient, metrics, observations }: Props) {
  const result: AKEFullResult = {
    left:  (metrics.left  as AKESideResult | null | undefined) ?? null,
    right: (metrics.right as AKESideResult | null | undefined) ?? null,
  };
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(result);

  return (
    <AKEReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
