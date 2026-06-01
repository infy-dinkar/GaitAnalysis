"use client";
// Saved-report-viewer adapter for the Straight Leg Raise test. Pulls
// the per-side blobs out of metrics + the cached interpretation out
// of observations and hands them to the live SLRReport component.

import { SLRReport } from "@/components/orthopedic/SLRReport";
import {
  buildInterpretation,
  type SLRFullResult,
  type SLRSideResult,
} from "@/lib/orthopedic/slr";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedSLRReport({ patientName, patient, metrics, observations }: Props) {
  const result: SLRFullResult = {
    left:  (metrics.left  as SLRSideResult | null | undefined) ?? null,
    right: (metrics.right as SLRSideResult | null | undefined) ?? null,
  };
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(result);

  return (
    <SLRReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
