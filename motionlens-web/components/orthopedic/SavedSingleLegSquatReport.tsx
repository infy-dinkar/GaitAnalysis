"use client";
// Saved-report-viewer adapter for the Single-Leg Squat test. Pulls
// the per-side blobs out of metrics + the cached interpretation out
// of observations, and hands them to the live SingleLegSquatReport
// component.

import { SingleLegSquatReport } from "@/components/orthopedic/SingleLegSquatReport";
import {
  buildInterpretation,
  type SingleLegSquatFullResult,
  type SingleLegSquatSideResult,
} from "@/lib/orthopedic/singleLegSquat";

interface Props {
  patientName: string | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedSingleLegSquatReport({ patientName, metrics, observations }: Props) {
  const result: SingleLegSquatFullResult = {
    left:  (metrics.left  as SingleLegSquatSideResult | null | undefined) ?? null,
    right: (metrics.right as SingleLegSquatSideResult | null | undefined) ?? null,
  };
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(result);

  return (
    <SingleLegSquatReport
      patientName={patientName}
      result={result}
      interpretation={interpretation}
    />
  );
}
