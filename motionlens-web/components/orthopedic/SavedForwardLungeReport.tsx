"use client";
// Saved-report-viewer adapter for the Forward Lunge (B3) test. Pulls
// the per-side blobs out of metrics + the cached interpretation out
// of observations and hands them to the live ForwardLungeReport
// component.

import { ForwardLungeReport } from "@/components/orthopedic/ForwardLungeReport";
import {
  buildInterpretation,
  type ForwardLungeFullResult,
  type ForwardLungeSideResult,
} from "@/lib/orthopedic/forwardLunge";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedForwardLungeReport({ patientName, patient, metrics, observations }: Props) {
  const result: ForwardLungeFullResult = {
    left:  (metrics.left  as ForwardLungeSideResult | null | undefined) ?? null,
    right: (metrics.right as ForwardLungeSideResult | null | undefined) ?? null,
  };
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(result);

  return (
    <ForwardLungeReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
