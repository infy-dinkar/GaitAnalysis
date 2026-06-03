"use client";
// Saved-report-viewer adapter for the Sit-to-Stand Quality (B4) test.
// Pulls the result blob out of metrics + the cached interpretation
// out of observations and hands them to the live STSQualityReport
// component.

import { STSQualityReport } from "@/components/orthopedic/STSQualityReport";
import {
  buildInterpretation,
  type STSQualityResult,
} from "@/lib/orthopedic/stsQuality";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedSTSQualityReport({ patientName, patient, metrics, observations }: Props) {
  // The capture component saves { result, chair_seat_height_cm }; the
  // whole result blob is the source of truth for rendering.
  const stored = metrics.result as STSQualityResult | undefined;
  if (!stored) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm text-foreground">
        Saved STS-quality report is missing its `result` payload. Re-record to refresh.
      </div>
    );
  }
  const cached = observations.interpretation;
  const interpretation =
    typeof cached === "string" && cached.length > 0
      ? cached
      : buildInterpretation(stored);

  return (
    <STSQualityReport
      patientName={patientName}
      patient={patient}
      result={stored}
      interpretation={interpretation}
    />
  );
}
