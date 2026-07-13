"use client";
// Saved-report-viewer adapter for the D2 Tuck Jump test. Re-hydrates
// the TuckJumpResult blob from the persisted metrics and hands it to
// the live TuckJumpReport component.

import {
  TuckJumpReport,
  buildTuckJumpInterpretation,
} from "@/components/orthopedic/TuckJumpReport";
import { type TuckJumpResult } from "@/lib/orthopedic/tuckJump";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedTuckJumpReport({
  patientName,
  patient,
  metrics,
  observations,
}: Props) {
  const result = (metrics.result as TuckJumpResult | undefined) ?? null;
  if (!result) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm">
        <p className="font-medium">
          This saved Tuck Jump report is missing its result blob.
        </p>
      </div>
    );
  }

  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildTuckJumpInterpretation(result);

  return (
    <TuckJumpReport
      patientName={patientName}
      patient={patient}
      result={result}
      interpretation={interpretation}
    />
  );
}
