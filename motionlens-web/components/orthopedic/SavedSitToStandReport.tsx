"use client";
// Saved-report-viewer adapter for the 5x Sit-to-Stand test.

import { SitToStandReport } from "@/components/orthopedic/SitToStandReport";
import {
  buildInterpretation,
  type SitToStandResult,
} from "@/lib/orthopedic/sitToStand";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedSitToStandReport({ patientName, patient, metrics, observations }: Props) {
  const trial = metrics.trial as SitToStandResult | null | undefined;
  if (!trial) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
        Saved sit-to-stand report is missing trial data.
      </div>
    );
  }
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(trial);
  return (
    <SitToStandReport
      patientName={patientName}
      patient={patient}
      result={trial}
      interpretation={interpretation}
    />
  );
}
