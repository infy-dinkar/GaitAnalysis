"use client";
// Saved-report-viewer adapter for the SPPB composite test.
//
// SPPBReport is fed `previousScores` when historical SPPB sessions
// exist for this patient — when no history is available, the trend
// chart is hidden automatically.

import { SPPBReport, type SPPBHistoryEntry } from "@/components/orthopedic/SPPBReport";
import type { SPPBResult } from "@/lib/orthopedic/sppb";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
  /** Optional — passed by the saved-report viewer when it has the
   *  patient's earlier SPPB session list available. */
  previousScores?: SPPBHistoryEntry[];
}

export function SavedSPPBReport({
  patientName, patient, metrics, previousScores,
}: Props) {
  const result = metrics.result as SPPBResult | null | undefined;
  if (!result) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
        Saved SPPB report is missing composite data.
      </div>
    );
  }
  return (
    <SPPBReport
      result={result}
      patient={patient}
      patientName={patientName}
      previousScores={previousScores}
    />
  );
}
