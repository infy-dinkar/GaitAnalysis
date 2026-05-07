"use client";
// Saved-report-viewer adapter for the 30-Second Chair Stand test.

import { ChairStand30sReport } from "@/components/orthopedic/ChairStand30sReport";
import {
  buildInterpretation,
  type ChairStand30sResult,
} from "@/lib/orthopedic/chairStand30s";

interface Props {
  patientName: string | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedChairStand30sReport({ patientName, metrics, observations }: Props) {
  const trial = metrics.trial as ChairStand30sResult | null | undefined;
  if (!trial) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
        Saved chair-stand-30s report is missing trial data.
      </div>
    );
  }
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(trial);
  return (
    <ChairStand30sReport
      patientName={patientName}
      result={trial}
      interpretation={interpretation}
    />
  );
}
