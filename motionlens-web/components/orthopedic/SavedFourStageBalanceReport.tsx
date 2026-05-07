"use client";
// Saved-report-viewer adapter for the 4-Stage Balance Test.

import { FourStageBalanceReport } from "@/components/orthopedic/FourStageBalanceReport";
import {
  buildInterpretation,
  type SessionResult,
} from "@/lib/orthopedic/fourStageBalance";

interface Props {
  patientName: string | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedFourStageBalanceReport({ patientName, metrics, observations }: Props) {
  const session = metrics.session as SessionResult | null | undefined;
  if (!session) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
        Saved 4-Stage Balance report is missing session data.
      </div>
    );
  }
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(session);
  return (
    <FourStageBalanceReport
      patientName={patientName}
      session={session}
      interpretation={interpretation}
    />
  );
}
