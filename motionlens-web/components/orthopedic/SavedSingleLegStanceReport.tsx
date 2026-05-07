"use client";
// Saved-report-viewer adapter for the Single-Leg Stance test.

import { SingleLegStanceReport } from "@/components/orthopedic/SingleLegStanceReport";
import {
  buildInterpretation,
  type SessionResult,
} from "@/lib/orthopedic/singleLegStance";

interface Props {
  patientName: string | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedSingleLegStanceReport({ patientName, metrics, observations }: Props) {
  const session = metrics.session as SessionResult | null | undefined;
  if (!session) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
        Saved Single-Leg Stance report is missing session data.
      </div>
    );
  }
  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(session);
  return (
    <SingleLegStanceReport
      patientName={patientName}
      session={session}
      interpretation={interpretation}
    />
  );
}
