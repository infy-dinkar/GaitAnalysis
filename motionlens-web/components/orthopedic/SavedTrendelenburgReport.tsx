"use client";
// Saved-report viewer for Trendelenburg sessions. Reads the metrics +
// observations + keypoints out of the persisted Report doc and hands
// them to the same TrendelenburgReport component the live flow uses.
// The keypoints field is preserved on save but only the time-series
// + screenshot are needed at render time.

import { TrendelenburgReport } from "@/components/orthopedic/TrendelenburgReport";
import {
  buildInterpretation,
  type TrendelenburgFullResult,
  type TrendelenburgSideResult,
} from "@/lib/orthopedic/trendelenburg";

interface Props {
  patientName: string | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedTrendelenburgReport({ patientName, metrics, observations }: Props) {
  const result: TrendelenburgFullResult = {
    left:  (metrics.left  as TrendelenburgSideResult | null | undefined) ?? null,
    right: (metrics.right as TrendelenburgSideResult | null | undefined) ?? null,
  };

  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildInterpretation(result);

  return (
    <TrendelenburgReport
      patientName={patientName}
      result={result}
      interpretation={interpretation}
    />
  );
}
