"use client";
// Saved-report-viewer adapter for the D3 Single-Leg Hop test.
// Re-hydrates the per-leg result blobs (or the pre-combined object
// the capture's Save action persisted) and hands them to the live
// SingleLegHopReport component.

import {
  SingleLegHopReport,
  buildSingleLegHopInterpretation,
} from "@/components/orthopedic/SingleLegHopReport";
import {
  buildCombinedResult,
  type CalibrationResult,
  type SingleLegHopCombinedResult,
  type SingleLegHopResult,
} from "@/lib/orthopedic/singleLegHop";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

export function SavedSingleLegHopReport({
  patientName,
  patient,
  metrics,
  observations,
}: Props) {
  // The Save action persists the SingleLegHopCombinedResult shape
  // flattened into metrics (see SingleLegHopCapture's buildPayload).
  // Older / experimental saves may have used metrics.result with a
  // SingleLegHopCombinedResult — accept either shape.
  const combined = parseSavedCombined(metrics);
  if (!combined) {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm">
        <p className="font-medium">
          This saved Single-Leg Hop report is missing its result blob.
        </p>
      </div>
    );
  }

  const stored = observations.interpretation;
  const interpretation =
    typeof stored === "string" && stored.length > 0
      ? stored
      : buildSingleLegHopInterpretation(combined);

  return (
    <SingleLegHopReport
      patientName={patientName}
      patient={patient}
      combined={combined}
      interpretation={interpretation}
    />
  );
}

/** Tolerant re-hydration — accepts either the flattened combined
 *  shape we save today, or a `metrics.result` wrapper for forward
 *  compat. Returns null when neither yields a usable structure. */
function parseSavedCombined(
  metrics: Record<string, unknown>,
): SingleLegHopCombinedResult | null {
  // Path 1 — wrapper.result.
  const wrappedRaw = metrics.result;
  if (
    wrappedRaw &&
    typeof wrappedRaw === "object" &&
    !Array.isArray(wrappedRaw)
  ) {
    const w = wrappedRaw as Record<string, unknown>;
    if ("left" in w || "right" in w) {
      const left = w.left as SingleLegHopResult | null | undefined;
      const right = w.right as SingleLegHopResult | null | undefined;
      return buildCombinedResult(left ?? null, right ?? null);
    }
  }

  // Path 2 — flat metrics with per-leg keys.
  if ("left" in metrics || "right" in metrics) {
    const left = metrics.left as SingleLegHopResult | null | undefined;
    const right = metrics.right as SingleLegHopResult | null | undefined;
    const combined = buildCombinedResult(left ?? null, right ?? null);
    // Honor the persisted calibration explicitly if available — it
    // may differ from whatever buildCombinedResult derived from the
    // per-leg blobs (e.g. caller persisted the calibration even
    // when one leg ran uncalibrated).
    const cal = metrics.calibration as CalibrationResult | null | undefined;
    if (cal && typeof cal === "object" && !combined.calibration) {
      return { ...combined, calibration: cal };
    }
    return combined;
  }

  return null;
}
