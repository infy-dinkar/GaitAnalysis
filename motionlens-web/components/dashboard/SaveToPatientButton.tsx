"use client";
// Explicit "Save to patient history" button. Renders only when the
// page is opened from the doctor flow (?patientId=xxx in URL).
//
// Usage:
//   <SaveToPatientButton
//     buildPayload={() => ({
//       module: "biomech",
//       body_part: "shoulder",
//       movement: "flexion",
//       metrics: { ... },
//     })}
//   />
//
// After successful save the button becomes a green "Saved" pill so
// the doctor sees confirmation and can't double-save the same report.

import { useState } from "react";
import Link from "next/link";
import { CheckCircle2, Loader2, Save, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { ReportCreatePayload } from "@/lib/reports";

interface Props {
  /** Called when the user clicks "Save". Should return the report
   * payload — built lazily so we capture the latest analysis state. */
  buildPayload: () => ReportCreatePayload | null;
  /** Optional label override. */
  label?: string;
}

export function SaveToPatientButton({
  buildPayload,
  label = "Save to patient history",
}: Props) {
  const { isDoctorFlow, patient, patientId, saveReport } = usePatientContext();
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isDoctorFlow) return null;

  async function handle() {
    setError(null);
    const payload = buildPayload();
    if (!payload) {
      setError("Analysis not ready — please run it first.");
      return;
    }
    setBusy(true);
    try {
      const result = await saveReport(payload);
      if (result.ok) {
        setSaved(true);
      } else {
        setError(result.message || "Could not save report.");
      }
    } finally {
      setBusy(false);
    }
  }

  if (saved) {
    return (
      <div className="flex items-center gap-3 rounded-card border border-emerald-500/30 bg-emerald-500/5 px-5 py-3 text-sm">
        <CheckCircle2 className="h-5 w-5 text-emerald-600" />
        <span className="font-medium text-foreground">
          Saved to {patient?.name || "patient"}&apos;s history
        </span>
        {patientId && (
          <Link
            href={`/dashboard/patients/${patientId}`}
            className="ml-auto text-xs font-medium text-emerald-700 hover:underline"
          >
            View patient →
          </Link>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-col items-center gap-3 rounded-card border border-accent/30 bg-accent/5 p-5 sm:flex-row sm:justify-between">
        <div className="text-sm">
          <p className="font-medium text-foreground">
            Save this report to{" "}
            <span className="text-accent">{patient?.name || "the patient"}</span>
            &apos;s history?
          </p>
          <p className="mt-0.5 text-xs text-muted">
            Stores all metrics, charts, and observations for later review.
          </p>
        </div>
        <Button onClick={handle} disabled={busy}>
          {busy ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Saving…
            </>
          ) : (
            <>
              <Save className="h-4 w-4" />
              {label}
            </>
          )}
        </Button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-card border border-error/30 bg-error/5 px-4 py-2.5 text-sm">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}
    </div>
  );
}
