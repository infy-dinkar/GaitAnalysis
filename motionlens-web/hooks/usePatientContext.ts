"use client";
// Detects "doctor flow" — when an analysis page is visited from
// /dashboard/patients/{id}/analyze, the URL carries ?patientId=xxx.
// This hook:
//   1. Pulls the patientId from the query string
//   2. Fetches the patient's name (so we can show "Saving to Priya…")
//   3. Exposes saveReport() that POSTs to /api/patients/{id}/reports
//      and shows a small toast.
//
// Public-flow visits (no patientId) get a no-op saveReport — analysis
// keeps working exactly as before.

import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getPatient, type PatientDTO } from "@/lib/patients";
import { createReport, type ReportCreatePayload } from "@/lib/reports";

interface SaveOutcome {
  ok: boolean;
  message: string;
}

export function usePatientContext() {
  const params = useSearchParams();
  const patientId = params.get("patientId");

  const [patient, setPatient] = useState<PatientDTO | null>(null);
  const [loading, setLoading] = useState<boolean>(!!patientId);
  const [lastSave, setLastSave] = useState<SaveOutcome | null>(null);

  // Hydrate patient name on mount (only when patientId present)
  useEffect(() => {
    if (!patientId) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    getPatient(patientId)
      .then((p) => !cancelled && setPatient(p))
      .catch(() => {})
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  const saveReport = useCallback(
    async (payload: ReportCreatePayload): Promise<SaveOutcome> => {
      if (!patientId) {
        // Public flow — silently skip
        return { ok: false, message: "" };
      }
      try {
        await createReport(patientId, payload);
        const out: SaveOutcome = {
          ok: true,
          message: patient
            ? `Saved to ${patient.name}'s record.`
            : "Report saved.",
        };
        setLastSave(out);
        return out;
      } catch (e) {
        const out: SaveOutcome = {
          ok: false,
          message: e instanceof Error ? e.message : "Could not save report.",
        };
        setLastSave(out);
        return out;
      }
    },
    [patientId, patient],
  );

  return {
    /** True if the page is opened from the doctor dashboard. */
    isDoctorFlow: !!patientId,
    patientId,
    patient,
    loading,
    /** Last save attempt's result — used by the UI to show a banner. */
    lastSave,
    saveReport,
  };
}
