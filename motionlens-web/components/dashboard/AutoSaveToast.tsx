"use client";
// AutoSaveToast — fires an auto-save on mount and shows a floating
// confirmation banner with a 10-second "Undo" (deleteReport) window.
//
// Meant for reduced-click doctor flow: when analysis phase reaches
// "done" the parent renders this component, and the save happens in
// the background without a click. Toast shows:
//   • saving…
//   • saved · Undo · View  (10s countdown)
//   • dismisses itself after 10s
//
// Undo is a soft-delete on the server; if the network call fails we
// leave the report saved and surface the error inline so the doctor
// can retry manually from the patient page.

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AlertCircle, CheckCircle2, Loader2, Undo2, X } from "lucide-react";

import { deleteReport } from "@/lib/reports";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { ReportCreatePayload } from "@/lib/reports";

interface Props {
  /** Called once on mount to obtain the report payload. Returning
   *  null aborts the save silently (analysis wasn't ready). */
  buildPayload: () => ReportCreatePayload | null;
  /** Undo window in seconds. Defaults to 10. */
  undoSeconds?: number;
}

type State =
  | { kind: "saving" }
  | { kind: "saved"; reportId: string; secondsLeft: number }
  | { kind: "undoing" }
  | { kind: "undone" }
  | { kind: "error"; message: string }
  | { kind: "dismissed" };

export function AutoSaveToast({ buildPayload, undoSeconds = 10 }: Props) {
  const { isDoctorFlow, patient, patientId, saveReport } = usePatientContext();
  const [state, setState] = useState<State>({ kind: "saving" });
  const firedRef = useRef(false);
  const buildPayloadRef = useRef(buildPayload);
  buildPayloadRef.current = buildPayload;

  // Fire the save exactly once on mount. `firedRef` guards against
  // React 18 dev-mode double invocation of effects.
  useEffect(() => {
    if (!isDoctorFlow) {
      setState({ kind: "dismissed" });
      return;
    }
    if (firedRef.current) return;
    firedRef.current = true;
    (async () => {
      const payload = buildPayloadRef.current();
      if (!payload) {
        setState({ kind: "error", message: "Analysis not ready — nothing to save." });
        return;
      }
      const out = await saveReport(payload);
      if (out.ok && out.reportId) {
        setState({ kind: "saved", reportId: out.reportId, secondsLeft: undoSeconds });
      } else {
        setState({ kind: "error", message: out.message || "Could not save report." });
      }
    })();
  }, [isDoctorFlow, saveReport, undoSeconds]);

  // Countdown tick for the "saved" state — decrements once per second
  // and auto-dismisses at zero.
  useEffect(() => {
    if (state.kind !== "saved") return;
    if (state.secondsLeft <= 0) {
      setState({ kind: "dismissed" });
      return;
    }
    const id = window.setTimeout(() => {
      setState((prev) =>
        prev.kind === "saved"
          ? { ...prev, secondsLeft: prev.secondsLeft - 1 }
          : prev,
      );
    }, 1000);
    return () => window.clearTimeout(id);
  }, [state]);

  const onUndo = useCallback(async () => {
    if (state.kind !== "saved") return;
    const id = state.reportId;
    setState({ kind: "undoing" });
    try {
      await deleteReport(id);
      setState({ kind: "undone" });
      window.setTimeout(() => setState({ kind: "dismissed" }), 3000);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Undo failed.";
      setState({ kind: "error", message: msg });
    }
  }, [state]);

  const onDismiss = useCallback(() => setState({ kind: "dismissed" }), []);

  if (!isDoctorFlow || state.kind === "dismissed") return null;

  return (
    <div className="fixed inset-x-0 top-4 z-[60] mx-auto flex w-full max-w-md justify-center px-4">
      <div className="pointer-events-auto flex w-full items-center gap-3 rounded-full border border-border bg-elevated/95 px-4 py-2.5 text-sm shadow-lg backdrop-blur">
        {state.kind === "saving" && (
          <>
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-accent" />
            <span className="flex-1 text-foreground">
              Saving to {patient?.name ?? "patient"}&apos;s record…
            </span>
          </>
        )}

        {state.kind === "saved" && (
          <>
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
            <span className="flex-1 text-foreground">
              Saved to {patient?.name ?? "patient"}&apos;s record
              <span className="ml-2 tabular text-xs text-muted">
                ({state.secondsLeft}s)
              </span>
            </span>
            <button
              type="button"
              onClick={onUndo}
              className="inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium text-muted transition hover:bg-surface hover:text-foreground"
            >
              <Undo2 className="h-3 w-3" /> Undo
            </button>
            {patientId && (
              <Link
                href={`/dashboard/patients/${patientId}`}
                className="rounded-full px-2 py-1 text-xs font-medium text-accent hover:underline"
              >
                View
              </Link>
            )}
            <button
              type="button"
              aria-label="Dismiss"
              onClick={onDismiss}
              className="rounded-full p-1 text-muted hover:bg-surface hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </>
        )}

        {state.kind === "undoing" && (
          <>
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted" />
            <span className="flex-1 text-foreground">Undoing save…</span>
          </>
        )}

        {state.kind === "undone" && (
          <>
            <Undo2 className="h-4 w-4 shrink-0 text-muted" />
            <span className="flex-1 text-foreground">
              Save undone — this run was deleted.
            </span>
            <button
              type="button"
              aria-label="Dismiss"
              onClick={onDismiss}
              className="rounded-full p-1 text-muted hover:bg-surface hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </>
        )}

        {state.kind === "error" && (
          <>
            <AlertCircle className="h-4 w-4 shrink-0 text-error" />
            <span className="flex-1 text-foreground">{state.message}</span>
            <button
              type="button"
              aria-label="Dismiss"
              onClick={onDismiss}
              className="rounded-full p-1 text-muted hover:bg-surface hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
