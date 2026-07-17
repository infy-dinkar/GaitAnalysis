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
  /** Optional idempotency key. When set, a successful save records
   *  the key in sessionStorage and a later mount with the same key
   *  silently skips the save — needed for result pages that persist
   *  their data in sessionStorage and can be revisited (gait results),
   *  where a naive re-mount would file a duplicate report. Undo
   *  clears the key so the doctor can re-save deliberately. */
  dedupeKey?: string;
}

const DEDUPE_PREFIX = "motionlens.autosaved:";

type State =
  | { kind: "saving" }
  | { kind: "saved"; reportId: string; secondsLeft: number }
  | { kind: "undoing" }
  | { kind: "undone" }
  | { kind: "error"; message: string }
  | { kind: "dismissed" };

export function AutoSaveToast({ buildPayload, undoSeconds = 10, dedupeKey }: Props) {
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
    // Idempotency: an earlier mount already saved this exact result
    // (page revisit on a sessionStorage-backed results page) — skip.
    if (dedupeKey) {
      try {
        if (window.sessionStorage.getItem(DEDUPE_PREFIX + dedupeKey)) {
          setState({ kind: "dismissed" });
          return;
        }
      } catch { /* storage unavailable — proceed with the save */ }
    }
    (async () => {
      const payload = buildPayloadRef.current();
      if (!payload) {
        setState({ kind: "error", message: "Analysis not ready — nothing to save." });
        return;
      }
      const out = await saveReport(payload);
      if (out.ok && out.reportId) {
        if (dedupeKey) {
          try {
            window.sessionStorage.setItem(DEDUPE_PREFIX + dedupeKey, out.reportId);
          } catch { /* non-fatal */ }
        }
        setState({ kind: "saved", reportId: out.reportId, secondsLeft: undoSeconds });
      } else {
        setState({ kind: "error", message: out.message || "Could not save report." });
      }
    })();
  }, [isDoctorFlow, saveReport, undoSeconds, dedupeKey]);

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
      if (dedupeKey) {
        try {
          window.sessionStorage.removeItem(DEDUPE_PREFIX + dedupeKey);
        } catch { /* non-fatal */ }
      }
      setState({ kind: "undone" });
      window.setTimeout(() => setState({ kind: "dismissed" }), 3000);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Undo failed.";
      setState({ kind: "error", message: msg });
    }
  }, [state, dedupeKey]);

  const onDismiss = useCallback(() => setState({ kind: "dismissed" }), []);

  if (!isDoctorFlow || state.kind === "dismissed") return null;

  // "saved" gets a BIG celebratory banner — the doctor should see at
  // a glance, from across the room, that the report stored itself.
  // The other states stay as a compact pill.
  if (state.kind === "saved") {
    return (
      <div className="fixed inset-x-0 top-4 z-[60] mx-auto flex w-full max-w-2xl justify-center px-4">
        <div className="pointer-events-auto w-full rounded-card border-2 border-emerald-500/60 bg-emerald-600 px-6 py-4 shadow-2xl">
          <div className="flex items-center gap-4">
            <CheckCircle2 className="h-10 w-10 shrink-0 text-white" />
            <div className="min-w-0 flex-1">
              <p className="text-xl font-bold leading-tight text-white">
                Report auto-saved ✓
              </p>
              <p className="mt-0.5 truncate text-sm text-emerald-50">
                Stored in {patient?.name ?? "the patient"}&apos;s record
                <span className="ml-2 tabular text-emerald-200">
                  · undo closes in {state.secondsLeft}s
                </span>
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                onClick={onUndo}
                className="inline-flex items-center gap-1.5 rounded-md bg-white/15 px-3 py-2 text-sm font-semibold text-white transition hover:bg-white/25"
              >
                <Undo2 className="h-4 w-4" /> Undo
              </button>
              {patientId && (
                <Link
                  href={`/dashboard/patients/${patientId}`}
                  className="rounded-md bg-white px-3 py-2 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-50"
                >
                  View
                </Link>
              )}
              <button
                type="button"
                aria-label="Dismiss"
                onClick={onDismiss}
                className="rounded-md p-2 text-emerald-100 transition hover:bg-white/15 hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
