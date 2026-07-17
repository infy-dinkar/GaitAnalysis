"use client";
// AutoFlowChrome — shared UI pieces for the reduced-click rehab flow
// (extracted from the validated Controlled Squat prototype so all
// exercise pages render the identical countdown / complete / footer
// chrome).
//
//   <AutoFlowCountdownOverlay />   camera overlay: big 3-2-1
//   <AutoFlowCompleteOverlay />    camera overlay: "Session complete"
//   <AutoFlowCountdownCard />      sidebar card: countdown + skip
//   <AutoFlowFooter />             sidebar footer: manual save while
//                                  live, auto-save toast + hint once
//                                  complete (doctor flow), friendly
//                                  hint in the public flow.

import { Button } from "@/components/ui/Button";
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { ReportCreatePayload } from "@/lib/reports";

export function AutoFlowCountdownOverlay({
  countdown,
  label = "Starting in",
}: {
  countdown: number;
  label?: string;
}) {
  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/30 backdrop-blur-[1px]">
      <div className="rounded-full bg-black/70 px-10 py-6 text-center text-white shadow-2xl ring-2 ring-white/20">
        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-emerald-300">
          {label}
        </p>
        <p className="tabular text-7xl font-semibold leading-none">
          {countdown}
        </p>
        <p className="mt-2 text-[10px] uppercase tracking-[0.14em] text-white/60">
          Space / Esc to skip
        </p>
      </div>
    </div>
  );
}

export function AutoFlowCompleteOverlay() {
  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-emerald-500/20 backdrop-blur-[1px]">
      <div className="rounded-full bg-emerald-600/90 px-8 py-4 text-center text-white shadow-2xl ring-2 ring-emerald-300/50">
        <p className="text-sm font-semibold uppercase tracking-[0.14em]">
          Session complete ✓
        </p>
      </div>
    </div>
  );
}

export function AutoFlowCountdownCard({
  countdown,
  onSkip,
  hint,
}: {
  countdown: number;
  onSkip: () => void;
  /** Exercise-specific positioning cue shown under the counter. */
  hint: string;
}) {
  return (
    <div className="rounded-card border border-accent/40 bg-accent/10 p-3 text-sm">
      <p className="font-medium text-foreground">
        Session starts in {countdown}s
      </p>
      <p className="mt-1 text-[11px] text-muted">
        {hint} Press Space / Escape or use the button below to skip.
      </p>
      <div className="mt-3">
        <Button variant="secondary" size="sm" onClick={onSkip}>
          Skip countdown
        </Button>
      </div>
    </div>
  );
}

export function AutoFlowFooter({
  complete,
  buildPayload,
  completeHint = "Target reached — saving to record automatically.",
}: {
  complete: boolean;
  buildPayload: () => ReportCreatePayload | null;
  completeHint?: string;
}) {
  const { isDoctorFlow } = usePatientContext();
  if (!complete) {
    return (
      <RehabSessionFooter
        buildPayload={buildPayload}
        label="Save session"
        compact
      />
    );
  }
  return (
    <>
      {/* AutoSaveToast fires the save exactly once on mount and
          renders as a fixed top-of-page toast with a 10s undo. It
          no-ops in the public flow. */}
      <AutoSaveToast buildPayload={buildPayload} />
      <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs text-muted">
        {isDoctorFlow ? completeHint : "Target reached. Nice session."}
      </div>
    </>
  );
}
