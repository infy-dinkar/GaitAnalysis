"use client";
// Biomech Auto Mode — sequence runner.
//
// Reads the queue from the URL and mounts LiveAssessment for one
// step at a time. COMPLETION-DRIVEN, no timer: each step runs until
// the patient finishes the 5-rep cycle target (LiveAssessment's own
// counter — merged tests need both directions), the report renders
// + auto-saves, then a short "next test" countdown advances the
// queue. The operator's Skip / Show Analysis remain the manual
// exits for a stuck step. (The legacy ?d= duration param is
// accepted but ignored.)
//
// Every step passes the FULL movement definition (merged, both
// direction labels, secondary target) looked up from the same
// per-joint catalogs the standalone /biomech/{joint}/live pages use
// — so merged tests capture BOTH angles here exactly like normal
// mode, not just a single peak.

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { CheckCircle2, ChevronRight, Loader2, Pause, Play, RotateCcw, SkipForward, Undo2, X, XCircle } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { LiveAssessment } from "@/components/biomech/LiveAssessment";
import {
  decodeQueue,
  stepTitle,
  type AutoStep,
  type Joint,
} from "@/lib/biomech/autoModeCatalog";
import { SHOULDER_MOVEMENTS } from "@/lib/biomech/shoulder";
import { NECK_MOVEMENTS } from "@/lib/biomech/neck";
import { KNEE_MOVEMENTS } from "@/lib/biomech/knee";
import { HIP_MOVEMENTS } from "@/lib/biomech/hip";
import { ANKLE_MOVEMENTS } from "@/lib/biomech/ankle-live";
import { createReport, deleteReport } from "@/lib/reports";

const READY_COUNTDOWN_SEC = 3;
// Pause between a completed report and the next step — long enough
// to glance at the result + the auto-save banner.
const NEXT_COUNTDOWN_SEC = 8;

// ── Full movement lookup ─────────────────────────────────────────
// The auto-mode catalog stores only id/label/target (picker
// metadata). The capture needs the movement's FULL definition —
// merged flag, per-direction labels, secondary target — exactly as
// the standalone live pages pass it. Same source catalogs.
interface FullMovementDef {
  id: string;
  label: string;
  description: string;
  target: [number, number];
  merged?: boolean;
  primaryLabel?: string;
  secondaryLabel?: string;
  secondaryTarget?: [number, number];
}

const CATALOG_BY_JOINT: Record<Joint, ReadonlyArray<FullMovementDef>> = {
  shoulder: SHOULDER_MOVEMENTS as ReadonlyArray<FullMovementDef>,
  neck: NECK_MOVEMENTS as ReadonlyArray<FullMovementDef>,
  knee: KNEE_MOVEMENTS as ReadonlyArray<FullMovementDef>,
  hip: HIP_MOVEMENTS as ReadonlyArray<FullMovementDef>,
  ankle: ANKLE_MOVEMENTS as unknown as ReadonlyArray<FullMovementDef>,
};

function fullMovement(step: AutoStep): FullMovementDef | null {
  return (
    CATALOG_BY_JOINT[step.joint]?.find((m) => m.id === step.movementId) ?? null
  );
}

export default function BiomechAutoRunPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <Suspense fallback={null}>
            <Inner />
          </Suspense>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function Inner() {
  const router = useRouter();
  const params = useSearchParams();
  const queueRaw = params.get("q") ?? "";
  const patientId = params.get("patientId");
  const qs = patientId ? `?patientId=${patientId}` : "";

  const queue = useMemo(() => decodeQueue(queueRaw), [queueRaw]);

  const [stepIdx, setStepIdx] = useState(0);
  // Bumped on every retest — part of the LiveAssessment key so the
  // step remounts completely fresh (camera auto-starts again).
  const [attempt, setAttempt] = useState(0);
  const [phase, setPhase] = useState<"ready" | "running" | "nexting" | "done">(
    "ready",
  );
  const [readyLeft, setReadyLeft] = useState(READY_COUNTDOWN_SEC);
  const [nextLeft, setNextLeft] = useState(NEXT_COUNTDOWN_SEC);
  const [paused, setPaused] = useState(false);

  // ── Cumulative-save bookkeeping ─────────────────────────────
  // NOTHING is saved per step. Each completed test's result is
  // collected here (keyed by step index) and the whole set is saved as
  // ONE combined is_batch report when the sequence finishes — same
  // report shape BatchSession produces, so the saved-report viewer
  // renders it stacked. A retest just overwrites that step's collected
  // entry (the fresh attempt re-emits via onResult); there is no
  // per-step report to delete.
  const stepIdxRef = useRef(0);
  useEffect(() => {
    stepIdxRef.current = stepIdx;
  }, [stepIdx]);
  const collectedRef = useRef<Record<number, Record<string, unknown>>>({});

  const advance = useCallback(() => {
    setStepIdx((prev) => {
      const next = prev + 1;
      if (next >= queue.length) {
        setPhase("done");
        return prev;
      }
      setPhase("ready");
      setReadyLeft(READY_COUNTDOWN_SEC);
      setNextLeft(NEXT_COUNTDOWN_SEC);
      return next;
    });
  }, [queue.length]);

  // Retest the CURRENT step: discard its collected result, then remount
  // the step fresh (get-ready → camera → countdown → capture). The new
  // attempt re-emits its entry via onResult, overwriting the slot.
  const retest = useCallback(() => {
    delete collectedRef.current[stepIdxRef.current];
    setAttempt((a) => a + 1);
    setPhase("ready");
    setReadyLeft(READY_COUNTDOWN_SEC);
    setNextLeft(NEXT_COUNTDOWN_SEC);
  }, []);

  // Step completed (rep target hit or manual Show Analysis) — the
  // per-test report is on screen. Give the operator a beat to see it,
  // then advance. (No save happens here — see the cumulative save.)
  const handleStepCompleted = useCallback(() => {
    setNextLeft(NEXT_COUNTDOWN_SEC);
    setPhase((p) => (p === "running" ? "nexting" : p));
  }, []);

  // ── Cumulative auto-save — ONE is_batch report at the very end ───
  const [autoBanner, setAutoBanner] = useState<
    | { kind: "saved"; reportId: string; secondsLeft: number }
    | { kind: "undoing" }
    | { kind: "undone" }
    | { kind: "error"; message: string }
    | null
  >(null);
  const cumulativeFiredRef = useRef(false);
  const [savedCount, setSavedCount] = useState(0);

  useEffect(() => {
    if (phase !== "done" || cumulativeFiredRef.current) return;
    cumulativeFiredRef.current = true;
    if (!patientId) return; // public flow — nothing to persist
    const items = Object.keys(collectedRef.current)
      .map(Number)
      .sort((a, b) => a - b)
      .map((k) => collectedRef.current[k]);
    if (items.length === 0) return;
    setSavedCount(items.length);
    void (async () => {
      try {
        const created = await createReport(patientId, {
          module: "biomech",
          movement: "batch",
          metrics: { is_batch: true, items },
        });
        setAutoBanner({ kind: "saved", reportId: created.id, secondsLeft: 10 });
      } catch (e) {
        setAutoBanner({
          kind: "error",
          message: e instanceof Error ? e.message : "Auto-save failed.",
        });
      }
    })();
  }, [phase, patientId]);

  // Countdown tick for the auto-saved banner (mirrors AutoSaveToast).
  useEffect(() => {
    if (!autoBanner || autoBanner.kind !== "saved") return;
    if (autoBanner.secondsLeft <= 0) {
      setAutoBanner(null);
      return;
    }
    const id = window.setTimeout(() => {
      setAutoBanner((prev) =>
        prev && prev.kind === "saved"
          ? { ...prev, secondsLeft: prev.secondsLeft - 1 }
          : prev,
      );
    }, 1000);
    return () => window.clearTimeout(id);
  }, [autoBanner]);

  async function undoAutoSave() {
    if (!autoBanner || autoBanner.kind !== "saved") return;
    const id = autoBanner.reportId;
    setAutoBanner({ kind: "undoing" });
    try {
      await deleteReport(id);
      setAutoBanner({ kind: "undone" });
      window.setTimeout(() => setAutoBanner(null), 3000);
    } catch (e) {
      setAutoBanner({
        kind: "error",
        message: e instanceof Error ? e.message : "Undo failed.",
      });
    }
  }

  // ── Ready countdown ─────────────────────────────────────────
  useEffect(() => {
    if (phase !== "ready" || paused) return;
    if (queue.length === 0) return;
    if (readyLeft <= 0) {
      setPhase("running");
      return;
    }
    const id = window.setTimeout(() => setReadyLeft((n) => n - 1), 1000);
    return () => window.clearTimeout(id);
  }, [phase, paused, readyLeft, queue.length]);

  // ── Next-step countdown (report on screen) ──────────────────
  useEffect(() => {
    if (phase !== "nexting" || paused) return;
    if (nextLeft <= 0) {
      advance();
      return;
    }
    const id = window.setTimeout(() => setNextLeft((n) => n - 1), 1000);
    return () => window.clearTimeout(id);
  }, [phase, paused, nextLeft, advance]);

  const currentStep: AutoStep | null =
    queue.length > 0 && stepIdx < queue.length ? queue[stepIdx] : null;

  if (queue.length === 0) {
    return (
      <div className="rounded-card border border-error/40 bg-error/5 p-6 text-sm">
        <p className="font-semibold text-error">No tests in the queue.</p>
        <p className="mt-2 text-muted">
          Head back to Auto Mode and pick some movements.
        </p>
        <Link href={`/biomech/auto${qs}`} className="mt-4 inline-block">
          <Button variant="secondary" size="sm">← Back to Auto Mode</Button>
        </Link>
      </div>
    );
  }

  if (phase === "done") {
    return (
      <>
        {/* Big "records auto-saved" banner with a 10s Undo — fires once
            when the whole sequence finishes and the single combined
            report has been saved. Mirrors BatchSession's banner. */}
        {autoBanner?.kind === "saved" && (
          <div className="fixed inset-x-0 top-4 z-[60] mx-auto flex w-full max-w-2xl justify-center px-4">
            <div className="pointer-events-auto w-full rounded-card border-2 border-emerald-500/60 bg-emerald-600 px-6 py-4 shadow-2xl">
              <div className="flex items-center gap-4">
                <CheckCircle2 className="h-10 w-10 shrink-0 text-white" />
                <div className="min-w-0 flex-1">
                  <p className="text-xl font-bold leading-tight text-white">
                    Records auto-saved ✓
                  </p>
                  <p className="mt-0.5 truncate text-sm text-emerald-50">
                    {savedCount} test{savedCount === 1 ? "" : "s"} saved as one
                    combined report
                    <span className="ml-2 tabular text-emerald-200">
                      · undo closes in {autoBanner.secondsLeft}s
                    </span>
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    onClick={undoAutoSave}
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
                    onClick={() => setAutoBanner(null)}
                    className="rounded-md p-2 text-emerald-100 transition hover:bg-white/15 hover:text-white"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        {autoBanner && autoBanner.kind !== "saved" && (
          <div className="fixed inset-x-0 top-4 z-[60] mx-auto flex w-full max-w-md justify-center px-4">
            <div className="pointer-events-auto flex w-full items-center gap-3 rounded-full border border-border bg-background/95 px-4 py-2.5 text-sm shadow-lg backdrop-blur">
              {autoBanner.kind === "undoing" && (
                <>
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted" />
                  <span className="flex-1 text-foreground">Undoing save…</span>
                </>
              )}
              {autoBanner.kind === "undone" && (
                <>
                  <Undo2 className="h-4 w-4 shrink-0 text-muted" />
                  <span className="flex-1 text-foreground">
                    Save undone — nothing was kept.
                  </span>
                </>
              )}
              {autoBanner.kind === "error" && (
                <>
                  <XCircle className="h-4 w-4 shrink-0 text-error" />
                  <span className="flex-1 text-error">{autoBanner.message}</span>
                </>
              )}
            </div>
          </div>
        )}
        <DoneScreen queue={queue} patientQs={qs} savedCount={savedCount} />
      </>
    );
  }

  return (
    <>
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <Badge>
            Auto Mode · Step {stepIdx + 1} of {queue.length}
          </Badge>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
            {currentStep ? stepTitle(currentStep) : ""}
          </h1>
          {currentStep && (
            <p className="mt-2 text-sm text-muted">{currentStep.description}</p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setPaused((p) => !p)}
          >
            {paused ? (
              <>
                <Play className="h-4 w-4" /> Resume
              </>
            ) : (
              <>
                <Pause className="h-4 w-4" /> Pause
              </>
            )}
          </Button>
          <Button variant="secondary" size="sm" onClick={advance}>
            <SkipForward className="h-4 w-4" />
            Skip
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => router.push(`/biomech/auto${qs}`)}
            className="text-error hover:bg-error/10"
          >
            <XCircle className="h-4 w-4" />
            Stop
          </Button>
        </div>
      </div>

      {/* ── Body ────────────────────────────────────────────── */}
      {phase === "ready" && (
        <ReadyOverlay
          step={currentStep!}
          nextStep={
            stepIdx + 1 < queue.length ? queue[stepIdx + 1] : null
          }
          countdown={readyLeft}
        />
      )}
      {(phase === "running" || phase === "nexting") && currentStep && (() => {
        // Full movement definition from the SAME catalog the
        // standalone live page uses — merged tests get both
        // directions (labels + secondary target), so both angles
        // are captured, not just one peak.
        const move = fullMovement(currentStep);
        const inst = `${stepIdx}:${attempt}`;
        return (
          <div
            className="mt-8"
            key={`${inst}-${currentStep.joint}-${currentStep.movementId}-${currentStep.side ?? "x"}`}
          >
            <LiveAssessment
              bodyPart={currentStep.joint}
              movementId={currentStep.movementId}
              movementLabel={stepTitle(currentStep)}
              movementName={move?.label ?? currentStep.movementLabel}
              description={move?.description ?? currentStep.description}
              target={move?.target ?? currentStep.target}
              side={currentStep.side ?? undefined}
              merged={move?.merged}
              primaryLabel={move?.primaryLabel}
              secondaryLabel={move?.secondaryLabel}
              secondaryTarget={move?.secondaryTarget}
              autoEnter
              // Cumulative mode: don't save this test individually —
              // collect its result (keyed by step index) so the whole
              // sequence saves as ONE combined report at the end.
              deferSave
              onTestResult={(item) => {
                collectedRef.current[stepIdx] = item;
              }}
              onCompleted={handleStepCompleted}
              onRetest={retest}
            />
          </div>
        );
      })()}

      {/* "Test complete → next" card — pinned bottom-right while the
          finished report (+ auto-save banner) is on screen. */}
      {phase === "nexting" && (
        <div className="fixed bottom-6 right-6 z-40 w-72 rounded-card border border-emerald-500/40 bg-background/95 p-4 shadow-2xl backdrop-blur">
          <p className="inline-flex items-center gap-1.5 text-sm font-semibold text-foreground">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            Test complete
          </p>
          <p className="mt-1 text-xs text-muted">
            {stepIdx + 1 < queue.length ? (
              <>
                Next: <span className="text-foreground">{stepTitle(queue[stepIdx + 1])}</span>{" "}
                in <span className="tabular font-semibold text-accent">{nextLeft}s</span>
              </>
            ) : (
              <>
                Finishing in{" "}
                <span className="tabular font-semibold text-accent">{nextLeft}s</span>
              </>
            )}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button size="sm" onClick={advance}>
              <SkipForward className="h-4 w-4" />
              {stepIdx + 1 < queue.length ? "Next now" : "Finish now"}
            </Button>
            <Button variant="secondary" size="sm" onClick={retest}>
              <RotateCcw className="h-4 w-4" />
              Retest
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setPaused((p) => !p)}
            >
              {paused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
              {paused ? "Resume" : "Hold"}
            </Button>
          </div>
          <p className="mt-2 text-[10px] text-muted">
            Retest discards this test&apos;s result and runs it again.
            Nothing is saved until the whole sequence finishes.
          </p>
        </div>
      )}
    </>
  );
}


function ReadyOverlay({
  step,
  nextStep,
  countdown,
}: {
  step: AutoStep;
  nextStep: AutoStep | null;
  countdown: number;
}) {
  return (
    <div className="mt-10 flex flex-col items-center justify-center gap-6 rounded-card border border-border bg-surface py-16 text-center">
      <p className="text-xs uppercase tracking-[0.16em] text-muted">
        Get ready for
      </p>
      <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
        {stepTitle(step)}
      </h2>
      <p className="max-w-md text-sm text-muted">{step.description}</p>
      <div className="mt-2 flex h-24 w-24 items-center justify-center rounded-full bg-accent/10 ring-4 ring-accent/30">
        <span className="tabular text-5xl font-bold text-accent">
          {countdown}
        </span>
      </div>
      {nextStep && (
        <p className="mt-2 text-xs text-muted">
          After this: {stepTitle(nextStep)}
        </p>
      )}
    </div>
  );
}

function DoneScreen({
  queue,
  patientQs,
  savedCount,
}: {
  queue: AutoStep[];
  patientQs: string;
  savedCount: number;
}) {
  return (
    <div className="mx-auto max-w-2xl">
      <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-8 text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/15 ring-2 ring-emerald-500/40">
          <CheckCircle2 className="h-8 w-8 text-emerald-500" />
        </div>
        <h1 className="mt-4 text-2xl font-semibold tracking-tight md:text-3xl">
          Sequence complete
        </h1>
        <p className="mt-2 text-sm text-muted">
          Ran {queue.length} test{queue.length === 1 ? "" : "s"} back to
          back. All {savedCount > 0 ? savedCount : ""} completed test
          {savedCount === 1 ? "" : "s"}&apos; angles (both directions on
          merged tests) + flagged compensations were saved together as{" "}
          <span className="font-medium text-foreground">
            one combined report
          </span>{" "}
          in the patient record.
        </p>

        <ul className="mt-6 space-y-2 text-left">
          {queue.map((s, i) => (
            <li
              key={i}
              className="flex items-center gap-3 rounded-md border border-border bg-background px-4 py-2 text-sm"
            >
              <ChevronRight className="h-4 w-4 shrink-0 text-muted" />
              <span className="font-medium text-foreground">
                {stepTitle(s)}
              </span>
            </li>
          ))}
        </ul>

        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <Link href={`/biomech/auto${patientQs}`}>
            <Button variant="secondary">
              <RotateCcw className="h-4 w-4" />
              Run another set
            </Button>
          </Link>
          <Link href={`/biomech${patientQs}`}>
            <Button variant="ghost">← Biomech</Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
