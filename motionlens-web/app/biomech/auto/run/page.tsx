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
import { CheckCircle2, ChevronRight, Pause, Play, RotateCcw, SkipForward, XCircle } from "lucide-react";
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
import { deleteReport } from "@/lib/reports";

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

  // ── Retest bookkeeping ──────────────────────────────────────
  // Each mounted attempt is identified by "stepIdx:attempt". A
  // retest discards that attempt's auto-saved report: if the save
  // already landed we delete it now; if it lands late (save is
  // async) the instance is in `retestedRef` and handleSaved deletes
  // it on arrival. Reports from ADVANCED (kept) steps are never
  // touched.
  const currentInstRef = useRef("0:0");
  useEffect(() => {
    currentInstRef.current = `${stepIdx}:${attempt}`;
  }, [stepIdx, attempt]);
  const retestedRef = useRef<Set<string>>(new Set());
  const lastReportIdRef = useRef<string | null>(null);

  const handleSaved = useCallback((reportId: string, inst: string) => {
    if (retestedRef.current.has(inst)) {
      // Save landed after the operator already discarded the attempt.
      deleteReport(reportId).catch(() => { /* already gone — fine */ });
      return;
    }
    if (inst === currentInstRef.current) {
      lastReportIdRef.current = reportId;
    }
  }, []);

  const advance = useCallback(() => {
    lastReportIdRef.current = null; // advanced step's report is KEPT
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

  // Retest the CURRENT step: delete this attempt's auto-saved
  // report, then remount the step fresh (get-ready → camera →
  // countdown → capture).
  const retest = useCallback(() => {
    const inst = currentInstRef.current;
    retestedRef.current.add(inst);
    const id = lastReportIdRef.current;
    lastReportIdRef.current = null;
    if (id) deleteReport(id).catch(() => { /* already undone — fine */ });
    setAttempt((a) => a + 1);
    setPhase("ready");
    setReadyLeft(READY_COUNTDOWN_SEC);
    setNextLeft(NEXT_COUNTDOWN_SEC);
  }, []);

  // Step completed (rep target hit or manual Show Analysis) — the
  // report + auto-save banner are on screen. Give the operator a
  // beat to see them, then advance.
  const handleStepCompleted = useCallback(() => {
    setNextLeft(NEXT_COUNTDOWN_SEC);
    setPhase((p) => (p === "running" ? "nexting" : p));
  }, []);

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
    return <DoneScreen queue={queue} patientQs={qs} />;
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
              onCompleted={handleStepCompleted}
              onRetest={retest}
              onSaved={(id) => handleSaved(id, inst)}
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
            Retest discards this attempt&apos;s saved report and runs
            the same test again.
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
}: {
  queue: AutoStep[];
  patientQs: string;
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
          back. Each completed test&apos;s angles (both directions on
          merged tests) + flagged compensations were captured and
          auto-saved to the patient record as it finished.
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
