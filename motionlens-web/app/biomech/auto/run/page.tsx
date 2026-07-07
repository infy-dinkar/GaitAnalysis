"use client";
// Biomech Auto Mode — sequence runner.
//
// Reads the queue + duration from the URL and mounts LiveAssessment
// for one step at a time, with a countdown timer that auto-advances
// to the next step. Between steps a short "get ready" overlay
// counts down 3-2-1 before mounting the next assessment so the
// patient has time to re-position.
//
// Additive: LiveAssessment is used as-is; no engine changes.

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
} from "@/lib/biomech/autoModeCatalog";

const READY_COUNTDOWN_SEC = 3;

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
  const durationRaw = params.get("d") ?? "60";
  const patientId = params.get("patientId");
  const qs = patientId ? `?patientId=${patientId}` : "";

  const queue = useMemo(() => decodeQueue(queueRaw), [queueRaw]);
  const duration = useMemo(() => {
    const n = parseInt(durationRaw, 10);
    return Number.isFinite(n) && n > 0 ? n : 60;
  }, [durationRaw]);

  const [stepIdx, setStepIdx] = useState(0);
  const [phase, setPhase] = useState<"ready" | "running" | "done">("ready");
  const [readyLeft, setReadyLeft] = useState(READY_COUNTDOWN_SEC);
  const [runLeft, setRunLeft] = useState(duration);
  const [paused, setPaused] = useState(false);

  const advance = useCallback(() => {
    setStepIdx((prev) => {
      const next = prev + 1;
      if (next >= queue.length) {
        setPhase("done");
        return prev;
      }
      setPhase("ready");
      setReadyLeft(READY_COUNTDOWN_SEC);
      setRunLeft(duration);
      return next;
    });
  }, [queue.length, duration]);

  // ── Ready countdown ─────────────────────────────────────────
  useEffect(() => {
    if (phase !== "ready" || paused) return;
    if (queue.length === 0) return;
    if (readyLeft <= 0) {
      setPhase("running");
      setRunLeft(duration);
      return;
    }
    const id = window.setTimeout(() => setReadyLeft((n) => n - 1), 1000);
    return () => window.clearTimeout(id);
  }, [phase, paused, readyLeft, duration, queue.length]);

  // ── Run countdown ───────────────────────────────────────────
  useEffect(() => {
    if (phase !== "running" || paused) return;
    if (runLeft <= 0) {
      advance();
      return;
    }
    const id = window.setTimeout(() => setRunLeft((n) => n - 1), 1000);
    return () => window.clearTimeout(id);
  }, [phase, paused, runLeft, advance]);

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
      {phase === "running" && currentStep && (
        <div
          className="mt-8"
          key={`${stepIdx}-${currentStep.joint}-${currentStep.movementId}-${currentStep.side ?? "x"}`}
        >
          <LiveAssessment
            bodyPart={currentStep.joint}
            movementId={currentStep.movementId}
            movementLabel={stepTitle(currentStep)}
            description={currentStep.description}
            target={currentStep.target}
            side={currentStep.side ?? undefined}
          />
        </div>
      )}

      {/* Fixed circular countdown — pinned bottom-right so it stays
          visible while the patient is doing the movement. Uses an
          SVG ring that drains as the second-hand ticks. Only
          rendered during the running phase — the "get ready"
          countdown is rendered inline inside ReadyOverlay. */}
      {phase === "running" && (
        <FloatingCountdown
          value={runLeft}
          total={duration}
          paused={paused}
        />
      )}
    </>
  );
}

function FloatingCountdown({
  value,
  total,
  paused,
}: {
  value: number;
  total: number;
  paused: boolean;
}) {
  const size = 84;
  const stroke = 6;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value / total));
  const dashOffset = c * (1 - pct);
  // Late-game colour ramp — green > 50%, amber 20-50%, red < 20%.
  const ringColor =
    pct > 0.5
      ? "rgb(34, 197, 94)"
      : pct > 0.2
        ? "rgb(251, 191, 36)"
        : "rgb(248, 113, 113)";
  return (
    <div className="pointer-events-none fixed bottom-6 right-6 z-40">
      <div
        className="pointer-events-auto flex items-center justify-center rounded-full border border-border bg-background/85 shadow-2xl backdrop-blur"
        style={{ width: size + 8, height: size + 8 }}
        aria-label="Countdown timer"
      >
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className={paused ? "opacity-60" : ""}
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="rgba(255,255,255,0.10)"
            strokeWidth={stroke}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={ringColor}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={c}
            strokeDashoffset={dashOffset}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
            style={{ transition: "stroke-dashoffset 900ms linear, stroke 300ms" }}
          />
          <text
            x="50%"
            y="50%"
            dominantBaseline="central"
            textAnchor="middle"
            fill="currentColor"
            className="tabular fill-foreground"
            style={{ fontSize: 22, fontWeight: 700 }}
          >
            {value}
          </text>
        </svg>
      </div>
    </div>
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
          back. Peak angles + any flagged compensations were tracked
          per test — save each below or head back to run a new set.
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
