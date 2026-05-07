"use client";
// 4-Stage Balance Test (Test C4) — capture flow.
//
// Sequential-progression test: side-by-side → semi-tandem → tandem
// → single-leg, 10 s each. Patient progresses to the next stage only
// if the current stage was held for the full 10 s (PDF mandate).
// Test STOPS at first failure — no retry, no skip.
//
// Per-stage state machine inside one capture run:
//
//   "preparing"  — patient is getting into position. We watch the
//                  ankles for the stage's geometry; once valid for
//                  POSITION_LOCK_MS continuously, transition to
//                  "holding". If the patient never gets into position
//                  inside POSITION_TIMEOUT_SEC, fail as position_lost.
//   "holding"    — 10 s countdown. Sway tracked from hip-mid; arm
//                  grab / foot-touchdown / sustained position-drift
//                  fail the stage immediately.
//   "passed"     — 10 s held. Brief hand-off card before advancing
//                  to the next stage. Doctor confirms when the
//                  patient is ready.
//   "failed"     — show the failure card; "Generate report" finalises.
//
// Top-level flow phases:
//
//   "idle"       — patient setup, "Start test" button.
//   "running"    — one of the per-stage phases above is active.
//   "done"       — render the final SessionResult report.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Play,
  RotateCcw,
  XCircle,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { FourStageBalanceLiveCamera } from "@/components/orthopedic/FourStageBalanceLiveCamera";
import { FourStageBalanceReport } from "@/components/orthopedic/FourStageBalanceReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  POSITION_LOCK_MS,
  POSITION_DRIFT_GRACE_MS,
  POSITION_TIMEOUT_SEC,
  SAMPLE_INTERVAL_MS,
  STAGE_HOLD_SEC,
  STAGE_INSTRUCTION,
  STAGE_LABEL,
  ankleMergeWarning,
  buildInterpretation,
  buildSession,
  computeHipMidpoint,
  detectStage4Stance,
  isArmGrab,
  isStage4FootTouchdown,
  isStagePosition,
  readAnkles,
  summarizeStage,
  type FailureMode,
  type FrameSample,
  type SessionResult,
  type StageIndex,
  type StageResult,
} from "@/lib/orthopedic/fourStageBalance";

type StagePhase = "preparing" | "holding" | "passed" | "failed";
type Phase = "idle" | "running" | "done";

interface RunState {
  stage: StageIndex;
  stagePhase: StagePhase;
  /** When the doctor clicked "Start" or the previous stage advanced. */
  preparingSinceMs: number;
  /** First time the stage's geometry was seen valid (resets if drift). */
  positionFirstValidMs: number | null;
  /** Continuously valid since this timestamp — used for the lock-in window. */
  positionContinuousSinceMs: number | null;
  /** When the 10 s hold timer began (after position-lock). */
  holdStartedAtMs: number | null;
  /** Last time we saw a valid position during the hold (drift tracking). */
  lastValidDuringHoldMs: number | null;
  /** Stage 4 only — the foot still on the ground. */
  stage4StanceSide: "left" | "right" | null;
  lastSampleAt: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  hipPath: Array<{ x: number; y: number }>;
  screenshot: string | null;
  pendingFailure: FailureMode | null;
}

const STAGES: readonly StageIndex[] = [1, 2, 3, 4];

export function FourStageBalanceCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [phase, setPhase] = useState<Phase>("idle");
  const [stageResults, setStageResults] = useState<SessionResult["stages"]>({});
  const [error, setError] = useState<string | null>(null);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");
  // Tick the UI for live timers without re-rendering the whole tree.
  const [tick, setTick] = useState<number>(0);

  const runRef = useRef<RunState | null>(null);

  useEffect(() => {
    if (phase !== "running") return;
    const id = window.setInterval(() => setTick((v) => v + 1), 200);
    return () => window.clearInterval(id);
  }, [phase]);

  const setCoachIfChanged = useCallback((msg: string) => {
    if (lastCoachRef.current === msg) return;
    lastCoachRef.current = msg;
    setCoachMsg(msg);
  }, []);

  // ─── Stage finalisation ──────────────────────────────────────
  const finalizeStage = useCallback((outcome: "pass" | "fail", failureMode: FailureMode | null) => {
    const run = runRef.current;
    if (!run) return;

    // Try to grab a screenshot at the moment the stage ended (whether
    // pass or fail) — same fallback pattern as C5.
    if (!run.screenshot) {
      const grab = (window as unknown as {
        __fourStageBalanceCapture?: () => string | null;
      }).__fourStageBalanceCapture;
      if (grab) {
        const url = grab();
        if (url) run.screenshot = url;
      }
    }

    const startedMs = run.holdStartedAtMs ?? run.preparingSinceMs;
    const summary: StageResult = summarizeStage({
      stage: run.stage,
      outcome,
      failureMode,
      startedAtMs: startedMs,
      endedAtMs: Date.now(),
      hipPath: run.hipPath,
      samples: run.samples,
      keypoints: run.keypoints,
      screenshotDataUrl: run.screenshot,
    });

    setStageResults((prev) => ({ ...prev, [run.stage]: summary }));
    if (outcome === "pass") {
      run.stagePhase = "passed";
    } else {
      run.stagePhase = "failed";
      run.pendingFailure = failureMode;
    }
    setCoachMsg("");
    lastCoachRef.current = "";
    // Force a re-render so the pass/fail card appears.
    setTick((v) => v + 1);
  }, []);

  // ─── Per-frame loop ──────────────────────────────────────────
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    if (phase !== "running" || !runRef.current) return;
    const run = runRef.current;
    if (run.stagePhase === "passed" || run.stagePhase === "failed") return;

    const tNow = Date.now();
    if (tNow - run.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    run.lastSampleAt = tNow;

    const ankles = readAnkles(kp);
    const hipMid = computeHipMidpoint(kp);

    // Always record a sample so the saved landmarks JSON covers the
    // whole stage duration (including the preparation window).
    const baseT = run.holdStartedAtMs ?? run.preparingSinceMs;
    run.samples.push({
      t_ms: tNow - baseT,
      hip_x: hipMid?.x ?? null,
      hip_y: hipMid?.y ?? null,
      ankle_l_x: ankles?.lx ?? null,
      ankle_l_y: ankles?.ly ?? null,
      ankle_r_x: ankles?.rx ?? null,
      ankle_r_y: ankles?.ry ?? null,
    });
    run.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Always-on terminations.
    if (isArmGrab(kp)) {
      finalizeStage("fail", "arm_grab");
      return;
    }
    if (run.stage === 4 && run.stage4StanceSide && isStage4FootTouchdown(kp, run.stage4StanceSide)) {
      finalizeStage("fail", "foot_touchdown");
      return;
    }

    // ─── Preparing phase ─────────────────────────────────────
    if (run.stagePhase === "preparing") {
      const elapsed = (tNow - run.preparingSinceMs) / 1000;
      if (!ankles) {
        setCoachIfChanged("Step into the camera frame — ankles not visible.");
        if (elapsed > POSITION_TIMEOUT_SEC) {
          finalizeStage("fail", "position_lost");
        }
        return;
      }
      if (ankleMergeWarning(ankles) && (run.stage === 1 || run.stage === 2)) {
        setCoachIfChanged(
          "Position not detected clearly — please separate your feet slightly so both ankles are visible.",
        );
        run.positionContinuousSinceMs = null;
        if (elapsed > POSITION_TIMEOUT_SEC) {
          finalizeStage("fail", "position_lost");
        }
        return;
      }
      const inPosition = isStagePosition(run.stage, ankles);
      if (inPosition) {
        if (run.positionFirstValidMs === null) run.positionFirstValidMs = tNow;
        if (run.positionContinuousSinceMs === null) run.positionContinuousSinceMs = tNow;
        const lockedFor = tNow - run.positionContinuousSinceMs;
        if (lockedFor >= POSITION_LOCK_MS) {
          // Lock in — start the 10 s hold.
          run.holdStartedAtMs = tNow;
          run.lastValidDuringHoldMs = tNow;
          // Reset the sway buffer so it only contains the hold itself.
          run.hipPath = [];
          if (run.stage === 4) {
            run.stage4StanceSide = detectStage4Stance(ankles);
          }
          run.stagePhase = "holding";
          setCoachIfChanged(
            `Hold steady for ${STAGE_HOLD_SEC} s. Avoid stepping out or reaching for support.`,
          );
          return;
        }
        const remaining = Math.max(0, POSITION_LOCK_MS - lockedFor) / 1000;
        setCoachIfChanged(
          `Hold the ${STAGE_LABEL[run.stage].toLowerCase()} position — locking in (${remaining.toFixed(1)} s)…`,
        );
      } else {
        run.positionContinuousSinceMs = null;
        setCoachIfChanged(STAGE_INSTRUCTION[run.stage]);
        if (elapsed > POSITION_TIMEOUT_SEC) {
          finalizeStage("fail", "position_lost");
          return;
        }
      }
      return;
    }

    // ─── Holding phase ───────────────────────────────────────
    if (run.stagePhase === "holding" && run.holdStartedAtMs !== null) {
      if (hipMid) run.hipPath.push({ x: hipMid.x, y: hipMid.y });

      const heldSec = (tNow - run.holdStartedAtMs) / 1000;

      // Position-drift watch (with grace window).
      if (ankles) {
        const driftOK = isStagePosition(run.stage, ankles);
        if (driftOK) {
          run.lastValidDuringHoldMs = tNow;
        } else if (
          run.lastValidDuringHoldMs !== null &&
          tNow - run.lastValidDuringHoldMs > POSITION_DRIFT_GRACE_MS
        ) {
          finalizeStage("fail", "position_lost");
          return;
        }
      }

      if (heldSec >= STAGE_HOLD_SEC) {
        finalizeStage("pass", null);
        return;
      }

      setCoachIfChanged(
        `Holding — ${heldSec.toFixed(1)} s of ${STAGE_HOLD_SEC} s.`,
      );
    }
  }, [phase, finalizeStage, setCoachIfChanged]);

  // ─── Lifecycle ───────────────────────────────────────────────

  function startTest() {
    setStageResults({});
    setError(null);
    runRef.current = freshRunState(1);
    setCoachMsg(STAGE_INSTRUCTION[1]);
    lastCoachRef.current = STAGE_INSTRUCTION[1];
    setPhase("running");
  }

  function advanceToNextStage() {
    const run = runRef.current;
    if (!run) return;
    if (run.stage === 4) {
      finalizeRun();
      return;
    }
    const next = (run.stage + 1) as StageIndex;
    runRef.current = freshRunState(next);
    setCoachMsg(STAGE_INSTRUCTION[next]);
    lastCoachRef.current = STAGE_INSTRUCTION[next];
    setTick((v) => v + 1);
  }

  function finalizeRun() {
    runRef.current = null;
    setPhase("done");
  }

  function stopEarly() {
    const run = runRef.current;
    if (!run) return;
    if (run.stagePhase === "holding" || run.stagePhase === "preparing") {
      finalizeStage("fail", "stopped");
    }
  }

  function reset() {
    runRef.current = null;
    setStageResults({});
    setPhase("idle");
    setError(null);
    setCoachMsg("");
    lastCoachRef.current = "";
  }

  // ─── Done view ──────────────────────────────────────────────
  if (phase === "done") {
    const session = buildSession(stageResults, patient?.age ?? null);
    const interpretation = buildInterpretation(session);
    return (
      <div className="space-y-8">
        <FourStageBalanceReport
          patientName={patient?.name ?? null}
          session={session}
          interpretation={interpretation}
        />
        <SaveToPatientButton
          buildPayload={() => ({
            module: "four_stage_balance",
            metrics: { session },
            observations: { interpretation },
          })}
        />
        <div className="flex justify-center border-t border-border pt-6">
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
            Run again
          </Button>
        </div>
      </div>
    );
  }

  // ─── Capture view ───────────────────────────────────────────
  const run = runRef.current;
  const heldSec =
    run && run.stagePhase === "holding" && run.holdStartedAtMs
      ? (Date.now() - run.holdStartedAtMs) / 1000
      : 0;

  // Suppress unused warning — `tick` is referenced solely to force
  // re-renders when the live timer advances.
  void tick;

  return (
    <div className="space-y-6">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <div className="rounded-card border border-accent/30 bg-accent/5 p-4 text-sm">
        <p className="font-medium text-foreground">4-Stage Balance Test</p>
        <p className="mt-1 text-muted">
          CDC fall-risk progression. Patient holds 4 progressively harder
          static stances for {STAGE_HOLD_SEC} s each. Test stops at the
          first stage the patient cannot complete — no retries, no skips.
        </p>
      </div>

      {/* Sticky right-aligned camera dock — same convention as C5 / C3. */}
      <div className="sticky top-20 z-20 ml-auto w-full max-w-md rounded-card bg-background/85 p-1 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/65">
        <FourStageBalanceLiveCamera onFrame={handleFrame} onError={setError} />
      </div>

      {/* Stage progression strip — always visible during the run. */}
      <StageProgressStrip
        currentStage={run?.stage ?? null}
        results={stageResults}
      />

      {/* Idle state — start button. */}
      {phase === "idle" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-sm font-medium text-foreground">Ready to begin?</p>
          <p className="mt-1 text-xs text-muted">
            Patient barefoot, level surface, no support within arm&apos;s
            reach. The camera should be at hip height, ~2 m from the
            patient. Start when you&apos;re both ready.
          </p>
          <div className="mt-4">
            <Button onClick={startTest}>
              <Play className="h-4 w-4" />
              Start test
            </Button>
          </div>
        </div>
      )}

      {/* Running — preparing or holding. */}
      {phase === "running" && run && (run.stagePhase === "preparing" || run.stagePhase === "holding") && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-foreground">
              {STAGE_LABEL[run.stage]}{" "}
              <span className="text-muted">·{" "}
                {run.stagePhase === "preparing" ? "Get into position" : "Holding"}
              </span>
            </p>
            {run.stagePhase === "holding" && (
              <p className="tabular text-2xl font-semibold text-accent">
                {heldSec.toFixed(1)}s
              </p>
            )}
          </div>
          {run.stagePhase === "holding" && (
            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
              <div
                className="h-full bg-accent transition-all"
                style={{ width: `${Math.min(100, (heldSec / STAGE_HOLD_SEC) * 100)}%` }}
              />
            </div>
          )}
          {coachMsg && (
            <p className="mt-3 rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
              {coachMsg}
            </p>
          )}
          <div className="mt-3 flex justify-end">
            <Button variant="ghost" size="sm" onClick={stopEarly}>
              Stop test
            </Button>
          </div>
        </div>
      )}

      {/* Stage passed — handoff card. */}
      {phase === "running" && run && run.stagePhase === "passed" && (
        <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <p className="text-sm font-medium text-foreground">
              {STAGE_LABEL[run.stage]} held for the full {STAGE_HOLD_SEC} s.
            </p>
          </div>
          {run.stage < 4 ? (
            <>
              <p className="mt-1 text-xs text-muted">
                Get the patient into the next position, then advance when
                they&apos;re ready.
              </p>
              <div className="mt-3 flex gap-2">
                <Button onClick={advanceToNextStage}>
                  <ChevronRight className="h-4 w-4" />
                  Advance to {STAGE_LABEL[(run.stage + 1) as StageIndex]}
                </Button>
                <Button variant="ghost" onClick={finalizeRun}>
                  Finish here
                </Button>
              </div>
            </>
          ) : (
            <>
              <p className="mt-1 text-xs text-muted">
                All four stages completed.
              </p>
              <div className="mt-3 flex gap-2">
                <Button onClick={finalizeRun}>
                  <Play className="h-4 w-4" />
                  Generate report
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Stage failed — terminal card. */}
      {phase === "running" && run && run.stagePhase === "failed" && (
        <div className="rounded-card border border-red-500/30 bg-red-500/5 p-5">
          <div className="flex items-center gap-2">
            <XCircle className="h-4 w-4 text-red-600" />
            <p className="text-sm font-medium text-foreground">
              {STAGE_LABEL[run.stage]} ended — {failureLabel(run.pendingFailure)}.
            </p>
          </div>
          <p className="mt-1 text-xs text-muted">
            Per the CDC protocol, the test stops at the first stage the
            patient cannot hold for {STAGE_HOLD_SEC} s. Generate the
            report to see the classification.
          </p>
          <div className="mt-3 flex gap-2">
            <Button onClick={finalizeRun}>
              <Play className="h-4 w-4" />
              Generate report
            </Button>
            <Button variant="ghost" onClick={reset}>
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}

      <p className="text-xs text-muted">
        CDC fall-risk thresholds (PDF Test C4): unable to hold tandem
        (stage 3) for {STAGE_HOLD_SEC} s = significantly elevated fall
        risk. Single-leg (stage 4) &lt; 5 s for age &gt; 60 = high
        fall risk. Sway path length and 95% ellipse area are reported
        in pixels (relative units — suitable for trend tracking within
        the same patient).
      </p>
    </div>
  );
}

// ─── Helpers ────────────────────────────────────────────────────

function freshRunState(stage: StageIndex): RunState {
  const now = Date.now();
  return {
    stage,
    stagePhase: "preparing",
    preparingSinceMs: now,
    positionFirstValidMs: null,
    positionContinuousSinceMs: null,
    holdStartedAtMs: null,
    lastValidDuringHoldMs: null,
    stage4StanceSide: null,
    lastSampleAt: 0,
    samples: [],
    keypoints: [],
    hipPath: [],
    screenshot: null,
    pendingFailure: null,
  };
}

function failureLabel(mode: FailureMode | null): string {
  switch (mode) {
    case "foot_touchdown": return "lifted foot returned to the ground";
    case "arm_grab":       return "patient reached for support";
    case "position_lost":  return "stance position drifted out of tolerance";
    case "stopped":        return "operator stopped the test";
    default:               return "stage ended";
  }
}

// ─── Stage progression strip (4 traffic lights) ─────────────────

function StageProgressStrip({
  currentStage,
  results,
}: {
  currentStage: StageIndex | null;
  results: SessionResult["stages"];
}) {
  return (
    <div className="grid grid-cols-4 gap-2">
      {STAGES.map((s) => {
        const r = results[s];
        const isCurrent = currentStage === s && !r;
        let tone: string;
        let icon: React.ReactNode;
        if (r?.outcome === "pass") {
          tone = "border-emerald-500/40 bg-emerald-500/5 text-foreground";
          icon = <CheckCircle2 className="h-4 w-4 text-emerald-600" />;
        } else if (r?.outcome === "fail") {
          tone = "border-red-500/40 bg-red-500/5 text-foreground";
          icon = <XCircle className="h-4 w-4 text-red-600" />;
        } else if (isCurrent) {
          tone = "border-accent/50 bg-accent/5 text-foreground";
          icon = <span className="h-2 w-2 rounded-full bg-accent" />;
        } else {
          tone = "border-border bg-surface text-subtle";
          icon = <span className="h-2 w-2 rounded-full bg-border" />;
        }
        return (
          <div
            key={s}
            className={`rounded-card border p-3 text-xs transition ${tone}`}
          >
            <div className="flex items-center gap-2">
              {icon}
              <p className="font-semibold uppercase tracking-[0.12em]">Stage {s}</p>
            </div>
            <p className="mt-1 text-[11px] leading-tight">
              {STAGE_LABEL[s].split(" · ")[1]}
            </p>
            {r && (
              <p className="mt-1 tabular text-[11px]">
                {r.duration_seconds.toFixed(1)} s
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
