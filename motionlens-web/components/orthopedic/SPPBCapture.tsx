"use client";
// SPPB (Short Physical Performance Battery) — orchestrator capture flow.
//
// HYBRID architecture: one component uses backend MediaPipe pose
// detection, two components use frontend MoveNet (live):
//
//   Component 1 (Balance)     → backend MediaPipe BlazePose Full.
//                               Frontend records video only; server
//                               does the heavy stage-detection work
//                               using heel + foot_index landmarks.
//                               Math: sppb_balance_engine.py +
//                               SPPBBalanceRecorder.tsx.
//   Component 2 (Gait Speed)  → MoveNet live (unchanged).
//                               Math: lib/orthopedic/sppbGaitSpeed.ts.
//   Component 3 (Chair Stand) → MoveNet live (unchanged).
//                               Math: reuses C2 sitToStand library.
//
// Why MediaPipe for Component 1 only: the SPPB balance protocol
// distinguishes stages by precise foot positioning (heel-beside-toe
// for semi-tandem, heel-to-toe for tandem). MoveNet has no foot
// landmarks, only ankles. Components 2 and 3 use hip / timing-based
// math that MoveNet handles fine, so they stay on the live path.
//
// Camera positioning differs by component — frontal for balance,
// lateral for gait + chair. The orchestrator displays a clear
// reposition prompt between Component 1 and Component 2.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Loader2,
  Play,
  RotateCcw,
  Square,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { SitToStandLiveCamera } from "@/components/orthopedic/SitToStandLiveCamera";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import { SPPBReport } from "@/components/orthopedic/SPPBReport";
import { SPPBBalanceRecorder } from "@/components/orthopedic/SPPBBalanceRecorder";

// ── Balance (Component 1) — types only ───────────────────────
// Backend produces StageResult-shaped objects; we just need the
// type and a couple of display helpers from the C4 module here.
// All live-pose math (handleBalanceFrame, isStagePosition, etc.)
// is gone — the recorder handles it via the backend pipeline.
import type {
  StageResult,
} from "@/lib/orthopedic/fourStageBalance";
import type { SPPBBalanceDiagnostics } from "@/lib/orthopedic/sppbBalance";

// ── Gait Speed (Component 2) ────────────────────────────────
import {
  PATH_LENGTH_M,
  endGaitTrialManually,
  newGaitTrialState,
  stepGaitTrial,
  trialDurationSec,
  type GaitSpeedTrialState,
} from "@/lib/orthopedic/sppbGaitSpeed";

// ── Chair Stand (Component 3) — reuse C2 math ───────────────
import {
  SAMPLE_INTERVAL_MS as CHAIR_SAMPLE_MS,
  TARGET_REP_COUNT,
  TRIAL_TIMEOUT_SEC,
  areArmsCrossed,
  computeHipMidY as computeChairHipMidY,
  computeKneeAngle as computeChairKneeAngle,
  computeLegLengthPx,
  newRepDetector,
  stepRepDetector,
  summarizeTrial as summarizeChairTrial,
  type FrameSample as ChairFrameSample,
  type RepDetectorState,
  type RepMetrics,
  type SitToStandResult,
  type Termination as ChairTermination,
} from "@/lib/orthopedic/sitToStand";

// ── Composite scoring ────────────────────────────────────────
import {
  buildBalanceComponent,
  buildChairStandComponent,
  buildGaitSpeedComponent,
  buildSPPBResult,
  type GaitSpeedTrial,
  type SPPBResult,
} from "@/lib/orthopedic/sppb";

// ─── State machine types ────────────────────────────────────

type Component = "balance" | "gait" | "chair";

type Phase =
  | "setup"
  | "balance"
  | "transition_to_gait"
  | "gait"
  | "transition_to_chair"
  | "chair"
  | "done";

type GaitTrialPhase = "ready" | "active" | "complete";

interface GaitRunState {
  trial: 1 | 2;
  trialPhase: GaitTrialPhase;
  state: GaitSpeedTrialState;
}

type ChairPhase = "ready" | "armed" | "recording" | "done";

interface ChairRunState {
  startedAt: number;
  lastSampleAt: number;
  samples: ChairFrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  reps: RepMetrics[];
  detector: RepDetectorState;
  armsUncrossedSeen: boolean;
  prevSitMs: number;
  lastRepScreenshot: string | null;
}

// ─── Component ─────────────────────────────────────────────

export function SPPBCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [phase, setPhase] = useState<Phase>("setup");
  const [error, setError] = useState<string | null>(null);
  const [, setTick] = useState(0);

  // Setup confirmation flags
  const [chairReady, setChairReady] = useState(false);
  const [floorReady, setFloorReady] = useState(false);
  const [cameraFrontalReady, setCameraFrontalReady] = useState(false);
  const [cameraLateralReady, setCameraLateralReady] = useState(false);

  // Per-component state + results
  // Component 1 (Balance) — only the result object is needed here;
  // the recorder owns its own state (camera, MediaRecorder, upload).
  const [balanceStages, setBalanceStages] = useState<{
    1?: StageResult;
    2?: StageResult;
    3?: StageResult;
  }>({});
  // Diagnostics from the last backend balance analysis. Surfaced in
  // the result panel when stage detection went poorly (operator gets
  // visibility into WHY detection failed without checking devtools).
  const [balanceDiagnostics, setBalanceDiagnostics] =
    useState<SPPBBalanceDiagnostics | null>(null);
  const gaitRef = useRef<GaitRunState | null>(null);
  const [gaitTrials, setGaitTrials] = useState<{
    1: GaitSpeedTrial | null;
    2: GaitSpeedTrial | null;
  }>({ 1: null, 2: null });
  const chairRef = useRef<ChairRunState | null>(null);
  const [chairResult, setChairResult] = useState<SitToStandResult | null>(null);
  const [chairPhase, setChairPhase] = useState<ChairPhase>("ready");

  // Coach message + live ticker
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  // 5 Hz UI tick during live-pose active phases so live timers
  // refresh. Component 1 (balance) is now backend — its recorder
  // owns its own ticking, so we don't need to drive UI updates here
  // for balance any more. Gait + chair still use MoveNet live.
  useEffect(() => {
    if (phase !== "gait" && phase !== "chair") return;
    const id = window.setInterval(() => setTick((v) => v + 1), 200);
    return () => window.clearInterval(id);
  }, [phase]);

  // ─── Component 1 — Balance (backend MediaPipe) ─────────
  //
  // The recorder owns the camera + MediaRecorder + upload pipeline.
  // We just provide an entry point (startBalance) and a callback the
  // recorder fires once the backend responds with per-stage results.

  function startBalance() {
    setBalanceStages({});
    setPhase("balance");
  }

  // Called by SPPBBalanceRecorder once /api/sppb/balance returns. The
  // payload is already the {1?: StageResult, 2?: StageResult,
  // 3?: StageResult} shape buildBalanceComponent() expects. The
  // diagnostics object goes into separate state so the result panel
  // can surface "what the engine saw" when detection didn't fire.
  const onBalanceAnalyzed = useCallback(
    (
      stages: { 1?: StageResult; 2?: StageResult; 3?: StageResult },
      diagnostics: SPPBBalanceDiagnostics | null,
    ) => {
      setBalanceStages(stages);
      setBalanceDiagnostics(diagnostics);
    },
    [],
  );

  // Advance from balance result view → gait transition. Called by
  // the result panel's "Continue" button after the operator has had a
  // chance to review the per-stage breakdown.
  function advanceFromBalance() {
    setPhase("transition_to_gait");
  }

  // ─── Component 2 — Gait Speed ───────────────────────────

  const handleGaitFrame = useCallback(
    (kp: Keypoint[]) => {
      if (phase !== "gait" || !gaitRef.current) return;
      const run = gaitRef.current;
      if (run.trialPhase !== "active") return;
      const tNow = Date.now();
      const newPhase = stepGaitTrial(run.state, kp, tNow);
      if (newPhase === "done") {
        const dur = trialDurationSec(run.state);
        const t: GaitSpeedTrial = {
          duration_sec: dur ?? 0,
          completed: run.state.auto_completed && dur !== null,
          started_at_ms: run.state.start_ms ?? 0,
        };
        setGaitTrials((prev) => ({ ...prev, [run.trial]: t }));
        run.trialPhase = "complete";
        setTick((v) => v + 1);
      } else {
        // Live coach text — re-render
        setTick((v) => v + 1);
      }
    },
    [phase],
  );

  function startGaitTrial(trial: 1 | 2) {
    gaitRef.current = {
      trial,
      trialPhase: "active",
      state: newGaitTrialState(),
    };
    setTick((v) => v + 1);
  }

  function endGaitTrialNow() {
    const run = gaitRef.current;
    if (!run || run.trialPhase !== "active") return;
    endGaitTrialManually(run.state, Date.now());
    const dur = trialDurationSec(run.state);
    const t: GaitSpeedTrial = {
      duration_sec: dur ?? 0,
      completed: dur !== null,
      started_at_ms: run.state.start_ms ?? 0,
    };
    setGaitTrials((prev) => ({ ...prev, [run.trial]: t }));
    run.trialPhase = "complete";
    setTick((v) => v + 1);
  }

  function advanceFromGait() {
    const run = gaitRef.current;
    if (!run) return;
    if (run.trial === 1) {
      // Start trial 2 immediately (operator can give a rest break)
      startGaitTrial(2);
    } else {
      gaitRef.current = null;
      setPhase("transition_to_chair");
    }
  }

  function startGait() {
    setGaitTrials({ 1: null, 2: null });
    setPhase("gait");
    startGaitTrial(1);
  }

  // ─── Component 3 — Chair Stand ──────────────────────────

  const finalizeChairTrial = useCallback(
    (termination: ChairTermination) => {
      const rec = chairRef.current;
      if (!rec) return;
      const result = summarizeChairTrial(
        rec.startedAt,
        Date.now(),
        termination,
        rec.reps,
        rec.armsUncrossedSeen,
        rec.samples,
        rec.keypoints,
        rec.lastRepScreenshot,
      );
      setChairResult(result);
      setChairPhase("done");
      chairRef.current = null;
    },
    [],
  );

  const handleChairFrame = useCallback(
    (kp: Keypoint[]) => {
      if (phase !== "chair" || chairPhase !== "recording" || !chairRef.current) return;
      const rec = chairRef.current;
      const tNow = Date.now();
      if (tNow - rec.lastSampleAt < CHAIR_SAMPLE_MS) return;
      rec.lastSampleAt = tNow;

      const tMs = tNow - rec.startedAt;
      const hipY = computeChairHipMidY(kp);
      const kneeAngle = computeChairKneeAngle(kp);
      const arms = areArmsCrossed(kp);
      if (!rec.armsUncrossedSeen && !arms) {
        rec.armsUncrossedSeen = true;
      }

      rec.samples.push({
        t_ms: tMs,
        hip_mid_y: hipY,
        knee_angle_deg: kneeAngle,
        arms_crossed: arms,
      });
      rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

      // Lazy-capture leg length once landmarks are stable.
      if (rec.detector.legLengthPx === null) {
        const leg = computeLegLengthPx(kp);
        if (leg !== null) rec.detector.legLengthPx = leg;
      }

      const res = stepRepDetector(rec.detector, hipY, kneeAngle, tMs);
      if (res.completedRep) {
        const prev = rec.prevSitMs;
        const dur = (tMs - prev) / 1000;
        rec.reps.push({
          rep_index: rec.reps.length + 1,
          duration_seconds: dur,
          min_knee_angle_deg: rec.detector.currentMinKneeAngle,
        });
        rec.detector.currentMinKneeAngle = 180;
        rec.prevSitMs = tMs;

        // On final rep, grab screenshot
        if (rec.reps.length === TARGET_REP_COUNT) {
          const grab = (window as unknown as {
            __sitToStandCapture?: () => string | null;
          }).__sitToStandCapture;
          if (grab) {
            const url = grab();
            if (url) rec.lastRepScreenshot = url;
          }
          finalizeChairTrial("completed");
          return;
        }
      }

      // Timeout guard.
      if (tMs / 1000 >= TRIAL_TIMEOUT_SEC) {
        finalizeChairTrial("timeout");
      }
    },
    [phase, chairPhase, finalizeChairTrial],
  );

  function startChairTrial() {
    chairRef.current = {
      startedAt: Date.now(),
      lastSampleAt: 0,
      samples: [],
      keypoints: [],
      reps: [],
      detector: newRepDetector(),
      armsUncrossedSeen: false,
      prevSitMs: 0,
      lastRepScreenshot: null,
    };
    setChairResult(null);
    setChairPhase("recording");
  }

  function stopChairEarly() {
    finalizeChairTrial("stopped");
  }

  function startChair() {
    setChairPhase("ready");
    setPhase("chair");
  }

  // ─── Explicit component switcher ─────────────────────────
  //
  // Lets the operator jump between Component 1, 2, 3 in any order
  // — useful when:
  //   - they need to redo a component the patient didn't perform well
  //   - they want to skip a component the patient can't safely do
  //   - the patient is already positioned for a later component and
  //     they want to grab data while convenient
  //
  // Switching DOES NOT destroy existing results. Each component's
  // result state is preserved; switching to a component that already
  // ran shows its results panel (with a "Re-record" affordance).
  //
  // Switching AWAY from an active recording effectively cancels it
  // (the live camera / MediaRecorder unmounts), which is the
  // expected behaviour — the operator made a deliberate switch.
  function switchToComponent(c: Component) {
    if (c === "balance") {
      if (phase === "balance") return;
      setPhase("balance");
      return;
    }
    if (c === "gait") {
      if (phase === "gait") return;
      if (gaitRef.current === null) {
        if (gaitTrials[2] !== null) {
          // Both trials complete — restore the "trial-2 complete"
          // view so the operator sees the Continue / Redo buttons.
          gaitRef.current = {
            trial: 2,
            trialPhase: "complete",
            state: newGaitTrialState(),
          };
        } else if (gaitTrials[1] !== null) {
          // Trial 1 done, Trial 2 not started yet — show the
          // "trial-1 complete" panel which advances to Trial 2.
          gaitRef.current = {
            trial: 1,
            trialPhase: "complete",
            state: newGaitTrialState(),
          };
        } else {
          // Nothing recorded yet — start fresh on Trial 1.
          startGaitTrial(1);
        }
      }
      setPhase("gait");
      return;
    }
    if (c === "chair") {
      if (phase === "chair") return;
      // First time visiting chair — ensure setup phase shows.
      if (chairRef.current === null && chairResult === null) {
        setChairPhase("ready");
      } else if (chairResult !== null) {
        // Returning to view existing results.
        setChairPhase("done");
      }
      setPhase("chair");
      return;
    }
  }

  // Used by the per-component "Re-record" buttons to clear that
  // component's results and re-enter its capture flow.
  function restartBalance() {
    setBalanceStages({});
    setBalanceDiagnostics(null);
    setPhase("balance");
  }

  function restartGait() {
    gaitRef.current = null;
    setGaitTrials({ 1: null, 2: null });
    startGaitTrial(1);
    setPhase("gait");
  }

  function restartChair() {
    chairRef.current = null;
    setChairResult(null);
    setChairPhase("ready");
    setPhase("chair");
  }

  // ─── Final composite ────────────────────────────────────

  function buildResult(): SPPBResult | null {
    if (!chairResult) return null;
    const balance = buildBalanceComponent(balanceStages);
    const gait = buildGaitSpeedComponent(gaitTrials[1], gaitTrials[2]);
    const chair = buildChairStandComponent(chairResult);
    return buildSPPBResult(balance, gait, chair, patient?.age ?? null);
  }

  function reset() {
    gaitRef.current = null;
    chairRef.current = null;
    setBalanceStages({});
    setBalanceDiagnostics(null);
    setGaitTrials({ 1: null, 2: null });
    setChairResult(null);
    setChairPhase("ready");
    setCoachMsg("");
    lastCoachRef.current = "";
    setError(null);
    setChairReady(false);
    setFloorReady(false);
    setCameraFrontalReady(false);
    setCameraLateralReady(false);
    setPhase("setup");
  }

  // ─── Render ─────────────────────────────────────────────

  if (phase === "done") {
    const result = buildResult();
    if (!result) {
      return (
        <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          Could not build SPPB result — chair-stand component missing data.
        </div>
      );
    }
    return (
      <div className="space-y-8">
        <SPPBReport
          patient={patient ?? null}
          patientName={patient?.name ?? null}
          result={result}
        />
        <SaveToPatientButton
          buildPayload={() => ({
            module: "sppb",
            metrics: { result },
            observations: {
              interpretation: result.interpretation,
              recommendation: result.recommendation,
            },
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

  return (
    <div className="space-y-8">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <ProgressStrip phase={phase} />

      {/* phase === "done" is handled by the early-return above, so by
          the time we reach this render branch `phase` can never be
          "done". Only "setup" needs to be excluded here. */}
      {phase !== "setup" && (
        <ComponentSwitcher
          phase={phase}
          balanceStages={balanceStages}
          gaitTrials={gaitTrials}
          chairResult={chairResult}
          onSwitch={switchToComponent}
        />
      )}

      {phase === "setup" && (
        <SetupCard
          chairReady={chairReady}
          setChairReady={setChairReady}
          floorReady={floorReady}
          setFloorReady={setFloorReady}
          cameraFrontalReady={cameraFrontalReady}
          setCameraFrontalReady={setCameraFrontalReady}
          canStart={chairReady && floorReady && cameraFrontalReady}
          onStart={startBalance}
        />
      )}

      {phase === "balance" && (
        <BalanceComponent
          stages={balanceStages}
          diagnostics={balanceDiagnostics}
          onAnalyzed={onBalanceAnalyzed}
          onAdvance={advanceFromBalance}
          onRestart={restartBalance}
        />
      )}

      {phase === "transition_to_gait" && (
        <TransitionToGait
          balanceStages={balanceStages}
          cameraLateralReady={cameraLateralReady}
          setCameraLateralReady={setCameraLateralReady}
          onContinue={startGait}
        />
      )}

      {phase === "gait" && (
        <GaitComponent
          gaitRef={gaitRef}
          trials={gaitTrials}
          onFrame={handleGaitFrame}
          onError={setError}
          onEndManually={endGaitTrialNow}
          onAdvanceTrial={advanceFromGait}
          onRestart={restartGait}
        />
      )}

      {phase === "transition_to_chair" && (
        <TransitionToChair
          gaitTrials={gaitTrials}
          onContinue={startChair}
        />
      )}

      {phase === "chair" && (
        <ChairComponent
          phase={chairPhase}
          result={chairResult}
          onFrame={handleChairFrame}
          onError={setError}
          onStart={() => setChairPhase("armed")}
          onGo={startChairTrial}
          onStopEarly={stopChairEarly}
          onFinish={() => setPhase("done")}
          onRestart={restartChair}
          coachMsg={coachMsg}
        />
      )}

      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}
    </div>
  );
}

// ─── Setup card ────────────────────────────────────────────

function SetupCard(props: {
  chairReady: boolean;
  setChairReady: (b: boolean) => void;
  floorReady: boolean;
  setFloorReady: (b: boolean) => void;
  cameraFrontalReady: boolean;
  setCameraFrontalReady: (b: boolean) => void;
  canStart: boolean;
  onStart: () => void;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-5">
      <p className="text-sm font-semibold text-foreground">SPPB setup</p>
      <p className="mt-1 text-xs text-muted">
        SPPB combines 3 tests in one session — balance, gait speed, and
        chair stand. Confirm the room is ready before starting.
      </p>
      <div className="mt-4 space-y-3">
        <SetupCheck
          checked={props.chairReady}
          onChange={props.setChairReady}
          label="Chair is positioned and ready (~45 cm seat, no armrests)."
        />
        <SetupCheck
          checked={props.floorReady}
          onChange={props.setFloorReady}
          label="4-metre floor distance is marked (line / tape / cone at the 4 m point)."
        />
        <SetupCheck
          checked={props.cameraFrontalReady}
          onChange={props.setCameraFrontalReady}
          label="Camera is in FRONT VIEW position (patient will face the camera for the balance test)."
        />
      </div>
      <div className="mt-4">
        <Button onClick={props.onStart} disabled={!props.canStart}>
          <Play className="h-4 w-4" />
          Start SPPB
        </Button>
      </div>
    </div>
  );
}

function SetupCheck({
  checked, onChange, label,
}: {
  checked: boolean;
  onChange: (b: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 text-sm">
      <input
        type="checkbox"
        className="mt-0.5 h-4 w-4 shrink-0 rounded border-border accent-accent"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="text-foreground">{label}</span>
    </label>
  );
}

// ─── Component 1 panel — backend MediaPipe ─────────────────

function BalanceComponent({
  stages,
  diagnostics,
  onAnalyzed,
  onAdvance,
  onRestart,
}: {
  stages: { 1?: StageResult; 2?: StageResult; 3?: StageResult };
  diagnostics: SPPBBalanceDiagnostics | null;
  onAnalyzed: (
    s: { 1?: StageResult; 2?: StageResult; 3?: StageResult },
    d: SPPBBalanceDiagnostics | null,
  ) => void;
  onAdvance: () => void;
  onRestart: () => void;
}) {
  const haveResult =
    stages[1] !== undefined || stages[2] !== undefined || stages[3] !== undefined;

  return (
    <div className="space-y-4">
      <ComponentHeader
        number={1}
        title="Balance — record"
        subtitle="3 stages × 10 s · backend pose analysis"
      />

      {!haveResult && <SPPBBalanceRecorder onComplete={onAnalyzed} />}

      {haveResult && (
        <BalanceResultPanel
          stages={stages}
          diagnostics={diagnostics}
          onAdvance={onAdvance}
          onRestart={onRestart}
        />
      )}
    </div>
  );
}

function BalanceResultPanel({
  stages,
  diagnostics,
  onAdvance,
  onRestart,
}: {
  stages: { 1?: StageResult; 2?: StageResult; 3?: StageResult };
  diagnostics: SPPBBalanceDiagnostics | null;
  onAdvance: () => void;
  onRestart: () => void;
}) {
  const stagesPassed = [1, 2, 3].filter(
    (s) => stages[s as 1 | 2 | 3]?.outcome === "pass",
  ).length;
  const totalAttempted = [1, 2, 3].filter(
    (s) => stages[s as 1 | 2 | 3] !== undefined &&
           stages[s as 1 | 2 | 3]?.outcome !== "not_attempted",
  ).length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        {([1, 2, 3] as const).map((s) => {
          const r = stages[s];
          const tone =
            r?.outcome === "pass"
              ? "border-emerald-500/40 bg-emerald-500/5"
              : r?.outcome === "fail"
                ? "border-red-500/40 bg-red-500/5"
                : "border-border bg-surface";
          const labels: Record<number, string> = {
            1: "Side-by-side",
            2: "Semi-tandem",
            3: "Tandem",
          };
          return (
            <div key={s} className={`rounded-card border p-3 text-xs ${tone}`}>
              <p className="font-semibold uppercase tracking-[0.12em] text-foreground">
                Stage {s}
              </p>
              <p className="mt-1 text-[11px] text-muted">{labels[s]}</p>
              {r ? (
                <p className="mt-1 tabular text-[11px] text-foreground">
                  {r.duration_seconds.toFixed(1)} s ·{" "}
                  {r.outcome === "pass"
                    ? "Pass"
                    : r.outcome === "fail"
                      ? "Fail"
                      : "Not attempted"}
                </p>
              ) : (
                <p className="mt-1 tabular text-[11px] text-muted">Not attempted</p>
              )}
            </div>
          );
        })}
      </div>

      {totalAttempted === 0 ? (
        // The engine couldn't detect ANY of the three stage positions.
        // This is almost always a setup / framing issue rather than a
        // patient-balance issue; surface diagnostic detail so the
        // operator can fix the cause and re-record without guessing.
        <NoStagesDetectedPanel
          diagnostics={diagnostics}
          onAdvance={onAdvance}
          onRestart={onRestart}
        />
      ) : (
        <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <p className="text-sm font-medium text-foreground">
              Balance analysed — {stagesPassed} of {totalAttempted} attempted stage
              {totalAttempted === 1 ? "" : "s"} held the full 10 s.
            </p>
          </div>
          <p className="mt-1 text-xs text-muted">
            The composite SPPB score will be computed after all three components
            finish.
          </p>

          {/* When not every stage passed, surface the per-stage
              longest-run info so the operator can see what the engine
              detected for each (including stages that progression
              blocked — those wouldn't otherwise appear in the
              outcome cards above). */}
          {stagesPassed < 3 && diagnostics?.longest_runs_per_stage && (
            <div className="mt-3 rounded-md border border-border/60 bg-background/60 p-3 text-[11px]">
              <p className="font-semibold uppercase tracking-[0.12em] text-subtle">
                Longest run detected per stage
              </p>
              <div className="mt-2 grid grid-cols-3 gap-3 text-muted">
                {([1, 2, 3] as const).map((s) => {
                  const lr = diagnostics.longest_runs_per_stage![`${s}` as "1" | "2" | "3"];
                  return (
                    <div key={s}>
                      <span className="text-foreground">Stage {s}:</span>{" "}
                      <span className="tabular">
                        {lr.seconds.toFixed(1)} s ({lr.frames} fr)
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="mt-2 text-[10px] text-subtle">
                A stage needs 10 s of contiguous detection to count as a Pass.
                If a stage shows time here but is &quot;Not attempted&quot; above, it&apos;s
                because SPPB protocol stops progression at the first failed stage.
              </p>
            </div>
          )}

          <div className="mt-3 flex flex-wrap gap-2">
            <Button onClick={onAdvance}>
              <ArrowRight className="h-4 w-4" />
              Continue to gait speed
            </Button>
            <Button variant="secondary" onClick={onRestart}>
              <RotateCcw className="h-4 w-4" />
              Re-record balance
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function NoStagesDetectedPanel({
  diagnostics,
  onAdvance,
  onRestart,
}: {
  diagnostics: SPPBBalanceDiagnostics | null;
  onAdvance: () => void;
  onRestart: () => void;
}) {
  // Diagnose the most likely cause given what the engine reported.
  // Three failure modes we know about:
  //   A. Feet not visible (visible_foot_ratio low) → framing problem
  //   B. Feet visible but geometry didn't match any stage → patient
  //      didn't perform the protocol clearly, or thresholds need
  //      further tuning
  //   C. Some frames classified but runs too short → patient changed
  //      positions too quickly
  let likelyCause = "Unknown — check diagnostics below.";
  let suggestion = "Re-record with the camera positioned so the patient's full body is visible head-to-toes.";
  const fc = diagnostics?.frame_classification_counts;
  if (diagnostics && fc) {
    const totalClassified = fc.stage_1 + fc.stage_2 + fc.stage_3;
    if (diagnostics.visible_foot_ratio < 0.30) {
      likelyCause = "The patient's feet were not visible in most frames.";
      suggestion = "Frame the FULL body — head to toes — and check that the feet stay in view throughout the test.";
    } else if (totalClassified === 0) {
      likelyCause = "Feet were visible but the engine couldn't match any frame to Stage 1, 2, or 3 geometry.";
      suggestion =
        "Make sure each stance is held still for ~10 s, and that the patient is facing the camera squarely (not at an angle). Stages: feet-together → semi-tandem (heel-beside-toe) → tandem (heel-to-toe).";
    } else if (totalClassified < (diagnostics.min_run_frames ?? 3)) {
      likelyCause = `Found ${totalClassified} matching frames total, but none of the stages were held long enough to count.`;
      suggestion = "Hold each stance still for the full 10 s before transitioning to the next stage.";
    }
  }

  return (
    <div className="rounded-card border border-amber-500/40 bg-amber-500/5 p-5">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div>
          <p className="text-sm font-medium text-foreground">
            No balance stages were detected in the recording.
          </p>
          <p className="mt-1 text-xs text-muted">
            <span className="text-foreground">Likely cause:</span> {likelyCause}
          </p>
          <p className="mt-2 text-xs text-foreground">
            <span className="font-medium">Suggestion:</span> {suggestion}
          </p>
        </div>
      </div>

      {diagnostics && (
        <div className="mt-3 space-y-3">
          <div className="rounded-md border border-border/60 bg-background/60 p-3 text-[11px]">
            <p className="font-semibold uppercase tracking-[0.12em] text-subtle">
              What the engine saw
            </p>
            <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-muted">
              <span>Frames with feet visible:</span>
              <span className="tabular text-foreground">
                {diagnostics.visible_foot_frames} ({(diagnostics.visible_foot_ratio * 100).toFixed(0)} %)
              </span>
              <span>Frames classified as Stage 1:</span>
              <span className="tabular text-foreground">{diagnostics.frame_classification_counts.stage_1}</span>
              <span>Frames classified as Stage 2:</span>
              <span className="tabular text-foreground">{diagnostics.frame_classification_counts.stage_2}</span>
              <span>Frames classified as Stage 3:</span>
              <span className="tabular text-foreground">{diagnostics.frame_classification_counts.stage_3}</span>
              <span>Frames unclassified:</span>
              <span className="tabular text-foreground">{diagnostics.frame_classification_counts.unclassified}</span>
              {diagnostics.body_h_failed_count !== undefined && diagnostics.body_h_failed_count > 0 && (
                <>
                  <span>Frames body-height failed:</span>
                  <span className="tabular text-foreground">{diagnostics.body_h_failed_count}</span>
                </>
              )}
              {diagnostics.geometry_unmatched_count !== undefined && (
                <>
                  <span>Frames geometry didn&apos;t match:</span>
                  <span className="tabular text-foreground">{diagnostics.geometry_unmatched_count}</span>
                </>
              )}
              <span>Min run required (frames):</span>
              <span className="tabular text-foreground">{diagnostics.min_run_frames}</span>
              {diagnostics.frame_width !== undefined && diagnostics.frame_height !== undefined && (
                <>
                  <span>Video resolution:</span>
                  <span className="tabular text-foreground">
                    {diagnostics.frame_width} × {diagnostics.frame_height} px
                    {(diagnostics.frame_width < 640 || diagnostics.frame_height < 480) && (
                      <span className="ml-2 text-warning">⚠ low</span>
                    )}
                  </span>
                </>
              )}
            </div>
          </div>

          {diagnostics.dx_heel_n && diagnostics.thresholds && (
            <div className="rounded-md border border-border/60 bg-background/60 p-3 text-[11px]">
              <p className="font-semibold uppercase tracking-[0.12em] text-subtle">
                Foot-geometry measurements (body-height fraction)
              </p>
              <p className="mt-1 text-[10px] text-subtle">
                The classifier checks if these per-frame values fall in
                the threshold ranges below. If a measurement&apos;s median
                sits outside every range, no frame matches a stage.
              </p>
              <table className="mt-2 w-full text-left tabular text-muted">
                <thead className="text-[10px] uppercase tracking-[0.08em] text-subtle">
                  <tr>
                    <th className="py-1 font-medium">Measurement</th>
                    <th className="py-1 font-medium">min</th>
                    <th className="py-1 font-medium">median</th>
                    <th className="py-1 font-medium">max</th>
                    <th className="py-1 font-medium">Threshold range</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="py-1 text-foreground">dx_heel</td>
                    <td className="py-1">{diagnostics.dx_heel_n.min?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">{diagnostics.dx_heel_n.median?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">{diagnostics.dx_heel_n.max?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">
                      Stage 1: {diagnostics.thresholds.stage1_x_min}–{diagnostics.thresholds.stage1_x_max}
                    </td>
                  </tr>
                  <tr>
                    <td className="py-1 text-foreground">dy_heel</td>
                    <td className="py-1">{diagnostics.dy_heel_n?.min?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">{diagnostics.dy_heel_n?.median?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">{diagnostics.dy_heel_n?.max?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">
                      S1 &lt; {diagnostics.thresholds.stage_dy_tight} · S2 {diagnostics.thresholds.stage_dy_med_min}–{diagnostics.thresholds.stage_dy_med_max} · S3 &gt; {diagnostics.thresholds.stage_dy_large}
                    </td>
                  </tr>
                  <tr>
                    <td className="py-1 text-foreground">dx_tandem</td>
                    <td className="py-1">{diagnostics.dx_tandem_n?.min?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">{diagnostics.dx_tandem_n?.median?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">{diagnostics.dx_tandem_n?.max?.toFixed(3) ?? "—"}</td>
                    <td className="py-1">
                      S2 &lt; {diagnostics.thresholds.semi_x} · S3 &lt; {diagnostics.thresholds.tandem_x}
                    </td>
                  </tr>
                  {diagnostics.body_h_px && (
                    <tr>
                      <td className="py-1 text-foreground">body_h_px</td>
                      <td className="py-1">{diagnostics.body_h_px.min?.toFixed(0) ?? "—"}</td>
                      <td className="py-1">{diagnostics.body_h_px.median?.toFixed(0) ?? "—"}</td>
                      <td className="py-1">{diagnostics.body_h_px.max?.toFixed(0) ?? "—"}</td>
                      <td className="py-1">— (raw px)</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <Button onClick={onRestart}>
          <RotateCcw className="h-4 w-4" />
          Re-record balance
        </Button>
        <Button variant="secondary" onClick={onAdvance}>
          <ArrowRight className="h-4 w-4" />
          Skip — continue to gait speed
        </Button>
      </div>
    </div>
  );
}

// ─── Transitions ────────────────────────────────────────────

function TransitionToGait({
  balanceStages,
  cameraLateralReady,
  setCameraLateralReady,
  onContinue,
}: {
  balanceStages: { 1?: StageResult; 2?: StageResult; 3?: StageResult };
  cameraLateralReady: boolean;
  setCameraLateralReady: (b: boolean) => void;
  onContinue: () => void;
}) {
  const passed = [1, 2, 3].filter(
    (s) => balanceStages[s as 1 | 2 | 3]?.outcome === "pass",
  ).length;
  return (
    <div className="space-y-4">
      <ComponentHeader number={1} title="Balance — complete" />
      <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-5">
        <p className="text-sm font-medium text-foreground">
          Balance component complete. {passed} of 3 stages passed.
        </p>
      </div>
      <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
        <p className="text-base font-semibold text-foreground">
          Reposition camera to SIDE VIEW
        </p>
        <p className="mt-2 text-sm text-muted">
          The next test (gait speed) needs a lateral camera. Place the
          camera perpendicular to the walking path so the entire 4 m is
          visible end-to-end.
        </p>
        <label className="mt-4 flex cursor-pointer items-start gap-3 text-sm">
          <input
            type="checkbox"
            className="mt-0.5 h-4 w-4 shrink-0 rounded border-border accent-accent"
            checked={cameraLateralReady}
            onChange={(e) => setCameraLateralReady(e.target.checked)}
          />
          <span className="text-foreground">
            Camera is now in side-view position and the full 4 m walking path is in frame.
          </span>
        </label>
        <div className="mt-4">
          <Button onClick={onContinue} disabled={!cameraLateralReady}>
            <ArrowRight className="h-4 w-4" />
            Start gait speed
          </Button>
        </div>
      </div>
    </div>
  );
}

function TransitionToChair({
  gaitTrials,
  onContinue,
}: {
  gaitTrials: { 1: GaitSpeedTrial | null; 2: GaitSpeedTrial | null };
  onContinue: () => void;
}) {
  return (
    <div className="space-y-4">
      <ComponentHeader number={2} title="Gait speed — complete" />
      <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-5">
        <p className="text-sm font-medium text-foreground">
          Gait speed component complete.
        </p>
        <div className="mt-2 grid grid-cols-2 gap-3 text-xs text-muted">
          <p>
            Trial 1: {gaitTrials[1] ? `${gaitTrials[1].duration_sec.toFixed(2)} s` : "—"}
          </p>
          <p>
            Trial 2: {gaitTrials[2] ? `${gaitTrials[2].duration_sec.toFixed(2)} s` : "—"}
          </p>
        </div>
      </div>
      <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
        <p className="text-base font-semibold text-foreground">
          Camera stays in side view
        </p>
        <p className="mt-2 text-sm text-muted">
          The chair-stand test uses the same lateral camera setup as
          gait speed. Position the patient in the chair, arms crossed
          at the chest, back against the backrest.
        </p>
        <div className="mt-4">
          <Button onClick={onContinue}>
            <ArrowRight className="h-4 w-4" />
            Start chair stand
          </Button>
        </div>
      </div>
    </div>
  );
}

// ─── Component 2 panel ─────────────────────────────────────

function GaitComponent({
  gaitRef,
  trials,
  onFrame,
  onError,
  onEndManually,
  onAdvanceTrial,
  onRestart,
}: {
  gaitRef: React.MutableRefObject<GaitRunState | null>;
  trials: { 1: GaitSpeedTrial | null; 2: GaitSpeedTrial | null };
  onFrame: (kp: Keypoint[]) => void;
  onError: (m: string) => void;
  onEndManually: () => void;
  onAdvanceTrial: () => void;
  onRestart: () => void;
}) {
  const run = gaitRef.current;
  const trialNum = run?.trial ?? 1;
  const tPhase = run?.state.phase;
  const elapsedSec = run?.state.start_ms
    ? (Date.now() - run.state.start_ms) / 1000
    : 0;

  return (
    <div className="space-y-4">
      <ComponentHeader number={2} title={`Gait speed — Trial ${trialNum}`} subtitle="4 m walk at usual pace" />

      <div className="sticky top-20 z-20 ml-auto w-full max-w-md rounded-card bg-background/85 p-1 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/65">
        <SitToStandLiveCamera onFrame={onFrame} onError={onError} />
      </div>

      {/* Protocol — what the patient needs to do. SPPB-spec aligned
          (Guralnik 1994); kept visible throughout the trial so the
          operator can read each step to the patient before the
          'Go' cue. */}
      <div className="rounded-card border border-border bg-surface p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Patient instructions — read these out before Trial 1
        </p>
        <ol className="mt-3 space-y-2 text-sm text-foreground">
          <li className="flex gap-2">
            <span className="tabular shrink-0 text-accent">1.</span>
            <span>
              <strong>Stand behind the START line</strong>, feet together,
              arms relaxed at your sides. Use your usual walking aid
              (cane / walker) if you normally use one — note in the
              chart that an aid was used.
            </span>
          </li>
          <li className="flex gap-2">
            <span className="tabular shrink-0 text-accent">2.</span>
            <span>
              On <strong>&quot;Go&quot;</strong>, walk straight to the marker at the
              end of the 4-metre path — <strong>at your usual, comfortable
              pace</strong>. Don&apos;t hurry, don&apos;t slow down on purpose.
            </span>
          </li>
          <li className="flex gap-2">
            <span className="tabular shrink-0 text-accent">3.</span>
            <span>
              <strong>Do not stop until you pass the 4 m marker.</strong>
              You may continue a step or two past the marker — the timer
              ends when motion stops.
            </span>
          </li>
          <li className="flex gap-2">
            <span className="tabular shrink-0 text-accent">4.</span>
            <span>
              Two trials total. The faster of the two is used. There is
              a 30-second rest break between trials — sit if you need to.
            </span>
          </li>
        </ol>
        <p className="mt-3 rounded-md border border-warning/30 bg-warning/5 p-2 text-[11px] text-warning">
          <strong>Stop the trial immediately</strong> if the patient loses
          balance, has chest pain, severe shortness of breath, or needs
          to reach for support.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {([1, 2] as const).map((t) => {
          const tr = trials[t];
          const tone = tr
            ? "border-emerald-500/40 bg-emerald-500/5"
            : run?.trial === t
              ? "border-accent/40 bg-accent/5"
              : "border-border bg-surface";
          return (
            <div key={t} className={`rounded-card border p-3 text-xs ${tone}`}>
              <p className="font-semibold uppercase tracking-[0.12em] text-foreground">
                Trial {t}
              </p>
              <p className="mt-1 tabular text-[11px] text-foreground">
                {tr ? `${tr.duration_sec.toFixed(2)} s · ${tr.completed ? "auto" : "manual"}` : "—"}
              </p>
            </div>
          );
        })}
      </div>

      {run && run.trialPhase === "active" && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-foreground">
              Trial {trialNum} ·{" "}
              {tPhase === "walking"
                ? "Walking"
                : tPhase === "baselined"
                  ? "Ready — give the 'Go' cue when patient is set"
                  : "Capturing baseline — patient stand still on the start line"}
            </p>
            {tPhase === "walking" && (
              <p className="tabular text-2xl font-semibold text-accent">
                {elapsedSec.toFixed(2)}s
              </p>
            )}
          </div>
          <p className="mt-2 text-xs text-muted">
            {tPhase === "waiting_baseline" &&
              "Tell the patient: “Stand still on the START line, feet together, arms relaxed.” Locking start position…"}
            {tPhase === "baselined" &&
              "Tell the patient: “On 'Go', walk at your usual pace to the marker. Don't stop until you pass it.” Then give the cue: “Ready, set, GO!”"}
            {tPhase === "walking" &&
              "Patient is walking. Timer auto-ends once motion stops past the 4 m marker. Click End walk to stop manually."}
          </p>
          {tPhase === "walking" && (
            <div className="mt-3">
              <Button variant="secondary" size="sm" onClick={onEndManually}>
                <Square className="h-4 w-4" />
                End walk
              </Button>
            </div>
          )}
        </div>
      )}

      {run && run.trialPhase === "complete" && (
        <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <p className="text-sm font-medium text-foreground">
              Trial {trialNum} recorded: {(trials[trialNum]?.duration_sec ?? 0).toFixed(2)} s
            </p>
          </div>
          <p className="mt-1 text-xs text-muted">
            {trialNum === 1
              ? "Patient may sit and rest for ~30 seconds. Reset to the start line, then begin Trial 2."
              : "Both trials recorded. Better-of-two will be used for SPPB scoring."}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button onClick={onAdvanceTrial}>
              <ChevronRight className="h-4 w-4" />
              {trialNum === 1 ? "Start Trial 2" : "Continue to chair stand"}
            </Button>
            <Button variant="secondary" onClick={onRestart}>
              <RotateCcw className="h-4 w-4" />
              {trialNum === 1 ? "Redo Trial 1" : "Redo gait speed"}
            </Button>
          </div>
        </div>
      )}

      <p className="text-xs text-muted">
        Walking speed = {PATH_LENGTH_M.toFixed(1)} m / best trial time.
        Better (faster) trial is used for scoring per SPPB protocol
        (Guralnik 1994). Spec cutoffs: ≥ 0.77 m/s = 4 pts · 0.60-0.76 = 3 ·
        0.43-0.59 = 2 · &lt; 0.43 = 1 · unable = 0.
      </p>
    </div>
  );
}

// ─── Component 3 panel ─────────────────────────────────────

function ChairComponent({
  phase,
  result,
  onFrame,
  onError,
  onStart,
  onGo,
  onStopEarly,
  onFinish,
  onRestart,
  coachMsg,
}: {
  phase: ChairPhase;
  result: SitToStandResult | null;
  onFrame: (kp: Keypoint[]) => void;
  onError: (m: string) => void;
  onStart: () => void;
  onGo: () => void;
  onStopEarly: () => void;
  onFinish: () => void;
  onRestart: () => void;
  coachMsg: string;
}) {
  return (
    <div className="space-y-4">
      <ComponentHeader number={3} title="Chair stand" subtitle="5x sit-to-stand, arms crossed" />

      <div className="sticky top-20 z-20 ml-auto w-full max-w-md rounded-card bg-background/85 p-1 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/65">
        <SitToStandLiveCamera onFrame={onFrame} onError={onError} />
      </div>

      {phase === "ready" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-sm font-semibold text-foreground">
            Patient setup &amp; protocol
          </p>
          <p className="mt-1 text-xs text-muted">
            5x sit-to-stand test (Guralnik 1994). Read these instructions
            to the patient before clicking Arm timer.
          </p>

          <p className="mt-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Starting position
          </p>
          <ol className="mt-2 space-y-2 text-sm text-foreground">
            <li className="flex gap-2">
              <span className="tabular shrink-0 text-accent">1.</span>
              <span>
                <strong>Sit in the chair</strong>, back against the backrest,
                feet flat on the floor (not on tiptoes, not crossed).
              </span>
            </li>
            <li className="flex gap-2">
              <span className="tabular shrink-0 text-accent">2.</span>
              <span>
                <strong>Cross your arms over your chest</strong> — each hand
                on the opposite shoulder. Keep them there the whole time.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="tabular shrink-0 text-accent">3.</span>
              <span>
                The chair must be a firm, armless chair (~45 cm seat
                height). Make sure it doesn&apos;t slide — back the chair
                against a wall if needed.
              </span>
            </li>
          </ol>

          <p className="mt-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            The test
          </p>
          <ol className="mt-2 space-y-2 text-sm text-foreground">
            <li className="flex gap-2">
              <span className="tabular shrink-0 text-accent">4.</span>
              <span>
                On <strong>&quot;Go&quot;</strong>, stand up <strong>fully</strong>
                — knees straight, body upright — then sit back down
                completely. That is one rep.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="tabular shrink-0 text-accent">5.</span>
              <span>
                Repeat <strong>5 times in a row, as fast as you can</strong>.
                Don&apos;t pause between reps — go straight from sit back to
                stand without resting.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="tabular shrink-0 text-accent">6.</span>
              <span>
                Time stops at the moment you are <strong>fully seated</strong>
                after the 5th rep.
              </span>
            </li>
          </ol>

          <p className="mt-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Common errors to watch
          </p>
          <ul className="mt-2 space-y-1.5 text-xs text-muted">
            <li className="flex gap-2">
              <span className="shrink-0 text-warning">•</span>
              <span>
                Using arms to push off the chair or thighs — invalidates
                the trial (auto-flagged).
              </span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0 text-warning">•</span>
              <span>
                Not standing up fully (knees still bent) — counts as a
                partial rep with the new knee-angle cross-check.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0 text-warning">•</span>
              <span>
                Hovering above the seat between reps instead of fully
                sitting back down.
              </span>
            </li>
          </ul>

          <div className="mt-5">
            <Button onClick={onStart}>
              <ArrowRight className="h-4 w-4" />
              Arm timer
            </Button>
          </div>
        </div>
      )}

      {phase === "armed" && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <p className="text-sm font-medium text-foreground">
            Timer armed. Order of operations matters — get this right or
            the rep counter won&apos;t work.
          </p>
          <ol className="mt-3 space-y-1.5 text-xs text-muted">
            <li>
              <strong>1.</strong> Confirm the patient is seated <strong>still</strong>,
              back against the chair, arms crossed. They must NOT be moving yet.
            </li>
            <li>
              <strong>2.</strong> Click <strong>Go</strong> RIGHT NOW — while
              the patient is still sitting still. The system needs to see
              the seated position before the patient moves.
            </li>
            <li>
              <strong>3.</strong> After clicking Go, say to the patient:
              <em>&quot;Ready, set, GO!&quot;</em> — they start their 5 reps then.
            </li>
          </ol>
          <p className="mt-3 rounded-md border border-warning/30 bg-warning/5 p-2 text-[11px] text-warning">
            <strong>Important:</strong> if you click Go after the patient
            already started standing, the seated baseline is captured
            mid-motion and reps won&apos;t count. Click Go first, verbal cue
            second.
          </p>
          <div className="mt-3">
            <Button onClick={onGo}>
              <Play className="h-4 w-4" />
              Go (while patient is still seated)
            </Button>
          </div>
        </div>
      )}

      {phase === "recording" && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <p className="text-sm font-medium text-foreground">
            Recording sit-to-stand reps…
          </p>
          <p className="mt-2 text-xs text-muted">
            Watching for 5 complete sit↔stand cycles. Each rep must reach
            full knee extension on the stand and a full sit-back to count.
            Trial auto-ends after the 5th rep (or 30 s timeout).
          </p>
          {coachMsg && (
            <p className="mt-2 rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-xs font-medium text-foreground">
              {coachMsg}
            </p>
          )}
          <div className="mt-3">
            <Button variant="ghost" size="sm" onClick={onStopEarly}>
              Stop early
            </Button>
          </div>
        </div>
      )}

      {phase === "done" && result && (
        <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <p className="text-sm font-medium text-foreground">
              Chair stand complete: {result.reps.length} reps in {result.total_time_seconds.toFixed(2)} s
            </p>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button onClick={onFinish}>
              <ArrowRight className="h-4 w-4" />
              Generate SPPB report
            </Button>
            <Button variant="secondary" onClick={onRestart}>
              <RotateCcw className="h-4 w-4" />
              Redo chair stand
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Explicit component switcher (tab bar) ─────────────────
//
// Sits below the progress strip and lets the operator JUMP between
// Component 1, 2, 3 without following the linear flow. Useful for
// re-doing a bad take, skipping a component the patient can't do,
// or grabbing data out of order while the patient is already
// positioned for a later component.
//
// Each tab shows:
//   - the component number + label
//   - a status icon: ✓ done, ▶ active, • pending
//   - the captured score / time when available
//
// The currently-active component's tab is highlighted; clicking
// another tab fires `onSwitch` which the parent uses to flip phase
// state. Switching does NOT destroy results.

function ComponentSwitcher({
  phase,
  balanceStages,
  gaitTrials,
  chairResult,
  onSwitch,
}: {
  phase: Phase;
  balanceStages: { 1?: StageResult; 2?: StageResult; 3?: StageResult };
  gaitTrials: { 1: GaitSpeedTrial | null; 2: GaitSpeedTrial | null };
  chairResult: SitToStandResult | null;
  onSwitch: (c: Component) => void;
}) {
  const balanceHasAnyResult =
    balanceStages[1] !== undefined ||
    balanceStages[2] !== undefined ||
    balanceStages[3] !== undefined;
  const gaitHasAnyResult = gaitTrials[1] !== null || gaitTrials[2] !== null;
  const chairHasResult = chairResult !== null;

  const currentComponent: Component | null =
    phase === "balance" || phase === "transition_to_gait" ? "balance"
    : phase === "gait" || phase === "transition_to_chair" ? "gait"
    : phase === "chair" ? "chair"
    : null;

  // Short status summaries for each tab.
  const balanceSummary = balanceHasAnyResult
    ? `${[1, 2, 3].filter((s) => balanceStages[s as 1 | 2 | 3]?.outcome === "pass").length}/3 passed`
    : "not started";
  const gaitSummary = (() => {
    const completed = [gaitTrials[1], gaitTrials[2]].filter(
      (t) => t !== null,
    ) as GaitSpeedTrial[];
    if (completed.length === 0) return "not started";
    const best = completed.reduce((a, b) =>
      a.duration_sec < b.duration_sec ? a : b,
    );
    return `best ${best.duration_sec.toFixed(2)} s`;
  })();
  const chairSummary = chairHasResult
    ? `${chairResult.reps.length} reps · ${chairResult.total_time_seconds.toFixed(2)} s`
    : "not started";

  const tabs: Array<{
    key: Component;
    num: number;
    label: string;
    done: boolean;
    summary: string;
  }> = [
    { key: "balance", num: 1, label: "Balance",     done: balanceHasAnyResult, summary: balanceSummary },
    { key: "gait",    num: 2, label: "Gait speed",  done: gaitHasAnyResult,    summary: gaitSummary    },
    { key: "chair",   num: 3, label: "Chair stand", done: chairHasResult,      summary: chairSummary   },
  ];

  return (
    <div className="rounded-card border border-border bg-surface p-3">
      <p className="px-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
        Jump to component
      </p>
      <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-3">
        {tabs.map((tab) => {
          const active = currentComponent === tab.key;
          const tone = active
            ? "border-accent bg-accent/10 text-foreground"
            : tab.done
              ? "border-emerald-500/30 bg-emerald-500/5 text-foreground hover:border-emerald-500/60"
              : "border-border bg-elevated text-foreground hover:border-accent/50";
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => onSwitch(tab.key)}
              disabled={active}
              className={`group flex items-center gap-3 rounded-md border p-3 text-left transition disabled:cursor-default ${tone}`}
            >
              <span
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                  tab.done
                    ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
                    : active
                      ? "bg-accent text-white"
                      : "bg-elevated text-muted ring-1 ring-border"
                }`}
              >
                {tab.done ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  tab.num
                )}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium text-foreground">
                  Component {tab.num} — {tab.label}
                </span>
                <span className="block truncate text-[11px] text-muted">
                  {active ? "Currently active" : tab.summary}
                </span>
              </span>
            </button>
          );
        })}
      </div>
      <p className="mt-2 px-1 text-[11px] text-subtle">
        Switching to another component while a recording is in progress
        will cancel the current capture.
      </p>
    </div>
  );
}

// ─── Progress strip ────────────────────────────────────────

function ProgressStrip({ phase }: { phase: Phase }) {
  const items: Array<{ label: string; active: boolean; done: boolean }> = [
    {
      label: "1. Balance",
      active: phase === "balance" || phase === "transition_to_gait",
      done: phase === "gait" || phase === "transition_to_chair" || phase === "chair" || phase === "done",
    },
    {
      label: "2. Gait speed",
      active: phase === "gait" || phase === "transition_to_chair",
      done: phase === "chair" || phase === "done",
    },
    {
      label: "3. Chair stand",
      active: phase === "chair",
      done: phase === "done",
    },
  ];
  return (
    <div className="flex items-center gap-2 text-xs">
      {items.map((it, i) => (
        <span
          key={it.label}
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 ${
            it.done
              ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
              : it.active
                ? "bg-accent/15 text-accent"
                : "bg-elevated text-muted"
          }`}
        >
          {it.done ? <CheckCircle2 className="h-3 w-3" /> : <span className="h-1.5 w-1.5 rounded-full bg-current" />}
          {it.label}
          {i < items.length - 1 && <ChevronRight className="h-3 w-3 opacity-60" />}
        </span>
      ))}
    </div>
  );
}

function ComponentHeader({
  number,
  title,
  subtitle,
}: {
  number: number;
  title: string;
  subtitle?: string;
}) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-accent">
        Component {number} of 3
      </p>
      <h3 className="mt-1 text-xl font-semibold tracking-tight text-foreground">
        {title}
      </h3>
      {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
    </div>
  );
}

// Suppress unused-warning for icons that are conditionally rendered.
void AlertCircle;
void Loader2;
