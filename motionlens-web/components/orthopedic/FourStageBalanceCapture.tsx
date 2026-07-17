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
  Camera,
  CheckCircle2,
  ChevronRight,
  FileVideo,
  Loader2,
  Play,
  RotateCcw,
  Upload,
  Video,
  XCircle,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { FourStageBalanceLiveCamera } from "@/components/orthopedic/FourStageBalanceLiveCamera";
import { FourStageBalanceReport } from "@/components/orthopedic/FourStageBalanceReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  POSITION_LOCK_MS,
  POSITION_DRIFT_GRACE_MS,
  POSITION_TIMEOUT_SEC,
  SAMPLE_INTERVAL_MS,
  STAGE_HOLD_SEC,
  STAGE_INSTRUCTION,
  STAGE_LABEL,
  STAGE_PROTOCOL,
  analyzeFourStageBalanceUpload,
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

type Mode = "live" | "upload";
type UploadPhase = "idle" | "analyzing" | "done" | "error";

const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = [
  "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
];

function errorMessage(e: unknown): string | null {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  return null;
}

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

  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [stageResults, setStageResults] = useState<SessionResult["stages"]>({});
  const [error, setError] = useState<string | null>(null);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");
  // Tick the UI for live timers without re-rendering the whole tree.
  const [tick, setTick] = useState<number>(0);

  const runRef = useRef<RunState | null>(null);

  // ── Auto-flow (fullscreen less-click live mode) ────────────────
  // One click ("Start Assessment") opens the fullscreen shell; the
  // camera auto-starts. The WHOLE 4-stage machine runs inside this
  // single fullscreen session: a 3-2-1 countdown runs before EACH
  // stage (`countdownFor` re-arms the hook's started input between
  // stances), then the stage's own position-lock → 10 s hold machine
  // takes over. The final "Generate report" leaves fullscreen and
  // the done view auto-saves the combined session (doctor flow).
  const [liveFullscreen, setLiveFullscreen] = useState<boolean>(false);
  const [camActive, setCamActive] = useState<boolean>(false);
  const [countdownFor, setCountdownFor] = useState<StageIndex | null>(null);

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

  // Countdown runs once the camera stream is live AND a stage is
  // armed (`countdownFor`). Clearing countdownFor inside onLive lets
  // the hook's started input toggle naturally between stances, so
  // every stage gets its own 3-2-1 before its position-lock window
  // starts ticking. startTest / beginStage are declared below
  // (hoisted function declarations).
  const {
    phase: flowPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(
    liveFullscreen && camActive && countdownFor !== null,
    () => {
      const stage = countdownFor;
      setCountdownFor(null);
      if (stage === null) return;
      if (stage === 1) startTest();
      else beginStage(stage);
    },
  );

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
      const inPosition = isStagePosition(run.stage, ankles, kp);
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
            run.stage4StanceSide = detectStage4Stance(ankles, kp);
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
        const driftOK = isStagePosition(run.stage, ankles, kp);
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

  // ── Upload-mode state ──────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadFiles, setUploadFiles] = useState<Record<StageIndex, File | null>>({
    1: null, 2: null, 3: null, 4: null,
  });
  const [uploadProgress, setUploadProgress] = useState<Record<StageIndex, number>>({
    1: 0, 2: 0, 3: 0, 4: 0,
  });
  const [uploadErrors, setUploadErrors] = useState<Record<StageIndex, string | null>>({
    1: null, 2: null, 3: null, 4: null,
  });
  const [filePickError, setFilePickError] = useState<string | null>(null);

  function setStageProgress(stage: StageIndex, pct: number) {
    setUploadProgress((prev) => ({ ...prev, [stage]: pct }));
  }

  function validateAndSetStage(stage: StageIndex, file: File | null) {
    setFilePickError(null);
    if (!file) {
      setUploadFiles((prev) => ({ ...prev, [stage]: null }));
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFilePickError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_MB} MB.`);
      return;
    }
    if (file.type && !ACCEPTED_VIDEO_TYPES.includes(file.type)) {
      setFilePickError(`Unsupported file type (${file.type}). Use MP4, WebM, MOV, or MKV.`);
      return;
    }
    setUploadFiles((prev) => ({ ...prev, [stage]: file }));
  }

  async function analyzeUpload() {
    const stages: StageIndex[] = ([1, 2, 3, 4] as StageIndex[])
      .filter((s) => uploadFiles[s] !== null);
    if (stages.length === 0) return;

    setUploadPhase("analyzing");
    setUploadProgress({ 1: 0, 2: 0, 3: 0, 4: 0 });
    setUploadErrors({ 1: null, 2: null, 3: null, 4: null });
    setError(null);

    // Sequential, not parallel. The backend has only 2 gunicorn workers
    // and each loads its MediaPipe BlazePose model on the first request
    // post-deploy; firing up to 4 stages in parallel can blow past
    // Vercel's ~30 s upstream-response budget on cold workers. Iterating
    // the stages in order keeps each request comfortably warm. Per-stage
    // analysis math + result shape unchanged.
    const newResults: SessionResult["stages"] = {};
    const newErrors: Record<StageIndex, string | null> = { 1: null, 2: null, 3: null, 4: null };
    let anySuccess = false;
    for (const stage of stages) {
      const file = uploadFiles[stage]!;
      try {
        const result = await analyzeFourStageBalanceUpload(
          file, stage, (pct) => setStageProgress(stage, pct),
        );
        newResults[stage] = result;
        anySuccess = true;
      } catch (e) {
        newErrors[stage] = errorMessage(e) ?? `Stage ${stage} analysis failed.`;
      }
    }

    // Stop-at-first-failure rule: if stage N failed, mark stages
    // N+1..4 as not_attempted on the assembled session.
    let firstFail: StageIndex | null = null;
    for (const s of [1, 2, 3, 4] as StageIndex[]) {
      const r = newResults[s];
      if (r && r.outcome === "fail" && firstFail === null) {
        firstFail = s;
        break;
      }
    }
    if (firstFail !== null) {
      for (const s of [1, 2, 3, 4] as StageIndex[]) {
        if (s > firstFail && newResults[s] === undefined && newErrors[s] === null) {
          newResults[s] = {
            stage: s,
            outcome: "not_attempted",
            hold_seconds: 0,
            failure_mode: null,
            sway_path_px: 0,
            sway_95_ellipse_px2: 0,
            hip_path: [],
            samples: [],
            keypoints: [],
            screenshot_data_url: null,
            duration_seconds: 0,
          };
        }
      }
    }

    setStageResults(newResults);
    setUploadErrors(newErrors);
    if (anySuccess) {
      setUploadPhase("done");
      setPhase("done");
    } else {
      setUploadPhase("error");
    }
  }

  function resetUpload() {
    setUploadFiles({ 1: null, 2: null, 3: null, 4: null });
    setUploadProgress({ 1: 0, 2: 0, 3: 0, 4: 0 });
    setUploadErrors({ 1: null, 2: null, 3: null, 4: null });
    setFilePickError(null);
    setUploadPhase("idle");
    setStageResults({});
    setPhase("idle");
    setError(null);
  }

  function switchMode(next: Mode) {
    if (next === mode) return;
    if (phase === "running" || uploadPhase === "analyzing") return;
    runRef.current = null;
    setStageResults({});
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    setError(null);
    setLiveFullscreen(false);
    setCamActive(false);
    setCountdownFor(null);
    resetUpload();
    setMode(next);
  }

  // Enter the fullscreen auto-flow shell (the single click of the
  // live mode). Camera auto-starts inside; a countdown for stage 1
  // begins once frames are flowing.
  function enterLive() {
    setError(null);
    setLiveFullscreen(true);
    setCountdownFor(1);
  }

  function exitLive() {
    runRef.current = null;
    setStageResults({});
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    setCountdownFor(null);
    setLiveFullscreen(false);
    setCamActive(false);
  }

  function startTest() {
    setStageResults({});
    setError(null);
    runRef.current = freshRunState(1);
    setCoachMsg(STAGE_INSTRUCTION[1]);
    lastCoachRef.current = STAGE_INSTRUCTION[1];
    setPhase("running");
  }

  // Start a later stage directly (called by the auto-flow hook once
  // the between-stance countdown finishes). Identical to the old
  // advance path — per-stage machine untouched.
  function beginStage(stage: StageIndex) {
    runRef.current = freshRunState(stage);
    setCoachMsg(STAGE_INSTRUCTION[stage]);
    lastCoachRef.current = STAGE_INSTRUCTION[stage];
    setTick((v) => v + 1);
  }

  // "Advance" on the passed card — arms the countdown for the next
  // stage instead of starting it immediately, so the patient gets a
  // 3-2-1 before the position-lock window starts ticking.
  function advanceToNextStage() {
    const run = runRef.current;
    if (!run) return;
    if (run.stage === 4) {
      finalizeRun();
      return;
    }
    setCountdownFor((run.stage + 1) as StageIndex);
  }

  function finalizeRun() {
    runRef.current = null;
    setCountdownFor(null);
    setPhase("done");
    // Leave the fullscreen shell — the done view renders the report.
    setLiveFullscreen(false);
    setCamActive(false);
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
    setCountdownFor(null);
  }

  // ─── Done view ──────────────────────────────────────────────
  if (phase === "done") {
    const session = buildSession(stageResults, patient?.age ?? null);
    const interpretation = buildInterpretation(session);
    const buildPayload = () => ({
      module: "four_stage_balance" as const,
      metrics: { session },
      observations: { interpretation },
    });
    return (
      <div className="space-y-8">
        {/* The FINAL combined session auto-saves in the doctor flow
            (toast with a 10s undo) for both live and upload runs. */}
        <AutoSaveToast buildPayload={buildPayload} />

        <FourStageBalanceReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          session={session}
          interpretation={interpretation}
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

  const modeSwitchDisabled =
    phase === "running" || uploadPhase === "analyzing";

  return (
    <div className="space-y-10">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      {/* ─── Mode toggle: Live Camera vs Upload Video ───────────── */}
      <div className="inline-flex rounded-card border border-border bg-surface p-1">
        <button
          type="button"
          onClick={() => switchMode("live")}
          disabled={modeSwitchDisabled}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "live"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${modeSwitchDisabled ? "opacity-50" : ""}`}
        >
          <Camera className="h-4 w-4" />
          Live camera
        </button>
        <button
          type="button"
          onClick={() => switchMode("upload")}
          disabled={modeSwitchDisabled}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "upload"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${modeSwitchDisabled ? "opacity-50" : ""}`}
        >
          <Upload className="h-4 w-4" />
          Upload video
        </button>
      </div>

      {/* ─── UPLOAD MODE ────────────────────────────────────────── */}
      {mode === "upload" && (
        <div className="space-y-6">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Setup checklist
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Record one short clip per stage (~10–15 seconds each).",
                "Each clip starts with the patient in position; end after the 10s hold.",
                "Camera frontal, hip height, full body visible.",
                "CDC protocol: if a stage fails, downstream stages are skipped on the report.",
                "Clips are analysed one after the other (stage 1 → 4).",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            {([1, 2, 3, 4] as StageIndex[]).map((s) => (
              <StageSlotPicker
                key={s}
                stage={s}
                label={STAGE_LABEL[s]}
                file={uploadFiles[s]}
                onPick={(f) => validateAndSetStage(s, f)}
                progress={uploadProgress[s]}
                busy={uploadPhase === "analyzing" && uploadFiles[s] !== null}
                error={uploadErrors[s]}
              />
            ))}
          </div>

          {filePickError && (
            <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">{filePickError}</p>
            </div>
          )}

          {uploadPhase === "idle" && (
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={analyzeUpload}
                disabled={!Object.values(uploadFiles).some((f) => f !== null)}
              >
                <Upload className="h-4 w-4" />
                Analyse selected
              </Button>
            </div>
          )}

          {uploadPhase === "analyzing" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-accent" />
                <p className="text-sm text-foreground">
                  Uploading and analysing each stage one after the other.
                </p>
              </div>
            </div>
          )}

          {uploadPhase === "error" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-start gap-3 rounded-md border border-error/40 bg-error/5 p-3 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                <p className="text-foreground">All stages failed — re-check the videos and try again.</p>
              </div>
              <div className="mt-3">
                <Button variant="secondary" onClick={resetUpload}>
                  <RotateCcw className="h-4 w-4" />
                  Try again
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─── LIVE MODE — pre-fullscreen: instructions + one click ── */}
      {mode === "live" && !liveFullscreen && (
        <>
          <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
            {/* LEFT — full 4-stage protocol. */}
            <div className="space-y-5">
              <div className="rounded-card border border-border bg-surface p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                  What the patient will do
                </p>
                <p className="mt-2 text-sm text-muted">
                  The test progresses through 4 stages. Each stage must be
                  held for {STAGE_HOLD_SEC} seconds before advancing. Hold
                  your arms relaxed at your sides throughout.
                </p>
                <div className="mt-4 space-y-3">
                  {([1, 2, 3, 4] as const).map((s) => {
                    const p = STAGE_PROTOCOL[s];
                    return (
                      <div
                        key={s}
                        className="rounded-card border border-border bg-elevated p-4"
                      >
                        <div className="flex items-baseline justify-between gap-2">
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-accent">
                              Stage {s}
                            </p>
                            <p className="mt-0.5 text-sm font-semibold text-foreground">
                              {p.headline}
                            </p>
                          </div>
                          <pre className="whitespace-pre text-lg leading-tight text-foreground">
                            {p.visual}
                          </pre>
                        </div>
                        <ol className="mt-3 space-y-1.5 text-xs leading-relaxed text-foreground">
                          {p.steps.map((step, i) => (
                            <li key={i} className="flex gap-2">
                              <span className="tabular shrink-0 text-accent">
                                {i + 1}.
                              </span>
                              <span>{step}</span>
                            </li>
                          ))}
                        </ol>
                        <p className="mt-3 rounded-md border border-border bg-surface px-3 py-2 text-[11px] leading-relaxed text-muted">
                          <span className="font-semibold text-foreground">
                            Why this stage:
                          </span>{" "}
                          {p.note}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              <p className="text-xs text-muted">
                CDC fall-risk thresholds (PDF Test C4): unable to hold tandem
                (stage 3) for {STAGE_HOLD_SEC} s = significantly elevated fall
                risk. Single-leg (stage 4) &lt; 5 s for age &gt; 60 = high
                fall risk.
              </p>
            </div>

            {/* RIGHT — one-click start card. */}
            <div className="space-y-4">
              <div className="rounded-card border border-border bg-surface p-6 text-center">
                <p className="text-sm text-muted">
                  One click — the camera opens fullscreen and all four
                  stages run in a single session. A 3-2-1 countdown runs
                  before each stage; positions are auto-detected and each{" "}
                  {STAGE_HOLD_SEC} s hold is auto-timed. The test stops at
                  the first failure and the combined report saves to the
                  patient record.
                </p>
                <div className="mt-4 flex justify-center">
                  <Button onClick={enterLive}>
                    <Camera className="h-4 w-4" />
                    Start Assessment
                  </Button>
                </div>
              </div>
              <p className="text-xs text-muted">
                Patient barefoot, level surface, no support within
                arm&apos;s reach. The camera should be at hip height,
                ~2 m from the patient, with the full body in frame.
              </p>
            </div>
          </div>

          {error && (
            <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">{error}</p>
            </div>
          )}
        </>
      )}

      {/* ─── LIVE MODE — fullscreen auto-flow shell ──────────────── */}
      {mode === "live" && liveFullscreen && (
        <LiveModeLayout
          title="4-Stage Balance Test"
          subtitle={
            run
              ? `${STAGE_LABEL[run.stage]} · ${
                  run.stagePhase === "preparing"
                    ? "Get into position"
                    : run.stagePhase === "holding"
                      ? "Holding"
                      : run.stagePhase === "passed"
                        ? "Stage passed"
                        : "Stage ended"
                }`
              : countdownFor !== null
                ? `${STAGE_LABEL[countdownFor]} — starting`
                : "Patient barefoot, facing the camera, full body in frame"
          }
          onExit={exitLive}
          camera={(
            <FourStageBalanceLiveCamera
              onFrame={handleFrame}
              onError={setError}
              autoStart
              hideControls
              fill
              onActiveChange={setCamActive}
            >
              {!camActive && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <p className="rounded-full bg-black/60 px-4 py-2 text-sm text-white/80">
                    Starting camera…
                  </p>
                </div>
              )}
              {flowPhase === "countdown" && countdown !== null && countdownFor !== null && (
                <AutoFlowCountdownOverlay
                  countdown={countdown}
                  label={countdownFor === 1 ? "Test starts in" : `Stage ${countdownFor} starts in`}
                />
              )}
              {run && (run.stagePhase === "preparing" || run.stagePhase === "holding") && (
                <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-rose-300">
                    ● Stage {run.stage} of 4
                  </p>
                  <p className="tabular text-2xl font-semibold text-white">
                    {run.stagePhase === "holding" ? `${heldSec.toFixed(1)}s` : "—"}
                  </p>
                  <p className="text-[10px] text-white/70">
                    {run.stagePhase === "preparing"
                      ? "Get into position"
                      : `Hold ${STAGE_HOLD_SEC}s`}
                  </p>
                </div>
              )}
            </FourStageBalanceLiveCamera>
          )}
          sidebar={(
            <>
              {/* Stage progression strip — the 4 traffic lights. */}
              <StageProgressStrip
                currentStage={run?.stage ?? countdownFor}
                results={stageResults}
              />

              {flowPhase === "countdown" && countdown !== null && countdownFor !== null && (
                <AutoFlowCountdownCard
                  countdown={countdown}
                  onSkip={skipCountdown}
                  hint={STAGE_INSTRUCTION[countdownFor]}
                />
              )}

              {/* Running — preparing or holding. */}
              {phase === "running" && run && (run.stagePhase === "preparing" || run.stagePhase === "holding") && (
                <div className="rounded-card border border-accent/40 bg-accent/5 p-4">
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
                  {/* Show the full stage instruction list during the
                      preparing phase so the patient (and operator) have
                      the protocol in front of them while getting into
                      position — not just the brief coach line. */}
                  {run.stagePhase === "preparing" && (
                    <div className="mt-3 rounded-md border border-accent/20 bg-background/40 px-3 py-2 text-xs leading-relaxed">
                      <p className="font-semibold text-foreground">
                        {STAGE_PROTOCOL[run.stage].headline} — how to do it
                      </p>
                      <ol className="mt-1.5 space-y-1 text-muted">
                        {STAGE_PROTOCOL[run.stage].steps.map((step, i) => (
                          <li key={i} className="flex gap-2">
                            <span className="tabular shrink-0 text-accent">
                              {i + 1}.
                            </span>
                            <span>{step}</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
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

              {/* Stage passed — handoff card (hidden while the next
                  stage's countdown is already running). */}
              {phase === "running" && run && run.stagePhase === "passed" && countdownFor === null && (
                <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    <p className="text-sm font-medium text-foreground">
                      {STAGE_LABEL[run.stage]} held for the full {STAGE_HOLD_SEC} s.
                    </p>
                  </div>
                  {run.stage < 4 ? (
                    <>
                      <p className="mt-1 text-xs text-muted">
                        Get the patient into the next position, then advance
                        when they&apos;re ready — a 3-2-1 countdown runs
                        before the stage begins.
                      </p>
                      {/* Preview next-stage instructions inline so the
                          operator knows exactly what to coach the patient
                          into BEFORE clicking Advance. */}
                      <div className="mt-3 rounded-md border border-emerald-500/20 bg-background/40 p-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-emerald-700 dark:text-emerald-400">
                          Next — {STAGE_PROTOCOL[(run.stage + 1) as StageIndex].headline}
                        </p>
                        <ol className="mt-2 space-y-1 text-xs leading-relaxed text-muted">
                          {STAGE_PROTOCOL[(run.stage + 1) as StageIndex].steps.map(
                            (step, i) => (
                              <li key={i} className="flex gap-2">
                                <span className="tabular shrink-0 text-emerald-700 dark:text-emerald-400">
                                  {i + 1}.
                                </span>
                                <span>{step}</span>
                              </li>
                            ),
                          )}
                        </ol>
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2">
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
                <div className="rounded-card border border-red-500/30 bg-red-500/5 p-4">
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
                  </div>
                </div>
              )}

              {error && (
                <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-foreground">
                  <AlertTriangle className="mr-2 inline h-4 w-4 text-rose-500" />
                  {error}
                </div>
              )}

              <div className="mt-auto flex flex-wrap gap-2">
                <Button variant="ghost" size="sm" onClick={exitLive}>
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
              </div>
            </>
          )}
        />
      )}
    </div>
  );
}

// ─── Upload-mode per-stage picker ────────────────────────────────
function StageSlotPicker({
  stage,
  label,
  file,
  onPick,
  progress,
  busy,
  error,
}: {
  stage: StageIndex;
  label: string;
  file: File | null;
  onPick: (f: File | null) => void;
  progress: number;
  busy: boolean;
  error: string | null;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-accent">
        Stage {stage}
      </p>
      <p className="mt-0.5 text-sm font-medium text-foreground">{label}</p>
      {!file && (
        <label className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border border-dashed border-border bg-elevated p-4 text-center transition hover:border-accent/60">
          <FileVideo className="h-6 w-6 text-muted" />
          <p className="text-xs font-medium text-foreground">Choose video file</p>
          <input
            type="file"
            accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
            className="hidden"
            onChange={(e) => onPick(e.target.files?.[0] ?? null)}
            disabled={busy}
          />
        </label>
      )}
      {file && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-2 rounded-md bg-elevated p-2 text-xs">
            <Video className="h-3.5 w-3.5 shrink-0 text-accent" />
            <span className="min-w-0 flex-1 truncate font-medium text-foreground">{file.name}</span>
            {!busy && (
              <button type="button" onClick={() => onPick(null)} className="text-[10px] text-muted hover:text-error">
                remove
              </button>
            )}
          </div>
          {busy && (
            <div className="space-y-1">
              <div className="h-1 w-full overflow-hidden rounded-full bg-elevated">
                <div className="h-full bg-accent transition-all" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
              </div>
              <p className="text-[10px] text-muted">{Math.round(progress)}%</p>
            </div>
          )}
          {error && (
            <p className="rounded-md border border-error/40 bg-error/5 px-2 py-1.5 text-[10px] text-foreground">{error}</p>
          )}
        </div>
      )}
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
