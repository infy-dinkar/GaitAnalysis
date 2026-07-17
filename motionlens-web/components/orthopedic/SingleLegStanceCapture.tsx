"use client";
// Single-Leg Stance test (Test C5) — capture flow.
//
// Bilateral, two-condition test. Layout shows EYES OPEN and EYES
// CLOSED as two top-level groups, each with Left-leg / Right-leg
// trial buttons. Doctor picks any of the (up to four) trials in
// any order. Recording starts immediately on click — no audio cue
// (operator gives the verbal "close your eyes" instruction in
// person; this keeps the flow simple and reliable across devices).
//
// Each trial: patient lifts the leg, stance auto-detects, timer
// starts. Trial auto-terminates on foot-touchdown / arm-grab /
// hop / max-time / manual stop.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Eye,
  EyeOff,
  FileVideo,
  Loader2,
  Play,
  RotateCcw,
  Upload,
  Video,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { SingleLegStanceLiveCamera } from "@/components/orthopedic/SingleLegStanceLiveCamera";
import { SingleLegStanceReport } from "@/components/orthopedic/SingleLegStanceReport";
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
  HOP_WINDOW_DURATION_MS,
  MAX_EYES_CLOSED_SEC,
  MAX_EYES_OPEN_SEC,
  ONSET_TIMEOUT_SEC,
  SAMPLE_INTERVAL_MS,
  analyzeSingleLegStanceUpload,
  buildInterpretation,
  buildSession,
  computeBodyHeightPx,
  computeHipMidpoint,
  computeShoulderMidpoint,
  computeTrunkLean,
  detectStanceSide,
  isArmGrab,
  isFootTouchdown,
  isHopInWindow,
  isLegLifted,
  summarizeTrial,
  type Condition,
  type FrameSample,
  type SessionResult,
  type Side,
  type Termination,
  type TrialResult,
} from "@/lib/orthopedic/singleLegStance";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

type Mode = "live" | "upload";
type UploadPhase = "idle" | "analyzing" | "done" | "error";
type TrialSlot = "left_open" | "right_open" | "left_closed" | "right_closed";

const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = [
  "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
];

function errorMessage(e: unknown): string | null {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  return null;
}

function slotToSideCondition(slot: TrialSlot): { side: Side; condition: Condition } {
  const [side, cond] = slot.split("_") as [Side, "open" | "closed"];
  return { side, condition: cond === "open" ? "eyes_open" : "eyes_closed" };
}

type TrialKey = "left_open" | "right_open" | "left_closed" | "right_closed";

function trialKey(side: Side, condition: Condition): TrialKey {
  return `${side}_${condition === "eyes_open" ? "open" : "closed"}` as TrialKey;
}

type Phase = "idle" | "recording" | "done";

interface RecordingState {
  side: Side;
  condition: Condition;
  startedAt: number;        // wall-clock ms when timer began
  firstStanceAt: number | null;
  lastSampleAt: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  hipPath: Array<{ x: number; y: number }>;
  trunkLeans: number[];
  /** Rolling window of stance-ankle Y for hop detection. */
  stanceAnkleWindow: Array<{ t_ms: number; y: number }>;
  screenshot: string | null;
}


// ─── Capture component ──────────────────────────────────────────

export function SingleLegStanceCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [trials, setTrials] = useState<SessionResult["trials"]>({});
  const [now, setNow] = useState<number>(0);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  const recordingRef = useRef<RecordingState | null>(null);

  // ── Auto-flow (fullscreen less-click live mode) ────────────────
  // One click ("Start Assessment") opens the fullscreen shell; the
  // camera auto-starts. Inside the shell the doctor picks a trial
  // (side x eyes condition); a 3-2-1 countdown runs before EACH
  // trial and the trial starts by itself. Trials auto-terminate as
  // before; "Generate report" leaves fullscreen and the done view
  // auto-saves (doctor flow).
  const [liveFullscreen, setLiveFullscreen] = useState<boolean>(false);
  const [camActive, setCamActive] = useState<boolean>(false);
  const [pendingTrial, setPendingTrial] = useState<{
    side: Side;
    condition: Condition;
  } | null>(null);

  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [phase]);

  const setCoachIfChanged = useCallback((msg: string) => {
    if (lastCoachRef.current === msg) return;
    lastCoachRef.current = msg;
    setCoachMsg(msg);
  }, []);

  const finishTrial = useCallback((termination: Termination) => {
    const rec = recordingRef.current;
    if (!rec) return;

    // Fallback screenshot — same pattern as the other tests.
    if (!rec.screenshot) {
      const grab = (window as unknown as {
        __singleLegStanceCapture?: () => string | null;
      }).__singleLegStanceCapture;
      if (grab) {
        const url = grab();
        if (url) rec.screenshot = url;
      }
    }

    const startMs = rec.firstStanceAt ?? rec.startedAt;
    const summary: TrialResult = summarizeTrial({
      side: rec.side,
      condition: rec.condition,
      startedAtMs: startMs,
      endedAtMs: Date.now(),
      termination,
      hipPath: rec.hipPath,
      trunkLeans: rec.trunkLeans,
      samples: rec.samples,
      keypoints: rec.keypoints,
      screenshotDataUrl: rec.screenshot,
      patientAge: patient?.age ?? null,
    });

    const key = trialKey(rec.side, rec.condition);
    setTrials((prev) => ({ ...prev, [key]: summary }));
    recordingRef.current = null;
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    // Clear the auto-flow gate so the next trial pick re-arms the
    // countdown machine.
    setPendingTrial(null);
  // patient may change identity between trials in theory, capture once
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient]);

  // Countdown starts once the camera stream is live AND a trial has
  // been picked — the hook's started input toggles naturally between
  // trials, so every trial gets its own 3-2-1. onLive fires the
  // existing beginRecording (declared below; hoisted).
  const {
    phase: flowPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(
    liveFullscreen && camActive && pendingTrial !== null,
    () => {
      if (pendingTrial) beginRecording(pendingTrial.side, pendingTrial.condition);
    },
  );

  // Per-frame callback ----------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    if (phase !== "recording" || !recordingRef.current) return;
    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    const elapsedSinceStart = (tNow - rec.startedAt) / 1000;
    const cap = rec.condition === "eyes_open" ? MAX_EYES_OPEN_SEC : MAX_EYES_CLOSED_SEC;

    const detected = detectStanceSide(kp);
    const hipMid = computeHipMidpoint(kp);
    const shMid  = computeShoulderMidpoint(kp);
    const bodyH  = computeBodyHeightPx(kp);
    const trunkLean = computeTrunkLean(kp);

    // Record a sample even before stance is achieved, so the saved
    // landmarks JSON covers the whole trial duration.
    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      hip_x: hipMid?.x ?? null,
      hip_y: hipMid?.y ?? null,
      trunk_lean_deg: trunkLean,
    });
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Pre-onset (waiting for leg lift).
    if (rec.firstStanceAt === null) {
      // Three-tier check, ordered most→least specific:
      //   1. Exact stance-side match (best — patient lifted the
      //      expected leg, MoveNet labels are confident).
      //   2. Any leg lifted (fallback — MoveNet labels can flip on
      //      back-of-camera or partial-occlusion frames; we trust the
      //      operator's clicked side and start the timer anyway).
      //   3. Otherwise keep waiting.
      const lifted = detected === rec.side || isLegLifted(kp);
      if (lifted) {
        // Stance achieved. Reset baseline timestamps so hold-time
        // measures from this point.
        rec.firstStanceAt = tNow;
        rec.startedAt = tNow;
        rec.hipPath = [];
        rec.trunkLeans = [];
        rec.stanceAnkleWindow = [];
        setCoachIfChanged(
          `Hold steady — max ${cap.toFixed(0)} s. Trial ends if your foot touches down or you reach for support.`,
        );
        return;
      }
      // No lift yet — coach the patient.
      if (elapsedSinceStart > ONSET_TIMEOUT_SEC) {
        finishTrial("no_lift_detected");
        return;
      }
      setCoachIfChanged(
        `Lift your ${rec.side === "left" ? "right" : "left"} leg — keep the ${rec.side} foot planted.`,
      );
      return;
    }

    // POST-ONSET — track hip-mid for sway, watch terminations.
    if (hipMid) rec.hipPath.push({ x: hipMid.x, y: hipMid.y });
    if (trunkLean !== null) rec.trunkLeans.push(trunkLean);

    // Foot touchdown.
    if (isFootTouchdown(kp, rec.side)) {
      finishTrial("foot_touchdown");
      return;
    }

    // Arm grab.
    if (isArmGrab(kp)) {
      finishTrial("arm_grab");
      return;
    }

    // Hop / stance foot reposition. Tracks a rolling window of
    // stance-ankle Y; if the spread exceeds the threshold inside
    // the window, we call it a hop.
    const stanceAnkleIdx = rec.side === "left" ? LM.LEFT_ANKLE : LM.RIGHT_ANKLE;
    const stanceAnkle = kp[stanceAnkleIdx];
    if (stanceAnkle && (stanceAnkle.score ?? 0) >= 0.3 && bodyH) {
      const t = tNow - rec.startedAt;
      rec.stanceAnkleWindow.push({ t_ms: t, y: stanceAnkle.y });
      while (
        rec.stanceAnkleWindow.length > 0 &&
        t - rec.stanceAnkleWindow[0].t_ms > HOP_WINDOW_DURATION_MS
      ) {
        rec.stanceAnkleWindow.shift();
      }
      if (isHopInWindow(rec.stanceAnkleWindow, bodyH)) {
        finishTrial("hop");
        return;
      }
    }

    // Max-time ceiling.
    const heldSec = (tNow - rec.firstStanceAt) / 1000;
    if (heldSec >= cap) {
      finishTrial("max_time");
      return;
    }

    // Live coaching (hold-time readout).
    setCoachIfChanged(
      `Holding — ${heldSec.toFixed(1)} s of up to ${cap.toFixed(0)} s.`,
    );

    // Suppress unused warning in dev — shoulder midpoint isn't
    // currently surfaced live but we capture it for any future
    // overlay.
    void shMid;
  }, [phase, finishTrial, setCoachIfChanged]);

  // ── Upload-mode state ──────────────────────────────────────────
  // 4 file slots × per-slot progress + error. Promise.allSettled
  // runs selected slots one after the other; combined SessionResult
  // assembles client-side.
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadFiles, setUploadFiles] = useState<Record<TrialSlot, File | null>>({
    left_open: null, right_open: null, left_closed: null, right_closed: null,
  });
  const [uploadProgress, setUploadProgress] = useState<Record<TrialSlot, number>>({
    left_open: 0, right_open: 0, left_closed: 0, right_closed: 0,
  });
  const [uploadErrors, setUploadErrors] = useState<Record<TrialSlot, string | null>>({
    left_open: null, right_open: null, left_closed: null, right_closed: null,
  });
  const [filePickError, setFilePickError] = useState<string | null>(null);

  function setSlotProgress(slot: TrialSlot, pct: number) {
    setUploadProgress((prev) => ({ ...prev, [slot]: pct }));
  }

  function validateAndSetSlot(slot: TrialSlot, file: File | null) {
    setFilePickError(null);
    if (!file) {
      setUploadFiles((prev) => ({ ...prev, [slot]: null }));
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
    setUploadFiles((prev) => ({ ...prev, [slot]: file }));
  }

  async function analyzeUpload() {
    const slots: TrialSlot[] = (Object.keys(uploadFiles) as TrialSlot[])
      .filter((s) => uploadFiles[s] !== null);
    if (slots.length === 0) return;

    setUploadPhase("analyzing");
    setUploadProgress({ left_open: 0, right_open: 0, left_closed: 0, right_closed: 0 });
    setUploadErrors({ left_open: null, right_open: null, left_closed: null, right_closed: null });
    setError(null);

    const age = patient?.age ?? null;

    // Sequential, not parallel. The backend has only 2 gunicorn workers
    // and each loads its MediaPipe BlazePose model on the first request
    // post-deploy; firing up to 4 trial slots in parallel can blow past
    // Vercel's ~30 s upstream-response budget on cold workers. Iterating
    // the slots keeps each request comfortably warm. Per-slot analysis
    // math + result shape unchanged.
    const newTrials: SessionResult["trials"] = {};
    const newErrors: Record<TrialSlot, string | null> = {
      left_open: null, right_open: null, left_closed: null, right_closed: null,
    };
    let anySuccess = false;
    for (const slot of slots) {
      const file = uploadFiles[slot]!;
      const { side, condition } = slotToSideCondition(slot);
      try {
        const result = await analyzeSingleLegStanceUpload(
          file, side, condition, age,
          (pct) => setSlotProgress(slot, pct),
        );
        newTrials[slot] = result;
        anySuccess = true;
      } catch (e) {
        newErrors[slot] = errorMessage(e) ?? `${slot} analysis failed.`;
      }
    }

    setTrials(newTrials);
    setUploadErrors(newErrors);
    if (anySuccess) {
      setUploadPhase("done");
      setPhase("done");
    } else {
      setUploadPhase("error");
    }
  }

  function resetUpload() {
    setUploadFiles({ left_open: null, right_open: null, left_closed: null, right_closed: null });
    setUploadProgress({ left_open: 0, right_open: 0, left_closed: 0, right_closed: 0 });
    setUploadErrors({ left_open: null, right_open: null, left_closed: null, right_closed: null });
    setFilePickError(null);
    setUploadPhase("idle");
    setTrials({});
    setPhase("idle");
    setError(null);
  }

  function switchMode(next: Mode) {
    if (next === mode) return;
    if (phase === "recording" || uploadPhase === "analyzing") return;
    recordingRef.current = null;
    setTrials({});
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    setError(null);
    setLiveFullscreen(false);
    setCamActive(false);
    setPendingTrial(null);
    resetUpload();
    setMode(next);
  }

  // Enter the fullscreen auto-flow shell (the single click of the
  // live mode). Camera auto-starts inside; the doctor then picks a
  // trial and a countdown leads into recording.
  function enterLive() {
    setError(null);
    setLiveFullscreen(true);
  }

  function exitLive() {
    recordingRef.current = null;
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    setPendingTrial(null);
    setLiveFullscreen(false);
    setCamActive(false);
  }

  // Arm the auto-flow countdown for a specific trial slot.
  function pickTrial(side: Side, condition: Condition) {
    setError(null);
    setPendingTrial({ side, condition });
  }

  // "Generate report" — leaves the fullscreen shell; the done view
  // auto-saves in the doctor flow.
  function generateReport() {
    setPhase("done");
    setPendingTrial(null);
    setLiveFullscreen(false);
    setCamActive(false);
  }

  // Begin a trial directly. No audio cue — operator gives the
  // verbal "close your eyes" instruction in person before clicking.
  function beginRecording(side: Side, condition: Condition) {
    recordingRef.current = {
      side,
      condition,
      startedAt: Date.now(),
      firstStanceAt: null,
      lastSampleAt: 0,
      samples: [],
      keypoints: [],
      hipPath: [],
      trunkLeans: [],
      stanceAnkleWindow: [],
      screenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg(
      `Lift your ${side === "left" ? "right" : "left"} leg to begin — timer starts when stance is detected.`,
    );
    setPhase("recording");
  }

  function stopEarly() {
    finishTrial("stopped");
  }

  function reset() {
    recordingRef.current = null;
    setTrials({});
    setPhase("idle");
    setError(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setPendingTrial(null);
  }

  // Done view ---------------------------------------------------------
  // Show the "Generate report" CTA once both eyes-open trials are
  // captured (the spec-mandated minimum for a complete session).
  // Eyes-closed trials remain optional and can be added before
  // generating the report.
  const allDesiredDone = !!trials.left_open && !!trials.right_open;
  const showReport = phase === "done";
  const session = buildSession(trials, patient?.age ?? null);

  if (showReport) {
    const interpretation = buildInterpretation(session);
    const buildPayload = () => ({
      module: "single_leg_stance" as const,
      metrics: { session },
      observations: { interpretation },
    });
    return (
      <div className="space-y-8">
        {/* Results auto-save in the doctor flow (toast with a 10s
            undo) for both live and upload runs. */}
        <AutoSaveToast buildPayload={buildPayload} />

        <SingleLegStanceReport
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

  // Capture view ------------------------------------------------------
  const elapsedSec =
    phase === "recording" && recordingRef.current?.firstStanceAt !== null && recordingRef.current
      ? (now - (recordingRef.current.firstStanceAt ?? now)) / 1000
      : 0;
  const liveCondition = recordingRef.current?.condition ?? null;
  const cap = liveCondition === "eyes_closed" ? MAX_EYES_CLOSED_SEC : MAX_EYES_OPEN_SEC;
  const liveSide = recordingRef.current?.side ?? null;
  const capturedCount = Object.keys(trials).length;

  const modeSwitchDisabled =
    phase === "recording" || uploadPhase === "analyzing";

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
                "Record one video per trial — patient stands on one leg, lifts the other.",
                "Camera frontal, hip height, full body in frame.",
                "Eyes-open trials: max 60 s hold. Eyes-closed trials: max 30 s hold.",
                "At least one eyes-open trial per side recommended for a complete session.",
                "Clips are analysed one after the other — combined report below.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="space-y-4">
            <div>
              <div className="mb-2 flex items-center gap-2">
                <Eye className="h-4 w-4 text-accent" />
                <p className="text-sm font-semibold uppercase tracking-[0.12em] text-foreground">Eyes open</p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <StanceSlotPicker
                  label="Left-leg stance"
                  file={uploadFiles.left_open}
                  onPick={(f) => validateAndSetSlot("left_open", f)}
                  progress={uploadProgress.left_open}
                  busy={uploadPhase === "analyzing" && uploadFiles.left_open !== null}
                  error={uploadErrors.left_open}
                />
                <StanceSlotPicker
                  label="Right-leg stance"
                  file={uploadFiles.right_open}
                  onPick={(f) => validateAndSetSlot("right_open", f)}
                  progress={uploadProgress.right_open}
                  busy={uploadPhase === "analyzing" && uploadFiles.right_open !== null}
                  error={uploadErrors.right_open}
                />
              </div>
            </div>
            <div>
              <div className="mb-2 flex items-center gap-2">
                <EyeOff className="h-4 w-4 text-accent" />
                <p className="text-sm font-semibold uppercase tracking-[0.12em] text-foreground">Eyes closed (optional)</p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <StanceSlotPicker
                  label="Left-leg stance"
                  file={uploadFiles.left_closed}
                  onPick={(f) => validateAndSetSlot("left_closed", f)}
                  progress={uploadProgress.left_closed}
                  busy={uploadPhase === "analyzing" && uploadFiles.left_closed !== null}
                  error={uploadErrors.left_closed}
                />
                <StanceSlotPicker
                  label="Right-leg stance"
                  file={uploadFiles.right_closed}
                  onPick={(f) => validateAndSetSlot("right_closed", f)}
                  progress={uploadProgress.right_closed}
                  busy={uploadPhase === "analyzing" && uploadFiles.right_closed !== null}
                  error={uploadErrors.right_closed}
                />
              </div>
            </div>
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
                  Uploading and analysing — this can take 10-30 seconds per trial.
                </p>
              </div>
            </div>
          )}

          {uploadPhase === "error" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-start gap-3 rounded-md border border-error/40 bg-error/5 p-3 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                <p className="text-foreground">All trials failed — re-check the videos and try again.</p>
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
          {!patient?.age && isDoctorFlow && (
            <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
              <p className="font-medium text-foreground">Patient age missing</p>
              <p className="mt-1 text-muted">
                The age-based norm comparison will fall back to the strictest
                band (under 60). Add the patient&apos;s age to the profile for
                an accurate comparison.
              </p>
            </div>
          )}

          <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
            <div className="space-y-5">
              <div className="rounded-card border border-border bg-surface p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                  Movement instructions
                </p>
                <ol className="mt-3 space-y-2.5 text-sm text-foreground">
                  {[
                    "Stand barefoot facing the camera, full body in frame.",
                    "Hands on hips or relaxed at your sides. Don't reach for support.",
                    "Lift one leg by bending the hip and knee to about 90°.",
                    "Stand on the supporting leg and hold balance for as long as you can.",
                    `Eyes-open trials run up to ${MAX_EYES_OPEN_SEC} seconds; eyes-closed up to ${MAX_EYES_CLOSED_SEC} seconds.`,
                    "The trial ends if the lifted foot touches down, if you reach for support, or if you hop.",
                  ].map((s, i) => (
                    <li key={i} className="flex gap-2.5">
                      <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                      <span className="leading-relaxed">{s}</span>
                    </li>
                  ))}
                </ol>
              </div>
              <p className="text-xs text-muted">
                Cutoffs (PDF Test C5): age &lt;60 ≥ 10 s eyes-open, age 60–69 ≥ 7 s,
                age 70+ ≥ 5 s. Eyes-closed thresholds halved per PDF guidance.
                L–R asymmetry &gt; 30% indicates targeted intervention.
              </p>
            </div>

            <div className="space-y-4">
              <div className="rounded-card border border-border bg-surface p-6 text-center">
                <p className="text-sm text-muted">
                  One click — the camera opens fullscreen. Pick a trial
                  (left / right, eyes open / closed); a 3-2-1 countdown
                  runs and the trial starts by itself. Each trial ends
                  automatically, and generating the report saves it to
                  the patient record.
                </p>
                <div className="mt-4 flex justify-center">
                  <Button onClick={enterLive}>
                    <Camera className="h-4 w-4" />
                    {capturedCount > 0 ? "Resume Assessment" : "Start Assessment"}
                  </Button>
                </div>
              </div>

              {capturedCount > 0 && (
                <div className="rounded-card border border-border bg-surface p-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                    Captured trials
                  </p>
                  <ul className="mt-3 space-y-1.5 text-sm">
                    {([
                      ["left_open", "Left leg · eyes open"],
                      ["right_open", "Right leg · eyes open"],
                      ["left_closed", "Left leg · eyes closed"],
                      ["right_closed", "Right leg · eyes closed"],
                    ] as const).map(([k, label]) => (
                      <li key={k} className="flex items-center gap-2 text-foreground">
                        {trials[k] ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                        ) : (
                          <span className="h-3.5 w-3.5 rounded-full border border-border" />
                        )}
                        {label}
                      </li>
                    ))}
                  </ul>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {allDesiredDone && (
                      <Button onClick={generateReport}>
                        <Play className="h-4 w-4" />
                        Generate report
                      </Button>
                    )}
                    <Button variant="ghost" onClick={reset}>
                      <RotateCcw className="h-4 w-4" />
                      Reset
                    </Button>
                  </div>
                </div>
              )}
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
          title="Single-Leg Stance"
          subtitle={
            phase === "recording" && liveSide
              ? `${liveSide === "left" ? "Left" : "Right"} leg · ${liveCondition === "eyes_closed" ? "eyes closed" : "eyes open"}`
              : "Pick a trial — patient barefoot, facing the camera, full body in frame"
          }
          onExit={exitLive}
          camera={(
            <SingleLegStanceLiveCamera
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
              {flowPhase === "countdown" && countdown !== null && (
                <AutoFlowCountdownOverlay countdown={countdown} label="Trial starts in" />
              )}
              {phase === "recording" && recordingRef.current && (
                <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-rose-300">
                    ● Recording
                  </p>
                  <p className="tabular text-2xl font-semibold text-white">
                    {recordingRef.current.firstStanceAt === null ? "—" : `${elapsedSec.toFixed(1)}s`}
                  </p>
                  <p className="text-[10px] text-white/70">
                    {liveSide === "left" ? "Left" : "Right"} leg ·{" "}
                    {liveCondition === "eyes_closed" ? "eyes closed" : "eyes open"} · max {cap.toFixed(0)}s
                  </p>
                </div>
              )}
            </SingleLegStanceLiveCamera>
          )}
          sidebar={(
            <>
              {flowPhase === "countdown" && countdown !== null && pendingTrial && (
                <AutoFlowCountdownCard
                  countdown={countdown}
                  onSkip={skipCountdown}
                  hint={`${pendingTrial.side === "left" ? "Left" : "Right"}-leg stance (${pendingTrial.condition === "eyes_closed" ? "eyes closed" : "eyes open"}) — the timer starts once the stance is detected.`}
                />
              )}

              {/* Trial picker — idle, nothing armed. */}
              {phase === "idle" && pendingTrial === null && (
                <>
                  <ConditionGroup
                    title="Eyes open"
                    subtitle={`Max hold ${MAX_EYES_OPEN_SEC} s per side.`}
                    icon={<Eye className="h-4 w-4 text-accent" />}
                    leftDone={!!trials.left_open}
                    rightDone={!!trials.right_open}
                    onLeft={() => pickTrial("left", "eyes_open")}
                    onRight={() => pickTrial("right", "eyes_open")}
                  />
                  <ConditionGroup
                    title="Eyes closed"
                    subtitle={`Max hold ${MAX_EYES_CLOSED_SEC} s per side. Tell the patient to close their eyes before clicking.`}
                    icon={<EyeOff className="h-4 w-4 text-accent" />}
                    leftDone={!!trials.left_closed}
                    rightDone={!!trials.right_closed}
                    onLeft={() => pickTrial("left", "eyes_closed")}
                    onRight={() => pickTrial("right", "eyes_closed")}
                  />
                  {allDesiredDone && (
                    <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-3">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                        <p className="text-sm font-medium text-foreground">
                          All required trials captured
                        </p>
                      </div>
                      <p className="mt-1 text-xs text-muted">
                        Eyes-closed trials are optional — add them or generate
                        the report now.
                      </p>
                      <div className="mt-2">
                        <Button size="sm" onClick={generateReport}>
                          <Play className="h-4 w-4" />
                          Generate report
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Recording panel */}
              {phase === "recording" && recordingRef.current && (
                <div className="rounded-card border border-border bg-surface p-4 space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm font-medium text-foreground">
                      {recordingRef.current.firstStanceAt === null
                        ? `Waiting for stance — ${liveSide === "left" ? "Left" : "Right"}-leg ${liveCondition === "eyes_closed" ? "(eyes closed)" : "(eyes open)"}`
                        : `Recording — ${liveSide === "left" ? "Left" : "Right"}-leg ${liveCondition === "eyes_closed" ? "(eyes closed)" : "(eyes open)"}`}
                    </p>
                    <p className="tabular text-2xl font-semibold text-accent">
                      {recordingRef.current.firstStanceAt === null ? "—" : `${elapsedSec.toFixed(1)}s`}
                    </p>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                    <div
                      className="h-full bg-accent transition-all"
                      style={{ width: `${Math.min(100, (elapsedSec / cap) * 100)}%` }}
                    />
                  </div>
                  {coachMsg && (
                    <p className="rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
                      {coachMsg}
                    </p>
                  )}
                </div>
              )}

              <div className="rounded-card border border-border bg-surface p-3 text-xs text-muted">
                <p className="font-semibold text-foreground">Session brief</p>
                <ol className="mt-2 list-decimal space-y-1 pl-4">
                  <li>Patient barefoot, facing the camera, full body in frame.</li>
                  <li>Pick a trial — 3-2-1 countdown, then the timer starts on leg lift.</li>
                  <li>Trials end on foot touchdown, arm grab, hop, or max time.</li>
                </ol>
              </div>

              {error && (
                <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-foreground">
                  <AlertTriangle className="mr-2 inline h-4 w-4 text-rose-500" />
                  {error}
                </div>
              )}

              <div className="mt-auto flex flex-wrap gap-2">
                {phase === "recording" && (
                  <Button variant="secondary" onClick={stopEarly}>Stop early</Button>
                )}
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

// ─── Upload-mode per-slot picker ─────────────────────────────────
function StanceSlotPicker({
  label,
  file,
  onPick,
  progress,
  busy,
  error,
}: {
  label: string;
  file: File | null;
  onPick: (f: File | null) => void;
  progress: number;
  busy: boolean;
  error: string | null;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">{label}</p>
      {!file && (
        <label className="mt-2 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border border-dashed border-border bg-elevated p-4 text-center transition hover:border-accent/60">
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
        <div className="mt-2 space-y-2">
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

// Top-level condition section ("Eyes open" / "Eyes closed") with
// Left- and Right-leg buttons inside it. Re-running a captured side
// is allowed (clicking again replaces the previous result).
function ConditionGroup({
  title,
  subtitle,
  icon,
  leftDone,
  rightDone,
  onLeft,
  onRight,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  leftDone: boolean;
  rightDone: boolean;
  onLeft: () => void;
  onRight: () => void;
}) {
  return (
    <section className="rounded-card border border-border bg-surface p-5">
      <div className="flex items-center gap-2">
        {icon}
        <p className="text-sm font-semibold uppercase tracking-[0.12em] text-foreground">
          {title}
        </p>
      </div>
      <p className="mt-1 text-xs text-muted">{subtitle}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <SideButton label="Left-leg stance" done={leftDone} onClick={onLeft} />
        <SideButton label="Right-leg stance" done={rightDone} onClick={onRight} />
      </div>
    </section>
  );
}

function SideButton({
  label,
  done,
  onClick,
}: {
  label: string;
  done: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center justify-between rounded-card border p-3 text-left text-sm transition ${
        done
          ? "border-emerald-500/40 bg-emerald-500/5 hover:border-emerald-500/60"
          : "border-border bg-elevated hover:border-accent"
      }`}
    >
      <span className="flex items-center gap-2 font-medium text-foreground">
        {done && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />}
        {label}
      </span>
      <span className="text-[10px] uppercase tracking-[0.12em] text-subtle">
        {done ? "Re-run" : "Start"}
      </span>
    </button>
  );
}
