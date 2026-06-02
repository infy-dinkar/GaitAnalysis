"use client";
// Active Knee Extension (AKE) capture flow.
//
// Patient lies supine; camera sits to the side (lateral view).
// Operator picks which leg to test, the camera tracks the test-side
// thigh + shin, and the system records the maximum knee extension
// reached while the thigh was held at ~90° from the bed.
//
// State machine (per side):
//   armed → ready_to_record (test-side leg trackable, waiting for click)
//         → recording      (trial running, capturing max knee at the
//                           moments where the thigh was held at 90°)
//         → finished side  (auto on TRIAL_DURATION_SEC timeout OR stop)
// Both sides done → done state with side-by-side report.
//
// Pre-record gate: keep evaluating whether the test-side hip + knee
// + ankle (and torso anchor) are visible. Start button is disabled
// until the chain is trackable.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  FileVideo,
  Loader2,
  Play,
  RotateCcw,
  Upload,
  Video,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { AKELiveCamera } from "@/components/orthopedic/AKELiveCamera";
import { AKEReport } from "@/components/orthopedic/AKEReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG,
  SAMPLE_INTERVAL_MS,
  THIGH_HELD_MAX_DEG,
  THIGH_HELD_MIN_DEG,
  TRIAL_DURATION_SEC,
  analyzeAKEUpload,
  buildInterpretation,
  computeHipFlexAngle,
  computeKneeAngle,
  isTestSideTrackable,
  isThighHeld,
  summarizeSide,
  type AKEFullResult,
  type AKESideResult,
  type FrameSample,
  type Side,
} from "@/lib/orthopedic/ake";

type Mode = "live" | "upload";
type Phase = "idle" | "armed" | "recording" | "done";
type UploadPhase = "idle" | "analyzing" | "done" | "error";

const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = [
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/x-matroska",
];

function errorMessage(e: unknown): string | null {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  return null;
}

interface RecordingState {
  side: Side;
  startedAt: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  lastSampleAt: number;
  maxValidKneeSoFar: number;
  peakScreenshot: string | null;
}

export function AKECapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  // Mode toggle — live (browser BlazePose WASM, per-side) vs upload
  // (backend MediaPipe, both sides sequentially). Both modes converge
  // on the same AKEFullResult + AKEReport.
  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [armedSide, setArmedSide] = useState<Side | null>(null);
  const [completedSides, setCompletedSides] = useState<Set<Side>>(new Set());
  const [result, setResult] = useState<AKEFullResult>({ left: null, right: null });
  const [now, setNow] = useState<number>(0);

  // Live coaching: whether the test-side leg is currently trackable +
  // real-time knee + hip-flex readouts.
  const [legTrackable, setLegTrackable] = useState<boolean>(false);
  const [liveKneeDeg, setLiveKneeDeg] = useState<number | null>(null);
  const [liveHipFlexDeg, setLiveHipFlexDeg] = useState<number | null>(null);
  const [liveThighHeld, setLiveThighHeld] = useState<boolean>(false);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  const recordingRef = useRef<RecordingState | null>(null);
  const resultRef = useRef(result);
  useEffect(() => { resultRef.current = result; }, [result]);

  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [phase]);

  const setCoachIfChanged = useCallback((msg: string) => {
    if (lastCoachRef.current === msg) return;
    lastCoachRef.current = msg;
    setCoachMsg(msg);
  }, []);

  const finishSide = useCallback((termination: AKESideResult["termination"]) => {
    const rec = recordingRef.current;
    if (!rec) return;
    // Fallback screenshot: if no peak frame was captured (e.g. trial
    // stopped early or no valid extension reached), grab the current
    // frame so the saved report still has a visual reference.
    if (!rec.peakScreenshot) {
      const grab = (window as unknown as {
        __akeCapture?: () => string | null;
      }).__akeCapture;
      if (grab) {
        const url = grab();
        if (url) rec.peakScreenshot = url;
      }
    }
    const summary = summarizeSide(
      rec.side,
      rec.startedAt,
      Date.now(),
      termination,
      rec.samples,
      rec.keypoints,
      rec.peakScreenshot,
    );
    setResult((prev) => ({ ...prev, [rec.side]: summary }));
    setCompletedSides((prev) => {
      const next = new Set(prev);
      next.add(rec.side);
      return next;
    });
    recordingRef.current = null;
    setArmedSide(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setLiveKneeDeg(null);
    setLiveHipFlexDeg(null);
    setLiveThighHeld(false);

    setPhase(() => {
      const r = resultRef.current;
      const otherDone = rec.side === "left" ? r.right !== null : r.left !== null;
      return otherDone ? "done" : "idle";
    });
  }, []);

  // Per-frame callback ------------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    // Always evaluate trackability for the side currently armed/recording
    // so the pre-record gate UI updates smoothly.
    const sideForGate: Side | null =
      recordingRef.current?.side ?? armedSide ?? null;
    if (sideForGate) {
      setLegTrackable(isTestSideTrackable(kp, sideForGate));
    } else {
      // No side selected yet — show "tracking ready" once either leg is fully visible.
      setLegTrackable(
        isTestSideTrackable(kp, "left") || isTestSideTrackable(kp, "right"),
      );
    }

    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    // Trial-timeout safety net.
    const elapsedSec = (tNow - rec.startedAt) / 1000;
    if (elapsedSec >= TRIAL_DURATION_SEC) {
      finishSide("timeout");
      return;
    }

    const knee = computeKneeAngle(kp, rec.side);
    const hipFlex = computeHipFlexAngle(kp, rec.side);
    const thighHeld = isThighHeld(hipFlex);

    setLiveKneeDeg(knee);
    setLiveHipFlexDeg(hipFlex);
    setLiveThighHeld(thighHeld);

    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      knee_angle_deg: knee,
      hip_flex_angle_deg: hipFlex,
      thigh_held: thighHeld,
    });
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Track peak valid knee extension + grab a screenshot at that moment.
    if (
      thighHeld && knee !== null && knee > rec.maxValidKneeSoFar
    ) {
      rec.maxValidKneeSoFar = knee;
      const grab = (window as unknown as {
        __akeCapture?: () => string | null;
      }).__akeCapture;
      if (grab) {
        const url = grab();
        if (url) rec.peakScreenshot = url;
      }
    }

    // Coaching message
    if (knee === null || hipFlex === null) {
      setCoachIfChanged(
        "Test-side leg not fully visible — keep the patient's hip, knee, and ankle in frame.",
      );
    } else if (!thighHeld) {
      setCoachIfChanged(
        `Hold the thigh vertical (hip flex ${hipFlex.toFixed(0)}°, need ${THIGH_HELD_MIN_DEG}–${THIGH_HELD_MAX_DEG}°).`,
      );
    } else {
      setCoachIfChanged(
        `Holding — knee now ${knee.toFixed(0)}°. Slowly straighten as far as the patient can.`,
      );
    }
  }, [phase, armedSide, finishSide, setCoachIfChanged]);

  // ── Upload-mode state ──────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [leftFile,  setLeftFile]  = useState<File | null>(null);
  const [rightFile, setRightFile] = useState<File | null>(null);
  const [leftProgress,  setLeftProgress]  = useState<number>(0);
  const [rightProgress, setRightProgress] = useState<number>(0);
  const [uploadErrors, setUploadErrors] = useState<{
    left: string | null;
    right: string | null;
  }>({ left: null, right: null });
  const [filePickError, setFilePickError] = useState<string | null>(null);

  function validateAndSetFile(side: Side, file: File | null) {
    setFilePickError(null);
    if (!file) {
      if (side === "left") setLeftFile(null);
      else setRightFile(null);
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFilePickError(
        `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_MB} MB.`,
      );
      return;
    }
    if (file.type && !ACCEPTED_VIDEO_TYPES.includes(file.type)) {
      setFilePickError(
        `Unsupported file type (${file.type}). Use MP4, WebM, MOV, or MKV.`,
      );
      return;
    }
    if (side === "left") setLeftFile(file);
    else setRightFile(file);
  }

  async function analyzeUpload() {
    if (!leftFile && !rightFile) return;
    setUploadPhase("analyzing");
    setLeftProgress(0);
    setRightProgress(0);
    setUploadErrors({ left: null, right: null });
    setError(null);

    // Sequential, not parallel. The backend has only 2 gunicorn workers
    // and each loads its MediaPipe BlazePose model on the first request
    // post-deploy; two parallel cold loads can blow past Vercel's ~30 s
    // upstream-response budget. Running left then right keeps each
    // request comfortably warm. Same mitigation as the other orthopedic
    // multi-clip modules.
    let leftResult:  AKESideResult | null = null;
    let rightResult: AKESideResult | null = null;
    let leftErr:  string | null = null;
    let rightErr: string | null = null;

    if (leftFile) {
      try {
        leftResult = await analyzeAKEUpload(leftFile, "left", setLeftProgress);
      } catch (e) {
        leftErr = errorMessage(e) ?? "Left-side analysis failed.";
      }
    }
    if (rightFile) {
      try {
        rightResult = await analyzeAKEUpload(rightFile, "right", setRightProgress);
      } catch (e) {
        rightErr = errorMessage(e) ?? "Right-side analysis failed.";
      }
    }

    setResult({ left: leftResult, right: rightResult });
    setUploadErrors({ left: leftErr, right: rightErr });
    setUploadPhase(leftResult || rightResult ? "done" : "error");
  }

  function resetUpload() {
    setLeftFile(null);
    setRightFile(null);
    setLeftProgress(0);
    setRightProgress(0);
    setUploadErrors({ left: null, right: null });
    setFilePickError(null);
    setUploadPhase("idle");
    setResult({ left: null, right: null });
    setError(null);
  }

  function switchMode(next: Mode) {
    if (next === mode) return;
    if (phase === "recording" || uploadPhase === "analyzing") return;
    recordingRef.current = null;
    setResult({ left: null, right: null });
    setCompletedSides(new Set());
    setArmedSide(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    setError(null);
    resetUpload();
    setMode(next);
  }

  // Arming + start ----------------------------------------------------
  function arm(side: Side) {
    setError(null);
    setArmedSide(side);
    setPhase("armed");
    setCoachMsg(
      `Patient is supine. Camera on the ${side} side. The ${side} thigh should ` +
      `be held vertical (~90° from the bed) with the knee initially bent. ` +
      `Start when the leg is fully visible — patient will slowly extend the knee.`,
    );
  }

  function startRecording() {
    if (!armedSide) return;
    if (!legTrackable) {
      setError(
        "Test-side leg not yet trackable. Make sure the patient's hip, knee, " +
        "and ankle are clearly visible in frame before starting.",
      );
      return;
    }
    setError(null);
    recordingRef.current = {
      side: armedSide,
      startedAt: Date.now(),
      samples: [],
      keypoints: [],
      lastSampleAt: 0,
      maxValidKneeSoFar: 0,
      peakScreenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg(
      `Hold the ${armedSide} thigh vertical. Patient slowly straightens the ` +
      `${armedSide} knee — we'll capture the max angle reached.`,
    );
    setLiveKneeDeg(null);
    setLiveHipFlexDeg(null);
    setLiveThighHeld(false);
    setPhase("recording");
  }

  function stopEarly() {
    finishSide("stopped");
  }

  function reset() {
    recordingRef.current = null;
    setResult({ left: null, right: null });
    setCompletedSides(new Set());
    setArmedSide(null);
    setPhase("idle");
    setError(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setLiveKneeDeg(null);
    setLiveHipFlexDeg(null);
    setLiveThighHeld(false);
  }

  // Done view ---------------------------------------------------------
  const isLiveDone = phase === "done";
  const isUploadDone = uploadPhase === "done";
  if (isLiveDone || isUploadDone) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone ? () => { resetUpload(); } : reset;
    return (
      <div className="space-y-8">
        {isUploadDone && (uploadErrors.left || uploadErrors.right) && (
          <div className="space-y-2">
            {uploadErrors.left && (
              <div className="flex items-start gap-3 rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
                <p className="text-foreground">
                  <span className="font-medium">Left-leg video: </span>
                  {uploadErrors.left}
                </p>
              </div>
            )}
            {uploadErrors.right && (
              <div className="flex items-start gap-3 rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
                <p className="text-foreground">
                  <span className="font-medium">Right-leg video: </span>
                  {uploadErrors.right}
                </p>
              </div>
            )}
          </div>
        )}

        <AKEReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "ake",
            metrics: { left: result.left, right: result.right },
            observations: { interpretation },
          })}
        />

        <div className="flex justify-center border-t border-border pt-6">
          <Button variant="secondary" onClick={onRunAgain}>
            <RotateCcw className="h-4 w-4" />
            Run again
          </Button>
        </div>
      </div>
    );
  }

  // Capture view ------------------------------------------------------
  const sidesRemaining: Side[] = (["left", "right"] as Side[]).filter(
    (s) => !completedSides.has(s),
  );
  const liveSide = recordingRef.current?.side ?? armedSide ?? null;
  const elapsedSec =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remainingSec = Math.max(0, TRIAL_DURATION_SEC - elapsedSec);

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
                "Patient lies supine on a flat surface — face up, arms relaxed.",
                "Camera is positioned to the SIDE (lateral view) at hip height.",
                "Record TWO clips — one for the left leg, one for the right leg. " +
                  "For each, place the camera on the SAME side as the leg being tested.",
                "In each clip, the patient raises the test thigh to ~90° (vertical), holds it stable, " +
                  "then slowly straightens the knee as far as they can.",
                "Both clips are analysed one after the other (left first, then right).",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <SidePicker
              label="Left-leg AKE"
              hint="Camera on patient's LEFT side, testing the LEFT leg."
              file={leftFile}
              onPick={(f) => validateAndSetFile("left", f)}
              progress={leftProgress}
              busy={uploadPhase === "analyzing" && leftFile !== null}
              error={uploadErrors.left}
            />
            <SidePicker
              label="Right-leg AKE"
              hint="Camera on patient's RIGHT side, testing the RIGHT leg."
              file={rightFile}
              onPick={(f) => validateAndSetFile("right", f)}
              progress={rightProgress}
              busy={uploadPhase === "analyzing" && rightFile !== null}
              error={uploadErrors.right}
            />
          </div>

          {filePickError && (
            <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">{filePickError}</p>
            </div>
          )}

          {uploadPhase === "idle" && (
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={analyzeUpload} disabled={!leftFile && !rightFile}>
                <Upload className="h-4 w-4" />
                Analyse {leftFile && rightFile ? "both" : "selected"}
              </Button>
              {(leftFile || rightFile) && !(leftFile && rightFile) && (
                <p className="text-xs text-warning">
                  Only one side selected — the other side will be missing from the
                  report. For a complete bilateral comparison, upload both clips.
                </p>
              )}
            </div>
          )}

          {uploadPhase === "analyzing" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-accent" />
                <p className="text-sm text-foreground">
                  Uploading and analysing — left side first, then right.
                  About 15-30 seconds per side.
                </p>
              </div>
            </div>
          )}

          {uploadPhase === "error" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-start gap-3 rounded-md border border-error/40 bg-error/5 p-3 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                <div>
                  <p className="font-medium text-foreground">Analysis failed for both sides.</p>
                  {uploadErrors.left && (
                    <p className="mt-1 text-foreground">Left: {uploadErrors.left}</p>
                  )}
                  {uploadErrors.right && (
                    <p className="mt-1 text-foreground">Right: {uploadErrors.right}</p>
                  )}
                </div>
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

      {/* ─── LIVE MODE ────────────────────────────────────────── */}
      {mode === "live" && (
      <>
      <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
        {/* LEFT — instructions + controls */}
        <div className="space-y-5">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Movement instructions
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Patient lies supine (face up) on a flat surface; arms relaxed by the sides.",
                "Place the camera on the SIDE of the patient at hip height (lateral view).",
                "Choose which leg to test — set the camera on that same side so the leg is fully visible.",
                "Patient lifts the test thigh to a vertical position (~90° from the bed) with the knee bent.",
                "Holding the thigh steady, the patient slowly straightens the knee as far as they can without pain.",
                "The non-test leg stays flat against the surface throughout.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Pre-record gate */}
          {phase !== "recording" && (
            <div
              className={`rounded-card border p-4 text-sm ${
                legTrackable
                  ? "border-emerald-500/30 bg-emerald-500/5"
                  : "border-amber-500/40 bg-amber-500/5"
              }`}
            >
              <div className="flex items-center gap-2">
                {legTrackable ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                )}
                <span className="font-medium text-foreground">
                  {legTrackable
                    ? "Leg fully trackable"
                    : "Waiting for full leg + torso visibility…"}
                </span>
              </div>
              {!legTrackable && (
                <p className="mt-1 text-xs text-muted">
                  Adjust the camera so the patient's hips, knees, ankles, and shoulders
                  are visible in the same frame.
                </p>
              )}
            </div>
          )}

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Live status
            </p>

            {/* Recording panel */}
            {phase === "recording" && recordingRef.current && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground">
                    Recording — {liveSide === "left" ? "Left" : "Right"}-leg AKE
                  </p>
                  <p className="tabular text-2xl font-semibold text-accent">
                    {liveKneeDeg !== null
                      ? `${liveKneeDeg.toFixed(0)}°`
                      : "—"}
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span
                    className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 ${
                      liveThighHeld
                        ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                        : "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                    }`}
                  >
                    Thigh {liveThighHeld ? "held ✓" : "drifting ✗"}
                    {liveHipFlexDeg !== null
                      ? ` (${liveHipFlexDeg.toFixed(0)}°)`
                      : ""}
                  </span>
                  <span className="text-muted">
                    {remainingSec.toFixed(0)} s remaining
                  </span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full bg-accent transition-all"
                    style={{ width: `${(elapsedSec / TRIAL_DURATION_SEC) * 100}%` }}
                  />
                </div>
                {coachMsg && (
                  <p className="rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
                    {coachMsg}
                  </p>
                )}
                <div className="flex justify-end">
                  <Button variant="ghost" size="sm" onClick={stopEarly}>
                    Stop early
                  </Button>
                </div>
              </div>
            )}

            {/* Side picker / start */}
            {phase !== "recording" && (
              <div className="mt-3">
                {sidesRemaining.length === 0 ? (
                  <p className="text-sm text-muted">Both sides recorded. Compiling the report…</p>
                ) : armedSide ? (
                  <div className="space-y-3">
                    <p className="text-sm">
                      Ready to record:{" "}
                      <span className="font-medium text-foreground">
                        {armedSide === "left" ? "Left" : "Right"}-leg AKE
                      </span>
                      .
                    </p>
                    <p className="text-xs text-muted">
                      Position the camera on the {armedSide} side of the patient so the
                      {" "}{armedSide} hip, knee, and ankle are clearly visible. Click
                      {" "}<em>Start</em> when the leg shows trackable — the trial runs for up
                      to {TRIAL_DURATION_SEC} s.
                    </p>
                    {coachMsg && (
                      <p className="rounded-md bg-background/40 px-3 py-2 text-sm text-foreground">
                        {coachMsg}
                      </p>
                    )}
                    <div className="flex gap-2">
                      <Button onClick={startRecording} disabled={!legTrackable}>
                        <Play className="h-4 w-4" />
                        Start ({armedSide})
                      </Button>
                      <Button variant="ghost" onClick={() => setArmedSide(null)}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-foreground">Choose which leg to record next:</p>
                    <div className="flex flex-wrap gap-3">
                      {sidesRemaining.map((s) => (
                        <Button key={s} onClick={() => arm(s)}>
                          {s === "left" ? "Left" : "Right"}-leg AKE
                        </Button>
                      ))}
                    </div>
                    {completedSides.size > 0 && (
                      <p className="inline-flex items-center gap-1 text-xs text-emerald-600">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        {completedSides.size === 1 ? "1 side recorded" : "Both sides recorded"}
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            <p className="mt-4 text-xs text-muted">
              Deficit cutoffs: ≤ 10° = normal · 11–20° = mild · 21–35° = moderate ·
              &gt; 35° = severe. Thigh must stay at {THIGH_HELD_MIN_DEG}–{THIGH_HELD_MAX_DEG}°
              from the bed for the knee angle to count.
            </p>
          </div>
        </div>

        {/* RIGHT — sticky camera */}
        <div className="lg:sticky lg:top-28">
          <AKELiveCamera onFrame={handleFrame} onError={setError} />
          <p className="mt-3 text-xs text-subtle">
            Start the camera and frame the patient lying on a flat surface
            from the side. The on-screen skeleton tracks the hip, knee, and
            ankle of the leg being tested.
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
    </div>
  );
}

// ─── Upload-mode per-side picker ─────────────────────────────────
function SidePicker({
  label,
  hint,
  file,
  onPick,
  progress,
  busy,
  error,
}: {
  label: string;
  hint: string;
  file: File | null;
  onPick: (f: File | null) => void;
  progress: number;
  busy: boolean;
  error: string | null;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {label}
      </p>
      <p className="mt-1 text-xs text-muted">{hint}</p>

      {!file && (
        <label className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border border-dashed border-border bg-elevated p-6 text-center transition hover:border-accent/60">
          <FileVideo className="h-7 w-7 text-muted" />
          <p className="text-sm font-medium text-foreground">
            Choose video file
          </p>
          <p className="text-[11px] text-muted">
            MP4, WebM, MOV, or MKV
          </p>
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
          <div className="flex items-center gap-2 rounded-md bg-elevated p-2.5 text-sm">
            <Video className="h-4 w-4 shrink-0 text-accent" />
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium text-foreground">{file.name}</p>
              <p className="text-[11px] text-muted">
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </p>
            </div>
            {!busy && (
              <button
                type="button"
                onClick={() => onPick(null)}
                className="text-[11px] text-muted hover:text-error"
              >
                remove
              </button>
            )}
          </div>

          {busy && (
            <div className="space-y-1.5">
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                <div
                  className="h-full bg-accent transition-all"
                  style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                />
              </div>
              <p className="inline-flex items-center gap-1.5 text-[11px] text-muted">
                <Loader2 className="h-3 w-3 animate-spin" />
                Analysing — {Math.round(progress)}%
              </p>
            </div>
          )}

          {error && (
            <p className="rounded-md border border-error/40 bg-error/5 px-2.5 py-2 text-[11px] text-foreground">
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
