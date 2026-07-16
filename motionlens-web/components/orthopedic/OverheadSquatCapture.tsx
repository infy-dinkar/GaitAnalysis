"use client";
// B2 Overhead Squat capture flow.
//
// Phase progression (live mode):
//   calibration (pre-camera) → calibration (fullscreen) → armed →
//   recording → uploading → done
//
// LIVE mode flow:
//   1. Tab toggle chooses "Live capture" (default).
//   2. Pre-camera step (normal page layout): patient height input +
//      "Start camera" button.
//   3. Operator clicks "Start camera" → fullscreen `LiveModeLayout`
//      mounts with `AssessmentCameraShell` (autoStart) on the left
//      and a compact calibration sidebar on the right.
//   4. Body pixel height stabilises → "Use this calibration" locks
//      the scale → phase transitions to "armed" (still fullscreen).
//   5. "Start recording" → recording (auto-stops at
//      RECORDING_DURATION_SEC) → uploading → done.
//
// UPLOAD mode: boxed form — file picker + height input + analyse.
//
// Both modes POST video to the same `/api/analyze-overhead-squat`
// endpoint. Structure mirrors D2 Tuck Jump byte-for-byte in flow;
// only labels + result-type imports differ.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  FileVideo,
  Loader2,
  Play,
  RefreshCw,
  RotateCcw,
  SkipForward,
  Upload,
  Video,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import {
  MAX_HEIGHT_CM,
  MIN_HEIGHT_CM,
  STABLE_FRAMES_REQUIRED,
  areReadingsStable,
  buildHeightCalibration,
  checkBodyInFrame,
  computeBodyPixelHeight,
  type BodyHeightReading,
} from "@/lib/calibration/heightCalibration";
import { AssessmentCameraShell } from "@/components/orthopedic/AssessmentCameraShell";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  OverheadSquatReport,
  buildOverheadSquatInterpretation,
} from "@/components/orthopedic/OverheadSquatReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  analyzeOverheadSquatUpload,
  RECORDING_DURATION_SEC,
  TARGET_SESSION_SEC,
  type CalibrationResult,
  type OverheadSquatResult,
} from "@/lib/orthopedic/overheadSquat";

type Mode = "live" | "upload";
type LivePhase = "calibration" | "armed" | "recording" | "uploading" | "done";
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

export function OverheadSquatCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");
  const [phase, setPhase] = useState<LivePhase>("calibration");
  const [error, setError] = useState<string | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [result, setResult] = useState<OverheadSquatResult | null>(null);
  const [now, setNow] = useState<number>(0);

  const [cameraStarted, setCameraStarted] = useState<boolean>(false);

  const [heightInput, setHeightInput] = useState<string>(
    patient?.height_cm && patient.height_cm > 0
      ? patient.height_cm.toFixed(0)
      : "",
  );

  // Auto-skip the pre-camera height step when patient's height is on
  // file — no manual entry needed. See TuckJumpCapture for details.
  useEffect(() => {
    if (
      phase === "calibration"
      && !cameraStarted
      && patient?.height_cm
      && patient.height_cm > 0
    ) {
      setCameraStarted(true);
    }
  }, [phase, cameraStarted, patient?.height_cm]);

  // ── Recording ────────────────────────────────────────────────
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingStartedAtRef = useRef<number>(0);

  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [phase]);

  function getLiveVideoStream(): MediaStream | null {
    const vid = document.querySelector(
      "video[playsinline]",
    ) as HTMLVideoElement | null;
    if (!vid) return null;
    const stream = vid.srcObject;
    return stream instanceof MediaStream ? stream : null;
  }

  function startRecording() {
    const stream = getLiveVideoStream();
    if (!stream) {
      setError(
        "Camera stream is not available. Click Start camera before recording.",
      );
      return;
    }
    const chunks: Blob[] = [];
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, { mimeType: "video/webm" });
    } catch {
      try {
        rec = new MediaRecorder(stream);
      } catch (e) {
        setError(
          `Could not start recording: ${errorMessage(e) ?? "MediaRecorder unavailable"}`,
        );
        return;
      }
    }
    rec.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data);
    };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: rec.mimeType || "video/webm" });
      mediaRecorderRef.current = null;
      recordingChunksRef.current = [];
      void uploadAndAnalyze(blob);
    };
    recordingChunksRef.current = chunks;
    mediaRecorderRef.current = rec;
    recordingStartedAtRef.current = Date.now();
    setError(null);
    setPhase("recording");
    rec.start();
  }

  async function uploadAndAnalyze(blob: Blob) {
    const file = new File([blob], "overhead_squat.webm", {
      type: blob.type,
    });
    setPhase("uploading");
    setError(null);
    try {
      const data = await analyzeOverheadSquatUpload(file, calibration, null);
      setResult(data);
      setPhase("done");
    } catch (e) {
      setError(errorMessage(e) ?? "Analysis failed");
      setPhase("armed");
    }
  }

  function stopRecording() {
    const rec = mediaRecorderRef.current;
    if (rec && rec.state === "recording") {
      rec.stop();
    } else {
      setPhase("armed");
    }
  }

  useEffect(() => {
    if (phase !== "recording") return;
    const elapsedMs = now - recordingStartedAtRef.current;
    if (elapsedMs >= RECORDING_DURATION_SEC * 1000) {
      stopRecording();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [now, phase]);

  const handleCalibrated = useCallback((cal: CalibrationResult | null) => {
    setCalibration(cal);
    setPhase("armed");
  }, []);

  function reset() {
    if (mediaRecorderRef.current) {
      try { mediaRecorderRef.current.stop(); } catch { /* ignore */ }
    }
    mediaRecorderRef.current = null;
    recordingChunksRef.current = [];
    setResult(null);
    setCalibration(null);
    setError(null);
    setPhase("calibration");
  }

  function exitLive() {
    reset();
    setCameraStarted(false);
  }

  const handleFrame = useCallback(
    (_kp: Keypoint[], _video: HTMLVideoElement) => {
      // No live coaching — the backend processes the uploaded clip.
    },
    [],
  );

  // ── Upload mode ──────────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [filePickError, setFilePickError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<OverheadSquatResult | null>(null);
  const [uploadHeightInput, setUploadHeightInput] = useState<string>(
    patient?.height_cm && patient.height_cm > 0
      ? patient.height_cm.toFixed(0)
      : "",
  );
  const [allowUncalibratedUpload, setAllowUncalibratedUpload] =
    useState<boolean>(false);
  const parsedUploadHeightCm = Number.parseFloat(uploadHeightInput);
  const uploadHeightCmValid =
    Number.isFinite(parsedUploadHeightCm) &&
    parsedUploadHeightCm >= MIN_HEIGHT_CM &&
    parsedUploadHeightCm <= MAX_HEIGHT_CM;

  function validateFile(f: File | null): File | null {
    setFilePickError(null);
    if (!f) return null;
    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      setFilePickError(
        `File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_MB} MB.`,
      );
      return null;
    }
    if (f.type && !ACCEPTED_VIDEO_TYPES.includes(f.type)) {
      setFilePickError(
        `Unsupported file type (${f.type}). Use MP4, WebM, MOV, or MKV.`,
      );
      return null;
    }
    return f;
  }

  async function analyzeUpload() {
    if (!uploadFile) {
      setUploadError("Pick a video to analyse.");
      return;
    }
    if (!uploadHeightCmValid && !allowUncalibratedUpload) {
      setUploadError(
        `Enter the patient's height (${MIN_HEIGHT_CM}-${MAX_HEIGHT_CM} cm) or ` +
          `tick "Continue without calibration" to run in relative-units mode.`,
      );
      return;
    }
    setUploadPhase("analyzing");
    setUploadError(null);
    const heightCm = uploadHeightCmValid ? parsedUploadHeightCm : null;
    try {
      const r = await analyzeOverheadSquatUpload(uploadFile, null, heightCm);
      setUploadResult(r);
      setUploadPhase("done");
    } catch (e) {
      setUploadError(errorMessage(e) ?? "Analysis failed");
      setUploadPhase("error");
    }
  }

  function resetUpload() {
    setUploadPhase("idle");
    setUploadFile(null);
    setUploadResult(null);
    setUploadError(null);
    setFilePickError(null);
  }

  // ── Render: done view ────────────────────────────────────────
  const liveDone = mode === "live" && phase === "done" && result !== null;
  const uploadDone =
    mode === "upload" && uploadPhase === "done" && uploadResult !== null;

  if (liveDone || uploadDone) {
    const r = (mode === "live" ? result : uploadResult)!;
    const interpretation = buildOverheadSquatInterpretation(r);
    return (
      <DoneView
        patientName={patient?.name ?? null}
        patient={patient ?? null}
        result={r}
        interpretation={interpretation}
        onReset={mode === "live" ? exitLive : resetUpload}
      />
    );
  }

  // ── Render: capture phases ───────────────────────────────────
  return (
    <div className="space-y-8">
      {isDoctorFlow && (
        <SaveStatusBanner patient={patient} saveStatus={null} />
      )}

      <div className="inline-flex rounded-card border border-border bg-surface p-1">
        <button
          type="button"
          onClick={() => setMode("live")}
          disabled={uploadPhase === "analyzing"}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "live"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          }`}
        >
          <Video className="h-4 w-4" /> Live capture
        </button>
        <button
          type="button"
          onClick={() => setMode("upload")}
          disabled={
            phase === "recording" ||
            phase === "uploading" ||
            uploadPhase === "analyzing"
          }
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "upload"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          }`}
        >
          <Upload className="h-4 w-4" /> Upload video
        </button>
      </div>

      {mode === "live" ? (
        <LiveSection
          phase={phase}
          calibration={calibration}
          patientHeightCm={patient?.height_cm ?? null}
          heightInput={heightInput}
          onHeightInputChange={setHeightInput}
          cameraStarted={cameraStarted}
          onStartCamera={() => setCameraStarted(true)}
          onCalibrated={handleCalibrated}
          handleFrame={handleFrame}
          startRecording={startRecording}
          stopRecording={stopRecording}
          now={now}
          recordingStartedAt={recordingStartedAtRef.current}
          onResetSession={reset}
          onExitLive={exitLive}
          error={error}
        />
      ) : (
        <UploadSection
          phase={uploadPhase}
          file={uploadFile}
          onFile={(f) => setUploadFile(validateFile(f))}
          filePickError={filePickError}
          uploadError={uploadError}
          heightInput={uploadHeightInput}
          onHeightChange={setUploadHeightInput}
          heightValid={uploadHeightCmValid}
          allowUncalibrated={allowUncalibratedUpload}
          onAllowUncalibratedChange={setAllowUncalibratedUpload}
          onAnalyze={analyzeUpload}
          onReset={resetUpload}
        />
      )}
    </div>
  );
}

// ─── Live section ───────────────────────────────────────────────
interface LiveSectionProps {
  phase: LivePhase;
  calibration: CalibrationResult | null;
  patientHeightCm: number | null;
  heightInput: string;
  onHeightInputChange: (s: string) => void;
  cameraStarted: boolean;
  onStartCamera: () => void;
  onCalibrated: (c: CalibrationResult | null) => void;
  handleFrame: (kp: Keypoint[], v: HTMLVideoElement) => void;
  startRecording: () => void;
  stopRecording: () => void;
  now: number;
  recordingStartedAt: number;
  onResetSession: () => void;
  onExitLive: () => void;
  error: string | null;
}

function LiveSection(props: LiveSectionProps) {
  const {
    phase,
    calibration,
    patientHeightCm,
    heightInput,
    onHeightInputChange,
    cameraStarted,
    onStartCamera,
    onCalibrated,
    handleFrame,
    startRecording,
    stopRecording,
    now,
    recordingStartedAt,
    onResetSession,
    onExitLive,
    error,
  } = props;

  const heightCm = Number.parseFloat(heightInput);
  const heightCmValid =
    Number.isFinite(heightCm) &&
    heightCm >= MIN_HEIGHT_CM &&
    heightCm <= MAX_HEIGHT_CM;

  if (phase === "calibration" && !cameraStarted) {
    return (
      <div className="space-y-6">
        <div>
          <p className="eyebrow">Step 1 · Patient height</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight">
            Enter height, then open the camera
          </h2>
          <p className="mt-2 max-w-xl text-sm text-muted">
            The system will convert pixel measurements to centimetres
            using the patient&apos;s standing height. Skipping is allowed —
            classification thresholds are fraction-of-leg-length so the
            score stays valid in relative units.
          </p>
        </div>

        <div className="rounded-card border border-border bg-surface p-5">
          <label
            htmlFor="overheadsquat-live-height"
            className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle"
          >
            Patient height (cm)
          </label>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <input
              id="overheadsquat-live-height"
              type="number"
              inputMode="numeric"
              min={MIN_HEIGHT_CM}
              max={MAX_HEIGHT_CM}
              step={1}
              value={heightInput}
              onChange={(e) => onHeightInputChange(e.target.value)}
              placeholder="e.g. 170"
              className="w-32 rounded-md border border-border bg-background px-3 py-2 text-base tabular text-foreground outline-none focus:border-accent"
            />
            <span className="text-xs text-muted">
              {MIN_HEIGHT_CM}–{MAX_HEIGHT_CM} cm
            </span>
          </div>
          {!heightCmValid && heightInput.length > 0 && (
            <p className="mt-2 text-xs text-error">
              Enter a value between {MIN_HEIGHT_CM} and {MAX_HEIGHT_CM} cm.
            </p>
          )}
          {patientHeightCm && patientHeightCm > 0 && (
            <p className="mt-2 text-xs text-muted">
              Pre-filled from patient record: {patientHeightCm.toFixed(0)} cm.
            </p>
          )}
        </div>

        <div className="flex flex-wrap gap-3">
          <Button onClick={onStartCamera}>
            <Camera className="h-4 w-4" />
            Start camera (fullscreen)
          </Button>
          <p className="w-full text-xs text-muted">
            Clicking &quot;Start camera&quot; opens the immersive full-
            screen capture shell. You can Exit any time from the header.
          </p>
        </div>
      </div>
    );
  }

  return (
    <FullscreenLiveShell
      phase={phase}
      calibration={calibration}
      heightInput={heightInput}
      onHeightInputChange={onHeightInputChange}
      heightCm={heightCm}
      heightCmValid={heightCmValid}
      patientHeightCm={patientHeightCm}
      onCalibrated={onCalibrated}
      handleFrame={handleFrame}
      startRecording={startRecording}
      stopRecording={stopRecording}
      now={now}
      recordingStartedAt={recordingStartedAt}
      onResetSession={onResetSession}
      onExitLive={onExitLive}
      error={error}
    />
  );
}

// ─── Fullscreen live shell ─────────────────────────────────────
interface FullscreenLiveShellProps {
  phase: LivePhase;
  calibration: CalibrationResult | null;
  heightInput: string;
  onHeightInputChange: (s: string) => void;
  heightCm: number;
  heightCmValid: boolean;
  patientHeightCm: number | null;
  onCalibrated: (c: CalibrationResult | null) => void;
  handleFrame: (kp: Keypoint[], v: HTMLVideoElement) => void;
  startRecording: () => void;
  stopRecording: () => void;
  now: number;
  recordingStartedAt: number;
  onResetSession: () => void;
  onExitLive: () => void;
  error: string | null;
}

function FullscreenLiveShell(props: FullscreenLiveShellProps) {
  const {
    phase,
    calibration,
    heightInput,
    onHeightInputChange,
    heightCm,
    heightCmValid,
    patientHeightCm,
    onCalibrated,
    handleFrame,
    startRecording,
    stopRecording,
    now,
    recordingStartedAt,
    onResetSession,
    onExitLive,
    error,
  } = props;

  const [frameReason, setFrameReason] = useState<string>("torso_missing");
  const [stableCount, setStableCount] = useState<number>(0);
  const [calibLocked, setCalibLocked] = useState<boolean>(false);
  const recentReadingsRef = useRef<BodyHeightReading[]>([]);
  const lockedResultRef = useRef<CalibrationResult | null>(null);

  const onLandmarks = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      handleFrame(kp, video);
      if (phase !== "calibration" || calibLocked) return;
      const sh = video.videoHeight;
      const sw = video.videoWidth;
      if (!sh || !sw) return;

      const reading = computeBodyPixelHeight(kp);
      const frameCheck = checkBodyInFrame(kp, reading, sh);
      setFrameReason(frameCheck.ok ? "" : frameCheck.reason);

      if (frameCheck.ok && reading && heightCmValid) {
        const recent = recentReadingsRef.current;
        if (recent.length > 0) {
          const last = recent[recent.length - 1];
          if (!areReadingsStable(last, reading)) recent.length = 0;
        }
        recent.push(reading);
        if (recent.length > STABLE_FRAMES_REQUIRED) recent.shift();
        setStableCount(recent.length);

        if (recent.length >= STABLE_FRAMES_REQUIRED) {
          const sorted = [...recent]
            .map((r) => r.body_pixel_height_px)
            .sort((a, b) => a - b);
          const median = sorted[Math.floor(sorted.length / 2)];
          const cal = buildHeightCalibration(
            median,
            heightCm,
            { width: sw, height: sh },
          );
          if (cal) {
            lockedResultRef.current = cal;
            setCalibLocked(true);
          }
        }
      } else {
        recentReadingsRef.current = [];
        setStableCount(0);
      }
    },
    [handleFrame, phase, calibLocked, heightCmValid, heightCm],
  );

  function calibConfirm() {
    const r = lockedResultRef.current;
    if (!r) return;
    onCalibrated(r);
  }
  function calibRetake() {
    recentReadingsRef.current = [];
    setStableCount(0);
    setCalibLocked(false);
    lockedResultRef.current = null;
  }
  function calibSkip() {
    onCalibrated(null);
  }

  const coachingMessage = (() => {
    if (calibLocked) return null;
    if (!heightCmValid) {
      return `Enter the patient's height in cm (${MIN_HEIGHT_CM}–${MAX_HEIGHT_CM}).`;
    }
    switch (frameReason) {
      case "torso_missing": return "Body not yet tracked — make sure the patient's torso is in frame.";
      case "head_missing":  return "Head not visible — raise the camera or step the patient closer.";
      case "feet_missing":  return "Feet not visible — lower the camera or step the patient back.";
      case "body_partial":  return "Patient not fully in frame yet.";
      case "head_at_frame_edge": return "Head is cropped at the top — patient should step back or camera should be raised.";
      case "feet_at_frame_edge": return "Feet are cropped at the bottom — patient should step back or camera should be lowered.";
      default: return `Stand straight, full body in frame — holding for stable reading (${stableCount}/${STABLE_FRAMES_REQUIRED}).`;
    }
  })();

  const isCalibrating = phase === "calibration";
  const isRecording = phase === "recording";
  const isUploading = phase === "uploading";
  const elapsedSec = Math.max(0, (now - recordingStartedAt) / 1000);
  const remainingSec = Math.max(0, RECORDING_DURATION_SEC - elapsedSec);
  const targetProgress = Math.min(
    100,
    Math.round((elapsedSec / TARGET_SESSION_SEC) * 100),
  );

  return (
    <LiveModeLayout
      title="Overhead Squat Assessment"
      subtitle={
        calibration
          ? `Calibrated · ${calibration.pixels_per_cm.toFixed(2)} px/cm`
          : isCalibrating
            ? "Scale calibration — stand full-body in frame"
            : "Uncalibrated — classification stays valid in relative units"
      }
      onExit={onExitLive}
      camera={(
        <AssessmentCameraShell onFrame={onLandmarks} hideControls autoStart>
          {isCalibrating && calibLocked && (
            <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-emerald-500/95 px-3 py-1.5 text-xs font-medium text-white shadow-lg">
              <CheckCircle2 className="h-4 w-4" />
              Calibrated (height)
            </div>
          )}
          {isCalibrating && !calibLocked && stableCount > 0 && (
            <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
              <p className="text-[10px] uppercase tracking-[0.14em] text-amber-300">
                Acquiring stable reading
              </p>
              <div className="mt-1 h-1.5 w-40 overflow-hidden rounded-full bg-white/15">
                <div
                  className="h-full bg-amber-400 transition-all"
                  style={{
                    width: `${(stableCount / STABLE_FRAMES_REQUIRED) * 100}%`,
                  }}
                />
              </div>
            </div>
          )}
          {isRecording && (
            <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
              <p className="text-[10px] uppercase tracking-[0.14em] text-rose-300">
                ● Recording
              </p>
              <p className="tabular text-2xl font-semibold text-white">
                {remainingSec.toFixed(1)}s
              </p>
              <p className="text-[10px] text-white/70">
                {targetProgress}% of {TARGET_SESSION_SEC}s target
              </p>
            </div>
          )}
          {isUploading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
              <div className="flex items-center gap-2 rounded-lg border border-white/15 bg-black/70 px-4 py-3 text-sm text-white">
                <Loader2 className="h-4 w-4 animate-spin" />
                Analysing session…
              </div>
            </div>
          )}
        </AssessmentCameraShell>
      )}
      sidebar={(
        <>
          {isCalibrating ? (
            <>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-subtle">
                  Calibration
                </p>
                <p className="mt-1 text-xs text-muted">
                  Patient stands straight, arms straight overhead, full body
                  facing the camera. Skip to run in relative-units mode —
                  classification stays valid.
                </p>
              </div>

              <div className="rounded-card border border-border bg-surface p-3">
                <label
                  htmlFor="overheadsquat-fs-height"
                  className="block text-[10px] font-semibold uppercase tracking-[0.12em] text-subtle"
                >
                  Patient height (cm)
                </label>
                <input
                  id="overheadsquat-fs-height"
                  type="number"
                  inputMode="numeric"
                  min={MIN_HEIGHT_CM}
                  max={MAX_HEIGHT_CM}
                  step={1}
                  value={heightInput}
                  onChange={(e) => onHeightInputChange(e.target.value)}
                  disabled={calibLocked}
                  placeholder="e.g. 170"
                  className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm tabular text-foreground outline-none focus:border-accent disabled:opacity-60"
                />
                {!heightCmValid && heightInput.length > 0 && (
                  <p className="mt-1 text-[11px] text-error">
                    Enter {MIN_HEIGHT_CM}–{MAX_HEIGHT_CM} cm.
                  </p>
                )}
                {patientHeightCm && patientHeightCm > 0 && (
                  <p className="mt-1 text-[11px] text-muted">
                    Pre-filled from patient record: {patientHeightCm.toFixed(0)} cm.
                  </p>
                )}
              </div>

              <div
                className={`rounded-card border p-3 text-sm ${
                  calibLocked
                    ? "border-emerald-500/30 bg-emerald-500/5"
                    : "border-border bg-surface"
                }`}
              >
                <div className="flex items-start gap-2">
                  {calibLocked ? (
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
                  ) : (
                    <Loader2
                      className={`mt-0.5 h-4 w-4 shrink-0 ${
                        heightCmValid ? "animate-spin text-amber-600" : "text-muted"
                      }`}
                    />
                  )}
                  <div>
                    <p className="font-medium text-foreground">
                      {calibLocked ? "Calibration locked" : "Acquiring stable reading…"}
                    </p>
                    {coachingMessage && (
                      <p className="mt-1 text-xs text-muted">{coachingMessage}</p>
                    )}
                  </div>
                </div>
                {!calibLocked && (
                  <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                    <div
                      className="h-full bg-accent transition-all"
                      style={{ width: `${(stableCount / STABLE_FRAMES_REQUIRED) * 100}%` }}
                    />
                  </div>
                )}
                {calibLocked && lockedResultRef.current && (
                  <p className="mt-3 text-[11px] tabular text-muted">
                    Scale:{" "}
                    <span className="font-semibold text-foreground">
                      {lockedResultRef.current.pixels_per_cm.toFixed(2)}
                    </span>{" "}
                    px/cm
                  </p>
                )}
              </div>

              <div className="mt-auto flex flex-wrap gap-2">
                {calibLocked ? (
                  <>
                    <Button onClick={calibConfirm}>
                      <CheckCircle2 className="h-4 w-4" />
                      Use this calibration
                    </Button>
                    <Button variant="secondary" onClick={calibRetake}>
                      <RefreshCw className="h-4 w-4" />
                      Re-take
                    </Button>
                  </>
                ) : (
                  <Button variant="ghost" onClick={calibSkip}>
                    <SkipForward className="h-4 w-4" />
                    Skip · use relative units
                  </Button>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                {calibration ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-400/40">
                    <CheckCircle2 className="h-3 w-3" />
                    Calibrated
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/15 px-3 py-1 text-xs font-semibold text-amber-200 ring-1 ring-amber-400/40">
                    <AlertTriangle className="h-3 w-3" />
                    Uncalibrated
                  </span>
                )}
              </div>

              <div className="rounded-card border border-border bg-surface p-3 text-xs text-muted">
                <p className="font-semibold text-foreground">Session brief</p>
                <ol className="mt-2 list-decimal space-y-1 pl-4">
                  <li>Stand still ~1 s facing the camera, arms straight overhead.</li>
                  <li>
                    Perform 3-5 slow overhead squats to about parallel depth
                    within ~{TARGET_SESSION_SEC}s.
                  </li>
                  <li>Return fully to standing between reps, arms overhead throughout.</li>
                </ol>
              </div>

              {isRecording && (
                <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-3 text-sm">
                  <p className="font-medium text-foreground">
                    ● Recording — {remainingSec.toFixed(1)}s remaining
                  </p>
                  <p className="mt-1 text-[11px] text-muted">
                    {targetProgress}% of target session.
                  </p>
                </div>
              )}
              {isUploading && (
                <div className="rounded-card border border-blue-500/30 bg-blue-500/10 p-3 text-sm">
                  <p className="flex items-center gap-2 font-medium text-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analysing session…
                  </p>
                </div>
              )}

              {error && (
                <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-foreground">
                  <AlertTriangle className="mr-2 inline h-4 w-4 text-rose-500" />
                  {error}
                </div>
              )}

              <div className="mt-auto flex flex-wrap gap-2">
                {phase === "armed" && (
                  <Button onClick={startRecording}>
                    <Play className="h-4 w-4" />
                    Start recording
                  </Button>
                )}
                {isRecording && (
                  <Button variant="secondary" onClick={stopRecording}>
                    Stop early
                  </Button>
                )}
                <Button variant="ghost" size="sm" onClick={onResetSession}>
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
              </div>
            </>
          )}
        </>
      )}
    />
  );
}

// ─── Upload section ─────────────────────────────────────────────
interface UploadSectionProps {
  phase: UploadPhase;
  file: File | null;
  onFile: (f: File | null) => void;
  filePickError: string | null;
  uploadError: string | null;
  heightInput: string;
  onHeightChange: (s: string) => void;
  heightValid: boolean;
  allowUncalibrated: boolean;
  onAllowUncalibratedChange: (v: boolean) => void;
  onAnalyze: () => void;
  onReset: () => void;
}

function UploadSection(props: UploadSectionProps) {
  const {
    phase,
    file,
    onFile,
    filePickError,
    uploadError,
    heightInput,
    onHeightChange,
    heightValid,
    allowUncalibrated,
    onAllowUncalibratedChange,
    onAnalyze,
    onReset,
  } = props;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          Upload Overhead Squat video
        </h2>
        <p className="mt-2 text-sm text-muted">
          Single clip containing 3-5 slow overhead squats (FRONTAL view,
          arms overhead throughout). The backend derives scale calibration
          from the patient&apos;s height + the standing window of the clip.
        </p>
      </div>

      <div className="rounded-card border border-border bg-surface p-5">
        <label htmlFor="overheadsquat-height" className="block text-sm font-medium">
          Patient height (cm)
        </label>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <input
            id="overheadsquat-height"
            type="number"
            inputMode="numeric"
            min={MIN_HEIGHT_CM}
            max={MAX_HEIGHT_CM}
            step={1}
            value={heightInput}
            onChange={(e) => onHeightChange(e.target.value)}
            disabled={phase === "analyzing"}
            className="w-32 rounded-md border border-border bg-background px-3 py-2 text-sm tabular"
            placeholder="e.g. 170"
          />
          <span className="text-xs text-muted">
            {MIN_HEIGHT_CM}–{MAX_HEIGHT_CM} cm
          </span>
        </div>
        <label className="mt-3 flex items-center gap-2 text-xs text-muted">
          <input
            type="checkbox"
            checked={allowUncalibrated}
            onChange={(e) => onAllowUncalibratedChange(e.target.checked)}
            disabled={phase === "analyzing"}
          />
          Continue without calibration (classification uses fraction-of-
          leg-length thresholds and stays valid without cm scaling)
        </label>
      </div>

      <FilePickerCard
        label="Overhead-squat clip"
        file={file}
        onPick={onFile}
        disabled={phase === "analyzing"}
      />
      {filePickError && (
        <p className="text-xs text-rose-600">{filePickError}</p>
      )}

      {uploadError && (
        <div className="rounded-card border border-rose-500/30 bg-rose-500/5 p-4 text-sm text-foreground">
          <AlertTriangle className="mr-2 inline h-4 w-4 text-rose-600" />
          {uploadError}
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <Button
          onClick={onAnalyze}
          disabled={
            phase === "analyzing" ||
            !file ||
            (!heightValid && !allowUncalibrated)
          }
        >
          {phase === "analyzing" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          Analyse session
        </Button>
        <Button
          variant="secondary"
          onClick={onReset}
          disabled={phase === "analyzing"}
        >
          <RotateCcw className="h-4 w-4" />
          Reset
        </Button>
      </div>
    </div>
  );
}

function FilePickerCard({
  label,
  file,
  onPick,
  disabled,
}: {
  label: string;
  file: File | null;
  onPick: (f: File | null) => void;
  disabled: boolean;
}) {
  return (
    <label
      className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border-2 border-dashed border-border bg-surface p-6 text-center text-sm transition hover:bg-elevated ${
        disabled ? "opacity-60 cursor-not-allowed" : ""
      }`}
    >
      <input
        type="file"
        accept={ACCEPTED_VIDEO_TYPES.join(",")}
        className="hidden"
        disabled={disabled}
        onChange={(e) => onPick(e.target.files?.[0] ?? null)}
      />
      <FileVideo className="h-6 w-6 text-muted" />
      <p className="font-medium">{label}</p>
      {file ? (
        <p className="text-xs text-muted">
          {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
        </p>
      ) : (
        <p className="text-xs text-muted">Drop or click to choose a video</p>
      )}
    </label>
  );
}

// ─── Done view ─────────────────────────────────────────────────
function DoneView({
  patientName,
  patient,
  result,
  interpretation,
  onReset,
}: {
  patientName: string | null;
  patient: ReturnType<typeof usePatientContext>["patient"] | null;
  result: OverheadSquatResult;
  interpretation: string;
  onReset: () => void;
}) {
  return (
    <div className="space-y-10">
      <OverheadSquatReport
        patientName={patientName}
        patient={patient ?? null}
        result={result}
        interpretation={interpretation}
      />

      <div className="no-pdf flex flex-wrap items-center justify-between gap-3 border-t border-border pt-6">
        <Button variant="secondary" onClick={onReset}>
          <RotateCcw className="h-4 w-4" />
          New session
        </Button>

        <SaveToPatientButton
          buildPayload={() => ({
            module: "overhead_squat",
            metrics: {
              result,
            },
            observations: { interpretation },
          })}
        />
      </div>
    </div>
  );
}
