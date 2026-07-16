"use client";
// D4 Counter-Movement Jump capture flow.
//
// Phase progression (live mode):
//   calibration → armed → recording → done
//
// Simpler than D3 — no side_picker (both legs together, single
// recording). Upload mode is a single file picker.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  FileVideo,
  Loader2,
  Play,
  RotateCcw,
  Upload,
  Video,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { HeightCalibrationStep } from "@/components/calibration/HeightCalibrationStep";
import {
  MAX_HEIGHT_CM,
  MIN_HEIGHT_CM,
} from "@/lib/calibration/heightCalibration";
import { CMJLiveCamera } from "@/components/orthopedic/CMJLiveCamera";
import {
  CMJReport,
  buildCMJInterpretation,
} from "@/components/orthopedic/CMJReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  analyzeCounterMovementJumpUpload,
  RECORDING_DURATION_SEC,
  type CalibrationResult,
  type CMJResult,
} from "@/lib/orthopedic/counterMovementJump";

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

export function CMJCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");
  const [phase, setPhase] = useState<LivePhase>("calibration");
  const [error, setError] = useState<string | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [result, setResult] = useState<CMJResult | null>(null);
  const [now, setNow] = useState<number>(0);

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
    const file = new File([blob], "counter_movement_jump.webm", {
      type: blob.type,
    });
    setPhase("uploading");
    setError(null);
    try {
      const data = await analyzeCounterMovementJumpUpload(
        file,
        calibration,
        patient?.height_cm ?? null,
      );
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

  // Auto-stop after RECORDING_DURATION_SEC
  useEffect(() => {
    if (phase !== "recording") return;
    const elapsedMs = now - recordingStartedAtRef.current;
    if (elapsedMs >= RECORDING_DURATION_SEC * 1000) {
      stopRecording();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [now, phase]);

  const handleCalibrated = useCallback(
    (cal: CalibrationResult | null) => {
      setCalibration(cal);
      setPhase("armed");
    },
    [],
  );

  function reset() {
    mediaRecorderRef.current = null;
    recordingChunksRef.current = [];
    setResult(null);
    setCalibration(null);
    setError(null);
    setPhase("calibration");
  }

  const handleFrame = useCallback(
    (_kp: Keypoint[], _video: HTMLVideoElement) => {
      // No live coaching state; the backend processes the uploaded
      // clip.
    },
    [],
  );

  // ── Upload mode ──────────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [filePickError, setFilePickError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<CMJResult | null>(null);
  const [uploadHeightInput, setUploadHeightInput] = useState<string>(
    patient?.height_cm && patient.height_cm > 0
      ? patient.height_cm.toFixed(0)
      : "",
  );
  useEffect(() => {
    if (patient?.height_cm && patient.height_cm > 0) {
      setUploadHeightInput((prev) =>
        prev === "" ? patient.height_cm!.toFixed(0) : prev,
      );
    }
  }, [patient?.height_cm]);
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
      const r = await analyzeCounterMovementJumpUpload(
        uploadFile,
        null,
        heightCm,
      );
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
    const interpretation = buildCMJInterpretation(r);
    return (
      <DoneView
        patientName={patient?.name ?? null}
        patient={patient ?? null}
        result={r}
        interpretation={interpretation}
        onReset={mode === "live" ? reset : resetUpload}
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
          onCalibrated={handleCalibrated}
          handleFrame={handleFrame}
          startRecording={startRecording}
          stopRecording={stopRecording}
          now={now}
          recordingStartedAt={recordingStartedAtRef.current}
          onResetSession={reset}
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
  onCalibrated: (c: CalibrationResult | null) => void;
  handleFrame: (kp: Keypoint[], v: HTMLVideoElement) => void;
  startRecording: () => void;
  stopRecording: () => void;
  now: number;
  recordingStartedAt: number;
  onResetSession: () => void;
  error: string | null;
}

function LiveSection(props: LiveSectionProps) {
  const {
    phase,
    calibration,
    patientHeightCm,
    onCalibrated,
    handleFrame,
    startRecording,
    stopRecording,
    now,
    recordingStartedAt,
    onResetSession,
    error,
  } = props;

  if (phase === "calibration") {
    return (
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          Scale calibration
        </h2>
        <p className="mt-2 text-sm text-muted">
          Patient stands straight, fully in frame. The system measures
          body pixel height and converts to centimetres using their
          standing height. Skipping the step is allowed — the test still
          runs in relative-units mode (flight time + physics estimate
          remain valid).
        </p>
        <div className="mt-6">
          <HeightCalibrationStep
            defaultHeightCm={patientHeightCm}
            onCalibrated={onCalibrated}
          />
        </div>
      </div>
    );
  }

  const isRecording = phase === "recording";
  const isUploading = phase === "uploading";
  const elapsedSec = Math.max(0, (now - recordingStartedAt) / 1000);
  const remainingSec = Math.max(0, 25 - elapsedSec);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        {calibration ? (
          <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-700 dark:text-emerald-300">
            Calibrated · {calibration.pixels_per_cm.toFixed(2)} px/cm
          </span>
        ) : (
          <span className="rounded-full bg-amber-500/15 px-3 py-1 text-xs font-medium text-amber-700 dark:text-amber-300">
            <AlertTriangle className="mr-1 inline h-3 w-3" />
            Uncalibrated — pixel heights only (flight time still valid)
          </span>
        )}
      </div>

      <CMJLiveCamera onFrame={handleFrame} onError={() => {}} />

      {isRecording && (
        <div className="rounded-card border border-rose-500/30 bg-rose-500/5 p-4 text-sm">
          <p className="font-medium text-foreground">
            ● Recording —{" "}
            <span className="tabular">{remainingSec.toFixed(1)}s</span>{" "}
            remaining
          </p>
          <p className="mt-1 text-xs text-muted">
            Stand still for ~1 s, then perform up to 3 counter-movement
            jumps. Land on both feet each time.
          </p>
        </div>
      )}
      {isUploading && (
        <div className="rounded-card border border-blue-500/30 bg-blue-500/5 p-4 text-sm">
          <p className="flex items-center gap-2 font-medium text-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Analysing jump…
          </p>
        </div>
      )}

      {error && (
        <div className="rounded-card border border-rose-500/30 bg-rose-500/5 p-4 text-sm text-foreground">
          <AlertTriangle className="mr-2 inline h-4 w-4 text-rose-600" />
          {error}
        </div>
      )}

      <div className="flex flex-wrap gap-3">
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
        <Button variant="secondary" onClick={onResetSession}>
          <RotateCcw className="h-4 w-4" />
          Reset session
        </Button>
      </div>
    </div>
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
          Upload CMJ video
        </h2>
        <p className="mt-2 text-sm text-muted">
          Single clip containing 1–3 counter-movement jumps. The backend
          derives scale calibration from the patient&apos;s height + the
          standing window of the clip.
        </p>
      </div>

      <div className="rounded-card border border-border bg-surface p-5">
        <label htmlFor="cmj-height" className="block text-sm font-medium">
          Patient height (cm)
        </label>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <input
            id="cmj-height"
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
          Continue without calibration (relative units only; flight time
          + physics estimate stay valid)
        </label>
      </div>

      <FilePickerCard
        label="Jump clip"
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
          Analyse jump
        </Button>
        <Button variant="secondary" onClick={onReset} disabled={phase === "analyzing"}>
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
  result: CMJResult;
  interpretation: string;
  onReset: () => void;
}) {
  return (
    <div className="space-y-10">
      <CMJReport
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
            module: "counter_movement_jump",
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
