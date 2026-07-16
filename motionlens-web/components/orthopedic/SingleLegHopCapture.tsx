"use client";
// D3 Single-Leg Hop capture flow.
//
// Phase progression (live mode, sequential two-leg):
//   side_picker → calibration → armed_left → recording_left
//   → armed_right → recording_right → done
//
// Upload mode is simpler — two file pickers (one per leg) + an
// optional patient_height_cm field that the backend uses to
// derive pixels_per_cm from the standing window of each clip.
//
// LSI is computed CLIENT-side via buildCombinedResult() after both
// per-leg backend results are in.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
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
import { HeightCalibrationStep } from "@/components/calibration/HeightCalibrationStep";
import {
  MAX_HEIGHT_CM,
  MIN_HEIGHT_CM,
} from "@/lib/calibration/heightCalibration";
import { SingleLegHopLiveCamera } from "@/components/orthopedic/SingleLegHopLiveCamera";
import {
  SingleLegHopReport,
  buildSingleLegHopInterpretation,
} from "@/components/orthopedic/SingleLegHopReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  analyzeSingleLegHopUpload,
  buildCombinedResult,
  RECORDING_DURATION_SEC,
  type CalibrationResult,
  type Side,
  type SingleLegHopCombinedResult,
  type SingleLegHopResult,
} from "@/lib/orthopedic/singleLegHop";

type Mode = "live" | "upload";
type LivePhase =
  | "side_picker"
  | "calibration"
  | "armed_left"
  | "recording_left"
  | "armed_right"
  | "recording_right"
  | "uploading_left"
  | "uploading_right"
  | "done";
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
  /** Local recording is delegated to the browser MediaRecorder via
   *  a temporary File blob; this state captures only the timing
   *  + blob reference for the per-leg backend upload. */
  blob: Blob | null;
}

export function SingleLegHopCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");
  const [phase, setPhase] = useState<LivePhase>("side_picker");
  const [error, setError] = useState<string | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [firstSide, setFirstSide] = useState<Side | null>(null);
  const [leftResult, setLeftResult] = useState<SingleLegHopResult | null>(null);
  const [rightResult, setRightResult] = useState<SingleLegHopResult | null>(null);
  const [now, setNow] = useState<number>(0);

  // ── Recording (per leg) ──────────────────────────────────────
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingStartedAtRef = useRef<number>(0);
  const currentRecordingSideRef = useRef<Side | null>(null);

  useEffect(() => {
    if (phase !== "recording_left" && phase !== "recording_right") return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [phase]);

  /** Pull the live camera's <video> element via the global capture
   *  hook the LiveCamera component installs. Used to wire up the
   *  MediaRecorder against the same stream. */
  function getLiveVideoStream(): MediaStream | null {
    const vid = document.querySelector(
      "video[playsinline]",
    ) as HTMLVideoElement | null;
    if (!vid) return null;
    const stream = vid.srcObject;
    return stream instanceof MediaStream ? stream : null;
  }

  function startRecording(side: Side) {
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
      const recordedSide = currentRecordingSideRef.current;
      currentRecordingSideRef.current = null;
      mediaRecorderRef.current = null;
      recordingChunksRef.current = [];
      void uploadAndAnalyze(blob, recordedSide);
    };
    recordingChunksRef.current = chunks;
    mediaRecorderRef.current = rec;
    currentRecordingSideRef.current = side;
    recordingStartedAtRef.current = Date.now();
    setError(null);
    setPhase(side === "left" ? "recording_left" : "recording_right");
    rec.start();
  }

  async function uploadAndAnalyze(blob: Blob, side: Side | null) {
    if (!side) return;
    const filename =
      side === "left" ? "single_leg_hop_left.webm" : "single_leg_hop_right.webm";
    const file = new File([blob], filename, { type: blob.type });
    setPhase(side === "left" ? "uploading_left" : "uploading_right");
    setError(null);
    try {
      const result = await analyzeSingleLegHopUpload(
        file,
        side,
        calibration,
        patient?.height_cm ?? null,
      );
      if (side === "left") setLeftResult(result);
      else setRightResult(result);
      // Advance to the other side, or finish if both done.
      const otherSide: Side = side === "left" ? "right" : "left";
      const otherDone =
        otherSide === "left" ? leftResult !== null : rightResult !== null;
      if (otherDone) {
        setPhase("done");
      } else {
        setPhase(otherSide === "left" ? "armed_left" : "armed_right");
      }
    } catch (e) {
      setError(
        errorMessage(e) ??
          `${side === "left" ? "Left" : "Right"} leg analysis failed`,
      );
      // Park the operator at the armed phase for the failed side so
      // they can retry without resetting calibration.
      setPhase(side === "left" ? "armed_left" : "armed_right");
    }
  }

  function stopRecording() {
    const rec = mediaRecorderRef.current;
    if (rec && rec.state === "recording") {
      rec.stop();
    } else {
      // Safety net — if recorder isn't running, drop straight back
      // to armed state.
      const side = currentRecordingSideRef.current;
      setPhase(side === "left" ? "armed_left" : "armed_right");
      currentRecordingSideRef.current = null;
    }
  }

  // Auto-stop after RECORDING_DURATION_SEC
  useEffect(() => {
    if (phase !== "recording_left" && phase !== "recording_right") return;
    const elapsedMs = now - recordingStartedAtRef.current;
    if (elapsedMs >= RECORDING_DURATION_SEC * 1000) {
      stopRecording();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [now, phase]);

  // ── Phase transitions ────────────────────────────────────────
  function pickFirstSide(s: Side) {
    setFirstSide(s);
    setError(null);
    setPhase("calibration");
  }

  const handleCalibrated = useCallback(
    (cal: CalibrationResult | null) => {
      setCalibration(cal);
      setPhase(firstSide === "left" ? "armed_left" : "armed_right");
    },
    [firstSide],
  );

  // Auto-skip HeightCalibrationStep when patient height is on file —
  // backend derives px/cm from patient_height_cm at analyse time. Runs
  // only after the side_picker step has resolved firstSide.
  useEffect(() => {
    if (
      mode === "live" &&
      phase === "calibration" &&
      firstSide !== null &&
      !calibration &&
      patient?.height_cm &&
      patient.height_cm > 0
    ) {
      handleCalibrated(null);
    }
  }, [
    mode,
    phase,
    firstSide,
    calibration,
    patient?.height_cm,
    handleCalibrated,
  ]);

  function reset() {
    mediaRecorderRef.current = null;
    recordingChunksRef.current = [];
    currentRecordingSideRef.current = null;
    setLeftResult(null);
    setRightResult(null);
    setCalibration(null);
    setFirstSide(null);
    setError(null);
    setPhase("side_picker");
  }

  // Frame handler — kept minimal in the new D3 flow. The backend
  // detects takeoff/landing from the uploaded clip; live preview
  // just needs to draw the skeleton (handled inside LiveCamera).
  const handleFrame = useCallback(
    (_kp: Keypoint[], _video: HTMLVideoElement) => {
      // No live coaching state yet — could be wired later (live
      // takeoff/landing detection mirrors the backend math; out of
      // scope for the initial ship).
    },
    [],
  );

  // ── Upload mode ──────────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadLeftFile, setUploadLeftFile] = useState<File | null>(null);
  const [uploadRightFile, setUploadRightFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [filePickError, setFilePickError] = useState<string | null>(null);
  const [uploadLeftResult, setUploadLeftResult] =
    useState<SingleLegHopResult | null>(null);
  const [uploadRightResult, setUploadRightResult] =
    useState<SingleLegHopResult | null>(null);
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

  async function analyzeUploads() {
    if (!uploadLeftFile && !uploadRightFile) {
      setUploadError("Pick at least one video.");
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
      const tasks: Array<Promise<void>> = [];
      if (uploadLeftFile) {
        tasks.push(
          analyzeSingleLegHopUpload(uploadLeftFile, "left", null, heightCm)
            .then((r) => {
              setUploadLeftResult(r);
            }),
        );
      }
      if (uploadRightFile) {
        tasks.push(
          analyzeSingleLegHopUpload(uploadRightFile, "right", null, heightCm)
            .then((r) => {
              setUploadRightResult(r);
            }),
        );
      }
      await Promise.all(tasks);
      setUploadPhase("done");
    } catch (e) {
      setUploadError(errorMessage(e) ?? "Analysis failed");
      setUploadPhase("error");
    }
  }

  function resetUpload() {
    setUploadPhase("idle");
    setUploadLeftFile(null);
    setUploadRightFile(null);
    setUploadLeftResult(null);
    setUploadRightResult(null);
    setUploadError(null);
    setFilePickError(null);
  }

  // ── Render: choose between live "done" and upload "done" combined ─
  const liveAllDone = leftResult !== null && rightResult !== null;
  const uploadAllDone =
    uploadPhase === "done" &&
    (uploadLeftResult !== null || uploadRightResult !== null);

  if ((mode === "live" && liveAllDone) || uploadAllDone) {
    const left = mode === "live" ? leftResult : uploadLeftResult;
    const right = mode === "live" ? rightResult : uploadRightResult;
    const combined = buildCombinedResult(left, right);
    const interpretation = buildSingleLegHopInterpretation(combined);
    return (
      <DoneView
        patientName={patient?.name ?? null}
        patient={patient ?? null}
        combined={combined}
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

      {/* Mode toggle */}
      <div className="inline-flex rounded-card border border-border bg-surface p-1">
        <button
          type="button"
          onClick={() => setMode("live")}
          disabled={
            phase !== "side_picker" && phase !== "calibration" || uploadPhase === "analyzing"
          }
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
            (phase !== "side_picker" && phase !== "calibration") ||
            uploadPhase === "analyzing"
          }
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "upload"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          }`}
        >
          <Upload className="h-4 w-4" /> Upload videos
        </button>
      </div>

      {mode === "live" ? (
        <LiveSection
          phase={phase}
          firstSide={firstSide}
          onPickFirstSide={pickFirstSide}
          onCalibrated={handleCalibrated}
          calibration={calibration}
          patientHeightCm={patient?.height_cm ?? null}
          handleFrame={handleFrame}
          startRecording={startRecording}
          stopRecording={stopRecording}
          now={now}
          recordingStartedAt={recordingStartedAtRef.current}
          leftDone={leftResult !== null}
          rightDone={rightResult !== null}
          onResetSession={reset}
          error={error}
        />
      ) : (
        <UploadSection
          phase={uploadPhase}
          leftFile={uploadLeftFile}
          rightFile={uploadRightFile}
          onLeftFile={(f) => setUploadLeftFile(validateFile(f))}
          onRightFile={(f) => setUploadRightFile(validateFile(f))}
          filePickError={filePickError}
          uploadError={uploadError}
          heightInput={uploadHeightInput}
          onHeightChange={setUploadHeightInput}
          heightValid={uploadHeightCmValid}
          allowUncalibrated={allowUncalibratedUpload}
          onAllowUncalibratedChange={setAllowUncalibratedUpload}
          onAnalyze={analyzeUploads}
          onReset={resetUpload}
        />
      )}
    </div>
  );
}

// ─── Live section ────────────────────────────────────────────────
interface LiveSectionProps {
  phase: LivePhase;
  firstSide: Side | null;
  onPickFirstSide: (s: Side) => void;
  onCalibrated: (c: CalibrationResult | null) => void;
  calibration: CalibrationResult | null;
  patientHeightCm: number | null;
  handleFrame: (kp: Keypoint[], v: HTMLVideoElement) => void;
  startRecording: (s: Side) => void;
  stopRecording: () => void;
  now: number;
  recordingStartedAt: number;
  leftDone: boolean;
  rightDone: boolean;
  onResetSession: () => void;
  error: string | null;
}

function LiveSection(props: LiveSectionProps) {
  const {
    phase,
    onPickFirstSide,
    onCalibrated,
    calibration,
    patientHeightCm,
    handleFrame,
    startRecording,
    stopRecording,
    now,
    recordingStartedAt,
    leftDone,
    rightDone,
    onResetSession,
    error,
  } = props;

  if (phase === "side_picker") {
    return (
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          Choose the first leg to test
        </h2>
        <p className="mt-2 text-sm text-muted">
          Hop on ONE leg at a time. We&apos;ll record three trials per leg.
        </p>
        <div className="mt-6 grid gap-3 md:grid-cols-2">
          <Button onClick={() => onPickFirstSide("left")}>Start with LEFT leg</Button>
          <Button onClick={() => onPickFirstSide("right")}>Start with RIGHT leg</Button>
        </div>
      </div>
    );
  }

  if (phase === "calibration") {
    return (
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          Scale calibration
        </h2>
        <p className="mt-2 text-sm text-muted">
          Patient stands straight, fully in frame. We measure body pixel
          height and convert to centimetres using their standing height.
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

  const currentSide: Side =
    phase === "armed_left" ||
    phase === "recording_left" ||
    phase === "uploading_left"
      ? "left"
      : "right";

  const isRecording =
    phase === "recording_left" || phase === "recording_right";
  const isUploading =
    phase === "uploading_left" || phase === "uploading_right";
  const elapsedSec = Math.max(0, (now - recordingStartedAt) / 1000);
  const remainingSec = Math.max(0, RECORDING_DURATION_SEC - elapsedSec);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <span className="rounded-full bg-elevated px-3 py-1 text-xs font-medium uppercase tracking-wide text-subtle">
          {currentSide === "left" ? "Left leg" : "Right leg"} ·{" "}
          {leftDone || rightDone ? "Second of 2" : "First of 2"}
        </span>
        {calibration && (
          <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-700 dark:text-emerald-300">
            <CheckCircle2 className="mr-1 inline h-3 w-3" />
            Calibrated · {calibration.pixels_per_cm.toFixed(2)} px/cm
          </span>
        )}
        {!calibration && (
          <span className="rounded-full bg-amber-500/15 px-3 py-1 text-xs font-medium text-amber-700 dark:text-amber-300">
            <AlertTriangle className="mr-1 inline h-3 w-3" />
            Uncalibrated — relative units only
          </span>
        )}
      </div>

      <SingleLegHopLiveCamera onFrame={handleFrame} onError={() => {}} />

      {isRecording && (
        <div className="rounded-card border border-rose-500/30 bg-rose-500/5 p-4 text-sm">
          <p className="font-medium text-foreground">
            ● Recording {currentSide} leg —{" "}
            <span className="tabular">{remainingSec.toFixed(1)}s</span>{" "}
            remaining
          </p>
          <p className="mt-1 text-xs text-muted">
            Stand still on the {currentSide} leg for ~1 s, then hop forward.
            Up to 3 trials. Auto-stops at {RECORDING_DURATION_SEC} s.
          </p>
        </div>
      )}
      {isUploading && (
        <div className="rounded-card border border-blue-500/30 bg-blue-500/5 p-4 text-sm">
          <p className="flex items-center gap-2 font-medium text-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Analysing {currentSide} leg recording…
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
        {(phase === "armed_left" || phase === "armed_right") && (
          <Button onClick={() => startRecording(currentSide)}>
            <Play className="h-4 w-4" />
            Start recording {currentSide} leg
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
  leftFile: File | null;
  rightFile: File | null;
  onLeftFile: (f: File | null) => void;
  onRightFile: (f: File | null) => void;
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
    leftFile,
    rightFile,
    onLeftFile,
    onRightFile,
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
          Upload Single-Leg Hop videos
        </h2>
        <p className="mt-2 text-sm text-muted">
          Two clips — one per leg. The backend derives scale calibration from
          the patient&apos;s height + the standing window of each clip.
        </p>
      </div>

      <div className="rounded-card border border-border bg-surface p-5">
        <label htmlFor="hop-height" className="block text-sm font-medium">
          Patient height (cm)
        </label>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <input
            id="hop-height"
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
          Continue without calibration (relative units only, no LSI)
        </label>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <FilePickerCard
          label="LEFT leg clip"
          file={leftFile}
          onPick={onLeftFile}
          disabled={phase === "analyzing"}
        />
        <FilePickerCard
          label="RIGHT leg clip"
          file={rightFile}
          onPick={onRightFile}
          disabled={phase === "analyzing"}
        />
      </div>
      {filePickError && (
        <p className="text-xs text-rose-600">{filePickError}</p>
      )}

      {(uploadError) && (
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
            (!leftFile && !rightFile) ||
            (!heightValid && !allowUncalibrated)
          }
        >
          {phase === "analyzing" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          Analyse uploads
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
  combined,
  interpretation,
  onReset,
}: {
  patientName: string | null;
  patient: ReturnType<typeof usePatientContext>["patient"] | null;
  combined: SingleLegHopCombinedResult;
  interpretation: string;
  onReset: () => void;
}) {
  return (
    <div className="space-y-10">
      <SingleLegHopReport
        patientName={patientName}
        patient={patient ?? null}
        combined={combined}
        interpretation={interpretation}
      />

      <div className="no-pdf flex flex-wrap items-center justify-between gap-3 border-t border-border pt-6">
        <Button variant="secondary" onClick={onReset}>
          <RotateCcw className="h-4 w-4" />
          New session
        </Button>

        <SaveToPatientButton
          buildPayload={() => ({
            module: "single_leg_hop",
            metrics: {
              left: combined.left,
              right: combined.right,
              lsi_pct: combined.lsi_pct,
              lsi_class: combined.lsi_class,
              weaker_cm: combined.weaker_cm,
              stronger_cm: combined.stronger_cm,
              weaker_side: combined.weaker_side,
              calibration: combined.calibration,
            },
            observations: { interpretation },
          })}
        />
      </div>
    </div>
  );
}
