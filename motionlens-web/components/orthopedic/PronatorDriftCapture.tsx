"use client";
// Pronator Drift (E2) capture flow.
//
// Single trial — patient holds both arms extended forward at shoulder
// height with palms up, eyes closed, for ~20 s. Audio cues mark the
// start + end of the hold (eyes are closed, so no visual cue lands).
//
// State machine:
//   idle
//     → countdown   (3-sec audible countdown, arms must be trackable)
//     → recording   (start-beep, 20 s hold, sampling at 10 Hz)
//     → done        (end-beep; summarizeTrial computes baseline +
//                    drift + velocity + classification)
//
// Frontal-view static-hold module — closest references are Tandem
// Walk (frontal, full-body) and Modified Thomas (settle-then-capture
// math). No L/R split: ONE clip captures BOTH arms simultaneously.

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
import { PronatorDriftLiveCamera } from "@/components/orthopedic/PronatorDriftLiveCamera";
import { PronatorDriftReport } from "@/components/orthopedic/PronatorDriftReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  COUNTDOWN_SEC,
  SAMPLE_INTERVAL_MS,
  TARGET_HOLD_DURATION_SEC,
  analyzePronatorDriftUpload,
  buildInterpretation,
  computeShoulderWidthPx,
  computeShoulderY,
  computeWristY,
  isBothArmsTrackable,
  playCountdownBeep,
  playEndBeep,
  playStartBeep,
  summarizeTrial,
  type FrameSample,
  type PronatorDriftResult,
} from "@/lib/orthopedic/pronatorDrift";

type Mode = "live" | "upload";
type Phase = "idle" | "countdown" | "recording" | "done";
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
  startedAt: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  lastSampleAt: number;
  screenshot: string | null;
}

export function PronatorDriftCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PronatorDriftResult | null>(null);
  const [now, setNow] = useState<number>(0);

  // Pre-record gate + visible-status badge.
  const [armsTrackable, setArmsTrackable] = useState<boolean>(false);

  // Countdown state (ticking 3 → 2 → 1 → GO).
  const [countdownRemaining, setCountdownRemaining] = useState<number>(0);

  const recordingRef = useRef<RecordingState | null>(null);
  const countdownTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [phase]);

  // Cleanup any pending countdown timers on unmount / phase change.
  useEffect(() => {
    return () => {
      for (const h of countdownTimersRef.current) clearTimeout(h);
      countdownTimersRef.current = [];
    };
  }, []);

  const finishTrial = useCallback((termination: "completed" | "timeout" | "stopped") => {
    const rec = recordingRef.current;
    if (!rec) return;
    playEndBeep();
    if (!rec.screenshot) {
      const grab = (window as unknown as {
        __pronatorDriftCapture?: () => string | null;
      }).__pronatorDriftCapture;
      if (grab) {
        const url = grab();
        if (url) rec.screenshot = url;
      }
    }
    const summary = summarizeTrial(
      rec.startedAt,
      Date.now(),
      termination,
      rec.samples,
      rec.keypoints,
      rec.screenshot,
    );
    setResult(summary);
    recordingRef.current = null;
    setPhase("done");
  }, []);

  // Per-frame callback ------------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    setArmsTrackable(isBothArmsTrackable(kp));

    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    const elapsedSec = (tNow - rec.startedAt) / 1000;
    if (elapsedSec >= TARGET_HOLD_DURATION_SEC) {
      finishTrial("completed");
      return;
    }

    const sample: FrameSample = {
      t_ms: tNow - rec.startedAt,
      left_wrist_y:     computeWristY(kp, "left"),
      right_wrist_y:    computeWristY(kp, "right"),
      left_shoulder_y:  computeShoulderY(kp, "left"),
      right_shoulder_y: computeShoulderY(kp, "right"),
      shoulder_width_px: computeShoulderWidthPx(kp),
    };
    rec.samples.push(sample);
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));
  }, [phase, finishTrial]);

  // ── Upload-mode state ──────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [filePickError, setFilePickError] = useState<string | null>(null);

  function validateAndSetFile(file: File | null) {
    setFilePickError(null);
    if (!file) { setUploadFile(null); return; }
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
    setUploadFile(file);
  }

  async function analyzeUpload() {
    if (!uploadFile) return;
    setUploadPhase("analyzing");
    setUploadProgress(0);
    setUploadError(null);
    setError(null);
    try {
      const r = await analyzePronatorDriftUpload(uploadFile, setUploadProgress);
      setResult(r);
      setUploadPhase("done");
    } catch (e) {
      setUploadError(errorMessage(e) ?? "Analysis failed.");
      setUploadPhase("error");
    }
  }

  function resetUpload() {
    setUploadFile(null);
    setUploadProgress(0);
    setUploadError(null);
    setFilePickError(null);
    setUploadPhase("idle");
    setResult(null);
    setError(null);
  }

  function switchMode(next: Mode) {
    if (next === mode) return;
    if (phase === "countdown" || phase === "recording" || uploadPhase === "analyzing") return;
    recordingRef.current = null;
    for (const h of countdownTimersRef.current) clearTimeout(h);
    countdownTimersRef.current = [];
    setResult(null);
    setPhase("idle");
    setError(null);
    resetUpload();
    setMode(next);
  }

  // Live-mode session start: 3-second audible countdown, then begin
  // sampling. The first audio call here MUST happen inside this user-
  // gesture handler so the browser's autoplay policy lets the beeps
  // through.
  function startRecording() {
    if (!armsTrackable) {
      setError(
        "Both arms aren't yet trackable. Position the camera so the patient's " +
        "extended arms (shoulders + wrists) are fully visible before starting.",
      );
      return;
    }
    setError(null);
    setPhase("countdown");
    setCountdownRemaining(COUNTDOWN_SEC);

    // Schedule countdown ticks. Each tick plays a tone; the final
    // tick transitions to recording + plays the higher START beep.
    for (const h of countdownTimersRef.current) clearTimeout(h);
    countdownTimersRef.current = [];
    playCountdownBeep();
    for (let i = 1; i < COUNTDOWN_SEC; i++) {
      countdownTimersRef.current.push(setTimeout(() => {
        setCountdownRemaining(COUNTDOWN_SEC - i);
        playCountdownBeep();
      }, i * 1000));
    }
    countdownTimersRef.current.push(setTimeout(() => {
      setCountdownRemaining(0);
      playStartBeep();
      recordingRef.current = {
        startedAt: Date.now(),
        samples: [],
        keypoints: [],
        lastSampleAt: 0,
        screenshot: null,
      };
      setPhase("recording");
    }, COUNTDOWN_SEC * 1000));
  }

  function stopEarly() {
    finishTrial("stopped");
  }

  function reset() {
    for (const h of countdownTimersRef.current) clearTimeout(h);
    countdownTimersRef.current = [];
    recordingRef.current = null;
    setResult(null);
    setPhase("idle");
    setError(null);
    setCountdownRemaining(0);
  }

  // Done view ---------------------------------------------------------
  const isLiveDone   = phase === "done";
  const isUploadDone = uploadPhase === "done";
  if ((isLiveDone || isUploadDone) && result) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone ? () => { resetUpload(); } : reset;
    return (
      <div className="space-y-8">
        <PronatorDriftReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "pronator_drift",
            metrics: { result },
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
  const elapsedSec =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remainingSec = Math.max(0, TARGET_HOLD_DURATION_SEC - elapsedSec);

  const modeSwitchDisabled =
    phase === "countdown" || phase === "recording" || uploadPhase === "analyzing";

  return (
    <div className="space-y-10">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      {/* Mode toggle */}
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

      {/* 2D limitation banner — surfaced BEFORE the operator starts
          so they tell the patient the right thing. */}
      <div className="rounded-card border border-amber-500/40 bg-amber-500/5 p-4 text-sm">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
          <div className="space-y-1 text-foreground">
            <p className="font-medium">
              2D system — vertical drop only
            </p>
            <p className="text-xs text-muted">
              True clinical pronator drift also involves the forearm rotating
              (pronating) as it drops. This 2D camera CANNOT measure rotation —
              only the vertical drop. Use clinical judgement; this is a screen,
              not a diagnostic test.
            </p>
          </div>
        </div>
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
                "Patient stands or sits facing the camera, both arms extended forward at shoulder height (90° flexion), elbows straight, palms up.",
                "Camera at CHEST height, ~6 ft (2 m) away. Both extended arms must be in frame.",
                "Patient closes their eyes and holds the position for the full clip.",
                `Recording should be ${TARGET_HOLD_DURATION_SEC}-30 seconds of the hold (a brief settle at the start is fine — the engine skips the first 0.5 s).`,
                "Upload the single clip below.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <SinglePicker
            label="Pronator drift clip"
            hint={`Single clip of the ${TARGET_HOLD_DURATION_SEC}-30 s eyes-closed hold, frontal view.`}
            file={uploadFile}
            onPick={validateAndSetFile}
            progress={uploadProgress}
            busy={uploadPhase === "analyzing"}
            error={uploadError}
          />

          {filePickError && (
            <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">{filePickError}</p>
            </div>
          )}

          {uploadPhase === "idle" && (
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={analyzeUpload} disabled={!uploadFile}>
                <Upload className="h-4 w-4" />
                Analyse
              </Button>
            </div>
          )}

          {uploadPhase === "analyzing" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-accent" />
                <p className="text-sm text-foreground">
                  Uploading and analysing — about 15-30 seconds.
                </p>
              </div>
            </div>
          )}

          {uploadPhase === "error" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-start gap-3 rounded-md border border-error/40 bg-error/5 p-3 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                <p className="text-foreground">{uploadError ?? "Analysis failed."}</p>
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
        <div className="space-y-5">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Movement instructions
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Patient stands or sits facing the camera. Camera at CHEST height, ~2 m away.",
                "Both arms extended FORWARD at shoulder height (90° flexion), elbows straight, palms UP.",
                "Patient closes their eyes when the START beep sounds.",
                `Hold the position still for ${TARGET_HOLD_DURATION_SEC} seconds. An END beep marks completion.`,
                "Patient may open their eyes and lower the arms on the END beep.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          {phase === "idle" && (
            <div
              className={`rounded-card border p-4 text-sm ${
                armsTrackable
                  ? "border-emerald-500/30 bg-emerald-500/5"
                  : "border-amber-500/40 bg-amber-500/5"
              }`}
            >
              <div className="flex items-center gap-2">
                {armsTrackable ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                )}
                <span className="font-medium text-foreground">
                  {armsTrackable
                    ? "Both arms trackable"
                    : "Waiting for both wrists + shoulders to be visible…"}
                </span>
              </div>
              {!armsTrackable && (
                <p className="mt-1 text-xs text-muted">
                  Adjust framing so both extended wrists AND both shoulders sit
                  inside the frame simultaneously.
                </p>
              )}
            </div>
          )}

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Live status
            </p>

            {phase === "countdown" && (
              <div className="mt-3 space-y-3 text-center">
                <p className="text-sm text-muted">
                  Patient closes eyes when GO sounds. Get ready…
                </p>
                <p className="tabular text-6xl font-bold text-accent">
                  {countdownRemaining > 0 ? countdownRemaining : "GO"}
                </p>
              </div>
            )}

            {phase === "recording" && recordingRef.current && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground">
                    Recording — Pronator drift hold
                  </p>
                  <p className="tabular text-2xl font-semibold text-accent">
                    {remainingSec.toFixed(0)} s
                  </p>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full bg-accent transition-all"
                    style={{ width: `${(elapsedSec / TARGET_HOLD_DURATION_SEC) * 100}%` }}
                  />
                </div>
                <p className="rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
                  Patient: hold the position still, eyes closed. The end beep will
                  fire automatically.
                </p>
                <div className="flex justify-end">
                  <Button variant="ghost" size="sm" onClick={stopEarly}>
                    Stop early
                  </Button>
                </div>
              </div>
            )}

            {phase === "idle" && (
              <div className="mt-3 space-y-3">
                <p className="text-sm font-medium text-foreground">
                  Ready when the patient is in position with both arms extended.
                </p>
                <p className="text-xs text-muted">
                  Click Start for a {COUNTDOWN_SEC}-second audible countdown, then a START
                  beep marks the beginning of the {TARGET_HOLD_DURATION_SEC}-second hold.
                </p>
                <div className="flex gap-2">
                  <Button onClick={startRecording} disabled={!armsTrackable}>
                    <Play className="h-4 w-4" />
                    Start
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="lg:sticky lg:top-28">
          <PronatorDriftLiveCamera onFrame={handleFrame} onError={setError} />
          <p className="mt-3 text-xs text-subtle">
            Frontal view — both extended arms must be in frame. The skeleton
            overlay tracks both wrists and shoulders for drift measurement.
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

// ─── Single-file picker ──────────────────────────────────────────
function SinglePicker({
  label, hint, file, onPick, progress, busy, error,
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
          <p className="text-sm font-medium text-foreground">Choose video file</p>
          <p className="text-[11px] text-muted">MP4, WebM, MOV, or MKV</p>
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
