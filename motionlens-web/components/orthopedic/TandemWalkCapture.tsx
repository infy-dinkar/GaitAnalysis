"use client";
// Tandem Walk (E1) capture flow.
//
// SINGLE trial — 10 footstrikes total, no L/R split. Patient walks
// heel-to-toe along a taped line, toward the camera. Frontal view,
// full body in frame, eyes open.
//
// State machine:
//   idle → ready (patient trackable, waiting for click)
//        → recording (per-foot step detector running independently
//                     for left and right; each swing→planted
//                     transition fires a footstrike event)
//        → done (10 strikes captured OR timeout OR stop; full sample
//                stream passes through summarizeTrial which fits the
//                walking line, computes per-step deviation in cm,
//                counts arm-grabs and missteps)

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
import { TandemWalkLiveCamera } from "@/components/orthopedic/TandemWalkLiveCamera";
import { TandemWalkReport } from "@/components/orthopedic/TandemWalkReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  MIN_SWING_DISPLACEMENT_RATIO,
  PLANTED_VELOCITY_PX_PER_SEC,
  SAMPLE_HZ,
  SAMPLE_INTERVAL_MS,
  TARGET_STEP_COUNT,
  TRIAL_TIMEOUT_SEC,
  analyzeTandemWalkUpload,
  buildInterpretation,
  computeArmAbductionDeg,
  computeFootPos,
  computeHipMidX,
  computeHipMidY,
  computeShoulderMidX,
  computeShoulderWidthPx,
  detectFootstrike,
  isPatientTrackable,
  newFootState,
  summarizeTrial,
  type FootSide,
  type FrameSample,
  type TandemWalkResult,
} from "@/lib/orthopedic/tandemWalk";

type Mode = "live" | "upload";
type Phase = "idle" | "recording" | "done";
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
  frameIdx: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  rawStrikes: Array<{ side: FootSide; sample_index: number }>;
  leftState: ReturnType<typeof newFootState>;
  rightState: ReturnType<typeof newFootState>;
  lastSampleAt: number;
  screenshot: string | null;
}

export function TandemWalkCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TandemWalkResult | null>(null);
  const [now, setNow] = useState<number>(0);

  // Live coaching
  const [patientTrackable, setPatientTrackable] = useState<boolean>(false);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  const recordingRef = useRef<RecordingState | null>(null);

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

  const finishTrial = useCallback((termination: "completed" | "timeout" | "stopped") => {
    const rec = recordingRef.current;
    if (!rec) return;
    if (!rec.screenshot) {
      const grab = (window as unknown as {
        __tandemWalkCapture?: () => string | null;
      }).__tandemWalkCapture;
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
      rec.rawStrikes,
      rec.keypoints,
      rec.screenshot,
      patient?.age ?? null,
    );
    setResult(summary);
    recordingRef.current = null;
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("done");
  }, [patient]);

  // Per-frame callback ------------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    setPatientTrackable(isPatientTrackable(kp));

    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;
    rec.frameIdx += 1;

    const elapsedSec = (tNow - rec.startedAt) / 1000;
    if (elapsedSec >= TRIAL_TIMEOUT_SEC) {
      finishTrial("timeout");
      return;
    }

    // Per-frame metrics
    const hipMidX = computeHipMidX(kp);
    const hipMidY = computeHipMidY(kp);
    const shMidX  = computeShoulderMidX(kp);
    const shWidth = computeShoulderWidthPx(kp);
    const leftFoot  = computeFootPos(kp, "left");
    const rightFoot = computeFootPos(kp, "right");
    const lArmAbd = computeArmAbductionDeg(kp, "left");
    const rArmAbd = computeArmAbductionDeg(kp, "right");

    const sample: FrameSample = {
      t_ms: tNow - rec.startedAt,
      hip_mid_x: hipMidX,
      hip_mid_y: hipMidY,
      shoulder_mid_x: shMidX,
      shoulder_width_px: shWidth,
      left_foot_x:  leftFoot?.x  ?? null,
      left_foot_y:  leftFoot?.y  ?? null,
      right_foot_x: rightFoot?.x ?? null,
      right_foot_y: rightFoot?.y ?? null,
      left_arm_abduction_deg:  lArmAbd,
      right_arm_abduction_deg: rArmAbd,
    };
    rec.samples.push(sample);
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Per-foot step detector. Velocity threshold converts from per-sec
    // spec to per-frame at SAMPLE_HZ. Minimum swing displacement is a
    // fraction of the current shoulder width in pixels — falls back to
    // a small fixed value if shoulder width isn't tracked yet (rare,
    // only at the very first few frames before pose lock).
    const vThreshPerFrame = PLANTED_VELOCITY_PX_PER_SEC / SAMPLE_HZ;
    const minSwingDispPx = shWidth !== null
      ? shWidth * MIN_SWING_DISPLACEMENT_RATIO
      : 20;
    const sampleIdx = rec.samples.length - 1;

    const leftStrike = detectFootstrike(
      rec.leftState, leftFoot?.y ?? null, rec.frameIdx, vThreshPerFrame, minSwingDispPx,
    );
    if (leftStrike) {
      rec.rawStrikes.push({ side: "left", sample_index: sampleIdx });
      if (!rec.screenshot) {
        const grab = (window as unknown as {
          __tandemWalkCapture?: () => string | null;
        }).__tandemWalkCapture;
        if (grab) {
          const url = grab();
          if (url) rec.screenshot = url;
        }
      }
    }

    const rightStrike = detectFootstrike(
      rec.rightState, rightFoot?.y ?? null, rec.frameIdx, vThreshPerFrame, minSwingDispPx,
    );
    if (rightStrike) {
      rec.rawStrikes.push({ side: "right", sample_index: sampleIdx });
    }

    const stepsSoFar = rec.rawStrikes.length;
    if (stepsSoFar >= TARGET_STEP_COUNT) {
      finishTrial("completed");
      return;
    }

    // Coaching
    if (!isPatientTrackable(kp)) {
      setCoachIfChanged(
        "Patient not fully tracked — keep the full body in frame.",
      );
    } else if (stepsSoFar === 0) {
      setCoachIfChanged("Start walking heel-to-toe toward the camera.");
    } else {
      const left = TARGET_STEP_COUNT - stepsSoFar;
      setCoachIfChanged(
        `Step ${stepsSoFar} captured — ${left} more.`,
      );
    }
  }, [phase, finishTrial, setCoachIfChanged]);

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
      const r = await analyzeTandemWalkUpload(
        uploadFile, patient?.age ?? null, setUploadProgress,
      );
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
    if (phase === "recording" || uploadPhase === "analyzing") return;
    recordingRef.current = null;
    setResult(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    setError(null);
    resetUpload();
    setMode(next);
  }

  function startRecording() {
    if (!patientTrackable) {
      setError(
        "Patient not yet trackable. Make sure the full body is in frame " +
        "before starting.",
      );
      return;
    }
    setError(null);
    recordingRef.current = {
      startedAt: Date.now(),
      frameIdx: 0,
      samples: [],
      keypoints: [],
      rawStrikes: [],
      leftState: newFootState(),
      rightState: newFootState(),
      lastSampleAt: 0,
      screenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg("Start walking heel-to-toe toward the camera.");
    setPhase("recording");
  }

  function stopEarly() {
    finishTrial("stopped");
  }

  function reset() {
    recordingRef.current = null;
    setResult(null);
    setPhase("idle");
    setError(null);
    setCoachMsg("");
    lastCoachRef.current = "";
  }

  // Done view ---------------------------------------------------------
  const isLiveDone   = phase === "done";
  const isUploadDone = uploadPhase === "done";
  if ((isLiveDone || isUploadDone) && result) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone ? () => { resetUpload(); } : reset;
    return (
      <div className="space-y-8">
        <TandemWalkReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "tandem_walk",
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
  const stepsCaptured = recordingRef.current?.rawStrikes.length ?? 0;
  const elapsedSec =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remainingSec = Math.max(0, TRIAL_TIMEOUT_SEC - elapsedSec);

  const modeSwitchDisabled =
    phase === "recording" || uploadPhase === "analyzing";

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

      {/* ─── UPLOAD MODE ────────────────────────────────────────── */}
      {mode === "upload" && (
        <div className="space-y-6">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Setup checklist
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Mark a straight line on the floor (tape) about 3 m long.",
                "Camera at the END of the line, framing the full body. Patient walks TOWARD it.",
                "Patient stands at the far end, eyes open, hands relaxed at the sides.",
                `Patient walks heel-to-toe along the line, one foot in front of the other, for ${TARGET_STEP_COUNT} steps.`,
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
            label="Tandem walk clip"
            hint={`Single clip of the ${TARGET_STEP_COUNT}-step heel-to-toe walk, frontal view.`}
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
                "Mark a straight line on the floor with tape (about 3 m long).",
                "Camera at the END of the line, at hip height. Patient stands at the FAR end.",
                "Eyes open, hands relaxed at the sides. No socks (bare feet or grippy shoes).",
                `Patient walks heel-to-toe along the line — each new step plants the advancing heel touching the previous foot's toe. ${TARGET_STEP_COUNT} steps total.`,
                "Trial auto-finishes after the 10th step, or stop early if the patient cannot continue.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          {phase !== "recording" && (
            <div
              className={`rounded-card border p-4 text-sm ${
                patientTrackable
                  ? "border-emerald-500/30 bg-emerald-500/5"
                  : "border-amber-500/40 bg-amber-500/5"
              }`}
            >
              <div className="flex items-center gap-2">
                {patientTrackable ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                )}
                <span className="font-medium text-foreground">
                  {patientTrackable
                    ? "Patient fully trackable"
                    : "Waiting for full body visibility…"}
                </span>
              </div>
              {!patientTrackable && (
                <p className="mt-1 text-xs text-muted">
                  Adjust the camera so both shoulders, both hips, and both ankles
                  are in the same frame.
                </p>
              )}
            </div>
          )}

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Live status
            </p>

            {phase === "recording" && recordingRef.current && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground">
                    Recording — Tandem walk
                  </p>
                  <p className="tabular text-2xl font-semibold text-accent">
                    {stepsCaptured} / {TARGET_STEP_COUNT}
                  </p>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full bg-accent transition-all"
                    style={{ width: `${(stepsCaptured / TARGET_STEP_COUNT) * 100}%` }}
                  />
                </div>
                <p className="text-xs text-muted">
                  {remainingSec.toFixed(0)} s remaining before timeout.
                </p>
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

            {phase !== "recording" && (
              <div className="mt-3 space-y-3">
                <p className="text-sm font-medium text-foreground">
                  Ready when the patient is at the far end of the line.
                </p>
                <p className="text-xs text-muted">
                  Trial auto-stops after the {TARGET_STEP_COUNT}th footstrike or
                  after {TRIAL_TIMEOUT_SEC} s.
                </p>
                <div className="flex gap-2">
                  <Button onClick={startRecording} disabled={!patientTrackable}>
                    <Play className="h-4 w-4" />
                    Start
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="lg:sticky lg:top-28">
          <TandemWalkLiveCamera onFrame={handleFrame} onError={setError} />
          <p className="mt-3 text-xs text-subtle">
            Frontal view — the patient should walk DIRECTLY toward the camera.
            The on-screen skeleton tracks both ankles, both wrists, and the trunk.
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
