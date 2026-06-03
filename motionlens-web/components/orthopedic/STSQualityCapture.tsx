"use client";
// Sit-to-Stand QUALITY (B4) capture flow.
//
// SINGLE trial — 3 reps total, no L/R split. Patient sits on a
// standard chair (~45 cm seat, no armrests), arms crossed at chest,
// and stands up + sits down 3 times at a comfortable pace. Camera
// is lateral on whichever side the operator chooses.
//
// State machine:
//   idle → configured (side + seat-height picked, leg trackable)
//        → recording  (rep detector running — fed `-hipY` so the
//                      "peak" trigger fires at the standing moment)
//        → done       (3 standing-moments captured OR timeout OR stop;
//                      sample stream is passed through summarizeTrial
//                      which extracts phase boundaries, computes
//                      smoothness, scans for hand-use)
//
// COMPLETELY SEPARATE from the existing 5x Sit-to-Stand (C2) capture
// at SitToStandCapture.tsx — different module, different state shape,
// different metrics. They share zero code.

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
import { STSQualityLiveCamera } from "@/components/orthopedic/STSQualityLiveCamera";
import { STSQualityReport } from "@/components/orthopedic/STSQualityReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  SAMPLE_INTERVAL_MS,
  TARGET_REP_COUNT,
  TRIAL_TIMEOUT_SEC,
  analyzeSTSQualityUpload,
  buildInterpretation,
  computeHipY,
  computeKneeAngle,
  computeLegLengthPx,
  computeShoulderY,
  computeTrunkLeanDeg,
  computeWristY,
  detectPeak,
  isTestSideTrackable,
  newPeakState,
  summarizeTrial,
  type FrameSample,
  type STSQualityResult,
  type Side,
} from "@/lib/orthopedic/stsQuality";

type Mode = "live" | "upload";
type Phase = "idle" | "configured" | "recording" | "done";
type UploadPhase = "idle" | "analyzing" | "done" | "error";

const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = [
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/x-matroska",
];

const DEFAULT_SEAT_HEIGHT_CM = 45;

function errorMessage(e: unknown): string | null {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  return null;
}

interface RecordingState {
  side: Side;
  seatHeightCm: number | null;
  startedAt: number;
  frameIdx: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  topOfStandIndices: number[];
  peakState: ReturnType<typeof newPeakState>;
  lastSampleAt: number;
  worstScreenshot: string | null;
}

export function STSQualityCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [cameraSide, setCameraSide] = useState<Side>("right");
  const [seatHeightCm, setSeatHeightCm] = useState<number>(DEFAULT_SEAT_HEIGHT_CM);
  const [result, setResult] = useState<STSQualityResult | null>(null);
  const [now, setNow] = useState<number>(0);

  // Live coaching
  const [legTrackable, setLegTrackable] = useState<boolean>(false);
  const [liveKneeDeg, setLiveKneeDeg] = useState<number | null>(null);
  const [liveTrunkDeg, setLiveTrunkDeg] = useState<number | null>(null);
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
    if (!rec.worstScreenshot) {
      const grab = (window as unknown as {
        __stsQualityCapture?: () => string | null;
      }).__stsQualityCapture;
      if (grab) {
        const url = grab();
        if (url) rec.worstScreenshot = url;
      }
    }
    const summary = summarizeTrial(
      rec.side,
      rec.seatHeightCm,
      rec.startedAt,
      Date.now(),
      termination,
      rec.samples,
      rec.topOfStandIndices,
      rec.keypoints,
      rec.worstScreenshot,
    );
    setResult(summary);
    recordingRef.current = null;
    setCoachMsg("");
    lastCoachRef.current = "";
    setLiveKneeDeg(null);
    setLiveTrunkDeg(null);
    setPhase("done");
  }, []);

  // Per-frame callback ------------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    // Pre-record gate evaluation uses whichever side is currently picked.
    setLegTrackable(isTestSideTrackable(kp, cameraSide));

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
    const hipY      = computeHipY(kp, rec.side);
    const kneeDeg   = computeKneeAngle(kp, rec.side);
    const trunkDeg  = computeTrunkLeanDeg(kp, rec.side);
    const wristY    = computeWristY(kp, rec.side);
    const shoulderY = computeShoulderY(kp, rec.side);
    const legLength = computeLegLengthPx(kp, rec.side);

    const sample: FrameSample = {
      t_ms: tNow - rec.startedAt,
      hip_y: hipY,
      knee_angle_deg: kneeDeg,
      trunk_lean_deg: trunkDeg,
      wrist_y: wristY,
      shoulder_y: shoulderY,
      leg_length_px: legLength,
    };
    rec.samples.push(sample);
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    setLiveKneeDeg(kneeDeg);
    setLiveTrunkDeg(trunkDeg);

    // Rep detection — peak detector is fed `-hipY` so the "peak"
    // trigger corresponds to the patient at standing height (the
    // valley in actual hipY in image y-down).
    const signal = hipY === null ? null : -hipY;
    const peaked = detectPeak(rec.peakState, signal, rec.frameIdx);
    if (peaked) {
      // Top-of-stand frame = previous sample (the standing moment).
      const topIndex = Math.max(0, rec.samples.length - 2);
      rec.topOfStandIndices.push(topIndex);

      // Grab a screenshot at the standing moment so the saved report
      // has a representative frame even if no rep ends up "worst".
      if (!rec.worstScreenshot) {
        const grab = (window as unknown as {
          __stsQualityCapture?: () => string | null;
        }).__stsQualityCapture;
        if (grab) {
          const url = grab();
          if (url) rec.worstScreenshot = url;
        }
      }

      const repsLeft = TARGET_REP_COUNT - rec.topOfStandIndices.length;
      if (repsLeft <= 0) {
        finishTrial("completed");
        return;
      }
      setCoachIfChanged(
        `Rep ${rec.topOfStandIndices.length} captured — ${repsLeft} more to go.`,
      );
      return;
    }

    // Coaching while between standing moments
    if (kneeDeg === null || hipY === null) {
      setCoachIfChanged(
        "Test-side leg not fully visible — keep the patient's hip, knee, and ankle in frame.",
      );
    } else if (rec.topOfStandIndices.length === 0) {
      setCoachIfChanged("Patient stands up at a comfortable pace.");
    } else {
      setCoachIfChanged(
        `Sit back down with control, then stand for rep ${rec.topOfStandIndices.length + 1}.`,
      );
    }
  }, [phase, cameraSide, finishTrial, setCoachIfChanged]);

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
      const r = await analyzeSTSQualityUpload(
        uploadFile, cameraSide, seatHeightCm || null, setUploadProgress,
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
    if (!legTrackable) {
      setError(
        "Test-side leg not yet trackable. Make sure the patient's hip, knee, " +
        "and ankle are clearly visible in frame before starting.",
      );
      return;
    }
    setError(null);
    recordingRef.current = {
      side: cameraSide,
      seatHeightCm,
      startedAt: Date.now(),
      frameIdx: 0,
      samples: [],
      keypoints: [],
      topOfStandIndices: [],
      peakState: newPeakState(),
      lastSampleAt: 0,
      worstScreenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg(
      `Patient stands up at a comfortable pace — ${TARGET_REP_COUNT} reps to capture.`,
    );
    setLiveKneeDeg(null);
    setLiveTrunkDeg(null);
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
    setLiveKneeDeg(null);
    setLiveTrunkDeg(null);
  }

  // Done view ---------------------------------------------------------
  const isLiveDone   = phase === "done";
  const isUploadDone = uploadPhase === "done";
  if ((isLiveDone || isUploadDone) && result) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone ? () => { resetUpload(); } : reset;
    return (
      <div className="space-y-8">
        <STSQualityReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "sts_quality",
            metrics: {
              result,
              chair_seat_height_cm: result.chair_seat_height_cm,
            },
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
  const repsCaptured = recordingRef.current?.topOfStandIndices.length ?? 0;
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

      {/* Shared session inputs — side + seat height */}
      <div className="rounded-card border border-border bg-surface p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Session setup
        </p>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <label className="block">
            <span className="text-xs text-muted">Camera-facing side</span>
            <select
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground"
              value={cameraSide}
              onChange={(e) => setCameraSide(e.target.value === "left" ? "left" : "right")}
              disabled={phase === "recording" || uploadPhase === "analyzing"}
            >
              <option value="right">Right side facing the camera</option>
              <option value="left">Left side facing the camera</option>
            </select>
          </label>
          <label className="block">
            <span className="text-xs text-muted">Chair seat height (cm)</span>
            <input
              type="number"
              min={30}
              max={70}
              step={1}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground"
              value={seatHeightCm}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (Number.isFinite(v)) setSeatHeightCm(v);
              }}
              disabled={phase === "recording" || uploadPhase === "analyzing"}
            />
          </label>
        </div>
        <p className="mt-2 text-[11px] text-muted">
          Standard chair = 45 cm. Seat height affects sit-to-stand mechanics — record it for context.
        </p>
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
                "Patient sits on a standard chair (no armrests), feet flat, arms crossed at the chest.",
                "Camera on the SIDE (lateral view), full body in frame, perpendicular to the patient.",
                "Patient stands up at a self-selected (comfortable) pace, then sits back down with control.",
                `Repeat ${TARGET_REP_COUNT} times. The system reports the median across the reps.`,
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
            label="Sit-to-stand quality clip"
            hint="One clip with the 3 reps. Camera on the chosen side."
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
        {/* LEFT — instructions + controls */}
        <div className="space-y-5">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Movement instructions
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Patient sits on a standard chair (no armrests), feet flat, arms CROSSED at the chest.",
                "Place the camera on the SIDE at hip height, ~2 m away. The full body should be in frame.",
                "Patient stands up at a comfortable pace, pauses briefly, then sits back down with control.",
                `Repeat ${TARGET_REP_COUNT} times. The trial auto-finishes after the 3rd stand.`,
                "Hands stay crossed throughout — pushing off the thighs is a compensation we will flag.",
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
                    ? `${cameraSide === "left" ? "Left" : "Right"}-side body fully trackable`
                    : "Waiting for shoulder + hip + knee + ankle visibility…"}
                </span>
              </div>
              {!legTrackable && (
                <p className="mt-1 text-xs text-muted">
                  Adjust the camera so the chosen-side shoulder, hip, knee, and ankle
                  are all in the same frame.
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
                    Recording — Sit-to-stand quality
                  </p>
                  <p className="tabular text-2xl font-semibold text-accent">
                    {repsCaptured} / {TARGET_REP_COUNT}
                  </p>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full bg-accent transition-all"
                    style={{ width: `${(repsCaptured / TARGET_REP_COUNT) * 100}%` }}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-md bg-elevated p-2">
                    <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">Knee</p>
                    <p className="tabular text-sm font-semibold text-foreground">
                      {liveKneeDeg !== null ? `${liveKneeDeg.toFixed(0)}°` : "—"}
                    </p>
                  </div>
                  <div className="rounded-md bg-elevated p-2">
                    <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">Trunk</p>
                    <p className="tabular text-sm font-semibold text-foreground">
                      {liveTrunkDeg !== null ? `${liveTrunkDeg.toFixed(0)}°` : "—"}
                    </p>
                  </div>
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
                  Ready when the patient is seated and arms crossed.
                </p>
                <p className="text-xs text-muted">
                  Camera-facing side: <span className="font-medium text-foreground">{cameraSide}</span>{" "}
                  · Seat height: <span className="font-medium text-foreground">{seatHeightCm} cm</span>.
                  Trial auto-stops after the {TARGET_REP_COUNT}rd stand or after {TRIAL_TIMEOUT_SEC} s.
                </p>
                <div className="flex gap-2">
                  <Button onClick={startRecording} disabled={!legTrackable}>
                    <Play className="h-4 w-4" />
                    Start
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT — sticky camera */}
        <div className="lg:sticky lg:top-28">
          <STSQualityLiveCamera onFrame={handleFrame} onError={setError} />
          <p className="mt-3 text-xs text-subtle">
            Start the camera and frame the patient from the side. The on-screen
            skeleton tracks the camera-side hip, knee, and trunk during each rep.
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

// ─── Single-file picker (no L/R split for B4) ────────────────────
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
