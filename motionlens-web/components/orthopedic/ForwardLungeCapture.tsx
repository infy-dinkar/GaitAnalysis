"use client";
// Forward Lunge (B3) capture flow.
//
// Patient steps forward into a lunge with the TEST leg, lowers until
// the back knee approaches the floor (front knee ≈ 90°), holds ~1 s,
// then pushes back to standing. Five reps per side. Camera is on the
// side of the front (test) leg, lateral view, full body in frame.
//
// State machine (per side):
//   armed → ready_to_record (leg trackable, waiting for click)
//         → recording      (rep detector running, target 5 reps)
//         → finished side  (auto on 5th rep OR timeout OR stop)
// Both sides done → done state with side-by-side report.
//
// The rep detector mirrors SLS's PeakState/detectPeak byte-for-byte
// (re-exported from lib/orthopedic/forwardLunge.ts) but is fed the
// TEST-side hip Y since in lateral view the contralateral hip is
// occluded.

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
import { ForwardLungeLiveCamera } from "@/components/orthopedic/ForwardLungeLiveCamera";
import { ForwardLungeReport } from "@/components/orthopedic/ForwardLungeReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  KNEE_TARGET_MAX_DEG,
  KNEE_TARGET_MIN_DEG,
  KOT_FLAG_RATIO,
  SAMPLE_INTERVAL_MS,
  TARGET_REP_COUNT,
  TRIAL_TIMEOUT_SEC,
  TRUNK_LEAN_FLAG_DEG,
  analyzeForwardLungeUpload,
  buildInterpretation,
  computeHipY,
  computeKneeAngle,
  computeKneeOverToeRatio,
  computeTrunkLeanDeg,
  detectPeak,
  isTestSideTrackable,
  newPeakState,
  summarizeSide,
  type ForwardLungeFullResult,
  type ForwardLungeSideResult,
  type FrameSample,
  type RepMetrics,
  type Side,
} from "@/lib/orthopedic/forwardLunge";

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
  frameIdx: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  reps: RepMetrics[];
  peakState: ReturnType<typeof newPeakState>;
  lastSampleAt: number;
  worstCompositeSoFar: number;
  worstScreenshot: string | null;
}

export function ForwardLungeCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  // Mode toggle — live (browser BlazePose WASM, per-side) vs upload
  // (backend MediaPipe, both sides sequentially).
  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [armedSide, setArmedSide] = useState<Side | null>(null);
  const [completedSides, setCompletedSides] = useState<Set<Side>>(new Set());
  const [result, setResult] = useState<ForwardLungeFullResult>({ left: null, right: null });
  const [now, setNow] = useState<number>(0);

  // Live coaching state
  const [legTrackable, setLegTrackable] = useState<boolean>(false);
  const [liveKneeDeg, setLiveKneeDeg] = useState<number | null>(null);
  const [liveKotRatio, setLiveKotRatio] = useState<number | null>(null);
  const [liveTrunkDeg, setLiveTrunkDeg] = useState<number | null>(null);
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

  const finishSide = useCallback((termination: ForwardLungeSideResult["termination"]) => {
    const rec = recordingRef.current;
    if (!rec) return;
    if (!rec.worstScreenshot) {
      const grab = (window as unknown as {
        __forwardLungeCapture?: () => string | null;
      }).__forwardLungeCapture;
      if (grab) {
        const url = grab();
        if (url) rec.worstScreenshot = url;
      }
    }
    const summary = summarizeSide(
      rec.side,
      rec.startedAt,
      Date.now(),
      termination,
      rec.reps,
      rec.samples,
      rec.keypoints,
      rec.worstScreenshot,
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
    setLiveKotRatio(null);
    setLiveTrunkDeg(null);

    setPhase(() => {
      const r = resultRef.current;
      const otherDone = rec.side === "left" ? r.right !== null : r.left !== null;
      return otherDone ? "done" : "idle";
    });
  }, []);

  // Per-frame callback ------------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    const sideForGate: Side | null =
      recordingRef.current?.side ?? armedSide ?? null;
    if (sideForGate) {
      setLegTrackable(isTestSideTrackable(kp, sideForGate));
    } else {
      setLegTrackable(
        isTestSideTrackable(kp, "left") || isTestSideTrackable(kp, "right"),
      );
    }

    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;
    rec.frameIdx += 1;

    // Trial-timeout safety net.
    const elapsedSec = (tNow - rec.startedAt) / 1000;
    if (elapsedSec >= TRIAL_TIMEOUT_SEC) {
      finishSide("timeout");
      return;
    }

    // Per-frame metrics
    const hipY      = computeHipY(kp, rec.side);
    const kneeDeg   = computeKneeAngle(kp, rec.side);
    const kotRatio  = computeKneeOverToeRatio(kp, rec.side);
    const trunkDeg  = computeTrunkLeanDeg(kp, rec.side);

    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      hip_y: hipY,
      knee_angle_deg: kneeDeg,
      knee_over_toe_ratio: kotRatio,
      trunk_lean_deg: trunkDeg,
    });
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    setLiveKneeDeg(kneeDeg);
    setLiveKotRatio(kotRatio);
    setLiveTrunkDeg(trunkDeg);

    // Rep detection on test-side hip Y trajectory.
    const peaked = detectPeak(rec.peakState, hipY, rec.frameIdx);
    if (peaked) {
      // Peak frame = previous sample. Pull metrics from there.
      const peakSample = rec.samples[rec.samples.length - 2] ?? rec.samples[rec.samples.length - 1];
      const newRep: RepMetrics = {
        rep_index: rec.reps.length + 1,
        t_ms: peakSample?.t_ms ?? tNow - rec.startedAt,
        knee_angle_at_bottom_deg: peakSample?.knee_angle_deg ?? null,
        knee_over_toe_ratio: peakSample?.knee_over_toe_ratio ?? null,
        trunk_lean_deg: peakSample?.trunk_lean_deg ?? null,
      };
      rec.reps.push(newRep);

      // Track worst-rep screenshot via the composite score so the
      // saved frame reflects whichever rep showed the most concerning
      // form (depth off-target / knee forward / trunk lean).
      const kneeDev = newRep.knee_angle_at_bottom_deg !== null
        ? Math.abs(newRep.knee_angle_at_bottom_deg - 90)
        : 0;
      const kotComp = newRep.knee_over_toe_ratio !== null
        ? Math.max(0, newRep.knee_over_toe_ratio) * 200
        : 0;
      const trunkComp = newRep.trunk_lean_deg ?? 0;
      const composite = kneeDev + kotComp + trunkComp;
      if (composite > rec.worstCompositeSoFar) {
        rec.worstCompositeSoFar = composite;
        const grab = (window as unknown as {
          __forwardLungeCapture?: () => string | null;
        }).__forwardLungeCapture;
        if (grab) {
          const url = grab();
          if (url) rec.worstScreenshot = url;
        }
      }

      const repsLeft = TARGET_REP_COUNT - rec.reps.length;
      if (repsLeft <= 0) {
        finishSide("completed");
        return;
      }
      setCoachIfChanged(
        `Rep ${newRep.rep_index} captured — ${repsLeft} more to go.`,
      );
      return;
    }

    // Coaching while between rep bottoms
    if (kneeDeg === null || hipY === null) {
      setCoachIfChanged(
        "Test-side leg not fully visible — keep the patient's hip, knee, and ankle in frame.",
      );
    } else if (rec.reps.length === 0) {
      setCoachIfChanged("Step forward into the first lunge.");
    } else {
      setCoachIfChanged(
        `Push back up to standing, then step into rep ${rec.reps.length + 1}.`,
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

    // Sequential, not parallel. Same cold-worker 502 mitigation as
    // SLR / AKE / MTT — the backend has only 2 gunicorn workers and
    // each loads its MediaPipe BlazePose model on the first request
    // post-deploy; serialising left → right keeps both requests warm.
    let leftResult:  ForwardLungeSideResult | null = null;
    let rightResult: ForwardLungeSideResult | null = null;
    let leftErr:  string | null = null;
    let rightErr: string | null = null;

    if (leftFile) {
      try {
        leftResult = await analyzeForwardLungeUpload(leftFile, "left", setLeftProgress);
      } catch (e) {
        leftErr = errorMessage(e) ?? "Left-side analysis failed.";
      }
    }
    if (rightFile) {
      try {
        rightResult = await analyzeForwardLungeUpload(rightFile, "right", setRightProgress);
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
      `Camera on the patient's ${side} side. ${side === "left" ? "Left" : "Right"} leg ` +
      `is the front (test) leg — they'll step forward with it. Start once the leg is ` +
      `fully visible.`,
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
      frameIdx: 0,
      samples: [],
      keypoints: [],
      reps: [],
      peakState: newPeakState(),
      lastSampleAt: 0,
      worstCompositeSoFar: 0,
      worstScreenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg(
      `Step forward into the first lunge with the ${armedSide} leg — five reps to capture.`,
    );
    setLiveKneeDeg(null);
    setLiveKotRatio(null);
    setLiveTrunkDeg(null);
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
    setLiveKotRatio(null);
    setLiveTrunkDeg(null);
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

        <ForwardLungeReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "forward_lunge",
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
  const repsCaptured = recordingRef.current?.reps.length ?? 0;
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
                "Patient stands tall, hands on hips or at the sides. Open space ahead so they can step forward.",
                "Camera is positioned to the SIDE (lateral view) on the side of the FRONT (test) leg.",
                `Record TWO clips — one for each side. In each, the patient performs ${TARGET_REP_COUNT} forward lunges on the test leg.`,
                "Each lunge should descend until the front knee is bent close to 90°, hold ~1 s, then return to standing.",
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
              label="Left-leg lunge"
              hint="LEFT leg is the front (test) leg. Camera on patient's LEFT side."
              file={leftFile}
              onPick={(f) => validateAndSetFile("left", f)}
              progress={leftProgress}
              busy={uploadPhase === "analyzing" && leftFile !== null}
              error={uploadErrors.left}
            />
            <SidePicker
              label="Right-leg lunge"
              hint="RIGHT leg is the front (test) leg. Camera on patient's RIGHT side."
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
                "Patient stands tall, hands on hips or at the sides. Camera ~2 m away on the test-leg side.",
                "Step forward into a lunge with the TEST leg. Front knee bends to ≈ 90°, back knee approaches the floor.",
                "Hold the bottom for ~1 s, then push back to standing.",
                `Perform ${TARGET_REP_COUNT} reps at a steady tempo — the trial auto-stops on the ${TARGET_REP_COUNT}th rep.`,
                "Repeat on the other side (swap which leg steps forward).",
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
                    : "Waiting for shoulder + hip + knee + ankle visibility…"}
                </span>
              </div>
              {!legTrackable && (
                <p className="mt-1 text-xs text-muted">
                  Position the camera so the patient's full body fits in frame, side-on
                  to the test leg.
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
                    Recording — {liveSide === "left" ? "Left" : "Right"}-leg lunge
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
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="rounded-md bg-elevated p-2">
                    <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">Knee</p>
                    <p className="tabular text-sm font-semibold text-foreground">
                      {liveKneeDeg !== null ? `${liveKneeDeg.toFixed(0)}°` : "—"}
                    </p>
                  </div>
                  <div className="rounded-md bg-elevated p-2">
                    <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">KOT</p>
                    <p className="tabular text-sm font-semibold text-foreground">
                      {liveKotRatio !== null ? `${(liveKotRatio * 100).toFixed(0)}%` : "—"}
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
                        {armedSide === "left" ? "Left" : "Right"}-leg lunge
                      </span>
                      .
                    </p>
                    <p className="text-xs text-muted">
                      The {armedSide} leg is the front leg (the one stepping forward).
                      Click <em>Start</em> once the leg is visible — {TARGET_REP_COUNT} reps
                      will be captured automatically.
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
                          {s === "left" ? "Left" : "Right"}-leg lunge
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
              Target: knee {KNEE_TARGET_MIN_DEG}–{KNEE_TARGET_MAX_DEG}° at bottom · KOT
              &lt; {(KOT_FLAG_RATIO * 100).toFixed(0)}% of leg length · trunk lean
              &lt; {TRUNK_LEAN_FLAG_DEG}°.
            </p>
          </div>
        </div>

        {/* RIGHT — sticky camera */}
        <div className="lg:sticky lg:top-28">
          <ForwardLungeLiveCamera onFrame={handleFrame} onError={setError} />
          <p className="mt-3 text-xs text-subtle">
            Start the camera and frame the patient from the side, perpendicular to the
            body. The on-screen skeleton tracks the front knee and trunk in real time.
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
