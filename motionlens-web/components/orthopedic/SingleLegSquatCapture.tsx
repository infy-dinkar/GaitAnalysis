"use client";
// Single-Leg Squat capture flow.
//
// State machine (per side):
//   armed → ready_to_record (camera square, waiting for click)
//         → recording      (rep detector running, target 5 reps)
//         → finished side  (auto on 5th rep OR 30 s timeout OR stop)
// Both sides done → done state with side-by-side report.
//
// Pre-record gate: continuously evaluate camera-squareness from the
// shoulder line. Start button is disabled while shoulders aren't level
// (within ±5° per spec). Live coaching banner explains what to fix.

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
import { SingleLegSquatLiveCamera } from "@/components/orthopedic/SingleLegSquatLiveCamera";
import { SingleLegSquatReport } from "@/components/orthopedic/SingleLegSquatReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  PELVIC_DROP_THRESHOLD_DEG,
  SAMPLE_INTERVAL_MS,
  SQUARENESS_TOLERANCE_DEG,
  TARGET_REP_COUNT,
  TRIAL_TIMEOUT_SEC,
  TRUNK_LEAN_THRESHOLD_DEG,
  analyzeSingleLegSquatUpload,
  buildInterpretation,
  computeHipMidY,
  computeKFPPA,
  computeLegLengthPx,
  computePelvicTilt,
  computeShoulderHorizontalDeg,
  computeTrunkLean,
  detectPeak,
  isCameraSquare,
  newPeakState,
  summarizeSide,
  type FrameSample,
  type RepMetrics,
  type Side,
  type SingleLegSquatFullResult,
  type SingleLegSquatSideResult,
} from "@/lib/orthopedic/singleLegSquat";

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
  legLengthPx: number | null;     // captured at start, used to normalise depth
  baselineHipY: number | null;    // captured at start, for depth subtraction
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  reps: RepMetrics[];
  peakState: ReturnType<typeof newPeakState>;
  lastSampleAt: number;
  worstKFPPASoFar: number;
  worstScreenshot: string | null;
}

export function SingleLegSquatCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  // Mode toggle — live (browser BlazePose WASM, per-side) vs upload
  // (backend MediaPipe, both sides in parallel). Both modes converge
  // on the same SingleLegSquatFullResult + SingleLegSquatReport.
  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [armedSide, setArmedSide] = useState<Side | null>(null);
  const [completedSides, setCompletedSides] = useState<Set<Side>>(new Set());
  const [result, setResult] = useState<SingleLegSquatFullResult>({ left: null, right: null });
  // re-render driver — for the rep counter / countdown
  const [now, setNow] = useState<number>(0);

  // Live shoulder-horizontal angle, updated ~10 Hz so the squareness
  // gate UI re-renders smoothly. Null when both shoulders aren't yet
  // visible.
  const [shoulderAngle, setShoulderAngle] = useState<number | null>(null);
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

  const finishSide = useCallback((termination: SingleLegSquatSideResult["termination"]) => {
    const rec = recordingRef.current;
    if (!rec) return;
    // Fallback screenshot: if no worst-rep frame was captured during
    // the trial (e.g. trial was stopped early or no reps were
    // detected), grab the current frame so the saved report still
    // has a visual reference for the doctor.
    if (!rec.worstScreenshot) {
      const grab = (window as unknown as {
        __singleLegSquatCapture?: () => string | null;
      }).__singleLegSquatCapture;
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

    // If both done, render report; else go back to idle for the next side.
    setPhase(() => {
      const r = resultRef.current;
      const otherDone = rec.side === "left" ? r.right !== null : r.left !== null;
      return otherDone ? "done" : "idle";
    });
  }, []);

  // Per-frame callback ------------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    // Always evaluate camera squareness so the pre-record gate updates
    // even before the user has armed a side.
    const shoulder = computeShoulderHorizontalDeg(kp);
    setShoulderAngle(shoulder);

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

    // Continuous squareness flag during the trial — non-fatal warning.
    if (shoulder !== null && !isCameraSquare(shoulder)) {
      setCoachIfChanged(
        `Patient rotated ${Math.abs(shoulder).toFixed(1)}° — try to face the camera squarely (KFPPA accuracy degrades).`,
      );
    } else if (rec.reps.length === 0) {
      setCoachIfChanged("Begin squatting on the test leg — five reps to capture.");
    }

    // Per-frame metrics
    const hipMidY = computeHipMidY(kp);
    const kfppa = computeKFPPA(kp, rec.side);
    const pelvicDrop = computePelvicTilt(kp);
    const trunkLean = computeTrunkLean(kp);

    // Capture leg length once — first valid frame after start.
    if (rec.legLengthPx === null) {
      rec.legLengthPx = computeLegLengthPx(kp, rec.side);
    }
    if (rec.baselineHipY === null && hipMidY !== null) {
      rec.baselineHipY = hipMidY;
    }

    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      hip_mid_y: hipMidY,
      kfppa_deg: kfppa,
      pelvic_drop_deg: pelvicDrop,
      trunk_lean_deg: trunkLean,
    });
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Track running worst KFPPA + grab a screenshot at that moment.
    if (kfppa !== null && kfppa > rec.worstKFPPASoFar) {
      rec.worstKFPPASoFar = kfppa;
      const grab = (window as unknown as {
        __singleLegSquatCapture?: () => string | null;
      }).__singleLegSquatCapture;
      if (grab) {
        const url = grab();
        if (url) rec.worstScreenshot = url;
      }
    }

    // Rep detection on hip-midpoint Y trajectory.
    const peaked = detectPeak(rec.peakState, hipMidY, rec.frameIdx);
    if (peaked) {
      // The peak frame was the PREVIOUS sample. Pull metrics from there
      // to commit the rep.
      const peakSample = rec.samples[rec.samples.length - 2] ?? rec.samples[rec.samples.length - 1];
      const depthPct =
        peakSample && rec.baselineHipY !== null && rec.legLengthPx
          ? ((peakSample.hip_mid_y ?? rec.baselineHipY) - rec.baselineHipY) / rec.legLengthPx * 100
          : null;
      const newRep: RepMetrics = {
        rep_index: rec.reps.length + 1,
        t_ms: peakSample?.t_ms ?? tNow - rec.startedAt,
        kfppa_deg: peakSample?.kfppa_deg ?? null,
        pelvic_drop_deg: peakSample?.pelvic_drop_deg ?? null,
        trunk_lean_deg: peakSample?.trunk_lean_deg ?? null,
        depth_pct: depthPct,
      };
      rec.reps.push(newRep);
      const repsLeft = TARGET_REP_COUNT - rec.reps.length;
      if (repsLeft <= 0) {
        finishSide("completed");
        return;
      }
      setCoachIfChanged(
        `Rep ${newRep.rep_index} captured — ${repsLeft} more to go.`,
      );
    }
  }, [phase, finishSide, setCoachIfChanged]);

  // ── Upload-mode state ──────────────────────────────────────────
  // Per-side errors are tracked independently so one bad clip
  // doesn't lose the other side's result. Mirrors TrendelenburgCapture.
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

  // ── Upload-mode handlers ───────────────────────────────────────
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

    const age = patient?.age ?? null;
    const tasks: Array<Promise<SingleLegSquatSideResult | null>> = [
      leftFile
        ? analyzeSingleLegSquatUpload(leftFile, "left", age, setLeftProgress)
        : Promise.resolve(null),
      rightFile
        ? analyzeSingleLegSquatUpload(rightFile, "right", age, setRightProgress)
        : Promise.resolve(null),
    ];
    const [leftSettled, rightSettled] = await Promise.allSettled(tasks);

    let leftResult:  SingleLegSquatSideResult | null = null;
    let rightResult: SingleLegSquatSideResult | null = null;
    let leftErr:  string | null = null;
    let rightErr: string | null = null;

    if (leftFile) {
      if (leftSettled.status === "fulfilled") {
        leftResult = leftSettled.value;
      } else {
        leftErr = errorMessage(leftSettled.reason) ?? "Left-side analysis failed.";
      }
    }
    if (rightFile) {
      if (rightSettled.status === "fulfilled") {
        rightResult = rightSettled.value;
      } else {
        rightErr = errorMessage(rightSettled.reason) ?? "Right-side analysis failed.";
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
    // Block mode switching while either flow is mid-operation.
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
      `Stand on the ${side === "left" ? "left" : "right"} leg, lift the other knee. ` +
      `Face the camera squarely — start when shoulders are level.`,
    );
  }

  function startRecording() {
    if (!armedSide) return;
    if (!isCameraSquare(shoulderAngle)) {
      setError("Patient is rotated. Face the camera squarely before starting.");
      return;
    }
    setError(null);
    recordingRef.current = {
      side: armedSide,
      startedAt: Date.now(),
      frameIdx: 0,
      legLengthPx: null,
      baselineHipY: null,
      samples: [],
      keypoints: [],
      reps: [],
      peakState: newPeakState(),
      lastSampleAt: 0,
      worstKFPPASoFar: 0,
      worstScreenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg(`Begin squatting on the ${armedSide} leg — five reps to capture.`);
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
  }

  // Done view ---------------------------------------------------------
  // Live mode: triggered when both sides have been recorded.
  // Upload mode: triggered when at least one side analysis succeeded.
  const isLiveDone = phase === "done";
  const isUploadDone = uploadPhase === "done";
  if (isLiveDone || isUploadDone) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone ? () => { resetUpload(); } : reset;
    return (
      <div className="space-y-8">
        {/* Per-side upload errors — surface alongside the report so
            the operator can see which side failed and re-record it
            without losing the side that succeeded. */}
        {isUploadDone && (uploadErrors.left || uploadErrors.right) && (
          <div className="space-y-2">
            {uploadErrors.left && (
              <div className="flex items-start gap-3 rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
                <p className="text-foreground">
                  <span className="font-medium">Left-stance video: </span>
                  {uploadErrors.left}
                </p>
              </div>
            )}
            {uploadErrors.right && (
              <div className="flex items-start gap-3 rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
                <p className="text-foreground">
                  <span className="font-medium">Right-stance video: </span>
                  {uploadErrors.right}
                </p>
              </div>
            )}
          </div>
        )}

        <SingleLegSquatReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "single_leg_squat",
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
  const cameraSquare = isCameraSquare(shoulderAngle);

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
                "Patient stands facing the camera squarely — both shoulders level in frame.",
                "Record TWO clips — one for left-leg squats, one for right-leg squats.",
                `Each clip should show the patient performing ${TARGET_REP_COUNT} single-leg squats on the test leg at a steady tempo.`,
                "Both clips are uploaded together and analysed in parallel.",
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
              label="Left-leg squat"
              hint="Patient stands on the RIGHT leg, lifts the LEFT."
              file={leftFile}
              onPick={(f) => validateAndSetFile("left", f)}
              progress={leftProgress}
              busy={uploadPhase === "analyzing" && leftFile !== null}
              error={uploadErrors.left}
            />
            <SidePicker
              label="Right-leg squat"
              hint="Patient stands on the LEFT leg, lifts the RIGHT."
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
                  Only one side selected — the other side will be missing
                  from the report. For a complete bilateral comparison,
                  upload both clips.
                </p>
              )}
            </div>
          )}

          {uploadPhase === "analyzing" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-accent" />
                <p className="text-sm text-foreground">
                  Uploading and analysing — this can take 10-30 seconds per side.
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

      {/* ─── LIVE MODE (unchanged behaviour) ─────────────────────── */}
      {mode === "live" && (
      <>
      {/* ─── 2-column layout (instructions+status | camera) ─────── */}
      <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
        {/* LEFT — instructions + controls */}
        <div className="space-y-5">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Movement instructions
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Stand facing the camera squarely — both shoulders level in frame.",
                "Lift the non-test leg by bending the knee out to the side. Stand on the test leg only.",
                "Squat down on the test leg to comfortable depth (thigh approaching horizontal).",
                "Move smoothly — about 2 seconds down, 2 seconds up. Don't let the knee collapse inward.",
                `Perform ${TARGET_REP_COUNT} squats in a row at this steady tempo.`,
                "After completing the set, repeat on the other side.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Camera-squareness gate */}
          {phase !== "recording" && (
            <div
              className={`rounded-card border p-4 text-sm ${
                cameraSquare
                  ? "border-emerald-500/30 bg-emerald-500/5"
                  : "border-amber-500/40 bg-amber-500/5"
              }`}
            >
              <div className="flex items-center gap-2">
                {cameraSquare ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                )}
                <span className="font-medium text-foreground">
                  Camera-squareness:{" "}
                  {shoulderAngle === null
                    ? "waiting for shoulders…"
                    : cameraSquare
                      ? `square (${Math.abs(shoulderAngle).toFixed(1)}° tilt)`
                      : `tilted ${Math.abs(shoulderAngle).toFixed(1)}° — must be ≤ ${SQUARENESS_TOLERANCE_DEG}°`}
                </span>
              </div>
              {!cameraSquare && (
                <p className="mt-1 text-xs text-muted">
                  Patient&apos;s shoulders should sit level relative to the camera. Adjust the camera height
                  or have the patient face the lens directly before starting.
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
                    Recording — {liveSide === "left" ? "Left" : "Right"}-leg squat
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
                <p className="text-xs text-muted">
                  {remainingSec.toFixed(0)} s remaining before timeout. Squat to comfortable depth
                  (thigh approaching horizontal). Trial auto-stops on the {TARGET_REP_COUNT}th rep
                  or after {TRIAL_TIMEOUT_SEC} s.
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
                        {armedSide === "left" ? "Left" : "Right"}-leg squat
                      </span>
                      .
                    </p>
                    <p className="text-xs text-muted">
                      Patient stands on the {armedSide} leg with the
                      contralateral knee lifted. Click <em>Start</em> when shoulders
                      are level — {TARGET_REP_COUNT} squats will be captured.
                    </p>
                    {coachMsg && (
                      <p className="rounded-md bg-background/40 px-3 py-2 text-sm text-foreground">
                        {coachMsg}
                      </p>
                    )}
                    <div className="flex gap-2">
                      <Button onClick={startRecording} disabled={!cameraSquare}>
                        <Play className="h-4 w-4" />
                        Start ({armedSide})
                      </Button>
                      <Button variant="ghost" onClick={() => setArmedSide(null)}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-foreground">Choose which side to record next:</p>
                    <div className="flex flex-wrap gap-3">
                      {sidesRemaining.map((s) => (
                        <Button key={s} onClick={() => arm(s)}>
                          {s === "left" ? "Left" : "Right"}-leg squat
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
              Cutoffs (PDF Test B1): KFPPA &lt;10° good, 10–15° borderline, &gt;15° valgus.
              Pelvic drop &gt;{PELVIC_DROP_THRESHOLD_DEG}° = hip abductor insufficiency.
              Trunk lateral lean &gt;{TRUNK_LEAN_THRESHOLD_DEG}° = compensatory pattern.
            </p>
          </div>
        </div>

        {/* RIGHT — sticky camera */}
        <div className="lg:sticky lg:top-28">
          <SingleLegSquatLiveCamera onFrame={handleFrame} onError={setError} />
          <p className="mt-3 text-xs text-subtle">
            Start the camera and have the patient stand facing the lens.
            The on-screen skeleton tracks the test leg's knee, hip, and
            trunk in real time — keep both shoulders inside the frame.
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
