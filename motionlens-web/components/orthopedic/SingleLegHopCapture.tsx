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
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import {
  useRehabAutoFlow,
  type RehabAutoFlowPhase,
} from "@/lib/rehab/useAutoFlow";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  analyzeSingleLegHopUpload,
  buildCombinedResult,
  MAX_TRIALS,
  RECORDING_DURATION_SEC,
  type CalibrationResult,
  type Side,
  type SingleLegHopCombinedResult,
  type SingleLegHopResult,
} from "@/lib/orthopedic/singleLegHop";
import {
  SingleLegHopLiveAnalyzer,
  type SLHLiveFrame,
} from "@/lib/orthopedic/singleLegHopLive";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";

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

  // ── Auto-flow (fullscreen less-click live mode) ───────────────
  // After calibration, both legs' armed/recording phases render
  // inside one fullscreen LiveModeLayout. A 3-2-1 countdown runs
  // before EACH leg's recording: the hook's `started` input is
  // driven by `phase.startsWith("armed")`, so after the first leg's
  // result lands and phase flips to armed_<other> the input toggles
  // false→true (armed → recording/uploading → armed) and the
  // countdown re-seeds automatically.
  const [camActive, setCamActive] = useState<boolean>(false);

  const {
    phase: flowPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(
    mode === "live" &&
      (phase === "armed_left" || phase === "armed_right") &&
      camActive,
    () => {
      // The hook always calls the latest closure — `phase` here is
      // the armed_<side> value at countdown end.
      if (phase === "armed_left") startRecording("left");
      else if (phase === "armed_right") startRecording("right");
    },
  );

  // ── Live in-browser session (biomech-style — no video upload) ──
  // Each leg's hops are detected in the browser on the BlazePose-tfjs
  // keypoint stream; the per-leg SingleLegHopResult is built client-
  // side (see lib/orthopedic/singleLegHopLive). LSI is then computed
  // from both legs via buildCombinedResult().
  const analyzerRef = useRef<SingleLegHopLiveAnalyzer | null>(null);
  const recordingStartedAtRef = useRef<number>(0);
  const currentRecordingSideRef = useRef<Side | null>(null);
  const [hud, setHud] = useState<SLHLiveFrame | null>(null);
  const phaseRef = useRef<LivePhase>(phase);
  useEffect(() => { phaseRef.current = phase; }, [phase]);

  useEffect(() => {
    if (phase !== "recording_left" && phase !== "recording_right") return;
    const id = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [phase]);

  function startRecording(side: Side) {
    const analyzer = new SingleLegHopLiveAnalyzer(side);
    analyzer.reset(calibration);
    analyzerRef.current = analyzer;
    currentRecordingSideRef.current = side;
    setHud(null);
    recordingStartedAtRef.current = Date.now();
    setError(null);
    setPhase(side === "left" ? "recording_left" : "recording_right");
  }

  // End the current leg's live session, build its result, and advance
  // to the other leg (or the done view). Fired by the auto-stop timer,
  // after MAX_TRIALS valid hops, or the manual "Stop early" button.
  function finishLegSession() {
    const analyzer = analyzerRef.current;
    const side = currentRecordingSideRef.current;
    if (!analyzer || !side) return;
    const elapsed = Math.max(
      0, (Date.now() - recordingStartedAtRef.current) / 1000,
    );
    const result = analyzer.finalize(patient?.height_cm ?? null, elapsed);
    analyzerRef.current = null;
    currentRecordingSideRef.current = null;
    if (!result) {
      setError(
        analyzer.hasBaseline()
          ? `No hops detected on the ${side} leg. Stand still on the ${side} leg ~1 s, then hop forward and land on the same leg.`
          : `Could not lock a standing baseline — stand still on the ${side} leg, full body in frame, for ~1 s before hopping.`,
      );
      setPhase(side === "left" ? "armed_left" : "armed_right");
      return;
    }
    if (side === "left") setLeftResult(result);
    else setRightResult(result);
    const otherSide: Side = side === "left" ? "right" : "left";
    const otherDone =
      otherSide === "left" ? leftResult !== null : rightResult !== null;
    setPhase(
      otherDone ? "done" : otherSide === "left" ? "armed_left" : "armed_right",
    );
  }

  // Auto-stop when the session length elapses OR MAX_TRIALS valid hops
  // have been recorded.
  useEffect(() => {
    if (phase !== "recording_left" && phase !== "recording_right") return;
    const elapsedMs = now - recordingStartedAtRef.current;
    const analyzer = analyzerRef.current;
    if (analyzer && analyzer.validTrialCount() >= MAX_TRIALS) {
      finishLegSession();
      return;
    }
    if (elapsedMs >= RECORDING_DURATION_SEC * 1000) {
      finishLegSession();
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

  // Full reset back to the side picker. Also the fullscreen shell's
  // Exit handler.
  function reset() {
    analyzerRef.current = null;
    currentRecordingSideRef.current = null;
    setHud(null);
    setLeftResult(null);
    setRightResult(null);
    setCalibration(null);
    setFirstSide(null);
    setError(null);
    setCamActive(false);
    setPhase("side_picker");
  }

  // Frame handler — feeds the in-browser per-leg analyzer during a
  // live recording and mirrors its HUD state up for the overlay.
  const handleFrame = useCallback(
    (kp: Keypoint[], _video: HTMLVideoElement) => {
      const ph = phaseRef.current;
      if (ph !== "recording_left" && ph !== "recording_right") return;
      const analyzer = analyzerRef.current;
      if (!analyzer) return;
      const tMs = Date.now() - recordingStartedAtRef.current;
      setHud(analyzer.pushFrame(kp as unknown as LiveKeypoint[], tMs));
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
          stopRecording={finishLegSession}
          now={now}
          recordingStartedAt={recordingStartedAtRef.current}
          leftDone={leftResult !== null}
          rightDone={rightResult !== null}
          onResetSession={reset}
          error={error}
          hud={hud}
          camActive={camActive}
          onCamActiveChange={setCamActive}
          flowPhase={flowPhase}
          countdown={countdown}
          skipCountdown={skipCountdown}
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
  /** Exit the fullscreen shell / full reset to the side picker. */
  onResetSession: () => void;
  error: string | null;
  camActive: boolean;
  onCamActiveChange: (v: boolean) => void;
  flowPhase: RehabAutoFlowPhase | null;
  countdown: number | null;
  skipCountdown: () => void;
  hud: SLHLiveFrame | null;
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
    camActive,
    onCamActiveChange,
    flowPhase,
    countdown,
    skipCountdown,
    hud,
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
            autoConfirm
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

  const isArmed = phase === "armed_left" || phase === "armed_right";
  const isRecording =
    phase === "recording_left" || phase === "recording_right";
  const isUploading =
    phase === "uploading_left" || phase === "uploading_right";
  const elapsedSec = Math.max(0, (now - recordingStartedAt) / 1000);
  const remainingSec = Math.max(0, RECORDING_DURATION_SEC - elapsedSec);
  const legLabel = currentSide === "left" ? "Left leg" : "Right leg";
  const legOrdinal = leftDone || rightDone ? "second of 2" : "first of 2";

  // ── Fullscreen auto-flow shell: both legs' armed → recording →
  //    uploading phases live here; countdown re-runs per leg. ─────
  return (
    <LiveModeLayout
      title="Single-Leg Hop"
      subtitle={
        isRecording
          ? `${legLabel} (${legOrdinal}) — recording, ${remainingSec.toFixed(1)}s remaining`
          : isUploading
            ? `${legLabel} (${legOrdinal}) — analysing…`
            : `${legLabel} (${legOrdinal}) — ready`
      }
      onExit={onResetSession}
      camera={(
        <SingleLegHopLiveCamera
          onFrame={handleFrame}
          onError={() => {}}
          autoStart
          hideControls
          fill
          onActiveChange={onCamActiveChange}
        >
          {!camActive && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="rounded-full bg-black/60 px-4 py-2 text-sm text-white/80">
                Starting camera…
              </p>
            </div>
          )}
          {flowPhase === "countdown" && countdown !== null && (
            <AutoFlowCountdownOverlay
              countdown={countdown}
              label={`Recording ${currentSide} leg in`}
            />
          )}
          {isRecording && (
            <div className="absolute left-3 top-3 w-56 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
              <div className="flex items-center justify-between">
                <p className="text-[10px] uppercase tracking-[0.14em] text-rose-300">
                  ● Live · {currentSide} leg
                </p>
                <p className="tabular text-sm font-semibold text-white">
                  {remainingSec.toFixed(0)}s
                </p>
              </div>
              {!hud?.baselineLocked ? (
                <div className="mt-2">
                  <p className="text-[10px] text-amber-300">
                    {hud?.tracking === false
                      ? "Step fully into frame…"
                      : `Hold still on the ${currentSide} leg — locking baseline…`}
                  </p>
                  <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-white/15">
                    <div
                      className="h-full bg-amber-400 transition-all"
                      style={{ width: `${(hud?.baselineProgress ?? 0) * 100}%` }}
                    />
                  </div>
                </div>
              ) : (
                <>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="tabular text-3xl font-semibold text-white">
                      {hud.validTrials}
                    </span>
                    <span className="text-[11px] text-white/70">
                      / {hud.maxTrials} hops
                    </span>
                    {hud.airborne && (
                      <span className="ml-auto rounded-full bg-sky-500/25 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-sky-200">
                        airborne
                      </span>
                    )}
                  </div>
                  <div className="mt-1.5 grid grid-cols-2 gap-2 text-[10px]">
                    <div className="rounded bg-white/5 px-2 py-1">
                      <p className="uppercase tracking-[0.12em] text-white/50">Last</p>
                      <p className="tabular text-sm font-semibold text-white">
                        {hud.lastHopCm !== null
                          ? `${hud.lastHopCm.toFixed(0)} cm`
                          : hud.lastHopPx !== null
                            ? `${hud.lastHopPx.toFixed(0)} px`
                            : "—"}
                        {hud.lastValid === false && (
                          <span className="ml-1 text-[9px] text-rose-300">✗</span>
                        )}
                      </p>
                    </div>
                    <div className="rounded bg-white/5 px-2 py-1">
                      <p className="uppercase tracking-[0.12em] text-white/50">Best</p>
                      <p className="tabular text-sm font-semibold text-emerald-300">
                        {hud.bestValidCm !== null
                          ? `${hud.bestValidCm.toFixed(0)} cm`
                          : hud.bestValidPx !== null
                            ? `${hud.bestValidPx.toFixed(0)} px`
                            : "—"}
                      </p>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </SingleLegHopLiveCamera>
      )}
      sidebar={(
        <>
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-elevated px-3 py-1 text-xs font-medium uppercase tracking-wide text-subtle">
              {legLabel} · {leftDone || rightDone ? "Second of 2" : "First of 2"}
            </span>
            {calibration ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-400/40">
                <CheckCircle2 className="h-3 w-3" />
                Calibrated · {calibration.pixels_per_cm.toFixed(2)} px/cm
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
              <li>
                Stand still on the {currentSide} leg ~1 s to lock the
                baseline.
              </li>
              <li>
                Hop forward as far as you can and land on the SAME leg —
                the hops are detected live ({MAX_TRIALS} needed).
              </li>
              <li>
                Auto-saves after {MAX_TRIALS} valid hops or{" "}
                {RECORDING_DURATION_SEC}s; the other leg runs next.
              </li>
            </ol>
          </div>

          {flowPhase === "countdown" && countdown !== null && (
            <AutoFlowCountdownCard
              countdown={countdown}
              onSkip={skipCountdown}
              hint={`Patient standing on the ${currentSide} leg, full body in frame.`}
            />
          )}

          {isRecording && (
            <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-3 text-sm">
              <p className="font-medium text-foreground">
                ● Live · {currentSide} leg —{" "}
                {hud?.baselineLocked
                  ? `${hud.validTrials}/${hud.maxTrials} hops`
                  : "locking baseline…"}
              </p>
              <p className="mt-1 text-[11px] text-muted">
                {hud?.baselineLocked
                  ? `Hop forward and land on the ${currentSide} leg. ${remainingSec.toFixed(0)}s left.`
                  : `Stand still on the ${currentSide} leg, full body in frame.`}
              </p>
            </div>
          )}
          {isUploading && (
            <div className="rounded-card border border-blue-500/30 bg-blue-500/10 p-3 text-sm">
              <p className="flex items-center gap-2 font-medium text-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Analysing {currentSide} leg recording…
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
            {/* Manual fallback — only reachable if the auto-start
                failed (e.g. camera stream error at countdown end). */}
            {isArmed && flowPhase === "live" && (
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
            <Button variant="ghost" size="sm" onClick={onResetSession}>
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          </div>
        </>
      )}
    />
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
  const buildPayload = useCallback(
    () => ({
      module: "single_leg_hop" as const,
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
    }),
    [combined, interpretation],
  );

  return (
    <div className="space-y-10">
      {/* Auto-fire save for live and upload results — no-ops in public flow. */}
      <AutoSaveToast buildPayload={buildPayload} />

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
      </div>
    </div>
  );
}
