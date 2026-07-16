"use client";
// Functional Reach (C6) capture flow.
//
// Phase progression (live mode):
//   side_picker → calibration → armed → recording → done
//
// • side_picker  — operator selects which arm is being tested
// • calibration  — drop-in <HeightCalibrationStep>: doctor enters
//                  patient height (pre-filled from the patient
//                  record where available), system measures full-
//                  body pixel height while the patient stands
//                  straight in frame and derives pixels_per_cm.
//                  Skipping the step is allowed — the test still
//                  runs but in relative-units mode.
// • armed        — camera up, body trackability gate.
// • recording    — own camera + skeleton overlay, RECORDING_DURATION_SEC
//                  countdown, per-frame samples, peak screenshot at the
//                  best valid trial. Baseline locks automatically once
//                  the arm has been at shoulder height for ~1 s.
// • done         — renders <FunctionalReachReport> + Save-to-patient.

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
import { HeightCalibrationStep } from "@/components/calibration/HeightCalibrationStep";
import {
  MAX_HEIGHT_CM,
  MIN_HEIGHT_CM,
} from "@/lib/calibration/heightCalibration";
import { FunctionalReachLiveCamera } from "@/components/orthopedic/FunctionalReachLiveCamera";
import { FunctionalReachReport } from "@/components/orthopedic/FunctionalReachReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  RECORDING_DURATION_SEC,
  SAMPLE_INTERVAL_MS,
  analyzeFunctionalReachUpload,
  buildInterpretation,
  buildSample,
  isArmRaisedToShoulder,
  isTestSideTrackable,
  summarizeTrial,
  type CalibrationResult,
  type FrameSample,
  type FunctionalReachResult,
  type Side,
} from "@/lib/orthopedic/functionalReach";

type Mode = "live" | "upload";
type Phase =
  | "side_picker"
  | "calibration"
  | "armed"
  | "recording"
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
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  lastSampleAt: number;
  /** Running-max absolute wrist displacement seen so far. Updated
   *  whenever a new sample beats the previous maximum. */
  peakDisplacementPx: number;
  /** Screenshot captured at the moment `peakDisplacementPx` was last
   *  updated. By end-of-recording this is the screenshot of the
   *  true maximum-reach frame — NOT a frame grabbed at finish time
   *  (which would be the patient returning to baseline). */
  peakScreenshot: string | null;
}

export function FunctionalReachCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");
  const [phase, setPhase] = useState<Phase>("side_picker");
  const [error, setError] = useState<string | null>(null);
  const [side, setSide] = useState<Side | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [result, setResult] = useState<FunctionalReachResult | null>(null);
  const [now, setNow] = useState<number>(0);

  const [armTrackable, setArmTrackable] = useState<boolean>(false);
  const [armRaised, setArmRaised] = useState<boolean>(false);
  const [liveBaselineLocked, setLiveBaselineLocked] = useState<boolean>(false);
  const [livePeakPx, setLivePeakPx] = useState<number>(0);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  const recordingRef = useRef<RecordingState | null>(null);
  const baselineWristXRef = useRef<number | null>(null);
  const baselineHoldCountRef = useRef<number>(0);
  const BASELINE_HOLD_FRAMES_LIVE = 10; // ~1 s at ~10 Hz sampling

  // Ticking clock for the recording-window countdown.
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

  const finishRecording = useCallback((termination: "completed" | "stopped") => {
    const rec = recordingRef.current;
    if (!rec) return;

    // Prefer the screenshot stamped at the running-max moment
    // during recording (the true peak-reach frame). Only fall back
    // to a current-frame grab if no peak was ever recorded — which
    // can only happen when the patient never moved at all.
    let peakScreenshot: string | null = rec.peakScreenshot;
    if (!peakScreenshot) {
      const grab = (window as unknown as {
        __functionalReachCapture?: () => string | null;
      }).__functionalReachCapture;
      if (grab) peakScreenshot = grab();
    }

    const summary = summarizeTrial({
      side: rec.side,
      startedAtMs: rec.startedAt,
      endedAtMs: Date.now(),
      termination,
      samples: rec.samples,
      keypoints: rec.keypoints,
      calibration,
      peakScreenshotDataUrl: peakScreenshot,
    });

    setResult(summary);
    recordingRef.current = null;
    baselineWristXRef.current = null;
    baselineHoldCountRef.current = 0;
    setLiveBaselineLocked(false);
    setLivePeakPx(0);
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("done");
  }, [calibration]);

  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    if (!side) return;

    setArmTrackable(isTestSideTrackable(kp, side));
    setArmRaised(isArmRaisedToShoulder(kp, side));

    if (phase !== "recording" || !recordingRef.current) return;
    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    // Recording-window timeout safety net.
    const elapsedSec = (tNow - rec.startedAt) / 1000;
    if (elapsedSec >= RECORDING_DURATION_SEC) {
      finishRecording("completed");
      return;
    }

    const sample = buildSample(tNow - rec.startedAt, kp, rec.side);
    rec.samples.push(sample);
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Update live coaching + an approximate baseline + live peak
    // so the operator can see progress on screen. The authoritative
    // baseline + peaks are computed by summarizeTrial at the end.
    if (baselineWristXRef.current === null) {
      if (sample.arm_raised && sample.wrist_x_px !== null) {
        baselineHoldCountRef.current += 1;
        if (baselineHoldCountRef.current >= BASELINE_HOLD_FRAMES_LIVE) {
          // Lock baseline at the median wrist x over the last N hold frames.
          const tail = rec.samples
            .slice(-BASELINE_HOLD_FRAMES_LIVE)
            .map((s) => s.wrist_x_px)
            .filter((v): v is number => v !== null);
          if (tail.length > 0) {
            tail.sort((a, b) => a - b);
            baselineWristXRef.current = tail[Math.floor(tail.length / 2)];
            setLiveBaselineLocked(true);
          }
        }
      } else {
        baselineHoldCountRef.current = 0;
      }
    } else {
      if (sample.wrist_x_px !== null) {
        const disp = Math.abs(sample.wrist_x_px - baselineWristXRef.current);
        if (disp > rec.peakDisplacementPx) {
          // New running peak — stamp a screenshot of the CURRENT
          // frame as the candidate peak-reach screenshot. By
          // end-of-recording the most recent stamp is the screenshot
          // of the absolute max-reach frame. Replaces the old
          // behaviour of grabbing a single screenshot at finish time
          // (which was the post-reach return-to-baseline frame).
          rec.peakDisplacementPx = disp;
          const grab = (window as unknown as {
            __functionalReachCapture?: () => string | null;
          }).__functionalReachCapture;
          if (grab) {
            const url = grab();
            if (url) rec.peakScreenshot = url;
          }
          if (disp > livePeakPx) setLivePeakPx(disp);
        }
      }
    }

    // Coaching
    if (!armTrackable) {
      setCoachIfChanged(
        "Test-side body not fully visible — make sure the shoulder, wrist, hip, ankle and heel are in frame.",
      );
    } else if (baselineWristXRef.current === null) {
      if (!sample.arm_raised) {
        setCoachIfChanged(
          "Raise the test arm to shoulder height (90°), fist closed, and hold steady for ~1 second.",
        );
      } else {
        setCoachIfChanged(
          `Hold steady — locking baseline (${baselineHoldCountRef.current}/${BASELINE_HOLD_FRAMES_LIVE}).`,
        );
      }
    } else {
      setCoachIfChanged(
        "Baseline locked. Reach forward as far as comfortable; brief hold; return. Do this 3 times.",
      );
    }
  }, [side, phase, armTrackable, livePeakPx, setCoachIfChanged, finishRecording]);

  // ── Side picker ───────────────────────────────────────────────
  function pickSide(s: Side) {
    setSide(s);
    setError(null);
    setPhase("calibration");
  }

  const handleCalibrated = useCallback((cal: CalibrationResult | null) => {
    setCalibration(cal);
    setPhase("armed");
  }, []);

  // ── Recording controls ────────────────────────────────────────
  function startRecording() {
    if (!side) return;
    if (!armTrackable) {
      setError(
        "Test-side body landmarks aren't fully trackable yet — make sure the shoulder, " +
        "wrist, hip, ankle and heel are all in frame.",
      );
      return;
    }
    setError(null);
    recordingRef.current = {
      side,
      startedAt: Date.now(),
      samples: [],
      keypoints: [],
      lastSampleAt: 0,
      peakDisplacementPx: 0,
      peakScreenshot: null,
    };
    baselineWristXRef.current = null;
    baselineHoldCountRef.current = 0;
    setLiveBaselineLocked(false);
    setLivePeakPx(0);
    lastCoachRef.current = "";
    setCoachMsg(
      "Raise the test arm to shoulder height and hold for ~1 s, then start reaching.",
    );
    setPhase("recording");
  }

  function stopEarly() {
    finishRecording("stopped");
  }

  function reset() {
    recordingRef.current = null;
    baselineWristXRef.current = null;
    baselineHoldCountRef.current = 0;
    setLiveBaselineLocked(false);
    setLivePeakPx(0);
    setResult(null);
    setSide(null);
    setCalibration(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setError(null);
    setPhase("side_picker");
  }

  // ── Upload mode ──────────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadSide, setUploadSide] = useState<Side>("right");
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [filePickError, setFilePickError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<FunctionalReachResult | null>(null);

  // Patient height for upload-mode calibration. Pre-filled from the
  // patient record where available (doctor-flow launches), editable
  // otherwise. If empty/invalid the operator must explicitly opt
  // into "continue uncalibrated" before Analyse is enabled — keeps
  // accidental uncalibrated runs from silently slipping through.
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
  const [allowUncalibratedUpload, setAllowUncalibratedUpload] = useState<boolean>(false);
  const parsedUploadHeightCm = Number.parseFloat(uploadHeightInput);
  const uploadHeightCmValid =
    Number.isFinite(parsedUploadHeightCm) &&
    parsedUploadHeightCm >= MIN_HEIGHT_CM &&
    parsedUploadHeightCm <= MAX_HEIGHT_CM;

  function validateAndSetFile(f: File | null) {
    setFilePickError(null);
    if (!f) {
      setFile(null);
      return;
    }
    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      setFilePickError(
        `File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_MB} MB.`,
      );
      return;
    }
    if (f.type && !ACCEPTED_VIDEO_TYPES.includes(f.type)) {
      setFilePickError(
        `Unsupported file type (${f.type}). Use MP4, WebM, MOV, or MKV.`,
      );
      return;
    }
    setFile(f);
  }

  async function analyzeUpload() {
    if (!file) return;
    // Block accidental uncalibrated runs: the operator must either
    // enter a valid height or explicitly opt in to relative-units
    // mode via the checkbox below.
    if (!uploadHeightCmValid && !allowUncalibratedUpload) {
      setUploadError(
        `Enter the patient's height (${MIN_HEIGHT_CM}-${MAX_HEIGHT_CM} cm) or ` +
        `tick "Continue without calibration" to run in relative-units mode.`,
      );
      return;
    }
    setUploadPhase("analyzing");
    setUploadProgress(0);
    setUploadError(null);
    setError(null);
    try {
      // The backend measures body pixel height in the clip's standing
      // window and combines it with this height_cm to derive
      // pixels_per_cm — same math as the live-mode HeightCalibrationStep.
      // When the operator explicitly continued uncalibrated, send null
      // so the report renders the amber "relative units" banner.
      const heightCm = uploadHeightCmValid ? parsedUploadHeightCm : null;
      const r = await analyzeFunctionalReachUpload(
        file,
        uploadSide,
        null,
        heightCm,
        setUploadProgress,
      );
      setUploadResult(r);
      setUploadPhase("done");
    } catch (e) {
      setUploadError(errorMessage(e) ?? "Functional reach analysis failed.");
      setUploadPhase("error");
    }
  }

  function resetUpload() {
    setFile(null);
    setUploadProgress(0);
    setUploadError(null);
    setFilePickError(null);
    setUploadPhase("idle");
    setUploadResult(null);
    setError(null);
    setAllowUncalibratedUpload(false);
    // Re-prime the height input from the patient record on reset so
    // a back-to-back second test reuses it.
    setUploadHeightInput(
      patient?.height_cm && patient.height_cm > 0
        ? patient.height_cm.toFixed(0)
        : "",
    );
  }

  function switchMode(next: Mode) {
    if (next === mode) return;
    if (phase === "recording" || uploadPhase === "analyzing") return;
    recordingRef.current = null;
    baselineWristXRef.current = null;
    baselineHoldCountRef.current = 0;
    setResult(null);
    setSide(null);
    setCalibration(null);
    setLiveBaselineLocked(false);
    setLivePeakPx(0);
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("side_picker");
    setError(null);
    resetUpload();
    setMode(next);
  }

  // ── Done view ─────────────────────────────────────────────────
  const isLiveDone = phase === "done" && result !== null;
  const isUploadDone = uploadPhase === "done" && uploadResult !== null;
  if (isLiveDone || isUploadDone) {
    const r: FunctionalReachResult = isLiveDone
      ? (result as FunctionalReachResult)
      : (uploadResult as FunctionalReachResult);
    const interpretation = buildInterpretation(r);
    const onRunAgain = isUploadDone ? resetUpload : reset;
    return (
      <div className="space-y-8">
        <FunctionalReachReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={r}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "functional_reach",
            side: r.side_tested,
            metrics: { result: r },
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

  // ── Capture view ──────────────────────────────────────────────
  const elapsedSec =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remainingSec = Math.max(0, RECORDING_DURATION_SEC - elapsedSec);
  const modeSwitchDisabled = phase === "recording" || uploadPhase === "analyzing";

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

      {/* UPLOAD MODE */}
      {mode === "upload" && (
        <div className="space-y-6">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Setup checklist
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Enter the patient's height (cm) below — the backend will measure body pixel height in the clip's standing window and combine the two to derive scale.",
                "Patient starts standing straight, full body in frame for at least 1-2 seconds BEFORE the first reach.",
                "Patient stands SIDE-ON to the camera, near arm raised to ~90° (shoulder height), fist closed.",
                "Patient reaches forward as far as comfortable — without stepping or lifting the heels — three times in one clip.",
                "Camera at hip-to-shoulder height, ~2 m away, capturing the full body in lateral view.",
                "Total clip length: about 30 seconds.",
                "If height isn't entered, the report falls back to relative pixel units (no fall-risk classification).",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Side tested
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {(["left", "right"] as Side[]).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setUploadSide(s)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                    uploadSide === s
                      ? "bg-accent text-white"
                      : "bg-elevated text-muted hover:text-foreground"
                  }`}
                >
                  {s === "left" ? "Left arm" : "Right arm"}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs text-muted">
              The arm closest to the camera is the test side — the camera sits
              on that same side of the patient.
            </p>
          </div>

          {/* PATIENT HEIGHT FOR HEIGHT-BASED CALIBRATION */}
          <div
            className={`rounded-card border p-5 ${
              uploadHeightCmValid
                ? "border-emerald-500/30 bg-emerald-500/5"
                : "border-border bg-surface"
            }`}
          >
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Patient height (cm) — for scale calibration
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <input
                type="number"
                inputMode="decimal"
                min={MIN_HEIGHT_CM}
                max={MAX_HEIGHT_CM}
                step={0.5}
                value={uploadHeightInput}
                onChange={(e) => {
                  setUploadHeightInput(e.target.value);
                  // Typing a height implicitly cancels the "continue
                  // uncalibrated" opt-in.
                  if (allowUncalibratedUpload) setAllowUncalibratedUpload(false);
                }}
                placeholder={
                  patient?.height_cm && patient.height_cm > 0
                    ? patient.height_cm.toFixed(0)
                    : "e.g. 170"
                }
                disabled={uploadPhase === "analyzing"}
                className="w-32 rounded-md border border-border bg-background px-3 py-2 text-base tabular text-foreground outline-none focus:border-accent disabled:opacity-60"
                aria-label="Patient height in centimetres"
              />
              {uploadHeightCmValid ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2.5 py-1 text-xs font-medium text-emerald-700 dark:text-emerald-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Will calibrate (height)
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/15 px-2.5 py-1 text-xs font-medium text-amber-700 dark:text-amber-400">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  Uncalibrated unless height entered
                </span>
              )}
            </div>
            <p className="mt-2 text-xs text-muted">
              {patient?.height_cm && patient.height_cm > 0 ? (
                <>Pre-filled from patient record ({patient.height_cm.toFixed(0)} cm). </>
              ) : null}
              {uploadHeightCmValid
                ? "Reach will be reported in centimetres with fall-risk classification."
                : `Enter a height between ${MIN_HEIGHT_CM} and ${MAX_HEIGHT_CM} cm to enable height-based calibration.`}
            </p>

            {!uploadHeightCmValid && (
              <label className="mt-3 inline-flex cursor-pointer items-center gap-2 text-xs text-muted">
                <input
                  type="checkbox"
                  checked={allowUncalibratedUpload}
                  onChange={(e) => setAllowUncalibratedUpload(e.target.checked)}
                  disabled={uploadPhase === "analyzing"}
                  className="h-3.5 w-3.5 rounded border-border accent-accent"
                />
                Continue without calibration (relative units only — no fall-risk cutoff).
              </label>
            )}
          </div>

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Video file
            </p>

            {!file && (
              <label className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border border-dashed border-border bg-elevated p-6 text-center transition hover:border-accent/60">
                <FileVideo className="h-7 w-7 text-muted" />
                <p className="text-sm font-medium text-foreground">Choose video file</p>
                <p className="text-[11px] text-muted">MP4, WebM, MOV, or MKV</p>
                <input
                  type="file"
                  accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
                  className="hidden"
                  onChange={(e) => validateAndSetFile(e.target.files?.[0] ?? null)}
                  disabled={uploadPhase === "analyzing"}
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
                  {uploadPhase !== "analyzing" && (
                    <button
                      type="button"
                      onClick={() => validateAndSetFile(null)}
                      className="text-[11px] text-muted hover:text-error"
                    >
                      remove
                    </button>
                  )}
                </div>
                {uploadPhase === "analyzing" && (
                  <div className="space-y-1.5">
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                      <div
                        className="h-full bg-accent transition-all"
                        style={{ width: `${Math.min(100, Math.max(0, uploadProgress))}%` }}
                      />
                    </div>
                    <p className="inline-flex items-center gap-1.5 text-[11px] text-muted">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Analysing — {Math.round(uploadProgress)}%
                    </p>
                  </div>
                )}
                {uploadError && (
                  <p className="rounded-md border border-error/40 bg-error/5 px-2.5 py-2 text-[11px] text-foreground">
                    {uploadError}
                  </p>
                )}
              </div>
            )}
          </div>

          {filePickError && (
            <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">{filePickError}</p>
            </div>
          )}

          {uploadPhase === "idle" && (
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={analyzeUpload}
                disabled={!file || (!uploadHeightCmValid && !allowUncalibratedUpload)}
              >
                <Upload className="h-4 w-4" />
                Analyse video
                {uploadHeightCmValid
                  ? <span className="ml-1 text-[11px] opacity-80">(calibrated · {parsedUploadHeightCm.toFixed(0)} cm)</span>
                  : allowUncalibratedUpload
                    ? <span className="ml-1 text-[11px] opacity-80">(uncalibrated)</span>
                    : null}
              </Button>
              {!file && (
                <p className="text-xs text-muted">Choose a video file to enable analysis.</p>
              )}
              {file && !uploadHeightCmValid && !allowUncalibratedUpload && (
                <p className="text-xs text-amber-700 dark:text-amber-400">
                  Enter a valid height OR tick &quot;Continue without calibration&quot; above to enable Analyse.
                </p>
              )}
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

      {/* LIVE MODE */}
      {mode === "live" && (
        <>
          {/* SIDE PICKER */}
          {phase === "side_picker" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Pick the test side
              </p>
              <p className="mt-2 text-sm text-muted">
                The patient stands SIDE-ON to the camera. The arm closest to the
                camera is the test arm — pick that side. The camera sits on the
                same side as the test arm.
              </p>
              <div className="mt-4 flex flex-wrap gap-3">
                <Button onClick={() => pickSide("left")}>Left arm</Button>
                <Button onClick={() => pickSide("right")}>Right arm</Button>
              </div>
            </div>
          )}

          {/* CALIBRATION STEP */}
          {phase === "calibration" && side && (
            <div className="space-y-4">
              <div className="rounded-card border border-accent/30 bg-accent/5 p-4 text-sm text-foreground">
                <p className="font-medium">
                  Step 1 of 2 — calibrate scale ({side === "left" ? "Left" : "Right"} arm test)
                </p>
                <p className="mt-1 text-xs text-muted">
                  Take the patient&apos;s height reading BEFORE they raise the
                  arm or lean forward. Skipping is OK — distances will be
                  reported in relative units only.
                </p>
              </div>
              <HeightCalibrationStep
                defaultHeightCm={patient?.height_cm ?? null}
                onCalibrated={handleCalibrated}
                allowSkip
              />
            </div>
          )}

          {/* ARMED + RECORDING */}
          {(phase === "armed" || phase === "recording") && side && (
            <div className="space-y-4">
              <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-4 text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  <p className="font-medium text-foreground">
                    Step 2 of 2 — record the reach trial
                    {calibration ? (
                      <>
                        {" "}· calibrated at{" "}
                        <span className="tabular">{calibration.pixels_per_cm.toFixed(2)}</span>{" "}
                        px/cm
                      </>
                    ) : (
                      <> · uncalibrated (relative units only)</>
                    )}
                  </p>
                </div>
              </div>

              <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
                {/* LEFT — instructions + controls */}
                <div className="space-y-5">
                  <div className="rounded-card border border-border bg-surface p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                      Movement instructions
                    </p>
                    <ol className="mt-3 space-y-2.5 text-sm text-foreground">
                      {[
                        "Patient stands SIDE-ON to the camera, feet flat, shoulder-width apart.",
                        `Raise the ${side === "left" ? "left" : "right"} arm to ~90° (shoulder height), fist closed.`,
                        "Hold steady for ~1 s — the system locks the baseline (point A) automatically.",
                        "Reach forward as far as comfortable, briefly hold the peak, then return.",
                        "Repeat three times within the 30 s recording window.",
                        "Do NOT step or lift the heels — those trials are voided.",
                      ].map((s, i) => (
                        <li key={i} className="flex gap-2.5">
                          <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                          <span className="leading-relaxed">{s}</span>
                        </li>
                      ))}
                    </ol>
                  </div>

                  {/* Pre-record gate */}
                  {phase === "armed" && (
                    <div
                      className={`rounded-card border p-4 text-sm ${
                        armTrackable
                          ? "border-emerald-500/30 bg-emerald-500/5"
                          : "border-amber-500/40 bg-amber-500/5"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        {armTrackable ? (
                          <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-amber-600" />
                        )}
                        <span className="font-medium text-foreground">
                          {armTrackable
                            ? "Body fully trackable"
                            : "Waiting for full body visibility…"}
                        </span>
                      </div>
                      {!armTrackable && (
                        <p className="mt-1 text-xs text-muted">
                          Adjust the camera so the patient&apos;s shoulder, wrist, hip,
                          ankle and heel are all in the same frame.
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
                            Recording — {side === "left" ? "Left" : "Right"} arm reach
                          </p>
                          <p className="tabular text-2xl font-semibold text-accent">
                            {remainingSec.toFixed(0)} s
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 ${
                              armRaised
                                ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                                : "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                            }`}
                          >
                            Arm {armRaised ? "at shoulder ✓" : "below shoulder"}
                          </span>
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 ${
                              liveBaselineLocked
                                ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                                : "bg-elevated text-muted"
                            }`}
                          >
                            Baseline {liveBaselineLocked ? "locked ✓" : "pending"}
                          </span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                          <div
                            className="h-full bg-accent transition-all"
                            style={{ width: `${(elapsedSec / RECORDING_DURATION_SEC) * 100}%` }}
                          />
                        </div>
                        {liveBaselineLocked && (
                          <p className="tabular text-xs text-muted">
                            Live peak so far:{" "}
                            <span className="font-medium text-foreground">
                              {calibration
                                ? `${(livePeakPx / calibration.pixels_per_cm).toFixed(1)} cm`
                                : `${livePeakPx.toFixed(0)} px (relative)`}
                            </span>
                          </p>
                        )}
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

                    {phase === "armed" && (
                      <div className="mt-3 space-y-3">
                        <p className="text-sm text-foreground">
                          Ready to record the{" "}
                          <span className="font-medium">{side === "left" ? "Left" : "Right"}</span>{" "}
                          arm reach. The recording runs for up to{" "}
                          {RECORDING_DURATION_SEC} s.
                        </p>
                        {coachMsg && (
                          <p className="rounded-md bg-background/40 px-3 py-2 text-sm text-foreground">
                            {coachMsg}
                          </p>
                        )}
                        <div className="flex gap-2">
                          <Button onClick={startRecording} disabled={!armTrackable}>
                            <Play className="h-4 w-4" />
                            Start recording
                          </Button>
                          <Button variant="ghost" onClick={() => setPhase("side_picker")}>
                            Cancel
                          </Button>
                        </div>
                      </div>
                    )}

                    <p className="mt-4 text-xs text-muted">
                      {calibration
                        ? "Cutoffs: ≥ 25 cm = low fall risk · 15–25 cm = moderate · 10–15 cm = high · < 10 cm = very high."
                        : "Uncalibrated mode — distances reported in pixels only. No fall-risk classification."}
                    </p>
                  </div>
                </div>

                {/* RIGHT — sticky camera */}
                <div className="lg:sticky lg:top-28">
                  <FunctionalReachLiveCamera onFrame={handleFrame} onError={setError} />
                  <p className="mt-3 text-xs text-subtle">
                    Frame the patient side-on with the {side === "left" ? "left" : "right"} arm
                    nearest the camera. The skeleton tracks the wrist, shoulder,
                    hip, ankle, and heel of the test side.
                  </p>
                </div>
              </div>
            </div>
          )}

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
