"use client";
// Modified Thomas Test (MTT) capture flow.
//
// Patient on the edge of a table, supine, with one knee held to chest
// and the OTHER (test) leg hanging off the edge. Camera on the test
// side, lateral view, tall frame from shoulder down to ankle.
//
// State machine (per side):
//   armed → ready_to_record (test-side leg trackable, waiting for click)
//         → recording      (trial running — sampling at SAMPLE_HZ,
//                           computing hip + knee angles every frame,
//                           checking the rolling stability window;
//                           auto-finishes the moment the window goes
//                           stable, or on TRIAL_DURATION_SEC timeout
//                           or on manual Stop)
//         → finished side  (captured angles latched into result)
// Both sides done → done state with side-by-side report.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  FileVideo,
  Loader2,
  RotateCcw,
  Upload,
  Video,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { ModifiedThomasLiveCamera } from "@/components/orthopedic/ModifiedThomasLiveCamera";
import { ModifiedThomasReport } from "@/components/orthopedic/ModifiedThomasReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  SAMPLE_INTERVAL_MS,
  STABILITY_JITTER_MAX_DEG,
  STABILITY_WINDOW_FRAMES,
  TRIAL_DURATION_SEC,
  analyzeModifiedThomasUpload,
  buildInterpretation,
  computeHipAngle,
  computeKneeAngle,
  detectStableTail,
  fallbackReduction,
  isTestSideTrackable,
  summarizeSide,
  type FrameSample,
  type ModifiedThomasFullResult,
  type ModifiedThomasSideResult,
  type Side,
  type StableWindow,
} from "@/lib/orthopedic/modifiedThomas";

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
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  lastSampleAt: number;
  captureScreenshot: string | null;
}

export function ModifiedThomasCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  // Mode toggle — live (browser BlazePose WASM, per-side) vs upload
  // (backend MediaPipe, both sides sequentially).
  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [armedSide, setArmedSide] = useState<Side | null>(null);
  const [completedSides, setCompletedSides] = useState<Set<Side>>(new Set());
  const [result, setResult] = useState<ModifiedThomasFullResult>({ left: null, right: null });
  const [now, setNow] = useState<number>(0);

  // Live coaching: trackability + real-time hip/knee + stability lock.
  const [legTrackable, setLegTrackable] = useState<boolean>(false);
  const [liveHipDeg, setLiveHipDeg] = useState<number | null>(null);
  const [liveKneeDeg, setLiveKneeDeg] = useState<number | null>(null);
  const [liveStable, setLiveStable] = useState<boolean>(false);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  const recordingRef = useRef<RecordingState | null>(null);
  const resultRef = useRef(result);
  useEffect(() => { resultRef.current = result; }, [result]);

  // ── Auto-flow (fullscreen less-click live mode) ────────────────
  // One click on a side button opens the fullscreen shell; the camera
  // auto-starts; once frames are flowing a 3-2-1 countdown runs and
  // the trial starts by itself. Each side is its own fullscreen pass
  // (the shell closes after a side captures), so the countdown runs
  // before EVERY trial. The done view auto-saves (doctor flow).
  const [liveFullscreen, setLiveFullscreen] = useState<boolean>(false);
  const [camActive, setCamActive] = useState<boolean>(false);

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

  const finishSide = useCallback((
    termination: ModifiedThomasSideResult["termination"],
    capture: StableWindow | null,
    lowConfidence: boolean,
  ) => {
    const rec = recordingRef.current;
    if (!rec) return;
    // Fallback screenshot if we didn't latch one earlier.
    if (!rec.captureScreenshot) {
      const grab = (window as unknown as {
        __modifiedThomasCapture?: () => string | null;
      }).__modifiedThomasCapture;
      if (grab) {
        const url = grab();
        if (url) rec.captureScreenshot = url;
      }
    }
    const summary = summarizeSide(
      rec.side,
      rec.startedAt,
      Date.now(),
      termination,
      rec.samples,
      rec.keypoints,
      capture,
      lowConfidence,
      rec.captureScreenshot,
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
    setLiveHipDeg(null);
    setLiveKneeDeg(null);
    setLiveStable(false);
    // Leave the fullscreen shell — back to the side picker (or the
    // done view once both sides have landed).
    setLiveFullscreen(false);
    setCamActive(false);

    setPhase(() => {
      const r = resultRef.current;
      const otherDone = rec.side === "left" ? r.right !== null : r.left !== null;
      return otherDone ? "done" : "idle";
    });
  }, []);

  // Countdown starts only once the camera stream is actually live —
  // otherwise the trial timer would eat the getUserMedia permission
  // delay. onLive fires startRecording (declared below; hoisted).
  const {
    phase: flowPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(liveFullscreen && camActive, () => {
    startRecording();
  });

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

    // Trial-timeout safety net — use the fallback reduction if nothing
    // ever stabilised. Returns a low-confidence summary instead of an
    // error so the operator can still see what was captured.
    const elapsedSec = (tNow - rec.startedAt) / 1000;
    if (elapsedSec >= TRIAL_DURATION_SEC) {
      const fallback = fallbackReduction(rec.samples);
      finishSide("timeout", fallback, /* lowConfidence */ true);
      return;
    }

    const hip = computeHipAngle(kp, rec.side);
    const knee = computeKneeAngle(kp, rec.side);

    // Record the sample WITHOUT a stable flag for now; we'll update
    // its `stable` field below if the rolling window passes.
    const sample: FrameSample = {
      t_ms: tNow - rec.startedAt,
      hip_angle_deg: hip,
      knee_angle_deg: knee,
      stable: false,
    };
    rec.samples.push(sample);
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    setLiveHipDeg(hip);
    setLiveKneeDeg(knee);

    // Stability check on the rolling tail.
    const stableWindow = detectStableTail(rec.samples);
    setLiveStable(stableWindow !== null);

    if (stableWindow !== null) {
      // Mark every sample in the window as stable (for the chart).
      const startIdx = Math.max(0, rec.samples.length - STABILITY_WINDOW_FRAMES);
      for (let i = startIdx; i < rec.samples.length; i++) {
        rec.samples[i].stable = true;
      }
      // Latch a capture screenshot the instant stability is detected.
      if (!rec.captureScreenshot) {
        const grab = (window as unknown as {
          __modifiedThomasCapture?: () => string | null;
        }).__modifiedThomasCapture;
        if (grab) {
          const url = grab();
          if (url) rec.captureScreenshot = url;
        }
      }
      finishSide("captured", stableWindow, /* lowConfidence */ false);
      return;
    }

    // Coaching message
    if (hip === null || knee === null) {
      setCoachIfChanged(
        "Test-side leg not fully visible — keep the patient's shoulder, hip, knee, and ankle in frame.",
      );
    } else if (rec.samples.length < STABILITY_WINDOW_FRAMES) {
      setCoachIfChanged(
        `Settle into position — measuring hip ${hip.toFixed(0)}°, knee ${knee.toFixed(0)}°…`,
      );
    } else {
      setCoachIfChanged(
        `Hold still — hip ${hip.toFixed(0)}°, knee ${knee.toFixed(0)}° (need to settle within ±${STABILITY_JITTER_MAX_DEG}° for ${(STABILITY_WINDOW_FRAMES / 10).toFixed(1)} s).`,
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

    // Sequential, not parallel — same cold-worker 502 mitigation as SLR/AKE.
    let leftResult:  ModifiedThomasSideResult | null = null;
    let rightResult: ModifiedThomasSideResult | null = null;
    let leftErr:  string | null = null;
    let rightErr: string | null = null;

    if (leftFile) {
      try {
        leftResult = await analyzeModifiedThomasUpload(leftFile, "left", setLeftProgress);
      } catch (e) {
        leftErr = errorMessage(e) ?? "Left-side analysis failed.";
      }
    }
    if (rightFile) {
      try {
        rightResult = await analyzeModifiedThomasUpload(rightFile, "right", setRightProgress);
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
    setLiveFullscreen(false);
    setCamActive(false);
    resetUpload();
    setMode(next);
  }

  // Enter the fullscreen auto-flow shell for one side (the single
  // click of the live mode). Camera auto-starts inside; countdown →
  // recording. The old pre-record trackability gate is advisory now —
  // the camera only starts inside the shell, so the per-frame coach
  // message covers visibility instead of blocking the start.
  function enterLive(side: Side) {
    setError(null);
    setArmedSide(side);
    setPhase("armed");
    setLiveFullscreen(true);
  }

  function exitLive() {
    recordingRef.current = null;
    setArmedSide(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setLiveHipDeg(null);
    setLiveKneeDeg(null);
    setLiveStable(false);
    setPhase("idle");
    setLiveFullscreen(false);
    setCamActive(false);
  }

  function startRecording() {
    if (!armedSide) return;
    setError(null);
    recordingRef.current = {
      side: armedSide,
      startedAt: Date.now(),
      samples: [],
      keypoints: [],
      lastSampleAt: 0,
      captureScreenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg(
      `Let the ${armedSide} leg settle. We'll auto-capture the moment hip and ` +
      `knee angles stop moving.`,
    );
    setLiveHipDeg(null);
    setLiveKneeDeg(null);
    setLiveStable(false);
    setPhase("recording");
  }

  function stopEarly() {
    const rec = recordingRef.current;
    if (!rec) return;
    const fallback = fallbackReduction(rec.samples);
    finishSide("stopped", fallback, /* lowConfidence */ true);
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
    setLiveHipDeg(null);
    setLiveKneeDeg(null);
    setLiveStable(false);
  }

  // Done view ---------------------------------------------------------
  const isLiveDone = phase === "done";
  const isUploadDone = uploadPhase === "done";
  if (isLiveDone || isUploadDone) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone ? () => { resetUpload(); } : reset;
    const buildPayload = () => ({
      module: "modified_thomas" as const,
      metrics: { left: result.left, right: result.right },
      observations: { interpretation },
    });
    return (
      <div className="space-y-8">
        {/* Results auto-save in the doctor flow (toast with a 10s
            undo) for both live and upload runs. */}
        <AutoSaveToast buildPayload={buildPayload} />

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

        <ModifiedThomasReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
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
  const elapsedSec =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remainingSec = Math.max(0, TRIAL_DURATION_SEC - elapsedSec);

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
                "Patient sits on the edge of a flat table, then lies back so the upper body rests on the table and both legs hang off the edge.",
                "Camera is positioned to the SIDE (lateral view) — frame TALL so the whole body, from shoulder down to the hanging ankle, fits in view.",
                "Record TWO clips. For each, the OPPOSITE knee is pulled to the chest, and the LEG BEING TESTED hangs naturally off the edge.",
                "In each clip, the patient settles into the hanging position and holds still for at least 2 seconds.",
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
              label="Left-leg MTT"
              hint="Right knee to chest; LEFT leg hangs and is measured. Camera on patient's LEFT side."
              file={leftFile}
              onPick={(f) => validateAndSetFile("left", f)}
              progress={leftProgress}
              busy={uploadPhase === "analyzing" && leftFile !== null}
              error={uploadErrors.left}
            />
            <SidePicker
              label="Right-leg MTT"
              hint="Left knee to chest; RIGHT leg hangs and is measured. Camera on patient's RIGHT side."
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

      {/* ─── LIVE MODE — pre-fullscreen: instructions + side pick ── */}
      {mode === "live" && !liveFullscreen && (
        <>
          <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
            <div className="space-y-5">
              <div className="rounded-card border border-border bg-surface p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                  Movement instructions
                </p>
                <ol className="mt-3 space-y-2.5 text-sm text-foreground">
                  {[
                    "Patient sits on the edge of a flat table, then lies back — upper body supported, both legs hanging off the edge.",
                    "Place the camera on the SIDE of the patient, framed TALL: shoulder at the top, hanging ankle at the bottom.",
                    "Choose which leg to test. The OPPOSITE knee is pulled up and held against the chest (this stabilises the pelvis — it is NOT measured).",
                    "The TEST leg is released and allowed to hang naturally off the edge.",
                    "Once the patient is settled, the trial auto-captures the moment hip and knee angles stop changing for 1.5 s.",
                    "Repeat on the other side after the first capture lands.",
                  ].map((s, i) => (
                    <li key={i} className="flex gap-2.5">
                      <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                      <span className="leading-relaxed">{s}</span>
                    </li>
                  ))}
                </ol>
              </div>
              <p className="text-xs text-muted">
                Hip cutoffs: ≥ 170° = normal · 155–170° = mild · &lt; 155° = significant tightness.
                Knee cutoffs: ≤ 100° = relaxed · &gt; 100° = rectus femoris tightness.
              </p>
            </div>

            <div className="space-y-4">
              <div className="rounded-card border border-border bg-surface p-6 text-center">
                {sidesRemaining.length === 0 ? (
                  <p className="text-sm text-muted">
                    Both sides recorded. Compiling the report…
                  </p>
                ) : (
                  <>
                    <p className="text-sm text-muted">
                      One click per side — the camera opens fullscreen, a 3-2-1
                      countdown runs, and the trial starts by itself. Each side
                      auto-captures the moment the hanging leg settles; once both
                      sides land the report saves to the patient record.
                    </p>
                    <div className="mt-4 flex flex-wrap justify-center gap-3">
                      {sidesRemaining.map((s) => (
                        <Button key={s} onClick={() => enterLive(s)}>
                          <Camera className="h-4 w-4" />
                          {s === "left" ? "Left" : "Right"}-leg MTT
                        </Button>
                      ))}
                    </div>
                    {completedSides.size > 0 && (
                      <p className="mt-3 inline-flex items-center gap-1 text-xs text-emerald-600">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        {completedSides.size === 1 ? "1 side recorded" : "Both sides recorded"}
                      </p>
                    )}
                  </>
                )}
              </div>
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

      {/* ─── LIVE MODE — fullscreen auto-flow shell ──────────────── */}
      {mode === "live" && liveFullscreen && (
        <LiveModeLayout
          title="Modified Thomas Test"
          subtitle={
            phase === "recording"
              ? `${liveSide === "left" ? "Left" : "Right"}-leg MTT · ${liveStable ? "Settled ✓" : "Settling…"}`
              : `${liveSide === "left" ? "Left" : "Right"}-leg MTT — ${liveSide === "left" ? "RIGHT" : "LEFT"} knee to chest`
          }
          onExit={exitLive}
          camera={(
            <ModifiedThomasLiveCamera
              onFrame={handleFrame}
              onError={setError}
              autoStart
              hideControls
              fill
              onActiveChange={setCamActive}
            >
              {!camActive && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <p className="rounded-full bg-black/60 px-4 py-2 text-sm text-white/80">
                    Starting camera…
                  </p>
                </div>
              )}
              {flowPhase === "countdown" && countdown !== null && (
                <AutoFlowCountdownOverlay countdown={countdown} label="Trial starts in" />
              )}
              {phase === "recording" && (
                <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-rose-300">
                    ● Recording · {liveSide === "left" ? "Left" : "Right"} leg
                  </p>
                  <p className="tabular text-2xl font-semibold text-white">
                    {liveHipDeg !== null ? `${liveHipDeg.toFixed(0)}°` : "—"} hip ·{" "}
                    {liveKneeDeg !== null ? `${liveKneeDeg.toFixed(0)}°` : "—"} knee
                  </p>
                  <p className="text-[10px] text-white/70">
                    {liveStable ? "Settled ✓" : "Settling…"} · timeout in {remainingSec.toFixed(0)}s
                  </p>
                </div>
              )}
            </ModifiedThomasLiveCamera>
          )}
          sidebar={(
            <>
              {flowPhase === "countdown" && countdown !== null && (
                <AutoFlowCountdownCard
                  countdown={countdown}
                  onSkip={skipCountdown}
                  hint={`Patient supine on the table edge, ${armedSide === "left" ? "RIGHT" : "LEFT"} knee to the chest, ${armedSide ?? "test"} leg hanging naturally.`}
                />
              )}

              {/* Trackability status — advisory in the auto-flow. */}
              <div
                className={`rounded-card border p-3 text-sm ${
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
              </div>

              {phase === "recording" && (
                <div className="rounded-card border border-border bg-surface p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-foreground">
                      {liveSide === "left" ? "Left" : "Right"}-leg MTT
                    </p>
                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs ${
                        liveStable
                          ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                          : "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                      }`}
                    >
                      {liveStable ? "Settled ✓" : "Settling…"}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-md bg-elevated p-2.5">
                      <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">Hip</p>
                      <p className="tabular text-xl font-semibold text-foreground">
                        {liveHipDeg !== null ? `${liveHipDeg.toFixed(0)}°` : "—"}
                      </p>
                    </div>
                    <div className="rounded-md bg-elevated p-2.5">
                      <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">Knee</p>
                      <p className="tabular text-xl font-semibold text-foreground">
                        {liveKneeDeg !== null ? `${liveKneeDeg.toFixed(0)}°` : "—"}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <span>{remainingSec.toFixed(0)} s remaining</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                    <div
                      className="h-full bg-accent transition-all"
                      style={{ width: `${(elapsedSec / TRIAL_DURATION_SEC) * 100}%` }}
                    />
                  </div>
                  {coachMsg && (
                    <p className="rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
                      {coachMsg}
                    </p>
                  )}
                </div>
              )}

              <div className="rounded-card border border-border bg-surface p-3 text-xs text-muted">
                <p className="font-semibold text-foreground">Session brief</p>
                <ol className="mt-2 list-decimal space-y-1 pl-4">
                  <li>Supine on the table edge, camera side-on, framed tall.</li>
                  <li>{(liveSide ?? "test") === "left" ? "RIGHT" : "LEFT"} knee to the chest; the {liveSide ?? "test"} leg hangs naturally.</li>
                  <li>Auto-captures once hip + knee angles hold still for 1.5 s.</li>
                </ol>
              </div>

              {error && (
                <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-foreground">
                  <AlertTriangle className="mr-2 inline h-4 w-4 text-rose-500" />
                  {error}
                </div>
              )}

              <div className="mt-auto flex flex-wrap gap-2">
                {phase === "recording" && (
                  <Button variant="secondary" onClick={stopEarly}>Stop early</Button>
                )}
                <Button variant="ghost" size="sm" onClick={exitLive}>
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
              </div>
            </>
          )}
        />
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
