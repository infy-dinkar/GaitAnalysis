"use client";
// Trendelenburg test capture flow.
//
// State machine:
//   idle → ready_left → recording_left → done_left →
//          ready_right → recording_right → done_right →
//          done
//
// During `recording_*`:
//   - Sample pelvic-tilt + trunk-lean at 10 Hz
//   - Save per-frame keypoints
//   - Track running max-drop and capture a screenshot at the moment
//     of max drop
//   - Auto-terminate if the lifted foot returns to the ground OR
//     pelvic tilt spikes past PELVIC_SPIKE_TERMINATION_DEG
//   - End normally after TARGET_HOLD_SECONDS (30s)

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
import { TrendelenburgLiveCamera } from "@/components/orthopedic/TrendelenburgLiveCamera";
import { TrendelenburgReport } from "@/components/orthopedic/TrendelenburgReport";
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
  COMPENSATORY_TRUNK_LEAN_DEG,
  PELVIC_SPIKE_TERMINATION_DEG,
  SAMPLE_INTERVAL_MS,
  TARGET_HOLD_SECONDS,
  analyzeTrendelenburgUpload,
  buildInterpretation,
  computePelvicTilt,
  computeTrunkLean,
  detectStanceSide,
  dropForStance,
  summarizeSide,
  type Side,
  type TrendelenburgFrameSample,
  type TrendelenburgFullResult,
  type TrendelenburgSideResult,
} from "@/lib/orthopedic/trendelenburg";

type Mode = "live" | "upload";

type Phase =
  | "idle"
  | "ready"
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

// Once the patient has reached single-leg stance, a brief loss of
// the detected stance (foot wobble) is allowed before the test
// terminates as a foot-touch. Was 1s; bumped to 2s so a momentary
// dip doesn't end the recording.
const POST_LIFT_LOSS_GRACE_SEC = 2.0;
// Hard ceiling on "armed but never lifted" — if the patient hasn't
// achieved stance after this many seconds we still terminate so the
// UI doesn't sit indefinitely.
const PRE_LIFT_TIMEOUT_SEC = 12.0;

interface RecordingState {
  side: Side;
  startedAt: number;
  samples: TrendelenburgFrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  lastSampleAt: number;
  maxDropSoFar: number;
  peakScreenshotDataUrl: string | null;
  /** Set the first time `detectStanceSide` returns the expected side.
   *  Determines whether subsequent stance loss is "foot wobble" (post-
   *  lift, terminate) or "still getting ready" (pre-lift, wait). */
  firstStanceAt: number | null;
  /** Most recent timestamp at which `detectStanceSide` returned a
   *  non-null value (used for the post-lift loss grace window). */
  lastStanceSeenAt: number | null;
}

export function TrendelenburgCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  // Mode toggle: live (browser BlazePose WASM, per-side) vs upload
  // (backend MediaPipe, both sides sequentially). Both modes
  // converge on the same TrendelenburgFullResult + TrendelenburgReport.
  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrendelenburgFullResult>({ left: null, right: null });
  const [now, setNow] = useState<number>(0); // re-render driver for the countdown
  // Live coaching message shown to the operator + patient during the
  // hold. Updated from `handleFrame` only when the message changes
  // (otherwise React would re-render on every detection tick).
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachMsgRef = useRef<string>("");

  // Active-recording state lives in a ref because the per-frame
  // callback fires from a rAF loop and can't depend on React state
  // closures without going stale.
  const recordingRef = useRef<RecordingState | null>(null);
  // Which side(s) are still pending so the user can do them in
  // any order. After one side completes, the OTHER side is offered.
  const [completedSides, setCompletedSides] = useState<Set<Side>>(new Set());
  // Currently-armed side (set when user clicks "Start <side>-leg stance").
  const [armedSide, setArmedSide] = useState<Side | null>(null);

  // ── Auto-flow (fullscreen less-click live mode) ────────────────
  // Picking a side opens the fullscreen shell; the camera auto-starts;
  // once frames are flowing a 3-2-1 countdown runs and the hold starts
  // by itself. The countdown re-runs before EACH side: finishing a
  // side clears `armedSide` (resetting the hook), and arming the next
  // side flips it back on.
  const [liveFullscreen, setLiveFullscreen] = useState<boolean>(false);
  const [camActive, setCamActive] = useState<boolean>(false);

  const {
    phase: flowPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(
    liveFullscreen && camActive && armedSide !== null,
    () => {
      startRecording();
    },
  );

  // 250 ms tick for the live UI (countdown timer + current-tilt readout).
  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [phase]);

  const finishSide = useCallback((termination: TrendelenburgSideResult["termination"]) => {
    const rec = recordingRef.current;
    if (!rec) return;
    // Fallback screenshot: if no peak-drop frame was captured during
    // the trial (e.g. drop never exceeded 0 because the test was
    // negative), grab the current frame so the report still has a
    // visual reference of the stance.
    if (!rec.peakScreenshotDataUrl) {
      const grab = (window as unknown as {
        __trendelenburgCapture?: () => string | null;
      }).__trendelenburgCapture;
      if (grab) {
        const url = grab();
        if (url) rec.peakScreenshotDataUrl = url;
      }
    }
    const summary = summarizeSide(
      rec.side,
      rec.startedAt,
      Date.now(),
      termination,
      rec.samples,
      rec.keypoints,
      rec.peakScreenshotDataUrl,
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
    lastCoachMsgRef.current = "";
    // If both sides done → render report (leave the fullscreen shell
    // so the done view renders). Else go back to ready — the sidebar
    // offers the other side and the countdown re-runs on arm.
    const bothDone = completedSidesIncluding("ready", rec.side);
    setPhase(bothDone ? "done" : "ready");
    if (bothDone) {
      setLiveFullscreen(false);
      setCamActive(false);
    }
  }, []);

  // Helper: check whether THE OTHER side has already been recorded
  // by inspecting the current `result` state at the moment of the
  // finishSide call. Captured via a separate ref to avoid a stale
  // closure.
  const resultRef = useRef(result);
  useEffect(() => { resultRef.current = result; }, [result]);

  function completedSidesIncluding(_prev: Phase, justFinished: Side): boolean {
    const r = resultRef.current;
    const otherDone = justFinished === "left" ? r.right !== null : r.left !== null;
    return otherDone;
  }

  // Helper: publish a coaching message to the UI without spamming
  // setState on every frame.
  const setCoachIfChanged = useCallback((msg: string) => {
    if (lastCoachMsgRef.current === msg) return;
    lastCoachMsgRef.current = msg;
    setCoachMsg(msg);
  }, []);

  // ── Per-frame callback from the live camera ────────────────────
  const handleFrame = useCallback((kp: Keypoint[], video: HTMLVideoElement) => {
    // PRE-RECORD: just show coaching info — no recording in this branch.
    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    const expectedSide = rec.side;
    const expectedStr = expectedSide === "left" ? "left" : "right";
    const liftedStr   = expectedSide === "left" ? "right" : "left";
    const elapsedSec  = (tNow - rec.startedAt) / 1000;

    // Auto-terminate guards — but only AFTER stance has been achieved
    // at least once. Before that we sit in a "waiting for lift" state
    // and surface a coaching prompt instead of cutting the test short.
    const detectedStance = detectStanceSide(kp);

    if (detectedStance === expectedSide) {
      // Good stance. Mark first-detection + last-detection.
      if (rec.firstStanceAt === null) rec.firstStanceAt = tNow;
      rec.lastStanceSeenAt = tNow;
    } else {
      // Either no detection, or the wrong leg is on the ground.
      if (rec.firstStanceAt === null) {
        // Pre-lift: never reached stance yet. Coach the patient and
        // give them up to PRE_LIFT_TIMEOUT_SEC before bailing.
        if (detectedStance === null) {
          setCoachIfChanged(`Lift your ${liftedStr} leg — keep the ${expectedStr} foot planted.`);
        } else {
          setCoachIfChanged(`Switch sides — please stand on the ${expectedStr} leg.`);
        }
        if (elapsedSec > PRE_LIFT_TIMEOUT_SEC) {
          finishSide("foot_touch");
        }
        return; // don't sample / advance until the lift is detected
      }
      // Post-lift: stance was achieved before — this is a foot wobble
      // or full touch-down. Allow POST_LIFT_LOSS_GRACE_SEC of slack
      // before terminating.
      const sinceLastStance = rec.lastStanceSeenAt
        ? (tNow - rec.lastStanceSeenAt) / 1000
        : 0;
      if (sinceLastStance > POST_LIFT_LOSS_GRACE_SEC) {
        finishSide("foot_touch");
        return;
      }
      setCoachIfChanged(
        detectedStance === null
          ? "Stay up — keep that leg lifted."
          : `Wrong leg on the ground — return to standing on the ${expectedStr} leg.`,
      );
    }

    const pelvic = computePelvicTilt(kp);
    const lean = computeTrunkLean(kp);

    if (pelvic !== null && Math.abs(pelvic) > PELVIC_SPIKE_TERMINATION_DEG) {
      finishSide("spike");
      return;
    }

    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      pelvic_tilt_deg: pelvic,
      trunk_lean_deg: lean,
    });
    rec.keypoints.push(
      kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })),
    );

    // Track max drop for the peak-frame screenshot.
    if (pelvic !== null) {
      const drop = dropForStance(pelvic, rec.side);
      if (drop > rec.maxDropSoFar) {
        rec.maxDropSoFar = drop;
        const grab = (window as unknown as {
          __trendelenburgCapture?: () => string | null;
        }).__trendelenburgCapture;
        if (grab) {
          const url = grab();
          if (url) rec.peakScreenshotDataUrl = url;
        }
      }
    }

    // Live coaching while in good stance — based on the current pelvic
    // drop magnitude.
    if (rec.firstStanceAt !== null && pelvic !== null) {
      const drop = dropForStance(pelvic, rec.side);
      const absLean = lean !== null ? Math.abs(lean) : 0;
      if (drop > 5) {
        setCoachIfChanged("Pelvic drop is high — try to level your hips.");
      } else if (drop > 2) {
        setCoachIfChanged("Slight pelvic drop — engage the stance-side hip.");
      } else if (absLean > COMPENSATORY_TRUNK_LEAN_DEG) {
        setCoachIfChanged("Trunk leaning — try to keep the upper body upright.");
      } else {
        setCoachIfChanged("Holding well — stay steady.");
      }
    }

    // Voluntary 30-second completion.
    if (elapsedSec >= TARGET_HOLD_SECONDS) {
      finishSide("completed");
    }

    // Avoid TS "video unused" warning — we actually only need the
    // <video> reference for the parent's screenshot path, which goes
    // through the global captureFrame helper exposed by the camera.
    void video;
  }, [phase, finishSide, setCoachIfChanged]);

  // ── Upload-mode state ──────────────────────────────────────────
  // The frontend runs both sides sequentially and assembles the
  // combined result client-side. Errors per side are tracked
  // independently so one bad clip doesn't lose the other side's
  // result.
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

    // Sequential, not parallel. The backend has only 2 gunicorn workers
    // and each loads its MediaPipe BlazePose model on the first request
    // post-deploy; two parallel cold loads can blow past Vercel's ~30 s
    // upstream-response budget. Running left then right keeps each
    // request comfortably warm. Analysis math + result shape unchanged.
    let leftResult:  TrendelenburgSideResult | null = null;
    let rightResult: TrendelenburgSideResult | null = null;
    let leftErr:  string | null = null;
    let rightErr: string | null = null;

    if (leftFile) {
      try {
        leftResult = await analyzeTrendelenburgUpload(
          leftFile, "left", age, setLeftProgress,
        );
      } catch (e) {
        leftErr = errorMessage(e) ?? "Left-side analysis failed.";
      }
    }
    if (rightFile) {
      try {
        rightResult = await analyzeTrendelenburgUpload(
          rightFile, "right", age, setRightProgress,
        );
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
    // Block mode switching while either flow is mid-operation.
    if (phase === "recording" || uploadPhase === "analyzing") return;
    // Resetting back to a clean slate when switching avoids carrying
    // a half-finished trial across modes.
    recordingRef.current = null;
    setResult({ left: null, right: null });
    setCompletedSides(new Set());
    setArmedSide(null);
    setCoachMsg("");
    lastCoachMsgRef.current = "";
    setPhase("idle");
    setError(null);
    setLiveFullscreen(false);
    setCamActive(false);
    resetUpload();
    setMode(next);
  }

  // ── Arming + start ─────────────────────────────────────────────
  function arm(side: Side) {
    setError(null);
    setArmedSide(side);
    setPhase("ready");
  }

  // Enter the fullscreen auto-flow shell — picking a side is the
  // single click of the live mode. Camera auto-starts inside;
  // countdown → recording of that side.
  function enterLive(side: Side) {
    arm(side);
    setLiveFullscreen(true);
  }

  function exitLive() {
    recordingRef.current = null;
    setArmedSide(null);
    setCoachMsg("");
    lastCoachMsgRef.current = "";
    setPhase("idle");
    setLiveFullscreen(false);
    setCamActive(false);
  }

  function startRecording() {
    if (!armedSide) return;
    recordingRef.current = {
      side: armedSide,
      startedAt: Date.now(),
      samples: [],
      keypoints: [],
      lastSampleAt: 0,
      maxDropSoFar: 0,
      peakScreenshotDataUrl: null,
      firstStanceAt: null,
      lastStanceSeenAt: null,
    };
    lastCoachMsgRef.current = "";
    setCoachMsg(`Lift your ${armedSide === "left" ? "right" : "left"} leg to begin.`);
    setPhase("recording");
  }

  function stopEarly() {
    finishSide("foot_touch");
  }

  function reset() {
    recordingRef.current = null;
    setResult({ left: null, right: null });
    setCompletedSides(new Set());
    setArmedSide(null);
    setCoachMsg("");
    lastCoachMsgRef.current = "";
    setPhase("idle");
    setError(null);
    setLiveFullscreen(false);
    setCamActive(false);
  }

  // ── Done view (shared between live + upload modes) ────────────
  // Live mode: triggered when both sides have been recorded.
  // Upload mode: triggered when at least one side analysis succeeded.
  const isLiveDone = phase === "done";
  const isUploadDone = uploadPhase === "done";
  if (isLiveDone || isUploadDone) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone
      ? () => { resetUpload(); }
      : reset;
    const buildPayload = () => ({
      module: "trendelenburg" as const,
      // Per-frame keypoints + samples + screenshot data-URL all
      // live inside each per-side TrendelenburgSideResult on the
      // metrics blob (PDF Section 2 (a) compliance). The top-
      // level `keypoints` field is reserved for single-snapshot
      // modules (posture) and is left unset here.
      metrics: {
        left:  result.left,
        right: result.right,
      },
      observations: { interpretation },
    });
    return (
      <div className="space-y-8">
        {/* Results auto-save in the doctor flow (toast with a 10s
            undo) for both live and upload runs. */}
        <AutoSaveToast buildPayload={buildPayload} />

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

        <TrendelenburgReport
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

  // ── Capture view ───────────────────────────────────────────────
  const elapsed =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remaining = Math.max(0, TARGET_HOLD_SECONDS - elapsed);
  const liveSide = recordingRef.current?.side ?? armedSide ?? null;
  const sidesRemaining: Side[] = (["left", "right"] as Side[]).filter(
    (s) => !completedSides.has(s),
  );

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
                "Patient stands barefoot facing the camera, both hips visible end-to-end.",
                "Record TWO clips — one for left-leg stance, one for right-leg stance.",
                `Each clip should show the patient holding the single-leg stance for up to ${TARGET_HOLD_SECONDS} seconds.`,
                "Both clips are analysed one after the other (left first, then right).",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Two side-by-side file pickers */}
          <div className="grid gap-4 md:grid-cols-2">
            <SidePicker
              label="Left-leg stance"
              hint="Patient stands on the LEFT leg, lifts the RIGHT."
              file={leftFile}
              onPick={(f) => validateAndSetFile("left", f)}
              progress={leftProgress}
              busy={uploadPhase === "analyzing" && leftFile !== null}
              error={uploadErrors.left}
            />
            <SidePicker
              label="Right-leg stance"
              hint="Patient stands on the RIGHT leg, lifts the LEFT."
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
              <Button
                onClick={analyzeUpload}
                disabled={!leftFile && !rightFile}
              >
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

      {/* ─── LIVE MODE — pre-fullscreen: instructions + one click ── */}
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
                    "Stand barefoot facing the camera, both hips visible end-to-end.",
                    "Keep your arms relaxed at your sides or crossed across the chest.",
                    `Lift one leg by bending the hip and knee to about 90°. Don't reach for support.`,
                    `Hold the position steady for ${TARGET_HOLD_SECONDS} seconds. Keep the standing leg straight.`,
                    "When the timer ends, lower the leg and repeat the same hold on the opposite side.",
                  ].map((s, i) => (
                    <li key={i} className="flex gap-2.5">
                      <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                      <span className="leading-relaxed">{s}</span>
                    </li>
                  ))}
                </ol>
              </div>
              <p className="text-xs text-muted">
                Recording auto-stops if the lifted foot touches down for
                longer than {POST_LIFT_LOSS_GRACE_SEC.toFixed(0)} s, or if
                pelvic tilt exceeds {PELVIC_SPIKE_TERMINATION_DEG}°.
                Compensatory trunk lean beyond {COMPENSATORY_TRUNK_LEAN_DEG}°
                toward the stance side will be flagged.
              </p>
            </div>

            <div className="space-y-4">
              <div className="rounded-card border border-border bg-surface p-6 text-center">
                <p className="text-sm text-muted">
                  Pick a side — the camera opens fullscreen, a 3-2-1
                  countdown runs, and the {TARGET_HOLD_SECONDS}-second
                  hold records by itself. After both sides the report
                  saves to the patient record.
                </p>
                <div className="mt-4 flex flex-wrap justify-center gap-3">
                  {sidesRemaining.map((s) => (
                    <Button key={s} onClick={() => enterLive(s)}>
                      <Camera className="h-4 w-4" />
                      {s === "left" ? "Left" : "Right"}-leg stance
                    </Button>
                  ))}
                </div>
                {completedSides.size > 0 && (
                  <p className="mt-3 inline-flex items-center gap-1 text-xs text-emerald-600">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    {completedSides.size === 1
                      ? "1 side recorded"
                      : "Both sides recorded"}
                  </p>
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
          title="Trendelenburg Test"
          subtitle={
            phase === "recording"
              ? `${liveSide === "left" ? "Left" : "Right"}-leg stance — hold ${TARGET_HOLD_SECONDS}s`
              : armedSide
                ? `${armedSide === "left" ? "Left" : "Right"}-leg stance — get ready`
                : "Choose the next side"
          }
          onExit={exitLive}
          camera={(
            <TrendelenburgLiveCamera
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
                <AutoFlowCountdownOverlay countdown={countdown} label="Hold starts in" />
              )}
              {phase === "recording" && recordingRef.current && (
                <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-rose-300">
                    ● Recording — {liveSide === "left" ? "Left" : "Right"} stance
                  </p>
                  <p className="tabular text-2xl font-semibold text-white">
                    {recordingRef.current.firstStanceAt === null
                      ? "—"
                      : `${remaining.toFixed(1)}s`}
                  </p>
                  <p className="text-[10px] text-white/70">
                    {recordingRef.current.firstStanceAt === null
                      ? "Waiting for the leg lift"
                      : "Hold steady"}
                  </p>
                </div>
              )}
            </TrendelenburgLiveCamera>
          )}
          sidebar={(
            <>
              {flowPhase === "countdown" && countdown !== null && (
                <AutoFlowCountdownCard
                  countdown={countdown}
                  onSkip={skipCountdown}
                  hint={`Patient facing the camera, ready to stand on the ${armedSide ?? "test"} leg.`}
                />
              )}

              {phase === "recording" && recordingRef.current && (
                <div className="rounded-card border border-border bg-surface p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-foreground">
                      {recordingRef.current.firstStanceAt === null
                        ? `Waiting for stance — ${liveSide === "left" ? "Left" : "Right"}-leg test`
                        : `Recording — ${liveSide === "left" ? "Left" : "Right"}-leg stance`}
                    </p>
                    <p className="tabular text-2xl font-semibold text-accent">
                      {recordingRef.current.firstStanceAt === null
                        ? "—"
                        : `${remaining.toFixed(1)}s`}
                    </p>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                    <div
                      className="h-full bg-accent transition-all"
                      style={{
                        width: `${Math.min(
                          100,
                          recordingRef.current.firstStanceAt === null
                            ? 0
                            : (elapsed / TARGET_HOLD_SECONDS) * 100,
                        )}%`,
                      }}
                    />
                  </div>
                  {coachMsg && (
                    <p className="rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
                      {coachMsg}
                    </p>
                  )}
                </div>
              )}

              {phase !== "recording" && !armedSide && sidesRemaining.length > 0 && (
                <div className="rounded-card border border-border bg-surface p-4 space-y-3">
                  <p className="text-sm font-medium text-foreground">
                    {completedSides.size === 0
                      ? "Choose which side to record:"
                      : "Now record the other side:"}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {sidesRemaining.map((s) => (
                      <Button key={s} onClick={() => arm(s)}>
                        {s === "left" ? "Left" : "Right"}-leg stance
                      </Button>
                    ))}
                  </div>
                  {completedSides.size > 0 && (
                    <p className="inline-flex items-center gap-1 text-xs text-emerald-600">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      1 side recorded
                    </p>
                  )}
                </div>
              )}

              <div className="rounded-card border border-border bg-surface p-3 text-xs text-muted">
                <p className="font-semibold text-foreground">Session brief</p>
                <ol className="mt-2 list-decimal space-y-1 pl-4">
                  <li>Stand facing the camera, both hips in frame.</li>
                  <li>Lift the opposite leg — hold {TARGET_HOLD_SECONDS}s on the stance leg.</li>
                  <li>Auto-stops on foot touch-down or pelvic spike; then the other side.</li>
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
              <p className="truncate font-medium text-foreground">
                {file.name}
              </p>
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
