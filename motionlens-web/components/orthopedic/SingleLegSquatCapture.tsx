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
  CheckCircle2,
  Play,
  RotateCcw,
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

type Phase = "idle" | "armed" | "recording" | "done";

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
  if (phase === "done") {
    const interpretation = buildInterpretation(result);
    return (
      <div className="space-y-8">
        <SingleLegSquatReport
          patientName={patient?.name ?? null}
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
          <Button variant="secondary" onClick={reset}>
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

  return (
    <div className="space-y-6">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <div className="rounded-card border border-accent/30 bg-accent/5 p-4 text-sm">
        <p className="font-medium text-foreground">
          Single-leg squat / step-down test
        </p>
        <p className="mt-1 text-muted">
          Patient stands on the test leg with the contralateral leg lifted,
          then performs {TARGET_REP_COUNT} squats to comfortable depth at
          a steady tempo (~2 s down, 2 s up). The system auto-detects each
          rep and captures KFPPA, pelvic drop, and trunk lean per rep.
        </p>
      </div>

      {/* Sticky camera dock — capped to ~480 px wide so it doesn't
          dominate the page, and pinned to the top of the viewport
          (top-20 ≈ below Nav) so the operator can keep watching the
          live skeleton while scrolling through controls / coaching. */}
      <div className="sticky top-20 z-20 ml-auto w-full max-w-md rounded-card bg-background/85 p-1 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/65">
        <SingleLegSquatLiveCamera onFrame={handleFrame} onError={setError} />
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

      {/* Recording panel */}
      {phase === "recording" && recordingRef.current && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-foreground">
              Recording — {liveSide === "left" ? "Left" : "Right"}-leg squat
            </p>
            <p className="tabular text-2xl font-semibold text-accent">
              {repsCaptured} / {TARGET_REP_COUNT}
            </p>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
            <div
              className="h-full bg-accent transition-all"
              style={{ width: `${(repsCaptured / TARGET_REP_COUNT) * 100}%` }}
            />
          </div>
          <p className="mt-3 text-xs text-muted">
            {remainingSec.toFixed(0)} s remaining before timeout. Squat to comfortable depth
            (thigh approaching horizontal). Trial auto-stops on the {TARGET_REP_COUNT}th rep
            or after {TRIAL_TIMEOUT_SEC} s.
          </p>
          {coachMsg && (
            <p className="mt-3 rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
              {coachMsg}
            </p>
          )}
          <div className="mt-3 flex justify-end">
            <Button variant="ghost" size="sm" onClick={stopEarly}>
              Stop early
            </Button>
          </div>
        </div>
      )}

      {/* Side picker / start */}
      {phase !== "recording" && (
        <div className="rounded-card border border-border bg-surface p-5">
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
                Patient should be standing on the {armedSide} leg with the
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
              <p className="text-sm font-medium">Choose which side to record next:</p>
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

      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}

      {/* Light-touch reference — show thresholds the operator can glance at */}
      <p className="text-xs text-muted">
        Cutoffs (PDF Test B1): KFPPA &lt;10° good, 10–15° borderline, &gt;15° valgus.
        Pelvic drop &gt;{PELVIC_DROP_THRESHOLD_DEG}° = hip abductor insufficiency.
        Trunk lateral lean &gt;{TRUNK_LEAN_THRESHOLD_DEG}° = compensatory pattern.
      </p>
    </div>
  );
}
