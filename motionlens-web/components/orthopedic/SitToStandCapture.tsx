"use client";
// 5x Sit-to-Stand capture flow.
//
// Single-trial test (no L/R split). State machine:
//   idle → armed (waiting for "Go" click) → recording → done
//
// Recording loop:
//   - Sample at 10 Hz
//   - Auto-detect baseline hip-mid Y (assumes patient is seated at click)
//   - Run sit↔stand state machine; each standing→sitting transition
//     closes a rep
//   - Track per-rep min knee angle (depth)
//   - Continuously evaluate arms-crossed posture; once a clear drop
//     is observed, set arm_uncrossed_flag = true (sticky for the trial)
//   - End on 5 reps OR 30 s timeout OR Stop early click

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Play,
  RotateCcw,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { SitToStandLiveCamera } from "@/components/orthopedic/SitToStandLiveCamera";
import { SitToStandReport } from "@/components/orthopedic/SitToStandReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  SAMPLE_INTERVAL_MS,
  TARGET_REP_COUNT,
  TRIAL_TIMEOUT_SEC,
  areArmsCrossed,
  buildInterpretation,
  computeHipMidY,
  computeKneeAngle,
  computeLegLengthPx,
  newRepDetector,
  stepRepDetector,
  summarizeTrial,
  type FrameSample,
  type RepDetectorState,
  type RepMetrics,
  type SitToStandResult,
  type Termination,
} from "@/lib/orthopedic/sitToStand";

type Phase = "idle" | "armed" | "recording" | "done";

interface RecordingState {
  startedAt: number;
  lastSampleAt: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  reps: RepMetrics[];
  detector: RepDetectorState;
  /** Sticky flag — set on first clear arm-uncross event. */
  armsUncrossedSeen: boolean;
  /** Ms since start when the most recent sit-back event was logged. */
  prevSitMs: number;
  /** Cached for the per-rep screenshot at the deepest part of the cycle. */
  deepestKneeSoFar: number;
  lastRepScreenshot: string | null;
}

export function SitToStandCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SitToStandResult | null>(null);
  const [now, setNow] = useState<number>(0);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  const recordingRef = useRef<RecordingState | null>(null);

  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [phase]);

  const setCoachIfChanged = useCallback((msg: string) => {
    if (lastCoachRef.current === msg) return;
    lastCoachRef.current = msg;
    setCoachMsg(msg);
  }, []);

  const finishTrial = useCallback((termination: Termination) => {
    const rec = recordingRef.current;
    if (!rec) return;
    const summary = summarizeTrial(
      rec.startedAt,
      Date.now(),
      termination,
      rec.reps,
      rec.armsUncrossedSeen,
      rec.samples,
      rec.keypoints,
      rec.lastRepScreenshot,
    );
    setResult(summary);
    recordingRef.current = null;
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("done");
  }, []);

  // Per-frame callback ----------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    const elapsed = (tNow - rec.startedAt) / 1000;
    if (elapsed >= TRIAL_TIMEOUT_SEC) {
      finishTrial("timeout");
      return;
    }

    const hipMidY = computeHipMidY(kp);
    const kneeAngle = computeKneeAngle(kp);
    const armsOk = areArmsCrossed(kp);

    // Capture leg length once for the rep-detector threshold.
    if (rec.detector.legLengthPx === null) {
      rec.detector.legLengthPx = computeLegLengthPx(kp);
    }

    // Sample for time-series.
    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      hip_mid_y: hipMidY,
      knee_angle_deg: kneeAngle,
      arms_crossed: armsOk,
    });
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Track deepest knee-bend frame (smallest knee angle) for the
    // last-rep screenshot. We grab a snapshot whenever a new minimum
    // is reached — by the end of the trial this is the deepest pose
    // captured.
    if (kneeAngle !== null && kneeAngle < rec.deepestKneeSoFar) {
      rec.deepestKneeSoFar = kneeAngle;
      const grab = (window as unknown as {
        __sitToStandCapture?: () => string | null;
      }).__sitToStandCapture;
      if (grab) {
        const url = grab();
        if (url) rec.lastRepScreenshot = url;
      }
    }

    if (!armsOk && !rec.armsUncrossedSeen) {
      rec.armsUncrossedSeen = true;
    }

    // Rep detection.
    const tMsRel = tNow - rec.startedAt;
    const stepResult = stepRepDetector(rec.detector, hipMidY, kneeAngle, tMsRel);

    // Coach message based on current detector state.
    if (rec.armsUncrossedSeen) {
      setCoachIfChanged("Arms uncrossed — keep both hands across the chest.");
    } else if (rec.detector.current === "standing") {
      setCoachIfChanged(`Sit back down — rep ${rec.reps.length + 1} of ${TARGET_REP_COUNT}.`);
    } else if (rec.reps.length === 0) {
      setCoachIfChanged("Stand up to start — five sit-to-stand cycles as fast as safely possible.");
    } else {
      setCoachIfChanged(`Stand back up — rep ${rec.reps.length + 1} of ${TARGET_REP_COUNT}.`);
    }

    if (stepResult.completedRep) {
      // Just transitioned standing → sitting. Close a rep.
      const sitEvents = rec.detector.sitEvents;
      const startMs = sitEvents[sitEvents.length - 2] ?? 0; // previous sit
      const endMs   = sitEvents[sitEvents.length - 1];      // this sit
      const duration = (endMs - startMs) / 1000;
      const newRep: RepMetrics = {
        rep_index: rec.reps.length + 1,
        duration_seconds: duration,
        min_knee_angle_deg: rec.detector.currentMinKneeAngle,
      };
      rec.reps.push(newRep);
      rec.prevSitMs = endMs;
      // Reset min-knee tracker for the next cycle.
      rec.detector.currentMinKneeAngle = 180;

      if (rec.reps.length >= TARGET_REP_COUNT) {
        finishTrial("completed");
        return;
      }
    }
  }, [phase, finishTrial, setCoachIfChanged]);

  // Arming + start ----------------------------------------------------
  function arm() {
    setError(null);
    setPhase("armed");
    setCoachMsg(
      "Patient should be seated, back against the chair, feet flat, arms crossed at chest. " +
      "Click Start when ready — timer begins immediately and patient should stand up.",
    );
  }

  function startRecording() {
    setError(null);
    recordingRef.current = {
      startedAt: Date.now(),
      lastSampleAt: 0,
      samples: [],
      keypoints: [],
      reps: [],
      detector: newRepDetector(),
      armsUncrossedSeen: false,
      prevSitMs: 0,
      deepestKneeSoFar: 180,
      lastRepScreenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg("Stand up to start — five sit-to-stand cycles as fast as safely possible.");
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
  if (phase === "done" && result) {
    const interpretation = buildInterpretation(result);
    return (
      <div className="space-y-8">
        <SitToStandReport
          patientName={patient?.name ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "sit_to_stand",
            metrics: { trial: result },
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
  const elapsedSec =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remainingSec = Math.max(0, TRIAL_TIMEOUT_SEC - elapsedSec);
  const repsCaptured = recordingRef.current?.reps.length ?? 0;
  const armsFlag = recordingRef.current?.armsUncrossedSeen ?? false;

  return (
    <div className="space-y-6">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <div className="rounded-card border border-accent/30 bg-accent/5 p-4 text-sm">
        <p className="font-medium text-foreground">5x Sit-to-Stand test</p>
        <p className="mt-1 text-muted">
          Patient is seated in profile to the camera, feet flat, arms crossed at the
          chest. On Start, patient performs {TARGET_REP_COUNT} full sit-to-stand
          cycles as fast as safely possible. Timer ends when buttocks contact the
          seat after the {TARGET_REP_COUNT}th rep.
        </p>
      </div>

      {/* Sticky right-aligned camera dock — same convention as A4 / B1 */}
      <div className="sticky top-20 z-20 ml-auto w-full max-w-md rounded-card bg-background/85 p-1 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/65">
        <SitToStandLiveCamera onFrame={handleFrame} onError={setError} />
      </div>

      {phase === "recording" && recordingRef.current && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-foreground">Recording</p>
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
          <div className="mt-3 flex items-center justify-between text-xs text-muted">
            <span>Elapsed: <span className="tabular text-foreground">{elapsedSec.toFixed(1)}s</span></span>
            <span>Timeout in {remainingSec.toFixed(0)} s</span>
          </div>
          {coachMsg && (
            <p className="mt-3 rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
              {coachMsg}
            </p>
          )}
          {armsFlag && (
            <p className="mt-2 rounded-md border border-warning/40 bg-warning/5 px-3 py-2 text-xs text-foreground">
              Arms-uncrossed event detected — the trial will be flagged on the report.
            </p>
          )}
          <div className="mt-3 flex justify-end">
            <Button variant="ghost" size="sm" onClick={stopEarly}>Stop early</Button>
          </div>
        </div>
      )}

      {phase !== "recording" && (
        <div className="rounded-card border border-border bg-surface p-5">
          {phase === "armed" ? (
            <div className="space-y-3">
              <p className="text-sm font-medium text-foreground">Ready to record</p>
              <p className="text-xs text-muted">
                Patient seated, back against backrest, feet flat, arms crossed at
                chest. The timer starts the moment you click <em>Start</em> — instruct
                the patient to begin standing immediately.
              </p>
              {coachMsg && (
                <p className="rounded-md bg-background/40 px-3 py-2 text-sm text-foreground">
                  {coachMsg}
                </p>
              )}
              <div className="flex gap-2">
                <Button onClick={startRecording}>
                  <Play className="h-4 w-4" />
                  Start trial
                </Button>
                <Button variant="ghost" onClick={() => setPhase("idle")}>Cancel</Button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm font-medium text-foreground">Begin a new trial</p>
              <Button onClick={arm}>
                Start
              </Button>
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

      <p className="text-xs text-muted">
        Cutoffs (PDF Test C2): &lt; 12 s normal, 12–15 s borderline, &gt; 15 s
        weakness / fall risk. Last-rep duration &gt; 60% of first-rep duration
        flags significant fatigue. Arm-uncrossing during the trial is reported
        as an integrity warning.
      </p>

      {/* Convenience indicator — when the patient is mid-trial and an
          arms-uncrossed event has fired, surface it again here so it
          stays in view if the operator scrolls past the recording panel. */}
      {phase === "recording" && armsFlag && (
        <p className="inline-flex items-center gap-1 rounded-full bg-warning/10 px-2.5 py-1 text-[11px] font-medium text-warning">
          <AlertTriangle className="h-3 w-3" />
          Arm uncrossed during trial
        </p>
      )}

      {phase === "recording" && !armsFlag && (
        <p className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-600">
          <CheckCircle2 className="h-3 w-3" />
          Arms crossed — good
        </p>
      )}
    </div>
  );
}
