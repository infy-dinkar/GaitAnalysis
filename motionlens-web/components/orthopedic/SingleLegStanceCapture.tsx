"use client";
// Single-Leg Stance test (Test C5) — capture flow.
//
// Bilateral, two-condition test. Layout shows EYES OPEN and EYES
// CLOSED as two top-level groups, each with Left-leg / Right-leg
// trial buttons. Doctor picks any of the (up to four) trials in
// any order. Recording starts immediately on click — no audio cue
// (operator gives the verbal "close your eyes" instruction in
// person; this keeps the flow simple and reliable across devices).
//
// Each trial: patient lifts the leg, stance auto-detects, timer
// starts. Trial auto-terminates on foot-touchdown / arm-grab /
// hop / max-time / manual stop.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  EyeOff,
  Play,
  RotateCcw,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { SingleLegStanceLiveCamera } from "@/components/orthopedic/SingleLegStanceLiveCamera";
import { SingleLegStanceReport } from "@/components/orthopedic/SingleLegStanceReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  HOP_WINDOW_DURATION_MS,
  MAX_EYES_CLOSED_SEC,
  MAX_EYES_OPEN_SEC,
  ONSET_TIMEOUT_SEC,
  SAMPLE_INTERVAL_MS,
  buildInterpretation,
  buildSession,
  computeBodyHeightPx,
  computeHipMidpoint,
  computeShoulderMidpoint,
  computeTrunkLean,
  detectStanceSide,
  isArmGrab,
  isFootTouchdown,
  isHopInWindow,
  summarizeTrial,
  type Condition,
  type FrameSample,
  type SessionResult,
  type Side,
  type Termination,
  type TrialResult,
} from "@/lib/orthopedic/singleLegStance";
import { LM } from "@/lib/pose/landmarks";

type TrialKey = "left_open" | "right_open" | "left_closed" | "right_closed";

function trialKey(side: Side, condition: Condition): TrialKey {
  return `${side}_${condition === "eyes_open" ? "open" : "closed"}` as TrialKey;
}

type Phase = "idle" | "recording" | "done";

interface RecordingState {
  side: Side;
  condition: Condition;
  startedAt: number;        // wall-clock ms when timer began
  firstStanceAt: number | null;
  lastSampleAt: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  hipPath: Array<{ x: number; y: number }>;
  trunkLeans: number[];
  /** Rolling window of stance-ankle Y for hop detection. */
  stanceAnkleWindow: Array<{ t_ms: number; y: number }>;
  screenshot: string | null;
}


// ─── Capture component ──────────────────────────────────────────

export function SingleLegStanceCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [trials, setTrials] = useState<SessionResult["trials"]>({});
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

    // Fallback screenshot — same pattern as the other tests.
    if (!rec.screenshot) {
      const grab = (window as unknown as {
        __singleLegStanceCapture?: () => string | null;
      }).__singleLegStanceCapture;
      if (grab) {
        const url = grab();
        if (url) rec.screenshot = url;
      }
    }

    const startMs = rec.firstStanceAt ?? rec.startedAt;
    const summary: TrialResult = summarizeTrial({
      side: rec.side,
      condition: rec.condition,
      startedAtMs: startMs,
      endedAtMs: Date.now(),
      termination,
      hipPath: rec.hipPath,
      trunkLeans: rec.trunkLeans,
      samples: rec.samples,
      keypoints: rec.keypoints,
      screenshotDataUrl: rec.screenshot,
      patientAge: patient?.age ?? null,
    });

    const key = trialKey(rec.side, rec.condition);
    setTrials((prev) => ({ ...prev, [key]: summary }));
    recordingRef.current = null;
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
  // patient may change identity between trials in theory, capture once
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient]);

  // Per-frame callback ----------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    if (phase !== "recording" || !recordingRef.current) return;
    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    const elapsedSinceStart = (tNow - rec.startedAt) / 1000;
    const cap = rec.condition === "eyes_open" ? MAX_EYES_OPEN_SEC : MAX_EYES_CLOSED_SEC;

    const detected = detectStanceSide(kp);
    const hipMid = computeHipMidpoint(kp);
    const shMid  = computeShoulderMidpoint(kp);
    const bodyH  = computeBodyHeightPx(kp);
    const trunkLean = computeTrunkLean(kp);

    // Record a sample even before stance is achieved, so the saved
    // landmarks JSON covers the whole trial duration.
    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      hip_x: hipMid?.x ?? null,
      hip_y: hipMid?.y ?? null,
      trunk_lean_deg: trunkLean,
    });
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    // Pre-onset (waiting for leg lift).
    if (rec.firstStanceAt === null) {
      if (detected === rec.side) {
        // Stance achieved. Reset baseline timestamps so hold-time
        // measures from this point.
        rec.firstStanceAt = tNow;
        rec.startedAt = tNow;
        rec.hipPath = [];
        rec.trunkLeans = [];
        rec.stanceAnkleWindow = [];
        setCoachIfChanged(
          `Hold steady — max ${cap.toFixed(0)} s. Trial ends if your foot touches down or you reach for support.`,
        );
        return;
      }
      // No lift yet — coach the patient.
      if (elapsedSinceStart > ONSET_TIMEOUT_SEC) {
        finishTrial("no_lift_detected");
        return;
      }
      setCoachIfChanged(
        `Lift your ${rec.side === "left" ? "right" : "left"} leg — keep the ${rec.side} foot planted.`,
      );
      return;
    }

    // POST-ONSET — track hip-mid for sway, watch terminations.
    if (hipMid) rec.hipPath.push({ x: hipMid.x, y: hipMid.y });
    if (trunkLean !== null) rec.trunkLeans.push(trunkLean);

    // Foot touchdown.
    if (isFootTouchdown(kp, rec.side)) {
      finishTrial("foot_touchdown");
      return;
    }

    // Arm grab.
    if (isArmGrab(kp)) {
      finishTrial("arm_grab");
      return;
    }

    // Hop / stance foot reposition. Tracks a rolling window of
    // stance-ankle Y; if the spread exceeds the threshold inside
    // the window, we call it a hop.
    const stanceAnkleIdx = rec.side === "left" ? LM.LEFT_ANKLE : LM.RIGHT_ANKLE;
    const stanceAnkle = kp[stanceAnkleIdx];
    if (stanceAnkle && (stanceAnkle.score ?? 0) >= 0.3 && bodyH) {
      const t = tNow - rec.startedAt;
      rec.stanceAnkleWindow.push({ t_ms: t, y: stanceAnkle.y });
      while (
        rec.stanceAnkleWindow.length > 0 &&
        t - rec.stanceAnkleWindow[0].t_ms > HOP_WINDOW_DURATION_MS
      ) {
        rec.stanceAnkleWindow.shift();
      }
      if (isHopInWindow(rec.stanceAnkleWindow, bodyH)) {
        finishTrial("hop");
        return;
      }
    }

    // Max-time ceiling.
    const heldSec = (tNow - rec.firstStanceAt) / 1000;
    if (heldSec >= cap) {
      finishTrial("max_time");
      return;
    }

    // Live coaching (hold-time readout).
    setCoachIfChanged(
      `Holding — ${heldSec.toFixed(1)} s of up to ${cap.toFixed(0)} s.`,
    );

    // Suppress unused warning in dev — shoulder midpoint isn't
    // currently surfaced live but we capture it for any future
    // overlay.
    void shMid;
  }, [phase, finishTrial, setCoachIfChanged]);

  // Begin a trial directly. No audio cue — operator gives the
  // verbal "close your eyes" instruction in person before clicking.
  function beginRecording(side: Side, condition: Condition) {
    recordingRef.current = {
      side,
      condition,
      startedAt: Date.now(),
      firstStanceAt: null,
      lastSampleAt: 0,
      samples: [],
      keypoints: [],
      hipPath: [],
      trunkLeans: [],
      stanceAnkleWindow: [],
      screenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg(
      `Lift your ${side === "left" ? "right" : "left"} leg to begin — timer starts when stance is detected.`,
    );
    setPhase("recording");
  }

  function stopEarly() {
    finishTrial("stopped");
  }

  function reset() {
    recordingRef.current = null;
    setTrials({});
    setPhase("idle");
    setError(null);
    setCoachMsg("");
    lastCoachRef.current = "";
  }

  // Done view ---------------------------------------------------------
  // Show the "Generate report" CTA once both eyes-open trials are
  // captured (the spec-mandated minimum for a complete session).
  // Eyes-closed trials remain optional and can be added before
  // generating the report.
  const allDesiredDone = !!trials.left_open && !!trials.right_open;
  const showReport = phase === "done";
  const session = buildSession(trials, patient?.age ?? null);

  if (showReport) {
    const interpretation = buildInterpretation(session);
    return (
      <div className="space-y-8">
        <SingleLegStanceReport
          patientName={patient?.name ?? null}
          session={session}
          interpretation={interpretation}
        />
        <SaveToPatientButton
          buildPayload={() => ({
            module: "single_leg_stance",
            metrics: { session },
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
    phase === "recording" && recordingRef.current?.firstStanceAt !== null && recordingRef.current
      ? (now - (recordingRef.current.firstStanceAt ?? now)) / 1000
      : 0;
  const liveCondition = recordingRef.current?.condition ?? null;
  const cap = liveCondition === "eyes_closed" ? MAX_EYES_CLOSED_SEC : MAX_EYES_OPEN_SEC;
  const liveSide = recordingRef.current?.side ?? null;

  return (
    <div className="space-y-6">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <div className="rounded-card border border-accent/30 bg-accent/5 p-4 text-sm">
        <p className="font-medium text-foreground">Single-Leg Stance test</p>
        <p className="mt-1 text-muted">
          Patient stands barefoot facing the camera, hands on hips or at the
          sides. Lift one leg to roughly 90° hip flexion and hold as long
          as possible. Eyes-open trials run up to {MAX_EYES_OPEN_SEC} s;
          eyes-closed up to {MAX_EYES_CLOSED_SEC} s. Trial ends on foot
          touchdown, arm grab for support, hop, or max time.
        </p>
      </div>

      {!patient?.age && isDoctorFlow && (
        <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
          <p className="font-medium text-foreground">Patient age missing</p>
          <p className="mt-1 text-muted">
            The age-based norm comparison will fall back to the strictest
            band (under 60). Add the patient&apos;s age to the profile for
            an accurate comparison.
          </p>
        </div>
      )}

      {/* Sticky right-aligned camera dock */}
      <div className="sticky top-20 z-20 ml-auto w-full max-w-md rounded-card bg-background/85 p-1 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/65">
        <SingleLegStanceLiveCamera onFrame={handleFrame} onError={setError} />
      </div>

      {/* Trial selection — two top-level groups: eyes-open / eyes-closed.
          Each has Left- and Right-leg buttons. Doctor picks any in any
          order. No audio cue — operator gives the verbal "close your
          eyes" instruction in person. */}
      {phase === "idle" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <ConditionGroup
            title="Eyes open"
            subtitle={`Max hold ${MAX_EYES_OPEN_SEC} s per side.`}
            icon={<Eye className="h-4 w-4 text-accent" />}
            leftDone={!!trials.left_open}
            rightDone={!!trials.right_open}
            onLeft={() => beginRecording("left", "eyes_open")}
            onRight={() => beginRecording("right", "eyes_open")}
          />
          <ConditionGroup
            title="Eyes closed"
            subtitle={`Max hold ${MAX_EYES_CLOSED_SEC} s per side. Tell the patient to close their eyes before clicking.`}
            icon={<EyeOff className="h-4 w-4 text-accent" />}
            leftDone={!!trials.left_closed}
            rightDone={!!trials.right_closed}
            onLeft={() => beginRecording("left", "eyes_closed")}
            onRight={() => beginRecording("right", "eyes_closed")}
          />
        </div>
      )}

      {/* Recording panel */}
      {phase === "recording" && recordingRef.current && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-foreground">
              {recordingRef.current.firstStanceAt === null
                ? `Waiting for stance — ${liveSide === "left" ? "Left" : "Right"}-leg ${liveCondition === "eyes_closed" ? "(eyes closed)" : "(eyes open)"}`
                : `Recording — ${liveSide === "left" ? "Left" : "Right"}-leg ${liveCondition === "eyes_closed" ? "(eyes closed)" : "(eyes open)"}`}
            </p>
            <p className="tabular text-2xl font-semibold text-accent">
              {recordingRef.current.firstStanceAt === null ? "—" : `${elapsedSec.toFixed(1)}s`}
            </p>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
            <div
              className="h-full bg-accent transition-all"
              style={{ width: `${Math.min(100, (elapsedSec / cap) * 100)}%` }}
            />
          </div>
          {coachMsg && (
            <p className="mt-3 rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
              {coachMsg}
            </p>
          )}
          <div className="mt-3 flex justify-end">
            <Button variant="ghost" size="sm" onClick={stopEarly}>Stop early</Button>
          </div>
        </div>
      )}

      {/* Done CTA */}
      {phase === "idle" && allDesiredDone && (
        <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-5">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <p className="text-sm font-medium text-foreground">All trials captured</p>
          </div>
          <p className="mt-1 text-xs text-muted">
            Generate the report to review results and (in the doctor flow) save
            to the patient&apos;s history.
          </p>
          <div className="mt-3 flex gap-2">
            <Button onClick={() => setPhase("done")}>
              <Play className="h-4 w-4" />
              Generate report
            </Button>
            <Button variant="ghost" onClick={reset}>
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}

      <p className="text-xs text-muted">
        Cutoffs (PDF Test C5): age &lt;60 ≥ 10 s eyes-open, age 60–69 ≥ 7 s,
        age 70+ ≥ 5 s. Eyes-closed thresholds halved per PDF guidance.
        L–R asymmetry &gt; 30% indicates targeted intervention. Sway path
        and 95% ellipse area are reported in pixels (relative units —
        suitable for trend tracking within the same patient).
      </p>
    </div>
  );
}

// Top-level condition section ("Eyes open" / "Eyes closed") with
// Left- and Right-leg buttons inside it. Re-running a captured side
// is allowed (clicking again replaces the previous result).
function ConditionGroup({
  title,
  subtitle,
  icon,
  leftDone,
  rightDone,
  onLeft,
  onRight,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  leftDone: boolean;
  rightDone: boolean;
  onLeft: () => void;
  onRight: () => void;
}) {
  return (
    <section className="rounded-card border border-border bg-surface p-5">
      <div className="flex items-center gap-2">
        {icon}
        <p className="text-sm font-semibold uppercase tracking-[0.12em] text-foreground">
          {title}
        </p>
      </div>
      <p className="mt-1 text-xs text-muted">{subtitle}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <SideButton label="Left-leg stance" done={leftDone} onClick={onLeft} />
        <SideButton label="Right-leg stance" done={rightDone} onClick={onRight} />
      </div>
    </section>
  );
}

function SideButton({
  label,
  done,
  onClick,
}: {
  label: string;
  done: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center justify-between rounded-card border p-3 text-left text-sm transition ${
        done
          ? "border-emerald-500/40 bg-emerald-500/5 hover:border-emerald-500/60"
          : "border-border bg-elevated hover:border-accent"
      }`}
    >
      <span className="flex items-center gap-2 font-medium text-foreground">
        {done && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />}
        {label}
      </span>
      <span className="text-[10px] uppercase tracking-[0.12em] text-subtle">
        {done ? "Re-run" : "Start"}
      </span>
    </button>
  );
}
