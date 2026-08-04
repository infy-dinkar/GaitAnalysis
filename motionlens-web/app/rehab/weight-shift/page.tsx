"use client";
// H3 — Weight-Shift Limits of Stability.
//
// Mechanic: Weight-Shift (lib/rehab/mechanics.ts weightShiftStep).
// The shell takes two driven props each frame:
//   • shift : number in [-1, +1]      — medio-lateral cursor position
//   • stepDetected : boolean          — true when the feet have left
//                                        the calibrated stance
// The shell renders a horizontal track of zones; capturing all
// zones (dwell inside each for its required duration) without
// stepping wins the round. A step auto-pauses dwell accumulation
// regardless of cursor position.
//
// Signals derived inside this page (no biomech file touched):
//   • shift  ← (mirrored hip-mid x − baseline) / scale
//              scale = 0.4 × shoulder-width (normalised)
//              mirrored so patient-shifts-right → cursor-goes-right,
//              consistent with the selfie-mirror skeleton view
//   • stepDetected ← either ankle.y RISE above baseline > 2 % of
//                    frame height, OR ankle.x DRIFT from baseline
//                    > 3 % of frame width
//
// Brief auto-calibration phase captures both baselines from the
// first ~10 valid frames after the patient is in stance — no
// explicit click required, just stand still centred.
//
// Reuses (no modifications):
//   • RehabCameraShell, WeightShiftShell, weightShiftStep
//   • computeHipMidX, computeShoulderWidth (NEW in
//     lib/rehab/poseMetrics.ts — biomech untouched)
//   • LM_LIVE ankle indices
//   • usePoseDetectionLive, useCamera, usePatientContext

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { WeightShiftShell } from "@/components/rehab/mechanics/WeightShiftShell";
import type { WeightShiftState, Score as MechanicScore } from "@/lib/rehab/gameState";
import {
  AutoFlowCompleteOverlay,
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
  AutoFlowFooter,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import {
  computeHipMidX,
  computeShoulderWidth,
} from "@/lib/rehab/poseMetrics";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

const CALIB_FRAMES = 10;
const ANKLE_VIS_THRESHOLD = 0.3;
const STEP_Y_THRESH_NORM = 0.02;   // 2 % frame height ≈ 15 px on 720p
const STEP_X_THRESH_NORM = 0.03;   // 3 % frame width
// Patient must shift the hip-mid by 40 % of shoulder-width to push
// the cursor all the way to ±1 — a clinically meaningful weight
// shift, but reachable without stepping.
const SHIFT_SCALE_FRACTION_OF_SHOULDER_WIDTH = 0.4;

// Four-zone limits-of-stability layout — skip centre so the
// patient MUST shift to capture each. Far zones at ±0.8 are
// LoS-extreme; ±0.4 are mid-range.
const WEIGHT_SHIFT_CONFIG = {
  zones: [
    { id: "far_left",  centre: -0.8, halfWidth: 0.15, dwellMs: 1500 },
    { id: "left",      centre: -0.4, halfWidth: 0.15, dwellMs: 1500 },
    { id: "right",     centre:  0.4, halfWidth: 0.15, dwellMs: 1500 },
    { id: "far_right", centre:  0.8, halfWidth: 0.15, dwellMs: 1500 },
  ],
  pointsPerCapture: 25,
  // Stepping auto-pauses dwell accumulation inside the engine —
  // no need for a continuous per-ms penalty on top.
  stepPausePenaltyPerMs: 0,
};

// ── Rep counting (full left↔right cycle) ─────────────────────────────
// One rep = the patient has reached BOTH a left and a right extreme
// since the last rep (order-independent). The session auto-completes
// (→ auto-save) after REP_TARGET full cycles. Independent of the
// zone-dwell visual: reaching a side counts even without holding.
const REP_TARGET = 5;
const REP_REACH = 0.4; // |lateral shift| that counts as "reached a side"

interface Baseline {
  hipMidNormX: number;
  shoulderWidthNorm: number;
  lAnkleNormX: number;
  lAnkleNormY: number;
  rAnkleNormX: number;
  rAnkleNormY: number;
}

interface CalibSample {
  hipMidNormX: number;
  shoulderWidthNorm: number;
  lAnkleNormX: number;
  lAnkleNormY: number;
  rAnkleNormX: number;
  rAnkleNormY: number;
}

function medianBy(arr: CalibSample[], key: keyof CalibSample): number {
  const vals = arr.map((s) => s[key]).sort((a, b) => a - b);
  const mid = Math.floor(vals.length / 2);
  return vals.length % 2 === 0 ? (vals[mid - 1] + vals[mid]) / 2 : vals[mid];
}

export default function WeightShiftPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  // Auto-calibration: starts capturing as soon as the camera + hip
  // + ankles are reliably visible; flips to "playing" after 10
  // stable samples.
  const [phase, setPhase] = useState<"calibrating" | "playing">("calibrating");
  const [calibProgress, setCalibProgress] = useState(0);
  const [shift, setShift] = useState(0);
  const [stepDetected, setStepDetected] = useState(false);
  // Full left↔right cycle rep count (0..REP_TARGET). Drives the rep
  // indicator + the 5-rep auto-complete → auto-save.
  const [reps, setReps] = useState(0);

  const calibSamplesRef = useRef<CalibSample[]>([]);
  const baselineRef = useRef<Baseline | null>(null);

  const sessionStartRef = useRef<number>(performance.now());
  const weightShiftStateRef = useRef<WeightShiftState | null>(null);
  const maxAbsShiftRef = useRef<number>(0);
  const stepCountRef = useRef<number>(0);
  const prevStepDetectedRef = useRef<boolean>(false);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakShiftRef = useRef<number>(0);
  // Which extremes the patient has reached since the last counted rep.
  const repVisitedRef = useRef<{ left: boolean; right: boolean }>({ left: false, right: false });
  // Authoritative rep count (ref) so the frame loop can read/advance it
  // without a stale closure; `reps` state mirrors it for display.
  const repsCountRef = useRef(0);

  // Auto-flow: baseline lock → 3-2-1 countdown → live → complete →
  // auto-save. Session-scoped refs reset at the live transition so
  // cursor motion during the countdown never counts toward the
  // payload's trackers or duration.
  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
    markComplete,
  } = useRehabAutoFlow(phase === "playing", () => {
    weightShiftStateRef.current = null;
    maxAbsShiftRef.current = 0;
    stepCountRef.current = 0;
    prevStepDetectedRef.current = false;
    bestPoseRef.current = null;
    peakShiftRef.current = 0;
    sessionStartRef.current = performance.now();
    // Reset the rep counter at the countdown→live edge so nothing
    // done during the countdown carries over.
    repVisitedRef.current = { left: false, right: false };
    repsCountRef.current = 0;
    setReps(0);
  });

  const handleWSShapshot = useCallback((state: WeightShiftState, _score: MechanicScore) => {
    // Harvest mechanic state for the saved payload. Completion is now
    // rep-based (see the frame handler), so no zone-count auto-complete
    // here — the zone track stays purely as the movement visual.
    weightShiftStateRef.current = state;
    const abs = Math.abs(state.cursor);
    if (abs > maxAbsShiftRef.current) maxAbsShiftRef.current = abs;
  }, []);

  const { patient, isDoctorFlow } = usePatientContext();
  // Setup-landing gate: the exercise is entered from a setup page with a
  // reference image + camera-setup help + Start button (matches every
  // other rehab exercise, e.g. Hip Abduction). The fullscreen live view
  // renders only once started; the Exit (X) collapses back to this
  // landing. The Start click is also the user gesture that lets the
  // camera autoplay policy pass.
  const [started, setStarted] = useState(false);

  const resetSession = useCallback(() => {
    calibSamplesRef.current = [];
    baselineRef.current = null;
    setCalibProgress(0);
    setShift(0);
    setStepDetected(false);
    setPhase("calibrating");
  }, []);

  // Fullscreen Exit (X): collapse back to the setup landing + reset so
  // the next start is fresh. LiveModeLayout unmounts → RehabCameraShell's
  // useCamera cleanup stops all tracks, releasing the camera (nothing
  // orphans). resetSession also stays wired to the Recalibrate button.
  const handleExit = useCallback(() => {
    resetSession();
    setStarted(false);
  }, [resetSession]);

  const buildRehabPayload = useCallback(() => {
    const peak = peakShiftRef.current;
    const interpretation =
      `Weight-shift session — peak medio-lateral shift ${peak.toFixed(2)} (of ±1 scale).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peak,
      null,
      `Peak weight shift — ${peak.toFixed(2)}× baseline`,
    );
    return {
      module: "rehab" as const,
      movement: "weight-shift",
      metrics: {
        exercise_slug: "weight-shift",
        mechanic_id: "weight_shift",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: weightShiftStateRef.current
          ? {
              reps,
              repTarget: REP_TARGET,
              zonesCaptured: weightShiftStateRef.current.capturedZoneIds.length,
              totalZones: WEIGHT_SHIFT_CONFIG.zones.length,
              capturedZoneIds: weightShiftStateRef.current.capturedZoneIds,
              maxExcursion: maxAbsShiftRef.current,
              stepPausedMs: weightShiftStateRef.current.stepPausedMs,
              stepCount: stepCountRef.current,
            }
          : null,
        signal: {
          name: "ml_shift",
          unit: "normalised",
          value_at_peak: peak,
        },
        config: WEIGHT_SHIFT_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [reps]);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw <= 0 || vh <= 0) return;

      const snap = kpToPoseSnapshot(kp, vw, vh);
      if (snap) lastKpRef.current = snap;

      const liveKp = kp as unknown as LiveKeypoint[];
      const hipMidPx = computeHipMidX(liveKp);
      const shoulderWidthPx = computeShoulderWidth(liveKp);
      const lAnkle = liveKp[LM.LEFT_ANKLE];
      const rAnkle = liveKp[LM.RIGHT_ANKLE];

      if (
        hipMidPx === null
        || shoulderWidthPx === null
        || !lAnkle
        || !rAnkle
        || (lAnkle.score ?? 0) < ANKLE_VIS_THRESHOLD
        || (rAnkle.score ?? 0) < ANKLE_VIS_THRESHOLD
      ) {
        return;
      }

      // Mirror x for hip / ankles so the cursor + step-checks
      // align with the selfie-mirror skeleton overlay.
      const hipMidNormX = 1 - hipMidPx / vw;
      const shoulderWidthNorm = shoulderWidthPx / vw;
      const lAnkleNormX = 1 - lAnkle.x / vw;
      const lAnkleNormY = lAnkle.y / vh;
      const rAnkleNormX = 1 - rAnkle.x / vw;
      const rAnkleNormY = rAnkle.y / vh;

      if (phase === "calibrating") {
        calibSamplesRef.current.push({
          hipMidNormX,
          shoulderWidthNorm,
          lAnkleNormX,
          lAnkleNormY,
          rAnkleNormX,
          rAnkleNormY,
        });
        setCalibProgress(calibSamplesRef.current.length / CALIB_FRAMES);
        if (calibSamplesRef.current.length >= CALIB_FRAMES) {
          const samples = calibSamplesRef.current;
          baselineRef.current = {
            hipMidNormX:       medianBy(samples, "hipMidNormX"),
            shoulderWidthNorm: medianBy(samples, "shoulderWidthNorm"),
            lAnkleNormX:       medianBy(samples, "lAnkleNormX"),
            lAnkleNormY:       medianBy(samples, "lAnkleNormY"),
            rAnkleNormX:       medianBy(samples, "rAnkleNormX"),
            rAnkleNormY:       medianBy(samples, "rAnkleNormY"),
          };
          setPhase("playing");
        }
        return;
      }

      // phase === "playing"
      const baseline = baselineRef.current;
      if (!baseline) return;

      const scale =
        Math.max(0.01, baseline.shoulderWidthNorm)
        * SHIFT_SCALE_FRACTION_OF_SHOULDER_WIDTH;
      const rawShift = (hipMidNormX - baseline.hipMidNormX) / scale;
      // Clamp a bit beyond ±1 so the marker still travels visibly
      // when the patient over-shoots a zone — the engine will clip
      // its zone-membership test internally.
      const clampedShift = Math.max(-1.2, Math.min(1.2, rawShift));
      setShift(clampedShift);

      // ── Full left↔right cycle rep counter ───────────────────────
      // Mark which extreme the patient has reached; once BOTH sides
      // have been touched, that's one full cycle → +1 rep. Reaching
      // REP_TARGET auto-completes the session (→ auto-save). Only
      // meaningful in live (the reset zeroes it at the live edge; five
      // full cycles can't fit in the 3 s countdown).
      if (clampedShift <= -REP_REACH) repVisitedRef.current.left = true;
      if (clampedShift >= REP_REACH) repVisitedRef.current.right = true;
      if (
        repVisitedRef.current.left
        && repVisitedRef.current.right
        && repsCountRef.current < REP_TARGET
      ) {
        repVisitedRef.current = { left: false, right: false };
        repsCountRef.current += 1;
        setReps(repsCountRef.current);
        if (repsCountRef.current >= REP_TARGET) markComplete();
      }

      const absShift = Math.abs(clampedShift);
      if (absShift > peakShiftRef.current) {
        peakShiftRef.current = absShift;
        if (absShift >= 0.3 && lastKpRef.current) {
          bestPoseRef.current = {
            landmarks: lastKpRef.current.landmarks,
            source_frame: lastKpRef.current.source_frame,
            angle: absShift,
            capturedAtMs: performance.now(),
          };
        }
      }

      // Step detection — either foot lifting OR drifting laterally.
      // y-rise is positive when the ankle goes UP in image (lower
      // y value), so we test `baseline - current > thresh`.
      const lAnkleYRise = baseline.lAnkleNormY - lAnkleNormY;
      const rAnkleYRise = baseline.rAnkleNormY - rAnkleNormY;
      const lAnkleXDrift = Math.abs(lAnkleNormX - baseline.lAnkleNormX);
      const rAnkleXDrift = Math.abs(rAnkleNormX - baseline.rAnkleNormX);
      const stepped =
        lAnkleYRise > STEP_Y_THRESH_NORM
        || rAnkleYRise > STEP_Y_THRESH_NORM
        || lAnkleXDrift > STEP_X_THRESH_NORM
        || rAnkleXDrift > STEP_X_THRESH_NORM;
      setStepDetected(stepped);
      // Rising-edge count — increment once per new step, not every frame.
      if (stepped && !prevStepDetectedRef.current) {
        stepCountRef.current += 1;
      }
      prevStepDetectedRef.current = stepped;
    },
    [phase, markComplete],
  );

  const calibratingPct = Math.round(calibProgress * 100);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>H3 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Weight-Shift Balance<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Limits-of-stability training. Patient stands feet
                fixed, shifts weight medio-laterally to drive a
                cursor through 4 target zones (±0.4 and ±0.8). A
                step or lateral foot drift auto-pauses dwell — only
                honest weight shifts count. Powered by the
                Weight-Shift mechanic.
              </p>
              {isDoctorFlow && patient && (
                <p className="mt-3 text-xs text-muted">
                  Connected to{" "}
                  <span className="font-semibold text-foreground">
                    {patient.name}
                  </span>
                  &apos;s record.
                </p>
              )}
            </div>
            <Link href="/rehab">
              <Button variant="ghost" size="sm">← Catalogue</Button>
            </Link>
          </div>

          {/* Setup landing — reference form + Start gate. The fullscreen
              live view (below) only mounts after Start, so the camera
              never opens until the patient is ready. */}
          {!started && (
            <div className="mt-10 max-w-md">
              {REHAB_EXERCISE_IMAGES["weight-shift"] && (
                <div className="overflow-hidden rounded-card border border-border bg-white">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={REHAB_EXERCISE_IMAGES["weight-shift"]}
                    alt="Weight Shift reference"
                    loading="lazy"
                    className="block w-full object-contain"
                    style={{ maxHeight: 260 }}
                  />
                  <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">
                    Reference form
                  </p>
                </div>
              )}
              <h2 className="mt-8 text-xl font-semibold tracking-tight">
                Get set up
              </h2>
              <p className="mt-1 text-sm text-muted">
                Stand centred and face the camera, feet flat and
                shoulder-width apart (use light support if you need it).
                When you start, a 3-2-1 countdown runs and the baseline
                locks — then shift your weight fully left, then fully
                right (one full swing = 1 rep). Do {REP_TARGET} reps and
                it saves automatically.
              </p>
              <Button className="mt-4" onClick={() => setStarted(true)}>
                Start exercise
              </Button>
            </div>
          )}

          {started && (
          <LiveModeLayout
            title="Weight Shift"
            subtitle={
              phase === "calibrating"
                ? `Calibrating · ${calibratingPct}%`
                : isDoctorFlow && patient
                  ? `Connected to ${patient.name}'s record.`
                  : "Baseline locked · play on"
            }
            onExit={handleExit}
            camera={(
              <RehabCameraShell onFrame={handleFrame} autoStart hideControls>
                <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">Lateral shift</p>
                  <p className="tabular text-2xl font-semibold text-white">{shift > 0 ? "+" : ""}{shift.toFixed(2)}</p>
                  {stepDetected && (
                    <p className="mt-1 inline-flex items-center gap-1 rounded-full bg-rose-500/30 px-1.5 py-0.5 text-[10px] font-semibold text-rose-100">STEP</p>
                  )}
                </div>
                {sessionPhase === "countdown" && countdown !== null && (
                  <AutoFlowCountdownOverlay countdown={countdown} />
                )}
                {sessionPhase === "complete" && <AutoFlowCompleteOverlay />}
              </RehabCameraShell>
            )}
            sidebar={(
              <>
                <div className="flex flex-wrap items-center gap-2">
                  {phase === "calibrating" ? (
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/15 px-3 py-1 text-xs font-semibold text-amber-200 ring-1 ring-amber-400/40">Calibrating · {calibratingPct}%</span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-400/40">Locked · go</span>
                  )}
                  <Button variant="ghost" size="sm" onClick={resetSession}>Recalibrate</Button>
                </div>
                {REHAB_EXERCISE_IMAGES["weight-shift"] && (
                  <div className="overflow-hidden rounded-md border border-border bg-white">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={REHAB_EXERCISE_IMAGES["weight-shift"]} alt="Weight Shift reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                    <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                  </div>
                )}
                {sessionPhase === "countdown" && countdown !== null && (
                  <AutoFlowCountdownCard
                    countdown={countdown}
                    onSkip={skipCountdown}
                    hint="Stand centred and face the camera, feet flat and shoulder-width apart. When it starts, slowly shift your weight from one foot to the other — side to side — without lifting either foot."
                  />
                )}
                {(sessionPhase === "countdown" || sessionPhase === "live") && (
                  <div className="rounded-md border border-border bg-surface p-3 text-xs text-muted">
                    <p className="font-semibold text-foreground">How to do it</p>
                    <p className="mt-1 leading-relaxed">
                      Keep both feet flat on the floor. Slowly shift your
                      body weight all the way onto your left foot, then all
                      the way onto your right — a full left-to-right swing
                      is <span className="font-semibold text-foreground">1 rep</span>.
                      Do {REP_TARGET} reps and it saves automatically.
                    </p>
                  </div>
                )}
                {(sessionPhase === "live" || sessionPhase === "complete") && (
                  <div className="flex min-h-0 flex-1 flex-col">
                    <WeightShiftShell shift={shift} stepDetected={stepDetected} config={WEIGHT_SHIFT_CONFIG} reps={reps} repTarget={REP_TARGET} compact onSnapshot={handleWSShapshot} />
                  </div>
                )}
                <div className="no-pdf">
                  <AutoFlowFooter
                    complete={sessionPhase === "complete"}
                    buildPayload={buildRehabPayload}
                    completeHint={`${REP_TARGET} reps done — saving to record automatically.`}
                  />
                </div>
              </>
            )}
          />
          )}

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2 m away, perpendicular to
                the patient — they face the camera (frontal view).
              </li>
              <li>
                Full body must be in frame from shoulders to ankles
                — the system tracks hip midpoint AND both ankles to
                detect stepping.
              </li>
              <li>
                Stand still and centred for ~1 second at the start
                — the system locks the baseline hip + ankle
                positions during this window. Stay stable until the
                green &quot;Baseline locked&quot; chip appears.
              </li>
              <li>
                Shift weight side-to-side without lifting either
                foot. The cursor moves with you toward the four
                target zones at ±0.4 and ±0.8.
              </li>
              <li>
                A lifted foot or a small lateral foot slide auto-
                pauses dwell and shows the STEP indicator — only
                honest weight shifts capture zones.
              </li>
              <li>
                Need to reset (e.g. you moved your feet between
                rounds)? Click &quot;Recalibrate&quot; to capture a
                fresh baseline.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
