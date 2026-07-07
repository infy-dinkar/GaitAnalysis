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
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
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

  const calibSamplesRef = useRef<CalibSample[]>([]);
  const baselineRef = useRef<Baseline | null>(null);

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakShiftRef = useRef<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const resetSession = useCallback(() => {
    calibSamplesRef.current = [];
    baselineRef.current = null;
    setCalibProgress(0);
    setShift(0);
    setStepDetected(false);
    setPhase("calibrating");
  }, []);

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
        mechanic_state: null,
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
  }, []);

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
    },
    [phase],
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

          <div className="mt-10 space-y-6">
            <div className="flex flex-wrap items-center gap-3">
              {phase === "calibrating" ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/15 px-3 py-1 text-xs font-semibold text-amber-200 ring-1 ring-amber-400/40">
                  Calibrating · stand still centred · {calibratingPct} %
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-400/40">
                  Baseline locked · play on
                </span>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={resetSession}
              >
                Recalibrate
              </Button>
            </div>

            {/* Reference image — shown during calibration prep,
                hides once the baseline locks (mirrors side-picker
                pages where the image disappears when the camera
                starts active capture). */}
            {phase === "calibrating" && REHAB_EXERCISE_IMAGES["weight-shift"] && (
              <div className="mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={REHAB_EXERCISE_IMAGES["weight-shift"]}
                  alt=""
                  aria-hidden="true"
                  loading="lazy"
                  className="block w-full object-contain"
                  style={{ maxHeight: 240 }}
                />
              </div>
            )}

            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <RehabCameraShell onFrame={handleFrame}>
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                      Lateral shift
                    </p>
                    <p className="tabular text-2xl font-semibold text-white">
                      {shift > 0 ? "+" : ""}
                      {shift.toFixed(2)}
                    </p>
                    {stepDetected && (
                      <p className="mt-1 inline-flex items-center gap-1 rounded-full bg-rose-500/30 px-1.5 py-0.5 text-[10px] font-semibold text-rose-100">
                        STEP
                      </p>
                    )}
                  </div>
                </RehabCameraShell>
              </div>

              <div>
                <WeightShiftShell
                  shift={shift}
                  stepDetected={stepDetected}
                  config={WEIGHT_SHIFT_CONFIG}
                />
              </div>
            </div>
            <div className="no-pdf mt-6">
              <RehabSessionFooter
                buildPayload={buildRehabPayload}
                label="Save rehab session"
              />
            </div>
          </div>

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
