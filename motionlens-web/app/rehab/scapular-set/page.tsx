"use client";
// S6 — Scapular Set / Row Pattern (COARSE coaching cue).
//
// PRD limitation — scapular motion is INFERRED from shoulder-
// landmark displacement only. The scapula itself is not landmarked
// in BlazePose. This is a COARSE coaching cue, NOT a precise
// scapular measurement. UI surfaces this caveat prominently.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Engine
// wants HIGH at rep top, LOW at rep depth. Here the proxy is
// already HIGH at retracted ("top of rep" = squeeze) and LOW at
// relaxed → fed directly, NO flip.
//
// Proxy math (computeScapularRetractionProxy in poseMetrics):
//   proxy = max(0, (1 − currentWidth / baselineWidth) × 100)
//   • Relaxed (baseline width):  proxy ≈ 0
//   • Mid-retraction:            proxy ≈ 3
//   • Strong retraction:         proxy ≈ 7-10
//
// Bilateral movement — no side picker. Auto-calibration phase
// (first 10 valid frames at relaxed posture) locks the baseline
// shoulder width + baseline shoulder Y for shrug detection.
// Pattern mirrors H3 Weight-Shift Balance's auto-calibration.
//
// Shrug detection (compensation coaching, NOT rep-voiding):
//   if currentShoulderY < baselineY − 0.07 × shoulderWidth
//   → shoulders rose by ≥ 7 % of body-scale → "Don't shrug" chip
//
// Reuses (no modifications):
//   • computeScapularRetractionProxy (NEW this turn) +
//     computeShoulderWidth — both in lib/rehab/poseMetrics
//   • RepCountShell, repCountStep, RehabCameraShell
//   • LM_LIVE shoulder indices
//   • usePoseDetectionLive, useCamera, usePatientContext
// NO biomech file touched.

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import {
  computeScapularRetractionProxy,
  computeShoulderWidth,
} from "@/lib/rehab/poseMetrics";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import type { RepCountState, Score } from "@/lib/rehab/gameState";
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

const CALIB_FRAMES = 10;
const SHOULDER_VIS_THRESHOLD = 0.3;
// Shrug threshold — shoulders rising by ≥ 7 % of shoulder-width
// from the calibrated baseline. Generous enough to ignore tiny
// posture micro-shifts, tight enough to catch deliberate shrug.
const SHRUG_RISE_FRACTION = 0.07;

const SCAPULAR_CONFIG = {
  // Proxy range under typical clinic geometry is ~0-10. 5 = 5 %
  // width narrowing required for the rep to clear "top".
  topThreshold: 5,
  // Back near baseline = relaxed. 2 = ≤2 % residual narrowing OK.
  depthThreshold: 2,
  // 3-point swing — distinguishes a real retract→release from
  // single-frame BlazePose noise (which is typically <1 % width).
  minAmplitude: 3,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

interface Baseline {
  shoulderWidth: number;
  shoulderY: number;
}

export default function ScapularSetExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  // Auto-calibration: starts capturing once shoulder landmarks are
  // reliably visible; flips to "playing" after CALIB_FRAMES stable
  // samples.
  const [phase, setPhase] = useState<"calibrating" | "playing">("calibrating");
  const [calibProgress, setCalibProgress] = useState(0);
  const [retractionProxy, setRetractionProxy] = useState(0);
  const [shrugDetected, setShrugDetected] = useState(false);

  const calibSamplesRef = useRef<Baseline[]>([]);
  const baselineRef = useRef<Baseline | null>(null);

  const sessionStartRef = useRef<number>(performance.now());
  const snapshotRef = useRef<{ state: RepCountState; score: Score } | null>(
    null,
  );
  const peakRetractionRef = useRef<number>(0);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);

  const { patient, isDoctorFlow } = usePatientContext();

  const resetSession = useCallback(() => {
    calibSamplesRef.current = [];
    baselineRef.current = null;
    setCalibProgress(0);
    setRetractionProxy(0);
    setShrugDetected(false);
    setPhase("calibrating");
  }, []);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const lSh = liveKp[LM.LEFT_SHOULDER];
      const rSh = liveKp[LM.RIGHT_SHOULDER];
      if (
        !lSh
        || !rSh
        || (lSh.score ?? 0) < SHOULDER_VIS_THRESHOLD
        || (rSh.score ?? 0) < SHOULDER_VIS_THRESHOLD
      ) {
        return;
      }
      const width = computeShoulderWidth(liveKp);
      if (width === null || width < 1) return;
      const shoulderY = (lSh.y + rSh.y) / 2;

      if (phase === "calibrating") {
        calibSamplesRef.current.push({ shoulderWidth: width, shoulderY });
        setCalibProgress(calibSamplesRef.current.length / CALIB_FRAMES);
        if (calibSamplesRef.current.length >= CALIB_FRAMES) {
          const samples = calibSamplesRef.current;
          const avgWidth =
            samples.reduce((s, x) => s + x.shoulderWidth, 0) / samples.length;
          const avgY =
            samples.reduce((s, x) => s + x.shoulderY, 0) / samples.length;
          baselineRef.current = { shoulderWidth: avgWidth, shoulderY: avgY };
          setPhase("playing");
        }
        return;
      }

      // phase === "playing"
      const baseline = baselineRef.current;
      if (!baseline) return;

      const proxy = computeScapularRetractionProxy(
        liveKp,
        baseline.shoulderWidth,
      );
      if (proxy !== null) {
        setRetractionProxy(proxy);
        if (proxy > peakRetractionRef.current) {
          peakRetractionRef.current = proxy;
          if (
            proxy >= SCAPULAR_CONFIG.depthThreshold
            && lastKpRef.current
          ) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: proxy,
              capturedAtMs: performance.now(),
            };
          }
        }
      }

      // Shrug = shoulders rise (Y decreases in image) beyond
      // 7 % × baseline shoulder-width.
      const rise = baseline.shoulderY - shoulderY;
      const shrugThreshold = SHRUG_RISE_FRACTION * baseline.shoulderWidth;
      setShrugDetected(rise > shrugThreshold);
    },
    [phase],
  );

  const handleSnapshot = useCallback(
    (state: RepCountState, score: Score) => {
      snapshotRef.current = { state, score };
    },
    [],
  );

  const buildRehabPayload = useCallback(() => {
    const snap = snapshotRef.current;
    const state = snap?.state ?? null;
    const score = snap?.score ?? { points: 0, streak: 0, bestStreak: 0 };
    const reps = state?.reps ?? 0;
    const goodReps = state?.goodReps ?? 0;
    const interpretation = reps > 0
      ? `${reps} scapular set${reps === 1 ? "" : "s"} completed`
        + (goodReps !== reps ? `, ${goodReps} clean` : ", all clean")
        + `. Peak retraction proxy: ${peakRetractionRef.current.toFixed(1)}%.`
      : "Session ended before any reps were counted.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peakRetractionRef.current,
      null,
      `Peak scapular retraction — ${peakRetractionRef.current.toFixed(1)}% narrowing`,
    );
    return {
      module: "rehab" as const,
      movement: "scapular-set",
      metrics: {
        exercise_slug: "scapular-set",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score,
        mechanic_state: state,
        signal: {
          name: "scapular_retraction_proxy",
          unit: "%",
          value_at_peak: peakRetractionRef.current,
          target_band: {
            min: SCAPULAR_CONFIG.depthThreshold,
            max: SCAPULAR_CONFIG.topThreshold,
          },
        },
        target_reps: TARGET_REPS,
        config: SCAPULAR_CONFIG,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, []);

  const calibratingPct = Math.round(calibProgress * 100);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>S6 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Scapular Set<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Scapular retraction / row-pattern rep counter.
                Patient stands frontal, squeezes the shoulder blades
                together, holds briefly, releases. Each squeeze =
                one rep. Powered by the Rep-Count mechanic.
              </p>
              {/* PRD-mandated coarse-cue caveat. */}
              <div className="mt-5 rounded-card border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p className="font-semibold uppercase tracking-[0.14em] text-amber-200 text-[10px]">
                  Coarse coaching cue — not a precise scapular measurement
                </p>
                <p className="mt-1 text-xs leading-relaxed">
                  The scapula itself is not landmarked. Retraction is
                  inferred from the small narrowing of shoulder-to-
                  shoulder pixel width when the blades squeeze
                  together. Useful as a session cue and within-
                  patient progress signal, NOT as a clinical scapular
                  ROM measurement.
                </p>
              </div>
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
                  Calibrating · stand relaxed · {calibratingPct} %
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-400/40">
                  Baseline locked · go
                </span>
              )}
              {shrugDetected && (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-500/20 px-3 py-1 text-xs font-semibold text-rose-200 ring-1 ring-rose-400/50">
                  Don&apos;t shrug · drop shoulders
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
                hides once baseline locks. Mirrors weight-shift
                + side-picker behavior. */}
            {phase === "calibrating" && REHAB_EXERCISE_IMAGES["scapular-set"] && (
              <div className="mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={REHAB_EXERCISE_IMAGES["scapular-set"]}
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
                      Scapular retraction · proxy
                    </p>
                    <p className="tabular text-2xl font-semibold text-white">
                      {retractionProxy.toFixed(1)}
                    </p>
                    <p className="mt-1 text-[10px] text-zinc-300">
                      {retractionProxy >= SCAPULAR_CONFIG.topThreshold
                        ? "retracted"
                        : retractionProxy <= SCAPULAR_CONFIG.depthThreshold
                        ? "relaxed"
                        : "engaging"}
                    </p>
                    <p className="mt-1 text-[9px] uppercase tracking-[0.12em] text-amber-200/80">
                      coarse cue
                    </p>
                  </div>
                </RehabCameraShell>
              </div>

              <div>
                <RepCountShell
                  signal={retractionProxy}
                  signalLabel="Scapular retraction (proxy) — coarse cue"
                  targetReps={TARGET_REPS}
                  config={SCAPULAR_CONFIG}
                  onSnapshot={handleSnapshot}
                />
              </div>
              <div className="no-pdf mt-4">
                <SaveToPatientButton
                  buildPayload={buildRehabPayload}
                  label="Save rehab session"
                />
              </div>
            </div>
          </div>

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at chest height, ~2 m away, perpendicular to
                the patient — <strong>frontal view</strong>. Both
                shoulders must stay clearly in frame throughout the
                session.
              </li>
              <li>
                Stand relaxed, neutral posture, arms at sides for
                ~1 second at the start — the system locks the
                baseline shoulder width + position during this
                window. Stay still until the green &quot;Baseline
                locked&quot; chip appears.
              </li>
              <li>
                Squeeze the <strong>shoulder blades together</strong>{" "}
                — imagine pinching a pencil between them — and hold
                briefly. The proxy climbs as the shoulder landmarks
                pull medially.
              </li>
              <li>
                Release back to relaxed. Each retract → release
                cycle closes one rep. The first retraction primes
                the engine and doesn&apos;t count; subsequent ones
                each count.
              </li>
              <li>
                Avoid <strong>shrugging</strong> (lifting the
                shoulders toward the ears) — the rose &quot;Don&apos;t
                shrug&quot; chip lights up when detected. Coaching cue
                only; the rep still counts.
              </li>
              <li>
                Need to reset (e.g. you repositioned)? Click{" "}
                <strong>Recalibrate</strong> to capture a fresh
                baseline.
              </li>
              <li className="text-amber-100/90">
                <strong>Coarse cue:</strong> the proxy reads small
                medial shoulder displacement, not true scapular
                position. Use it for within-patient progress, not
                against clinical scapular norms.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
