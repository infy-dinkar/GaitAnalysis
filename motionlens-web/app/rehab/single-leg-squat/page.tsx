"use client";
// K6 — Single-Leg Squat.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Same
// engine + direction convention as K1 Squat — HIGH at the rep
// "top" (extended knee, standing on one leg), LOW at "depth"
// (knee bent during the single-leg descent).
//
// Single-leg squat is harder than a bilateral squat: balance load
// on one limb means the descent is shallower (less depth), the
// amplitude swing is smaller, and patients fatigue faster. Thresholds
// tightened from K1 to reflect that:
//   • depthThreshold 120 (vs K1's 110) — accept shallower descent
//   • minAmplitude   35  (vs K1's 50)  — smaller ROM is still real
//   • targetReps     8   (vs K1's 10)  — fewer reps per set
//   • pointsPerRep   12  (vs K1's 10)  — harder exercise rewards more
//
// Direction flip is identical to K1: feed interior (180 − flexion)
// so the signal is HIGH at standing.
//
// NO biomech file modified — computeKneeAngle imported as-is.

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { computeKneeAngle } from "@/lib/biomech/knee-live";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
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

type Side = "left" | "right";

const SINGLE_LEG_SQUAT_CONFIG = {
  // Top — single-leg standing position, knee close to extended.
  topThreshold: 160,
  // Depth — single-leg squat is shallower than a bilateral squat.
  // 120 accepts a clinically meaningful unipedal descent without
  // forcing the depth a balance-challenged patient can't reach.
  depthThreshold: 120,
  // 35° excursion catches real single-leg squats. Smaller than K1's
  // 50° because the unipedal ROM is reduced by design.
  minAmplitude: 35,
  maxJerk: null as number | null,
  // Harder exercise (balance + unilateral load) → reward more per rep.
  pointsPerRep: 12,
};
const TARGET_REPS = 8;

export default function SingleLegSquatExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 180 = fully extended (single-leg standing). Engine
  // transitions init → above_top immediately on first valid frame.
  const [interior, setInterior] = useState<number>(180);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const snapshotRef = useRef<{ state: RepCountState; score: Score } | null>(
    null,
  );
  const peakInteriorRef = useRef<number>(180);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const flexion = computeKneeAngle(
        "flexion_extension",
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (flexion !== null) {
        const interiorAngle = 180 - flexion;
        setInterior(interiorAngle);
        if (interiorAngle < peakInteriorRef.current) {
          peakInteriorRef.current = interiorAngle;
          if (
            interiorAngle <= SINGLE_LEG_SQUAT_CONFIG.topThreshold
            && lastKpRef.current
          ) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: interiorAngle,
              capturedAtMs: performance.now(),
            };
          }
        }
      }
    },
    [side],
  );

  const handleSnapshot = useCallback(
    (state: RepCountState, score: Score) => {
      snapshotRef.current = { state, score };
    },
    [],
  );

  const buildRehabPayload = useCallback((supervised: boolean) => {
    if (!side) return null;
    const snap = snapshotRef.current;
    const state = snap?.state ?? null;
    const score = snap?.score ?? { points: 0, streak: 0, bestStreak: 0 };
    const reps = state?.reps ?? 0;
    const goodReps = state?.goodReps ?? 0;
    const interpretation = reps > 0
      ? `${reps} single-leg squat${reps === 1 ? "" : "s"} completed`
        + (goodReps !== reps ? `, ${goodReps} clean` : ", all clean")
        + `. Deepest knee interior: ${peakInteriorRef.current.toFixed(0)}°.`
      : "Session ended before any reps were counted.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peakInteriorRef.current,
      side,
      `Deepest single-leg squat — ${peakInteriorRef.current.toFixed(0)}° knee interior`,
    );
    return {
      module: "rehab" as const,
      movement: "single-leg-squat",
      side,
      metrics: {
        exercise_slug: "single-leg-squat",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score,
        mechanic_state: state,
        signal: {
          name: "knee_interior",
          unit: "deg",
          value_at_peak: peakInteriorRef.current,
          target_band: {
            min: SINGLE_LEG_SQUAT_CONFIG.depthThreshold,
            max: SINGLE_LEG_SQUAT_CONFIG.topThreshold,
          },
        },
        target_reps: TARGET_REPS,
        config: SINGLE_LEG_SQUAT_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        supervised,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [side]);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>K6 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Single-Leg Squat<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Unipedal squat — patient stands on the working leg,
                performs a controlled descent, returns to standing.
                Reduced amplitude vs the bilateral squat (35° gate
                instead of 50°), fewer reps per set (8 vs 10), more
                points per rep. Powered by the Rep-Count mechanic.
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

          {!side ? (
            <SidePicker onPick={setSide} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-500/15 px-3 py-1 text-xs font-semibold text-indigo-200 ring-1 ring-indigo-400/40">
                  Standing leg: {side === "left" ? "Left" : "Right"}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSide(null)}
                >
                  Change side
                </Button>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  <RehabCameraShell
                    onFrame={handleFrame}
                    angleArc={{
                      vertex: side === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                      armA: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                      armB: side === "left" ? LM_LIVE.LEFT_ANKLE : LM_LIVE.RIGHT_ANKLE,
                      currentDeg: interior,
                      band: {
                        min: SINGLE_LEG_SQUAT_CONFIG.depthThreshold,
                        max: SINGLE_LEG_SQUAT_CONFIG.topThreshold,
                      },
                    }}
                  >
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} knee
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {interior.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {interior >= SINGLE_LEG_SQUAT_CONFIG.topThreshold
                          ? "standing"
                          : interior <= SINGLE_LEG_SQUAT_CONFIG.depthThreshold
                          ? "squat"
                          : "transition"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <RepCountShell
                    signal={interior}
                    signalLabel={`${side === "left" ? "Left" : "Right"} knee angle (°)`}
                    targetReps={TARGET_REPS}
                    config={SINGLE_LEG_SQUAT_CONFIG}
                    onSnapshot={handleSnapshot}
                  />
                </div>
              </div>

              <div className="no-pdf">
                <RehabSessionFooter
                  buildPayload={buildRehabPayload}
                  label="Save rehab session"
                />
              </div>
            </div>
          )}

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2 m away, perpendicular to
                the body line — <strong>lateral view</strong>, test
                leg toward the camera.
              </li>
              <li>
                Keep a wall, chair, or counter within arm&apos;s
                reach for safety — the patient may need to touch
                lightly if balance fails. Light fingertip contact
                is fine; full-weight grab invalidates the rep.
              </li>
              <li>
                Stand on the working leg. The other foot floats
                free (lifted forward / behind — choose what
                stabilises). Hip, knee, and ankle on the test side
                must stay clearly in frame.
              </li>
              <li>
                Descend slowly until the knee interior reaches{" "}
                <strong>≤ {SINGLE_LEG_SQUAT_CONFIG.depthThreshold}°</strong>{" "}
                (status pill: &quot;squat&quot;).
              </li>
              <li>
                Return to standing — interior climbs back over{" "}
                <strong>{SINGLE_LEG_SQUAT_CONFIG.topThreshold}°</strong>{" "}
                — and the rep closes.
              </li>
              <li>
                Excursion gate: descents shorter than{" "}
                <strong>{SINGLE_LEG_SQUAT_CONFIG.minAmplitude}°</strong>{" "}
                of knee motion are flagged as shallow.
              </li>
              <li>
                Set target: {TARGET_REPS} reps. Stop earlier if
                balance fails or form breaks — quality over quantity.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function SidePicker({ onPick }: { onPick: (s: Side) => void }) {
  return (
    <div className="mt-10 max-w-xl">
      {REHAB_EXERCISE_IMAGES["single-leg-squat"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["single-leg-squat"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the standing leg
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the leg the patient will stand and squat on. We track
        that knee&apos;s interior angle every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
