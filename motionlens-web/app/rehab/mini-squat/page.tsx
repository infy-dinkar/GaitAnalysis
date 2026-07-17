"use client";
// K2 — Mini-Squat.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Same
// engine + direction convention as K1 Controlled Squat — HIGH at
// the rep "top" (extended knee, standing), LOW at "depth" (knee
// bent during the partial descent).
//
// Mini-squat = shallow partial squat, lower-intensity than K1. ROM
// is smaller, more reps, looser depth gate. Patient does NOT
// descend as deep as a full squat; thresholds tightened from K1:
//   • depthThreshold 140 (vs K1's 110)  — shallow descent counts
//   • minAmplitude   25  (vs K1's 50)   — small ROM is acceptable
//   • targetReps     12  (vs K1's 10)   — more volume, less depth
//   • pointsPerRep   8   (vs K1's 10)   — easier exercise, fewer pts
//
// Direction flip identical to K1: feed interior (180 − flexion)
// so the signal is HIGH at standing.
//
// NO biomech file modified — computeKneeAngle imported as-is.

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
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

const MINI_SQUAT_CONFIG = {
  topThreshold: 165,
  depthThreshold: 140,
  minAmplitude: 25,
  maxJerk: null as number | null,
  pointsPerRep: 8,
};
const TARGET_REPS = 12;

export default function MiniSquatExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [interior, setInterior] = useState<number>(180);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const snapshotRef = useRef<{ state: RepCountState; score: Score } | null>(
    null,
  );
  const peakInteriorRef = useRef<number>(180);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);

  // Reset every session-scoped ref when the operator picks a side
  // (session start) OR clears it (Change side / Exit). Prevents
  // pre-session framing noise from carrying into the payload.
  useEffect(() => {
    peakInteriorRef.current = 180;
    bestPoseRef.current = null;
    lastKpRef.current = null;
    snapshotRef.current = null;
    if (side !== null) sessionStartRef.current = performance.now();
  }, [side]);

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
        // Self-contained tracker gate — see squat/page.tsx for the
        // full rationale (top-5° = "left the standing window", ≥40°
        // = physical sanity clamp).
        if (
          interiorAngle < MINI_SQUAT_CONFIG.topThreshold - 5
          && interiorAngle >= 40
          && interiorAngle < peakInteriorRef.current
        ) {
          peakInteriorRef.current = interiorAngle;
          if (lastKpRef.current) {
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

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const snap = snapshotRef.current;
    const state = snap?.state ?? null;
    const score = snap?.score ?? { points: 0, streak: 0, bestStreak: 0 };
    const reps = state?.reps ?? 0;
    const goodReps = state?.goodReps ?? 0;
    // Interpretation + payload use FLEXION display convention
    // (0° standing, higher = deeper). Engine feed still runs on
    // interior — see lib/rehab/kneeAngleDisplay.js.
    const captured = peakInteriorRef.current < 180;
    const deepestFlexionDeg = captured ? 180 - peakInteriorRef.current : null;
    const interpretation = captured
      ? (reps > 0
          ? `${reps} mini-squat rep${reps === 1 ? "" : "s"} completed`
            + (goodReps !== reps ? `, ${goodReps} clean` : ", all clean")
            + `. Deepest knee angle: ${deepestFlexionDeg!.toFixed(0)}°.`
          : `Deepest knee angle: ${deepestFlexionDeg!.toFixed(0)}°.`)
      : (reps > 0
          ? `${reps} mini-squat rep${reps === 1 ? "" : "s"} counted. Knee depth not captured.`
          : "Knee depth not captured.");
    // Skeleton pose: inline construction so we can persist the
    // deepest-frame angle in FLEXION convention (the report reads
    // it that way) and set angle: null on the standing-fallback
    // path — a fallback frame must never be labelled with a depth
    // number. Non-knee rehab pages still use the shared helper.
    const best = captured ? bestPoseRef.current : null;
    const fallback = lastKpRef.current;
    const skeletonPose = best
      ? {
          landmarks: best.landmarks,
          source_frame: best.source_frame,
          angle: 180 - best.angle,
          angle_convention: "flexion" as const,
          captured_at_ms: best.capturedAtMs,
          side,
          label: `Deepest mini-squat — ${deepestFlexionDeg!.toFixed(0)}° knee angle`,
        }
      : fallback
        ? {
            landmarks: fallback.landmarks,
            source_frame: fallback.source_frame,
            angle: null,
            angle_convention: "flexion" as const,
            captured_at_ms: performance.now(),
            side,
            label: "Knee depth not captured",
          }
        : null;
    return {
      module: "rehab" as const,
      movement: "mini-squat",
      side,
      metrics: {
        exercise_slug: "mini-squat",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score,
        mechanic_state: state,
        signal: {
          name: "knee_flexion",
          unit: "deg",
          // null when the tracker gate never opened this session.
          value_at_peak: captured ? deepestFlexionDeg : null,
          target_band: {
            min: 180 - MINI_SQUAT_CONFIG.topThreshold,
            max: 180 - MINI_SQUAT_CONFIG.depthThreshold,
          },
        },
        target_reps: TARGET_REPS,
        config: MINI_SQUAT_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
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
              <Badge>K2 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Mini-Squat<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Shallow partial squat — lower intensity than K1
                Controlled Squat. Patient descends only to ~40° knee
                flexion (interior {MINI_SQUAT_CONFIG.depthThreshold}°),
                returns. Same Rep-Count engine; looser depth gate +
                smaller amplitude + higher target ({TARGET_REPS} reps)
                make it suitable for early-stage / deconditioned
                patients. Powered by the Rep-Count mechanic.
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

          {!side ? <SidePicker onPick={setSide} /> : null}

          {side && (
            <LiveModeLayout
              title={`Mini-Squat · ${side === "left" ? "Left" : "Right"} leg`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Goal ${TARGET_REPS} reps`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  angleArc={{
                    vertex: side === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                    armA: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                    armB: side === "left" ? LM_LIVE.LEFT_ANKLE : LM_LIVE.RIGHT_ANKLE,
                    currentDeg: interior,
                    band: { min: MINI_SQUAT_CONFIG.depthThreshold, max: MINI_SQUAT_CONFIG.topThreshold },
                  }}
                >
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                      {side === "left" ? "L" : "R"} knee
                    </p>
                    <p className="tabular text-2xl font-semibold text-white">
                      {interior.toFixed(0)}°
                    </p>
                  </div>
                </RehabCameraShell>
              )}
              sidebar={(
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-500/15 px-3 py-1 text-xs font-semibold text-indigo-200 ring-1 ring-indigo-400/40">
                      {side === "left" ? "Left" : "Right"} leg
                    </span>
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>
                      Change side
                    </Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["mini-squat"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["mini-squat"]} alt="Mini-Squat reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  <div className="flex min-h-0 flex-1 flex-col">
                    <RepCountShell
                      signal={interior}
                      signalLabel="Knee angle (°)"
                      signalDisplayName="knee_interior"
                      targetReps={TARGET_REPS}
                      config={MINI_SQUAT_CONFIG}
                      onSnapshot={handleSnapshot}
                      compact
                    />
                  </div>
                  <div className="no-pdf">
                    <RehabSessionFooter buildPayload={buildRehabPayload} label="Save session" compact />
                  </div>
                </>
              )}
            />
          )}

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2 m away, perpendicular to
                the stance line — <strong>lateral view</strong>.
              </li>
              <li>
                Patient stands side-on, test leg toward the camera.
                Test-side hip, knee, ankle must stay in frame.
              </li>
              <li>
                Descend slowly to a <strong>shallow</strong> mini-
                squat — interior ≤{" "}
                <strong>{MINI_SQUAT_CONFIG.depthThreshold}°</strong>{" "}
                (~40° flexion). Return to standing (interior ≥{" "}
                {MINI_SQUAT_CONFIG.topThreshold}°).
              </li>
              <li>
                Excursion gate: descents under{" "}
                <strong>{MINI_SQUAT_CONFIG.minAmplitude}°</strong> of
                knee motion are flagged as shallow.
              </li>
              <li>
                Target: {TARGET_REPS} reps. Higher volume reflects
                the lower per-rep intensity vs a full squat.
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
      {REHAB_EXERCISE_IMAGES["mini-squat"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["mini-squat"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the test leg
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the leg facing the camera. We track that knee&apos;s
        interior angle every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
