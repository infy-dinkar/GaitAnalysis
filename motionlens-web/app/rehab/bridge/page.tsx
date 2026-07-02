"use client";
// H4 — Bridge / Hip Extension.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Same
// engine that powers K1 Squat — HIGH at the rep "top", LOW at
// "depth". The state machine requires the value to traverse
// above_top → descending → below_depth → ascending → above_top
// for each rep to close.
//
// Direction analysis — computeHipAngle("flexion", kp, side):
//   • Returns 180 − interior(trunk, thigh) in degrees
//   • Supine knees-bent rest (hips on floor): interior ≈ 90°
//     so the helper returns ~90°
//   • Fully bridged (shoulder-hip-knee straight line): interior
//     ≈ 180° so the helper returns ~0°
//
// Signal DECREASES as the patient bridges up — the OPPOSITE of
// what the Rep-Count engine wants. Same flip K1 needed: feed the
// INTERIOR hip angle (180 − flexion) so it goes HIGH at the
// lifted/top position, LOW at the floor.
//
//   bridgeSignal = 180 − computeHipAngle("flexion", kp, side)
//
//   resting hips down ⇒ bridgeSignal ≈ 90
//   peak bridge       ⇒ bridgeSignal ≈ 180
//
// Reuses (no modifications):
//   • computeHipAngle (lib/biomech/hip-live.ts) — pure helper
//   • RepCountShell, repCountStep, RehabCameraShell — rehab library
//   • usePoseDetectionLive, useCamera, usePatientContext

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
import { computeHipAngle } from "@/lib/biomech/hip-live";
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

const BRIDGE_CONFIG = {
  // Bridge peak — interior angle near 180° (shoulder-hip-knee
  // line). Threshold is slightly below 180 to tolerate the
  // partial extension typical of real bridges.
  topThreshold: 150,
  // Bridge bottom — interior ≈ 90° at rest with knees bent.
  // Threshold sits above 90 so a partial drop still counts as
  // "depth reached".
  depthThreshold: 115,
  // Minimum excursion ~50° catches real bridges; rules out tiny
  // pelvic wiggles.
  minAmplitude: 50,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function BridgeExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 90 = resting bridge bottom (supine, knees bent). The
  // engine starts in "init" and waits for the value to cross
  // topThreshold — the patient's first bridge primes the
  // engine; subsequent bridges each count one rep.
  const [bridgeSignal, setBridgeSignal] = useState<number>(90);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const snapshotRef = useRef<{ state: RepCountState; score: Score } | null>(
    null,
  );
  const peakBridgeRef = useRef<number>(0);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      // "flexion" and "extension" run the same internal formula in
      // hip-live.ts — both return (180 − interior). Passing
      // "flexion" is just a movement label; the unsigned magnitude
      // it returns is what we invert below.
      const flexion = computeHipAngle(
        "flexion",
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (flexion !== null) {
        const bridgeValue = 180 - flexion;
        setBridgeSignal(bridgeValue);
        if (bridgeValue > peakBridgeRef.current) {
          peakBridgeRef.current = bridgeValue;
          if (
            bridgeValue >= BRIDGE_CONFIG.depthThreshold
            && lastKpRef.current
          ) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: bridgeValue,
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
    const interpretation = reps > 0
      ? `${reps} bridge${reps === 1 ? "" : "s"} completed`
        + (goodReps !== reps ? `, ${goodReps} clean` : ", all clean")
        + `. Best hip-interior peak: ${peakBridgeRef.current.toFixed(0)}°.`
      : "Session ended before any reps were counted.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peakBridgeRef.current,
      side,
      `Peak bridge — ${peakBridgeRef.current.toFixed(0)}° hip interior`,
    );
    return {
      module: "rehab" as const,
      movement: "bridge",
      side,
      metrics: {
        exercise_slug: "bridge",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score,
        mechanic_state: state,
        signal: {
          name: "hip_interior",
          unit: "deg",
          value_at_peak: peakBridgeRef.current,
          target_band: {
            min: BRIDGE_CONFIG.depthThreshold,
            max: BRIDGE_CONFIG.topThreshold,
          },
        },
        target_reps: TARGET_REPS,
        config: BRIDGE_CONFIG,
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
              <Badge>H4 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Bridge<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Supine glute bridge — lying on the back, knees bent,
                feet flat. Patient lifts hips toward a straight
                shoulder-hip-knee line, holds briefly, lowers under
                control. Each completed lift-and-lower counts as one
                rep with depth and amplitude gates. Powered by the
                Rep-Count mechanic.
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
                  Testing: {side === "left" ? "Left" : "Right"} side
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
                      vertex: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                      armA: side === "left" ? LM_LIVE.LEFT_SHOULDER : LM_LIVE.RIGHT_SHOULDER,
                      armB: side === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                      currentDeg: bridgeSignal,
                      band: {
                        min: BRIDGE_CONFIG.depthThreshold,
                        max: BRIDGE_CONFIG.topThreshold,
                      },
                    }}
                  >
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} hip
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {bridgeSignal.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {bridgeSignal >= BRIDGE_CONFIG.topThreshold
                          ? "bridged"
                          : bridgeSignal <= BRIDGE_CONFIG.depthThreshold
                          ? "resting"
                          : "transition"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <RepCountShell
                    signal={bridgeSignal}
                    signalLabel={`${side === "left" ? "Left" : "Right"} hip — interior angle (°)`}
                    targetReps={TARGET_REPS}
                    config={BRIDGE_CONFIG}
                    onSnapshot={handleSnapshot}
                  />
                </div>
              </div>

              <div className="no-pdf">
                <SaveToPatientButton
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
                Patient lies <strong>supine</strong> on the floor,
                knees bent ~90°, feet flat, shoulder-width apart.
              </li>
              <li>
                Camera at floor level, ~2 m from the patient&apos;s
                hip, perpendicular to the body line —
                <strong> lateral view</strong>. The test-side
                shoulder, hip, and knee must all stay in frame
                across the full lift.
              </li>
              <li>
                Resting position: hips on the floor, hip interior
                angle ≈ 90° (the live readout will show ~90°).
              </li>
              <li>
                Bridge: lift the pelvis until shoulder–hip–knee form
                a straight line — interior reaches{" "}
                <strong>≥ {BRIDGE_CONFIG.topThreshold}°</strong>{" "}
                (target ≈ 180°). Hold briefly, lower under control.
              </li>
              <li>
                Excursion gate: lifts that don&apos;t span at least{" "}
                <strong>{BRIDGE_CONFIG.minAmplitude}°</strong> of
                hip motion are flagged as shallow.
              </li>
              <li>
                Engine note: the patient&apos;s <em>first</em> bridge
                primes the rep state machine — subsequent bridges
                each count as one rep. Plan one extra lift if you
                want exactly {TARGET_REPS} counted.
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
      {REHAB_EXERCISE_IMAGES["bridge"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["bridge"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the test side
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the side facing the camera. We compute the hip interior
        angle (trunk vs thigh) on that side every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left side</Button>
        <Button onClick={() => onPick("right")}>Right side</Button>
      </div>
    </div>
  );
}
