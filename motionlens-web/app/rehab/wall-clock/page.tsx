"use client";
// S2 — Wall-Clock Multidirectional Reach.
//
// Mechanic: Target-Reach (lib/rehab/mechanics.ts targetReachStep +
// spawnReachTarget). Cursor is in [0..1] × [0..1] CSS-y-down —
// same space TargetReachShell expects.
//
// Cursor source — wrist position RELATIVE to the working shoulder.
// The patient's hand IS the cursor; reaching in any direction
// moves the cursor in that direction on screen. Mirror handled to
// match the selfie-skeleton overlay:
//
//   reachX_mirrored = (shoulder.x − wrist.x)   // raw pixels
//                                              // positive ⇒ patient
//                                              // reaches their RIGHT
//   reachY          = (wrist.y − shoulder.y)   // image y-down
//                                              // positive ⇒ patient
//                                              // reaches DOWN
//   scale           = shoulderWidth × 2.5      // ~1 arm-length spans
//                                              // ~35% of play area
//   cursor.x        = clamp(0.5 + reachX_mirrored / scale, 0, 1)
//   cursor.y        = clamp(0.5 + reachY / scale,          0, 1)
//
// Targets spawn at random positions in [0.15, 0.85] × [0.15, 0.85]
// — inherently multidirectional: targets pop in all four quadrants
// around the centre, so the patient must reach in all directions
// (the "wall-clock" pattern) to score, not just one axis.
//
// Reuses (no modifications):
//   • TargetReachShell, targetReachStep, spawnReachTarget,
//     RehabCameraShell — rehab mechanic library
//   • computeShoulderWidth — existing helper in lib/rehab/poseMetrics
//     (added for H3 weight-shift; reused here as the body-scale ref)
//   • LM_LIVE wrist + shoulder indices
//   • usePoseDetectionLive, useCamera, usePatientContext
//
// NO biomech file imported or touched — wrist + shoulder landmarks
// are read directly.

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { TargetReachShell } from "@/components/rehab/mechanics/TargetReachShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { RehabStartCard } from "@/components/rehab/RehabStartCard";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import { computeShoulderWidth } from "@/lib/rehab/poseMetrics";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const LANDMARK_VIS_THRESHOLD = 0.3;
// 2.5 × shoulder-width as the reach scale means a comfortable
// extended-arm reach (~1 arm length ≈ 1.2 × shoulder-width) shifts
// the cursor about 0.34 of the play area — close to the
// target-spawn edge at 0.85. Patients with restricted ROM still
// move the cursor a useful distance; full-ROM patients reach all
// quadrant edges.
const REACH_SCALE_FACTOR = 2.5;

const REACH_CONFIG = {
  hitRadiusMultiplier: 1.25,
  pointsPerHit: 10,
  pointsPerMiss: -2,
};

export default function WallClockExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [started, setStarted] = useState(false);
  // Default centre — patient starts at neutral stance, cursor sits
  // at the play-area centre.
  const [cursor, setCursor] = useState<{ x: number; y: number }>({
    x: 0.5,
    y: 0.5,
  });
  // For the on-camera overlay readout.
  const [reachMagnitude, setReachMagnitude] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakReachRef = useRef<number>(0);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const wrist =
        liveKp[side === "right" ? LM.RIGHT_WRIST : LM.LEFT_WRIST];
      const shoulder =
        liveKp[side === "right" ? LM.RIGHT_SHOULDER : LM.LEFT_SHOULDER];
      if (
        !wrist
        || !shoulder
        || (wrist.score ?? 0) < LANDMARK_VIS_THRESHOLD
        || (shoulder.score ?? 0) < LANDMARK_VIS_THRESHOLD
      ) {
        // Sticky last position on dropouts — no cursor snap.
        return;
      }
      const sw = computeShoulderWidth(liveKp);
      if (sw === null || sw < 1) return;

      // Mirror x so patient-reaches-right ⇒ cursor-goes-right
      // (matches the selfie-skeleton overlay convention).
      const reachXMirrored = shoulder.x - wrist.x;
      const reachY = wrist.y - shoulder.y;
      const scale = sw * REACH_SCALE_FACTOR;
      const cursorX = Math.max(0, Math.min(1, 0.5 + reachXMirrored / scale));
      const cursorY = Math.max(0, Math.min(1, 0.5 + reachY / scale));
      setCursor({ x: cursorX, y: cursorY });
      // Reach magnitude normalised — quick readout for the patient.
      const magnitude = Math.hypot(reachXMirrored, reachY) / scale;
      setReachMagnitude(magnitude);
      if (magnitude > peakReachRef.current) {
        peakReachRef.current = magnitude;
        if (magnitude >= 0.2 && lastKpRef.current) {
          bestPoseRef.current = {
            landmarks: lastKpRef.current.landmarks,
            source_frame: lastKpRef.current.source_frame,
            angle: magnitude,
            capturedAtMs: performance.now(),
          };
        }
      }
    },
    [side],
  );

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const peak = peakReachRef.current;
    const interpretation =
      `Peak reach magnitude: ${peak.toFixed(2)} (of shoulder-width baseline).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peak,
      side,
      `Peak wall-clock reach — ${peak.toFixed(2)}× baseline`,
    );
    return {
      module: "rehab" as const,
      movement: "wall-clock",
      side,
      metrics: {
        exercise_slug: "wall-clock",
        mechanic_id: "target_reach",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: null,
        signal: {
          name: "reach_magnitude",
          unit: "shoulder-widths",
          value_at_peak: peak,
        },
        config: REACH_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [side]);

  // Direction hint for the overlay — helps when the patient first
  // starts moving and isn't sure if the cursor mapping feels right.
  const dirHint =
    cursor.y < 0.4
      ? cursor.x > 0.6
        ? "reaching up-right"
        : cursor.x < 0.4
        ? "reaching up-left"
        : "reaching up"
      : cursor.y > 0.6
      ? cursor.x > 0.6
        ? "reaching down-right"
        : cursor.x < 0.4
        ? "reaching down-left"
        : "reaching down"
      : cursor.x > 0.6
      ? "reaching right"
      : cursor.x < 0.4
      ? "reaching left"
      : "centre";

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>S2 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Wall-Clock Reach<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Multidirectional shoulder reach — patient stands
                frontal, the chosen hand IS the cursor. Targets
                spawn around the play area (12, 3, 6, 9 o&apos;clock
                and in-between positions), forcing reach in every
                direction. Trains shoulder ROM across the full
                hemisphere + reach-coordination. Powered by the
                Target-Reach mechanic.
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
              title={`Wall Clock · ${side === "left" ? "Left" : "Right"} arm`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : "Reach in different directions"}
              onExit={() => { setSide(null); setStarted(false); }}
              camera={(
                <RehabCameraShell onFrame={handleFrame}>
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">{side === "left" ? "L" : "R"} reach</p>
                    <p className="tabular text-xs text-zinc-300">{dirHint}</p>
                  </div>
                </RehabCameraShell>
              )}
              sidebar={(
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40">{side === "left" ? "Left" : "Right"} arm</span>
                    <Button variant="ghost" size="sm" onClick={() => { setSide(null); setStarted(false); }}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["wall-clock"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["wall-clock"]} alt="Wall Clock reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  <div className="flex min-h-0 flex-1 flex-col">
                    {started ? (
                      <TargetReachShell cursor={cursor} config={REACH_CONFIG} compact />
                    ) : (
                      <RehabStartCard onStart={() => setStarted(true)} />
                    )}
                  </div>
                  <div className="no-pdf"><RehabSessionFooter buildPayload={buildRehabPayload} label="Save session" compact /></div>
                </>
              )}
            />
          )}

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at chest height, ~2 m away, perpendicular to
                the patient — <strong>frontal view</strong>. Both
                shoulders + the working wrist must stay in frame
                throughout.
              </li>
              <li>
                Patient stands neutral, feet shoulder-width, arms at
                sides. The hand on the working side IS the cursor —
                lift the arm out of the way of the body and reach
                in different directions.
              </li>
              <li>
                Try the cardinal directions first: 12 o&apos;clock
                (straight up), 3 (right), 6 (down — across the
                body), 9 (left), then the diagonals. Each direction
                lands the cursor in a different quadrant of the
                play area.
              </li>
              <li>
                Each successful hit awards points; misses (target
                TTL expires before the cursor arrives) cost points.
                Hold near a target to glue the cursor onto it as it
                spawns nearby.
              </li>
              <li>
                Stable shoulder + wrist visibility is required —
                avoid clothing that drapes over the shoulder; keep
                the working hand visible (no pocketed thumb,
                etc.).
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
      {REHAB_EXERCISE_IMAGES["wall-clock"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["wall-clock"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the reaching arm
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the arm the patient will use to reach. The cursor
        tracks that wrist&apos;s position relative to the shoulder
        every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
