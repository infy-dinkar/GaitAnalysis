"use client";
// H2 — Standing Hip Abduction to Target.
//
// Mechanic: Target-Reach (lib/rehab/mechanics.ts targetReachStep +
// spawnReachTarget). Cursor is in normalised [0..1] × [0..1]
// CSS-y-down — same space TargetReachShell expects.
//
// Mapping — the shared clinical metric IS the game control:
//   abductionNorm ← clamp(angle / MAX_ABDUCTION_DEG, 0, 1)
//   cursor.y      ← 1 − abductionNorm  (leg lifts ⇒ cursor.y drops
//                                       ⇒ cursor rises toward the
//                                       upper-band targets)
//   cursor.x      ← working-side ankle x, normalised + mirrored
//                  for selfie-skeleton consistency
//
// Same direction convention as S1 Shoulder Raise (which drives
// cursor.y from shoulder elevation). Top targets sit around
// cursor.y ≈ 0.15 → abductionNorm ≈ 0.85 → angle ≈ 38° — the
// upper end of typical active hip abduction ROM (~30-45°).
//
// Reuses (no modifications):
//   • TargetReachShell, targetReachStep, spawnReachTarget,
//     RehabCameraShell — rehab mechanic library
//   • computeHipAbductionDeg — NEW pure fn in lib/rehab/poseMetrics
//     (no equivalent in biomech; hip-live.ts only models sagittal)
//   • LM_LIVE ankle indices
//   • usePoseDetectionLive, useCamera, usePatientContext

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import type { RepCountState, Score as MechanicScore } from "@/lib/rehab/gameState";
import {
  AutoFlowCompleteOverlay,
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
  AutoFlowFooter,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import { computeHipAbductionDeg } from "@/lib/rehab/poseMetrics";
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

// Rep-Count on the abduction angle (° — high when the leg is lifted to
// the side). One rep = lift to target → lower back down. topThreshold =
// "reached" (lifted), depthThreshold = "relaxed" (leg near vertical).
// First lift primes; every lift-after-lower counts. Tune on camera if
// reps under/over count (active abduction ROM ~30-45°).
const REP_CONFIG = {
  topThreshold: 28,
  depthThreshold: 10,
  minAmplitude: 15,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function HipAbductionExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [liveAngle, setLiveAngle] = useState<number>(0);
  const [reps, setReps] = useState(0);
  const [elapsedSec, setElapsedSec] = useState(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const repStateRef = useRef<RepCountState | null>(null);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakAngleRef = useRef<number>(0);
  // EMA-smoothed abduction — kills per-frame pose jitter that would
  // otherwise spike phantom reps.
  const smoothAngleRef = useRef<number | null>(null);

  // Auto-flow: side pick → 3-2-1 countdown → live → complete (at
  // TARGET_REPS) → auto-save. Session-scoped refs reset at the live
  // transition so countdown framing noise never leaks into the payload.
  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
    markComplete,
  } = useRehabAutoFlow(side !== null, () => {
    peakAngleRef.current = 0;
    bestPoseRef.current = null;
    repStateRef.current = null;
    smoothAngleRef.current = null;
    setReps(0);
    setElapsedSec(0);
    sessionStartRef.current = performance.now();
  });

  const handleSnapshot = useCallback(
    (state: RepCountState, _score: MechanicScore) => {
      repStateRef.current = state;
      setReps(state.reps);
      if (state.reps >= TARGET_REPS) markComplete();
    },
    [markComplete],
  );

  useEffect(() => {
    if (sessionPhase !== "live") return;
    const id = window.setInterval(() => setElapsedSec((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [sessionPhase]);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const raw = computeHipAbductionDeg(liveKp, side);
      if (raw === null) return;

      const prev = smoothAngleRef.current;
      const angle = prev === null ? raw : prev * 0.65 + raw * 0.35;
      smoothAngleRef.current = angle;
      setLiveAngle(angle);
      if (angle > peakAngleRef.current) {
        peakAngleRef.current = angle;
        if (angle >= 10 && lastKpRef.current) {
          bestPoseRef.current = {
            landmarks: lastKpRef.current.landmarks,
            source_frame: lastKpRef.current.source_frame,
            angle,
            capturedAtMs: performance.now(),
          };
        }
      }
    },
    [side],
  );

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const peak = peakAngleRef.current;
    const interpretation =
      `Peak hip abduction: ${peak.toFixed(0)}° (target band 15–40°).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peak,
      side,
      `Peak hip abduction — ${peak.toFixed(0)}°`,
    );
    return {
      module: "rehab" as const,
      movement: "hip-abduction",
      side,
      metrics: {
        exercise_slug: "hip-abduction",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        reps,
        target_reps: TARGET_REPS,
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: repStateRef.current,
        signal: {
          name: "hip_abduction",
          unit: "deg",
          value_at_peak: peak,
          target_band: { min: 15, max: 40 },
        },
        config: REP_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [side, reps]);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>H2 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Hip Abduction<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Standing hip abduction — patient stands with light
                support (hand on chair / wall) and repeatedly lifts the
                working leg out to the side, then lowers it. Each
                lift-and-lower is one <strong>rep</strong>; the session
                auto-saves after {TARGET_REPS} reps. The shared clinical
                metric — <strong>hip abduction angle</strong> (thigh vs
                vertical) — is the control.
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
              title={`Hip Abduction · ${side === "left" ? "Left" : "Right"} leg`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Goal ${TARGET_REPS} reps`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  autoStart
                  hideControls
                  angleArc={{
                    vertex: side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP,
                    armA: side === "left" ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER,
                    armB: side === "left" ? LM.LEFT_KNEE : LM.RIGHT_KNEE,
                    currentDeg: liveAngle,
                    band: { min: 15, max: 40 },
                  }}
                >
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">{side === "left" ? "L" : "R"} hip · abd</p>
                    <p className="tabular text-2xl font-semibold text-white">{liveAngle.toFixed(0)}°</p>
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
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40">{side === "left" ? "Left" : "Right"} leg</span>
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["hip-abduction"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["hip-abduction"]} alt="Hip Abduction reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownCard
                      countdown={countdown}
                      onSkip={skipCountdown}
                      hint="Patient facing the camera, standing with light hand support."
                    />
                  )}
                  {(sessionPhase === "live" || sessionPhase === "complete") && (
                    <>
                      <div className="flex items-center justify-between rounded-lg border border-zinc-700 bg-zinc-900/80 px-3 py-2">
                        <div>
                          <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Time</p>
                          <p className="tabular text-2xl font-semibold leading-none text-white">
                            {Math.floor(elapsedSec / 60)}:{String(elapsedSec % 60).padStart(2, "0")}
                          </p>
                        </div>
                        <p className="text-[10px] text-zinc-400">Reps auto-save at {TARGET_REPS}</p>
                      </div>
                      <div className="flex min-h-0 flex-1 flex-col">
                        <RepCountShell
                          signal={liveAngle}
                          signalLabel="Hip abduction (°)"
                          targetReps={TARGET_REPS}
                          config={REP_CONFIG}
                          onSnapshot={handleSnapshot}
                          compact
                        />
                      </div>
                    </>
                  )}
                  <div className="no-pdf">
                    <AutoFlowFooter
                      complete={sessionPhase === "complete"}
                      buildPayload={buildRehabPayload}
                      completeHint={`${TARGET_REPS} reps done — saving to record automatically.`}
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
                the patient — <strong>frontal view</strong>.
                Both hips, the working knee, and the working ankle
                must stay in frame throughout the lift.
              </li>
              <li>
                Patient stands with light support (hand on a chair
                back or wall). Stance leg stays planted; working leg
                hangs neutral to start.
              </li>
              <li>
                Lift the working leg <strong>straight out to the
                side</strong> — purely lateral motion. Avoid lifting
                the leg forward (hip flexion) or tilting the trunk —
                both inflate the angle reading without producing real
                abduction.
              </li>
              <li>
                Lift the leg to <strong>≥ {REP_CONFIG.topThreshold}°</strong>
                {" "}abduction, then lower it back near vertical — that&apos;s
                one rep. The rep counter climbs with each clean
                lift-and-lower.
              </li>
              <li>
                After {TARGET_REPS} reps the session auto-saves. A full,
                controlled lift (not a quick swing) counts best.
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
      {REHAB_EXERCISE_IMAGES["hip-abduction"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["hip-abduction"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the working leg
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the leg the patient will lift sideways. We compute that
        hip&apos;s abduction angle (thigh vs vertical) every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
