"use client";
// S1 — Shoulder Raise to Target.
//
// Mechanic: Target-Reach (lib/rehab/mechanics.ts targetReachStep +
// spawnReachTarget). Cursor is normalised [0..1] × [0..1] in CSS
// coordinates (y=0 top, y=1 bottom).
//
// Mapping — the shared clinical metric IS the game control:
//   cursor.y ← shoulder elevation ANGLE
//             higher angle (arm up) ⇒ lower y (cursor near top)
//             via cursor.y = 1 − clamp(angle / MAX_RAISE, 0, 1)
//   cursor.x ← test-side wrist x, normalised against video width
//             and MIRRORED to match the selfie-mirror skeleton view
//             so the on-screen cursor moves the same direction as
//             the patient's arm in the camera feed.
//
// v1 movement: ABDUCTION (frontal view, arm raises to the side).
// Targets spawn randomly within [0.15, 0.85] (the shell's default
// behaviour) — hitting the highest targets requires patient to
// raise the arm to ~MAX_RAISE_ANGLE.
//
// Reuses (no modifications):
//   • RehabCameraShell, TargetReachShell, targetReachStep,
//     spawnReachTarget — rehab mechanic library
//   • computeShoulderAngle — lib/biomech/shoulder-live.ts (zero
//     modification — imported as-is)
//   • usePoseDetectionLive, useCamera (via RehabCameraShell)
//   • usePatientContext for ?patientId doctor flow

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
import { computeShoulderAngle } from "@/lib/biomech/shoulder-live";
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

// Maximum raise angle used to normalise the cursor. 160° lets
// patients with slightly restricted ROM still reach the top
// targets, and avoids requiring terminal-range strain at 180°.
// Patients with normal ROM cap out the cursor before maxing the
// shoulder; this is intentional — game targets clip slightly
// below the play-area edges so we never need cursor.y = 0.
const MAX_RAISE_ANGLE_DEG = 160;

// Rep-Count on the shoulder abduction angle (° — high when the arm is
// raised). One rep = raise arm to target → lower back down.
// topThreshold = "reached" (raised near/above horizontal),
// depthThreshold = "relaxed" (arm near the side). First raise primes;
// every raise-after-lower counts. Tune on camera as needed.
const REP_CONFIG = {
  topThreshold: 80,
  depthThreshold: 20,
  minAmplitude: 45,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function ShoulderRaisePage() {
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
  // EMA-smoothed abduction — kills per-frame pose jitter (phantom reps).
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
      // v1 movement: abduction (frontal view).
      const liveKp = kp as unknown as LiveKeypoint[];
      const raw = computeShoulderAngle("abduction", liveKp, side);
      if (raw === null) return;

      const prev = smoothAngleRef.current;
      const angle = prev === null ? raw : prev * 0.65 + raw * 0.35;
      smoothAngleRef.current = angle;
      setLiveAngle(angle);
      if (angle > peakAngleRef.current) {
        peakAngleRef.current = angle;
        if (angle >= 60 && lastKpRef.current) {
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
      `Peak shoulder abduction: ${peak.toFixed(0)}° (target band 90–160°).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peak,
      side,
      `Peak shoulder raise — ${peak.toFixed(0)}° abduction`,
    );
    return {
      module: "rehab" as const,
      movement: "shoulder-raise",
      side,
      metrics: {
        exercise_slug: "shoulder-raise",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        reps,
        target_reps: TARGET_REPS,
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: repStateRef.current,
        signal: {
          name: "shoulder_abduction",
          unit: "deg",
          value_at_peak: peak,
          target_band: { min: 90, max: MAX_RAISE_ANGLE_DEG },
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
              <Badge>S1 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Shoulder Raise<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Active shoulder abduction (v1) — patient repeatedly
                raises the test arm out to the side, then lowers it.
                Each raise-and-lower is one <strong>rep</strong>; the
                session auto-saves after {TARGET_REPS} reps. The shared{" "}
                <strong>shoulder elevation angle</strong> (same metric
                the assessment module reports) is the control, so the
                game IS the clinical signal.
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
              title={`Shoulder Raise · ${side === "left" ? "Left" : "Right"} arm`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Goal ${TARGET_REPS} reps`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  autoStart
                  hideControls
                  angleArc={{
                    vertex: side === "left" ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER,
                    armA: side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP,
                    armB: side === "left" ? LM.LEFT_ELBOW : LM.RIGHT_ELBOW,
                    currentDeg: liveAngle,
                    band: { min: 90, max: 160 },
                  }}
                >
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">{side === "left" ? "L" : "R"} shoulder</p>
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
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40">{side === "left" ? "Left" : "Right"} arm</span>
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["shoulder-raise"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["shoulder-raise"]} alt="Shoulder Raise reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownCard
                      countdown={countdown}
                      onSkip={skipCountdown}
                      hint="Patient facing the camera, test arm relaxed at the side."
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
                          signalLabel="Shoulder abduction (°)"
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
                Camera at shoulder height, ~2 m away, perpendicular
                to the patient — they face the camera (frontal view).
              </li>
              <li>
                Full upper body visible — shoulder, elbow, wrist, and
                hip on the test side must all stay in frame
                throughout the arm raise.
              </li>
              <li>
                Start with the arm relaxed at the side (angle ≈ 0°).
              </li>
              <li>
                Raise the arm <strong>sideways</strong> (true frontal-
                plane abduction) to <strong>≥ {REP_CONFIG.topThreshold}°</strong>,
                then lower it back near the side — that&apos;s one rep.
              </li>
              <li>
                The rep counter climbs with each clean raise-and-lower;
                after {TARGET_REPS} reps the session auto-saves.
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
      {REHAB_EXERCISE_IMAGES["shoulder-raise"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["shoulder-raise"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the test arm
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the arm the patient will raise. We track that shoulder&apos;s
        elevation angle and count a rep each time they raise to the
        target and lower.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
