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

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { TargetReachShell } from "@/components/rehab/mechanics/TargetReachShell";
import type { TargetReachState, Score as MechanicScore } from "@/lib/rehab/gameState";
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

const REACH_CONFIG = {
  hitRadiusMultiplier: 1.2,
  pointsPerHit: 10,
  pointsPerMiss: -2,
};

export default function ShoulderRaisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default cursor at bottom-centre — arm-down resting position.
  const [cursor, setCursor] = useState<{ x: number; y: number }>({
    x: 0.5,
    y: 1.0,
  });
  // Live shoulder angle for the on-camera overlay readout.
  const [liveAngle, setLiveAngle] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const reachStateRef = useRef<TargetReachState | null>(null);
  const handleReachSnapshot = useCallback((state: TargetReachState, _score: MechanicScore) => {
    reachStateRef.current = state;
  }, []);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakAngleRef = useRef<number>(0);

  // Auto-flow: side pick → 3-2-1 countdown → live. No auto-complete
  // — the Target-Reach spawner is open-ended (no finite target
  // count on this page), so the manual Save stays available.
  // Session-scoped refs reset at the live transition so countdown
  // framing noise never leaks into the payload.
  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(side !== null, () => {
    peakAngleRef.current = 0;
    bestPoseRef.current = null;
    reachStateRef.current = null;
    sessionStartRef.current = performance.now();
  });

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      // v1 movement: abduction (frontal view). The compute helper
      // is movement-keyed but pose-agnostic — same kp shape works.
      const liveKp = kp as unknown as LiveKeypoint[];
      const angle = computeShoulderAngle("abduction", liveKp, side);
      const wrist =
        liveKp[side === "right" ? LM.RIGHT_WRIST : LM.LEFT_WRIST];
      if (angle === null) return;

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
      // cursor.y from angle: arm up (high angle) → cursor.y near 0
      const yPct = Math.max(0, Math.min(1, angle / MAX_RAISE_ANGLE_DEG));
      const cursorY = 1 - yPct;
      // cursor.x from wrist — mirrored to match the skeleton view
      // (RehabCameraShell does `1 - p.x / sw` for selfie display).
      const vw = video.videoWidth;
      const cursorX =
        wrist && vw > 0
          ? Math.max(0, Math.min(1, 1 - wrist.x / vw))
          : 0.5;
      setCursor({ x: cursorX, y: cursorY });
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
        mechanic_id: "target_reach",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: reachStateRef.current,
        signal: {
          name: "shoulder_abduction",
          unit: "deg",
          value_at_peak: peak,
          target_band: { min: 90, max: MAX_RAISE_ANGLE_DEG },
        },
        config: REACH_CONFIG,
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
              <Badge>S1 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Shoulder Raise<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Active shoulder abduction (v1) to target — patient
                raises the test arm to the side and uses it to drive
                a cursor onto spawning targets. Cursor height is
                controlled by the shared{" "}
                <strong>shoulder elevation angle</strong> (same metric
                the assessment module reports), so the game control
                IS the clinical signal. Max-cursor at{" "}
                {MAX_RAISE_ANGLE_DEG}° abduction — patients with
                restricted ROM still hit mid-band targets without
                terminal-range strain. Powered by the Target-Reach
                mechanic.
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
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : "Drive cursor onto targets"}
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
                    <div className="flex min-h-0 flex-1 flex-col">
                      <TargetReachShell cursor={cursor} config={REACH_CONFIG} compact onSnapshot={handleReachSnapshot} />
                    </div>
                  )}
                  <div className="no-pdf">
                    <AutoFlowFooter
                      complete={sessionPhase === "complete"}
                      buildPayload={buildRehabPayload}
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
                Start with the arm relaxed at the side (angle ≈ 0°,
                cursor near the bottom).
              </li>
              <li>
                Raise the arm <strong>sideways</strong> (true frontal-
                plane abduction). The cursor rises as the angle
                grows. Aim for the spawning targets — each hit awards
                points, each timeout costs them.
              </li>
              <li>
                Top targets sit around{" "}
                <strong>{Math.round(MAX_RAISE_ANGLE_DEG * 0.85)}°</strong>
                {" "}abduction. Patients with restricted ROM still
                score on mid- and lower-band targets.
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
        elevation angle to drive the cursor.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
