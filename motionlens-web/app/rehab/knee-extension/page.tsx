"use client";
// K3 — Terminal Knee Extension to Target.
//
// Mechanic: Target-Reach (lib/rehab/mechanics.ts targetReachStep +
// spawnReachTarget). Cursor is in [0..1] × [0..1] CSS-y-down coords
// — same space TargetReachShell expects.
//
// Mapping — the shared clinical metric IS the game control:
//   extensionNorm ← clamp((180 − flexion) / 180, 0, 1)
//   cursor.y      ← 1 − extensionNorm  (extension grows ⇒ y drops ⇒
//                                       cursor rises toward top
//                                       targets — natural for an
//                                       "extend to reach" exercise)
//   cursor.x      ← test-side ankle x, normalised + mirrored to
//                  match the selfie-skeleton overlay (keeps the
//                  feel of "cursor follows my foot")
//
// computeKneeAngle returns FLEXION:
//   • Full extension (knee straight) → flexion ≈ 0°
//   • Knee bent 90° (typical start)  → flexion ≈ 90°
//
// Top targets sit at cursor.y ≈ 0.15 → extensionNorm ≈ 0.85
// → flexion ≤ 27° (the terminal-extension band — clinically the
// hardest portion of ROM after a knee surgery / quads inhibition).
//
// Reuses (no modifications):
//   • computeKneeAngle (lib/biomech/knee-live.ts) — pure helper
//   • RehabCameraShell, TargetReachShell, targetReachStep,
//     spawnReachTarget — rehab mechanic library
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
import { computeKneeAngle } from "@/lib/biomech/knee-live";
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

// Rep-Count on the knee EXTENSION angle (° — high at terminal
// extension). One rep = relax (bent) → extend to target → relax.
// repCountStep counts a "dip" starting above topThreshold, so with an
// extension signal: topThreshold = "reached / near-terminal",
// depthThreshold = "relaxed / bent". First extension primes; every
// extend-after-relax counts. minAmplitude rejects tiny flicks.
// NOTE: thresholds are first-pass — tune on camera if reps under/over
// count (terminal extension ROM varies a lot post-op).
const REP_CONFIG = {
  // Seated terminal extension swings the knee between ~90° (bent start)
  // and ~160-170° (straight). So "reached" = ≥150°, "relaxed" = the bent
  // start. depthThreshold must sit ABOVE the natural seated bend (~90°)
  // or the cycle never completes — 60° (flexion 120°) was unreachable
  // seated, so legit reps never counted. 115° catches a normal bend.
  topThreshold: 150,
  depthThreshold: 115,
  minAmplitude: 25,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function KneeExtensionPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Live knee extension (180 − flexion) — feeds both the on-camera
  // overlay AND the rep counter (signal).
  const [liveExtension, setLiveExtension] = useState<number>(90);
  const [reps, setReps] = useState(0);
  const [elapsedSec, setElapsedSec] = useState(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const repStateRef = useRef<RepCountState | null>(null);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakExtensionRef = useRef<number>(0);
  // EMA-smoothed extension — kills per-frame pose jitter (e.g. from
  // hand/upper-body movement) that would otherwise spike the signal
  // across both thresholds and count a phantom rep.
  const smoothExtRef = useRef<number | null>(null);

  // Auto-flow: side pick → 3-2-1 countdown → live → complete (at
  // TARGET_REPS) → auto-save. Session-scoped refs reset at the live
  // transition so countdown framing noise never leaks into the payload.
  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
    markComplete,
  } = useRehabAutoFlow(side !== null, () => {
    peakExtensionRef.current = 0;
    bestPoseRef.current = null;
    repStateRef.current = null;
    smoothExtRef.current = null;
    setReps(0);
    setElapsedSec(0);
    sessionStartRef.current = performance.now();
  });

  // Auto-complete once the patient hits the rep target.
  const handleSnapshot = useCallback(
    (state: RepCountState, _score: MechanicScore) => {
      repStateRef.current = state;
      setReps(state.reps);
      if (state.reps >= TARGET_REPS) markComplete();
    },
    [markComplete],
  );

  // Elapsed session timer — ticks once/second while live, freezes on
  // complete. Shows the operator how long the set took.
  useEffect(() => {
    if (sessionPhase !== "live") return;
    const id = window.setInterval(() => {
      setElapsedSec((s) => s + 1);
    }, 1000);
    return () => window.clearInterval(id);
  }, [sessionPhase]);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const flexion = computeKneeAngle("flexion_extension", liveKp, side);
      if (flexion === null) return;

      // Live readout uses INTERIOR (180 − flexion) which the clinic
      // is used to seeing as "extension angle" — 180° = full. Smooth it
      // (EMA) before it becomes the rep-count signal so pose jitter
      // doesn't spike phantom reps.
      const rawExt = 180 - flexion;
      const prev = smoothExtRef.current;
      const extension = prev === null ? rawExt : prev * 0.65 + rawExt * 0.35;
      smoothExtRef.current = extension;
      setLiveExtension(extension);
      if (extension > peakExtensionRef.current) {
        peakExtensionRef.current = extension;
        if (extension >= 120 && lastKpRef.current) {
          bestPoseRef.current = {
            landmarks: lastKpRef.current.landmarks,
            source_frame: lastKpRef.current.source_frame,
            angle: extension,
            capturedAtMs: performance.now(),
          };
        }
      }
    },
    [side],
  );

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const peak = peakExtensionRef.current;
    const interpretation =
      `Peak knee extension: ${peak.toFixed(0)}° (target near-terminal 165–180°).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peak,
      side,
      `Peak knee extension — ${peak.toFixed(0)}°`,
    );
    return {
      module: "rehab" as const,
      movement: "knee-extension",
      side,
      metrics: {
        exercise_slug: "knee-extension",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        reps,
        target_reps: TARGET_REPS,
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: repStateRef.current,
        signal: {
          name: "knee_extension",
          unit: "deg",
          value_at_peak: peak,
          target_band: { min: 150, max: 180 },
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
              <Badge>K3 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Terminal Knee Extension<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Active terminal-extension drill — patient seated or
                long-sitting, thigh supported, repeatedly straightens
                the knee toward full extension. Critical post-op
                (regaining the last 0–30° after ACL / TKR / quads
                inhibition). Each extend-and-relax counts as a{" "}
                <strong>rep</strong>; the session auto-saves after{" "}
                {TARGET_REPS} reps. The shared clinical metric —{" "}
                <strong>knee extension angle</strong> — is the control.
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
              title={`Terminal Knee Extension · ${side === "left" ? "Left" : "Right"} leg`}
              subtitle={
                isDoctorFlow && patient
                  ? `Connected to ${patient.name}'s record.`
                  : `Goal ${TARGET_REPS} reps`
              }
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  autoStart
                  hideControls
                  angleArc={{
                    vertex: side === "left" ? LM.LEFT_KNEE : LM.RIGHT_KNEE,
                    armA: side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP,
                    armB: side === "left" ? LM.LEFT_ANKLE : LM.RIGHT_ANKLE,
                    currentDeg: liveExtension,
                    band: { min: 165, max: 180 },
                  }}
                >
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                      {side === "left" ? "L" : "R"} knee · extension
                    </p>
                    <p className="tabular text-2xl font-semibold text-white">
                      {liveExtension.toFixed(0)}°
                    </p>
                    <p className="mt-1 text-[10px] text-zinc-300">
                      {liveExtension >= 165 ? "near terminal" : "extending"}
                    </p>
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
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40">
                      {side === "left" ? "Left" : "Right"} leg
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSide(null)}
                    >
                      Change side
                    </Button>
                  </div>

                  {REHAB_EXERCISE_IMAGES["knee-extension"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={REHAB_EXERCISE_IMAGES["knee-extension"]}
                        alt="Terminal Knee Extension reference"
                        loading="lazy"
                        className="block w-full object-contain"
                        style={{ maxHeight: 140 }}
                      />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">
                        Reference form
                      </p>
                    </div>
                  )}

                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownCard
                      countdown={countdown}
                      onSkip={skipCountdown}
                      hint="Patient seated or long-sitting, test leg side-on to the camera, knee bent."
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
                        <p className="text-[10px] text-zinc-400">
                          Reps auto-save at {TARGET_REPS}
                        </p>
                      </div>
                      <div className="flex min-h-0 flex-1 flex-col">
                        <RepCountShell
                          signal={liveExtension}
                          signalLabel="Knee extension (°)"
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
                Patient seated or long-sitting with the working
                thigh supported (bolster / rolled towel under the
                knee works). Camera at knee height, ~2 m from the
                test-side, perpendicular —
                <strong> lateral view</strong>.
              </li>
              <li>
                The test-side hip, knee, and ankle must all stay
                clearly in frame across the full extension.
              </li>
              <li>
                Start with the knee bent (typically 30–60° flexion).
                The live readout shows the current extension angle in
                degrees (180° = fully straight).
              </li>
              <li>
                Actively contract the quads to <strong>straighten
                the knee</strong> toward full extension (≥ 150°),
                then relax back to the bent start — that&apos;s one rep.
              </li>
              <li>
                The rep counter climbs with each clean extend-and-relax;
                after {TARGET_REPS} reps the session auto-saves. A
                steady, full extension (not a quick flick) counts best.
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
      {REHAB_EXERCISE_IMAGES["knee-extension"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["knee-extension"]}
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
        Pick the leg the patient will actively extend. We track that
        knee&apos;s extension angle every frame and count a rep each
        time they straighten to the target and relax.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
