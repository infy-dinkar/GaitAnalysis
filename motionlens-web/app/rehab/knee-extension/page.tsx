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
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import { RehabStartCard } from "@/components/rehab/RehabStartCard";
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

const REACH_CONFIG = {
  // Slightly more generous than S1 — terminal extension is hard;
  // we want to reward the effort once the patient is near the
  // target line.
  hitRadiusMultiplier: 1.3,
  pointsPerHit: 10,
  pointsPerMiss: -2,
};

export default function KneeExtensionPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [started, setStarted] = useState(false);
  // Default cursor at middle-centre — the cursor will glide into
  // position naturally once the patient is in long-sit / seated
  // and the first valid frame lands.
  const [cursor, setCursor] = useState<{ x: number; y: number }>({
    x: 0.5,
    y: 0.5,
  });
  // Live knee extension (180 − flexion) for the on-camera overlay.
  const [liveExtension, setLiveExtension] = useState<number>(90);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const reachStateRef = useRef<TargetReachState | null>(null);
  const handleReachSnapshot = useCallback((state: TargetReachState, _score: MechanicScore) => {
    reachStateRef.current = state;
  }, []);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakExtensionRef = useRef<number>(0);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const flexion = computeKneeAngle("flexion_extension", liveKp, side);
      const ankle =
        liveKp[side === "right" ? LM.RIGHT_ANKLE : LM.LEFT_ANKLE];
      if (flexion === null) return;

      // Live readout uses INTERIOR (180 − flexion) which the clinic
      // is used to seeing as "extension angle" — 180° = full.
      const extension = 180 - flexion;
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

      // extensionNorm grows as the knee straightens. Higher value
      // ⇒ cursor.y closer to 0 ⇒ cursor rises toward the upper
      // target band (where the "fully extended" targets live).
      const extensionNorm = Math.max(0, Math.min(1, extension / 180));
      const cursorY = 1 - extensionNorm;
      const vw = video.videoWidth;
      const cursorX =
        ankle && vw > 0
          ? Math.max(0, Math.min(1, 1 - ankle.x / vw))
          : 0.5;
      setCursor({ x: cursorX, y: cursorY });
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
        mechanic_id: "target_reach",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: reachStateRef.current,
        signal: {
          name: "knee_extension",
          unit: "deg",
          value_at_peak: peak,
          target_band: { min: 165, max: 180 },
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
              <Badge>K3 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Terminal Knee Extension<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Active terminal-extension drill — patient seated or
                long-sitting, thigh supported, straightens the knee
                to drive a cursor onto spawning targets. Critical
                post-op (regaining the last 0–30° after ACL / TKR
                / quads inhibition). Cursor height is the shared{" "}
                <strong>knee extension angle</strong> — the game
                control IS the clinical metric. Powered by the
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
              title={`Terminal Knee Extension · ${side === "left" ? "Left" : "Right"} leg`}
              subtitle={
                isDoctorFlow && patient
                  ? `Connected to ${patient.name}'s record.`
                  : "Drive the cursor onto spawning targets."
              }
              onExit={() => { setSide(null); setStarted(false); }}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
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
                      onClick={() => { setSide(null); setStarted(false); }}
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

                  <div className="flex min-h-0 flex-1 flex-col">
                    {started ? (
                      <TargetReachShell cursor={cursor} config={REACH_CONFIG} compact onSnapshot={handleReachSnapshot} />
                    ) : (
                      <RehabStartCard onStart={() => setStarted(true)} />
                    )}
                  </div>

                  <div className="no-pdf">
                    <RehabSessionFooter
                      buildPayload={buildRehabPayload}
                      label="Save session"
                      compact
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
                Start with the knee bent (typically 30–60° flexion)
                — cursor will sit around middle of the play area.
                Live readout shows the current extension angle in
                degrees (180° = full).
              </li>
              <li>
                Actively contract the quads to <strong>straighten
                the knee</strong>. The cursor rises toward the top
                of the play area; targets spawning near the top
                require ≥ 153° extension (≤ 27° flexion) — the
                terminal-extension band.
              </li>
              <li>
                Each successful hit awards points; misses (target
                TTL expires before the cursor reaches it) cost
                points. Hold a steady extension to glue the cursor
                onto each target as it spawns.
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
        knee&apos;s flexion angle every frame and drive the cursor
        from its extension.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
