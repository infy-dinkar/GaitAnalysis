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

import { Suspense, useCallback, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { TargetReachShell } from "@/components/rehab/mechanics/TargetReachShell";
import { computeHipAbductionDeg } from "@/lib/rehab/poseMetrics";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

// Cap the cursor at 45° abduction. Top targets reachable at ~38°
// (cursor.y ≈ 0.15); patients with restricted ROM still score on
// mid-band targets. Going higher than 45° rarely happens actively
// and would just pin the cursor against the top edge.
const MAX_ABDUCTION_DEG = 45;

const REACH_CONFIG = {
  hitRadiusMultiplier: 1.25,
  pointsPerHit: 10,
  pointsPerMiss: -2,
};

export default function HipAbductionExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default cursor at bottom-centre — patient starts with the leg
  // hanging straight down (0° abduction).
  const [cursor, setCursor] = useState<{ x: number; y: number }>({
    x: 0.5,
    y: 1.0,
  });
  const [liveAngle, setLiveAngle] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const liveKp = kp as unknown as LiveKeypoint[];
      const angle = computeHipAbductionDeg(liveKp, side);
      const ankle =
        liveKp[side === "right" ? LM.RIGHT_ANKLE : LM.LEFT_ANKLE];
      if (angle === null) return;

      setLiveAngle(angle);
      const yPct = Math.max(0, Math.min(1, angle / MAX_ABDUCTION_DEG));
      const cursorY = 1 - yPct;
      const vw = video.videoWidth;
      const cursorX =
        ankle && vw > 0
          ? Math.max(0, Math.min(1, 1 - ankle.x / vw))
          : 0.5;
      setCursor({ x: cursorX, y: cursorY });
    },
    [side],
  );

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
                Standing hip abduction to target — patient stands
                with light support (hand on chair / wall) and lifts
                the working leg out to the side. The cursor height
                IS the shared{" "}
                <strong>hip abduction angle</strong>: more lift ⇒
                higher cursor ⇒ higher targets. Max-cursor at{" "}
                {MAX_ABDUCTION_DEG}° — patients with restricted ROM
                still score on mid-band targets. Powered by the
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

          {!side ? (
            <SidePicker onPick={setSide} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40">
                  Working leg: {side === "left" ? "Left" : "Right"}
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
                      vertex: side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP,
                      armA: side === "left" ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER,
                      armB: side === "left" ? LM.LEFT_KNEE : LM.RIGHT_KNEE,
                      currentDeg: liveAngle,
                      band: { min: 15, max: 40 },
                    }}
                  >
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} hip · abduction
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {liveAngle.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        cursor y {cursor.y.toFixed(2)} ·{" "}
                        {liveAngle >= 30 ? "near target ROM" : "lift higher"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <TargetReachShell cursor={cursor} config={REACH_CONFIG} />
                </div>
              </div>
            </div>
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
                Cursor height tracks the abduction angle. Top
                targets sit around{" "}
                <strong>≥ {Math.round(MAX_ABDUCTION_DEG * 0.85)}°</strong>
                {" "}of abduction (the upper end of typical clinical
                active ROM).
              </li>
              <li>
                Each successful hit awards points; misses (target TTL
                expires before the cursor reaches it) cost points.
                Hold steady at target height to glue the cursor onto
                each spawn.
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
