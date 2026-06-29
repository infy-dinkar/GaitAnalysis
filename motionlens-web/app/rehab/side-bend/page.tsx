"use client";
// B3 — Side Bend (Lateral Trunk Flexion).
//
// Mechanic: Target-Reach (lib/rehab/mechanics.ts targetReachStep +
// spawnReachTarget). Cursor in normalised [0..1] × [0..1] CSS-y-down
// — same space TargetReachShell expects.
//
// Bilateral movement — patient bends to either side; cursor moves
// accordingly via SIGNED math (positive angle = right bend).
//
// Cursor mapping — single signal (lateral flexion) drives both axes
// so the rectangular play area gets meaningful coverage:
//
//   angle = computeLateralTrunkFlexionDeg(kp)   // signed
//                                               // +25° = full right bend
//                                               // −25° = full left bend
//   cursor.x = 0.5 + clamp(angle / (2 × MAX), −0.45, +0.45)
//                                               // signed lateral shift
//   cursor.y = 1 − clamp(|angle| / MAX, 0, 0.85)
//                                               // magnitude → height
//                                               // neutral ⇒ y = 1.0 (bottom)
//                                               // max bend ⇒ y ≈ 0.15 (top)
//
// Trajectory: at rest cursor sits at (0.5, 1.0) bottom-centre. As
// the patient bends right, the cursor sweeps up-and-right; bend
// left, up-and-left. Targets spawning anywhere in [0.15, 0.85] ×
// [0.15, 0.85] are reachable across the natural ROM arc.
//
// Mirror correctness: the camera shell renders a selfie-mirror
// (patient's right appears on screen-right). The helper returns
// POSITIVE when the patient bends to their anatomical right, which
// in the mirrored view = screen-right = cursor.x > 0.5. Natural
// directional feel without extra negation.
//
// Reuses (no modifications):
//   • TargetReachShell, targetReachStep, spawnReachTarget,
//     RehabCameraShell
//   • computeLateralTrunkFlexionDeg — NEW pure fn in poseMetrics
//   • usePatientContext

import { Suspense, useCallback, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { TargetReachShell } from "@/components/rehab/mechanics/TargetReachShell";
import { computeLateralTrunkFlexionDeg } from "@/lib/rehab/poseMetrics";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

// 25° = upper end of typical clinical active lateral-flexion ROM.
const MAX_BEND_DEG = 25;

const REACH_CONFIG = {
  hitRadiusMultiplier: 1.3,
  pointsPerHit: 10,
  pointsPerMiss: -2,
};

export default function SideBendExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  // Simple "ready → active" gate so the reference image renders
  // before the camera engages — mirrors the side-picker pattern on
  // other rehab pages.
  const [phase, setPhase] = useState<"ready" | "active">("ready");
  const [cursor, setCursor] = useState<{ x: number; y: number }>({
    x: 0.5,
    y: 1.0,
  });
  const [liveAngle, setLiveAngle] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], _video: HTMLVideoElement) => {
      const angle = computeLateralTrunkFlexionDeg(
        kp as unknown as LiveKeypoint[],
      );
      if (angle === null) return;
      setLiveAngle(angle);
      // Signed lateral shift, clamped so the cursor stays just
      // inside the target spawn band edges (0.05–0.95 visually,
      // 0.15-0.85 reachable).
      const xOffset = Math.max(
        -0.45,
        Math.min(0.45, angle / (2 * MAX_BEND_DEG)),
      );
      const yOffset = Math.max(
        0,
        Math.min(0.85, Math.abs(angle) / MAX_BEND_DEG),
      );
      setCursor({ x: 0.5 + xOffset, y: 1 - yOffset });
    },
    [],
  );

  // Direction hint for the live readout.
  const bendSide =
    Math.abs(liveAngle) < 2
      ? "neutral"
      : liveAngle > 0
      ? "bending right"
      : "bending left";

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>B3 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Side Bend<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Lateral trunk-flexion drill — patient stands frontal,
                bends sideways to either side, drives a cursor onto
                spawning targets. Cursor.x is the shared{" "}
                <strong>signed lateral-flexion angle</strong> (left
                bends move the cursor left, right bends move it
                right); cursor.y rises with the magnitude of bend, so
                top targets demand near-max ROM. Powered by the
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

          {phase === "ready" ? (
            <ReadyGate onStart={() => setPhase("active")} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40">
                  Side bend · bilateral
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setPhase("ready")}
                >
                  Show reference
                </Button>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  <RehabCameraShell onFrame={handleFrame}>
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        Lateral flexion
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {liveAngle > 0 ? "+" : ""}
                        {liveAngle.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {bendSide}
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

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at chest height, ~2 m away, perpendicular to
                the patient — <strong>frontal view</strong>. Both
                shoulders and both hips must stay clearly in frame
                across the full bend.
              </li>
              <li>
                Patient stands neutral, feet shoulder-width, arms
                relaxed at sides. Avoid forward bending (flexion) and
                rotation — keep the motion strictly lateral.
              </li>
              <li>
                Bend gently to one side, then the other. Targets
                spawn around the play area; reaching the top corners
                requires near-max ROM (~{MAX_BEND_DEG}° bend).
                Bottom-centre targets are reachable with minimal
                lean.
              </li>
              <li>
                Each successful hit awards points; misses (target
                TTL expires before the cursor reaches it) cost
                points.
              </li>
              <li>
                Keep the pelvis level — avoid hip-hiking to
                substitute for lateral flexion (the cursor follows
                trunk-vs-pelvis tilt; hiking will not score).
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function ReadyGate({ onStart }: { onStart: () => void }) {
  return (
    <div className="mt-10 max-w-xl">
      {REHAB_EXERCISE_IMAGES["side-bend"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["side-bend"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Ready when you are
      </h2>
      <p className="mt-2 text-sm text-muted">
        Bilateral movement — bend to either side. Set up frontal to
        the camera, then begin.
      </p>
      <div className="mt-6">
        <Button onClick={onStart}>Begin</Button>
      </div>
    </div>
  );
}
