"use client";
// K4 — Step-Up Control.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Same
// engine + same direction convention as K1 Squat — HIGH at the
// rep "top" (extended knee), LOW at "depth" (bent knee during the
// step-up phase).
//
// computeKneeAngle returns FLEXION:
//   • Fully extended knee → flexion ≈ 0°
//   • Knee bent ~90°      → flexion ≈ 90°
//
// Rep-Count needs HIGH at top → feed the INTERIOR angle:
//   interior = 180 − flexion
//     extended top of step → interior ≈ 180° (above topThreshold)
//     bent during stepping → interior ≈ 110–130° (below depth)
//
// Same flip pattern K1 already uses. NO biomech file modified —
// computeKneeAngle imported as-is.

import { Suspense, useCallback, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import { computeKneeAngle } from "@/lib/biomech/knee-live";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const STEP_UP_CONFIG = {
  // Top — fully extended at the apex of the step-up (foot planted
  // on platform, body upright). 165° tolerates slight residual
  // flexion at lockout.
  topThreshold: 165,
  // Depth — knee bent during the loading / step phase. 120°
  // captures the deepest portion without forcing terminal
  // flexion the patient may not reach.
  depthThreshold: 120,
  // 40° excursion catches real step-ups; rules out small body
  // sways or micro-shifts.
  minAmplitude: 40,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function StepUpExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 180 = fully extended (standing). Patient usually starts
  // standing with the working foot on the floor next to the
  // platform — engine transitions init → above_top quickly.
  const [interior, setInterior] = useState<number>(180);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], _v: HTMLVideoElement) => {
      if (!side) return;
      const flexion = computeKneeAngle(
        "flexion_extension",
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (flexion !== null) setInterior(180 - flexion);
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
              <Badge>K4 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Step-Up Control<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Stepping-leg knee control on a low platform. Patient
                steps up reaching full extension, lowers under
                control. The same Rep-Count engine K1 Squat uses
                gates depth and amplitude. Powered by the Rep-Count
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

          {!side ? (
            <SidePicker onPick={setSide} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-500/15 px-3 py-1 text-xs font-semibold text-indigo-200 ring-1 ring-indigo-400/40">
                  Stepping leg: {side === "left" ? "Left" : "Right"}
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
                      vertex: side === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                      armA: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                      armB: side === "left" ? LM_LIVE.LEFT_ANKLE : LM_LIVE.RIGHT_ANKLE,
                      currentDeg: interior,
                      band: {
                        min: STEP_UP_CONFIG.depthThreshold,
                        max: STEP_UP_CONFIG.topThreshold,
                      },
                    }}
                  >
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} knee
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {interior.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {interior >= STEP_UP_CONFIG.topThreshold
                          ? "extended"
                          : interior <= STEP_UP_CONFIG.depthThreshold
                          ? "loaded"
                          : "transition"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <RepCountShell
                    signal={interior}
                    signalLabel={`${side === "left" ? "Left" : "Right"} knee angle (°)`}
                    targetReps={TARGET_REPS}
                    config={STEP_UP_CONFIG}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Patient stands next to a low platform / step
                (~15–25 cm). Camera at hip height, ~2 m away,
                roughly perpendicular to the body — frontal or
                slightly lateral both work.
              </li>
              <li>
                Test-side hip, knee, and ankle must all stay in
                frame across the full step.
              </li>
              <li>
                Step up onto the platform with the working leg until
                fully upright — knee interior reaches{" "}
                <strong>≥ {STEP_UP_CONFIG.topThreshold}°</strong>{" "}
                (target ≈ 180°).
              </li>
              <li>
                Step down under control — knee interior must drop
                below <strong>{STEP_UP_CONFIG.depthThreshold}°</strong>{" "}
                during the loading phase for the rep to register.
              </li>
              <li>
                Excursion gate: lifts that don&apos;t span at least{" "}
                <strong>{STEP_UP_CONFIG.minAmplitude}°</strong> are
                flagged as shallow.
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
      {REHAB_EXERCISE_IMAGES["step-up"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["step-up"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the stepping leg
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the leg the patient will use to step up onto the
        platform. We track that knee&apos;s interior angle every
        frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
