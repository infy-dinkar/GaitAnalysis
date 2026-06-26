"use client";
// H6 — Lateral Step / Side-Step.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Same
// engine + same direction convention as K1 Squat — HIGH at the
// rep "top" (working knee extended), LOW at "depth" (knee bent
// during the side-step load).
//
// Lateral stepping keeps a steadier low squat than K4 step-up:
// the patient is in a quarter-squat throughout, so the
// flexion/extension swing per rep is smaller. Thresholds shift up
// (less ROM headroom) and minAmplitude drops to 30°.
//
// Direction flip is identical to K1 / K4 / Bridge — feed
// interior (180 − flexion) so the signal is HIGH at extension.
//
// NO biomech file modified — computeKneeAngle imported as-is.

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
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const LATERAL_STEP_CONFIG = {
  // Top — quarter-squat stance with the working leg extended as
  // the foot plants out. Less terminal extension than a step-up;
  // 160° tolerates the natural slight bend held throughout the
  // drill.
  topThreshold: 160,
  // Depth — knee bent during the lateral load. 125° captures the
  // step-load phase without forcing a deep squat (which the drill
  // doesn't ask for).
  depthThreshold: 125,
  // 30° excursion — smaller than step-up because the lateral-step
  // ROM is shallower by design. Still big enough to reject body
  // sway.
  minAmplitude: 30,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function LateralStepExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
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
              <Badge>H6 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Lateral Step<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Side-stepping drill in a maintained quarter-squat
                stance — patient steps sideways with the working
                leg, lands in a controlled load, returns to start.
                Working-knee interior drives the same Rep-Count
                engine K1 uses; tighter amplitude gate (30°)
                reflects the shallower ROM of the drill. Powered by
                the Rep-Count mechanic.
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
                  <RehabCameraShell onFrame={handleFrame}>
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} knee
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {interior.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {interior >= LATERAL_STEP_CONFIG.topThreshold
                          ? "extended"
                          : interior <= LATERAL_STEP_CONFIG.depthThreshold
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
                    config={LATERAL_STEP_CONFIG}
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
                Camera at hip height, ~2 m away, perpendicular to
                the patient — frontal view. Both knees and ankles
                must stay in frame across the step.
              </li>
              <li>
                Patient starts in a comfortable quarter-squat stance
                (~10–15° knee flexion), feet hip-width apart.
              </li>
              <li>
                Side-step with the working leg, planting wide and
                loading into the same quarter-squat depth on the
                other side. Knee interior drops to{" "}
                <strong>≤ {LATERAL_STEP_CONFIG.depthThreshold}°</strong>{" "}
                during load.
              </li>
              <li>
                Return to start by extending the working knee back
                to <strong>≥ {LATERAL_STEP_CONFIG.topThreshold}°</strong>{" "}
                — that closes the rep.
              </li>
              <li>
                Excursion gate: side-steps with less than{" "}
                <strong>{LATERAL_STEP_CONFIG.minAmplitude}°</strong>{" "}
                of knee motion are flagged as shallow.
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
      {REHAB_EXERCISE_IMAGES["lateral-step"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["lateral-step"]}
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
        Pick the leg the patient will side-step with. We track that
        knee&apos;s interior angle every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
