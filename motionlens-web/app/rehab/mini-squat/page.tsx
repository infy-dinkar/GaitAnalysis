"use client";
// K2 — Mini-Squat.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Same
// engine + direction convention as K1 Controlled Squat — HIGH at
// the rep "top" (extended knee, standing), LOW at "depth" (knee
// bent during the partial descent).
//
// Mini-squat = shallow partial squat, lower-intensity than K1. ROM
// is smaller, more reps, looser depth gate. Patient does NOT
// descend as deep as a full squat; thresholds tightened from K1:
//   • depthThreshold 140 (vs K1's 110)  — shallow descent counts
//   • minAmplitude   25  (vs K1's 50)   — small ROM is acceptable
//   • targetReps     12  (vs K1's 10)   — more volume, less depth
//   • pointsPerRep   8   (vs K1's 10)   — easier exercise, fewer pts
//
// Direction flip identical to K1: feed interior (180 − flexion)
// so the signal is HIGH at standing.
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
import { LM_LIVE } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const MINI_SQUAT_CONFIG = {
  topThreshold: 165,
  depthThreshold: 140,
  minAmplitude: 25,
  maxJerk: null as number | null,
  pointsPerRep: 8,
};
const TARGET_REPS = 12;

export default function MiniSquatExercisePage() {
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
              <Badge>K2 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Mini-Squat<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Shallow partial squat — lower intensity than K1
                Controlled Squat. Patient descends only to ~40° knee
                flexion (interior {MINI_SQUAT_CONFIG.depthThreshold}°),
                returns. Same Rep-Count engine; looser depth gate +
                smaller amplitude + higher target ({TARGET_REPS} reps)
                make it suitable for early-stage / deconditioned
                patients. Powered by the Rep-Count mechanic.
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
                  Testing: {side === "left" ? "Left" : "Right"} leg
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
                        min: MINI_SQUAT_CONFIG.depthThreshold,
                        max: MINI_SQUAT_CONFIG.topThreshold,
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
                        {interior >= MINI_SQUAT_CONFIG.topThreshold
                          ? "standing"
                          : interior <= MINI_SQUAT_CONFIG.depthThreshold
                          ? "mini-squat"
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
                    config={MINI_SQUAT_CONFIG}
                  />
                </div>
              </div>
            </div>
          )}

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2 m away, perpendicular to
                the stance line — <strong>lateral view</strong>.
              </li>
              <li>
                Patient stands side-on, test leg toward the camera.
                Test-side hip, knee, ankle must stay in frame.
              </li>
              <li>
                Descend slowly to a <strong>shallow</strong> mini-
                squat — interior ≤{" "}
                <strong>{MINI_SQUAT_CONFIG.depthThreshold}°</strong>{" "}
                (~40° flexion). Return to standing (interior ≥{" "}
                {MINI_SQUAT_CONFIG.topThreshold}°).
              </li>
              <li>
                Excursion gate: descents under{" "}
                <strong>{MINI_SQUAT_CONFIG.minAmplitude}°</strong> of
                knee motion are flagged as shallow.
              </li>
              <li>
                Target: {TARGET_REPS} reps. Higher volume reflects
                the lower per-rep intensity vs a full squat.
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
      {REHAB_EXERCISE_IMAGES["mini-squat"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["mini-squat"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the test leg
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the leg facing the camera. We track that knee&apos;s
        interior angle every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
