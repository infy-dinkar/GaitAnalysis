"use client";
// K1 — Controlled Squat. First wired rehab exercise.
//
// Mechanic: Rep-Count Gate (lib/rehab/mechanics.ts repCountStep).
// Signal: knee INTERIOR angle, derived from the reused
// lib/biomech/knee-live.ts:computeKneeAngle helper. The biomech
// helper returns flexion (0 = straight, ~140 = fully bent); the
// rep-count engine wants HIGH at the rep "top" and LOW at "depth".
// We compute interior = 180 − flexion so:
//   • Standing  → interior ~180  (above topThreshold 160)
//   • Deep squat → interior ~90-110 (below depthThreshold 110)
// Rep closes on return to standing. Matches the engine's state
// machine direction without modifying either side.
//
// Reuses (no modifications):
//   • RehabCameraShell      — generic camera + skeleton overlay
//   • computeKneeAngle      — knee-live.ts pure helper
//   • RepCountShell         — UI + scoring
//   • repCountStep engine   — driven indirectly by RepCountShell
//   • usePatientContext     — optional ?patientId attaches doctor flow

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

const SQUAT_CONFIG = {
  topThreshold: 160,
  depthThreshold: 110,
  minAmplitude: 50,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function SquatExercisePage() {
  // Next.js 16 requires routes that use useSearchParams (via
  // usePatientContext below) to be wrapped in Suspense for static
  // prerender. Mirrors the same pattern the orthopedic test pages
  // use — without this the route prerender bails out at build time.
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 180 = standing position, so the engine starts in the
  // "above_top" phase ready for a descent.
  const [interior, setInterior] = useState<number>(180);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], _v: HTMLVideoElement) => {
      if (!side) return;
      // Reuse the BlazePose-live biomech math without modification.
      // The Keypoint shape from @tensorflow-models/pose-detection
      // is structurally compatible with LiveKeypoint at runtime
      // (score is populated by the BlazePose detector); the cast
      // satisfies the stricter LiveKeypoint.score: number type.
      const flexion = computeKneeAngle(
        "flexion_extension",
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (flexion !== null) {
        setInterior(180 - flexion);
      }
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
              <Badge>K1 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Controlled Squat<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Quality-gated squat rep counter. Each rep is scored
                against depth ({SQUAT_CONFIG.depthThreshold}° interior
                knee), amplitude ({SQUAT_CONFIG.minAmplitude}° excursion),
                and starting-position (knee ≥ {SQUAT_CONFIG.topThreshold}°
                = standing). Shallow or partial reps are flagged
                transparently. Goal: {TARGET_REPS} good reps.
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

              {/* Two-column layout on lg+: camera left, game stats
                  right. Stacks on smaller screens. Keeps both
                  panels visible without scrolling on a typical
                  clinic laptop / desktop. */}
              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  {/* Knee-angle arc — vertex at the working-side knee,
                      arms to the hip and ankle on that side. Reuses
                      the same `interior` value already computed by
                      handleFrame; the shell just renders it as a
                      partial arc with a colour band tied to the
                      rep-count thresholds so the manager can see
                      the ViFive-style joint indicator. */}
                  <RehabCameraShell
                    onFrame={handleFrame}
                    angleArc={{
                      vertex: side === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                      armA: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                      armB: side === "left" ? LM_LIVE.LEFT_ANKLE : LM_LIVE.RIGHT_ANKLE,
                      currentDeg: interior,
                      // Band tied to the healthy rep sweep — green
                      // between the depth and top thresholds (active
                      // rep zone), amber at the edges, red outside.
                      band: {
                        min: SQUAT_CONFIG.depthThreshold,
                        max: SQUAT_CONFIG.topThreshold,
                      },
                    }}
                  >
                    {/* Live knee-angle readout — corner overlay on
                        the camera tile. Updates every frame the
                        detector returns a valid signal. */}
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} knee
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {interior.toFixed(0)}°
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <RepCountShell
                    signal={interior}
                    signalLabel={`${side === "left" ? "Left" : "Right"} knee angle (°)`}
                    targetReps={TARGET_REPS}
                    config={SQUAT_CONFIG}
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
                Camera at hip height, ~2 m away, perpendicular to the
                stance line.
              </li>
              <li>
                Patient stands <strong>side-on</strong> — the test leg
                toward the camera, contralateral leg behind it.
              </li>
              <li>
                Make sure the test-side hip, knee, and ankle are all
                visible throughout the squat.
              </li>
              <li>
                Stand tall (knee ≥ {SQUAT_CONFIG.topThreshold}°), squat
                until the knee interior angle drops below{" "}
                {SQUAT_CONFIG.depthThreshold}°, return to standing —
                that&apos;s one rep.
              </li>
              <li>
                Reps below the depth gate or with under{" "}
                {SQUAT_CONFIG.minAmplitude}° of excursion are flagged
                <span className="ml-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-[10px] font-medium text-rose-200">
                  shallow
                </span>{" "}
                — they still count toward the rep total but not the
                streak.
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
      {REHAB_EXERCISE_IMAGES["squat"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["squat"]}
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
        interior angle frame-by-frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
