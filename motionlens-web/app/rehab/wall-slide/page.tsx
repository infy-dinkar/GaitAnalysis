"use client";
// S4 — Wall Slide / Overhead Reach Hold.
//
// Mechanic: Hold-in-Zone (lib/rehab/mechanics.ts holdInZoneStep).
// Band-based, NO direction flip — feed the shoulder flexion angle
// directly. The engine checks value ∈ [min, max] each frame.
//
// Signal: computeShoulderAngle("flexion", kp, side) — returns the
// angle between the trunk vector (hip → shoulder, pointing UP) and
// the arm vector (shoulder → elbow):
//   • Arm at side (rest)       → flexion ≈ 0°
//   • Arm forward horizontal   → flexion ≈ 90°
//   • Arm fully overhead       → flexion ≈ 180°
//
// Wall slide target = near-overhead hold. Band 140°–160° catches
// the meaningful range:
//   • Hits the upper portion of clinical active flexion ROM
//   • Leaves ~20° of headroom under the helper's 180° ceiling so
//     real-world technique (slight elbow flexion, head tilt) still
//     scores
//
// NO biomech file modified — computeShoulderAngle imported as-is.

import { Suspense, useCallback, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { HoldInZoneShell } from "@/components/rehab/mechanics/HoldInZoneShell";
import { computeShoulderAngle } from "@/lib/biomech/shoulder-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";

type Side = "left" | "right";

const WALL_SLIDE_CONFIG = {
  // Band centred at 150° flexion (mid of the upper-ROM window).
  // ±10° tolerance gives the patient room to settle without
  // chattering the in-zone flag at the edges.
  min: 140,
  max: 160,
  // 20 s cumulative — standard endurance target for an overhead
  // hold in early/mid-stage shoulder rehab.
  targetHoldMs: 20_000,
  // 3° hysteresis — same edge-debounce K5 Wall Sit uses for the
  // knee-flexion band. Damps single-frame MediaPipe noise at the
  // boundary.
  hysteresis: 3,
};
// Axis: 0° (arm at side) to 180° (full overhead) — matches the
// helper's full return range, gives the band visual context.
const AXIS_MIN = 0;
const AXIS_MAX = 180;

export default function WallSlideExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 0 = arm at side. Patient starts below the band; the
  // engine accumulates dtMs only once they slide up into 140°-160°.
  const [shoulderAngle, setShoulderAngle] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], _v: HTMLVideoElement) => {
      if (!side) return;
      const angle = computeShoulderAngle(
        "flexion",
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (angle !== null) {
        // Feed directly — Hold-in-Zone is band-membership only,
        // not direction-based.
        setShoulderAngle(angle);
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
              <Badge>S4 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Wall Slide<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Overhead-reach hold against a wall. Patient stands
                with back against the wall, slides the working arm
                up the wall surface, and holds at the
                {" "}{WALL_SLIDE_CONFIG.min}°–{WALL_SLIDE_CONFIG.max}°
                {" "}shoulder flexion band. Target:{" "}
                {(WALL_SLIDE_CONFIG.targetHoldMs / 1000).toFixed(0)} s
                cumulative inside the band. Every ms inside counts;
                drift out and the timer pauses until the patient
                returns. Powered by the Hold-in-Zone mechanic.
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
                <span className="inline-flex items-center gap-1.5 rounded-full bg-teal-500/15 px-3 py-1 text-xs font-semibold text-teal-200 ring-1 ring-teal-400/40">
                  Working arm: {side === "left" ? "Left" : "Right"}
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
                        {side === "left" ? "Left" : "Right"} shoulder · flexion
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {shoulderAngle.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {shoulderAngle >= WALL_SLIDE_CONFIG.min
                          && shoulderAngle <= WALL_SLIDE_CONFIG.max
                          ? "in zone"
                          : shoulderAngle > WALL_SLIDE_CONFIG.max
                          ? "above band"
                          : "below band · slide higher"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <HoldInZoneShell
                    signal={shoulderAngle}
                    signalLabel={`${side === "left" ? "Left" : "Right"} shoulder flexion (°)`}
                    axisMin={AXIS_MIN}
                    axisMax={AXIS_MAX}
                    config={WALL_SLIDE_CONFIG}
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
                Patient stands with <strong>back against a wall</strong>,
                heels ~10 cm out from the wall base. Forearm and the
                back of the hand stay in contact with the wall
                throughout.
              </li>
              <li>
                Camera at shoulder height, ~2 m away, perpendicular
                to the body — <strong>lateral view</strong>, test
                side toward the camera.
              </li>
              <li>
                Test-side hip, shoulder, and elbow must all stay in
                frame across the full slide.
              </li>
              <li>
                Slide the arm <strong>up the wall</strong> (shoulder
                flexion). The arm must stay extended at the elbow;
                forearm flat to the wall.
              </li>
              <li>
                Hold inside the{" "}
                <strong>
                  {WALL_SLIDE_CONFIG.min}°–{WALL_SLIDE_CONFIG.max}°
                </strong>{" "}
                band — the in-zone timer accumulates as long as the
                angle stays inside. Drifting more than{" "}
                {WALL_SLIDE_CONFIG.hysteresis}° outside an edge
                pauses the timer.
              </li>
              <li>
                Target: cumulative{" "}
                <strong>
                  {(WALL_SLIDE_CONFIG.targetHoldMs / 1000).toFixed(0)} s
                </strong>{" "}
                inside the band. Rest, return to start, repeat as
                tolerated.
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
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the working arm
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the arm the patient will slide up the wall. We compute
        that shoulder&apos;s flexion angle (trunk vs arm) every
        frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
