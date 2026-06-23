"use client";
// K5 — Wall Sit. Second wired rehab exercise.
//
// Mechanic: Hold-in-Zone (lib/rehab/mechanics.ts holdInZoneStep).
// Signal: knee FLEXION fed directly — no direction flip needed
// because Hold-in-Zone is band-membership only, not high/low.
//   • Standing       → flexion ~0°   (below band, out of zone)
//   • Wall sit       → flexion ~90°  (inside band [80°, 100°], timer runs)
//   • Deeper squat   → flexion ~120° (above band, out of zone)
// Leaving the band pauses the timer; returning resumes it. Hysteresis
// debounces edge chatter so a single noisy frame can't break the
// dwell streak.
//
// Reuses (no modifications):
//   • RehabCameraShell    — generic camera + skeleton overlay
//   • computeKneeAngle    — knee-live.ts pure helper
//   • HoldInZoneShell     — UI + scoring
//   • holdInZoneStep      — driven indirectly by HoldInZoneShell
//   • usePatientContext   — ?patientId attaches doctor flow

import { Suspense, useCallback, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { HoldInZoneShell } from "@/components/rehab/mechanics/HoldInZoneShell";
import { computeKneeAngle } from "@/lib/biomech/knee-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";

type Side = "left" | "right";

const WALL_SIT_CONFIG = {
  // Target band: 90° ± 10° knee flexion. Thigh roughly parallel
  // to the floor when standing wall-sit posture is held.
  min: 80,
  max: 100,
  // 30 s is the standard clinical wall-sit goal for an average
  // adult. Easy to scale per patient later via a difficulty knob.
  targetHoldMs: 30_000,
  // 3° hysteresis around each band edge so MediaPipe knee jitter
  // doesn't chatter the in-zone classification at the boundary.
  hysteresis: 3,
};
const AXIS_MIN = 0;     // standing
const AXIS_MAX = 130;   // deep squat — gives the band visual headroom

export default function WallSitExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 0 = standing position, so the patient starts below the
  // wall-sit band and the timer doesn't accumulate until they
  // actually descend into the zone.
  const [kneeFlexion, setKneeFlexion] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], _v: HTMLVideoElement) => {
      if (!side) return;
      const flexion = computeKneeAngle(
        "flexion_extension",
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (flexion !== null) {
        // Feed flexion directly — no inversion. The band engine
        // doesn't care about direction, only band membership.
        setKneeFlexion(flexion);
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
              <Badge>K5 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Wall Sit<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Isometric hold at {WALL_SIT_CONFIG.min}°–{WALL_SIT_CONFIG.max}°
                {" "}knee flexion (thigh ≈ parallel to floor). Hold for{" "}
                {(WALL_SIT_CONFIG.targetHoldMs / 1000).toFixed(0)} s
                total. Powered by the Hold-in-Zone mechanic — every
                ms inside the band counts; drift out and the timer
                pauses until you&apos;re back in zone.
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
                  <RehabCameraShell onFrame={handleFrame}>
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} knee flexion
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {kneeFlexion.toFixed(0)}°
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <HoldInZoneShell
                    signal={kneeFlexion}
                    signalLabel={`${side === "left" ? "Left" : "Right"} knee flexion (°)`}
                    axisMin={AXIS_MIN}
                    axisMax={AXIS_MAX}
                    config={WALL_SIT_CONFIG}
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
                Patient stands <strong>side-on</strong> with back
                against a wall — the test leg toward the camera.
              </li>
              <li>
                Slide down the wall until the knee flexion lands
                between <strong>{WALL_SIT_CONFIG.min}°</strong> and{" "}
                <strong>{WALL_SIT_CONFIG.max}°</strong> (thigh roughly
                parallel to the floor).
              </li>
              <li>
                Hold the position — the in-zone timer accumulates as
                long as the knee stays inside the band. Drifting more
                than {WALL_SIT_CONFIG.hysteresis}° outside an edge
                pauses the timer.
              </li>
              <li>
                Target: cumulative{" "}
                <strong>
                  {(WALL_SIT_CONFIG.targetHoldMs / 1000).toFixed(0)} s
                </strong>{" "}
                inside the band.
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
        Choose the test leg
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the leg facing the camera. We track that knee&apos;s
        flexion frame-by-frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left leg</Button>
        <Button onClick={() => onPick("right")}>Right leg</Button>
      </div>
    </div>
  );
}
