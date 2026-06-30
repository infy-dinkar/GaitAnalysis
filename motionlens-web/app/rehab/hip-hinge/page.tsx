"use client";
// B5 — Hip Hinge.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep).
// Same engine + direction convention as B2 Back Extension —
// HIGH at "top" of the rep (hinged forward), LOW at "depth"
// (returned to upright). No flip — magnitude already grows in
// the right direction.
//
// Proxy math: computeHipHingeAngleDeg(kp) — unsigned trunk-tilt
// from vertical-up. Patient is instructed to hinge FORWARD only
// (back stays flat, hinge at the hips, no spinal flexion).
//   • Upright neutral:  ~0-5°
//   • Mid-hinge:        ~25-40°
//   • Deep hinge:       ~60-75° (trunk nearing parallel-to-floor)
//
// Reuses (no modifications):
//   • RepCountShell, repCountStep, RehabCameraShell
//   • computeHipHingeAngleDeg — NEW pure fn in poseMetrics
//   • usePatientContext
// NO biomech file modified.

import { Suspense, useCallback, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import { computeHipHingeAngleDeg } from "@/lib/rehab/poseMetrics";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const HIP_HINGE_CONFIG = {
  // Patient must reach ≥ 30° trunk tilt for the rep to register
  // as "hinged" — well clear of postural sway / breathing wobble
  // (typically <8°) yet reachable by most patients.
  topThreshold: 30,
  // Back near upright counts as "depth" / rest position.
  depthThreshold: 10,
  // 20° excursion gate — distinguishes a deliberate hinge from
  // small forward sways.
  minAmplitude: 20,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

export default function HipHingeExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [trunkAngle, setTrunkAngle] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], _v: HTMLVideoElement) => {
      if (!side) return;
      const angle = computeHipHingeAngleDeg(
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (angle !== null) setTrunkAngle(angle);
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
              <Badge>B5 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Hip Hinge<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Posterior-chain pattern training — patient hinges
                forward at the hips with a FLAT back, returns to
                upright. Each cycle = one rep. Powered by the
                Rep-Count mechanic.
              </p>
              <div className="mt-5 rounded-card border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p className="font-semibold uppercase tracking-[0.14em] text-amber-200 text-[10px]">
                  Keep the back FLAT — hinge at the hips
                </p>
                <p className="mt-1 text-xs leading-relaxed">
                  This is NOT a forward bend. Maintain a neutral
                  spine throughout — knees soft, shoulders blade
                  back, motion comes from the hip joints. Spinal
                  flexion during the drill inflates the proxy
                  reading without delivering the intended
                  posterior-chain load.
                </p>
              </div>
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
                  Camera side: {side === "left" ? "Left" : "Right"}
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
                        Trunk tilt
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {trunkAngle.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {trunkAngle >= HIP_HINGE_CONFIG.topThreshold
                          ? "hinged"
                          : trunkAngle <= HIP_HINGE_CONFIG.depthThreshold
                          ? "upright"
                          : "transition"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <RepCountShell
                    signal={trunkAngle}
                    signalLabel="Trunk hinge angle (°)"
                    targetReps={TARGET_REPS}
                    config={HIP_HINGE_CONFIG}
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
                the body — <strong>lateral view</strong>. Whole
                body in frame; head + torso + hip + knee + ankle
                all visible across the hinge.
              </li>
              <li>
                Stand with feet hip-width, knees soft (not locked).
                Hinge forward at the hips — push the hips BACK as
                the chest tips forward. Keep the back straight
                (flat, not rounded).
              </li>
              <li>
                Reach ≥ <strong>{HIP_HINGE_CONFIG.topThreshold}°</strong>{" "}
                of trunk tilt (status: &quot;hinged&quot;), then
                return to upright (≤ {HIP_HINGE_CONFIG.depthThreshold}°)
                — that&apos;s one rep.
              </li>
              <li>
                Excursion gate: hinges under{" "}
                <strong>{HIP_HINGE_CONFIG.minAmplitude}°</strong> of
                trunk motion flag as shallow.
              </li>
              <li>
                Engine note: the first hinge primes the rep state
                machine — count one extra if you want exactly{" "}
                {TARGET_REPS} counted.
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
      {REHAB_EXERCISE_IMAGES["hip-hinge"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["hip-hinge"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Which side is facing the camera?
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the side closest to the camera. We compute the trunk-
        tilt angle from the shoulder-mid + hip-mid pair — symmetric,
        but we keep the picker for consistency with other lateral-
        view drills.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left side</Button>
        <Button onClick={() => onPick("right")}>Right side</Button>
      </div>
    </div>
  );
}
