"use client";
// H1 — Pelvic-Level Hold (Trendelenburg retraining).
//
// Mechanic: Hold-in-Zone (lib/rehab/mechanics.ts holdInZoneStep).
// Signal: signed pelvic tilt (line from LEFT_HIP to RIGHT_HIP vs
// horizontal). The band is centred on 0° — symmetric around level
// — so the engine's pure min ≤ value ≤ max test works directly
// with negative bounds.
//
//   • Pelvis level   → tilt ≈ 0°    (inside band [−5°, +5°], timer runs)
//   • Right hip drop → tilt ≈ +10°  (above band, timer pauses)
//   • Left hip drop  → tilt ≈ −10°  (below band, timer pauses)
//
// Frontal view test — patient stands on one leg, faces camera.
// Optional stance-leg picker is for the on-screen label only; the
// math is symmetric so the picked leg doesn't change the
// computation.
//
// Reuses (no modifications):
//   • RehabCameraShell, HoldInZoneShell, holdInZoneStep — rehab
//     mechanic library
//   • computePelvicTiltDeg — NEW pure fn in lib/rehab/poseMetrics.ts
//     (no existing biomech file was modified)
//   • usePatientContext — ?patientId attaches doctor flow

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { HoldInZoneShell } from "@/components/rehab/mechanics/HoldInZoneShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { RehabStartCard } from "@/components/rehab/RehabStartCard";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import { computePelvicTiltDeg } from "@/lib/rehab/poseMetrics";
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

type StanceLeg = "left" | "right";

const PELVIC_HOLD_CONFIG = {
  // Symmetric ±5° band around level. Clinically a "drop" of more
  // than ~5° is the conventional Trendelenburg sign threshold.
  min: -5,
  max: 5,
  // 25 s cumulative target — sits between the 15-30 s range used
  // in standard Trendelenburg single-leg-stance protocols.
  targetHoldMs: 25_000,
  // 1.5° hysteresis — tighter than wall-sit's 3° because the band
  // itself is tighter (10° wide vs 20°), so the relative jitter
  // headroom stays comparable.
  hysteresis: 1.5,
};
// Visual axis range — gives the band visible context to either side.
const AXIS_MIN = -25;
const AXIS_MAX = 25;

export default function PelvicHoldExercisePage() {
  // Next.js 16 static prerender requires Suspense around
  // usePatientContext (which uses useSearchParams). Same pattern
  // K1/K5 use.
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [stance, setStance] = useState<StanceLeg | null>(null);
  const [started, setStarted] = useState(false);
  // Default 0 = perfectly level. Patient starts in zone before
  // they lift the contralateral foot.
  const [pelvicTilt, setPelvicTilt] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const bestSignalRef = useRef<number>(0);
  const totalInZoneMsRef = useRef<number>(0);
  const currentDwellMsRef = useRef<number>(0);
  const bestDwellMsRef = useRef<number>(0);
  const lastTickRef = useRef<number | null>(null);
  const wasInZoneRef = useRef<boolean>(false);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const tilt = computePelvicTiltDeg(
        kp as unknown as LiveKeypoint[],
      );
      if (tilt !== null) {
        setPelvicTilt(tilt);
        const inBand =
          tilt >= PELVIC_HOLD_CONFIG.min && tilt <= PELVIC_HOLD_CONFIG.max;
        const now = performance.now();
        if (lastTickRef.current !== null) {
          const dt = now - lastTickRef.current;
          if (inBand && wasInZoneRef.current) {
            totalInZoneMsRef.current += dt;
            currentDwellMsRef.current += dt;
            if (currentDwellMsRef.current > bestDwellMsRef.current) {
              bestDwellMsRef.current = currentDwellMsRef.current;
            }
          } else if (!inBand) {
            currentDwellMsRef.current = 0;
          }
        }
        lastTickRef.current = now;
        wasInZoneRef.current = inBand;
        if (inBand && lastKpRef.current) {
          // Prefer the closest-to-level frame for the skeleton.
          if (
            Math.abs(tilt) <= Math.abs(bestSignalRef.current)
            || bestPoseRef.current === null
          ) {
            bestSignalRef.current = tilt;
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: tilt,
              capturedAtMs: now,
            };
          }
        }
      }
    },
    [],
  );

  const buildRehabPayload = useCallback(() => {
    if (!stance) return null;
    const totalSec = totalInZoneMsRef.current / 1000;
    const bestDwellSec = bestDwellMsRef.current / 1000;
    const interpretation = totalInZoneMsRef.current > 0
      ? `Pelvic-level hold: ${totalSec.toFixed(1)}s cumulative inside ±${PELVIC_HOLD_CONFIG.max}° band `
        + `(longest single hold ${bestDwellSec.toFixed(1)}s). Target ${(PELVIC_HOLD_CONFIG.targetHoldMs / 1000).toFixed(0)}s.`
      : "Session ended before the patient held the pelvic-level band.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      bestSignalRef.current,
      stance,
      `Pelvic-level hold — ${bestSignalRef.current.toFixed(1)}° tilt`,
    );
    return {
      module: "rehab" as const,
      movement: "pelvic-hold",
      side: stance,
      metrics: {
        exercise_slug: "pelvic-hold",
        mechanic_id: "hold_in_zone",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: {
          totalMsInZone: totalInZoneMsRef.current,
          bestDwellMs: bestDwellMsRef.current,
          currentDwellMs: currentDwellMsRef.current,
        },
        signal: {
          name: "pelvic_tilt",
          unit: "deg",
          value_at_peak: bestSignalRef.current,
          target_band: {
            min: PELVIC_HOLD_CONFIG.min,
            max: PELVIC_HOLD_CONFIG.max,
          },
        },
        stance,
        target_hold_ms: PELVIC_HOLD_CONFIG.targetHoldMs,
        config: PELVIC_HOLD_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [stance]);

  // Direction hint shown next to the live readout — helps the
  // patient understand which side is dropping when they drift out
  // of the band.
  const tiltSide =
    Math.abs(pelvicTilt) < 0.5
      ? "level"
      : pelvicTilt > 0
      ? "right hip dropping"
      : "left hip dropping";

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>H1 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Pelvic-Level Hold<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Trendelenburg retraining — patient stands on one leg
                facing the camera and keeps the pelvis level.
                In-zone band is {PELVIC_HOLD_CONFIG.min}° to{" "}
                {PELVIC_HOLD_CONFIG.max}° tilt. Drop a hip more than{" "}
                {PELVIC_HOLD_CONFIG.max + PELVIC_HOLD_CONFIG.hysteresis}°
                and the timer pauses; return to level and it resumes.
                Target:{" "}
                {(PELVIC_HOLD_CONFIG.targetHoldMs / 1000).toFixed(0)} s
                cumulative. Powered by the Hold-in-Zone mechanic.
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

          {!stance ? <StancePicker onPick={setStance} /> : null}

          {stance && (
            <LiveModeLayout
              title={`Pelvic Level Hold · ${stance === "left" ? "Left" : "Right"} stance`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Hold ${(PELVIC_HOLD_CONFIG.targetHoldMs / 1000).toFixed(0)}s`}
              onExit={() => { setStance(null); setStarted(false); }}
              camera={(
                <RehabCameraShell onFrame={handleFrame}>
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">Pelvic tilt</p>
                    <p className="tabular text-2xl font-semibold text-white">{pelvicTilt > 0 ? "+" : ""}{pelvicTilt.toFixed(1)}°</p>
                    <p className="mt-1 text-[10px] text-zinc-300">{tiltSide}</p>
                  </div>
                </RehabCameraShell>
              )}
              sidebar={(
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-teal-500/15 px-3 py-1 text-xs font-semibold text-teal-200 ring-1 ring-teal-400/40">{stance === "left" ? "Left" : "Right"} stance</span>
                    <Button variant="ghost" size="sm" onClick={() => { setStance(null); setStarted(false); }}>Change stance</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["pelvic-hold"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["pelvic-hold"]} alt="Pelvic Hold reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  <div className="flex min-h-0 flex-1 flex-col">
                    {started ? (
                      <HoldInZoneShell signal={pelvicTilt} signalLabel="Pelvic tilt (°)" axisMin={AXIS_MIN} axisMax={AXIS_MAX} config={PELVIC_HOLD_CONFIG} compact />
                    ) : (
                      <RehabStartCard onStart={() => setStarted(true)} />
                    )}
                  </div>
                  <div className="no-pdf"><RehabSessionFooter buildPayload={buildRehabPayload} label="Save session" compact /></div>
                </>
              )}
            />
          )}

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2 m away, perpendicular to
                the patient — they face the camera directly (frontal
                view, NOT side-on).
              </li>
              <li>
                Both hips must be clearly visible — no loose clothing
                draped across the pelvis.
              </li>
              <li>
                Stand on the picked leg, lift the contralateral foot
                a few cm off the floor.
              </li>
              <li>
                Keep the pelvis <strong>level</strong> — keep tilt
                inside the {PELVIC_HOLD_CONFIG.min}° to{" "}
                {PELVIC_HOLD_CONFIG.max}° band. A drop of more than{" "}
                {PELVIC_HOLD_CONFIG.max + PELVIC_HOLD_CONFIG.hysteresis}°
                pauses the timer.
              </li>
              <li>
                Target: cumulative{" "}
                <strong>
                  {(PELVIC_HOLD_CONFIG.targetHoldMs / 1000).toFixed(0)} s
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

function StancePicker({ onPick }: { onPick: (s: StanceLeg) => void }) {
  return (
    <div className="mt-10 max-w-xl">
      {REHAB_EXERCISE_IMAGES["pelvic-hold"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["pelvic-hold"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the stance leg
      </h2>
      <p className="mt-2 text-sm text-muted">
        Which leg is the patient standing on? The pelvic-tilt math is
        symmetric — this is just a label so the on-screen banner is
        accurate.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Standing on LEFT</Button>
        <Button onClick={() => onPick("right")}>Standing on RIGHT</Button>
      </div>
    </div>
  );
}
