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

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { HoldInZoneShell } from "@/components/rehab/mechanics/HoldInZoneShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import { computeShoulderAngle } from "@/lib/biomech/shoulder-live";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
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
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const angle = computeShoulderAngle(
        "flexion",
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (angle !== null) {
        setShoulderAngle(angle);
        const inBand =
          angle >= WALL_SLIDE_CONFIG.min && angle <= WALL_SLIDE_CONFIG.max;
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
          bestSignalRef.current = angle;
          bestPoseRef.current = {
            landmarks: lastKpRef.current.landmarks,
            source_frame: lastKpRef.current.source_frame,
            angle,
            capturedAtMs: now,
          };
        }
      }
    },
    [side],
  );

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const totalSec = totalInZoneMsRef.current / 1000;
    const bestDwellSec = bestDwellMsRef.current / 1000;
    const interpretation = totalInZoneMsRef.current > 0
      ? `Wall slide: ${totalSec.toFixed(1)}s cumulative in the ${WALL_SLIDE_CONFIG.min}–${WALL_SLIDE_CONFIG.max}° band `
        + `(longest single hold ${bestDwellSec.toFixed(1)}s). Target ${(WALL_SLIDE_CONFIG.targetHoldMs / 1000).toFixed(0)}s.`
      : "Session ended before the patient held the wall-slide band.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      bestSignalRef.current,
      side,
      `Wall slide — ${bestSignalRef.current.toFixed(0)}° shoulder flexion`,
    );
    return {
      module: "rehab" as const,
      movement: "wall-slide",
      side,
      metrics: {
        exercise_slug: "wall-slide",
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
          name: "shoulder_flexion",
          unit: "deg",
          value_at_peak: bestSignalRef.current,
          target_band: {
            min: WALL_SLIDE_CONFIG.min,
            max: WALL_SLIDE_CONFIG.max,
          },
        },
        target_hold_ms: WALL_SLIDE_CONFIG.targetHoldMs,
        config: WALL_SLIDE_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [side]);

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

          {!side ? <SidePicker onPick={setSide} /> : null}

          {side && (
            <LiveModeLayout
              title={`Wall Slide · ${side === "left" ? "Left" : "Right"} arm`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Band ${WALL_SLIDE_CONFIG.min}–${WALL_SLIDE_CONFIG.max}°`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  angleArc={{
                    vertex: side === "left" ? LM_LIVE.LEFT_SHOULDER : LM_LIVE.RIGHT_SHOULDER,
                    armA: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                    armB: side === "left" ? LM_LIVE.LEFT_ELBOW : LM_LIVE.RIGHT_ELBOW,
                    currentDeg: shoulderAngle,
                    band: { min: WALL_SLIDE_CONFIG.min, max: WALL_SLIDE_CONFIG.max },
                  }}
                >
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">{side === "left" ? "L" : "R"} shoulder</p>
                    <p className="tabular text-2xl font-semibold text-white">{shoulderAngle.toFixed(0)}°</p>
                  </div>
                </RehabCameraShell>
              )}
              sidebar={(
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-teal-500/15 px-3 py-1 text-xs font-semibold text-teal-200 ring-1 ring-teal-400/40">{side === "left" ? "Left" : "Right"} arm</span>
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["wall-slide"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["wall-slide"]} alt="Wall Slide reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  <div className="flex min-h-0 flex-1 flex-col">
                    <HoldInZoneShell signal={shoulderAngle} signalLabel={`${side === "left" ? "L" : "R"} shoulder (°)`} axisMin={AXIS_MIN} axisMax={AXIS_MAX} config={WALL_SLIDE_CONFIG} compact />
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
      {REHAB_EXERCISE_IMAGES["wall-slide"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["wall-slide"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
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
