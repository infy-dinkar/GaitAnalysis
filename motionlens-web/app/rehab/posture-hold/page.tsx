"use client";
// B1 — Posture Hold (Forward-Head Reset).
//
// Mechanic: Hold-in-Zone (lib/rehab/mechanics.ts holdInZoneStep).
// Band-based, NO direction flip — feed the forward-head offset
// angle directly. The engine checks value ∈ [min, max] each frame.
//
// Proxy math (computeForwardHeadOffsetDeg in poseMetrics):
//   Returns the angle of the (shoulder → ear) vector from
//   vertical-up.
//     • Good posture (ear ≈ above shoulder):  ~0-5°
//     • Mild forward head:                     ~10-15°
//     • Pronounced forward-head posture:       ~20-30°
//
// Target band [0°, 12°] = good posture (ear stays close to vertical
// above shoulder). Hold 20 s cumulative. Drift > 12° + hysteresis →
// timer pauses; return to band → timer resumes.
//
// Reuses (no modifications):
//   • HoldInZoneShell, holdInZoneStep, RehabCameraShell
//   • computeForwardHeadOffsetDeg — NEW pure fn in poseMetrics
//   • usePatientContext

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
import { computeForwardHeadOffsetDeg } from "@/lib/rehab/poseMetrics";
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

const POSTURE_HOLD_CONFIG = {
  // Band starts at 0 (perfectly aligned ear above shoulder).
  min: 0,
  // 12° upper edge — accommodates mild natural variance (~5-8°)
  // and small head movements without breaking the streak.
  max: 12,
  // 20 s cumulative — standard early-stage postural endurance target.
  targetHoldMs: 20_000,
  // 2° hysteresis — damps single-frame BlazePose noise at the band
  // boundary. (Smaller than K5 Wall Sit's 3° because the band is
  // narrower.)
  hysteresis: 2,
};
const AXIS_MIN = 0;
const AXIS_MAX = 40;

export default function PostureHoldExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [offset, setOffset] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const bestSignalRef = useRef<number>(Infinity);
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
      const angle = computeForwardHeadOffsetDeg(
        kp as unknown as LiveKeypoint[],
        side,
      );
      if (angle !== null) {
        setOffset(angle);
        const inBand =
          angle >= POSTURE_HOLD_CONFIG.min && angle <= POSTURE_HOLD_CONFIG.max;
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
        // Prefer the frame with the smallest offset (best alignment).
        if (inBand && lastKpRef.current && angle < bestSignalRef.current) {
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
    const finalBest = Number.isFinite(bestSignalRef.current)
      ? bestSignalRef.current
      : 0;
    const totalSec = totalInZoneMsRef.current / 1000;
    const bestDwellSec = bestDwellMsRef.current / 1000;
    const interpretation = totalInZoneMsRef.current > 0
      ? `Posture hold: ${totalSec.toFixed(1)}s cumulative in the 0–${POSTURE_HOLD_CONFIG.max}° band `
        + `(longest single hold ${bestDwellSec.toFixed(1)}s). Best alignment ${finalBest.toFixed(1)}° from vertical.`
      : "Session ended before the patient held the posture band.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      finalBest,
      side,
      `Posture hold — ${finalBest.toFixed(1)}° forward-head offset`,
    );
    return {
      module: "rehab" as const,
      movement: "posture-hold",
      side,
      metrics: {
        exercise_slug: "posture-hold",
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
          name: "forward_head_offset",
          unit: "deg",
          value_at_peak: finalBest,
          target_band: {
            min: POSTURE_HOLD_CONFIG.min,
            max: POSTURE_HOLD_CONFIG.max,
          },
        },
        target_hold_ms: POSTURE_HOLD_CONFIG.targetHoldMs,
        config: POSTURE_HOLD_CONFIG,
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
              <Badge>B1 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Posture Hold<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Forward-head reset — patient sits or stands lateral
                to the camera and holds the ear stacked above the
                shoulder (good cervical alignment). Drift more than{" "}
                {POSTURE_HOLD_CONFIG.max}° forward and the in-zone
                timer pauses; return to alignment to resume. Target:{" "}
                {(POSTURE_HOLD_CONFIG.targetHoldMs / 1000).toFixed(0)} s
                cumulative inside the band. Powered by the
                Hold-in-Zone mechanic.
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
              title="Posture Hold"
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Hold ${(POSTURE_HOLD_CONFIG.targetHoldMs / 1000).toFixed(0)}s`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  angleArc={{
                    vertex: side === "left" ? LM_LIVE.LEFT_SHOULDER : LM_LIVE.RIGHT_SHOULDER,
                    armA: side === "left" ? LM_LIVE.LEFT_EAR : LM_LIVE.RIGHT_EAR,
                    armB: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                    currentDeg: offset,
                  }}
                >
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">Head offset</p>
                    <p className="tabular text-2xl font-semibold text-white">{offset.toFixed(1)}°</p>
                  </div>
                </RehabCameraShell>
              )}
              sidebar={(
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-teal-500/15 px-3 py-1 text-xs font-semibold text-teal-200 ring-1 ring-teal-400/40">{side === "left" ? "Left" : "Right"} view</span>
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["posture-hold"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["posture-hold"]} alt="Posture Hold reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  <div className="flex min-h-0 flex-1 flex-col">
                    <HoldInZoneShell signal={offset} signalLabel="Offset (°)" axisMin={AXIS_MIN} axisMax={AXIS_MAX} config={POSTURE_HOLD_CONFIG} compact />
                  </div>
                  <div className="no-pdf"><RehabSessionFooter buildPayload={buildRehabPayload} label="Save session" compact /></div>
                </>
              )}
            />
          )}

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at <strong>shoulder height</strong>, ~2 m
                away, perpendicular to the body —{" "}
                <strong>lateral view</strong>. The chosen side&apos;s
                ear and shoulder must both stay clearly visible.
              </li>
              <li>
                Patient sits or stands with the test side toward the
                camera. Tie hair back / move it clear of the ear if
                needed so the landmark detects.
              </li>
              <li>
                Stack the ear directly above the shoulder — neutral
                cervical alignment. The forward-head offset readout
                should sit below{" "}
                <strong>{POSTURE_HOLD_CONFIG.max}°</strong> (status
                pill: &quot;stacked&quot;).
              </li>
              <li>
                Hold. Every ms inside the band counts toward the{" "}
                <strong>
                  {(POSTURE_HOLD_CONFIG.targetHoldMs / 1000).toFixed(0)} s
                </strong>{" "}
                target. Drift more than{" "}
                {POSTURE_HOLD_CONFIG.hysteresis}° outside the band
                and the timer pauses; reset and resume.
              </li>
              <li>
                Coaching cue: imagine a string lifting the top of
                the head upward, not the chin forward.
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
      {REHAB_EXERCISE_IMAGES["posture-hold"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["posture-hold"]}
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
        Pick the side closest to the camera. We track that
        side&apos;s ear and shoulder for the forward-head offset.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left side</Button>
        <Button onClick={() => onPick("right")}>Right side</Button>
      </div>
    </div>
  );
}
