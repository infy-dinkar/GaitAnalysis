"use client";
// S2 — Wall-Clock Multidirectional Reach.
//
// Mechanic: Target-Reach (lib/rehab/mechanics.ts targetReachStep +
// spawnReachTarget). Cursor is in [0..1] × [0..1] CSS-y-down —
// same space TargetReachShell expects.
//
// Cursor source — wrist position RELATIVE to the working shoulder.
// The patient's hand IS the cursor; reaching in any direction
// moves the cursor in that direction on screen. Mirror handled to
// match the selfie-skeleton overlay:
//
//   reachX_mirrored = (shoulder.x − wrist.x)   // raw pixels
//                                              // positive ⇒ patient
//                                              // reaches their RIGHT
//   reachY          = (wrist.y − shoulder.y)   // image y-down
//                                              // positive ⇒ patient
//                                              // reaches DOWN
//   scale           = shoulderWidth × 2.5      // ~1 arm-length spans
//                                              // ~35% of play area
//   cursor.x        = clamp(0.5 + reachX_mirrored / scale, 0, 1)
//   cursor.y        = clamp(0.5 + reachY / scale,          0, 1)
//
// Targets spawn at random positions in [0.15, 0.85] × [0.15, 0.85]
// — inherently multidirectional: targets pop in all four quadrants
// around the centre, so the patient must reach in all directions
// (the "wall-clock" pattern) to score, not just one axis.
//
// Reuses (no modifications):
//   • TargetReachShell, targetReachStep, spawnReachTarget,
//     RehabCameraShell — rehab mechanic library
//   • computeShoulderWidth — existing helper in lib/rehab/poseMetrics
//     (added for H3 weight-shift; reused here as the body-scale ref)
//   • LM_LIVE wrist + shoulder indices
//   • usePoseDetectionLive, useCamera, usePatientContext
//
// NO biomech file imported or touched — wrist + shoulder landmarks
// are read directly.

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import {
  AutoFlowCompleteOverlay,
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
  AutoFlowFooter,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const WRIST_VIS_THRESHOLD = 0.3;

// Circle counter — the patient traces the clock face with the working
// hand (12 → 3 → 6 → 9 → back to 12). We track the wrist's angle around
// a slowly-adapting centre and add 2π of unwrapped rotation = 1 full
// circle. Direction-agnostic (CW/CCW). Auto-saves at TARGET_CIRCLES. No
// precise tracing / target-hitting, so cursor lag is irrelevant.
const TARGET_CIRCLES = 10;
// Only accumulate rotation when the wrist is a meaningful distance from
// the centre — stops jitter near the centre from spinning the angle.
const MIN_CIRCLE_RADIUS = 0.05;

/** Wrap an angle delta into (-π, π]. */
function normAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

export default function WallClockExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [tracking, setTracking] = useState(false);
  const [circles, setCircles] = useState(0);
  const [elapsedSec, setElapsedSec] = useState(0);
  // Brief pulse on the circle counter each time one lands.
  const [flash, setFlash] = useState(false);
  const flashTimeoutRef = useRef<number | null>(null);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const wristSeenRef = useRef(false);
  // Circle-detection state.
  const centerRef = useRef<{ x: number; y: number } | null>(null);
  const prevAngleRef = useRef<number | null>(null);
  const accumAngleRef = useRef<number>(0);
  const circlesCountRef = useRef<number>(0);
  const peakRadiusRef = useRef<number>(0);

  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
    markComplete,
  } = useRehabAutoFlow(side !== null, () => {
    bestPoseRef.current = null;
    wristSeenRef.current = false;
    centerRef.current = null;
    prevAngleRef.current = null;
    accumAngleRef.current = 0;
    circlesCountRef.current = 0;
    peakRadiusRef.current = 0;
    setTracking(false);
    setCircles(0);
    setElapsedSec(0);
    sessionStartRef.current = performance.now();
  });

  useEffect(() => {
    if (sessionPhase !== "live") return;
    const id = window.setInterval(() => setElapsedSec((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [sessionPhase]);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const wristIdx = side === "right" ? LM.RIGHT_WRIST : LM.LEFT_WRIST;
      const wrist = kp[wristIdx];
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (
        !wrist
        || (wrist.score ?? 0) < WRIST_VIS_THRESHOLD
        || vw <= 0
        || vh <= 0
      ) {
        return; // dropout — hold state
      }
      const cx = Math.max(0, Math.min(1, 1 - wrist.x / vw));
      const cy = Math.max(0, Math.min(1, wrist.y / vh));

      // Slowly-adapting centre = the point the wrist circles around.
      const c = centerRef.current;
      if (c === null) {
        centerRef.current = { x: cx, y: cy };
      } else {
        centerRef.current = { x: c.x * 0.98 + cx * 0.02, y: c.y * 0.98 + cy * 0.02 };
      }
      const ctr = centerRef.current;
      const dx = cx - ctr.x;
      const dy = cy - ctr.y;
      const radius = Math.hypot(dx, dy);
      if (radius > peakRadiusRef.current) peakRadiusRef.current = radius;

      if (radius >= MIN_CIRCLE_RADIUS) {
        const angle = Math.atan2(dy, dx);
        const prev = prevAngleRef.current;
        if (prev !== null) {
          accumAngleRef.current += normAngle(angle - prev);
          if (Math.abs(accumAngleRef.current) >= 2 * Math.PI) {
            accumAngleRef.current -= Math.sign(accumAngleRef.current) * 2 * Math.PI;
            const next = circlesCountRef.current + 1;
            circlesCountRef.current = next;
            setCircles(next);
            setFlash(true);
            if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
            flashTimeoutRef.current = window.setTimeout(() => setFlash(false), 700);
            if (next >= TARGET_CIRCLES) markComplete();
          }
        }
        prevAngleRef.current = angle;
      } else {
        prevAngleRef.current = null;
      }

      if (lastKpRef.current) {
        bestPoseRef.current = {
          landmarks: lastKpRef.current.landmarks,
          source_frame: lastKpRef.current.source_frame,
          angle: 0,
          capturedAtMs: performance.now(),
        };
      }
      if (!wristSeenRef.current) {
        wristSeenRef.current = true;
        setTracking(true);
      }
    },
    [side, markComplete],
  );

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const durationSec = elapsedSecondsSince(sessionStartRef.current);
    const peakRadius = peakRadiusRef.current;
    const interpretation =
      `Wall-clock — ${circles} circle${circles === 1 ? "" : "s"} in ${durationSec.toFixed(0)}s on the ${side} arm (peak radius ${(peakRadius * 100).toFixed(0)}%).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      0,
      side,
      "Wall-clock session",
    );
    return {
      module: "rehab" as const,
      movement: "wall-clock",
      side,
      metrics: {
        exercise_slug: "wall-clock",
        mechanic_id: "swing_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: durationSec,
        reps: circles,
        target_reps: TARGET_CIRCLES,
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: {
          swings: circles,
          targetSwings: TARGET_CIRCLES,
          peakAmplitude: peakRadius,
        },
        signal: {
          name: "circle_amplitude",
          unit: "play-widths",
          value_at_peak: peakRadius,
        },
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [side, circles]);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>S2 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Wall-Clock Reach<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Multidirectional shoulder circles — patient stands
                frontal and moves the working hand in a full circle
                around a clock face (12 → 3 → 6 → 9 → back to 12). Each
                full circle is one <strong>circle</strong>; the session
                auto-saves after {TARGET_CIRCLES} circles. No precise
                tracing — just circle the arm smoothly in either
                direction. Trains shoulder ROM across the full
                hemisphere + coordination.
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
              title={`Wall Clock · ${side === "left" ? "Left" : "Right"} arm`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Goal ${TARGET_CIRCLES} circles`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell onFrame={handleFrame} autoStart hideControls>
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">{side === "left" ? "L" : "R"} circles</p>
                    <p className="tabular text-2xl font-semibold text-white">{circles}<span className="text-sm text-zinc-400"> / {TARGET_CIRCLES}</span></p>
                    <p className="mt-1 text-[10px] text-zinc-300">{tracking ? "circle the clock" : "waiting…"}</p>
                  </div>
                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownOverlay countdown={countdown} />
                  )}
                  {sessionPhase === "complete" && <AutoFlowCompleteOverlay />}
                </RehabCameraShell>
              )}
              sidebar={(
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40">{side === "left" ? "Left" : "Right"} arm</span>
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["wall-clock"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["wall-clock"]} alt="Wall Clock reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownCard
                      countdown={countdown}
                      onSkip={skipCountdown}
                      hint="Patient facing the camera, working hand free to circle."
                    />
                  )}
                  {(sessionPhase === "live" || sessionPhase === "complete") && (
                    <>
                      <div className="flex items-center justify-between rounded-lg border border-zinc-700 bg-zinc-900/80 px-3 py-2">
                        <div>
                          <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Time</p>
                          <p className="tabular text-2xl font-semibold leading-none text-white">
                            {Math.floor(elapsedSec / 60)}:{String(elapsedSec % 60).padStart(2, "0")}
                          </p>
                        </div>
                        <p className="text-[10px] text-zinc-400">Circles auto-save at {TARGET_CIRCLES}</p>
                      </div>
                      <div
                        className={`flex items-center justify-between rounded-lg border px-3 py-2 transition-all duration-200 ${
                          flash
                            ? "border-emerald-400 bg-emerald-500/20 ring-2 ring-emerald-400/60"
                            : "border-zinc-700 bg-zinc-900/80"
                        }`}
                      >
                        <div>
                          <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Circles</p>
                          <p className="tabular text-3xl font-bold leading-none text-white">
                            {Math.min(circles, TARGET_CIRCLES)}
                            <span className="text-lg font-semibold text-zinc-500"> / {TARGET_CIRCLES}</span>
                          </p>
                        </div>
                        {flash ? (
                          <span className="rounded-full bg-emerald-500/30 px-2 py-0.5 text-[10px] font-semibold text-emerald-100 ring-1 ring-emerald-400/50">+1 circle</span>
                        ) : circles >= TARGET_CIRCLES ? (
                          <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[9px] font-semibold text-emerald-200">Complete</span>
                        ) : null}
                      </div>
                      <p className="text-[11px] leading-relaxed text-muted">
                        Move the hand in a full circle around the clock
                        (12 → 3 → 6 → 9 → back to 12). Each full circle = 1.
                      </p>
                    </>
                  )}
                  <div className="no-pdf">
                    <AutoFlowFooter
                      complete={sessionPhase === "complete"}
                      buildPayload={buildRehabPayload}
                      completeHint={`${TARGET_CIRCLES} circles done — saving to record automatically.`}
                    />
                  </div>
                </>
              )}
            />
          )}

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at chest height, ~2 m away, perpendicular to
                the patient — <strong>frontal view</strong>. Both
                shoulders + the working wrist must stay in frame
                throughout.
              </li>
              <li>
                Move the working hand in a <strong>full circle</strong>
                around a clock face: up (12) → right (3) → down (6) →
                left (9) → back up to 12. Either direction is fine.
              </li>
              <li>
                Keep the circles smooth and reasonably big — a full
                loop back to the start counts as one circle.
              </li>
              <li>
                The circle counter climbs with each full loop; after
                {" "}{TARGET_CIRCLES} circles the session auto-saves.
              </li>
              <li>
                Stable shoulder + wrist visibility is required —
                avoid clothing that drapes over the shoulder; keep
                the working hand visible (no pocketed thumb,
                etc.).
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
      {REHAB_EXERCISE_IMAGES["wall-clock"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["wall-clock"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the reaching arm
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the arm the patient will circle. We track that hand and
        count each full circle around the clock as one.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
