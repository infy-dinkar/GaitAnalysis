"use client";
// S3 — Pendulum / Circle Trace.
//
// Mechanic: Trace (lib/rehab/mechanics.ts traceStep). Cursor is
// in normalised [0..1] × [0..1] CSS-y-down coords — same space
// TraceShell expects.
//
// Cursor source — the WRIST landmark directly (no biomech math
// needed; the wrist x/y IS the game control):
//   cursor.x ← 1 − wrist.x / video.videoWidth   (mirrored for selfie
//                                                view, same direction
//                                                as the on-screen
//                                                skeleton)
//   cursor.y ← wrist.y / video.videoHeight
//
// Path: gentle counter-clockwise circle, centre (0.5, 0.5), radius
// 0.25 of the play area, 8 s per revolution. Tuned for early-stage
// frozen-shoulder / post-op pendulum mobility — slow enough that the
// patient can stay glued to the lead target without rushing.
//
// Reuses (no modifications):
//   • RehabCameraShell, TraceShell, traceStep — rehab mechanic library
//   • LM_LIVE wrist indices — lib/pose/landmarks-live
//   • usePoseDetectionLive, useCamera (via RehabCameraShell)
//   • usePatientContext for ?patientId doctor flow
// NO biomech file imported or touched — wrist position is the raw signal.

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

// Swing counter — no more precise circle tracing (a laggy wrist-cursor
// following a moving pacer was the frustration). Instead we count full
// pendulum revolutions: track the wrist's angle around a slowly-adapting
// centre and add 2π of unwrapped rotation = 1 swing. Direction-agnostic
// (CW or CCW). Auto-saves at TARGET_SWINGS.
const TARGET_SWINGS = 15;
// Only accumulate rotation when the wrist is a meaningful distance from
// the centre — stops jitter near the hang point from spinning the angle.
const MIN_SWING_RADIUS = 0.05;

/** Wrap an angle delta into (-π, π]. */
function normAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

export default function PendulumExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const wristSeenRef = useRef(false);
  const [tracking, setTracking] = useState(false);
  const [swings, setSwings] = useState(0);
  const [elapsedSec, setElapsedSec] = useState(0);
  // Brief pulse on the swing counter each time one lands.
  const [flash, setFlash] = useState(false);
  const flashTimeoutRef = useRef<number | null>(null);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  // Swing-detection state (see config comments above).
  const centerRef = useRef<{ x: number; y: number } | null>(null);
  const prevAngleRef = useRef<number | null>(null);
  const accumAngleRef = useRef<number>(0);
  const swingsCountRef = useRef<number>(0);
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
    swingsCountRef.current = 0;
    peakRadiusRef.current = 0;
    setTracking(false);
    setSwings(0);
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

      // Slowly-adapting centre = the average hang point the wrist
      // circles around. Seeds to the first sample, then EMA-drifts.
      const c = centerRef.current;
      if (c === null) {
        centerRef.current = { x: cx, y: cy };
      } else {
        centerRef.current = {
          x: c.x * 0.98 + cx * 0.02,
          y: c.y * 0.98 + cy * 0.02,
        };
      }
      const ctr = centerRef.current;
      const dx = cx - ctr.x;
      const dy = cy - ctr.y;
      const radius = Math.hypot(dx, dy);
      if (radius > peakRadiusRef.current) peakRadiusRef.current = radius;

      // Accumulate rotation only when the wrist is genuinely circling
      // (radius above the jitter floor).
      if (radius >= MIN_SWING_RADIUS) {
        const angle = Math.atan2(dy, dx);
        const prev = prevAngleRef.current;
        if (prev !== null) {
          accumAngleRef.current += normAngle(angle - prev);
          if (Math.abs(accumAngleRef.current) >= 2 * Math.PI) {
            accumAngleRef.current -=
              Math.sign(accumAngleRef.current) * 2 * Math.PI;
            const next = swingsCountRef.current + 1;
            swingsCountRef.current = next;
            setSwings(next);
            setFlash(true);
            if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
            flashTimeoutRef.current = window.setTimeout(() => setFlash(false), 700);
            if (next >= TARGET_SWINGS) markComplete();
          }
        }
        prevAngleRef.current = angle;
      } else {
        prevAngleRef.current = null; // reset so re-entry doesn't jump
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
      `Pendulum — ${swings} swing${swings === 1 ? "" : "s"} in ${durationSec.toFixed(0)}s on the ${side} arm (peak amplitude ${(peakRadius * 100).toFixed(0)}%).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      0,
      side,
      "Pendulum session",
    );
    return {
      module: "rehab" as const,
      movement: "pendulum",
      side,
      metrics: {
        exercise_slug: "pendulum",
        mechanic_id: "swing_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: durationSec,
        reps: swings,
        target_reps: TARGET_SWINGS,
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: {
          swings,
          targetSwings: TARGET_SWINGS,
          peakAmplitude: peakRadius,
        },
        signal: {
          name: "swing_amplitude",
          unit: "play-widths",
          value_at_peak: peakRadius,
        },
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [side, swings]);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>S3 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Pendulum / Circle Trace<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Gentle shoulder mobility (Codman&apos;s) — patient
                leans forward, lets the arm hang, and swings it in slow
                circles using body momentum. Each full circle counts as
                one <strong>swing</strong>; the session auto-saves after
                {" "}{TARGET_SWINGS} swings. No precise tracing needed —
                just swing freely; we track the wrist and count circles +
                measure the swing amplitude. Ideal for frozen-shoulder
                and early post-op range work.
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
              title={`Pendulum · ${side === "left" ? "Left" : "Right"} arm`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Goal ${TARGET_SWINGS} swings`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell onFrame={handleFrame} autoStart hideControls>
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">{side === "left" ? "L" : "R"} swings</p>
                    <p className="tabular text-2xl font-semibold text-white">{swings}<span className="text-sm text-zinc-400"> / {TARGET_SWINGS}</span></p>
                    <p className="mt-1 text-[10px] text-zinc-300">{tracking ? "circle the arm gently" : "waiting…"}</p>
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
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-purple-500/15 px-3 py-1 text-xs font-semibold text-purple-200 ring-1 ring-purple-400/40">{side === "left" ? "Left" : "Right"} arm</span>
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["pendulum"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["pendulum"]} alt="Pendulum reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownCard
                      countdown={countdown}
                      onSkip={skipCountdown}
                      hint="Patient leans forward, test arm hanging free, wrist in frame."
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
                        <p className="text-[10px] text-zinc-400">Swings auto-save at {TARGET_SWINGS}</p>
                      </div>
                      <div
                        className={`flex items-center justify-between rounded-lg border px-3 py-2 transition-all duration-200 ${
                          flash
                            ? "border-emerald-400 bg-emerald-500/20 ring-2 ring-emerald-400/60"
                            : "border-zinc-700 bg-zinc-900/80"
                        }`}
                      >
                        <div>
                          <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Swings</p>
                          <p className="tabular text-3xl font-bold leading-none text-white">
                            {Math.min(swings, TARGET_SWINGS)}
                            <span className="text-lg font-semibold text-zinc-500"> / {TARGET_SWINGS}</span>
                          </p>
                        </div>
                        {flash ? (
                          <span className="rounded-full bg-emerald-500/30 px-2 py-0.5 text-[10px] font-semibold text-emerald-100 ring-1 ring-emerald-400/50">+1 swing</span>
                        ) : swings >= TARGET_SWINGS ? (
                          <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[9px] font-semibold text-emerald-200">Complete</span>
                        ) : null}
                      </div>
                      <p className="text-[11px] leading-relaxed text-muted">
                        Lean forward, let the arm hang, and swing it in
                        slow circles. Each full circle counts as one swing.
                      </p>
                    </>
                  )}
                  <div className="no-pdf">
                    <AutoFlowFooter
                      complete={sessionPhase === "complete"}
                      buildPayload={buildRehabPayload}
                      completeHint={`${TARGET_SWINGS} swings done — saving to record automatically.`}
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
                the patient — they face the camera (frontal view).
              </li>
              <li>
                Patient leans forward slightly so the test arm hangs
                free in front of the chest — classic pendulum
                position.
              </li>
              <li>
                Make sure the wrist on the test side stays in frame
                across the full circle. The &quot;Wrist tracking&quot;
                badge above turns green once the landmark is locked.
              </li>
              <li>
                Swing the hanging arm in slow, easy <strong>circles</strong>
                {" "}(either direction) using body sway — not shoulder
                effort. Each full circle counts as one swing.
              </li>
              <li>
                The swing counter climbs with each full circle; after{" "}
                <strong>{TARGET_SWINGS}</strong> swings the session
                auto-saves. Bigger, smoother circles record a higher
                amplitude.
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
      {REHAB_EXERCISE_IMAGES["pendulum"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["pendulum"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the test arm
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the arm the patient will swing. We track that side&apos;s
        wrist and count each full circle as one swing.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
