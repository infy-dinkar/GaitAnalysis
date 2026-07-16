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

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import { computeKneeAngle } from "@/lib/biomech/knee-live";
import { DEFAULT_LEVEL_INDEX, SQUAT_LADDER } from "@/lib/rehab/progressionLadders";
import { useProgressionLevel } from "@/lib/rehab/useProgressionLevel";
import { createRoundingTracker, createHandUseTracker } from "@/lib/rehab/compensationChecks";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import type { RepCountState, Score } from "@/lib/rehab/gameState";
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

type SessionPhase = "countdown" | "live" | "complete";

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 180 = standing position, so the engine starts in the
  // "above_top" phase ready for a descent.
  const [interior, setInterior] = useState<number>(180);

  // Auto-flow session phase machine, driven by the doctor-flow UX
  // spec: side pick → 3-2-1 countdown → live scoring → complete when
  // TARGET_REPS is hit → auto-save toast fires. `null` while no side
  // is picked.
  const [sessionPhase, setSessionPhase] = useState<SessionPhase | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);

  const { patient, isDoctorFlow } = usePatientContext();

  // Progression: derive the current level from the patient's history
  // when the doctor flow supplied a patientId. Public flow falls back
  // to the ladder's default level (matches the hardcoded SQUAT_CONFIG).
  const progression = useProgressionLevel(
    patient?.id ?? null,
    "squat",
    SQUAT_LADDER,
  );
  const activeConfig = useMemo(
    () => (isDoctorFlow && progression.config ? progression.config : SQUAT_CONFIG),
    [isDoctorFlow, progression.config],
  );

  // Compensation trackers — additive; nothing here modifies biomech
  // engines. RoundingTracker catches trunk lean, HandUseTracker
  // catches wrist-at-hip offloading.
  const roundingRef = useRef(createRoundingTracker({ thresholdFraction: 0.18 }));
  const handUseRef = useRef(createHandUseTracker({ sustainedFrames: 10 }));

  // Session harvest refs — used to build the save payload without
  // peeking into RepCountShell's internals. sessionStartRef anchors
  // duration, snapshotRef holds the shell's latest {state, score},
  // peakInteriorRef tracks the deepest squat (min interior angle)
  // across the whole session. bestPoseRef holds a coordinate
  // snapshot of the landmarks at the deepest-so-far frame; lastKpRef
  // is the last-known-good pose used as a fallback if no deep rep
  // ever landed. Skeleton is redrawn from coords in the report —
  // NOT saved as a screenshot — so framing is controllable and the
  // payload is tiny (~1.5 KB vs ~30-70 KB JPEG).
  const sessionStartRef = useRef<number>(performance.now());
  const snapshotRef = useRef<{ state: RepCountState; score: Score } | null>(
    null,
  );
  const peakInteriorRef = useRef<number>(180);

  interface PoseSnapshot {
    landmarks: Array<{ x: number; y: number; score: number; name?: string }>;
    source_frame: { width: number; height: number };
  }
  const bestPoseRef = useRef<
    (PoseSnapshot & { angle: number; capturedAtMs: number }) | null
  >(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const sw = video.videoWidth;
      const sh = video.videoHeight;

      // Snapshot the current pose into lastKpRef every frame so
      // buildRehabPayload always has SOMETHING to save even if the
      // patient never crosses topThreshold. Rounded to 2 decimals
      // to keep the payload lean (~1.5 KB for 33 landmarks).
      if (sw > 0 && sh > 0 && kp.length > 0) {
        lastKpRef.current = {
          landmarks: kp.map((p) => ({
            x: Math.round(p.x * 100) / 100,
            y: Math.round(p.y * 100) / 100,
            score: Math.round((p.score ?? 0) * 1000) / 1000,
            ...(p.name ? { name: p.name } : {}),
          })),
          source_frame: { width: sw, height: sh },
        };
      }

      // Feed compensation trackers off the same keypoints.
      roundingRef.current.update(kp);
      handUseRef.current.update(kp);

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
        const interiorAngle = 180 - flexion;
        setInterior(interiorAngle);
        // Track session-wide deepest squat = smallest interior seen.
        // On a new deepest frame that is actually squatting (below
        // topThreshold — filters out standing-still noise), snapshot
        // the landmark coords. The report redraws from these coords
        // rather than saving pixels.
        if (interiorAngle < peakInteriorRef.current) {
          peakInteriorRef.current = interiorAngle;
          if (
            interiorAngle <= activeConfig.topThreshold
            && lastKpRef.current
          ) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: interiorAngle,
              capturedAtMs: performance.now(),
            };
          }
        }
      }
    },
    [side, activeConfig.topThreshold],
  );

  const handleSnapshot = useCallback(
    (state: RepCountState, score: Score) => {
      snapshotRef.current = { state, score };
      // Auto-complete + auto-save when the target reps are reached.
      // Guard against re-entering by only flipping the phase on the
      // upward transition from "live".
      const targetReps = activeConfig.targetReps ?? TARGET_REPS;
      if (state.reps >= targetReps) {
        setSessionPhase((prev) => (prev === "live" ? "complete" : prev));
      }
    },
    [activeConfig.targetReps],
  );

  // Start / reset the session phase machine when the operator picks a
  // side. Resetting on side change also nukes the countdown so a mid-
  // session "Change side" doesn't leave stale scoring.
  useEffect(() => {
    if (side === null) {
      setSessionPhase(null);
      setCountdown(null);
      return;
    }
    setSessionPhase("countdown");
    setCountdown(3);
    sessionStartRef.current = performance.now();
  }, [side]);

  // Countdown tick — decrements once per second, flips to "live" at 0.
  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) {
      setSessionPhase("live");
      setCountdown(null);
      // Session actually starts NOW — reset the anchor so duration in
      // the payload doesn't include the countdown seconds.
      sessionStartRef.current = performance.now();
      return;
    }
    const id = window.setTimeout(() => {
      setCountdown((c) => (c === null ? null : c - 1));
    }, 1000);
    return () => window.clearTimeout(id);
  }, [countdown]);

  const cancelCountdown = useCallback(() => {
    // Skip the countdown → go live immediately. Doctor can still
    // press Space / Escape via the keydown handler below.
    setCountdown(null);
    setSessionPhase("live");
    sessionStartRef.current = performance.now();
  }, []);

  useEffect(() => {
    if (countdown === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === " " || e.code === "Space") {
        e.preventDefault();
        cancelCountdown();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [countdown, cancelCountdown]);

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const snap = snapshotRef.current;
    const state = snap?.state ?? null;
    const score = snap?.score ?? { points: 0, streak: 0, bestStreak: 0 };
    const reps = state?.reps ?? 0;
    const goodReps = state?.goodReps ?? 0;
    const durationSec = Math.max(
      0,
      (performance.now() - sessionStartRef.current) / 1000,
    );
    const interpretation = reps > 0
      ? `${reps} rep${reps === 1 ? "" : "s"} completed`
        + (goodReps !== reps ? `, ${goodReps} clean` : ", all clean")
        + `. Deepest knee interior: ${peakInteriorRef.current.toFixed(0)}°.`
      : "Session ended before any reps were counted.";

    // Skeleton pose: prefer the deepest-rep pose we captured during
    // play. If none was captured (patient stood still, or never
    // dipped below topThreshold), fall back to the last-known-good
    // frame so the report always shows a skeleton.
    const best = bestPoseRef.current;
    const fallback = lastKpRef.current;
    const pose = best
      ? {
          landmarks: best.landmarks,
          source_frame: best.source_frame,
          angle: best.angle,
          captured_at_ms: best.capturedAtMs,
        }
      : fallback
        ? {
            landmarks: fallback.landmarks,
            source_frame: fallback.source_frame,
            angle: peakInteriorRef.current,
            captured_at_ms: performance.now(),
          }
        : null;
    const skeletonPose = pose
      ? {
          landmarks: pose.landmarks,
          source_frame: pose.source_frame,
          angle: pose.angle,
          captured_at_ms: pose.captured_at_ms,
          side,
          label: `Deepest squat — ${pose.angle.toFixed(0)}° knee interior`,
        }
      : null;
    return {
      module: "rehab" as const,
      // Reuse the existing `movement` slot for the exercise slug —
      // matches the audit's decision. `side` maps 1:1 to the shared
      // report column.
      movement: "squat",
      side,
      metrics: {
        exercise_slug: "squat",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: durationSec,
        score,
        mechanic_state: state,
        signal: {
          name: "knee_interior",
          unit: "deg",
          value_at_peak: peakInteriorRef.current,
          target_band: {
            min: activeConfig.depthThreshold,
            max: activeConfig.topThreshold,
          },
        },
        target_reps: TARGET_REPS,
        config: activeConfig,
        level_index: isDoctorFlow ? progression.level : DEFAULT_LEVEL_INDEX,
        compensation_flags: [
          roundingRef.current.finalize(),
          handUseRef.current.finalize(),
        ].filter(Boolean),
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [side, activeConfig, isDoctorFlow, progression.level]);

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
          ) : null}

          {sessionPhase === "complete" && (
            <AutoSaveToast buildPayload={buildRehabPayload} />
          )}

          {side && (
            <LiveModeLayout
              title={`Controlled Squat · ${side === "left" ? "Left" : "Right"} leg`}
              subtitle={
                isDoctorFlow && patient
                  ? `Connected to ${patient.name}'s record${
                      isDoctorFlow
                        ? ` · Level ${progression.level + 1}`
                        : ""
                    }`
                  : `Depth ${activeConfig.depthThreshold}° · Top ${activeConfig.topThreshold}° · Goal ${activeConfig.targetReps ?? TARGET_REPS} reps`
              }
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  autoStart
                  hideControls
                  angleArc={{
                    vertex: side === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                    armA: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                    armB: side === "left" ? LM_LIVE.LEFT_ANKLE : LM_LIVE.RIGHT_ANKLE,
                    currentDeg: interior,
                    band: {
                      min: activeConfig.depthThreshold,
                      max: activeConfig.topThreshold,
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
                  </div>
                  {/* Big 3-2-1 countdown overlay before the session
                      goes live. Cancelable via Space/Escape or the
                      sidebar "Skip countdown" button. */}
                  {sessionPhase === "countdown" && countdown !== null && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/30 backdrop-blur-[1px]">
                      <div className="rounded-full bg-black/70 px-10 py-6 text-center text-white shadow-2xl ring-2 ring-white/20">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-emerald-300">
                          Starting in
                        </p>
                        <p className="tabular text-7xl font-semibold leading-none">
                          {countdown}
                        </p>
                        <p className="mt-2 text-[10px] uppercase tracking-[0.14em] text-white/60">
                          Space / Esc to skip
                        </p>
                      </div>
                    </div>
                  )}
                  {sessionPhase === "complete" && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-emerald-500/20 backdrop-blur-[1px]">
                      <div className="rounded-full bg-emerald-600/90 px-8 py-4 text-center text-white shadow-2xl ring-2 ring-emerald-300/50">
                        <p className="text-sm font-semibold uppercase tracking-[0.14em]">
                          Session complete ✓
                        </p>
                      </div>
                    </div>
                  )}
                </RehabCameraShell>
              )}
              sidebar={(
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-500/15 px-3 py-1 text-xs font-semibold text-indigo-200 ring-1 ring-indigo-400/40">
                      Testing: {side === "left" ? "Left" : "Right"} leg
                    </span>
                    {isDoctorFlow && (
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-200 ring-1 ring-emerald-400/40">
                        Level {progression.level + 1} · {progression.hint}
                      </span>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSide(null)}
                    >
                      Change side
                    </Button>
                  </div>

                  {REHAB_EXERCISE_IMAGES["squat"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={REHAB_EXERCISE_IMAGES["squat"]}
                        alt="Squat reference"
                        loading="lazy"
                        className="block w-full object-contain"
                        style={{ maxHeight: 140 }}
                      />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">
                        Reference form
                      </p>
                    </div>
                  )}

                  {sessionPhase === "countdown" && (
                    <div className="rounded-card border border-accent/40 bg-accent/10 p-3 text-sm">
                      <p className="font-medium text-foreground">
                        Session starts in {countdown ?? 0}s
                      </p>
                      <p className="mt-1 text-[11px] text-muted">
                        Get the patient into the starting position — knee
                        straight, side-on to the camera. Press Space /
                        Escape or use the button below to skip.
                      </p>
                      <div className="mt-3">
                        <Button variant="secondary" size="sm" onClick={cancelCountdown}>
                          Skip countdown
                        </Button>
                      </div>
                    </div>
                  )}

                  {(sessionPhase === "live" || sessionPhase === "complete") && (
                    <div className="flex min-h-0 flex-1 flex-col">
                      <RepCountShell
                        signal={interior}
                        signalLabel={`${side === "left" ? "L" : "R"} knee (°)`}
                        targetReps={activeConfig.targetReps ?? TARGET_REPS}
                        config={activeConfig}
                        onSnapshot={handleSnapshot}
                        compact
                      />
                    </div>
                  )}

                  <div className="no-pdf">
                    {sessionPhase === "complete" ? (
                      isDoctorFlow ? (
                        /* Auto-save fires on mount; the toast shows above
                           the whole page. This inline slot only renders a
                           hint so the doctor knows the session ended. */
                        <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs text-muted">
                          Target reps reached — saving to record automatically.
                        </div>
                      ) : (
                        <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs text-muted">
                          Target reps reached. Nice session.
                        </div>
                      )
                    ) : (
                      <RehabSessionFooter
                        buildPayload={buildRehabPayload}
                        label="Save session"
                        compact
                      />
                    )}
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
