"use client";
// B2 — Back Extension.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). The
// engine wants HIGH at the rep "top" (extended position) and LOW
// at "depth" (back to neutral upright).
//
// Proxy math (computeTrunkExtensionAngleDeg in poseMetrics):
//   Returns the MAGNITUDE of trunk-tilt from vertical-up.
//     • Upright neutral:  ~0°
//     • Mid-extension:    ~7-12°
//     • End-range:        ~15-25°
//
// Already HIGH at extended ⇒ fed directly, NO flip needed.
// (The patient is instructed to extend BACKWARD only — forward
// flexion is not part of this drill, so the unsigned magnitude
// cleanly tracks the extension arc.)
//
// Bilateral movement — no side picker. PRD: SMALL controlled range
// only. Thresholds tuned for the modest active back-extension ROM.
//
// Reuses (no modifications):
//   • RepCountShell, repCountStep, RehabCameraShell
//   • computeTrunkExtensionAngleDeg — NEW pure fn in poseMetrics
//   • usePatientContext

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { computeTrunkExtensionAngleDeg } from "@/lib/rehab/poseMetrics";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import type { RepCountState, Score } from "@/lib/rehab/gameState";
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

const BACK_EXT_CONFIG = {
  // Small controlled extension range — top needs ≥ 12° of trunk
  // tilt from vertical (a real arch, not just postural sway).
  topThreshold: 12,
  // Back-near-upright counts as "depth" / rest position.
  depthThreshold: 5,
  // 7° excursion — distinguishes a deliberate extension cycle from
  // body sway / breathing wobble (typically < 3°).
  minAmplitude: 7,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 8;

export default function BackExtensionExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  // Single bilateral exercise — small "Ready → Active" gate so the
  // reference image renders before the camera engages, mirroring the
  // side-picker pattern on other pages.
  const [phase, setPhase] = useState<"ready" | "active">("ready");
  const [trunkAngle, setTrunkAngle] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const snapshotRef = useRef<{ state: RepCountState; score: Score } | null>(
    null,
  );
  const peakExtensionRef = useRef<number>(0);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const angle = computeTrunkExtensionAngleDeg(
        kp as unknown as LiveKeypoint[],
      );
      if (angle !== null) {
        setTrunkAngle(angle);
        if (angle > peakExtensionRef.current) {
          peakExtensionRef.current = angle;
          if (
            angle >= BACK_EXT_CONFIG.depthThreshold
            && lastKpRef.current
          ) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle,
              capturedAtMs: performance.now(),
            };
          }
        }
      }
    },
    [],
  );

  const handleSnapshot = useCallback(
    (state: RepCountState, score: Score) => {
      snapshotRef.current = { state, score };
    },
    [],
  );

  const buildRehabPayload = useCallback((supervised: boolean) => {
    const snap = snapshotRef.current;
    const state = snap?.state ?? null;
    const score = snap?.score ?? { points: 0, streak: 0, bestStreak: 0 };
    const reps = state?.reps ?? 0;
    const goodReps = state?.goodReps ?? 0;
    const interpretation = reps > 0
      ? `${reps} back-extension${reps === 1 ? "" : "s"} completed`
        + (goodReps !== reps ? `, ${goodReps} clean` : ", all clean")
        + `. Peak trunk extension: ${peakExtensionRef.current.toFixed(0)}°.`
      : "Session ended before any reps were counted.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peakExtensionRef.current,
      null,
      `Peak back-extension — ${peakExtensionRef.current.toFixed(0)}° trunk arch`,
    );
    return {
      module: "rehab" as const,
      movement: "back-extension",
      metrics: {
        exercise_slug: "back-extension",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score,
        mechanic_state: state,
        signal: {
          name: "trunk_extension",
          unit: "deg",
          value_at_peak: peakExtensionRef.current,
          target_band: {
            min: BACK_EXT_CONFIG.depthThreshold,
            max: BACK_EXT_CONFIG.topThreshold,
          },
        },
        target_reps: TARGET_REPS,
        config: BACK_EXT_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        supervised,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, []);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>B2 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Back Extension<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Standing or prone back-extension rep counter. Patient
                arches the trunk gently backward through a small
                controlled range, returns to neutral. Each
                extension-and-return = one rep. Powered by the
                Rep-Count mechanic.
              </p>
              <div className="mt-5 rounded-card border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p className="font-semibold uppercase tracking-[0.14em] text-amber-200 text-[10px]">
                  Small pain-free range only
                </p>
                <p className="mt-1 text-xs leading-relaxed">
                  Back extension stresses the lumbar spine — use a
                  modest ROM (≤ 25° from vertical). Stop immediately
                  if any pain, pinching, or radicular symptoms occur.
                  This drill is unsuitable for acute disc or facet
                  pathology unless cleared by the treating clinician.
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

          {phase === "ready" ? (
            <ReadyGate onStart={() => setPhase("active")} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-500/15 px-3 py-1 text-xs font-semibold text-indigo-200 ring-1 ring-indigo-400/40">
                  Back extension · bilateral
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setPhase("ready")}
                >
                  Show reference
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
                        {trunkAngle >= BACK_EXT_CONFIG.topThreshold
                          ? "extended"
                          : trunkAngle <= BACK_EXT_CONFIG.depthThreshold
                          ? "upright"
                          : "transition"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <RepCountShell
                    signal={trunkAngle}
                    signalLabel="Trunk extension (°)"
                    targetReps={TARGET_REPS}
                    config={BACK_EXT_CONFIG}
                    onSnapshot={handleSnapshot}
                  />
                </div>
              </div>

              <div className="no-pdf">
                <RehabSessionFooter
                  buildPayload={buildRehabPayload}
                  label="Save rehab session"
                />
              </div>
            </div>
          )}

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip / chest height, ~2 m away, perpendicular
                to the body — <strong>lateral view</strong>. Both
                shoulders and both hips must stay clearly in frame
                throughout the rep.
              </li>
              <li>
                Patient stands upright with feet shoulder-width apart
                (or in prone press-up position with elbows under
                shoulders for the floor variant).
              </li>
              <li>
                Arch the spine <strong>gently backward</strong> — the
                trunk-tilt readout climbs past{" "}
                <strong>{BACK_EXT_CONFIG.topThreshold}°</strong>{" "}
                (status flips to &quot;extended&quot;). Hold briefly.
              </li>
              <li>
                Return to upright — readout drops below{" "}
                <strong>{BACK_EXT_CONFIG.depthThreshold}°</strong>{" "}
                — and the rep closes.
              </li>
              <li>
                Excursion gate: extensions smaller than{" "}
                <strong>{BACK_EXT_CONFIG.minAmplitude}°</strong> are
                flagged as shallow.
              </li>
              <li>
                Engine note: the first extension primes the rep state
                machine — count one extra lift if you want exactly{" "}
                {TARGET_REPS} counted.
              </li>
              <li className="text-amber-100/90">
                <strong>Stop if anything hurts.</strong> Range stays
                pain-free; this is a mobility drill, not a strength
                test.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function ReadyGate({ onStart }: { onStart: () => void }) {
  return (
    <div className="mt-10 max-w-xl">
      {REHAB_EXERCISE_IMAGES["back-extension"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["back-extension"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Ready when you are
      </h2>
      <p className="mt-2 text-sm text-muted">
        Bilateral movement — no side picker. Set up lateral to the
        camera, then begin. Keep the range comfortable.
      </p>
      <div className="mt-6">
        <Button onClick={onStart}>Begin</Button>
      </div>
    </div>
  );
}
