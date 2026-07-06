"use client";
// B6 — Cat-Cow.
//
// Mechanic: Trace (lib/rehab/mechanics.ts traceStep). Cursor in
// normalised [0..1] × [0..1] CSS-y-down — same space TraceShell
// expects. A slow vertical sinusoid path guides the patient
// between CAT (spinal flexion, head tucked) and COW (spinal
// extension, head up).
//
// Cursor mapping — signed spine proxy → vertical cursor:
//   proxy = computeSpineFlexionProxyDeg(kp)  // signed degrees-like
//   • Cat (head dropped):     proxy ≈ +30 → cursor.y near 1 (bottom)
//   • Neutral quadruped:      proxy ≈ 0   → cursor.y = 0.5
//   • Cow (head lifted):      proxy ≈ −30 → cursor.y near 0 (top)
//
//   cursor.y = 0.5 + clamp(proxy / 60, −0.45, +0.45)
//   cursor.x = 0.5 (fixed — single-axis exercise)
//
// Path: vertical cosine at 6 s per cycle.
//   pathFn(t) = { x: 0.5, y: 0.5 + 0.35 * cos(2π · t) }
//   t = 0  → cat target (y ≈ 0.85)
//   t = ½  → cow target (y ≈ 0.15)
//   t = 1  → back to cat
//
// PRD: LOW-CONFIDENCE proxy — head position is a stand-in for
// spinal flexion since BlazePose has no mid-spine landmark. UI
// surfaces "trend only — gentle spinal mobility" caveat.
//
// Reuses (no modifications):
//   • TraceShell, traceStep, RehabCameraShell
//   • computeSpineFlexionProxyDeg — NEW pure fn in poseMetrics
//   • usePatientContext
// NO biomech file modified.

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { TraceShell } from "@/components/rehab/mechanics/TraceShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import { computeSpineFlexionProxyDeg } from "@/lib/rehab/poseMetrics";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import type { TracePathPoint } from "@/lib/rehab/gameState";
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

const PROXY_SCALE = 60;       // ±30 proxy → ±0.5 cursor.y swing
const LOOP_DURATION_MS = 6_000;

function catCowPath(t: number): TracePathPoint {
  // Cosine starts at +1 (t=0) → cat (cursor.y = 0.85), passes
  // through 0 at t=¼, hits −1 at t=½ → cow (cursor.y = 0.15).
  return {
    x: 0.5,
    y: 0.5 + 0.35 * Math.cos(2 * Math.PI * t),
  };
}

const TRACE_CONFIG = {
  accuracyTolerance: 0.10,    // 10 % play-area distance — generous;
                              // proxy is coarse so don't punish small
                              // misalignment
  smoothnessTolerance: 0.001,
  pointsPerSample: 1,
};

export default function CatCowExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [phase, setPhase] = useState<"ready" | "active">("ready");
  const [cursor, setCursor] = useState<{ x: number; y: number }>({
    x: 0.5,
    y: 0.5,
  });
  const [liveProxy, setLiveProxy] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakProxyRef = useRef<number>(0);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const proxy = computeSpineFlexionProxyDeg(
        kp as unknown as LiveKeypoint[],
      );
      if (proxy === null) return;
      setLiveProxy(proxy);
      const absProxy = Math.abs(proxy);
      if (absProxy > peakProxyRef.current) {
        peakProxyRef.current = absProxy;
        if (absProxy >= 5 && lastKpRef.current) {
          bestPoseRef.current = {
            landmarks: lastKpRef.current.landmarks,
            source_frame: lastKpRef.current.source_frame,
            angle: absProxy,
            capturedAtMs: performance.now(),
          };
        }
      }
      const yOffset = Math.max(-0.45, Math.min(0.45, proxy / PROXY_SCALE));
      setCursor({ x: 0.5, y: 0.5 + yOffset });
    },
    [],
  );

  const buildRehabPayload = useCallback((supervised: boolean) => {
    const peak = peakProxyRef.current;
    const interpretation =
      `Cat-Cow trace — peak spine flexion proxy ${peak.toFixed(0)}° across the session.`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peak,
      null,
      `Cat-Cow — peak proxy ${peak.toFixed(0)}°`,
    );
    return {
      module: "rehab" as const,
      movement: "cat-cow",
      metrics: {
        exercise_slug: "cat-cow",
        mechanic_id: "trace",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: null,
        signal: {
          name: "spine_flex_proxy",
          unit: "deg",
          value_at_peak: peak,
        },
        config: TRACE_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        supervised,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, []);

  const phaseHint =
    Math.abs(liveProxy) < 4
      ? "neutral"
      : liveProxy > 0
      ? "cat (flexion)"
      : "cow (extension)";

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>B6 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Cat-Cow<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Gentle spinal-mobility drill from quadruped position
                — alternate between CAT (round the back, chin tucks)
                and COW (arch the back, look up) following a slow
                vertical pacer. Powered by the Trace mechanic.
              </p>
              <div className="mt-5 rounded-card border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p className="font-semibold uppercase tracking-[0.14em] text-amber-200 text-[10px]">
                  Trend only — gentle spinal mobility
                </p>
                <p className="mt-1 text-xs leading-relaxed">
                  BlazePose has no mid-spine landmark, so spinal
                  flexion is inferred from HEAD position relative to
                  the shoulder line (chin tucks ⇒ &quot;cat&quot;, head
                  lifts ⇒ &quot;cow&quot;). Useful as a movement-quality
                  cue and within-patient progress tracker, NOT as an
                  absolute spinal-ROM measurement.
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
                <span className="inline-flex items-center gap-1.5 rounded-full bg-purple-500/15 px-3 py-1 text-xs font-semibold text-purple-200 ring-1 ring-purple-400/40">
                  Cat-Cow · {(LOOP_DURATION_MS / 1000).toFixed(0)}s cycle
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
                  <RehabCameraShell
                    onFrame={handleFrame}
                    angleArc={{
                      vertex: LM_LIVE.LEFT_HIP,
                      armA: LM_LIVE.LEFT_SHOULDER,
                      armB: LM_LIVE.LEFT_KNEE,
                      currentDeg: liveProxy,
                    }}
                  >
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        Spine proxy
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {liveProxy > 0 ? "+" : ""}
                        {liveProxy.toFixed(0)}
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {phaseHint}
                      </p>
                      <p className="mt-1 text-[9px] uppercase tracking-[0.12em] text-amber-200/80">
                        trend only
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <TraceShell
                    cursor={cursor}
                    pathFn={catCowPath}
                    loopDurationMs={LOOP_DURATION_MS}
                    config={TRACE_CONFIG}
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
                Camera at quadruped-height (~50 cm off the floor),
                ~2 m away, perpendicular to the patient&apos;s body
                line — <strong>lateral view</strong>. Both shoulders
                + the head must stay clearly in frame.
              </li>
              <li>
                Patient sets up on hands and knees (quadruped) —
                wrists under shoulders, knees under hips.
              </li>
              <li>
                Follow the vertical pacer: as the lead target
                drifts DOWN on screen, tuck the chin and round the
                back (CAT). As it drifts UP, lift the head and
                arch the back (COW).
              </li>
              <li>
                Tempo is{" "}
                <strong>
                  {(LOOP_DURATION_MS / 1000).toFixed(0)} s per cat-cow
                  cycle
                </strong>
                {" "}— breath-paced, not a strength move.
              </li>
              <li className="text-amber-100/90">
                <strong>Coarse cue:</strong> the readout is a head-
                position proxy, not a spinal-ROM measurement. Use
                it for movement quality and within-patient
                consistency, not absolute angle comparisons.
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
      {REHAB_EXERCISE_IMAGES["cat-cow"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["cat-cow"]}
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
        Quadruped position, lateral to the camera. Set up, then
        begin — follow the vertical pacer at the breath-paced
        cadence.
      </p>
      <div className="mt-6">
        <Button onClick={onStart}>Begin</Button>
      </div>
    </div>
  );
}
