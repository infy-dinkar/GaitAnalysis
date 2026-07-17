"use client";
// S5 — External Rotation at Side (TREND-ONLY proxy).
//
// PRD limitation — this exercise uses a 2-D forearm-position
// PROXY for shoulder external rotation. True gleno-humeral
// rotation is axial and is NOT directly observable in a single
// frontal-view image. The UI surfaces this caveat prominently;
// readings should be interpreted as a WITHIN-PATIENT TREND, not
// as an absolute clinical ER measurement.
//
// Mechanic: Rep-Count (lib/rehab/mechanics.ts repCountStep). Same
// engine + same direction convention as K1 Squat — HIGH at rep
// "top" (rest = forearm neutral, tucked, pointing forward), LOW
// at "depth" (forearm rotated externally to end-range).
//
// Proxy math (computeForearmRotationProxyDeg in lib/rehab/poseMetrics):
//   • Neutral, forearm-forward: lateral ≈ 0 ⇒ proxy ≈ 0°
//   • Full ER, forearm-lateral: lateral ≈ forearm len ⇒ proxy ≈ 90°
//
// Direction flip — like K1's "standing is top of rep":
//   signal = 90 − proxy
//   • Neutral (rest):       signal ≈ 90   → above topThreshold ✓
//   • ER ≈ 40° rotated:     signal ≈ 50   → at depthThreshold
//   • ER ≈ 60° (full):      signal ≈ 30   → below depth ✓
//
// Elbow-drift coaching — inline check: if the elbow moves laterally
// away from the shoulder beyond a body-scaled tolerance, show a
// warning ("keep elbow tucked"). Does NOT void the rep in v1; rep
// quality still ticks. Future versions could gate the rep on this.
//
// Reuses (no modifications):
//   • computeForearmRotationProxyDeg, computeShoulderWidth — both
//     in lib/rehab/poseMetrics (the proxy is NEW this turn; the
//     shoulder-width helper already exists from H3)
//   • RepCountShell, repCountStep, RehabCameraShell
//   • LM_LIVE elbow + shoulder + wrist indices
//   • usePoseDetectionLive, useCamera, usePatientContext
// NO biomech file touched.

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { RepCountShell } from "@/components/rehab/mechanics/RepCountShell";
import {
  AutoFlowCompleteOverlay,
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
  AutoFlowFooter,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  computeForearmRotationProxyDeg,
  computeShoulderWidth,
} from "@/lib/rehab/poseMetrics";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
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

type Side = "left" | "right";

const ER_CONFIG = {
  // Top — signal HIGH at neutral rest. 75 accommodates ±15°
  // natural variance in the resting forearm position.
  topThreshold: 75,
  // Depth — signal LOW after ≥40° of external rotation
  // (signal = 90 − 40 = 50).
  depthThreshold: 50,
  // 25° of rotation excursion is enough to count a real rep —
  // catches noisy / shallow attempts.
  minAmplitude: 25,
  maxJerk: null as number | null,
  pointsPerRep: 10,
};
const TARGET_REPS = 10;

// Elbow-drift coaching — show a warning when the elbow.x deviates
// from the shoulder.x by more than 40% of shoulder-width. Pure
// guidance signal; does NOT gate the rep.
const ELBOW_DRIFT_WARN_FRACTION = 0.4;

export default function ExternalRotationExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default 90 = forearm-neutral. Engine inits, transitions to
  // above_top immediately on first valid frame in rest position.
  const [signal, setSignal] = useState<number>(90);
  // Raw proxy for the on-camera readout (clinical-meaningful
  // value the operator can see).
  const [liveProxy, setLiveProxy] = useState<number>(0);
  // Elbow drift state — boolean coach indicator.
  const [elbowDrifted, setElbowDrifted] = useState<boolean>(false);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const snapshotRef = useRef<{ state: RepCountState; score: Score } | null>(
    null,
  );
  const peakProxyRef = useRef<number>(0);
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);

  // Auto-flow: side pick → 3-2-1 countdown → live → complete →
  // auto-save. Session-scoped refs reset at the live transition so
  // the countdown seconds never count into the payload's duration
  // or trackers.
  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
    markComplete,
  } = useRehabAutoFlow(side !== null, () => {
    peakProxyRef.current = 0;
    bestPoseRef.current = null;
    snapshotRef.current = null;
    sessionStartRef.current = performance.now();
  });

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const proxy = computeForearmRotationProxyDeg(liveKp, side);
      if (proxy !== null) {
        setLiveProxy(proxy);
        // Flip so signal is HIGH at neutral (rep "top") — matches
        // K1's rest-is-top convention.
        setSignal(90 - proxy);
        if (proxy > peakProxyRef.current) {
          peakProxyRef.current = proxy;
          // Guard: patient has actually rotated outward (clinical
          // rep-engagement zone) before we snapshot the pose.
          if (proxy >= 25 && lastKpRef.current) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: proxy,
              capturedAtMs: performance.now(),
            };
          }
        }
      }
      // Elbow drift coaching — only if both landmarks visible.
      const shoulder =
        liveKp[side === "right" ? LM.RIGHT_SHOULDER : LM.LEFT_SHOULDER];
      const elbow =
        liveKp[side === "right" ? LM.RIGHT_ELBOW : LM.LEFT_ELBOW];
      const sw = computeShoulderWidth(liveKp);
      if (shoulder && elbow && sw && sw > 1) {
        const drift = Math.abs(elbow.x - shoulder.x) / sw;
        setElbowDrifted(drift > ELBOW_DRIFT_WARN_FRACTION);
      }
    },
    [side],
  );

  const handleSnapshot = useCallback(
    (state: RepCountState, score: Score) => {
      snapshotRef.current = { state, score };
      if (state.reps >= TARGET_REPS) markComplete();
    },
    [markComplete],
  );

  const buildRehabPayload = useCallback(() => {
    if (!side) return null;
    const snap = snapshotRef.current;
    const state = snap?.state ?? null;
    const score = snap?.score ?? { points: 0, streak: 0, bestStreak: 0 };
    const reps = state?.reps ?? 0;
    const goodReps = state?.goodReps ?? 0;
    const interpretation = reps > 0
      ? `${reps} ER rep${reps === 1 ? "" : "s"} completed`
        + (goodReps !== reps ? `, ${goodReps} clean` : ", all clean")
        + `. Peak external-rotation proxy: ${peakProxyRef.current.toFixed(0)}°.`
      : "Session ended before any reps were counted.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peakProxyRef.current,
      side,
      `Peak external rotation — ${peakProxyRef.current.toFixed(0)}° proxy`,
    );
    return {
      module: "rehab" as const,
      movement: "external-rotation",
      side,
      metrics: {
        exercise_slug: "external-rotation",
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score,
        mechanic_state: state,
        signal: {
          name: "forearm_rotation_proxy",
          unit: "deg",
          value_at_peak: peakProxyRef.current,
          target_band: {
            min: ER_CONFIG.depthThreshold,
            max: ER_CONFIG.topThreshold,
          },
        },
        target_reps: TARGET_REPS,
        config: ER_CONFIG,
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
              <Badge>S5 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                External Rotation<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Elbow-at-side external rotation rep counter. Patient
                stands or sits with the upper arm tucked, elbow at
                90° flexion, rotates the forearm outward to end-
                range, returns to neutral. Each cycle = one rep.
                Powered by the Rep-Count mechanic.
              </p>
              {/* PRD-mandated trend-only caveat. */}
              <div className="mt-5 rounded-card border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p className="font-semibold uppercase tracking-[0.14em] text-amber-200 text-[10px]">
                  Trend only — not a precise rotation measurement
                </p>
                <p className="mt-1 text-xs leading-relaxed">
                  True shoulder external rotation is an axial motion
                  that cannot be isolated in a single 2-D frontal
                  view. The angle shown is a forearm-position
                  PROXY — useful for tracking within-patient
                  progress session to session, NOT for comparing
                  against absolute clinical ER ROM values.
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

          {!side ? <SidePicker onPick={setSide} /> : null}

          {side && (
            <LiveModeLayout
              title={`External Rotation · ${side === "left" ? "Left" : "Right"} arm`}
              subtitle={isDoctorFlow && patient ? `Connected to ${patient.name}'s record.` : `Goal ${TARGET_REPS} reps`}
              onExit={() => setSide(null)}
              camera={(
                <RehabCameraShell onFrame={handleFrame} autoStart hideControls>
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">{side === "left" ? "L" : "R"} ER · proxy</p>
                    <p className="tabular text-2xl font-semibold text-white">{liveProxy.toFixed(0)}°</p>
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
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-500/15 px-3 py-1 text-xs font-semibold text-indigo-200 ring-1 ring-indigo-400/40">{side === "left" ? "Left" : "Right"} arm</span>
                    {elbowDrifted && (
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-500/20 px-3 py-1 text-xs font-semibold text-rose-200 ring-1 ring-rose-400/50">Elbow</span>
                    )}
                    <Button variant="ghost" size="sm" onClick={() => setSide(null)}>Change side</Button>
                  </div>
                  {REHAB_EXERCISE_IMAGES["external-rotation"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={REHAB_EXERCISE_IMAGES["external-rotation"]} alt="External Rotation reference" loading="lazy" className="block w-full object-contain" style={{ maxHeight: 140 }} />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">Reference form</p>
                    </div>
                  )}
                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownCard
                      countdown={countdown}
                      onSkip={skipCountdown}
                      hint="Patient facing the camera, elbow tucked, forearm forward."
                    />
                  )}
                  {(sessionPhase === "live" || sessionPhase === "complete") && (
                    <div className="flex min-h-0 flex-1 flex-col">
                      <RepCountShell signal={signal} signalLabel={`${side === "left" ? "L" : "R"} ER (°)`} targetReps={TARGET_REPS} config={ER_CONFIG} onSnapshot={handleSnapshot} compact />
                    </div>
                  )}
                  <div className="no-pdf"><AutoFlowFooter complete={sessionPhase === "complete"} buildPayload={buildRehabPayload} /></div>
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
                shoulders, the working elbow, and the working wrist
                must stay in frame throughout the rep.
              </li>
              <li>
                Patient stands or sits with the working{" "}
                <strong>upper arm pressed against the body</strong>,
                elbow bent to 90°, forearm pointing forward at the
                start (rest position).
              </li>
              <li>
                Keep the elbow <strong>tucked to the side</strong>.
                The on-screen &quot;Elbow drifted&quot; chip lights up if
                the elbow wanders laterally — coaching cue only, the
                rep still counts. Future versions may gate.
              </li>
              <li>
                Rotate the forearm <strong>outward</strong> (external
                rotation) — wrist swings away from the body
                midline — to end-range. Hold briefly, return to
                neutral.
              </li>
              <li>
                Rep closes on return to neutral. Excursion gate:
                rotations smaller than{" "}
                <strong>{ER_CONFIG.minAmplitude}°</strong> of proxy
                change are flagged as shallow.
              </li>
              <li className="text-amber-100/90">
                <strong>Trend only:</strong> the angle shown is a
                2-D proxy, not absolute gleno-humeral ER. Use it to
                compare a patient&apos;s own performance over time,
                not against clinical ER norms.
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
      {REHAB_EXERCISE_IMAGES["external-rotation"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["external-rotation"]}
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
        Pick the arm the patient will externally rotate. We track
        that side&apos;s elbow + wrist position to derive the
        forearm rotation proxy every frame.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
