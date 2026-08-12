"use client";
// B6 — Cat-Cow (rep counter).
//
// Mechanic: Rep-count (one full CAT ↔ COW cycle = 1 rep). Replaces the
// old follow-the-pacer Trace game — that was latency-sensitive and
// scored "how well you chased a dot" instead of the actual movement.
//
// Signal: signed spine-flexion proxy (computeSpineFlexionProxyDeg) —
// nose position relative to the shoulder line, since BlazePose has no
// mid-spine landmark.
//   • Cat (round the back, chin tucks, head DROPS): proxy ≈ +30..+45
//   • Neutral quadruped:                            proxy ≈ 0
//   • Cow (arch the back, head LIFTS):              proxy ≈ −30
//
// One rep = the patient reaches BOTH the cat extreme (proxy ≥ +REACH)
// AND the cow extreme (proxy ≤ −REACH) — order doesn't matter. When
// both have been touched, rep++ and the visited flags reset. Auto-
// saves at TARGET_REPS. EMA smoothing rejects the coarse proxy's
// frame-to-frame jitter so a wobble near the threshold can't double-
// count.
//
// PRD: LOW-CONFIDENCE proxy — head position is a stand-in for spinal
// flexion. UI surfaces the "trend only — gentle spinal mobility"
// caveat; the rep count is a movement-completion tally, not a ROM
// measurement.
//
// Reuses (no modifications):
//   • RehabCameraShell, useRehabAutoFlow, LiveModeLayout
//   • computeSpineFlexionProxyDeg (existing poseMetrics helper)
//   • usePatientContext
// NO biomech file modified.

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
import { computeSpineFlexionProxyDeg } from "@/lib/rehab/poseMetrics";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
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

// Auto-saves at TARGET_REPS. Tune REP_REACH on camera if needed —
// cat reaches ~+45, cow ~−30, so ±15 leaves comfortable headroom.
const TARGET_REPS = 10;
const REP_REACH = 15;
// Physiological ceiling — real cat/cow proxy stays within ~±45°.
// Anything beyond is a pose glitch (e.g. shoulder-width collapsing to
// near-zero blows up the nose/width ratio → tens-of-thousands of
// "degrees"). Reject those frames before they poison the peak / reps.
const PROXY_MAX = 80;
// Visual axis: proxy → vertical position of the marker (0 top = cow,
// 1 bottom = cat). ±30 proxy maps to the ends of the visible track.
const AXIS_SCALE = 60;

function fmtClock(totalSec: number): string {
  const s = Math.max(0, Math.round(totalSec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

export default function CatCowExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [phase, setPhase] = useState<"ready" | "active">("ready");
  const [liveProxy, setLiveProxy] = useState<number>(0);
  const [reps, setReps] = useState<number>(0);
  const [visited, setVisited] = useState<{ cat: boolean; cow: boolean }>({
    cat: false,
    cow: false,
  });
  const [elapsedSec, setElapsedSec] = useState<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakProxyRef = useRef<number>(0);
  const repsCountRef = useRef<number>(0);
  const visitedRef = useRef<{ cat: boolean; cow: boolean }>({
    cat: false,
    cow: false,
  });
  // EMA-smoothed proxy — the head-position proxy is coarse, so smooth
  // out single-frame jitter before the rep edge-detection reads it.
  const smoothProxyRef = useRef<number | null>(null);

  // Auto-flow: Begin → 3-2-1 countdown → live → (TARGET_REPS) →
  // complete → auto-save. Session-scoped refs reset at the live
  // transition so countdown framing noise never leaks into the tally.
  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
    markComplete,
  } = useRehabAutoFlow(phase === "active", () => {
    bestPoseRef.current = null;
    peakProxyRef.current = 0;
    repsCountRef.current = 0;
    visitedRef.current = { cat: false, cow: false };
    smoothProxyRef.current = null;
    setReps(0);
    setVisited({ cat: false, cow: false });
    setElapsedSec(0);
    sessionStartRef.current = performance.now();
  });

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const rawProxy = computeSpineFlexionProxyDeg(
        kp as unknown as LiveKeypoint[],
      );
      if (rawProxy === null) return;
      // Reject pose-glitch frames (implausible proxy) before anything
      // reads them — a single bad frame otherwise locks the peak at an
      // absurd value (e.g. 95 580°) and can false-trip a rep extreme.
      if (Math.abs(rawProxy) > PROXY_MAX) return;
      // EMA smooth.
      const prev = smoothProxyRef.current;
      const proxy = prev === null ? rawProxy : prev * 0.65 + rawProxy * 0.35;
      smoothProxyRef.current = proxy;
      setLiveProxy(proxy);

      // Best skeleton frame = deepest cat/cow (largest |proxy|).
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

      // Rep counting only while live.
      if (sessionPhase !== "live") return;
      let changed = false;
      // Cat extreme (head dropped, proxy positive).
      if (proxy >= REP_REACH && !visitedRef.current.cat) {
        visitedRef.current.cat = true;
        changed = true;
      }
      // Cow extreme (head lifted, proxy negative).
      if (proxy <= -REP_REACH && !visitedRef.current.cow) {
        visitedRef.current.cow = true;
        changed = true;
      }
      // Both extremes touched → one full cat↔cow cycle = 1 rep.
      if (
        visitedRef.current.cat
        && visitedRef.current.cow
        && repsCountRef.current < TARGET_REPS
      ) {
        repsCountRef.current += 1;
        setReps(repsCountRef.current);
        visitedRef.current = { cat: false, cow: false };
        changed = true;
        if (repsCountRef.current >= TARGET_REPS) markComplete();
      }
      if (changed) setVisited({ ...visitedRef.current });
    },
    [sessionPhase, markComplete],
  );

  // Elapsed timer — how long the patient took to hit the target.
  useEffect(() => {
    if (sessionPhase !== "live") return;
    const id = window.setInterval(() => {
      setElapsedSec(elapsedSecondsSince(sessionStartRef.current));
    }, 1000);
    return () => window.clearInterval(id);
  }, [sessionPhase]);

  const buildRehabPayload = useCallback(() => {
    const peak = peakProxyRef.current;
    const done = repsCountRef.current;
    const interpretation =
      done > 0
        ? `Cat-Cow: ${done} full cat↔cow cycle${done === 1 ? "" : "s"} completed. Peak spine-flexion proxy ${peak.toFixed(0)}° (trend-only head-position proxy).`
        : "Session ended before a full cat↔cow cycle was completed.";
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
        mechanic_id: "rep_count",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        target_reps: TARGET_REPS,
        // rep_count report reads reps/goodReps from mechanic_state.
        mechanic_state: { reps: done, goodReps: done },
        signal: {
          name: "spine_flex_proxy",
          unit: "deg",
          value_at_peak: peak,
        },
        config: { targetReps: TARGET_REPS, repReach: REP_REACH },
        level_index: DEFAULT_LEVEL_INDEX,
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

  // Marker position on the vertical track: 0 = top (cow), 1 = bottom
  // (cat). Clamped so it never leaves the visible rail.
  const markerFrac = Math.max(
    0,
    Math.min(1, 0.5 + liveProxy / AXIS_SCALE),
  );

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
                Gentle spinal-mobility drill from quadruped position —
                alternate between CAT (round the back, chin tucks) and
                COW (arch the back, look up) at your own pace. One full
                cat↔cow cycle = one rep; the session auto-saves after{" "}
                {TARGET_REPS} reps.
              </p>
              <div className="mt-5 rounded-card border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p className="font-semibold uppercase tracking-[0.14em] text-amber-200 text-[10px]">
                  Trend only — gentle spinal mobility
                </p>
                <p className="mt-1 text-xs leading-relaxed">
                  BlazePose has no mid-spine landmark, so spinal
                  flexion is inferred from HEAD position relative to
                  the shoulder line (chin tucks ⇒ &quot;cat&quot;, head
                  lifts ⇒ &quot;cow&quot;). The rep count is a
                  movement-completion tally, NOT an absolute spinal-ROM
                  measurement.
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

          {phase === "ready" ? <ReadyGate onStart={() => setPhase("active")} /> : null}

          {phase !== "ready" && (
            <LiveModeLayout
              title="Cat-Cow"
              subtitle={
                isDoctorFlow && patient
                  ? `Connected to ${patient.name}'s record.`
                  : `${TARGET_REPS} reps · own pace`
              }
              onExit={() => setPhase("ready")}
              camera={(
                <RehabCameraShell onFrame={handleFrame} autoStart hideControls>
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                      Spine proxy
                    </p>
                    <p className="tabular text-2xl font-semibold text-white">
                      {liveProxy > 0 ? "+" : ""}
                      {liveProxy.toFixed(0)}
                    </p>
                    <p className="mt-1 text-[10px] text-zinc-300">{phaseHint}</p>
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
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-purple-500/15 px-3 py-1 text-xs font-semibold text-purple-200 ring-1 ring-purple-400/40">
                      Cat-Cow
                    </span>
                    {(sessionPhase === "live" || sessionPhase === "complete") && (
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-500/15 px-3 py-1 text-xs font-semibold text-sky-200 ring-1 ring-sky-400/40 tabular">
                        {fmtClock(elapsedSec)}
                      </span>
                    )}
                    <Button variant="ghost" size="sm" onClick={() => setPhase("ready")}>
                      Restart
                    </Button>
                  </div>

                  {(sessionPhase === "live" || sessionPhase === "complete") && (
                    <div className="rounded-card border border-border bg-surface p-4">
                      <div className="flex items-end justify-between">
                        <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
                          Reps
                        </p>
                        <p className="text-[10px] text-muted">
                          {sessionPhase === "complete" ? "Complete" : "own pace"}
                        </p>
                      </div>
                      <div className="mt-1 flex items-baseline gap-2">
                        <span className="tabular text-5xl font-semibold text-foreground">
                          {reps}
                        </span>
                        <span className="text-lg text-muted">/ {TARGET_REPS}</span>
                      </div>

                      {/* Cat / cow cycle progress for the current rep */}
                      <div className="mt-3 flex gap-2">
                        <span
                          className={`flex-1 rounded-md px-2 py-1.5 text-center text-xs font-semibold ring-1 transition-colors ${
                            visited.cow
                              ? "bg-emerald-500/20 text-emerald-200 ring-emerald-400/50"
                              : "bg-elevated text-subtle ring-border"
                          }`}
                        >
                          COW ▲ {visited.cow ? "✓" : ""}
                        </span>
                        <span
                          className={`flex-1 rounded-md px-2 py-1.5 text-center text-xs font-semibold ring-1 transition-colors ${
                            visited.cat
                              ? "bg-emerald-500/20 text-emerald-200 ring-emerald-400/50"
                              : "bg-elevated text-subtle ring-border"
                          }`}
                        >
                          CAT ▼ {visited.cat ? "✓" : ""}
                        </span>
                      </div>

                      {/* Vertical position rail — cow at top, cat at bottom */}
                      <div className="mt-3 flex items-stretch gap-3">
                        <div className="relative h-40 w-3 rounded-full bg-elevated">
                          {/* cow zone (top) */}
                          <div className="absolute inset-x-0 top-0 h-1/4 rounded-t-full bg-sky-500/25" />
                          {/* cat zone (bottom) */}
                          <div className="absolute inset-x-0 bottom-0 h-1/4 rounded-b-full bg-fuchsia-500/25" />
                          {/* marker */}
                          <div
                            className="absolute left-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-sky-400 ring-2 ring-white/70"
                            style={{ top: `${markerFrac * 100}%` }}
                          />
                        </div>
                        <div className="flex flex-col justify-between py-0.5 text-[10px] uppercase tracking-[0.12em] text-subtle">
                          <span>Cow (head up)</span>
                          <span className="text-muted">neutral</span>
                          <span>Cat (head down)</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {REHAB_EXERCISE_IMAGES["cat-cow"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={REHAB_EXERCISE_IMAGES["cat-cow"]}
                        alt="Cat-Cow reference"
                        loading="lazy"
                        className="block w-full object-contain"
                        style={{ maxHeight: 140 }}
                      />
                      <p className="border-t border-border bg-surface px-2 py-1 text-center text-[10px] uppercase tracking-[0.12em] text-muted">
                        Reference form
                      </p>
                    </div>
                  )}

                  {sessionPhase === "countdown" && countdown !== null && (
                    <AutoFlowCountdownCard
                      countdown={countdown}
                      onSkip={skipCountdown}
                      hint="Patient on hands and knees, side-on to the camera."
                    />
                  )}

                  <div className="no-pdf">
                    <AutoFlowFooter
                      complete={sessionPhase === "complete"}
                      buildPayload={buildRehabPayload}
                    />
                  </div>
                </>
              )}
            />
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
                <strong>One rep = one full cat↔cow cycle.</strong> Lift
                the head and arch the back (COW ▲), then tuck the chin
                and round the back (CAT ▼). Once BOTH ends are reached
                the rep counts — the COW / CAT chips tick as you hit
                each end.
              </li>
              <li>
                Move at your own breath-paced tempo — no dot to chase.
                The session auto-saves after{" "}
                <strong>{TARGET_REPS} reps</strong>.
              </li>
              <li className="text-amber-100/90">
                <strong>Coarse cue:</strong> the readout is a head-
                position proxy, not a spinal-ROM measurement. Use it
                for movement quality and within-patient consistency,
                not absolute angle comparisons.
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
        Quadruped position, lateral to the camera. Set up, then begin —
        move between cat and cow at your own breath-paced tempo. One
        full cat↔cow cycle counts as a rep.
      </p>
      <div className="mt-6">
        <Button onClick={onStart}>Begin</Button>
      </div>
    </div>
  );
}
