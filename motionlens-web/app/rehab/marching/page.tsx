"use client";
// H5 — Marching (bilateral, timed, angle-tracking).
//
// Patient marches in place with BOTH knees for a fixed duration
// (5 / 10 / 15 min). No beat grading — the session simply tracks each
// knee lift and records its PEAK hip + knee flexion, averaged PER SIDE
// for the report (left-hip, right-hip, left-knee, right-knee). A
// looping "marching beat" track plays as an optional audio pacing cue.
//
// Session is TIME-BOXED: the timer counts down the picked duration and
// auto-completes (auto-saves) when it hits zero.
//
// Knee-lift detection (per leg, independent):
//   computeHipAngle("flexion", kp, side) rises during a lift.
//   Edge-detect with threshold + hysteresis (LIFT 35° → RESET 15°)
//   so a single lift is counted exactly once. Each lift's peak hip +
//   knee flexion is captured on the falling edge and averaged per side.
//
// Audio:
//   • Plain HTML5 <audio> element owned by this PAGE, looping
//     /audio/rehab/marching-beat.mp3. Default ON, mute toggle in UI.
//   • The duration-picker click UNLOCKS the element (muted play→pause,
//     autoplay policy); the audible play() fires at the countdown→live
//     transition so the music only starts once counting finishes.
//
// Reuses (no modifications):
//   • RehabCameraShell, useRehabAutoFlow, LiveModeLayout
//   • computeHipAngle + computeKneeAngle (biomech — IMPORT ONLY)
//   • computePelvicTiltDeg (existing rehab helper)
//   • usePatientContext

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Volume2, VolumeX } from "lucide-react";
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
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import { computeHipAngle } from "@/lib/biomech/hip-live";
import { computeKneeAngle } from "@/lib/biomech/knee-live";
import { computePelvicTiltDeg } from "@/lib/rehab/poseMetrics";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const MUSIC_URL = "/audio/rehab/marching-beat.mp3";

// Hip-flexion thresholds for knee-lift edge detection.
const LIFT_THRESHOLD_DEG = 35;
const RESET_THRESHOLD_DEG = 15;

// Physiological ceilings — readings above these are pose glitches (a
// landmark jumping for a frame), NOT real flexion. Rejected before
// smoothing so a single bad frame can't inflate the peak / averages
// (e.g. the 179.7° "hip flexion" a mis-tracked knee/shoulder produces).
const HIP_FLEX_MAX_DEG = 130;
const KNEE_FLEX_MAX_DEG = 150;
// EMA weight for the incoming raw sample (rest = previous smoothed).
const SMOOTH_ALPHA = 0.4;

// Pelvic-tilt coaching threshold.
const PELVIS_TILT_WARN_DEG = 8;

// Session-length options (minutes).
const DURATION_OPTIONS = [5, 10, 15] as const;

// Music source tempo — kept for the config payload only (no beat
// grading on this exercise anymore).
const MUSIC_BPM = 90;

// Per-side running accumulators for the peak-per-lift averages.
type SideStats = { hipSum: number; kneeSum: number; count: number };
const emptyStats = (): { left: SideStats; right: SideStats } => ({
  left: { hipSum: 0, kneeSum: 0, count: 0 },
  right: { hipSum: 0, kneeSum: 0, count: 0 },
});

function fmtClock(totalSec: number): string {
  const s = Math.max(0, Math.round(totalSec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

export default function MarchingExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [durationMin, setDurationMin] = useState<number | null>(null);
  const [pelvisDrifted, setPelvisDrifted] = useState<boolean>(false);
  const [musicOn, setMusicOn] = useState<boolean>(true);
  const [timeLeftSec, setTimeLeftSec] = useState<number>(0);
  const [liftsDisplay, setLiftsDisplay] = useState<number>(0);

  // Live per-leg readouts (display only).
  const [liveLeftHip, setLiveLeftHip] = useState<number>(0);
  const [liveRightHip, setLiveRightHip] = useState<number>(0);
  const [liveLeftKnee, setLiveLeftKnee] = useState<number>(0);
  const [liveRightKnee, setLiveRightKnee] = useState<number>(0);

  // Edge-detector state per leg — true while that knee is in the
  // lifted position (after crossing LIFT_THRESHOLD).
  const inLiftedRef = useRef<{ left: boolean; right: boolean }>({
    left: false,
    right: false,
  });
  // Peak hip + knee flexion of the lift currently in progress (per leg).
  const liftPeakRef = useRef<{
    left: { hip: number; knee: number };
    right: { hip: number; knee: number };
  }>({ left: { hip: 0, knee: 0 }, right: { hip: 0, knee: 0 } });
  // Completed-lift peak sums + counts, per side → averages at save.
  const statsRef = useRef(emptyStats());
  // EMA-smoothed hip/knee per side — rejects single-frame jitter so a
  // glitch frame can't spike a lift's recorded peak.
  const smoothRef = useRef<{
    left: { hip: number | null; knee: number | null };
    right: { hip: number | null; knee: number | null };
  }>({ left: { hip: null, knee: null }, right: { hip: null, knee: null } });

  // HTMLAudioElement instance reference.
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const peakFlexionRef = useRef<number>(0);
  const liftCountRef = useRef<number>(0);

  const { patient, isDoctorFlow } = usePatientContext();

  // Auto-flow: duration pick → 3-2-1 countdown → live → (timer) →
  // complete → auto-save. Music stays silent through the countdown;
  // the audible play() fires HERE at the countdown→live transition
  // (the duration-pick click only UNLOCKED the element). All
  // session-scoped refs reset at the same transition so countdown
  // framing lifts never count into the payload.
  const {
    phase: sessionPhase,
    countdown,
    skipCountdown,
    markComplete,
  } = useRehabAutoFlow(durationMin !== null, () => {
    inLiftedRef.current = { left: false, right: false };
    liftPeakRef.current = {
      left: { hip: 0, knee: 0 },
      right: { hip: 0, knee: 0 },
    };
    statsRef.current = emptyStats();
    smoothRef.current = {
      left: { hip: null, knee: null },
      right: { hip: null, knee: null },
    };
    bestPoseRef.current = null;
    peakFlexionRef.current = 0;
    liftCountRef.current = 0;
    setLiftsDisplay(0);
    setTimeLeftSec((durationMin ?? 0) * 60);
    sessionStartRef.current = performance.now();
    // Start the music now — counting is done, live has begun.
    if (musicOn && audioRef.current) {
      audioRef.current.currentTime = 0;
      void audioRef.current.play().catch(() => {});
    }
  });

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (durationMin === null) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const live = sessionPhase === "live";

      for (const s of ["left", "right"] as const) {
        const rawHip = computeHipAngle("flexion", liveKp, s);
        const rawKnee = computeKneeAngle("flexion", liveKp, s);
        const sm = smoothRef.current[s];
        // Reject pose-glitch frames (implausible angles), then EMA-
        // smooth. sm.hip / sm.knee hold the previous smoothed value and
        // are only nudged by a plausible reading.
        if (rawHip !== null && rawHip >= 0 && rawHip <= HIP_FLEX_MAX_DEG) {
          sm.hip =
            sm.hip === null
              ? rawHip
              : sm.hip * (1 - SMOOTH_ALPHA) + rawHip * SMOOTH_ALPHA;
        }
        if (rawKnee !== null && rawKnee >= 0 && rawKnee <= KNEE_FLEX_MAX_DEG) {
          sm.knee =
            sm.knee === null
              ? rawKnee
              : sm.knee * (1 - SMOOTH_ALPHA) + rawKnee * SMOOTH_ALPHA;
        }
        const hip = sm.hip;
        const knee = sm.knee;
        // Live display always updates (even during the countdown).
        if (hip !== null) {
          if (s === "left") setLiveLeftHip(hip);
          else setLiveRightHip(hip);
        }
        if (knee !== null) {
          if (s === "left") setLiveLeftKnee(knee);
          else setLiveRightKnee(knee);
        }
        // Counting + averaging only while the session is live.
        if (!live || hip === null) continue;

        // Best skeleton frame = overall deepest hip flexion.
        if (hip > peakFlexionRef.current) {
          peakFlexionRef.current = hip;
          if (hip >= LIFT_THRESHOLD_DEG && lastKpRef.current) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: hip,
              capturedAtMs: performance.now(),
            };
          }
        }
        const lifted = inLiftedRef.current[s];
        if (!lifted && hip >= LIFT_THRESHOLD_DEG) {
          // Rising edge — start of a lift.
          inLiftedRef.current[s] = true;
          liftPeakRef.current[s] = { hip, knee: knee ?? 0 };
          liftCountRef.current += 1;
          setLiftsDisplay(liftCountRef.current);
        } else if (lifted) {
          // Track this lift's peak hip + knee.
          if (hip > liftPeakRef.current[s].hip) {
            liftPeakRef.current[s].hip = hip;
          }
          if (knee !== null && knee > liftPeakRef.current[s].knee) {
            liftPeakRef.current[s].knee = knee;
          }
          if (hip <= RESET_THRESHOLD_DEG) {
            // Falling edge — record the completed lift's peaks.
            const peak = liftPeakRef.current[s];
            statsRef.current[s].hipSum += peak.hip;
            statsRef.current[s].kneeSum += peak.knee;
            statsRef.current[s].count += 1;
            inLiftedRef.current[s] = false;
          }
        }
      }

      // Coaching: pelvic tilt magnitude.
      const tilt = computePelvicTiltDeg(liveKp);
      if (tilt !== null) {
        setPelvisDrifted(Math.abs(tilt) > PELVIS_TILT_WARN_DEG);
      }
    },
    [durationMin, sessionPhase],
  );

  // Session timer — counts down the picked duration and auto-completes
  // (which triggers the footer's auto-save) when it hits zero.
  useEffect(() => {
    if (sessionPhase !== "live" || durationMin === null) return;
    const totalSec = durationMin * 60;
    const id = window.setInterval(() => {
      const elapsed = (performance.now() - sessionStartRef.current) / 1000;
      const left = Math.max(0, totalSec - elapsed);
      setTimeLeftSec(left);
      if (left <= 0) markComplete();
    }, 250);
    return () => window.clearInterval(id);
  }, [sessionPhase, durationMin, markComplete]);

  // Duration picker callback — sets the session length AND unlocks the
  // audio for autoplay. The browser only permits play() from inside a
  // user gesture (this click), so we start the element MUTED and pause
  // it immediately — this "blesses" the element so the real, audible
  // play() at the countdown→live transition is allowed. The music
  // therefore stays silent through the 3-2-1 countdown.
  const handleDurationPick = useCallback(
    (min: number) => {
      setDurationMin(min);
      setTimeLeftSec(min * 60);
      if (musicOn && audioRef.current) {
        const a = audioRef.current;
        a.muted = true;
        a.currentTime = 0;
        void a
          .play()
          .then(() => {
            a.pause();
            a.currentTime = 0;
            a.muted = false;
          })
          .catch(() => {});
      }
    },
    [musicOn],
  );

  // "Restart" — pause music, clear the picked duration so the picker
  // shows again, reset the live edge state.
  const handleRestart = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    inLiftedRef.current = { left: false, right: false };
    setDurationMin(null);
    setLiftsDisplay(0);
  }, []);

  // Mute toggle — pause / resume the music. Resume is autoplay-
  // safe because mute toggle is a user click.
  const handleMusicToggle = useCallback(() => {
    if (!audioRef.current) {
      setMusicOn((v) => !v);
      return;
    }
    if (musicOn) {
      audioRef.current.pause();
      setMusicOn(false);
    } else {
      void audioRef.current.play().catch(() => {});
      setMusicOn(true);
    }
  }, [musicOn]);

  // Cleanup on unmount — stop the music so it doesn't keep playing
  // if the user navigates away mid-session.
  useEffect(() => {
    return () => {
      const a = audioRef.current;
      if (a) {
        a.pause();
        a.currentTime = 0;
      }
    };
  }, []);

  const buildRehabPayload = useCallback(() => {
    if (durationMin === null) return null;
    const lifts = liftCountRef.current;
    const peak = peakFlexionRef.current;
    const st = statsRef.current;
    const avg = (sum: number, n: number) => (n > 0 ? sum / n : 0);
    const avgHipLeft = avg(st.left.hipSum, st.left.count);
    const avgHipRight = avg(st.right.hipSum, st.right.count);
    const avgKneeLeft = avg(st.left.kneeSum, st.left.count);
    const avgKneeRight = avg(st.right.kneeSum, st.right.count);
    const interpretation =
      lifts > 0
        ? `Marching (${durationMin} min): ${lifts} knee lifts. Avg peak hip flexion — L ${avgHipLeft.toFixed(0)}° / R ${avgHipRight.toFixed(0)}°; avg peak knee flexion — L ${avgKneeLeft.toFixed(0)}° / R ${avgKneeRight.toFixed(0)}°.`
        : "Session ended before the patient completed a lift above threshold.";
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      peak,
      null,
      `Peak marching lift — ${peak.toFixed(0)}° hip flexion`,
    );
    return {
      module: "rehab" as const,
      movement: "marching",
      metrics: {
        exercise_slug: "marching",
        mechanic_id: "metronome",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: {
          liftCount: lifts,
          liftsLeft: st.left.count,
          liftsRight: st.right.count,
          avgHipLeft,
          avgHipRight,
          avgKneeLeft,
          avgKneeRight,
          targetDurationMin: durationMin,
        },
        config: { bpm: MUSIC_BPM },
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [durationMin]);

  // Which leg to draw the live angle arc on — the one currently higher
  // (i.e. the leg mid-lift).
  const activeSide: Side = liveLeftHip >= liveRightHip ? "left" : "right";
  const activeHip = activeSide === "left" ? liveLeftHip : liveRightHip;

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          {/* Hidden audio element — visually invisible, fully
              controlled via audioRef. Looped so a single ~12-min
              source covers any session length without restart. */}
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <audio
            ref={audioRef}
            src={MUSIC_URL}
            loop
            preload="auto"
            aria-hidden="true"
          />

          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>H5 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Marching<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Marching in place — the patient lifts BOTH knees
                alternately for a set duration. Every lift&apos;s peak
                hip and knee flexion is recorded and averaged per side
                for the report. Pick a session length; it auto-saves
                when the timer ends. A looping beat track plays as an
                optional audio pacing cue.
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

          {durationMin === null ? (
            <DurationPicker onPick={handleDurationPick} />
          ) : null}

          {durationMin !== null && (
            <LiveModeLayout
              title="Marching · both knees"
              subtitle={
                isDoctorFlow && patient
                  ? `Connected to ${patient.name}'s record.`
                  : `${durationMin} min session`
              }
              onExit={handleRestart}
              camera={(
                <RehabCameraShell
                  onFrame={handleFrame}
                  autoStart
                  hideControls
                  angleArc={{
                    vertex: activeSide === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                    armA: activeSide === "left" ? LM_LIVE.LEFT_SHOULDER : LM_LIVE.RIGHT_SHOULDER,
                    armB: activeSide === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                    currentDeg: activeHip,
                    band: { min: 35, max: 90 },
                  }}
                >
                  <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                      Hip / Knee flexion
                    </p>
                    <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-0.5 tabular text-white">
                      <p className="text-[10px] text-zinc-400">Left</p>
                      <p className="text-[10px] text-zinc-400">Right</p>
                      <p className="text-lg font-semibold">
                        {liveLeftHip.toFixed(0)}°
                        <span className="ml-1 text-xs text-zinc-400">
                          / {liveLeftKnee.toFixed(0)}°
                        </span>
                      </p>
                      <p className="text-lg font-semibold">
                        {liveRightHip.toFixed(0)}°
                        <span className="ml-1 text-xs text-zinc-400">
                          / {liveRightKnee.toFixed(0)}°
                        </span>
                      </p>
                    </div>
                    <p className="mt-1 text-[9px] text-zinc-500">
                      hip / knee
                    </p>
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
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-fuchsia-500/15 px-3 py-1 text-xs font-semibold text-fuchsia-200 ring-1 ring-fuchsia-400/40">
                      Both knees
                    </span>
                    {(sessionPhase === "live" || sessionPhase === "complete") && (
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-500/15 px-3 py-1 text-xs font-semibold text-sky-200 ring-1 ring-sky-400/40 tabular">
                        {sessionPhase === "complete" ? "0:00" : fmtClock(timeLeftSec)} left
                      </span>
                    )}
                    {pelvisDrifted && (
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-500/20 px-3 py-1 text-xs font-semibold text-rose-200 ring-1 ring-rose-400/50">
                        Pelvis
                      </span>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleMusicToggle}
                      aria-pressed={musicOn}
                      aria-label={musicOn ? "Mute music" : "Unmute music"}
                    >
                      {musicOn ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                    </Button>
                    <Button variant="ghost" size="sm" onClick={handleRestart}>
                      Restart
                    </Button>
                  </div>

                  {REHAB_EXERCISE_IMAGES["marching"] && (
                    <div className="overflow-hidden rounded-md border border-border bg-white">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={REHAB_EXERCISE_IMAGES["marching"]}
                        alt="Marching reference"
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
                      hint="Patient faces the camera, full body in frame."
                    />
                  )}
                  {(sessionPhase === "live" || sessionPhase === "complete") && (
                    <div className="rounded-card border border-border bg-surface p-4">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
                        {sessionPhase === "complete" ? "Session complete" : "Live"}
                      </p>
                      <div className="mt-2 flex items-baseline gap-2">
                        <span className="tabular text-4xl font-semibold text-foreground">
                          {liftsDisplay}
                        </span>
                        <span className="text-sm text-muted">knee lifts</span>
                      </div>
                      <p className="mt-1 text-[11px] text-muted">
                        {sessionPhase === "complete"
                          ? "Saving — per-side ROM averages are in the report."
                          : "Keep marching with both knees until the timer ends."}
                      </p>
                    </div>
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
                Camera at hip height, ~2 m away, perpendicular to
                the patient — <strong>frontal view</strong>. Full
                body in frame; both hips, both knees, and both ankles
                must stay visible.
              </li>
              <li>
                March in place on the spot — <strong>alternate both
                knees</strong>, lifting each up to ~ hip height. Every
                lift on either leg is counted, and its peak hip + knee
                flexion is recorded for that side.
              </li>
              <li>
                Keep a steady, comfortable pace for the whole session.
                The looping beat track is an optional audio cue — mute
                it any time without affecting the counting.
              </li>
              <li>
                Keep the pelvis LEVEL — avoid hip-hiking to compensate
                for the lift. The rose &quot;Pelvis&quot; chip lights
                up if tilt exceeds {PELVIS_TILT_WARN_DEG}°. Coaching
                only.
              </li>
              <li>
                Pick <strong>5, 10, or 15 minutes</strong> up front —
                the session auto-saves when the timer runs out. The
                report shows the average peak hip + knee flexion for
                each side.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}

// Custom-duration bounds (minutes).
const MIN_DURATION = 1;
const MAX_DURATION = 60;

function DurationPicker({ onPick }: { onPick: (min: number) => void }) {
  const [custom, setCustom] = useState<string>("");

  const parsedCustom = Math.floor(Number(custom));
  const customValid =
    custom.trim() !== ""
    && Number.isFinite(parsedCustom)
    && parsedCustom >= MIN_DURATION
    && parsedCustom <= MAX_DURATION;

  const startCustom = () => {
    if (customValid) onPick(parsedCustom);
  };

  return (
    <div className="mt-10 max-w-xl">
      {REHAB_EXERCISE_IMAGES["marching"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["marching"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Set the session length
      </h2>
      <p className="mt-2 text-sm text-muted">
        The patient marches in place with BOTH knees for the whole
        duration; the session auto-saves when the timer ends. Music
        starts once the 3-2-1 countdown finishes; you can mute it any
        time from the toolbar.
      </p>

      {/* Quick presets */}
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        {DURATION_OPTIONS.map((min) => (
          <Button key={min} variant="ghost" onClick={() => onPick(min)}>
            {min} min
          </Button>
        ))}
      </div>

      {/* Custom duration */}
      <div className="mt-4 rounded-card border border-border bg-surface p-4">
        <label
          htmlFor="marching-custom-min"
          className="text-[10px] uppercase tracking-[0.14em] text-subtle"
        >
          Or set your own ({MIN_DURATION}–{MAX_DURATION} min)
        </label>
        <div className="mt-2 flex items-center gap-3">
          <input
            id="marching-custom-min"
            type="number"
            inputMode="numeric"
            min={MIN_DURATION}
            max={MAX_DURATION}
            step={1}
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") startCustom();
            }}
            placeholder="e.g. 8"
            className="w-28 rounded-md border border-border bg-elevated px-3 py-2 text-lg font-semibold tabular text-foreground outline-none focus:border-accent"
          />
          <span className="text-sm text-muted">minutes</span>
          <Button
            className="ml-auto"
            onClick={startCustom}
            disabled={!customValid}
          >
            Start
          </Button>
        </div>
        {custom.trim() !== "" && !customValid && (
          <p className="mt-2 text-[11px] text-rose-400">
            Enter a whole number between {MIN_DURATION} and {MAX_DURATION}.
          </p>
        )}
      </div>
    </div>
  );
}
