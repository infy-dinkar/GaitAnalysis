"use client";
// H5 — Marching (Metronome + paced music loop).
//
// Mechanic: Metronome (lib/rehab/mechanics.ts metronomeStep).
// Patient marches in place; each knee lift on the tracked side is
// graded against a steady beat. Music loop adds the AUDIO cue;
// the metronome bpm is matched to the music's 90 BPM so the visual
// pulse + the music's downbeats align without drift.
//
// Knee-lift detection:
//   computeHipAngle("flexion", kp, side) rises during a lift.
//   Edge-detect with threshold + hysteresis (LIFT 35° → RESET 15°)
//   so a single lift produces exactly one event.
//
// Audio architecture (kept OUT of the shared MetronomeShell):
//   • MetronomeShell's built-in Web Audio click stays OFF
//     (audio={false}) — would conflict with the music loop.
//   • Music is a plain HTML5 <audio> element owned by this PAGE,
//     pointing at /audio/rehab/marching-beat.mp3 served from
//     public/. Looped, default ON, mute toggle in the active UI.
//   • Browser autoplay restrictions: audio.play() is called from
//     the side-picker's button onClick (a user gesture), so the
//     browser permits playback. Subsequent play()/pause() via the
//     mute toggle inherit that grant.
//
// Reuses (no modifications):
//   • MetronomeShell, metronomeStep, RehabCameraShell
//   • computeHipAngle (biomech — IMPORT ONLY)
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
import { MetronomeShell } from "@/components/rehab/mechanics/MetronomeShell";
import { computeHipAngle } from "@/lib/biomech/hip-live";
import { computePelvicTiltDeg } from "@/lib/rehab/poseMetrics";
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

// Pelvic-tilt coaching threshold.
const PELVIS_TILT_WARN_DEG = 8;

const MARCHING_CONFIG = {
  // 90 BPM matches the marching-beat.mp3 source tempo so the
  // visual pulse + the music's downbeats align.
  bpm: 90,
  perfectWindowMs: 150,
  goodWindowMs: 350,
  pointsPerfect: 10,
  pointsGood: 5,
};

export default function MarchingExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  const [eventTrigger, setEventTrigger] = useState<number>(0);
  const [liveFlexion, setLiveFlexion] = useState<number>(0);
  const [pelvisDrifted, setPelvisDrifted] = useState<boolean>(false);
  const [musicOn, setMusicOn] = useState<boolean>(true);

  // Edge-detector state — true while the knee is in the lifted
  // position (after crossing LIFT_THRESHOLD). Resets to false when
  // hip flexion drops back below RESET_THRESHOLD.
  const inLiftedRef = useRef<boolean>(false);
  // HTMLAudioElement instance reference.
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], _v: HTMLVideoElement) => {
      if (!side) return;
      const liveKp = kp as unknown as LiveKeypoint[];
      const flexion = computeHipAngle("flexion", liveKp, side);
      if (flexion !== null) {
        setLiveFlexion(flexion);
        // Rising-edge detection — fire one event per lift.
        if (!inLiftedRef.current && flexion >= LIFT_THRESHOLD_DEG) {
          inLiftedRef.current = true;
          setEventTrigger((n) => n + 1);
        } else if (inLiftedRef.current && flexion <= RESET_THRESHOLD_DEG) {
          inLiftedRef.current = false;
        }
      }
      // Coaching: pelvic tilt magnitude.
      const tilt = computePelvicTiltDeg(liveKp);
      if (tilt !== null) {
        setPelvisDrifted(Math.abs(tilt) > PELVIS_TILT_WARN_DEG);
      }
    },
    [side],
  );

  // Side picker callback — wraps setSide with audio start. The
  // play() call must happen INSIDE the user-gesture handler
  // (button click) to satisfy browser autoplay policy.
  const handleSidePick = useCallback(
    (s: Side) => {
      setSide(s);
      if (musicOn && audioRef.current) {
        audioRef.current.currentTime = 0;
        // Autoplay safe here — invoked from the button's click
        // event. Catch (and silently ignore) the promise rejection
        // some browsers throw if playback fails anyway.
        void audioRef.current.play().catch(() => {});
      }
    },
    [musicOn],
  );

  // "Change side" — pause music, reset position, clear session
  // state so the next pick starts fresh.
  const handleChangeSide = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    inLiftedRef.current = false;
    setSide(null);
    setEventTrigger(0);
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
                Cadence-paced marching in place — patient lifts the
                tracked knee on every beat. Each lift is graded
                against the {MARCHING_CONFIG.bpm}-bpm beat (perfect /
                good / miss). A looping marching-beat track plays in
                sync; the patient can mute the music while keeping
                the visual pulse + scoring active. Powered by the
                Metronome mechanic.
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
            <SidePicker onPick={handleSidePick} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-fuchsia-500/15 px-3 py-1 text-xs font-semibold text-fuchsia-200 ring-1 ring-fuchsia-400/40">
                  Marching: {side === "left" ? "Left" : "Right"} knee tracked
                </span>
                {pelvisDrifted && (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-500/20 px-3 py-1 text-xs font-semibold text-rose-200 ring-1 ring-rose-400/50">
                    Keep pelvis level
                  </span>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleMusicToggle}
                  aria-pressed={musicOn}
                  aria-label={musicOn ? "Mute music" : "Unmute music"}
                >
                  {musicOn ? (
                    <Volume2 className="h-4 w-4" />
                  ) : (
                    <VolumeX className="h-4 w-4" />
                  )}
                  {musicOn ? "Music on" : "Music off"}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleChangeSide}
                >
                  Change side
                </Button>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  <RehabCameraShell
                    onFrame={handleFrame}
                    angleArc={{
                      vertex: side === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                      armA: side === "left" ? LM_LIVE.LEFT_SHOULDER : LM_LIVE.RIGHT_SHOULDER,
                      armB: side === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                      currentDeg: liveFlexion,
                      band: { min: 35, max: 90 },
                    }}
                  >
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        Hip flexion ({side})
                      </p>
                      <p className="tabular text-2xl font-semibold text-white">
                        {liveFlexion.toFixed(0)}°
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-300">
                        {inLiftedRef.current ? "lifted" : "ready"}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <MetronomeShell
                    eventTrigger={eventTrigger}
                    audio={false}
                    config={MARCHING_CONFIG}
                  />
                </div>
              </div>
            </div>
          )}

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2 m away, perpendicular to
                the patient — <strong>frontal view</strong>. Full
                body in frame; both hips, the tracked knee, and
                both ankles must stay visible.
              </li>
              <li>
                March in place on the spot — alternate knees, lift
                each up to ~ hip height. The tracked side&apos;s
                lifts are graded; the contralateral lifts are not
                counted (single-side timing assessment).
              </li>
              <li>
                Match the beat — <strong>{MARCHING_CONFIG.bpm} bpm
                </strong> ({(60 / MARCHING_CONFIG.bpm).toFixed(2)} s
                per lift). The marching-beat music plays in sync;
                listen + watch the visual pulse together. Beat hit
                within ±{MARCHING_CONFIG.perfectWindowMs} ms =
                perfect, within ±{MARCHING_CONFIG.goodWindowMs} ms
                = good, outside = miss.
              </li>
              <li>
                Keep the pelvis LEVEL — avoid hip-hiking to compensate
                for the lift. The rose &quot;Keep pelvis level&quot;
                chip lights up if tilt exceeds {PELVIS_TILT_WARN_DEG}°.
                Coaching only; doesn&apos;t affect the beat grade.
              </li>
              <li>
                The music is on by default. Use the &quot;Music on /
                off&quot; toggle to mute or resume — the visual beat
                + scoring keep running either way.
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
        Choose the tracked knee
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the knee whose lifts the system should grade against
        the beat. The patient still marches with BOTH knees — the
        tracked one is the timing reference. Music starts playing
        when you click; you can mute it any time from the toolbar.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left knee</Button>
        <Button onClick={() => onPick("right")}>Right knee</Button>
      </div>
    </div>
  );
}
