"use client";
// Metronome mechanic — UI shell.
//
// Visual: a pulsing beat indicator that lights up on each scheduled
// beat + an optional audio click (Web Audio API) for cadence cues.
// Patient produces events (rep completions, steps, claps — whatever
// the exercise emits) by incrementing the `eventTrigger` prop. The
// shell records the timestamp when the trigger changes and feeds
// it to the pure metronomeStep engine.
//
// A scrolling history strip shows the deviation of the last ~8
// beats so the patient sees their cadence drifting.

import { useEffect, useRef, useState } from "react";
import {
  type MetronomeConfig,
  type MetronomeState,
  type Score,
  emptyMetronomeState,
  emptyScore,
} from "@/lib/rehab/gameState";
import {
  metronomeStep,
  metronomeUpcomingBeats,
} from "@/lib/rehab/mechanics";
import { ScoreHUD } from "@/components/rehab/mechanics/ScoreHUD";

interface Props {
  /** Increment this any time the patient produces a tappable event.
   *  The shell reads performance.now() at that moment and feeds it
   *  to the engine. Use a counter (0, 1, 2, …); the engine doesn't
   *  care about the value, only that it changed. */
  eventTrigger: number;
  /** Enable the audio click on each scheduled beat. Default true. */
  audio?: boolean;
  config: MetronomeConfig;
  /** Compact live-mode variant. */
  compact?: boolean;
}

export function MetronomeShell({
  eventTrigger,
  audio = true,
  config,
  compact = false,
}: Props) {
  const stateRef = useRef<MetronomeState>(emptyMetronomeState());
  const scoreRef = useRef<Score>(emptyScore());
  const audioCtxRef = useRef<AudioContext | null>(null);
  const lastClickedBeatRef = useRef<number>(-1);
  const sessionStartRef = useRef<number>(performance.now());
  const [feedback, setFeedback] = useState<string | null>(null);
  const [feedbackTone, setFeedbackTone] =
    useState<"good" | "bad" | "neutral">("neutral");
  const [pulse, setPulse] = useState(false);
  const [, setTick] = useState(0);

  // Process patient event ticks.
  useEffect(() => {
    if (eventTrigger === 0) return; // initial mount — don't grade
    const now = performance.now();
    const r = metronomeStep(stateRef.current, scoreRef.current, now, config, now);
    stateRef.current = r.state;
    scoreRef.current = r.score;
    const grade = r.event?.kind ?? "";
    if (grade === "beat_perfect") {
      setFeedback("Perfect!");
      setFeedbackTone("good");
    } else if (grade === "beat_good") {
      setFeedback("Good");
      setFeedbackTone("good");
    } else if (grade === "beat_miss") {
      setFeedback("Off beat");
      setFeedbackTone("bad");
    }
    setTick((t) => t + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eventTrigger]);

  // Drive the visual pulse + optional audio click on each scheduled
  // beat.
  useEffect(() => {
    let cancelled = false;
    let raf: number | null = null;
    const playClick = () => {
      if (!audio) return;
      try {
        if (!audioCtxRef.current) {
          const Ctx =
            (window as unknown as { AudioContext?: typeof AudioContext })
              .AudioContext ??
            (window as unknown as {
              webkitAudioContext?: typeof AudioContext;
            }).webkitAudioContext;
          if (!Ctx) return;
          audioCtxRef.current = new Ctx();
        }
        const ctx = audioCtxRef.current;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.value = 880;
        osc.connect(gain);
        gain.connect(ctx.destination);
        const t0 = ctx.currentTime;
        gain.gain.setValueAtTime(0.3, t0);
        gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.05);
        osc.start(t0);
        osc.stop(t0 + 0.06);
      } catch {
        // ignore audio failures (autoplay policy etc.)
      }
    };
    const tick = () => {
      if (cancelled) return;
      const now = performance.now();
      const upcoming = metronomeUpcomingBeats(
        { ...stateRef.current, sessionStartedAt: sessionStartRef.current },
        config,
        now,
        2000,
      );
      // Fire the click + pulse exactly once per beat as it crosses
      // t-minus = 0 (with a tiny tolerance to absorb rAF jitter).
      const due = upcoming.find(
        (b) => b.tMinusMs <= 0 && b.tMinusMs > -50,
      );
      if (due && due.beatIndex > lastClickedBeatRef.current) {
        lastClickedBeatRef.current = due.beatIndex;
        playClick();
        setPulse(true);
        window.setTimeout(() => setPulse(false), 120);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, [audio, config]);

  const s = stateRef.current;
  const score = scoreRef.current;
  const lastBeats = s.beats.slice(-8);
  const accuracyPct =
    s.beats.length > 0
      ? ((s.perfectCount + s.goodCount) / s.beats.length) * 100
      : 0;

  if (compact) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <ScoreHUD
          score={score}
          timer={`${accuracyPct.toFixed(0)}%`}
          feedback={feedback}
          feedbackTone={feedbackTone}
          compact
        />
        <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-zinc-700 bg-zinc-900/80 p-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-zinc-400">
                Tempo
              </p>
              <p className="tabular text-2xl font-semibold text-white">
                {config.bpm} <span className="text-xs text-zinc-400">bpm</span>
              </p>
            </div>
            <div
              className={`flex h-14 w-14 items-center justify-center rounded-full transition-all ${
                pulse
                  ? "scale-110 bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.8)]"
                  : "scale-100 bg-zinc-700"
              }`}
            >
              <span className="text-lg font-bold text-zinc-50">♩</span>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 text-[10px]">
            <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2">
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Perfect</p>
              <p className="tabular text-lg font-semibold text-emerald-300">{s.perfectCount}</p>
            </div>
            <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2">
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Good</p>
              <p className="tabular text-lg font-semibold text-amber-300">{s.goodCount}</p>
            </div>
            <div className="rounded-md border border-zinc-700 bg-zinc-950 p-2">
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Miss</p>
              <p className="tabular text-lg font-semibold text-rose-300">{s.missCount}</p>
            </div>
          </div>
          <div className="mt-2 min-h-0 flex-1">
            <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">
              Last 8 · deviation (ms)
            </p>
            <div className="mt-1 flex h-full max-h-24 items-end gap-1">
              {lastBeats.length === 0 && (
                <span className="text-[10px] text-zinc-500">Waiting for events…</span>
              )}
              {lastBeats.map((b, i) => {
                const dev = b.deviationMs ?? 0;
                const maxAbs = Math.max(config.goodWindowMs, Math.abs(dev));
                const h = (Math.abs(dev) / maxAbs) * 100;
                const colour =
                  b.grade === "perfect"
                    ? "bg-emerald-500"
                    : b.grade === "good"
                      ? "bg-amber-500"
                      : "bg-rose-500";
                return (
                  <div key={i} className="flex flex-1 flex-col items-center gap-0.5">
                    <div
                      className={`w-full rounded-sm ${colour}`}
                      style={{ height: `${Math.max(6, h)}%` }}
                    />
                    <span className="text-[8px] tabular text-zinc-500">
                      {dev > 0 ? "+" : ""}{dev.toFixed(0)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <ScoreHUD
        score={score}
        timer={`${accuracyPct.toFixed(0)}% on tempo`}
        feedback={feedback}
        feedbackTone={feedbackTone}
      />

      <div className="rounded-card border border-zinc-700 bg-zinc-900/80 p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-400">
              Tempo
            </p>
            <p className="tabular text-3xl font-semibold text-white">
              {config.bpm} <span className="text-sm text-zinc-400">bpm</span>
            </p>
            <p className="mt-1 text-[10px] uppercase tracking-[0.14em] text-zinc-500">
              Perfect ±{config.perfectWindowMs}ms · Good ±{config.goodWindowMs}ms
            </p>
          </div>

          {/* Beat pulse circle */}
          <div
            className={`flex h-20 w-20 items-center justify-center rounded-full transition-all ${
              pulse
                ? "scale-110 bg-cyan-400 shadow-[0_0_28px_rgba(34,211,238,0.8)]"
                : "scale-100 bg-zinc-700"
            }`}
          >
            <span className="text-xl font-bold text-zinc-50">♩</span>
          </div>
        </div>

        {/* Counts */}
        <div className="mt-6 grid grid-cols-3 gap-3 text-xs">
          <Stat label="Perfect" value={s.perfectCount} tone="good" />
          <Stat label="Good" value={s.goodCount} tone="neutral" />
          <Stat label="Miss" value={s.missCount} tone="bad" />
        </div>

        {/* Recent deviations */}
        <div className="mt-6">
          <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
            Last 8 events · deviation from beat (ms)
          </p>
          <div className="mt-2 flex h-16 items-end gap-1">
            {lastBeats.length === 0 && (
              <span className="text-xs text-zinc-500">
                Waiting for first event…
              </span>
            )}
            {lastBeats.map((b, i) => {
              const dev = b.deviationMs ?? 0;
              const maxAbs = Math.max(
                config.goodWindowMs,
                Math.abs(dev),
              );
              const h = (Math.abs(dev) / maxAbs) * 100;
              const sign = dev < 0 ? "early" : "late";
              const colour =
                b.grade === "perfect"
                  ? "bg-emerald-500"
                  : b.grade === "good"
                  ? "bg-amber-500"
                  : "bg-rose-500";
              return (
                <div
                  key={i}
                  className="flex flex-1 flex-col items-center gap-1"
                >
                  <div
                    className={`w-full rounded-sm ${colour}`}
                    style={{ height: `${Math.max(6, h)}%` }}
                  />
                  <span className="text-[9px] tabular text-zinc-500">
                    {dev > 0 ? "+" : ""}
                    {dev.toFixed(0)}
                    <span className="text-[8px] text-zinc-600"> {sign}</span>
                  </span>
                </div>
              );
            })}
          </div>
          {s.beats.length > 0 && (
            <p className="mt-2 text-[10px] text-zinc-400">
              Mean |dev| {s.meanAbsDeviationMs.toFixed(0)} ms
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "good" | "bad" | "neutral";
}) {
  const colour =
    tone === "good"
      ? "text-emerald-300"
      : tone === "bad"
      ? "text-rose-300"
      : "text-amber-300";
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-950 p-3">
      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
        {label}
      </p>
      <p className={`tabular mt-1 text-xl font-semibold ${colour}`}>
        {value}
      </p>
    </div>
  );
}
