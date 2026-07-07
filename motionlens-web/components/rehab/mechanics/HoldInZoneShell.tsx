"use client";
// Hold-in-Zone mechanic — UI shell.
//
// Visual: vertical track from min to max axis, target band shown
// as a green strip, current signal as a horizontal marker. Hold
// timer + progress bar across the bottom.
//
// Driven by the pure holdInZoneStep engine. The shell owns the
// state (via useRef so frame-rate updates don't churn React) and
// re-evaluates the engine on every `signal` prop change.

import { useEffect, useRef, useState } from "react";
import {
  type HoldInZoneConfig,
  type HoldInZoneState,
  type Score,
  emptyHoldInZoneState,
  emptyScore,
} from "@/lib/rehab/gameState";
import { holdInZoneStep } from "@/lib/rehab/mechanics";
import { ScoreHUD } from "@/components/rehab/mechanics/ScoreHUD";

interface Props {
  /** Latest value of the input signal (e.g. knee angle in degrees).
   *  Placeholder 0 is fine before an exercise is plugged in. */
  signal: number;
  /** Display label for the signal axis (e.g. "Knee flexion (°)"). */
  signalLabel?: string;
  /** Axis bounds — used to scale the marker position on the track. */
  axisMin: number;
  axisMax: number;
  config: HoldInZoneConfig;
  /** Compact live-mode variant. */
  compact?: boolean;
}

export function HoldInZoneShell({
  signal,
  signalLabel = "Signal",
  axisMin,
  axisMax,
  config,
  compact = false,
}: Props) {
  const stateRef = useRef<HoldInZoneState>(emptyHoldInZoneState());
  const scoreRef = useRef<Score>(emptyScore());
  const [, setTick] = useState(0);

  // Mirror live props in a ref so the rAF loop always reads the
  // latest values without re-creating itself on each prop change.
  // This effect does NOT setState — it only mutates the ref, so
  // running it on every signal update (~60 Hz) is safe.
  const propsRef = useRef({ signal, config });
  useEffect(() => {
    propsRef.current = { signal, config };
  }, [signal, config]);

  // Single rAF loop, started once on mount. The engine ticks every
  // frame regardless of signal stability so dtMs keeps accumulating
  // when the patient holds perfectly still in the band.
  //
  // The prior pattern — useEffect([signal, config]) calling
  // setTick inside — caused two bugs:
  //   1. Max-update-depth: 60 Hz setState-from-effect chain trips
  //      React 19's heuristic.
  //   2. Timer froze when signal stopped changing: useEffect
  //      didn't re-run, so dtMs never accumulated.
  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    const loop = () => {
      if (cancelled) return;
      const now = performance.now();
      const { signal: s, config: c } = propsRef.current;
      const r = holdInZoneStep(stateRef.current, scoreRef.current, s, c, now);
      stateRef.current = r.state;
      scoreRef.current = r.score;
      setTick((t) => (t + 1) % 1_000_000);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelled = true;
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const s = stateRef.current;
  const score = scoreRef.current;

  const axisSpan = Math.max(1e-6, axisMax - axisMin);
  const markerPct = ((signal - axisMin) / axisSpan) * 100;
  const bandTopPct = ((axisMax - config.max) / axisSpan) * 100;
  const bandHeightPct = ((config.max - config.min) / axisSpan) * 100;
  const progressPct = Math.min(
    100,
    (s.totalMsInZone / Math.max(1, config.targetHoldMs)) * 100,
  );

  const remainingMs = Math.max(0, config.targetHoldMs - s.totalMsInZone);
  const timer = `${(remainingMs / 1000).toFixed(1)}s left`;
  const feedback = s.inZone ? "In the zone" : null;

  if (compact) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <ScoreHUD
          score={score}
          timer={timer}
          feedback={feedback}
          feedbackTone={s.inZone ? "good" : "neutral"}
          compact
        />
        <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-zinc-700 bg-zinc-900/80 p-3">
          <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-zinc-400">
            {signalLabel}
          </p>
          <div className="mt-2 flex min-h-0 flex-1 items-stretch gap-3">
            <div className="relative w-10 overflow-hidden rounded-md border border-zinc-700 bg-zinc-950">
              <div
                className="absolute inset-x-0 bg-emerald-500/25 ring-1 ring-emerald-400/60"
                style={{
                  top: `${Math.max(0, bandTopPct)}%`,
                  height: `${Math.max(0, bandHeightPct)}%`,
                }}
              />
              <div
                className={`absolute inset-x-[-4px] h-1 rounded-full ${
                  s.inZone ? "bg-emerald-300" : "bg-rose-300"
                }`}
                style={{
                  bottom: `calc(${Math.min(100, Math.max(0, markerPct))}% - 2px)`,
                }}
              />
            </div>
            <div className="flex flex-col justify-between py-1 text-[10px] text-zinc-400">
              <span className="tabular">{axisMax.toFixed(0)}</span>
              <span className="tabular text-zinc-100">
                <span className="font-semibold">{signal.toFixed(0)}</span>
              </span>
              <span className="tabular">{axisMin.toFixed(0)}</span>
            </div>
            <div className="flex flex-1 flex-col justify-between text-[10px] text-zinc-300">
              <div>
                <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Band</p>
                <p className="tabular">{config.min.toFixed(0)}–{config.max.toFixed(0)}</p>
              </div>
              <div>
                <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Dwell</p>
                <p className="tabular">{(s.currentDwellMs / 1000).toFixed(1)}s</p>
              </div>
              <div>
                <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Total</p>
                <p className="tabular">{(s.totalMsInZone / 1000).toFixed(1)}s</p>
              </div>
            </div>
          </div>
          <div className="mt-2">
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full bg-emerald-500 transition-all"
                style={{ width: `${progressPct}%` }}
              />
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
        timer={timer}
        feedback={feedback}
        feedbackTone={s.inZone ? "good" : "neutral"}
      />
      <div className="rounded-card border border-zinc-700 bg-zinc-900/80 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-400">
          {signalLabel}
        </p>

        <div className="mt-4 flex items-stretch gap-6">
          {/* Vertical track */}
          <div className="relative h-64 w-12 overflow-hidden rounded-lg border border-zinc-700 bg-zinc-950">
            {/* Target band */}
            <div
              className="absolute inset-x-0 bg-emerald-500/25 ring-1 ring-emerald-400/60"
              style={{
                top: `${Math.max(0, bandTopPct)}%`,
                height: `${Math.max(0, bandHeightPct)}%`,
              }}
            />
            {/* Marker */}
            <div
              className={`absolute inset-x-[-6px] h-1 rounded-full ${
                s.inZone ? "bg-emerald-300" : "bg-rose-300"
              }`}
              style={{
                bottom: `calc(${Math.min(100, Math.max(0, markerPct))}% - 2px)`,
              }}
            />
          </div>
          {/* Axis labels + current value */}
          <div className="flex flex-col justify-between text-xs text-zinc-400">
            <span className="tabular">{axisMax.toFixed(0)}</span>
            <span className="tabular text-zinc-200">
              now: <span className="font-semibold">{signal.toFixed(1)}</span>
            </span>
            <span className="tabular">{axisMin.toFixed(0)}</span>
          </div>

          {/* Status pane */}
          <div className="flex-1 space-y-3">
            <div>
              <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
                Target band
              </p>
              <p className="tabular text-sm text-zinc-100">
                {config.min.toFixed(0)} – {config.max.toFixed(0)}
              </p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
                Current dwell
              </p>
              <p className="tabular text-sm text-zinc-100">
                {(s.currentDwellMs / 1000).toFixed(1)} s
              </p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
                Total in zone
              </p>
              <p className="tabular text-sm text-zinc-100">
                {(s.totalMsInZone / 1000).toFixed(1)} s
              </p>
            </div>
            <div className="pt-2">
              <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
                Hold progress
              </p>
              <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full bg-emerald-500 transition-all"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
