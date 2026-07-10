"use client";
// Trace mechanic — UI shell.
//
// Visual: a parametric path drawn across the play area + a moving
// "lead" target that advances along the path at a fixed cadence.
// The patient's cursor (driven by the `cursor` prop) tries to
// stay glued to the lead target. The pure traceStep engine scores
// per-sample accuracy and smoothness.

import { useEffect, useRef, useState } from "react";
import {
  type Score,
  type TraceConfig,
  type TracePathPoint,
  type TraceState,
  emptyScore,
  emptyTraceState,
} from "@/lib/rehab/gameState";
import { traceStep } from "@/lib/rehab/mechanics";
import { ScoreHUD } from "@/components/rehab/mechanics/ScoreHUD";

interface Props {
  /** Patient's cursor in normalised [0..1] coords. */
  cursor: { x: number; y: number };
  /** Parametric path — function of t ∈ [0, 1]. Defaults to a
   *  horizontal sine wave so the shell is visually verifiable
   *  before an exercise is plugged in. */
  pathFn?: (t: number) => TracePathPoint;
  /** Time (ms) for one full traversal of t = 0 → 1. */
  loopDurationMs?: number;
  /** Length of the cursor's trail visualisation, in samples. */
  trailLength?: number;
  config: TraceConfig;
  /** Optional session-state harvester — same additive pattern as
   *  RepCountShell.onSnapshot. Fires on delta-gated sample-count
   *  changes so pages can capture accuracy/smoothness/mean-dev at
   *  save time. */
  onSnapshot?: (state: TraceState, score: Score) => void;
  /** Compact live-mode variant. */
  compact?: boolean;
}

const DEFAULT_PATH: (t: number) => TracePathPoint = (t) => ({
  x: t,
  y: 0.5 + 0.25 * Math.sin(t * Math.PI * 4),
});

export function TraceShell({
  cursor,
  pathFn = DEFAULT_PATH,
  loopDurationMs = 6000,
  trailLength = 30,
  config,
  onSnapshot,
  compact = false,
}: Props) {
  const stateRef = useRef<TraceState>(emptyTraceState());
  const scoreRef = useRef<Score>(emptyScore());
  const startedAtRef = useRef<number>(performance.now());
  const trailRef = useRef<TracePathPoint[]>([]);
  const [tick, setTick] = useState(0);
  const onSnapshotRef = useRef(onSnapshot);
  useEffect(() => {
    onSnapshotRef.current = onSnapshot;
  }, [onSnapshot]);
  // Delta-gate: only fire when samples change by >=1 (~60Hz naturally).
  // Throttle to at most 5Hz to keep the consumer cheap.
  const lastEmitRef = useRef<{ samples: number; at: number } | null>(null);

  // Mirror live props in a ref so the rAF loop reads the latest
  // values without re-creating itself on each prop change. Cursor
  // changes ~60 Hz; including it in deps tears down + recreates
  // the rAF loop every frame.
  const propsRef = useRef({ cursor, config, pathFn, loopDurationMs, trailLength });
  useEffect(() => {
    propsRef.current = { cursor, config, pathFn, loopDurationMs, trailLength };
  }, [cursor, config, pathFn, loopDurationMs, trailLength]);

  // Single rAF loop — set up ONCE on mount, never restarted. Reads
  // latest props from propsRef each frame.
  useEffect(() => {
    let cancelled = false;
    let raf: number | null = null;
    const loop = () => {
      if (cancelled) return;
      const now = performance.now();
      const p = propsRef.current;
      const t = ((now - startedAtRef.current) % p.loopDurationMs) / p.loopDurationMs;
      const target = p.pathFn(t);
      const r = traceStep(stateRef.current, scoreRef.current, p.cursor, target, p.config, now);
      stateRef.current = r.state;
      scoreRef.current = r.score;
      trailRef.current = [...trailRef.current.slice(-p.trailLength), p.cursor];
      const last = lastEmitRef.current;
      if (
        !last
        || (r.state.samples !== last.samples && now - last.at >= 200)
      ) {
        lastEmitRef.current = { samples: r.state.samples, at: now };
        onSnapshotRef.current?.(r.state, r.score);
      }
      setTick((x) => (x + 1) % 1_000_000);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelled = true;
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, []);

  const s = stateRef.current;
  const score = scoreRef.current;
  const accuracyPct =
    s.samples > 0 ? (s.accurateSamples / s.samples) * 100 : 0;
  const smoothPct =
    s.samples > 0 ? (s.smoothSamples / s.samples) * 100 : 0;

  // Render path as a series of dots for a cheap visual.
  const pathSamples = 80;
  const dots: TracePathPoint[] = [];
  for (let i = 0; i <= pathSamples; i++) {
    dots.push(pathFn(i / pathSamples));
  }
  const now = performance.now();
  const tNow = ((now - startedAtRef.current) % loopDurationMs) / loopDurationMs;
  const lead = pathFn(tNow);

  void tick;
  if (compact) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <ScoreHUD
          score={score}
          timer={`${accuracyPct.toFixed(0)}%`}
          compact
        />
        <div className="relative min-h-0 flex-1 overflow-hidden rounded-lg border border-zinc-700 bg-zinc-950">
          {dots.map((p, i) => (
            <div
              key={i}
              className="absolute h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/20"
              style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
            />
          ))}
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${lead.x * 100}%`, top: `${lead.y * 100}%` }}
          >
            <div className="h-4 w-4 rounded-full bg-amber-400 ring-2 ring-amber-200 shadow-[0_0_14px_rgba(252,211,77,0.7)]" />
          </div>
          {trailRef.current.map((p, i) => (
            <div
              key={`trail-${i}`}
              className="absolute h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-400/60"
              style={{
                left: `${p.x * 100}%`,
                top: `${p.y * 100}%`,
                opacity: i / trailRef.current.length,
              }}
            />
          ))}
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${cursor.x * 100}%`, top: `${cursor.y * 100}%` }}
          >
            <div className="h-3 w-3 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.9)]" />
          </div>
          <div className="absolute bottom-2 left-2 right-2 rounded-md bg-black/60 px-2 py-1 backdrop-blur">
            <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full bg-emerald-500 transition-all"
                style={{ width: `${accuracyPct}%` }}
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
        timer={`${accuracyPct.toFixed(0)}% on path`}
        feedback={`Smoothness ${smoothPct.toFixed(0)}%`}
      />
      <div className="relative aspect-video w-full overflow-hidden rounded-card border border-zinc-700 bg-zinc-950">
        {/* Path */}
        {dots.map((p, i) => (
          <div
            key={i}
            className="absolute h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/20"
            style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
          />
        ))}
        {/* Lead target */}
        <div
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${lead.x * 100}%`, top: `${lead.y * 100}%` }}
        >
          <div className="h-6 w-6 rounded-full bg-amber-400 ring-2 ring-amber-200 shadow-[0_0_18px_rgba(252,211,77,0.7)]" />
        </div>
        {/* Cursor trail */}
        {trailRef.current.map((p, i) => (
          <div
            key={`trail-${i}`}
            className="absolute h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-400/60"
            style={{
              left: `${p.x * 100}%`,
              top: `${p.y * 100}%`,
              opacity: i / trailRef.current.length,
            }}
          />
        ))}
        {/* Cursor */}
        <div
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${cursor.x * 100}%`, top: `${cursor.y * 100}%` }}
        >
          <div className="h-4 w-4 rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.9)]" />
        </div>
        {/* Accuracy meter */}
        <div className="absolute bottom-3 left-3 right-3 rounded-md bg-black/60 px-3 py-2 backdrop-blur">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-zinc-300">
            <span>Mean dev {s.meanDeviation.toFixed(3)}</span>
            <span>Mean jerk {s.meanJerk.toFixed(4)}</span>
          </div>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full bg-emerald-500 transition-all"
              style={{ width: `${accuracyPct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
