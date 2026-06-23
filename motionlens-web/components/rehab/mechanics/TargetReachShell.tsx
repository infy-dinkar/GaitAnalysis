"use client";
// Target-Reach mechanic — UI shell.
//
// Visual: rectangular play area with the patient's cursor (driven
// by the input `cursor` prop in normalised [0..1] × [0..1] coords)
// and spawning targets. The shell auto-spawns a new target on a
// configurable interval and trims expired ones. Hit detection and
// scoring all live in the pure targetReachStep engine.

import { useEffect, useRef, useState } from "react";
import {
  type ReachTarget,
  type Score,
  type TargetReachConfig,
  type TargetReachState,
  emptyScore,
  emptyTargetReachState,
} from "@/lib/rehab/gameState";
import { spawnReachTarget, targetReachStep } from "@/lib/rehab/mechanics";
import { ScoreHUD } from "@/components/rehab/mechanics/ScoreHUD";

interface Props {
  /** Patient's cursor in normalised play-area coordinates. */
  cursor: { x: number; y: number };
  /** Auto-spawn settings. The shell injects a new target every
   *  spawnIntervalMs into a randomised position. Set 0 to disable
   *  auto-spawn (caller drives via spawnReachTarget externally). */
  spawnIntervalMs?: number;
  /** Default ttl for auto-spawned targets. */
  defaultTtlMs?: number;
  /** Default normalised radius. */
  defaultRadius?: number;
  config: TargetReachConfig;
}

export function TargetReachShell({
  cursor,
  spawnIntervalMs = 1800,
  defaultTtlMs = 3000,
  defaultRadius = 0.07,
  config,
}: Props) {
  const stateRef = useRef<TargetReachState>(emptyTargetReachState());
  const scoreRef = useRef<Score>(emptyScore());
  const lastSpawnRef = useRef<number>(0);
  const idCounterRef = useRef(0);
  const [, setTick] = useState(0);

  // Mirror live props in a ref so the rAF loop reads the latest
  // values without re-creating itself on each prop change. Cursor
  // changes ~60 Hz; including it in useEffect deps causes the
  // loop to be torn down + recreated every frame.
  const propsRef = useRef({
    cursor,
    config,
    spawnIntervalMs,
    defaultTtlMs,
    defaultRadius,
  });
  useEffect(() => {
    propsRef.current = {
      cursor,
      config,
      spawnIntervalMs,
      defaultTtlMs,
      defaultRadius,
    };
  }, [cursor, config, spawnIntervalMs, defaultTtlMs, defaultRadius]);

  // Single rAF loop — set up ONCE on mount, never restarted. Reads
  // latest cursor + config from propsRef each frame so per-frame
  // prop changes don't recreate the loop. Drives auto-spawn + engine
  // step + UI re-render.
  useEffect(() => {
    let cancelled = false;
    let raf: number | null = null;
    const tick = () => {
      if (cancelled) return;
      const now = performance.now();
      const p = propsRef.current;
      // Auto-spawn
      if (
        p.spawnIntervalMs > 0
        && (now - lastSpawnRef.current) >= p.spawnIntervalMs
      ) {
        const id = `t${++idCounterRef.current}`;
        const t: ReachTarget = {
          id,
          x: 0.15 + Math.random() * 0.7,
          y: 0.15 + Math.random() * 0.7,
          radius: p.defaultRadius,
          ttlMs: p.defaultTtlMs,
          spawnedAt: now,
        };
        stateRef.current = spawnReachTarget(stateRef.current, t);
        lastSpawnRef.current = now;
      }
      // Engine step (hit-test + ttl trim).
      const r = targetReachStep(
        stateRef.current,
        scoreRef.current,
        p.cursor,
        p.config,
        now,
      );
      stateRef.current = r.state;
      scoreRef.current = r.score;
      setTick((x) => (x + 1) % 1_000_000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, []);

  const s = stateRef.current;
  const score = scoreRef.current;
  const timer = `${s.hits} hits · ${s.misses} miss`;

  return (
    <div className="space-y-4">
      <ScoreHUD
        score={score}
        timer={timer}
        feedback={s.hits > 0 ? `Excursion ${(s.maxExcursion * 100).toFixed(0)}%` : null}
      />
      <div className="relative aspect-video w-full overflow-hidden rounded-card border border-zinc-700 bg-zinc-950">
        {/* Crosshair at the play-area centre */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-white/5" />
          <div className="absolute left-0 top-1/2 h-px w-full -translate-y-1/2 bg-white/5" />
        </div>
        {/* Targets */}
        {s.targets.map((t) => {
          const now = performance.now();
          const lifeLeft =
            t.ttlMs != null && t.ttlMs > 0
              ? Math.max(0, 1 - (now - t.spawnedAt) / t.ttlMs)
              : 1;
          return (
            <div
              key={t.id}
              className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-orange-400 transition-opacity"
              style={{
                left: `${t.x * 100}%`,
                top: `${t.y * 100}%`,
                width: `${t.radius * 200}%`,
                aspectRatio: "1",
                background: `radial-gradient(circle, rgba(251,146,60,${0.35 * lifeLeft}) 0%, rgba(251,146,60,0) 70%)`,
                opacity: 0.4 + 0.6 * lifeLeft,
              }}
            />
          );
        })}
        {/* Cursor */}
        <div
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${cursor.x * 100}%`, top: `${cursor.y * 100}%` }}
        >
          <div className="h-4 w-4 rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.8)]" />
        </div>
      </div>
    </div>
  );
}
