"use client";
// Match-Pose mechanic — UI shell.
//
// Visual: per-joint match bars + overall match % gauge + a hold
// progress bar. The patient sees how close each joint angle is to
// its target and the cumulative hold time at the achieved
// threshold.

import { useEffect, useRef, useState } from "react";
import {
  type MatchPoseConfig,
  type MatchPoseState,
  type Score,
  emptyMatchPoseState,
  emptyScore,
} from "@/lib/rehab/gameState";
import { matchPoseStep } from "@/lib/rehab/mechanics";
import { ScoreHUD } from "@/components/rehab/mechanics/ScoreHUD";

interface Props {
  /** Current joint angles keyed identically to config.pose. */
  currentAngles: Record<string, number>;
  config: MatchPoseConfig;
  /** Compact live-mode variant. */
  compact?: boolean;
}

export function MatchPoseShell({ currentAngles, config, compact = false }: Props) {
  const stateRef = useRef<MatchPoseState>(emptyMatchPoseState());
  const scoreRef = useRef<Score>(emptyScore());
  const [, setTick] = useState(0);

  // Mirror live props in a ref so the rAF loop reads the latest
  // values without re-creating itself on each prop change. No
  // setState here — safe at 60 Hz prop updates.
  const propsRef = useRef({ currentAngles, config });
  useEffect(() => {
    propsRef.current = { currentAngles, config };
  }, [currentAngles, config]);

  // Single rAF loop, started once on mount. Engine ticks every
  // frame regardless of currentAngles stability so dtMs keeps
  // accumulating when the patient holds the achieved pose perfectly
  // still.
  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    const loop = () => {
      if (cancelled) return;
      const now = performance.now();
      const { currentAngles: ca, config: c } = propsRef.current;
      const r = matchPoseStep(stateRef.current, scoreRef.current, ca, c, now);
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
  const overallPct = Math.max(0, Math.min(100, s.matchPct));
  const holdProgressPct = Math.min(
    100,
    (s.achievedDwellMs / Math.max(1, config.requiredHoldMs)) * 100,
  );
  const remainingMs = Math.max(0, config.requiredHoldMs - s.achievedDwellMs);

  const tone = s.achieved ? "good" : overallPct >= 60 ? "neutral" : "bad";
  const feedback = s.achieved
    ? "Pose locked"
    : overallPct >= 60
    ? "Close — refine"
    : "Adjust pose";

  if (compact) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <ScoreHUD
          score={score}
          timer={`${(remainingMs / 1000).toFixed(1)}s`}
          feedback={feedback}
          feedbackTone={tone}
          compact
        />
        <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-zinc-700 bg-zinc-900/80 p-3">
          <div className="flex items-center gap-3">
            <div
              className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-full border-2 ${
                s.achieved
                  ? "border-emerald-400 bg-emerald-500/20"
                  : overallPct >= 60
                    ? "border-amber-400 bg-amber-500/15"
                    : "border-rose-400 bg-rose-500/15"
              }`}
            >
              <p className="tabular text-lg font-bold text-white">
                {overallPct.toFixed(0)}%
              </p>
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">
                Threshold {config.achievedThresholdPct.toFixed(0)}%
              </p>
              <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full bg-cyan-400 transition-all"
                  style={{ width: `${holdProgressPct}%` }}
                />
              </div>
              <p className="mt-1 text-[9px] text-zinc-400">
                {(s.achievedDwellMs / 1000).toFixed(1)}s / {(config.requiredHoldMs / 1000).toFixed(1)}s
              </p>
            </div>
          </div>
          <div className="mt-2 min-h-0 flex-1 space-y-1.5 overflow-hidden">
            {Object.entries(config.pose).map(([joint, target]) => {
              const pct = Math.max(0, Math.min(100, s.perJoint[joint] ?? 0));
              return (
                <div key={joint}>
                  <div className="flex items-baseline justify-between text-[10px]">
                    <span className="truncate font-semibold text-zinc-100">{joint}</span>
                    <span className="tabular text-zinc-400">{pct.toFixed(0)}%</span>
                  </div>
                  <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                    <div
                      className={`h-full transition-all ${
                        pct >= 90 ? "bg-emerald-500" : pct >= 60 ? "bg-amber-500" : "bg-rose-500"
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <ScoreHUD
        score={score}
        timer={`${(remainingMs / 1000).toFixed(1)}s hold left`}
        feedback={feedback}
        feedbackTone={tone}
      />

      <div className="rounded-card border border-zinc-700 bg-zinc-900/80 p-6">
        <div className="flex items-start gap-6">
          {/* Overall match gauge */}
          <div className="flex flex-col items-center">
            <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
              Match
            </p>
            <div
              className={`mt-2 flex h-28 w-28 items-center justify-center rounded-full border-4 ${
                s.achieved
                  ? "border-emerald-400 bg-emerald-500/20"
                  : overallPct >= 60
                  ? "border-amber-400 bg-amber-500/15"
                  : "border-rose-400 bg-rose-500/15"
              }`}
            >
              <p className="tabular text-2xl font-bold text-white">
                {overallPct.toFixed(0)}%
              </p>
            </div>
            <p className="mt-2 text-[10px] uppercase tracking-[0.14em] text-zinc-500">
              Threshold {config.achievedThresholdPct.toFixed(0)}%
            </p>
          </div>

          {/* Per-joint bars */}
          <div className="flex-1 space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-400">
              Per-joint match
            </p>
            {Object.entries(config.pose).map(([joint, target]) => {
              const pct = Math.max(0, Math.min(100, s.perJoint[joint] ?? 0));
              const current = currentAngles[joint];
              return (
                <div key={joint}>
                  <div className="flex items-baseline justify-between text-[11px]">
                    <span className="font-semibold text-zinc-100">{joint}</span>
                    <span className="tabular text-zinc-400">
                      target {target.value.toFixed(0)}° ± {target.tolerance.toFixed(0)} ·
                      now {current !== undefined ? current.toFixed(0) + "°" : "—"}
                    </span>
                  </div>
                  <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                    <div
                      className={`h-full transition-all ${
                        pct >= 90
                          ? "bg-emerald-500"
                          : pct >= 60
                          ? "bg-amber-500"
                          : "bg-rose-500"
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}

            {/* Hold progress */}
            <div className="pt-4">
              <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
                Hold progress
              </p>
              <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full bg-cyan-400 transition-all"
                  style={{ width: `${holdProgressPct}%` }}
                />
              </div>
              <p className="mt-1 text-[10px] text-zinc-400">
                {(s.achievedDwellMs / 1000).toFixed(1)} s of{" "}
                {(config.requiredHoldMs / 1000).toFixed(1)} s required
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
