"use client";
// Weight-Shift mechanic — UI shell.
//
// Visual: horizontal track with target zones marked along it. The
// patient's lateral shift drives a cursor along the track; when
// the cursor sits inside a zone for its required dwell time, the
// zone is captured. A separate "step paused" banner appears when
// the stepDetected prop is true — the engine auto-pauses dwell
// accumulation in that case (the patient must stay grounded).

import { useEffect, useRef, useState } from "react";
import { Footprints } from "lucide-react";
import {
  type Score,
  type WeightShiftConfig,
  type WeightShiftState,
  emptyScore,
  emptyWeightShiftState,
} from "@/lib/rehab/gameState";
import { weightShiftStep } from "@/lib/rehab/mechanics";
import { ScoreHUD } from "@/components/rehab/mechanics/ScoreHUD";

interface Props {
  /** Lateral shift in [-1, +1]. -1 = full left, 0 = centre,
   *  +1 = full right. */
  shift: number;
  /** True if the patient has lifted a foot — game pauses dwell. */
  stepDetected: boolean;
  config: WeightShiftConfig;
}

export function WeightShiftShell({
  shift,
  stepDetected,
  config,
}: Props) {
  const stateRef = useRef<WeightShiftState>(emptyWeightShiftState());
  const scoreRef = useRef<Score>(emptyScore());
  const [feedback, setFeedback] = useState<string | null>(null);
  const [feedbackTone, setFeedbackTone] =
    useState<"good" | "bad" | "neutral">("neutral");
  const [, setTick] = useState(0);

  // Mirror live props in a ref so the rAF loop reads the latest
  // values. No setState here — safe at 60 Hz prop updates.
  const propsRef = useRef({ shift, stepDetected, config });
  useEffect(() => {
    propsRef.current = { shift, stepDetected, config };
  }, [shift, stepDetected, config]);

  // Single rAF loop, started once on mount. See HoldInZoneShell
  // for the rationale.
  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    const loop = () => {
      if (cancelled) return;
      const now = performance.now();
      const { shift: sh, stepDetected: sd, config: c } = propsRef.current;
      const r = weightShiftStep(stateRef.current, scoreRef.current, sh, sd, c, now);
      stateRef.current = r.state;
      scoreRef.current = r.score;
      if (r.event?.kind === "zone_captured") {
        setFeedback("Zone captured");
        setFeedbackTone("good");
      } else if (r.event?.kind === "step_paused") {
        setFeedback("Step detected — pausing");
        setFeedbackTone("bad");
      } else if (r.event?.kind === "exited_zone") {
        setFeedback("Out of zone");
        setFeedbackTone("neutral");
      }
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
  const captured = s.capturedZoneIds.length;
  const total = config.zones.length;
  const timer = `${captured} / ${total} captured`;

  // Cursor position 0..100% across the track. shift -1 maps to 0%,
  // +1 maps to 100%.
  const cursorPct = ((s.cursor + 1) / 2) * 100;

  return (
    <div className="space-y-4">
      <ScoreHUD
        score={score}
        timer={timer}
        feedback={feedback}
        feedbackTone={feedbackTone}
      />

      <div className="relative rounded-card border border-zinc-700 bg-zinc-900/80 p-6">
        {stepDetected && (
          <div className="absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-full bg-rose-500/20 px-3 py-1 text-xs font-semibold text-rose-200 ring-1 ring-rose-400/50">
            <Footprints className="h-3 w-3" />
            Step paused
          </div>
        )}

        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-400">
          Lateral weight shift
        </p>

        {/* Track */}
        <div className="relative mt-6 h-16 w-full rounded-full border border-zinc-700 bg-zinc-950">
          {/* Centre tick */}
          <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-white/10" />
          {/* Zones */}
          {config.zones.map((z) => {
            const leftPct = ((z.centre - z.halfWidth + 1) / 2) * 100;
            const widthPct = z.halfWidth * 2 * 50;
            const isCaptured = s.capturedZoneIds.includes(z.id);
            const isCurrent = s.currentZoneId === z.id;
            return (
              <div
                key={z.id}
                className={`absolute top-1 bottom-1 rounded-full ring-1 ${
                  isCaptured
                    ? "bg-emerald-500/30 ring-emerald-400"
                    : isCurrent
                    ? "bg-amber-500/30 ring-amber-400"
                    : "bg-cyan-500/15 ring-cyan-500/40"
                }`}
                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
              >
                <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap text-[10px] uppercase tracking-[0.12em] text-zinc-400">
                  {z.id}
                </span>
              </div>
            );
          })}
          {/* Cursor */}
          <div
            className="absolute top-1/2 h-10 w-3 -translate-x-1/2 -translate-y-1/2 rounded-md bg-white shadow-[0_0_10px_rgba(255,255,255,0.5)]"
            style={{ left: `${cursorPct}%` }}
          />
        </div>

        {/* Dwell progress under the current zone */}
        <div className="mt-10 grid grid-cols-3 gap-3 text-xs">
          <div>
            <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
              Current zone
            </p>
            <p className="tabular text-zinc-100">
              {s.currentZoneId ?? "—"}
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
              Dwell
            </p>
            <p className="tabular text-zinc-100">
              {(s.dwellMs / 1000).toFixed(1)} s
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
              Step paused
            </p>
            <p className="tabular text-zinc-100">
              {(s.stepPausedMs / 1000).toFixed(1)} s
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
