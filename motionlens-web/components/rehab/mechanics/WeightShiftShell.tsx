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
  /** Full left↔right cycle rep count (owned by the page) — the primary
   *  progress indicator + completion metric. */
  reps: number;
  /** Target reps that completes the session (auto-save fires there). */
  repTarget: number;
  /** Optional session-state harvester — same additive pattern as
   *  RepCountShell.onSnapshot. Fires when captured zones change so
   *  pages can persist mechanic_state without peeking. */
  onSnapshot?: (state: WeightShiftState, score: Score) => void;
  /** Compact live-mode variant. */
  compact?: boolean;
}

export function WeightShiftShell({
  shift,
  stepDetected,
  config,
  reps,
  repTarget,
  onSnapshot,
  compact = false,
}: Props) {
  const stateRef = useRef<WeightShiftState>(emptyWeightShiftState());
  const scoreRef = useRef<Score>(emptyScore());
  const [feedback, setFeedback] = useState<string | null>(null);
  const [feedbackTone, setFeedbackTone] =
    useState<"good" | "bad" | "neutral">("neutral");
  // Brief highlight pulse on the rep counter each time a rep lands —
  // immediate visual feedback. Driven by the `reps` prop increment.
  const [flash, setFlash] = useState(false);
  const flashTimeoutRef = useRef<number | null>(null);
  const prevRepsRef = useRef(reps);
  useEffect(() => {
    if (reps > prevRepsRef.current) {
      setFlash(true);
      if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
      flashTimeoutRef.current = window.setTimeout(() => setFlash(false), 700);
    }
    prevRepsRef.current = reps;
  }, [reps]);
  const [, setTick] = useState(0);
  const onSnapshotRef = useRef(onSnapshot);
  useEffect(() => {
    onSnapshotRef.current = onSnapshot;
  }, [onSnapshot]);
  const lastEmitRef = useRef<{ captured: number; points: number } | null>(null);
  // Elapsed session timer — starts on the first live frame and freezes
  // once the rep target is reached. Surfaced in the ScoreHUD "Time" slot.
  const startAtRef = useRef<number | null>(null);
  const elapsedMsRef = useRef(0);

  // Mirror live props in a ref so the rAF loop reads the latest
  // values. No setState here — safe at 60 Hz prop updates.
  const propsRef = useRef({ shift, stepDetected, config, reps, repTarget });
  useEffect(() => {
    propsRef.current = { shift, stepDetected, config, reps, repTarget };
  }, [shift, stepDetected, config, reps, repTarget]);

  // Single rAF loop, started once on mount. See HoldInZoneShell
  // for the rationale.
  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    const loop = () => {
      if (cancelled) return;
      const now = performance.now();
      const { shift: sh, stepDetected: sd, config: c, reps: rp, repTarget: rt } = propsRef.current;
      const r = weightShiftStep(stateRef.current, scoreRef.current, sh, sd, c, now);
      stateRef.current = r.state;
      scoreRef.current = r.score;
      const capturedNow = r.state.capturedZoneIds.length;
      // Tick the elapsed timer until the rep target is reached.
      if (startAtRef.current === null) startAtRef.current = now;
      if (rp < rt) {
        elapsedMsRef.current = now - startAtRef.current;
      }
      const last = lastEmitRef.current;
      if (
        !last
        || last.captured !== capturedNow
        || last.points !== r.score.points
      ) {
        lastEmitRef.current = { captured: capturedNow, points: r.score.points };
        onSnapshotRef.current?.(r.state, r.score);
      }
      if (r.event?.kind === "zone_captured") {
        setFeedback("Zone captured");
        setFeedbackTone("good");
        // (Rep-counter pulse is driven by the `reps` prop, not zones.)
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
      if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    };
  }, []);

  const s = stateRef.current;
  const score = scoreRef.current;
  // Elapsed session time as M:SS for the ScoreHUD "Time" slot. (The rep
  // count has its own prominent indicator below.)
  const elapsedSec = Math.floor(elapsedMsRef.current / 1000);
  const timer = `${Math.floor(elapsedSec / 60)}:${String(elapsedSec % 60).padStart(2, "0")}`;

  // Cursor position 0..100% across the track. shift -1 maps to 0%,
  // +1 maps to 100%.
  const cursorPct = ((s.cursor + 1) / 2) * 100;

  if (compact) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <ScoreHUD
          score={score}
          timer={timer}
          feedback={feedback}
          feedbackTone={feedbackTone}
          compact
        />
        {/* Prominent REP counter — one full left↔right cycle = 1 rep.
            Big number updates the instant a rep lands; a green pulse +
            "+1 rep" badge gives immediate feedback. Session auto-saves
            at repTarget. Mirrors RepCountShell's Reps block. */}
        <div
          className={`flex items-center justify-between rounded-lg border px-3 py-2 transition-all duration-200 ${
            flash
              ? "border-emerald-400 bg-emerald-500/20 ring-2 ring-emerald-400/60"
              : "border-zinc-700 bg-zinc-900/80"
          }`}
        >
          <div>
            <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">
              Reps
            </p>
            <p className="tabular text-3xl font-bold leading-none text-white">
              {Math.min(reps, repTarget)}
              <span className="text-lg font-semibold text-zinc-500"> / {repTarget}</span>
            </p>
          </div>
          {flash ? (
            <span className="rounded-full bg-emerald-500/30 px-2 py-0.5 text-[10px] font-semibold text-emerald-100 ring-1 ring-emerald-400/50">
              +1 rep
            </span>
          ) : reps >= repTarget ? (
            <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[9px] font-semibold text-emerald-200">
              Complete
            </span>
          ) : null}
        </div>
        <div className="relative flex min-h-0 flex-1 flex-col rounded-lg border border-zinc-700 bg-zinc-900/80 p-3">
          {stepDetected && (
            <div className="absolute right-2 top-2 inline-flex items-center gap-1 rounded-full bg-rose-500/20 px-2 py-0.5 text-[10px] font-semibold text-rose-200 ring-1 ring-rose-400/50">
              <Footprints className="h-3 w-3" /> Step paused
            </div>
          )}
          <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-zinc-400">
            Lateral weight shift
          </p>
          <div className="relative mt-4 h-12 w-full rounded-full border border-zinc-700 bg-zinc-950">
            <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-white/10" />
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
                />
              );
            })}
            <div
              className="absolute top-1/2 h-8 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-md bg-white shadow-[0_0_8px_rgba(255,255,255,0.5)]"
              style={{ left: `${cursorPct}%` }}
            />
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2 text-[10px]">
            <div>
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Zone</p>
              <p className="tabular text-zinc-100">{s.currentZoneId ?? "—"}</p>
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Dwell</p>
              <p className="tabular text-zinc-100">{(s.dwellMs / 1000).toFixed(1)}s</p>
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">Paused</p>
              <p className="tabular text-zinc-100">{(s.stepPausedMs / 1000).toFixed(1)}s</p>
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
