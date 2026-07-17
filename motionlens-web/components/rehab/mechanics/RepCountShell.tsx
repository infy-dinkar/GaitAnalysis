"use client";
// Rep-Count Gate mechanic — UI shell.
//
// Visual: large rep counter + depth bar visualising the current
// signal between depthThreshold and topThreshold. The bar fills
// downward as the patient descends. Streak displayed via the
// shared HUD; per-rep feedback ("shallow", "jerky") shown
// transiently after each rep closes.

import { useEffect, useRef, useState } from "react";
import {
  type RepCountConfig,
  type RepCountState,
  type Score,
  emptyRepCountState,
  emptyScore,
} from "@/lib/rehab/gameState";
import { repCountStep } from "@/lib/rehab/mechanics";
import { ScoreHUD } from "@/components/rehab/mechanics/ScoreHUD";
import {
  interiorToDisplayFlexion,
} from "@/lib/rehab/kneeAngleDisplay";

interface Props {
  signal: number;
  signalLabel?: string;
  /** Optional total rep target — when reps reach this, shell flips
   *  to a "session complete" state. */
  targetReps?: number;
  config: RepCountConfig;
  /** Optional session-state harvester. Called on each meaningful
   *  transition (delta-gated: only when reps or score changes) so
   *  the mounting page can build a save payload without peeking
   *  into shell internals. Additive; backward-compatible when
   *  omitted. */
  onSnapshot?: (state: RepCountState, score: Score) => void;
  /** Compact mode — tighter card + shorter depth bar + no verbose
   *  phase/currentMin/etc rows. Used inside the live-mode sidebar. */
  compact?: boolean;
  /** Display-side convention hint. When the caller feeds the engine
   *  with `interior` (squat-family pages), pass "knee_interior" and
   *  the HUD readouts (`now`, `top`, `depth`) are transformed via
   *  `interiorToDisplayFlexion` before render. The bar math and the
   *  engine input are UNCHANGED — this only relabels displayed
   *  numbers so physios see flexion (0° standing, ~70° deep). */
  signalDisplayName?: "knee_interior" | "knee_flexion";
}

export function RepCountShell({
  signal,
  signalLabel = "Signal",
  targetReps,
  config,
  onSnapshot,
  compact = false,
  signalDisplayName,
}: Props) {
  const stateRef = useRef<RepCountState>(emptyRepCountState());
  const scoreRef = useRef<Score>(emptyScore());
  const [feedback, setFeedback] = useState<string | null>(null);
  const [feedbackTone, setFeedbackTone] = useState<"good" | "bad" | "neutral">(
    "neutral",
  );
  const [, setTick] = useState(0);

  // Mirror live props in a ref so the rAF loop reads the latest
  // values. No setState here — safe at 60 Hz prop updates.
  const propsRef = useRef({ signal, config });
  useEffect(() => {
    propsRef.current = { signal, config };
  }, [signal, config]);

  // Latest onSnapshot in a ref so the memoised rAF loop reads it
  // without needing to rebind. Same pattern as RehabCameraShell's
  // onFrame / angleArc refs.
  const onSnapshotRef = useRef(onSnapshot);
  useEffect(() => {
    onSnapshotRef.current = onSnapshot;
  }, [onSnapshot]);

  // Delta-gate the snapshot emit: only fire when reps or points
  // actually changed. Both are monotonically non-decreasing in
  // normal play, so this covers every rep_counted event without
  // spamming the consumer 60× per second.
  const lastEmitRef = useRef<{ reps: number; points: number } | null>(null);

  // Single rAF loop, started once on mount. See HoldInZoneShell
  // for the rationale — fixes both max-update-depth at 60 Hz and
  // engine-stalling when the signal stops changing.
  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    const loop = () => {
      if (cancelled) return;
      const now = performance.now();
      const { signal: s, config: c } = propsRef.current;
      const r = repCountStep(stateRef.current, scoreRef.current, s, c, now);
      stateRef.current = r.state;
      scoreRef.current = r.score;
      // Delta-gated snapshot — fire on any rep-or-points change.
      const last = lastEmitRef.current;
      if (
        !last
        || last.reps !== r.state.reps
        || last.points !== r.score.points
      ) {
        lastEmitRef.current = {
          reps: r.state.reps,
          points: r.score.points,
        };
        onSnapshotRef.current?.(r.state, r.score);
      }
      if (r.event?.kind === "rep_counted") {
        const downgrade = r.event.payload?.downgrade as string | null;
        if (downgrade === "shallow") {
          setFeedback("Go deeper");
          setFeedbackTone("bad");
        } else if (downgrade === "jerky") {
          setFeedback("Smooth it out");
          setFeedbackTone("bad");
        } else {
          setFeedback("Good rep");
          setFeedbackTone("good");
        }
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
  const complete = targetReps != null && s.reps >= targetReps;

  // Depth bar: 0 % at topThreshold, 100 % at depthThreshold.
  // Math stays in raw engine (interior) units — the visual is
  // symmetric either way (deep = full bar). Only the number labels
  // below get transformed.
  const range = Math.max(
    1e-6,
    config.topThreshold - config.depthThreshold,
  );
  const depthPct = Math.min(
    100,
    Math.max(0, ((config.topThreshold - signal) / range) * 100),
  );

  // HUD label transform — flex when the caller opted into flexion
  // display. Applies to signal (now), topThreshold (top), and
  // depthThreshold (depth). Guards against non-finite/undefined.
  const isKneeFlexionDisplay = signalDisplayName === "knee_interior";
  const displaySignal = isKneeFlexionDisplay
    ? (interiorToDisplayFlexion(signal) ?? signal)
    : signal;
  const displayTopThreshold = isKneeFlexionDisplay
    ? (interiorToDisplayFlexion(config.topThreshold) ?? config.topThreshold)
    : config.topThreshold;
  const displayDepthThreshold = isKneeFlexionDisplay
    ? (interiorToDisplayFlexion(config.depthThreshold) ?? config.depthThreshold)
    : config.depthThreshold;

  const timer =
    targetReps != null
      ? `${s.reps} / ${targetReps}`
      : `${s.reps} reps`;

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
        {/* Depth card grows to fill the remaining sidebar space; the
            bar inside stretches with it so the operator always sees
            a full-height depth visualisation, not a token stub. */}
        <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-zinc-700 bg-zinc-900/80 p-3">
          <div className="flex min-h-0 flex-1 items-stretch gap-3">
            <div className="flex flex-col items-center justify-center">
              <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">
                Reps
              </p>
              <p className="tabular text-4xl font-bold leading-none text-white">
                {s.reps}
              </p>
              {complete && (
                <span className="mt-1 rounded-full bg-emerald-500/20 px-2 py-0.5 text-[9px] font-semibold text-emerald-200">
                  Complete
                </span>
              )}
            </div>
            <div className="flex min-h-0 flex-1 flex-col">
              <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-zinc-400">
                Depth · {signalLabel}
              </p>
              <div className="mt-1 flex min-h-0 flex-1 items-stretch gap-2">
                <div className="relative w-10 overflow-hidden rounded-md border border-zinc-700 bg-zinc-950">
                  <div
                    className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-orange-500 to-amber-300 transition-all"
                    style={{ height: `${depthPct}%` }}
                  />
                  <div className="absolute inset-x-0 top-0 h-[2px] bg-emerald-400/70" />
                </div>
                <div className="flex flex-1 flex-col justify-between py-1 text-[10px] text-zinc-400">
                  <span className="tabular">
                    top {displayTopThreshold.toFixed(0)}
                  </span>
                  <span className="tabular text-zinc-100">
                    now <span className="font-semibold">{displaySignal.toFixed(0)}</span>
                  </span>
                  <span className="tabular">
                    depth {displayDepthThreshold.toFixed(0)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <ScoreHUD score={score} timer={timer} feedback={feedback} feedbackTone={feedbackTone} />

      <div className="rounded-card border border-zinc-700 bg-zinc-900/80 p-6">
        <div className="flex items-start gap-6">
          {/* Rep counter */}
          <div className="flex flex-col items-center">
            <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
              Reps
            </p>
            <p className="tabular text-5xl font-bold text-white md:text-6xl">
              {s.reps}
            </p>
            {s.goodReps !== s.reps && (
              <p className="mt-1 text-[10px] text-zinc-400">
                <span className="text-emerald-400">{s.goodReps} good</span>{" "}
                · {s.reps - s.goodReps} flagged
              </p>
            )}
            {complete && (
              <span className="mt-3 rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-200">
                Set complete
              </span>
            )}
          </div>

          {/* Depth bar */}
          <div className="flex-1">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-400">
              Depth · {signalLabel}
            </p>
            <div className="mt-2 flex items-stretch gap-3">
              <div className="relative h-40 w-12 overflow-hidden rounded-lg border border-zinc-700 bg-zinc-950">
                <div
                  className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-orange-500 to-amber-300 transition-all"
                  style={{ height: `${depthPct}%` }}
                />
                {/* Depth threshold tick */}
                <div className="absolute inset-x-0 top-0 h-[2px] bg-emerald-400/70" />
              </div>
              <div className="flex flex-col justify-between text-[10px] text-zinc-400">
                <span className="tabular">
                  top {displayTopThreshold.toFixed(0)}
                </span>
                <span className="tabular text-zinc-200">
                  now <span className="font-semibold">{displaySignal.toFixed(1)}</span>
                </span>
                <span className="tabular">
                  depth {displayDepthThreshold.toFixed(0)}
                </span>
              </div>
              <div className="flex-1 space-y-2 text-xs text-zinc-300">
                <Row label="Phase" value={s.phase} />
                <Row
                  label="Current min"
                  value={
                    Number.isFinite(s.currentRepMin)
                      ? s.currentRepMin.toFixed(1)
                      : "—"
                  }
                />
                <Row
                  label="Current max"
                  value={
                    Number.isFinite(s.currentRepMax)
                      ? s.currentRepMax.toFixed(1)
                      : "—"
                  }
                />
                <Row
                  label="Min amplitude"
                  value={config.minAmplitude.toFixed(0)}
                />
                {s.lastRepDowngrade && (
                  <Row
                    label="Last rep"
                    value={s.lastRepDowngrade}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] uppercase tracking-[0.14em] text-zinc-500">
        {label}
      </span>
      <span className="tabular font-semibold text-zinc-100">{value}</span>
    </div>
  );
}
