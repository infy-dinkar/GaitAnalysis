"use client";
// Shared HUD for every rehab game mechanic. Renders the score,
// streak, an optional timer, optional best-streak badge, and a
// transient feedback message that fades out after a few seconds.
//
// Pure presentation — no game logic. Each mechanic shell passes in
// the data it wants displayed.

import { useEffect, useState } from "react";
import { Flame, Trophy } from "lucide-react";
import type { Score } from "@/lib/rehab/gameState";

interface Props {
  score: Score;
  /** Optional remaining-time string ("0:23" / "2 of 3 hops left" /
   *  whatever the mechanic wants to surface). */
  timer?: string | null;
  /** Optional transient feedback ("Perfect!", "Too shallow",
   *  "Step paused", ...). Cleared automatically after ~1.6 s. */
  feedback?: string | null;
  feedbackTone?: "good" | "bad" | "neutral";
  /** Compact single-row variant for space-constrained shells
   *  (live-mode sidebar). Renders as a single chip strip instead of
   *  the three-column card grid. */
  compact?: boolean;
}

export function ScoreHUD({
  score,
  timer = null,
  feedback = null,
  feedbackTone = "neutral",
  compact = false,
}: Props) {
  const [shownFeedback, setShownFeedback] = useState<string | null>(null);

  useEffect(() => {
    if (!feedback) return;
    setShownFeedback(feedback);
    const id = window.setTimeout(() => setShownFeedback(null), 1600);
    return () => window.clearTimeout(id);
  }, [feedback]);

  const toneClass =
    feedbackTone === "good"
      ? "bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-400/40"
      : feedbackTone === "bad"
      ? "bg-rose-500/20 text-rose-200 ring-1 ring-rose-400/40"
      : "bg-zinc-700/70 text-zinc-100 ring-1 ring-white/20";

  if (compact) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900/80 px-3 py-2">
        <div className="min-w-0 flex-1">
          <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">
            Score
          </p>
          <p className="tabular text-lg font-semibold leading-none text-white">
            {Math.round(score.points)}
          </p>
        </div>
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1 text-[9px] uppercase tracking-[0.14em] text-zinc-500">
            <Flame className="h-2.5 w-2.5 text-orange-400" /> Streak
          </p>
          <p className="tabular text-lg font-semibold leading-none text-white">
            {score.streak}
            {score.bestStreak > 0 && (
              <span className="ml-1 text-[10px] font-normal text-zinc-400">
                / {score.bestStreak}
              </span>
            )}
          </p>
        </div>
        {timer && (
          <div className="min-w-0 flex-1">
            <p className="text-[9px] uppercase tracking-[0.14em] text-zinc-500">
              Time
            </p>
            <p className="tabular text-lg font-semibold leading-none text-white">
              {timer}
            </p>
          </div>
        )}
        {shownFeedback && (
          <div
            className={`pointer-events-none fixed left-1/2 top-24 z-50 -translate-x-1/2 rounded-full px-5 py-2 text-sm font-semibold shadow-lg ${toneClass}`}
          >
            {shownFeedback}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <div className="rounded-lg border border-zinc-700 bg-zinc-900/80 p-4">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-400">
          Score
        </p>
        <p className="tabular mt-1 text-2xl font-semibold text-white md:text-3xl">
          {Math.round(score.points)}
        </p>
      </div>

      <div className="rounded-lg border border-zinc-700 bg-zinc-900/80 p-4">
        <p className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-400">
          <Flame className="h-3 w-3 text-orange-400" /> Streak
        </p>
        <p className="tabular mt-1 text-2xl font-semibold text-white md:text-3xl">
          {score.streak}
        </p>
        {score.bestStreak > 0 && (
          <p className="mt-1 inline-flex items-center gap-1 text-[10px] text-zinc-400">
            <Trophy className="h-3 w-3 text-yellow-400" /> Best{" "}
            <span className="tabular text-zinc-200">{score.bestStreak}</span>
          </p>
        )}
      </div>

      <div className="rounded-lg border border-zinc-700 bg-zinc-900/80 p-4">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-400">
          {timer ? "Time" : "Feedback"}
        </p>
        {timer ? (
          <p className="tabular mt-1 text-2xl font-semibold text-white md:text-3xl">
            {timer}
          </p>
        ) : (
          <p className="mt-1 text-sm text-zinc-300">
            {shownFeedback ?? "—"}
          </p>
        )}
      </div>

      {shownFeedback && timer && (
        <div
          className={`pointer-events-none fixed left-1/2 top-24 z-50 -translate-x-1/2 rounded-full px-5 py-2 text-sm font-semibold shadow-lg ${toneClass}`}
        >
          {shownFeedback}
        </div>
      )}
    </div>
  );
}
