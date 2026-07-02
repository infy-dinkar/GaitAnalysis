"use client";
// Rehab day-streak display badge.
//
// Kemtai / ViFive style: 🔥 fire icon + current streak count, with
// best-streak shown as a smaller secondary line when it's longer
// than the current run. Falls back to an encouraging "Start your
// streak!" state when the patient has no active streak.
//
// Presentational only — takes a pre-computed StreakResult prop.
// Fetching + streak computation live in the mounting page.

import { Flame } from "lucide-react";
import type { StreakResult } from "@/lib/rehab/streak";

interface Props {
  streak: StreakResult;
  /** Extra classes appended after the base container. */
  className?: string;
}

export function RehabStreakBadge({ streak, className }: Props) {
  const active = streak.currentStreak > 0;
  const extra = className ? ` ${className}` : "";

  if (!active) {
    return (
      <div
        className={
          "inline-flex items-center gap-3 rounded-card border border-border "
          + "bg-surface px-4 py-3 text-sm" + extra
        }
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-orange-500/10">
          <Flame className="h-5 w-5 text-orange-400 opacity-70" />
        </div>
        <div>
          <p className="font-semibold text-foreground">Start your streak!</p>
          <p className="text-xs text-muted">
            Complete a rehab session today to begin.
          </p>
        </div>
      </div>
    );
  }

  const showBest = streak.bestStreak > streak.currentStreak;

  return (
    <div
      className={
        "inline-flex items-center gap-3 rounded-card border border-orange-500/30 "
        + "bg-gradient-to-br from-orange-500/10 to-amber-500/5 px-4 py-3"
        + extra
      }
      role="status"
      aria-label={`${streak.currentStreak}-day rehab streak`}
    >
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-orange-500/20 ring-1 ring-orange-500/40">
        <Flame className="h-5 w-5 text-orange-500" />
      </div>
      <div className="min-w-0">
        <p className="tabular text-lg font-bold leading-tight text-foreground">
          {streak.currentStreak}-day streak
        </p>
        {showBest ? (
          <p className="mt-0.5 text-[11px] text-muted">
            Best: {streak.bestStreak} day{streak.bestStreak === 1 ? "" : "s"}
          </p>
        ) : (
          <p className="mt-0.5 text-[11px] text-muted">
            Keep it going!
          </p>
        )}
      </div>
    </div>
  );
}
