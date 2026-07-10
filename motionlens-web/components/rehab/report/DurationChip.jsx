"use client";
// DurationChip — small pill formatting a duration in seconds/minutes.
// Reused across every mechanic (all rehab reports save duration_sec).

import { Timer } from "lucide-react";

/**
 * @typedef {object} DurationChipProps
 * @property {number} seconds
 * @property {string} [label]     Defaults to "Duration"
 */

/**
 * @param {DurationChipProps} props
 */
export function DurationChip({ seconds, label = "Duration" }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1">
      <Timer className="h-3.5 w-3.5 text-muted" />
      <span className="text-[10px] uppercase tracking-[0.14em] text-subtle">
        {label}
      </span>
      <span className="tabular text-xs font-semibold text-foreground">
        {formatDuration(seconds)}
      </span>
    </div>
  );
}

function formatDuration(sec) {
  if (!Number.isFinite(sec) || sec <= 0) return "—";
  if (sec < 60) return `${sec.toFixed(0)} s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m} min ${s.toString().padStart(2, "0")} s`;
}
