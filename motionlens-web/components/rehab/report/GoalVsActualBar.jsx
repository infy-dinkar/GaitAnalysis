"use client";
// GoalVsActualBar — thin bar comparing a completed count against a
// target. Pure CSS / Tailwind — PDF-safe under html2canvas-pro.
//
// Layout: rounded track (bg-elevated) filled left→right in the accent
// tone. Numeric label sits above ("6 / 10 reps"). Under-fill = missed;
// over-fill (>100 %) clamps at 100 % and shows a check pill.

/**
 * @typedef {object} GoalVsActualBarProps
 * @property {string} label            e.g. "Reps"
 * @property {number} actual
 * @property {number} goal
 * @property {string} [unit]           e.g. "reps", "s", "zones"
 * @property {"accent"|"emerald"|"amber"|"sky"|"orange"} [tone]
 * @property {string} [caption]        Optional subtext under the bar
 */

const TONE = {
  accent: { fill: "bg-accent", text: "text-accent" },
  emerald: { fill: "bg-emerald-500", text: "text-emerald-500" },
  amber: { fill: "bg-amber-500", text: "text-amber-500" },
  sky: { fill: "bg-sky-500", text: "text-sky-500" },
  orange: { fill: "bg-orange-500", text: "text-orange-500" },
};

/**
 * @param {GoalVsActualBarProps} props
 */
export function GoalVsActualBar({
  label,
  actual,
  goal,
  unit = "",
  tone = "accent",
  caption,
}) {
  const safeGoal = Number.isFinite(goal) && goal > 0 ? goal : 0;
  const safeActual = Number.isFinite(actual) ? Math.max(0, actual) : 0;
  const pct =
    safeGoal > 0
      ? Math.min(100, Math.round((safeActual / safeGoal) * 100))
      : 0;
  const tones = TONE[tone] ?? TONE.accent;
  const complete = safeGoal > 0 && safeActual >= safeGoal;
  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="flex items-baseline justify-between">
        <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
          {label}
        </p>
        <p className="tabular text-xs text-muted">
          {formatNum(safeActual)}
          {safeGoal > 0 && (
            <>
              <span className="text-subtle"> / </span>
              {formatNum(safeGoal)}
            </>
          )}
          {unit ? <span className="ml-1 text-subtle">{unit}</span> : null}
        </p>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-elevated">
          <div
            className={`h-full ${tones.fill} transition-all`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className={`tabular text-[10px] font-semibold ${tones.text}`}>
          {pct}%
        </span>
      </div>
      {(caption || complete) && (
        <p className="mt-2 text-[11px] text-muted">
          {complete && !caption ? "Target reached." : caption}
        </p>
      )}
    </div>
  );
}

function formatNum(n) {
  if (!Number.isFinite(n)) return "—";
  if (Number.isInteger(n)) return n.toString();
  return n.toFixed(1);
}
