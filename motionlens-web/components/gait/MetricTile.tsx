import { cn } from "@/lib/utils";
import type { ReliabilityEntryDTO } from "@/lib/api";

export type Status = "good" | "fair" | "poor" | "neutral";

/** Dot colour + label per reliability tier. Kept here so every gait
 *  surface (tile, chart title, PDF) spells the tiers identically. */
export const RELIABILITY_UI: Record<
  ReliabilityEntryDTO["tier"],
  { label: string; color: string }
> = {
  // HARD-CODED on purpose. This theme's --color-accent is orange
  // (#EA580C light / #F97316 dark), so a "Reliable" dot on bg-accent was
  // indistinguishable from a "Caution" dot on bg-warning (#F59E0B) —
  // every tier read amber in screenshots. Inline styles bypass the
  // theme so the three tiers are always green / amber / red.
  reliable:     { label: "Reliable",     color: "#22c55e" },
  caution:      { label: "Caution",      color: "#f59e0b" },
  not_assessed: { label: "Not assessed", color: "#ef4444" },
};

/** Small status dot + tier label. Renders nothing when there is no
 *  entry, so reports saved before reliability existed are untouched. */
export function ReliabilityBadge({
  entry,
  className,
}: {
  entry?: ReliabilityEntryDTO | null;
  className?: string;
}) {
  if (!entry) return null;
  const ui = RELIABILITY_UI[entry.tier];
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 text-[11px]", className)}
      style={{ color: ui.color }}
      data-tier={entry.tier}
    >
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ backgroundColor: ui.color }}
        aria-hidden
      />
      {ui.label}
    </span>
  );
}

const tileBorder: Record<Status, string> = {
  good: "border-accent/50",
  fair: "border-warning/50",
  poor: "border-error/50",
  neutral: "border-border",
};
const tileText: Record<Status, string> = {
  good: "text-accent",
  fair: "text-warning",
  poor: "text-error",
  neutral: "text-foreground",
};

interface MetricTileProps {
  label: string;
  value: string;
  hint?: string;
  status?: Status;
  className?: string;
  /** Landmark reliability behind this number. When the tier is
   *  not_assessed the DISPLAYED value becomes "—" — the number is still
   *  in the saved data, it is just not shown as a measurement. */
  reliability?: ReliabilityEntryDTO | null;
  /** Optional second row (used by Knee peak to show the other leg):
   *  its own label, value and tier. A not_assessed tier shows "—". */
  secondary?: {
    label: string;
    value: string;
    reliability?: ReliabilityEntryDTO | null;
    reason?: string;
  } | null;
  /** Overrides the "—" suppression rule for the primary value (the knee
   *  tile decides suppression per side itself). */
  suppressWhenNotAssessed?: boolean;
}

export function MetricTile({
  label,
  value,
  hint,
  status = "neutral",
  className,
  reliability,
  secondary,
  suppressWhenNotAssessed = true,
}: MetricTileProps) {
  const suppressed = suppressWhenNotAssessed && reliability?.tier === "not_assessed";
  const shownValue = suppressed ? "—" : value;
  const shownStatus: Status = suppressed ? "neutral" : status;
  const secondarySuppressed = secondary?.reliability?.tier === "not_assessed";
  return (
    <div
      className={cn(
        "rounded-card border bg-surface p-5",
        tileBorder[shownStatus],
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="text-xs uppercase tracking-[0.12em] text-subtle">{label}</div>
        <ReliabilityBadge entry={reliability} />
      </div>
      <div className={cn("mt-2 tabular text-3xl font-semibold leading-none", tileText[shownStatus])}>
        {shownValue}
      </div>
      {hint && <div className="mt-2 text-xs text-muted">{hint}</div>}
      {reliability && (
        <div
          className="mt-2 text-[11px] leading-snug"
          style={{ color: RELIABILITY_UI[reliability.tier].color }}
        >
          {reliability.note}
        </div>
      )}
      {secondary && (
        <div className="mt-3 border-t border-border/60 pt-2 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="text-subtle">{secondary.label}</span>
            <ReliabilityBadge entry={secondary.reliability} />
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="tabular font-semibold text-foreground">
              {secondarySuppressed ? "—" : secondary.value}
            </span>
            {secondary.reason && (
              <span
                className="text-[11px] leading-snug"
                style={secondary.reliability
                  ? { color: RELIABILITY_UI[secondary.reliability.tier].color }
                  : undefined}
              >
                {secondary.reason}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
