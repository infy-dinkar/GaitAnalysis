import { cn } from "@/lib/utils";
import type { ReliabilityEntryDTO } from "@/lib/api";

export type Status = "good" | "fair" | "poor" | "neutral";

/** Dot colour + label per reliability tier. Kept here so every gait
 *  surface (tile, chart title, PDF) spells the tiers identically. */
export const RELIABILITY_UI: Record<
  ReliabilityEntryDTO["tier"],
  { label: string; dot: string; text: string }
> = {
  reliable:     { label: "Reliable",     dot: "bg-accent",  text: "text-accent" },
  caution:      { label: "Caution",      dot: "bg-warning", text: "text-warning" },
  not_assessed: { label: "Not assessed", dot: "bg-error",   text: "text-error" },
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
    <span className={cn("inline-flex items-center gap-1.5 text-[11px]", ui.text, className)}>
      <span className={cn("inline-block h-2 w-2 rounded-full", ui.dot)} aria-hidden />
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
}

export function MetricTile({
  label,
  value,
  hint,
  status = "neutral",
  className,
  reliability,
}: MetricTileProps) {
  const suppressed = reliability?.tier === "not_assessed";
  const shownValue = suppressed ? "—" : value;
  const shownStatus: Status = suppressed ? "neutral" : status;
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
        <div className={cn("mt-2 text-[11px] leading-snug", RELIABILITY_UI[reliability.tier].text)}>
          {reliability.note}
        </div>
      )}
    </div>
  );
}
