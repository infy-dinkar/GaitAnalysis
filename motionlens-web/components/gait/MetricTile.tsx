import { cn } from "@/lib/utils";

export type Status = "good" | "fair" | "poor" | "neutral";

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
}

export function MetricTile({
  label,
  value,
  hint,
  status = "neutral",
  className,
}: MetricTileProps) {
  return (
    <div
      className={cn(
        "rounded-card border bg-surface p-5",
        tileBorder[status],
        className,
      )}
    >
      <div className="text-xs uppercase tracking-[0.12em] text-subtle">{label}</div>
      <div className={cn("mt-2 tabular text-3xl font-semibold leading-none", tileText[status])}>
        {value}
      </div>
      {hint && <div className="mt-2 text-xs text-muted">{hint}</div>}
    </div>
  );
}
