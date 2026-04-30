import { cn } from "@/lib/utils";

export interface ResultRow {
  label: string;
  value: string;
  reference?: string;
  status?: "good" | "fair" | "poor" | "neutral";
}

const statusColor: Record<NonNullable<ResultRow["status"]>, string> = {
  good: "text-accent",
  fair: "text-warning",
  poor: "text-error",
  neutral: "text-foreground",
};

export function ResultsTable({
  rows,
  className,
}: {
  rows: ResultRow[];
  className?: string;
}) {
  return (
    <div
      className={cn(
        "overflow-hidden rounded-card border border-border bg-surface",
        className,
      )}
    >
      <table className="w-full text-left text-sm">
        <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
          <tr>
            <th className="px-5 py-3 font-medium">Metric</th>
            <th className="px-5 py-3 font-medium">Value</th>
            <th className="hidden px-5 py-3 font-medium md:table-cell">Reference</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.label}
              className={cn(
                "border-b border-border last:border-b-0",
                i % 2 === 1 && "bg-background/40",
              )}
            >
              <td className="px-5 py-4 text-muted">{row.label}</td>
              <td
                className={cn(
                  "px-5 py-4 tabular font-medium",
                  statusColor[row.status ?? "neutral"],
                )}
              >
                {row.value}
              </td>
              <td className="hidden px-5 py-4 text-muted md:table-cell">
                {row.reference ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
