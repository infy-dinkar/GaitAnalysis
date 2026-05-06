"use client";
// Single report row — used in patient detail page report list.

import { useState } from "react";
import Link from "next/link";
import {
  Activity,
  Download,
  Footprints,
  PersonStanding,
  ChevronRight,
  Loader2,
  StretchHorizontal,
  Trash2,
} from "lucide-react";
import { deleteReport, type ReportSummaryDTO } from "@/lib/reports";
import { formatISTDate, formatISTTime } from "@/lib/format/datetime";

const MODULE_META: Record<
  ReportSummaryDTO["module"],
  { label: string; icon: typeof Activity; tone: string }
> = {
  gait: {
    label: "Gait analysis",
    icon: Footprints,
    tone: "bg-cyan-500/10 text-cyan-700 dark:text-cyan-400",
  },
  biomech: {
    label: "Biomechanics",
    icon: Activity,
    tone: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  },
  posture: {
    label: "Posture screening",
    icon: PersonStanding,
    tone: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  },
  trendelenburg: {
    label: "Trendelenburg test",
    icon: StretchHorizontal,
    tone: "bg-violet-500/10 text-violet-700 dark:text-violet-400",
  },
};

export function ReportCard({
  report,
  onDeleted,
}: {
  report: ReportSummaryDTO;
  /** Called after a successful delete so the parent can refresh the list. */
  onDeleted?: (id: string) => void;
}) {
  const meta = MODULE_META[report.module];
  const Icon = meta.icon;

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Build a descriptive subline based on module-specific fields
  const subline = [
    report.body_part && capitalize(report.body_part),
    report.movement && humanize(report.movement),
    report.side && capitalize(report.side),
  ]
    .filter(Boolean)
    .join(" · ");

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (busy) return;
    const label = subline ? `${meta.label} (${subline})` : meta.label;
    if (!confirm(`Delete this ${label} report? This cannot be undone.`)) return;
    setBusy(true);
    setErr(null);
    try {
      await deleteReport(report.id);
      onDeleted?.(report.id);
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : "Delete failed");
      setBusy(false);
    }
  }

  return (
    <div className="space-y-1">
      <div className="group flex items-center gap-3 rounded-card border border-border bg-surface p-4 transition hover:border-accent">
        <Link
          href={`/dashboard/reports/${report.id}`}
          className="flex flex-1 items-center gap-4 min-w-0"
        >
          <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-card ${meta.tone}`}>
            <Icon className="h-5 w-5" />
          </div>

          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-foreground">{meta.label}</p>
            {subline && (
              <p className="truncate text-xs text-muted">{subline}</p>
            )}
          </div>

          <div className="shrink-0 text-right">
            <p className="text-xs tabular text-foreground">{formatISTDate(report.created_at)}</p>
            <p className="text-[11px] tabular text-subtle">{formatISTTime(report.created_at)} IST</p>
          </div>

          <ChevronRight className="h-4 w-4 text-muted transition group-hover:text-accent" />
        </Link>

        <a
          href={`/dashboard/reports/${report.id}?download=1`}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Download this report as PDF"
          title="Download this report as PDF"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-muted transition hover:bg-accent/10 hover:text-accent"
        >
          <Download className="h-4 w-4" />
        </a>

        <button
          type="button"
          onClick={handleDelete}
          disabled={busy}
          aria-label="Delete this report"
          title="Delete this report"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-muted transition hover:bg-error/10 hover:text-error disabled:opacity-50"
        >
          {busy ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Trash2 className="h-4 w-4" />
          )}
        </button>
      </div>

      {err && (
        <p className="px-3 text-xs text-error">{err}</p>
      )}
    </div>
  );
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
