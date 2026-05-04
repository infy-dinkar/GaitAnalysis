"use client";
// Single report row — used in patient detail page report list.

import Link from "next/link";
import { Activity, Footprints, PersonStanding, ChevronRight } from "lucide-react";
import type { ReportSummaryDTO } from "@/lib/reports";

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
};

export function ReportCard({ report }: { report: ReportSummaryDTO }) {
  const meta = MODULE_META[report.module];
  const Icon = meta.icon;

  // Build a descriptive subline based on module-specific fields
  const subline = [
    report.body_part && capitalize(report.body_part),
    report.movement && humanize(report.movement),
    report.side && capitalize(report.side),
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Link
      href={`/dashboard/reports/${report.id}`}
      className="group flex items-center gap-4 rounded-card border border-border bg-surface p-4 transition hover:border-accent"
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
        <p className="text-xs tabular text-subtle">{formatDateTime(report.created_at)}</p>
      </div>

      <ChevronRight className="h-4 w-4 text-muted transition group-hover:text-accent" />
    </Link>
  );
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatDateTime(iso: string): string {
  try {
    const d = new Date(iso);
    const today = new Date();
    const sameDay = d.toDateString() === today.toDateString();
    if (sameDay) {
      return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    }
    return d.toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
    });
  } catch {
    return "";
  }
}
