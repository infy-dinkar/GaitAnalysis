"use client";
// /dashboard/reports/[id] — view saved report (with all metrics + figures)

import { useEffect, useState, use as usePromise } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import {
  Activity,
  Calendar,
  FileText,
  Footprints,
  Loader2,
  PersonStanding,
} from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { getReport, type ReportDTO } from "@/lib/reports";

// Plotly is heavy + browser-only — load it lazily.
const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

const MODULE_META: Record<
  ReportDTO["module"],
  { label: string; icon: typeof Activity }
> = {
  gait: { label: "Gait analysis", icon: Footprints },
  biomech: { label: "Biomechanics", icon: Activity },
  posture: { label: "Posture screening", icon: PersonStanding },
};

export default function ReportViewPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = usePromise(params);
  return (
    <AuthGuard>
      <DashboardShell>
        <ReportView id={id} />
      </DashboardShell>
    </AuthGuard>
  );
}

function ReportView({ id }: { id: string }) {
  const [report, setReport] = useState<ReportDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getReport(id)
      .then((r) => !cancelled && setReport(r))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) {
    return (
      <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
        {error}
      </div>
    );
  }

  if (report === null) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-accent" />
      </div>
    );
  }

  const meta = MODULE_META[report.module];
  const Icon = meta.icon;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-card bg-accent/10 text-accent">
          <Icon className="h-6 w-6" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="eyebrow">{meta.label}</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight md:text-3xl">
            {[report.body_part, report.movement, report.side]
              .filter(Boolean)
              .map((s) => s && s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, " "))
              .join(" · ") || "Assessment report"}
          </h1>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-muted">
            <span className="inline-flex items-center gap-1">
              <Calendar className="h-3.5 w-3.5" />
              {new Date(report.created_at).toLocaleString()}
            </span>
            <Link
              href={`/dashboard/patients/${report.patient_id}`}
              className="inline-flex items-center gap-1 text-accent hover:underline"
            >
              <FileText className="h-3.5 w-3.5" />
              View patient
            </Link>
          </div>
        </div>
      </div>

      {/* Metrics */}
      {report.metrics && Object.keys(report.metrics).length > 0 && (
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Metrics
          </h2>
          <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
            <pre className="overflow-x-auto p-5 text-xs text-foreground">
              {JSON.stringify(report.metrics, null, 2)}
            </pre>
          </div>
        </section>
      )}

      {/* Figures */}
      {report.figures.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Charts ({report.figures.length})
          </h2>
          <div className="mt-3 space-y-4">
            {report.figures.map((fig, i) => {
              const data = (fig as { data?: unknown[] }).data;
              const layout = (fig as { layout?: Record<string, unknown> }).layout;
              if (!Array.isArray(data)) return null;
              return (
                <div
                  key={i}
                  className="rounded-card border border-border bg-surface p-4"
                >
                  <PlotlyChart
                    data={data as unknown[]}
                    layout={(layout as Record<string, unknown>) || {}}
                    height={360}
                  />
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Observations */}
      {report.observations && Object.keys(report.observations).length > 0 && (
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Observations
          </h2>
          <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
            <pre className="overflow-x-auto p-5 text-xs text-foreground">
              {JSON.stringify(report.observations, null, 2)}
            </pre>
          </div>
        </section>
      )}
    </div>
  );
}
