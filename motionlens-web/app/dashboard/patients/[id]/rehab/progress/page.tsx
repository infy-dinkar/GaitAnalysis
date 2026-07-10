"use client";
// Dedicated rehab-progress page. Two sections:
//   1. Proof of Progress — improvement graphs (baseline → rehab
//      trend → re-assessment) per exercise with a sharedMetric.
//   2. Session-history dashboard — reuses RehabProgressDashboard for
//      adherence / recent-sessions / charts.

import { useEffect, useMemo, useState, use as usePromise } from "react";
import Link from "next/link";
import { Dumbbell } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Button } from "@/components/ui/Button";
import { RehabStreakBadge } from "@/components/rehab/RehabStreakBadge";
import { RehabProgressDashboard } from "@/components/rehab/RehabProgressDashboard";
import { ProofChart } from "@/components/rehab/report/ProofChart";
import { computeStreak, type StreakResult } from "@/lib/rehab/streak";
import { listPatientReports, type ReportSummaryDTO } from "@/lib/reports";
import { computeProofArtifact } from "@/lib/rehab/proofArtifact";
import { SHARED_METRIC_SLUGS } from "@/lib/rehab/exerciseIndications";

// Loose type — the artifact is authored in JS with JSDoc, TS reads
// it as `any`. Widen to `unknown` in the collected list and cast at
// render time.
type ProofArtifact = {
  slug: string;
  unit: string;
  label: string;
  betterDirection: "higher" | "lower";
  caveat?: string;
  baseline: { date: string; value: number; reportId: string } | null;
  trend: { date: string; value: number; reportId: string }[];
  reassessment: { date: string; value: number; reportId: string } | null;
  improvementDelta?: number;
  assessmentsFound: number;
  sessionsFound: number;
};

export default function PatientRehabProgressPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = usePromise(params);
  return (
    <AuthGuard>
      <DashboardShell
        backHref={`/dashboard/patients/${id}/rehab`}
        backLabel="Rehab catalogue"
      >
        <Content patientId={id} />
      </DashboardShell>
    </AuthGuard>
  );
}

function Content({ patientId }: { patientId: string }) {
  const [reports, setReports] = useState<ReportSummaryDTO[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    listPatientReports(patientId)
      .then((res) => {
        if (!cancelled) setReports(res.data);
      })
      .catch(() => {
        if (!cancelled) setReports([]);
      });
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  const streak: StreakResult = useMemo(() => {
    const dates = (reports ?? [])
      .filter((r) => r.module === "rehab")
      .map((r) => r.created_at);
    return computeStreak(dates);
  }, [reports]);

  return (
    <div className="space-y-10">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Rehab progress</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            Proof of progress
          </h1>
          <p className="mt-2 max-w-xl text-sm text-muted">
            Baseline assessment → daily rehab sessions → re-assessment
            plotted on one axis in the same units. This is the closed-
            loop the platform was built for.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {reports !== null && <RehabStreakBadge streak={streak} />}
          <Link href={`/dashboard/patients/${patientId}/rehab`}>
            <Button variant="secondary">
              <Dumbbell className="h-4 w-4" />
              Start a session
            </Button>
          </Link>
        </div>
      </div>

      <ProofOfProgressSection patientId={patientId} />

      <div>
        <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-subtle">
          Session history & adherence
        </h2>
        <div className="mt-4">
          <RehabProgressDashboard patientId={patientId} />
        </div>
      </div>
    </div>
  );
}

function ProofOfProgressSection({ patientId }: { patientId: string }) {
  const [state, setState] = useState<
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; artifacts: ProofArtifact[] }
  >({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    Promise.all(
      SHARED_METRIC_SLUGS.map((slug) =>
        computeProofArtifact(patientId, slug).catch(() => null),
      ),
    )
      .then((results) => {
        if (cancelled) return;
        const artifacts = (results.filter(Boolean) as ProofArtifact[])
          // Show exercises that actually have SOMETHING to plot.
          .filter((a) => a.baseline || a.trend.length > 0 || a.reassessment)
          .sort((a, b) => {
            // Rank: (1) has reassessment first, (2) more trend points,
            // (3) alphabetical fallback.
            const aFull = a.reassessment ? 1 : 0;
            const bFull = b.reassessment ? 1 : 0;
            if (aFull !== bFull) return bFull - aFull;
            if (b.trend.length !== a.trend.length) {
              return b.trend.length - a.trend.length;
            }
            return a.slug.localeCompare(b.slug);
          });
        setState({ status: "ready", artifacts });
      })
      .catch((e) => {
        if (cancelled) return;
        setState({
          status: "error",
          message: e instanceof Error ? e.message : "Could not build proof artifacts.",
        });
      });
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  if (state.status === "loading") {
    return (
      <div className="rounded-card border border-border bg-surface p-8 text-center text-sm text-muted">
        Building improvement graphs…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="rounded-card border border-error/40 bg-error/5 p-5 text-sm text-error">
        {state.message}
      </div>
    );
  }
  if (state.artifacts.length === 0) {
    return (
      <div className="rounded-card border border-dashed border-border bg-surface p-6 text-sm text-muted">
        Complete an assessment (e.g. Trendelenburg, AKE, biomech ROM,
        gait) and a few rehab sessions of a linked exercise
        (pelvic-hold, knee-extension, shoulder-raise, wall-slide,
        pendulum, wall-clock, posture-hold, marching) to see the
        improvement graph populate.
      </div>
    );
  }
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {state.artifacts.map((a) => (
        <ProofChart key={a.slug} artifact={a} />
      ))}
    </div>
  );
}
