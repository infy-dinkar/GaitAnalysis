"use client";
// Dedicated rehab-progress page. Shows ONLY the streak + progress
// dashboard (adherence, charts, top exercises, recent sessions with
// delete) — no catalogue. Distinct from
// /dashboard/patients/[id]/rehab which is the "start a new session"
// launcher.

import { useEffect, useMemo, useState, use as usePromise } from "react";
import Link from "next/link";
import { ArrowLeft, Dumbbell } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Button } from "@/components/ui/Button";
import { RehabStreakBadge } from "@/components/rehab/RehabStreakBadge";
import { RehabProgressDashboard } from "@/components/rehab/RehabProgressDashboard";
import { computeStreak, type StreakResult } from "@/lib/rehab/streak";
import { listPatientReports, type ReportSummaryDTO } from "@/lib/reports";

export default function PatientRehabProgressPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = usePromise(params);
  return (
    <AuthGuard>
      <DashboardShell
        backHref={`/dashboard/patients/${id}`}
        backLabel="Patient"
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
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Rehab progress</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            Streak, adherence & session history
          </h1>
          <p className="mt-2 max-w-xl text-sm text-muted">
            Everything the patient has done so far. Jump into a new
            session or wipe a stray one from the record.
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

      <RehabProgressDashboard patientId={patientId} />
    </div>
  );
}
