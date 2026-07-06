"use client";
// Dashboard rehab launcher — parallel to /analyze/page.tsx. Shows
// the rehab exercise catalogue with the patient context already
// attached, so any session played from here saves against the
// patient's record.
//
// Exercise definitions live in the shared lib/rehab/exerciseCatalog
// so this page and the public /rehab catalogue stay in sync.

import { useEffect, useMemo, useState, use as usePromise } from "react";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";
import { RehabStreakBadge } from "@/components/rehab/RehabStreakBadge";
import { RehabProgressDashboard } from "@/components/rehab/RehabProgressDashboard";
import { computeStreak, type StreakResult } from "@/lib/rehab/streak";
import { listPatientReports, type ReportSummaryDTO } from "@/lib/reports";
import { groupExercisesByJoint } from "@/lib/rehab/exerciseCatalog";

export default function PatientRehabPage({
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

  const groups = useMemo(() => groupExercisesByJoint(), []);

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Choose rehab game</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            Which mechanic are we playing today?
          </h1>
          <p className="mt-2 max-w-xl text-sm text-muted">
            Pick a game. Session scores save automatically against this
            patient&apos;s record.
          </p>
        </div>
        {reports !== null && <RehabStreakBadge streak={streak} />}
      </div>

      <div className="space-y-12">
        {groups.map(({ joint, meta, items }) => (
          <section key={joint}>
            <div className="flex items-baseline justify-between">
              <h2 className="text-xl font-semibold tracking-tight md:text-2xl">
                {meta.label}
              </h2>
              <p className="text-xs text-muted">{meta.subtitle}</p>
            </div>
            <div className="mt-4 grid gap-5 md:grid-cols-3">
              {items.map((m) => {
                const Icon = m.icon;
                const href = `/rehab/${m.slug}?patientId=${patientId}`;
                const imageUrl = REHAB_EXERCISE_IMAGES[m.slug];
                const eyebrow = `${m.code} · ${m.title}`;
                const sharedClass = `group relative flex flex-col overflow-hidden rounded-hero border border-border bg-gradient-to-br ${m.tone} p-6 transition md:p-8 hover:border-accent hover:shadow-glow-sm`;
                return (
                  <Link key={m.slug} href={href} className={sharedClass}>
                    {imageUrl && (
                      <div className="mb-3 w-full overflow-hidden rounded-md bg-white">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imageUrl}
                          alt=""
                          aria-hidden="true"
                          loading="lazy"
                          className="block h-28 w-full object-contain"
                        />
                      </div>
                    )}
                    <div className="flex items-center justify-between">
                      <Icon className={`h-7 w-7 ${m.iconTone}`} />
                      <ArrowUpRight className="h-5 w-5 text-muted transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-accent" />
                    </div>
                    <div className="mt-8">
                      <p className="eyebrow">{eyebrow}</p>
                      <h3 className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">
                        {m.title}
                      </h3>
                      <p className="mt-3 text-sm leading-relaxed text-muted">
                        {m.patientBody}
                      </p>
                    </div>
                  </Link>
                );
              })}
            </div>
          </section>
        ))}
      </div>

      {/* Progress dashboard — Kemtai/ViFive-style. */}
      <div className="pt-4">
        <RehabProgressDashboard patientId={patientId} />
      </div>
    </div>
  );
}
