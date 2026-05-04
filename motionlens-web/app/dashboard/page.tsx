"use client";
// /dashboard — doctor's home. Shows greeting + recent patients + quick stats.

import { useEffect, useState } from "react";
import Link from "next/link";
import { UserPlus, Users, FileText, Activity } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { PatientCard } from "@/components/dashboard/PatientCard";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/contexts/AuthContext";
import { listPatients, type PatientDTO } from "@/lib/patients";

export default function DashboardPage() {
  return (
    <AuthGuard>
      <DashboardShell>
        <DashboardContent />
      </DashboardShell>
    </AuthGuard>
  );
}

function DashboardContent() {
  const { doctor } = useAuth();
  const [patients, setPatients] = useState<PatientDTO[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listPatients()
      .then((r) => !cancelled && setPatients(r.data))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  const totalReports = patients?.reduce((sum, p) => sum + p.report_count, 0) ?? 0;
  const recent = patients?.slice(0, 6) ?? [];

  return (
    <div className="space-y-10">
      {/* Greeting */}
      <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-center">
        <div>
          <p className="eyebrow">{greeting()},</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            {doctor?.name ?? "Doctor"}
          </h1>
          <p className="mt-2 text-sm text-muted">
            Manage patients, run movement assessments, and review report history.
          </p>
        </div>
        <Link href="/dashboard/patients/new">
          <Button>
            <UserPlus className="h-4 w-4" />
            Add patient
          </Button>
        </Link>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <StatCard
          icon={Users}
          label="Total patients"
          value={patients?.length ?? "…"}
          tone="text-cyan-600"
        />
        <StatCard
          icon={FileText}
          label="Reports generated"
          value={patients ? totalReports : "…"}
          tone="text-emerald-600"
        />
        <StatCard
          icon={Activity}
          label="Active assessments"
          value={
            patients
              ? patients.filter((p) => p.report_count > 0).length
              : "…"
          }
          tone="text-amber-600"
        />
      </div>

      {/* Recent patients */}
      <section>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold tracking-tight">Recent patients</h2>
          {(patients?.length ?? 0) > 0 && (
            <Link
              href="/dashboard/patients"
              className="text-sm text-accent hover:underline"
            >
              View all →
            </Link>
          )}
        </div>

        {error && (
          <div className="mt-4 rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
            {error}
          </div>
        )}

        {patients === null ? (
          <SkeletonGrid />
        ) : recent.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {recent.map((p) => (
              <PatientCard key={p.id} patient={p} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

function StatCard({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: typeof Users;
  label: string;
  value: number | string;
  tone: string;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          {label}
        </p>
        <Icon className={`h-4 w-4 ${tone}`} />
      </div>
      <p className="mt-2 text-3xl font-semibold tabular text-foreground">{value}</p>
    </div>
  );
}

function SkeletonGrid() {
  return (
    <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="h-28 animate-pulse rounded-card border border-border bg-surface"
        />
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mt-4 rounded-card border border-dashed border-border bg-surface/50 px-6 py-10 text-center">
      <Users className="mx-auto h-8 w-8 text-muted" />
      <h3 className="mt-3 text-base font-semibold text-foreground">No patients yet</h3>
      <p className="mt-1 text-sm text-muted">
        Add your first patient to start running assessments.
      </p>
      <Link href="/dashboard/patients/new" className="mt-4 inline-block">
        <Button>
          <UserPlus className="h-4 w-4" />
          Add first patient
        </Button>
      </Link>
    </div>
  );
}
