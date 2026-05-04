"use client";
// /dashboard/patients/[id] — patient profile + reports list

import { useEffect, useState, use as usePromise } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  ArrowRight,
  Calendar,
  FileText,
  Loader2,
  Phone,
  Ruler,
  Scale,
  Trash2,
  User,
} from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { ReportCard } from "@/components/dashboard/ReportCard";
import { Button } from "@/components/ui/Button";
import {
  deletePatient,
  getPatient,
  type PatientDTO,
} from "@/lib/patients";
import { listPatientReports, type ReportSummaryDTO } from "@/lib/reports";

export default function PatientDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = usePromise(params);
  return (
    <AuthGuard>
      <DashboardShell backHref="/dashboard/patients" backLabel="Patients">
        <PatientDetail id={id} />
      </DashboardShell>
    </AuthGuard>
  );
}

function PatientDetail({ id }: { id: string }) {
  const router = useRouter();
  const [patient, setPatient] = useState<PatientDTO | null>(null);
  const [reports, setReports] = useState<ReportSummaryDTO[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getPatient(id), listPatientReports(id)])
      .then(([p, r]) => {
        if (!cancelled) {
          setPatient(p);
          setReports(r.data);
        }
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [id]);

  async function handleDelete() {
    if (!confirm("Delete this patient and all their reports? This cannot be undone.")) {
      return;
    }
    setDeleting(true);
    try {
      await deletePatient(id);
      router.replace("/dashboard/patients");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
      setDeleting(false);
    }
  }

  if (error) {
    return (
      <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
        {error}
      </div>
    );
  }

  if (patient === null || reports === null) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-accent" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0 flex-1">
          <p className="eyebrow">Patient profile</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            {patient.name}
          </h1>
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-sm text-muted">
            <Stat icon={User} value={`${patient.gender}`} />
            <span>·</span>
            <Stat icon={Calendar} value={`${patient.age} yrs`} />
            <span>·</span>
            <Stat icon={Ruler} value={`${patient.height_cm} cm`} />
            {patient.weight_kg && (
              <>
                <span>·</span>
                <Stat icon={Scale} value={`${patient.weight_kg} kg`} />
              </>
            )}
            {patient.contact && (
              <>
                <span>·</span>
                <Stat icon={Phone} value={patient.contact} />
              </>
            )}
          </div>
        </div>

        <div className="flex shrink-0 gap-2">
          <Link href={`/dashboard/patients/${id}/analyze`}>
            <Button>
              <Activity className="h-4 w-4" />
              New assessment
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
          <Button
            variant="secondary"
            onClick={handleDelete}
            disabled={deleting}
            className="text-error hover:bg-error/10"
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </div>
      </div>

      {/* Notes */}
      {patient.medical_notes && (
        <section>
          <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Medical notes
          </h2>
          <p className="mt-2 rounded-card border border-border bg-surface p-4 text-sm leading-relaxed text-foreground">
            {patient.medical_notes}
          </p>
        </section>
      )}

      {/* Reports */}
      <section>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold tracking-tight">
            Reports{" "}
            <span className="text-sm font-normal text-muted">({reports.length})</span>
          </h2>
        </div>

        {reports.length === 0 ? (
          <div className="mt-4 rounded-card border border-dashed border-border bg-surface/50 px-6 py-10 text-center">
            <FileText className="mx-auto h-8 w-8 text-muted" />
            <h3 className="mt-3 text-base font-semibold text-foreground">
              No reports yet
            </h3>
            <p className="mt-1 text-sm text-muted">
              Run an assessment to generate the first report for this patient.
            </p>
            <Link href={`/dashboard/patients/${id}/analyze`} className="mt-4 inline-block">
              <Button>
                <Activity className="h-4 w-4" />
                Start assessment
              </Button>
            </Link>
          </div>
        ) : (
          <div className="mt-4 space-y-3">
            {reports.map((r) => (
              <ReportCard key={r.id} report={r} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({ icon: Icon, value }: { icon: typeof User; value: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <Icon className="h-3.5 w-3.5" />
      <span className="capitalize">{value}</span>
    </span>
  );
}
