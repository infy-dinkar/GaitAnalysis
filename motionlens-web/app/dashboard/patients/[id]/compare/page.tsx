"use client";
// /dashboard/patients/[id]/compare — pick two reports, render side-by-side.
//
// URL: ?a=<reportId>&b=<reportId>
// Without both query params, the page shows a selector. With both,
// it renders the comparison view (delta summary + two reports side-by-side).
//
// Comparison rules:
//   gait    → any gait report can be compared to any other gait report
//   biomech → must share body_part + movement (else apples-to-oranges)
//   posture → any posture-front+side pair can be compared

import { useEffect, useMemo, useState, use as usePromise } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, GitCompare, Loader2 } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Button } from "@/components/ui/Button";
import {
  getReport,
  listPatientReports,
  type ReportDTO,
  type ReportSummaryDTO,
} from "@/lib/reports";
import { getPatient, type PatientDTO } from "@/lib/patients";
import { formatIST, formatISTDate } from "@/lib/format/datetime";
import { ComparisonView } from "@/components/dashboard/ComparisonView";

export default function ComparePage({
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
        <CompareInner patientId={id} />
      </DashboardShell>
    </AuthGuard>
  );
}

function CompareInner({ patientId }: { patientId: string }) {
  const router = useRouter();
  const sp = useSearchParams();
  const aId = sp.get("a");
  const bId = sp.get("b");

  const [patient, setPatient] = useState<PatientDTO | null>(null);
  const [reports, setReports] = useState<ReportSummaryDTO[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getPatient(patientId), listPatientReports(patientId)])
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
  }, [patientId]);

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

  // Render the comparison if both report ids are pinned in the URL.
  if (aId && bId) {
    return (
      <ComparisonContainer
        patient={patient}
        aId={aId}
        bId={bId}
        onPickAgain={() => router.push(`/dashboard/patients/${patientId}/compare`)}
      />
    );
  }

  return (
    <Selector
      patientId={patientId}
      patientName={patient.name}
      reports={reports}
    />
  );
}

// ─── Selector ─────────────────────────────────────────────────────
function Selector({
  patientId,
  patientName,
  reports,
}: {
  patientId: string;
  patientName: string;
  reports: ReportSummaryDTO[];
}) {
  const router = useRouter();
  const [a, setA] = useState<ReportSummaryDTO | null>(null);
  const [b, setB] = useState<ReportSummaryDTO | null>(null);

  // Group reports by comparison key, so the doctor only sees compatible
  // pairs in the second column once they've picked A.
  const compatibleWithA = useMemo(() => {
    if (!a) return [] as ReportSummaryDTO[];
    return reports.filter((r) => r.id !== a.id && comparisonKey(r) === comparisonKey(a));
  }, [a, reports]);

  function go() {
    if (!a || !b) return;
    router.push(`/dashboard/patients/${patientId}/compare?a=${a.id}&b=${b.id}`);
  }

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Compare reports</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
          {patientName}
        </h1>
        <p className="mt-2 text-sm text-muted">
          Pick two reports of the same type — gait↔gait, posture↔posture, or
          biomech reports of the same body part and movement — to see them
          side-by-side with a delta summary.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Column
          title="Report A"
          reports={reports}
          selected={a}
          onSelect={(r) => {
            setA(r);
            // If the chosen B is no longer compatible, clear it.
            if (b && comparisonKey(b) !== comparisonKey(r)) setB(null);
          }}
        />
        <Column
          title="Report B"
          reports={a ? compatibleWithA : reports}
          selected={b}
          onSelect={setB}
          disabled={!a}
          emptyHint={a ? "No other reports of the same type yet." : "Pick Report A first."}
        />
      </div>

      <div className="flex justify-end gap-2 border-t border-border pt-6">
        <Link href={`/dashboard/patients/${patientId}`}>
          <Button variant="ghost">
            <ArrowLeft className="h-4 w-4" />
            Cancel
          </Button>
        </Link>
        <Button onClick={go} disabled={!a || !b}>
          <GitCompare className="h-4 w-4" />
          Compare
        </Button>
      </div>
    </div>
  );
}

function Column({
  title,
  reports,
  selected,
  onSelect,
  disabled = false,
  emptyHint,
}: {
  title: string;
  reports: ReportSummaryDTO[];
  selected: ReportSummaryDTO | null;
  onSelect: (r: ReportSummaryDTO) => void;
  disabled?: boolean;
  emptyHint?: string;
}) {
  return (
    <section className={disabled ? "opacity-50" : ""}>
      <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {title}
      </h2>
      <div className="mt-3 space-y-2">
        {reports.length === 0 ? (
          <p className="rounded-card border border-dashed border-border bg-surface/50 px-4 py-6 text-center text-sm text-muted">
            {emptyHint ?? "No reports available."}
          </p>
        ) : (
          reports.map((r) => {
            const active = selected?.id === r.id;
            return (
              <button
                key={r.id}
                type="button"
                disabled={disabled}
                onClick={() => onSelect(r)}
                className={`block w-full rounded-card border p-3 text-left transition ${
                  active
                    ? "border-accent bg-accent/5"
                    : "border-border bg-surface hover:border-accent/60"
                }`}
              >
                <div className="flex items-baseline justify-between gap-3">
                  <p className="text-sm font-medium text-foreground">
                    {moduleLabel(r)}
                  </p>
                  <p className="shrink-0 text-xs tabular text-subtle">
                    {formatISTDate(r.created_at)}
                  </p>
                </div>
                {subline(r) && (
                  <p className="mt-0.5 truncate text-xs text-muted">{subline(r)}</p>
                )}
                <p className="mt-0.5 text-[11px] tabular text-subtle">
                  {formatIST(r.created_at)} IST
                </p>
              </button>
            );
          })
        )}
      </div>
    </section>
  );
}

// ─── Comparison container ─────────────────────────────────────────
function ComparisonContainer({
  patient,
  aId,
  bId,
  onPickAgain,
}: {
  patient: PatientDTO;
  aId: string;
  bId: string;
  onPickAgain: () => void;
}) {
  const [a, setA] = useState<ReportDTO | null>(null);
  const [b, setB] = useState<ReportDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getReport(aId), getReport(bId)])
      .then(([rA, rB]) => {
        if (cancelled) return;
        setA(rA);
        setB(rB);
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [aId, bId]);

  if (error) {
    return (
      <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
        {error}
      </div>
    );
  }

  if (!a || !b) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-accent" />
      </div>
    );
  }

  if (comparisonKey(a) !== comparisonKey(b)) {
    return (
      <div className="space-y-4">
        <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
          These two reports aren&apos;t comparable — they need to be the same
          module (and for biomech, same body part + movement).
        </div>
        <Button variant="secondary" onClick={onPickAgain}>
          Pick again
        </Button>
      </div>
    );
  }

  return (
    <ComparisonView
      patient={patient}
      a={a}
      b={b}
      onPickAgain={onPickAgain}
    />
  );
}

// ─── Helpers ──────────────────────────────────────────────────────
function comparisonKey(r: { module: string; body_part?: string | null; movement?: string | null }) {
  if (r.module === "biomech") {
    return `biomech::${r.body_part ?? ""}::${r.movement ?? ""}`;
  }
  return r.module;
}

function moduleLabel(r: ReportSummaryDTO): string {
  if (r.module === "gait") return "Gait analysis";
  if (r.module === "posture") return "Posture screening";
  if (r.module === "trendelenburg") return "Trendelenburg test";
  if (r.module === "single_leg_squat") return "Single-leg squat";
  if (r.module === "sit_to_stand") return "5x Sit-to-Stand";
  if (r.module === "chair_stand_30s") return "30-Second Chair Stand";
  return "Biomechanics";
}

function subline(r: ReportSummaryDTO): string {
  return [
    r.body_part && cap(r.body_part),
    r.movement && humanize(r.movement),
    r.side && cap(r.side),
  ]
    .filter(Boolean)
    .join(" · ");
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
