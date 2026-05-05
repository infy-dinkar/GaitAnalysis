"use client";
// /dashboard/reports/[id] — view a saved report.
//
// Renders the same UI the doctor saw at capture time:
//   biomech → <AssessmentReport>     (table + Measured-vs-Normal chart + interpretation)
//   posture → <SavedPostureReport>   (front + side findings tables)
//   gait    → <GaitResultsView>      (full metrics + per-joint tabs)

import { useEffect, useState, use as usePromise } from "react";
import Link from "next/link";
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
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SavedPostureReport } from "@/components/posture/SavedPostureReport";
import { GaitResultsView } from "@/components/gait/GaitResultsView";
import { resolveMovement } from "@/lib/biomech/movements";
import { getReport, type ReportDTO } from "@/lib/reports";
import { getPatient, type PatientDTO } from "@/lib/patients";
import { formatIST, formatISTIsoDate } from "@/lib/format/datetime";
import type { GaitDataDTO } from "@/lib/api";
import type {
  FrontMeasurements,
  SideMeasurements,
  PostureFinding,
} from "@/lib/posture/measurements";

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
  const [patient, setPatient] = useState<PatientDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getReport(id)
      .then(async (r) => {
        if (cancelled) return;
        setReport(r);
        try {
          const p = await getPatient(r.patient_id);
          if (!cancelled) setPatient(p);
        } catch {
          // patient fetch failure is non-fatal — the report still renders
        }
      })
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
  const dateStr = `${formatIST(report.created_at)} IST`;
  const isoDate = `${formatIST(report.created_at)} IST`;

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
              {dateStr}
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

      {/* Module-specific body */}
      {report.module === "biomech" && (
        <BiomechBody
          report={report}
          patientName={patient?.name ?? null}
          patientCode={report.patient_id}
          isoDate={isoDate}
        />
      )}
      {report.module === "posture" && <PostureBody report={report} />}
      {report.module === "gait" && (
        <GaitBody
          report={report}
          patientNameOverride={patient?.name ?? null}
        />
      )}
    </div>
  );
}

// ─── Biomech body ─────────────────────────────────────────────────
function BiomechBody({
  report,
  patientName,
  patientCode,
  isoDate,
}: {
  report: ReportDTO;
  patientName: string | null;
  patientCode: string | null;
  isoDate: string;
}) {
  const bodyPart = report.body_part as
    | "shoulder" | "neck" | "knee" | "hip" | "ankle" | null;
  if (!bodyPart || !report.movement) {
    return <UnsupportedNotice reason="Missing body part or movement on this saved report." />;
  }

  const meta = resolveMovement(bodyPart, report.movement);
  const movementName = meta?.label ?? report.movement;

  // The saved metrics shape varies between live capture and upload analysis.
  const m = report.metrics as Record<string, unknown>;
  const measured = pickNumber(m, "peak_magnitude") ?? pickNumber(m, "peak_angle");

  const target = pickRange(m, "reference_range")
    ?? pickRange(m, "target")
    ?? meta?.target
    ?? null;

  if (measured === null || target === null) {
    return <UnsupportedNotice reason="This saved report is missing the measured angle or normal range." />;
  }

  const side = (report.side === "left" || report.side === "right") ? report.side : undefined;
  const interpretation = pickString(report.observations, "interpretation");

  return (
    <div className="space-y-8">
      <AssessmentReport
        bodyPart={bodyPart}
        movementName={movementName}
        movementId={report.movement}
        measured={measured}
        target={target}
        side={side}
        patientNameOverride={patientName}
        patientIdOverride={patientCode}
        dateOverride={isoDate}
      />
      {interpretation && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Saved interpretation</h3>
          <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
            {interpretation}
          </div>
        </section>
      )}
      <BackLink patientId={report.patient_id} />
    </div>
  );
}

// ─── Posture body ─────────────────────────────────────────────────
function PostureBody({ report }: { report: ReportDTO }) {
  const m = report.metrics as Record<string, unknown>;
  const front = (m.front as FrontMeasurements | null | undefined) ?? null;
  const side = (m.side as SideMeasurements | null | undefined) ?? null;

  const o = report.observations as Record<string, unknown>;
  const frontFindings = (o.front_findings as PostureFinding[] | undefined) ?? null;
  const sideFindings = (o.side_findings as PostureFinding[] | undefined) ?? null;

  return (
    <div className="space-y-8">
      <SavedPostureReport
        front={front}
        side={side}
        frontFindings={frontFindings}
        sideFindings={sideFindings}
      />
      <BackLink patientId={report.patient_id} />
    </div>
  );
}

// ─── Gait body ────────────────────────────────────────────────────
function GaitBody({
  report,
  patientNameOverride,
}: {
  report: ReportDTO;
  patientNameOverride: string | null;
}) {
  const m = report.metrics as Record<string, unknown>;

  // Required pieces — if any are missing the saved report can't be reconstructed.
  const required = [
    "metrics_total",
    "metrics_clean",
    "video_info",
    "joint_angles",
    "normalized_overview",
    "tabs_data",
  ];
  for (const key of required) {
    if (!(key in m)) {
      return <UnsupportedNotice reason={`This saved report is missing the "${key}" field.`} />;
    }
  }

  // Reconstruct the GaitDataDTO shape from saved metrics + patient_info pulled from the report.
  const data: GaitDataDTO = {
    patient_info: {
      name: patientNameOverride,
      // height_cm is needed for CalibrationHeader; saved video_info also carries calibration.
      height_cm: 0,
    },
    video_info: m.video_info as GaitDataDTO["video_info"],
    walking_direction: (m.walking_direction as string) ?? "—",
    metrics_total: m.metrics_total as GaitDataDTO["metrics_total"],
    metrics_clean: m.metrics_clean as GaitDataDTO["metrics_clean"],
    joint_angles: m.joint_angles as GaitDataDTO["joint_angles"],
    gait_cycle_data: (m.gait_cycle_data as GaitDataDTO["gait_cycle_data"]) ?? null,
    normalized_overview: m.normalized_overview as GaitDataDTO["normalized_overview"],
    tabs_data: m.tabs_data as GaitDataDTO["tabs_data"],
    observations: (report.observations as unknown as GaitDataDTO["observations"]) ?? {
      hip: [], knee: [], ankle: [], overall: [], suggestions: [],
    },
  };

  return (
    <div className="space-y-8">
      <GaitResultsView data={data} patientNameOverride={patientNameOverride} />
      <BackLink patientId={report.patient_id} />
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────
function pickNumber(obj: Record<string, unknown>, key: string): number | null {
  const v = obj?.[key];
  return typeof v === "number" && isFinite(v) ? v : null;
}

function pickString(obj: Record<string, unknown>, key: string): string | null {
  const v = obj?.[key];
  return typeof v === "string" && v.length > 0 ? v : null;
}

function pickRange(obj: Record<string, unknown>, key: string): [number, number] | null {
  const v = obj?.[key];
  if (Array.isArray(v) && v.length === 2 && typeof v[0] === "number" && typeof v[1] === "number") {
    return [v[0], v[1]];
  }
  return null;
}

function UnsupportedNotice({ reason }: { reason: string }) {
  return (
    <div className="rounded-card border border-warning/40 bg-warning/5 p-5 text-sm text-foreground">
      <p className="font-medium">This saved report can&apos;t be rendered as a polished view.</p>
      <p className="mt-1 text-muted">{reason}</p>
    </div>
  );
}

function BackLink({ patientId }: { patientId: string }) {
  return (
    <div className="border-t border-border pt-6">
      <Link
        href={`/dashboard/patients/${patientId}`}
        className="text-sm text-accent hover:underline"
      >
        ← Back to patient
      </Link>
    </div>
  );
}
