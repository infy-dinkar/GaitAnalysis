"use client";
// /dashboard/reports/[id] — view a saved report.
//
// Renders the same UI the doctor saw at capture time:
//   biomech → <AssessmentReport>     (table + Measured-vs-Normal chart + interpretation)
//   posture → <SavedPostureReport>   (front + side findings tables)
//   gait    → <GaitResultsView>      (full metrics + per-joint tabs)

import { useCallback, useEffect, useRef, useState, use as usePromise } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Activity,
  ArmchairIcon,
  Award,
  Calendar,
  Clock,
  FileText,
  Footprints,
  Layers,
  Loader2,
  Move3d,
  MoveUp,
  MoveDiagonal,
  Hourglass,
  ChevronsRight,
  PersonStanding,
  Scale,
  StretchHorizontal,
  TimerIcon,
} from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SavedPostureReport } from "@/components/posture/SavedPostureReport";
import { GaitResultsView } from "@/components/gait/GaitResultsView";
import { SavedTrendelenburgReport } from "@/components/orthopedic/SavedTrendelenburgReport";
import { SavedSingleLegSquatReport } from "@/components/orthopedic/SavedSingleLegSquatReport";
import { SavedSitToStandReport } from "@/components/orthopedic/SavedSitToStandReport";
import { SavedChairStand30sReport } from "@/components/orthopedic/SavedChairStand30sReport";
import { SavedSingleLegStanceReport } from "@/components/orthopedic/SavedSingleLegStanceReport";
import { SavedFourStageBalanceReport } from "@/components/orthopedic/SavedFourStageBalanceReport";
import { SavedTUGReport } from "@/components/orthopedic/SavedTUGReport";
import { SavedSPPBReport } from "@/components/orthopedic/SavedSPPBReport";
import { SavedSLRReport } from "@/components/orthopedic/SavedSLRReport";
import { SavedAKEReport } from "@/components/orthopedic/SavedAKEReport";
import { SavedModifiedThomasReport } from "@/components/orthopedic/SavedModifiedThomasReport";
import { SavedForwardLungeReport } from "@/components/orthopedic/SavedForwardLungeReport";
import { resolveMovement } from "@/lib/biomech/movements";
import { getReport, type ReportDTO } from "@/lib/reports";
import { getPatient, type PatientDTO } from "@/lib/patients";
import { formatIST, formatISTIsoDate } from "@/lib/format/datetime";
import { exportReportPdf } from "@/lib/pdf/exportReportPdf";
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
  trendelenburg: { label: "Trendelenburg test", icon: StretchHorizontal },
  single_leg_squat: { label: "Single-leg squat", icon: Move3d },
  sit_to_stand: { label: "5x Sit-to-Stand", icon: ArmchairIcon },
  chair_stand_30s: { label: "30-Second Chair Stand", icon: TimerIcon },
  single_leg_stance: { label: "Single-Leg Stance", icon: Scale },
  four_stage_balance: { label: "4-Stage Balance Test", icon: Layers },
  tug: { label: "Timed Up and Go (TUG)", icon: Clock },
  sppb: { label: "SPPB (Short Physical Performance Battery)", icon: Award },
  slr: { label: "Straight Leg Raise", icon: MoveUp },
  ake: { label: "Active Knee Extension", icon: MoveDiagonal },
  modified_thomas: { label: "Modified Thomas Test", icon: Hourglass },
  forward_lunge: { label: "Forward Lunge", icon: ChevronsRight },
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
  const sp = useSearchParams();
  const autoDownload = sp.get("download") === "1";

  const [report, setReport] = useState<ReportDTO | null>(null);
  const [patient, setPatient] = useState<PatientDTO | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const reportBodyRef = useRef<HTMLDivElement | null>(null);
  const autoDownloadFiredRef = useRef<boolean>(false);

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

  // All hooks (useCallback + auto-download useEffect) MUST run on every
  // render in the same order; defining them before the early-return
  // branches below keeps the hook order stable. Both guard against the
  // null-report state internally.
  const handleDownload = useCallback(async () => {
    if (!reportBodyRef.current || !report) return;
    setDownloading(true);
    try {
      const meta = MODULE_META[report.module];
      const dateStr = `${formatIST(report.created_at)} IST`;
      const safePatient = (patient?.name ?? "patient").replace(/[^a-z0-9]+/gi, "-").toLowerCase();
      const ymd = formatISTIsoDate(report.created_at);
      const filename = `motionlens-${report.module}-${safePatient}-${ymd}.pdf`;
      const moduleLabel =
        [report.body_part, report.movement, report.side]
          .filter(Boolean)
          .map((s) => s && s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, " "))
          .join(" · ") || meta.label;
      const title = `${meta.label} — ${patient?.name ?? "Patient"} · ${moduleLabel} · ${dateStr}`;
      await exportReportPdf(reportBodyRef.current, { filename, title });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "PDF export failed.";
      alert(`Could not generate PDF: ${msg}`);
    } finally {
      setDownloading(false);
    }
  }, [report, patient]);

  // When the page is reached via a "Download" icon on the patient
  // profile (?download=1), auto-fire the PDF export once the report
  // body has finished mounting + rendering its charts. The 600ms
  // delay gives Plotly a frame to paint into the canvas before
  // html2canvas snapshots it.
  useEffect(() => {
    if (!autoDownload) return;
    if (!report || !patient) return;
    if (autoDownloadFiredRef.current) return;
    autoDownloadFiredRef.current = true;
    const t = window.setTimeout(() => {
      handleDownload();
    }, 600);
    return () => window.clearTimeout(t);
  }, [autoDownload, report, patient, handleDownload]);

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
      {/* Header (buttons hidden in PDF capture via .no-pdf class) */}
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
        {downloading && (
          <div className="no-pdf flex shrink-0 items-center gap-2 rounded-card border border-accent/30 bg-accent/5 px-3 py-2 text-xs text-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />
            Generating PDF…
          </div>
        )}
      </div>

      {/* Body wrapped in a ref so the PDF exporter can capture every
          chart, table, image, and disclaimer the doctor sees on screen. */}
      <div ref={reportBodyRef} className="space-y-8">
        {report.module === "biomech" && (
          <BiomechBody
            report={report}
            patient={patient}
            patientName={patient?.name ?? null}
            patientCode={report.patient_id}
            isoDate={isoDate}
          />
        )}
        {report.module === "posture" && <PostureBody report={report} patient={patient} />}
        {report.module === "gait" && (
          <GaitBody
            report={report}
            patient={patient}
            patientNameOverride={patient?.name ?? null}
          />
        )}
        {report.module === "trendelenburg" && (
          <SavedTrendelenburgReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "single_leg_squat" && (
          <SavedSingleLegSquatReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "sit_to_stand" && (
          <SavedSitToStandReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "chair_stand_30s" && (
          <SavedChairStand30sReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "single_leg_stance" && (
          <SavedSingleLegStanceReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "four_stage_balance" && (
          <SavedFourStageBalanceReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "tug" && (
          <SavedTUGReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "sppb" && (
          <SavedSPPBReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "slr" && (
          <SavedSLRReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "ake" && (
          <SavedAKEReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "modified_thomas" && (
          <SavedModifiedThomasReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
        {report.module === "forward_lunge" && (
          <SavedForwardLungeReport
            patientName={patient?.name ?? null}
            patient={patient}
            metrics={report.metrics as Record<string, unknown>}
            observations={report.observations as Record<string, unknown>}
          />
        )}
      </div>
    </div>
  );
}

// ─── Biomech body ─────────────────────────────────────────────────
function BiomechBody({
  report,
  patient,
  patientName,
  patientCode,
  isoDate,
}: {
  report: ReportDTO;
  patient: PatientDTO | null;
  patientName: string | null;
  patientCode: string | null;
  isoDate: string;
}) {
  // Batch sentinel — see BatchSession.saveAll. A biomech batch save
  // packs N per-item entries into metrics.items with is_batch=true,
  // leaving body_part / movement / side null on the wrapper row.
  const rawMetrics = report.metrics as Record<string, unknown>;
  if (
    rawMetrics.is_batch === true &&
    Array.isArray(rawMetrics.items)
  ) {
    return (
      <BiomechBatchBody
        report={report}
        patient={patient}
        patientName={patientName}
        patientCode={patientCode}
        isoDate={isoDate}
      />
    );
  }

  const bodyPart = report.body_part as
    | "shoulder" | "neck" | "knee" | "hip" | "ankle" | null;
  if (!bodyPart || !report.movement) {
    return <UnsupportedNotice reason="Missing body part or movement on this saved report." />;
  }

  const meta = resolveMovement(bodyPart, report.movement);

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

  // ── Merged-test detection + secondary-direction extraction ────
  // For shoulder "rotation" and "abduction_adduction" the saved
  // payload carries primary_label / secondary_label / secondary_*
  // fields so the report can render two rows + two chart bars + two
  // interpretation sentences (same UX the live "Show Analysis" view
  // gives). The legacy "Abduction + Adduction" single-row label
  // appeared because the saved viewer used to ignore those fields.
  //
  // We fall back to the movement metadata for older saved records
  // that pre-date merged-test support — so even those render with
  // the primary direction's label rather than the combined chooser
  // label.
  const primaryLabelSaved = pickString(m, "primary_label");
  const secondaryLabelSaved = pickString(m, "secondary_label");
  const secondaryTargetSaved =
    pickRange(m, "secondary_reference_range") ?? pickRange(m, "secondary_target");
  const secondaryMeasuredSaved = pickNumber(m, "secondary_peak_magnitude");

  const isMerged =
    !!primaryLabelSaved ||
    !!secondaryLabelSaved ||
    !!meta?.merged;

  const movementName = isMerged
    ? (primaryLabelSaved ?? meta?.primaryLabel ?? meta?.label ?? report.movement)
    : (meta?.label ?? report.movement);
  const secondaryMovementName = isMerged
    ? (secondaryLabelSaved ?? meta?.secondaryLabel ?? undefined)
    : undefined;
  const secondaryTarget = isMerged
    ? (secondaryTargetSaved ?? meta?.secondaryTarget ?? undefined)
    : undefined;
  // 0 is a legitimate measurement (knee extension at full straight),
  // so the captured-vs-missing distinction is type-based not value-
  // based. pickNumber returns null for missing fields, a number for
  // present (including 0).
  const secondaryMeasured =
    isMerged && typeof secondaryMeasuredSaved === "number"
      ? secondaryMeasuredSaved
      : undefined;

  // Recover key-frame screenshots from saved metrics if present.
  // Backend ankle analyses persist these so the saved-report viewer
  // can show the same "Key frames" strip the original capture did.
  const savedKeyFrames = (() => {
    const raw = (m as Record<string, unknown>).key_frames;
    if (!Array.isArray(raw)) return undefined;
    const items: Array<{ label: string; frame_index: number; image_data_url: string }> = [];
    for (const entry of raw) {
      if (!entry || typeof entry !== "object") continue;
      const e = entry as Record<string, unknown>;
      const label = typeof e.label === "string" ? e.label : null;
      const frameIdx =
        typeof e.frame_index === "number"
          ? e.frame_index
          : typeof e.frame_index === "string"
            ? parseInt(e.frame_index, 10)
            : null;
      const url = typeof e.image_data_url === "string" ? e.image_data_url : null;
      if (label && frameIdx !== null && !Number.isNaN(frameIdx) && url) {
        items.push({ label, frame_index: frameIdx, image_data_url: url });
      }
    }
    return items.length > 0 ? items : undefined;
  })();

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
        patientOverride={patient}
        keyFrames={savedKeyFrames}
        secondaryMovementName={secondaryMovementName}
        secondaryMeasured={secondaryMeasured}
        secondaryTarget={secondaryTarget}
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

// ─── Biomech batch body ──────────────────────────────────────────
// Renders a saved biomech batch report (one DB row, N per-item
// AssessmentReports stacked). Mirrors BatchSession's report-phase
// layout so the saved view and the live view look identical.
function BiomechBatchBody({
  report,
  patient,
  patientName,
  patientCode,
  isoDate,
}: {
  report: ReportDTO;
  patient: PatientDTO | null;
  patientName: string | null;
  patientCode: string | null;
  isoDate: string;
}) {
  const m = report.metrics as Record<string, unknown>;
  const rawItems = Array.isArray(m.items) ? (m.items as unknown[]) : [];
  // Normalise each entry — discard malformed rows rather than crashing
  // the whole report. Each rendered item is a self-contained
  // AssessmentReport with the same merged-test handling the
  // single-item BiomechBody applies.
  const items: Array<{
    bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle";
    movement: string;
    side: "left" | "right" | undefined;
    measured: number;
    target: [number, number];
    movementName: string;
    secondaryMovementName?: string;
    secondaryMeasured?: number;
    secondaryTarget?: [number, number];
    keyFrames?: Array<{ label: string; frame_index: number; image_data_url: string }>;
  }> = [];

  for (const raw of rawItems) {
    if (!raw || typeof raw !== "object") continue;
    const e = raw as Record<string, unknown>;
    const bp = e.body_part;
    if (bp !== "shoulder" && bp !== "neck" && bp !== "knee" && bp !== "hip" && bp !== "ankle") {
      continue;
    }
    const movement = typeof e.movement === "string" ? e.movement : null;
    if (!movement) continue;
    const measured = pickNumber(e, "peak_magnitude") ?? pickNumber(e, "peak_angle");
    const target =
      pickRange(e, "reference_range") ??
      pickRange(e, "target");
    if (measured === null || !target) continue;
    const side = e.side === "left" || e.side === "right" ? e.side : undefined;

    const meta = resolveMovement(bp, movement);
    const primaryLabelSaved = pickString(e, "primary_label");
    const secondaryLabelSaved = pickString(e, "secondary_label");
    const secondaryTargetSaved =
      pickRange(e, "secondary_reference_range") ?? pickRange(e, "secondary_target");
    const secondaryMeasuredSaved = pickNumber(e, "secondary_peak_magnitude");
    const isMerged =
      !!primaryLabelSaved || !!secondaryLabelSaved || !!meta?.merged;

    const movementName = isMerged
      ? (primaryLabelSaved ?? meta?.primaryLabel ?? meta?.label ?? movement)
      : (meta?.label ?? movement);
    const secondaryMovementName = isMerged
      ? (secondaryLabelSaved ?? meta?.secondaryLabel ?? undefined)
      : undefined;
    const secondaryTarget = isMerged
      ? (secondaryTargetSaved ?? meta?.secondaryTarget ?? undefined)
      : undefined;
    const secondaryMeasured =
      isMerged && typeof secondaryMeasuredSaved === "number"
        ? secondaryMeasuredSaved
        : undefined;

    // Optional annotated screenshots — same shape as the single-item path.
    const rawKF = e.key_frames;
    const keyFrames = Array.isArray(rawKF)
      ? rawKF
          .map((kf) => {
            if (!kf || typeof kf !== "object") return null;
            const k = kf as Record<string, unknown>;
            const label = typeof k.label === "string" ? k.label : null;
            const frameIdx =
              typeof k.frame_index === "number"
                ? k.frame_index
                : typeof k.frame_index === "string"
                  ? parseInt(k.frame_index, 10)
                  : null;
            const url = typeof k.image_data_url === "string" ? k.image_data_url : null;
            return label && frameIdx !== null && !Number.isNaN(frameIdx) && url
              ? { label, frame_index: frameIdx, image_data_url: url }
              : null;
          })
          .filter((x): x is NonNullable<typeof x> => x !== null)
      : undefined;

    items.push({
      bodyPart: bp,
      movement,
      side,
      measured,
      target,
      movementName,
      secondaryMovementName,
      secondaryMeasured,
      secondaryTarget,
      keyFrames: keyFrames && keyFrames.length > 0 ? keyFrames : undefined,
    });
  }

  if (items.length === 0) {
    return (
      <UnsupportedNotice reason="This batch report has no readable item entries." />
    );
  }

  return (
    <div className="space-y-12">
      <div className="rounded-card border border-border bg-surface px-5 py-4 text-sm">
        <p className="font-semibold text-foreground">
          {patientName?.trim() || "Anonymous patient"}
        </p>
        <p className="mt-0.5 text-xs text-muted">
          Combined biomechanics report · {items.length} movement
          {items.length === 1 ? "" : "s"} captured · ID: {patientCode ?? "—"} ·{" "}
          {isoDate}
        </p>
      </div>

      {items.map((it, i) => (
        <div
          key={`${it.bodyPart}-${it.movement}-${it.side ?? "_"}-${i}`}
          className="border-t border-border pt-8 first:border-t-0 first:pt-0"
        >
          <AssessmentReport
            bodyPart={it.bodyPart}
            movementName={it.movementName}
            movementId={it.movement}
            measured={it.measured}
            target={it.target}
            side={it.side}
            patientNameOverride={patientName}
            patientIdOverride={patientCode}
            dateOverride={isoDate}
            patientOverride={patient}
            keyFrames={it.keyFrames}
            secondaryMovementName={it.secondaryMovementName}
            secondaryMeasured={it.secondaryMeasured}
            secondaryTarget={it.secondaryTarget}
          />
        </div>
      ))}

      <BackLink patientId={report.patient_id} />
    </div>
  );
}

// ─── Posture body ─────────────────────────────────────────────────
function PostureBody({
  report,
  patient,
}: {
  report: ReportDTO;
  patient: PatientDTO | null;
}) {
  const m = report.metrics as Record<string, unknown>;
  const front = (m.front as FrontMeasurements | null | undefined) ?? null;
  const side = (m.side as SideMeasurements | null | undefined) ?? null;

  // Source photos persisted by PostureCapture. May be null for reports
  // saved BEFORE the image-persistence schema landed — SavedPostureReport
  // handles that case by falling back to findings-only rendering.
  const frontImage =
    (m.front_image as { data_url: string; width: number; height: number } | null | undefined)
      ?? null;
  const sideImage =
    (m.side_image as { data_url: string; width: number; height: number } | null | undefined)
      ?? null;

  // Keypoint arrays live on report.keypoints, separated by view.
  const kpRoot = (report.keypoints ?? {}) as Record<string, unknown>;
  const frontKeypoints =
    (kpRoot.front as Parameters<typeof SavedPostureReport>[0]["frontKeypoints"]) ?? null;
  const sideKeypoints =
    (kpRoot.side as Parameters<typeof SavedPostureReport>[0]["sideKeypoints"]) ?? null;

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
        frontImage={frontImage}
        sideImage={sideImage}
        frontKeypoints={frontKeypoints}
        sideKeypoints={sideKeypoints}
        patient={patient}
        patientName={patient?.name ?? null}
      />
      <BackLink patientId={report.patient_id} />
    </div>
  );
}

// ─── Gait body ────────────────────────────────────────────────────
function GaitBody({
  report,
  patient,
  patientNameOverride,
}: {
  report: ReportDTO;
  patient: PatientDTO | null;
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
      <GaitResultsView data={data} patientNameOverride={patientNameOverride} patientOverride={patient} />
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
