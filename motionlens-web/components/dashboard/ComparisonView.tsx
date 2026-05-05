"use client";
// Side-by-side comparison of two reports of the same module.
//
// Top: delta summary ("Peak ROM 31° → 38° (+7°, +22%)").
// Below: two columns of the original polished reports.

import Link from "next/link";
import { ArrowDown, ArrowRight, ArrowUp, GitCompare, Minus } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SavedPostureReport } from "@/components/posture/SavedPostureReport";
import { GaitResultsView } from "@/components/gait/GaitResultsView";
import { resolveMovement } from "@/lib/biomech/movements";
import { formatIST } from "@/lib/format/datetime";
import type { ReportDTO } from "@/lib/reports";
import type { PatientDTO } from "@/lib/patients";
import type { GaitDataDTO } from "@/lib/api";
import type {
  FrontMeasurements,
  PostureFinding,
  SideMeasurements,
} from "@/lib/posture/measurements";

export function ComparisonView({
  patient,
  a,
  b,
  onPickAgain,
}: {
  patient: PatientDTO;
  a: ReportDTO;
  b: ReportDTO;
  onPickAgain: () => void;
}) {
  // Earlier report on the left, later report on the right — reads as
  // "before → after" by default.
  const [left, right] =
    new Date(a.created_at).getTime() <= new Date(b.created_at).getTime()
      ? [a, b]
      : [b, a];

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="eyebrow">Comparing reports</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            {patient.name}
          </h1>
          <p className="mt-2 text-sm text-muted">
            {moduleHeading(left)} · {formatIST(left.created_at)} IST → {formatIST(right.created_at)} IST
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button variant="secondary" onClick={onPickAgain}>
            <GitCompare className="h-4 w-4" />
            Pick different reports
          </Button>
        </div>
      </div>

      {/* Delta summary */}
      <DeltaSummary left={left} right={right} />

      {/* Side-by-side polished reports */}
      <div className="grid gap-8 lg:grid-cols-2">
        <ReportColumn label="Before" tone="muted" report={left} patient={patient} />
        <ReportColumn label="After" tone="accent" report={right} patient={patient} />
      </div>
    </div>
  );
}

// ─── Delta summary ────────────────────────────────────────────────
function DeltaSummary({ left, right }: { left: ReportDTO; right: ReportDTO }) {
  const rows = buildDeltaRows(left, right);
  if (rows.length === 0) return null;

  return (
    <section className="rounded-card border border-border bg-surface p-5">
      <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        Change summary
      </h2>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-3 py-2 font-medium">Metric</th>
              <th className="px-3 py-2 text-right font-medium">Before</th>
              <th className="px-3 py-2 text-center font-medium"> </th>
              <th className="px-3 py-2 text-right font-medium">After</th>
              <th className="px-3 py-2 text-right font-medium">Δ</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <DeltaRow key={i} row={r} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function DeltaRow({ row }: { row: DeltaRow }) {
  const { label, before, after, deltaText, direction, unit } = row;
  const Icon =
    direction === "up" ? ArrowUp : direction === "down" ? ArrowDown : Minus;
  const tone =
    direction === "improved"
      ? "text-emerald-600"
      : direction === "worsened"
        ? "text-error"
        : direction === "up"
          ? "text-accent"
          : direction === "down"
            ? "text-accent"
            : "text-muted";

  return (
    <tr className="border-b border-border/50 last:border-b-0">
      <td className="px-3 py-2.5 text-foreground">{label}</td>
      <td className="px-3 py-2.5 text-right tabular text-foreground">
        {fmtVal(before, unit)}
      </td>
      <td className="px-3 py-2.5 text-center text-subtle">
        <ArrowRight className="mx-auto h-3.5 w-3.5" />
      </td>
      <td className="px-3 py-2.5 text-right tabular text-foreground">
        {fmtVal(after, unit)}
      </td>
      <td className={`px-3 py-2.5 text-right tabular ${tone}`}>
        <span className="inline-flex items-center justify-end gap-1">
          {direction !== "flat" && <Icon className="h-3.5 w-3.5" />}
          {deltaText}
        </span>
      </td>
    </tr>
  );
}

// ─── Report column ────────────────────────────────────────────────
function ReportColumn({
  label,
  tone,
  report,
  patient,
}: {
  label: string;
  tone: "muted" | "accent";
  report: ReportDTO;
  patient: PatientDTO;
}) {
  const ringClass =
    tone === "accent"
      ? "ring-1 ring-accent/30"
      : "ring-1 ring-border/60";

  return (
    <section className={`rounded-card bg-surface/40 p-5 ${ringClass}`}>
      <div className="mb-5 flex items-baseline justify-between gap-3 border-b border-border pb-3">
        <p
          className={`text-xs font-semibold uppercase tracking-[0.12em] ${
            tone === "accent" ? "text-accent" : "text-subtle"
          }`}
        >
          {label}
        </p>
        <Link
          href={`/dashboard/reports/${report.id}`}
          className="text-xs text-accent hover:underline"
        >
          Open full report →
        </Link>
      </div>
      <div className="text-sm">
        <ReportBody report={report} patient={patient} />
      </div>
    </section>
  );
}

function ReportBody({
  report,
  patient,
}: {
  report: ReportDTO;
  patient: PatientDTO;
}) {
  const dateOverride = `${formatIST(report.created_at)} IST`;

  if (report.module === "biomech") {
    const bodyPart = report.body_part as
      | "shoulder" | "neck" | "knee" | "hip" | "ankle" | null;
    if (!bodyPart || !report.movement) {
      return <Notice>Saved report is missing body part / movement.</Notice>;
    }
    const meta = resolveMovement(bodyPart, report.movement);
    const movementName = meta?.label ?? report.movement;
    const m = report.metrics as Record<string, unknown>;
    const measured = pickNumber(m, "peak_magnitude") ?? pickNumber(m, "peak_angle");
    const target = pickRange(m, "reference_range") ?? pickRange(m, "target") ?? meta?.target ?? null;
    if (measured === null || target === null) {
      return <Notice>Missing measured angle or normal range.</Notice>;
    }
    const side = (report.side === "left" || report.side === "right") ? report.side : undefined;
    return (
      <AssessmentReport
        bodyPart={bodyPart}
        movementName={movementName}
        movementId={report.movement}
        measured={measured}
        target={target}
        side={side}
        patientNameOverride={patient.name}
        patientIdOverride={report.patient_id}
        dateOverride={dateOverride}
      />
    );
  }

  if (report.module === "posture") {
    const m = report.metrics as Record<string, unknown>;
    const front = (m.front as FrontMeasurements | null | undefined) ?? null;
    const side = (m.side as SideMeasurements | null | undefined) ?? null;
    const o = report.observations as Record<string, unknown>;
    const ff = (o.front_findings as PostureFinding[] | undefined) ?? null;
    const sf = (o.side_findings as PostureFinding[] | undefined) ?? null;
    return (
      <SavedPostureReport
        front={front}
        side={side}
        frontFindings={ff}
        sideFindings={sf}
      />
    );
  }

  if (report.module === "gait") {
    const m = report.metrics as Record<string, unknown>;
    const required = [
      "metrics_total", "metrics_clean", "video_info", "joint_angles",
      "normalized_overview", "tabs_data",
    ];
    for (const key of required) {
      if (!(key in m)) return <Notice>Saved report missing &quot;{key}&quot;.</Notice>;
    }
    const data: GaitDataDTO = {
      patient_info: { name: patient.name, height_cm: 0 },
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
    return <GaitResultsView data={data} patientNameOverride={patient.name} />;
  }

  return <Notice>Unsupported module: {report.module}</Notice>;
}

function Notice({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
      {children}
    </p>
  );
}

// ─── Delta builders (per module) ──────────────────────────────────
type Direction = "up" | "down" | "flat" | "improved" | "worsened";

interface DeltaRow {
  label: string;
  before: number | null;
  after: number | null;
  unit: string;
  deltaText: string;
  direction: Direction;
}

function buildDeltaRows(left: ReportDTO, right: ReportDTO): DeltaRow[] {
  if (left.module === "biomech") return biomechDeltas(left, right);
  if (left.module === "gait") return gaitDeltas(left, right);
  if (left.module === "posture") return postureDeltas(left, right);
  return [];
}

function biomechDeltas(left: ReportDTO, right: ReportDTO): DeltaRow[] {
  const ml = left.metrics as Record<string, unknown>;
  const mr = right.metrics as Record<string, unknown>;
  const lPeak = pickNumber(ml, "peak_magnitude") ?? pickNumber(ml, "peak_angle");
  const rPeak = pickNumber(mr, "peak_magnitude") ?? pickNumber(mr, "peak_angle");
  const target =
    pickRange(mr, "reference_range") ?? pickRange(mr, "target") ?? null;

  const rows: DeltaRow[] = [];
  rows.push(deltaRow("Peak angle", lPeak, rPeak, "°", "higher_is_better", target));
  return rows;
}

function gaitDeltas(left: ReportDTO, right: ReportDTO): DeltaRow[] {
  const lc = pick(left.metrics, "metrics_clean") as Record<string, unknown> | null;
  const rc = pick(right.metrics, "metrics_clean") as Record<string, unknown> | null;
  if (!lc || !rc) return [];

  const rows: DeltaRow[] = [];
  rows.push(deltaRow("Cadence",     pickNumber(lc, "cadence"),    pickNumber(rc, "cadence"),    " steps/min", "higher_is_better"));
  rows.push(deltaRow("Knee peak",   pickNumber(lc, "knee_peak"),  pickNumber(rc, "knee_peak"),  "°",           "neutral"));
  rows.push(deltaRow("Stride CV",   pickNumber(lc, "stride_cv"),  pickNumber(rc, "stride_cv"),  "%",           "lower_is_better"));
  rows.push(deltaRow("Step length", pickNumber(lc, "step_length"), pickNumber(rc, "step_length"), " m",         "higher_is_better"));
  const lSym = pickNumber(lc, "symmetry");
  const rSym = pickNumber(rc, "symmetry");
  rows.push(deltaRow(
    "Symmetry",
    lSym === null ? null : lSym * 100,
    rSym === null ? null : rSym * 100,
    "%",
    "higher_is_better",
  ));
  rows.push(deltaRow("Torso lean",  pickNumber(lc, "torso_lean"), pickNumber(rc, "torso_lean"), "°",           "lower_abs_is_better"));

  return rows.filter((r) => r.before !== null || r.after !== null);
}

function postureDeltas(left: ReportDTO, right: ReportDTO): DeltaRow[] {
  const ol = left.observations as Record<string, unknown>;
  const or = right.observations as Record<string, unknown>;
  const lf = (ol.front_findings as PostureFinding[] | undefined) ?? [];
  const lsd = (ol.side_findings as PostureFinding[] | undefined) ?? [];
  const rf = (or.front_findings as PostureFinding[] | undefined) ?? [];
  const rsd = (or.side_findings as PostureFinding[] | undefined) ?? [];

  const rows: DeltaRow[] = [];
  const lAll = [...lf, ...lsd];
  const rAll = [...rf, ...rsd];

  // Notable count
  const lNotable = lAll.filter((f) => f.severity === "notable").length;
  const rNotable = rAll.filter((f) => f.severity === "notable").length;
  rows.push(deltaRow("Notable issues", lNotable, rNotable, "", "lower_is_better"));

  const lMild = lAll.filter((f) => f.severity === "mild").length;
  const rMild = rAll.filter((f) => f.severity === "mild").length;
  rows.push(deltaRow("Mild issues", lMild, rMild, "", "lower_is_better"));

  return rows;
}

// ─── Generic delta-row builder ────────────────────────────────────
type Goal =
  | "higher_is_better"
  | "lower_is_better"
  | "lower_abs_is_better"
  | "neutral";

function deltaRow(
  label: string,
  before: number | null,
  after: number | null,
  unit: string,
  goal: Goal,
  target: [number, number] | null = null,
): DeltaRow {
  if (before === null && after === null) {
    return { label, before, after, unit, deltaText: "—", direction: "flat" };
  }
  if (before === null || after === null) {
    return { label, before, after, unit, deltaText: "—", direction: "flat" };
  }

  const diff = after - before;
  const abs = Math.abs(diff);
  const pct = before !== 0 ? (diff / before) * 100 : null;

  let direction: Direction;
  if (Math.abs(diff) < epsilon(unit)) {
    direction = "flat";
  } else if (goal === "higher_is_better") {
    direction = diff > 0 ? "improved" : "worsened";
  } else if (goal === "lower_is_better") {
    direction = diff < 0 ? "improved" : "worsened";
  } else if (goal === "lower_abs_is_better") {
    direction = Math.abs(after) < Math.abs(before) ? "improved" : "worsened";
  } else {
    direction = diff > 0 ? "up" : "down";
  }

  // Target band coloring override: if both before+after are within target,
  // call it flat (already in range — small wiggles aren't meaningful).
  if (target && inRange(before, target) && inRange(after, target)) {
    direction = "flat";
  }

  const sign = diff > 0 ? "+" : diff < 0 ? "−" : "";
  const magnitudeText = `${sign}${abs.toFixed(decimals(unit))}${unit}`;
  const pctText = pct !== null && Math.abs(pct) >= 1
    ? ` (${pct >= 0 ? "+" : "−"}${Math.abs(pct).toFixed(0)}%)`
    : "";
  const deltaText = direction === "flat" ? "—" : `${magnitudeText}${pctText}`;

  return { label, before, after, unit, deltaText, direction };
}

function inRange(v: number, t: [number, number]) {
  return v >= t[0] && v <= t[1];
}

function epsilon(unit: string): number {
  if (unit === "°" || unit === "%" || unit === "" || unit === " steps/min") return 0.5;
  if (unit === " m") return 0.01;
  return 0.5;
}

function decimals(unit: string): number {
  if (unit === " m") return 2;
  if (unit === "%" || unit === "°") return 1;
  if (unit === " steps/min" || unit === "") return 0;
  return 1;
}

function fmtVal(v: number | null, unit: string): string {
  if (v === null) return "—";
  return `${v.toFixed(decimals(unit))}${unit}`;
}

// ─── Helpers ──────────────────────────────────────────────────────
function pick(obj: Record<string, unknown> | null | undefined, key: string): unknown {
  if (!obj) return undefined;
  return obj[key];
}

function pickNumber(obj: Record<string, unknown> | null | undefined, key: string): number | null {
  const v = pick(obj, key);
  return typeof v === "number" && isFinite(v) ? v : null;
}

function pickRange(obj: Record<string, unknown>, key: string): [number, number] | null {
  const v = obj?.[key];
  if (Array.isArray(v) && v.length === 2 && typeof v[0] === "number" && typeof v[1] === "number") {
    return [v[0], v[1]];
  }
  return null;
}

function moduleHeading(r: ReportDTO): string {
  if (r.module === "gait") return "Gait analysis";
  if (r.module === "posture") return "Posture screening";
  const bp = r.body_part ? `${r.body_part.charAt(0).toUpperCase()}${r.body_part.slice(1)}` : "";
  const mv = r.movement ? `${r.movement.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}` : "";
  return [bp, mv].filter(Boolean).join(" · ") || "Biomechanics";
}
