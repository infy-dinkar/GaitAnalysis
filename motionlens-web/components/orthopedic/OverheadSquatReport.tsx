"use client";
// B2 Overhead Squat — result report.
//
// Sections:
//   • Classification pill (good / moderate / poor) + measurable-
//     fails badge and rep-count.
//   • Summary cards — max depth, worst valgus, max pelvic tilt.
//   • 7-item checklist table.
//   • Worst-rep screenshot (bottom frame).
//   • Clinical interpretation + honest frontal-view caveat.

import { CheckCircle2, MinusCircle, XCircle } from "lucide-react";

import {
  type OverheadSquatChecklistItem,
  type OverheadSquatChecklistStatus,
  type OverheadSquatClassification,
  type OverheadSquatResult,
} from "@/lib/orthopedic/overheadSquat";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  result: OverheadSquatResult;
  interpretation: string;
}

export function OverheadSquatReport({
  patientName,
  patient,
  result,
  interpretation,
}: Props) {
  const calibrated = result.calibration !== null;

  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Overhead Squat Assessment (frontal)"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Overhead Squat Assessment
        </h2>
        <p className="mt-2 text-sm text-muted">
          NASM/FMS-style movement screen · frontal-plane items only
        </p>
      </div>

      <ClassificationBanner result={result} />
      <SummaryCards result={result} calibrated={calibrated} />
      <ChecklistTable items={result.checklist} />

      {result.peak_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Worst-rep frame (bottom)
          </h3>
          <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.peak_screenshot_data_url}
              alt="Bottom of the worst-valgus rep"
              className="block w-full"
            />
          </div>
        </section>
      )}

      <section>
        <h3 className="text-base font-semibold tracking-tight">
          Clinical interpretation
        </h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
          <p className="mt-3 text-xs text-amber-700 dark:text-amber-400">
            ⚠ <strong>Frontal-view caveat.</strong> A single frontal camera
            cannot resolve sagittal squat depth (torso-tibia parallelism),
            excessive forward trunk lean, or heel rise. Depth here is a
            proxy from hip-Y descent as a fraction of leg length. For the
            full 5-view NASM Overhead Squat, add a lateral camera.
          </p>
          {!calibrated && (
            <p className="mt-3 text-xs text-amber-700 dark:text-amber-400">
              ⚠ This session did not include a scale calibration. Depth is
              reported in pixels + fraction-of-leg-length only. Classification
              thresholds are fraction-based so they remain valid without cm.
            </p>
          )}
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Classification banner ─────────────────────────────────────
function ClassificationBanner({ result }: { result: OverheadSquatResult }) {
  const tone = classificationTone(result.classification);
  const label = classificationLabel(result.classification);
  const durationSec = result.duration_seconds.toFixed(1);
  return (
    <div className={`rounded-card border p-5 ${tone.bg}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-subtle">
            Movement-screen classification
          </p>
          <div className="mt-2 flex flex-wrap items-baseline gap-3">
            <span
              className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ${tone.pill}`}
            >
              {label}
            </span>
            <span className="text-sm text-muted">
              {result.measurable_fails} of 5 measurable item
              {result.measurable_fails === 1 ? "" : "s"} failed
            </span>
          </div>
          <p className="mt-3 text-xs text-muted">
            Scored across {result.rep_count} rep
            {result.rep_count === 1 ? "" : "s"} over {durationSec}s.
            Items 6 (torso lean) and 7 (heel rise) are honestly not
            assessed — single frontal camera cannot resolve them.
          </p>
        </div>
      </div>
    </div>
  );
}

function classificationLabel(c: OverheadSquatClassification): string {
  return c === "good" ? "Good" : c === "moderate" ? "Moderate" : "Poor";
}
function classificationTone(c: OverheadSquatClassification): {
  bg: string;
  pill: string;
} {
  if (c === "good") {
    return {
      bg: "border-emerald-500/30 bg-emerald-500/5",
      pill: "bg-emerald-500/15 text-emerald-700 ring-emerald-500/40 dark:text-emerald-300",
    };
  }
  if (c === "moderate") {
    return {
      bg: "border-amber-500/40 bg-amber-500/5",
      pill: "bg-amber-500/15 text-amber-700 ring-amber-500/40 dark:text-amber-300",
    };
  }
  return {
    bg: "border-rose-500/40 bg-rose-500/5",
    pill: "bg-rose-500/15 text-rose-700 ring-rose-500/40 dark:text-rose-300",
  };
}

// ─── Summary cards ─────────────────────────────────────────────
function SummaryCards({
  result,
  calibrated,
}: {
  result: OverheadSquatResult;
  calibrated: boolean;
}) {
  const depthText = (() => {
    if (result.max_depth_cm !== null) {
      return `${result.max_depth_cm.toFixed(1)} cm`;
    }
    return `${(result.max_depth_frac * 100).toFixed(0)}% leg`;
  })();

  const valgusText =
    result.max_valgus_worse_deg > 0
      ? `${result.max_valgus_worse_deg.toFixed(1)}°`
      : "—";

  const pelvicText = `${(result.max_pelvic_tilt_frac * 100).toFixed(0)}%`;

  return (
    <section className="grid gap-4 md:grid-cols-3">
      <SummaryCard
        title="Deepest rep"
        primary={depthText}
        secondary={`Mean depth ${(result.mean_depth_frac * 100).toFixed(0)}% of leg length across reps`}
        relativeTag={!calibrated && result.max_depth_cm === null}
      />
      <SummaryCard
        title="Worst valgus at bottom"
        primary={valgusText}
        secondary={`Mean ${result.mean_valgus_worse_deg.toFixed(1)}° across all rep bottoms`}
      />
      <SummaryCard
        title="Max pelvic drop"
        primary={pelvicText}
        secondary={`Of horizontal hip span — foot-placement ${result.mean_ank_spread_ratio.toFixed(2)}× hip span`}
      />
    </section>
  );
}

function SummaryCard({
  title,
  primary,
  secondary,
  relativeTag,
}: {
  title: string;
  primary: string;
  secondary?: string;
  relativeTag?: boolean;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-6">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {title}
      </p>
      <div className="mt-3 flex flex-wrap items-baseline gap-2">
        <p className="tabular text-3xl font-semibold tracking-tight text-foreground md:text-4xl">
          {primary}
        </p>
        {relativeTag && (
          <span className="rounded-full bg-amber-500/10 px-2 py-1 text-[10px] font-medium text-amber-700 dark:text-amber-400">
            relative
          </span>
        )}
      </div>
      {secondary && (
        <p className="mt-3 text-xs text-muted">{secondary}</p>
      )}
    </div>
  );
}

// ─── Checklist table ───────────────────────────────────────────
function ChecklistTable({ items }: { items: OverheadSquatChecklistItem[] }) {
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Frontal-plane 7-item checklist
      </h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="w-14 px-4 py-3 font-semibold">#</th>
              <th className="px-4 py-3 font-semibold">Item</th>
              <th className="px-4 py-3 font-semibold">Result</th>
              <th className="px-4 py-3 font-semibold">Detail</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {items.map((item) => (
              <ChecklistRow key={item.index} item={item} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ChecklistRow({ item }: { item: OverheadSquatChecklistItem }) {
  return (
    <tr>
      <td className="px-4 py-3 tabular font-medium text-foreground">
        {item.index}
      </td>
      <td className="px-4 py-3 text-foreground">{item.label}</td>
      <td className="px-4 py-3">
        <StatusPill status={item.status} />
      </td>
      <td className="px-4 py-3 text-xs text-muted">
        {item.detail ?? "—"}
      </td>
    </tr>
  );
}

function StatusPill({ status }: { status: OverheadSquatChecklistStatus }) {
  if (status === "pass") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:text-emerald-300">
        <CheckCircle2 className="h-3.5 w-3.5" />
        Pass
      </span>
    );
  }
  if (status === "fail") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-xs font-medium text-rose-700 dark:text-rose-300">
        <XCircle className="h-3.5 w-3.5" />
        Fail
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-slate-500/15 px-2 py-0.5 text-xs font-medium text-slate-700 dark:text-slate-300">
      <MinusCircle className="h-3.5 w-3.5" />
      Not assessed
    </span>
  );
}

export function buildOverheadSquatInterpretation(
  result: OverheadSquatResult,
): string {
  if (result.interpretation && result.interpretation.length > 0) {
    return result.interpretation;
  }
  const label = classificationLabel(result.classification);
  const fails = result.measurable_fails;
  const notes: string[] = [];
  if (result.max_valgus_worse_deg > 12) {
    notes.push(
      `Peak knee valgus ${result.max_valgus_worse_deg.toFixed(1)}° at squat ` +
        "bottom — consider glute-medius activation and hip-abductor work.",
    );
  }
  if (result.max_pelvic_tilt_frac > 0.1) {
    notes.push(
      `Pelvic drop ${Math.round(result.max_pelvic_tilt_frac * 100)}% of ` +
        "hip span — check for hip abductor / core stability deficits.",
    );
  }
  if (result.min_arm_overhead_frac !== null && result.min_arm_overhead_frac < 0.3) {
    notes.push(
      "Arms drop from overhead during the squat — likely tight lats / " +
        "thoracic mobility limitation.",
    );
  }
  if (result.max_depth_frac < 0.2) {
    notes.push(
      `Deepest rep only reached ${Math.round(result.max_depth_frac * 100)}% ` +
        "hip descent (of leg length) — check ankle dorsiflexion and hip mobility.",
    );
  }
  if (notes.length === 0) {
    notes.push(
      fails === 0
        ? "No measurable frontal-plane items failed."
        : `${fails} item${fails === 1 ? "" : "s"} flagged (see table).`,
    );
  }
  return `${label}. ${notes.join(" ")} Items 6 (torso lean) and 7 (heel rise) require a lateral / feet close-up view and are honestly not assessed here.`;
}
