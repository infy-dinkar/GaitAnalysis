"use client";
// D2 Tuck Jump (Myer's TJA) — result report.
//
// Sections:
//   • Classification pill (good / moderate / poor) + measurable-
//     fails badge and jump-count.
//   • Summary cards — mean jump height, worst valgus, height fade.
//   • Myer's 10-item checklist table.
//   • Apex screenshot from the best jump.
//   • Clinical interpretation + disclaimer.

import { CheckCircle2, MinusCircle, XCircle } from "lucide-react";

import {
  physicsJumpHeightCm,
  type TuckJumpChecklistItem,
  type TuckJumpChecklistStatus,
  type TuckJumpClassification,
  type TuckJumpResult,
} from "@/lib/orthopedic/tuckJump";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  result: TuckJumpResult;
  interpretation: string;
}

export function TuckJumpReport({
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
        subtitle="Tuck Jump Assessment (Myer's TJA)"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Tuck Jump Assessment
        </h2>
        <p className="mt-2 text-sm text-muted">
          Myer&apos;s 10-item injury-risk screen · continuous frontal-view session
        </p>
      </div>

      <ClassificationBanner result={result} />
      <SummaryCards result={result} calibrated={calibrated} />
      <ChecklistTable items={result.checklist} />

      {result.peak_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Apex frame (best jump)
          </h3>
          <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.peak_screenshot_data_url}
              alt="Apex of best tuck jump"
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
          {!calibrated && (
            <p className="mt-3 text-xs text-amber-700 dark:text-amber-400">
              ⚠ This session did not include a scale calibration. Jump
              heights are reported in pixels (relative units). The Myer&apos;s
              scoring items are fraction-of-leg-length based and stay
              valid without calibration.
            </p>
          )}
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Classification banner ─────────────────────────────────────
function ClassificationBanner({ result }: { result: TuckJumpResult }) {
  const tone = classificationTone(result.classification);
  const label = classificationLabel(result.classification);
  const durationSec = result.duration_seconds.toFixed(1);
  return (
    <div className={`rounded-card border p-5 ${tone.bg}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-subtle">
            Injury-risk classification
          </p>
          <div className="mt-2 flex flex-wrap items-baseline gap-3">
            <span
              className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ${tone.pill}`}
            >
              {label}
            </span>
            <span className="text-sm text-muted">
              {result.measurable_fails} of 8 measurable item
              {result.measurable_fails === 1 ? "" : "s"} failed
            </span>
          </div>
          <p className="mt-3 text-xs text-muted">
            Scored across {result.jump_count} tuck jump
            {result.jump_count === 1 ? "" : "s"} over {durationSec}s. Items 5
            (foot yaw) and 7 (contact noise) are honestly not assessed —
            single frontal camera cannot resolve them.
          </p>
        </div>
      </div>
    </div>
  );
}

function classificationLabel(c: TuckJumpClassification): string {
  return c === "good" ? "Good" : c === "moderate" ? "Moderate risk" : "Poor";
}
function classificationTone(c: TuckJumpClassification): {
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
  result: TuckJumpResult;
  calibrated: boolean;
}) {
  const meanHeightText = (() => {
    if (result.mean_jump_height_cm !== null) {
      return `${result.mean_jump_height_cm.toFixed(1)} cm`;
    }
    if (result.mean_jump_height_px > 0) {
      return `${result.mean_jump_height_px.toFixed(0)} px`;
    }
    return "—";
  })();

  const worstValgusText =
    result.max_valgus_worse_deg > 0
      ? `${result.max_valgus_worse_deg.toFixed(1)}°`
      : "—";

  const heightFadeText = `${(result.height_fade_frac * 100).toFixed(0)}%`;

  const meanFlightSec =
    result.jumps.length > 0
      ? result.jumps.reduce((acc, j) => acc + j.flight_time_sec, 0) /
        result.jumps.length
      : 0;
  const physicsText =
    meanFlightSec > 0
      ? `${physicsJumpHeightCm(meanFlightSec).toFixed(1)} cm`
      : "—";

  return (
    <section className="grid gap-4 md:grid-cols-3">
      <SummaryCard
        title="Mean jump height"
        primary={meanHeightText}
        secondary={`Physics cross-check: ${physicsText}`}
        relativeTag={!calibrated && result.mean_jump_height_px > 0}
      />
      <SummaryCard
        title="Worst-side valgus (landing)"
        primary={worstValgusText}
        secondary={`Mean ${result.mean_valgus_worse_deg.toFixed(1)}° across all landings`}
      />
      <SummaryCard
        title="Fatigue — height fade"
        primary={heightFadeText}
        secondary={`Valgus growth ${result.valgus_growth_deg >= 0 ? "+" : ""}${result.valgus_growth_deg.toFixed(1)}° across the session`}
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
function ChecklistTable({ items }: { items: TuckJumpChecklistItem[] }) {
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Myer&apos;s 10-item checklist
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

function ChecklistRow({ item }: { item: TuckJumpChecklistItem }) {
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

function StatusPill({ status }: { status: TuckJumpChecklistStatus }) {
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

export function buildTuckJumpInterpretation(result: TuckJumpResult): string {
  if (result.interpretation && result.interpretation.length > 0) {
    return result.interpretation;
  }
  const label = classificationLabel(result.classification);
  const fails = result.measurable_fails;
  const notes: string[] = [];
  if (result.max_valgus_worse_deg > 12) {
    notes.push(
      `Peak landing valgus ${result.max_valgus_worse_deg.toFixed(1)}° — ` +
        "elevated knee-abduction load; consider single-leg squat and " +
        "landing-mechanics work.",
    );
  }
  if (result.height_fade_frac > 0.2) {
    notes.push(
      `Jump height faded ${Math.round(result.height_fade_frac * 100)}% ` +
        "across the session — indicates conditioning / neuromuscular fatigue.",
    );
  }
  if (result.footprint_drift_frac > 0.1) {
    notes.push(
      "Landing footprint drifted noticeably across jumps — " +
        "reinforces balance / landing-target training.",
    );
  }
  if (notes.length === 0) {
    notes.push(
      fails === 0
        ? "No measurable items failed — good frontal-plane mechanics."
        : `${fails} item${fails === 1 ? "" : "s"} flagged (see table).`,
    );
  }
  return `${label}. ${notes.join(" ")} Note: Myer items 5 (foot yaw) and 7 (contact noise) are honestly not assessed — a single frontal camera cannot resolve them.`;
}
