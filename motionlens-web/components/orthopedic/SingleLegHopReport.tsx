"use client";
// D3 Single-Leg Hop report.
//
// Sections:
//   • LSI summary — left vs right best valid hop + LSI %
//     classification (cleared / warning / deficit / incomplete).
//   • Calibration banner — cm if a CalibrationResult was applied,
//     otherwise amber "relative units only" warning.
//   • Per-side trial log — best valid hop per leg, plus a row per
//     attempted trial with takeoff/landing times and validity.
//   • Peak screenshots (one per leg, when available).
//   • Clinical interpretation paragraph + disclaimer.

import { AlertTriangle, CheckCircle2 } from "lucide-react";

import {
  LSI_CLEARED_PCT,
  LSI_WARNING_PCT,
  type LSIClass,
  type Side,
  type SingleLegHopCombinedResult,
  type SingleLegHopResult,
  type Trial,
} from "@/lib/orthopedic/singleLegHop";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  combined: SingleLegHopCombinedResult;
  interpretation: string;
}

const LSI_LABEL: Record<LSIClass, string> = {
  cleared: "Cleared — symmetric",
  warning: "Mild asymmetry",
  deficit: "Significant deficit",
  incomplete: "Incomplete — one side missing",
};

const LSI_TONE: Record<LSIClass, string> = {
  cleared: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  warning: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
  deficit: "bg-rose-500/15 text-rose-700 dark:text-rose-300",
  incomplete: "bg-slate-500/15 text-slate-700 dark:text-slate-300",
};

export function SingleLegHopReport({
  patientName,
  patient,
  combined,
  interpretation,
}: Props) {
  const calibrated = combined.calibration !== null;

  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Single-Leg Hop test"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Single-Leg Hop test
        </h2>
        <p className="mt-2 text-sm text-muted">
          Forward hop distance · Limb Symmetry Index
        </p>
      </div>

      <CalibrationBanner combined={combined} />
      <LSISummary combined={combined} />

      <SideSection
        title="Left leg"
        result={combined.left}
        calibrated={calibrated}
      />
      <SideSection
        title="Right leg"
        result={combined.right}
        calibrated={calibrated}
      />

      <section>
        <h3 className="text-base font-semibold tracking-tight">
          Clinical interpretation
        </h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
          {!calibrated && (
            <p className="mt-3 text-xs text-amber-700 dark:text-amber-400">
              ⚠ This session did not include a scale calibration. Distances are
              reported in pixels (relative units) only — the Limb Symmetry
              Index requires centimetre measurements. Re-run with calibration.
            </p>
          )}
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Calibration banner ─────────────────────────────────────────
function CalibrationBanner({
  combined,
}: {
  combined: SingleLegHopCombinedResult;
}) {
  if (combined.calibration) {
    const c = combined.calibration;
    return (
      <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-4 text-sm">
        <div className="flex items-start gap-2">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
          <div>
            <p className="font-medium text-foreground">
              Calibrated — distances reported in centimetres
            </p>
            <p className="mt-1 text-xs text-muted">
              Scale ={" "}
              <span className="tabular font-medium">
                {c.pixels_per_cm.toFixed(2)}
              </span>{" "}
              px/cm
              {c.source ? <> · source: {c.source.replace(/_/g, " ")}</> : null}.
            </p>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-card border border-amber-500/40 bg-amber-500/5 p-4 text-sm">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div>
          <p className="font-medium text-foreground">
            Uncalibrated — relative units only, no LSI classification
          </p>
          <p className="mt-1 text-xs text-muted">
            No scale calibration was applied. Hop distances are reported in
            raw pixel units; the Limb Symmetry Index (≥ {LSI_CLEARED_PCT} %
            cleared, {LSI_WARNING_PCT}–{LSI_CLEARED_PCT} % warning, &lt;{" "}
            {LSI_WARNING_PCT} % deficit) is NOT applied.
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── LSI summary card ──────────────────────────────────────────
function LSISummary({ combined }: { combined: SingleLegHopCombinedResult }) {
  const lsiText =
    combined.lsi_pct !== null ? `${combined.lsi_pct.toFixed(1)} %` : "—";
  const calibrated = combined.calibration !== null;
  return (
    <section className="grid gap-4 md:grid-cols-3">
      <BestSideCard
        title="Left best valid"
        result={combined.left}
        calibrated={calibrated}
      />
      <div className="rounded-card border border-border bg-surface p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Limb Symmetry Index
        </p>
        <div className="mt-3 flex flex-wrap items-baseline gap-3">
          <p className="tabular text-4xl font-semibold tracking-tight text-foreground md:text-5xl">
            {lsiText}
          </p>
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-medium ${LSI_TONE[combined.lsi_class]}`}
          >
            {LSI_LABEL[combined.lsi_class]}
          </span>
        </div>
        <p className="mt-3 text-xs text-muted">
          {combined.weaker_side
            ? `Weaker side: ${combined.weaker_side}.`
            : "Both legs required for the LSI calculation."}{" "}
          Threshold ≥ {LSI_CLEARED_PCT} % = cleared (standard ACL convention).
        </p>
      </div>
      <BestSideCard
        title="Right best valid"
        result={combined.right}
        calibrated={calibrated}
      />
    </section>
  );
}

function BestSideCard({
  title,
  result,
  calibrated,
}: {
  title: string;
  result: SingleLegHopResult | null;
  calibrated: boolean;
}) {
  const text = (() => {
    if (!result) return "—";
    if (result.best_valid_hop_cm !== null) {
      return `${result.best_valid_hop_cm.toFixed(1)} cm`;
    }
    if (result.best_valid_hop_px !== null) {
      return `${result.best_valid_hop_px.toFixed(0)} px`;
    }
    return "—";
  })();
  const validCount = result?.trials.filter((t) => t.valid).length ?? 0;
  const total = result?.trials.length ?? 0;
  return (
    <div className="rounded-card border border-border bg-surface p-6">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {title}
      </p>
      <div className="mt-3 flex flex-wrap items-baseline gap-2">
        <p className="tabular text-3xl font-semibold tracking-tight text-foreground md:text-4xl">
          {text}
        </p>
        {!calibrated && result?.best_valid_hop_px !== null && (
          <span className="rounded-full bg-amber-500/10 px-2 py-1 text-[10px] font-medium text-amber-700 dark:text-amber-400">
            relative
          </span>
        )}
      </div>
      <p className="mt-3 text-xs text-muted">
        {result
          ? `${validCount} of ${total} trial${total === 1 ? "" : "s"} valid · ${result.duration_seconds.toFixed(1)} s recording`
          : "Not recorded"}
      </p>
    </div>
  );
}

// ─── Per-side section ──────────────────────────────────────────
function SideSection({
  title,
  result,
  calibrated,
}: {
  title: string;
  result: SingleLegHopResult | null;
  calibrated: boolean;
}) {
  if (!result) {
    return (
      <section>
        <h3 className="text-base font-semibold tracking-tight">{title}</h3>
        <div className="mt-3 rounded-card border border-border bg-elevated p-5 text-sm text-muted">
          No recording captured for this leg.
        </div>
      </section>
    );
  }
  const sideTitle =
    result.side_tested === "left" ? "Left leg" : "Right leg";
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">{sideTitle}</h3>
      <p className="mt-1 text-xs text-muted">
        {result.trials.length} hop
        {result.trials.length === 1 ? "" : "s"} detected ·{" "}
        {result.duration_seconds.toFixed(1)} s recording.
      </p>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-4 py-3 font-semibold">Trial</th>
              <th className="px-4 py-3 font-semibold">Takeoff</th>
              <th className="px-4 py-3 font-semibold">Landing</th>
              <th className="px-4 py-3 font-semibold">Hop distance</th>
              <th className="px-4 py-3 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {result.trials.map((t) => (
              <TrialRow
                key={t.trial_index}
                trial={t}
                calibrated={calibrated}
              />
            ))}
          </tbody>
        </table>
      </div>
      {result.peak_screenshot_data_url && (
        <div className="mt-4 overflow-hidden rounded-card border border-border bg-surface">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.peak_screenshot_data_url}
            alt={`Best hop frame for ${sideTitle}`}
            className="block w-full"
          />
        </div>
      )}
    </section>
  );
}

function TrialRow({
  trial,
  calibrated,
}: {
  trial: Trial;
  calibrated: boolean;
}) {
  const distance =
    trial.hop_distance_cm !== null
      ? `${trial.hop_distance_cm.toFixed(1)} cm`
      : `${trial.hop_distance_px.toFixed(0)} px`;
  return (
    <tr>
      <td className="px-4 py-3 font-medium text-foreground">
        Trial {trial.trial_index}
      </td>
      <td className="px-4 py-3 tabular text-muted">
        {(trial.takeoff_t_ms / 1000).toFixed(2)} s
      </td>
      <td className="px-4 py-3 tabular text-muted">
        {(trial.landing_t_ms / 1000).toFixed(2)} s
      </td>
      <td className="px-4 py-3 tabular text-foreground">
        {distance}
        {!calibrated && (
          <span className="ml-1 text-[10px] text-amber-600">rel.</span>
        )}
      </td>
      <td className="px-4 py-3">
        {trial.valid ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:text-emerald-300">
            Valid
          </span>
        ) : (
          <>
            <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-xs font-medium text-rose-700 dark:text-rose-300">
              Invalid
            </span>
            {trial.invalidation_reason && (
              <p className="mt-1 text-[11px] text-muted">
                {trial.invalidation_reason}
              </p>
            )}
          </>
        )}
      </td>
    </tr>
  );
}

export function buildSingleLegHopInterpretation(
  combined: SingleLegHopCombinedResult,
): string {
  const { left, right, lsi_pct, lsi_class, weaker_side } = combined;
  if (!left && !right) {
    return "No recordings captured — please record both legs.";
  }
  const parts: string[] = [];
  if (left?.interpretation) parts.push(left.interpretation);
  if (right?.interpretation) parts.push(right.interpretation);
  if (lsi_pct !== null && weaker_side) {
    parts.push(
      `LSI ${lsi_pct.toFixed(1)} % (weaker side: ${weaker_side}) — ${LSI_LABEL[lsi_class].toLowerCase()}.`,
    );
  } else if (lsi_class === "incomplete") {
    parts.push(
      "LSI requires a valid hop on BOTH legs — please re-record the missing side.",
    );
  }
  return parts.join(" ");
}
