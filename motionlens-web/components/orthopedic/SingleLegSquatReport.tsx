"use client";
// Single-Leg Squat side-by-side report.
//
// Per-side column shows: classification badge + composite risk badge,
// per-rep table, KFPPA-vs-rep bar chart, worst-rep annotated frame,
// and aggregates (worst KFPPA, mean pelvic drop, mean trunk lean).
// Includes a plain-language interpretation paragraph + the unified
// disclaimer.

import dynamic from "next/dynamic";
import {
  ASYMMETRY_THRESHOLD_DEG,
  KFPPA_BORDERLINE_MAX_DEG,
  KFPPA_GOOD_MAX_DEG,
  KNEE_CLASSIFICATION_LABEL,
  KNEE_CLASSIFICATION_TONE,
  PELVIC_DROP_THRESHOLD_DEG,
  RISK_LABEL,
  RISK_TONE,
  TARGET_REP_COUNT,
  TRUNK_LEAN_THRESHOLD_DEG,
  type SingleLegSquatFullResult,
  type SingleLegSquatSideResult,
} from "@/lib/orthopedic/singleLegSquat";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

interface Props {
  patientName: string | null;
  result: SingleLegSquatFullResult;
  interpretation: string;
}

export function SingleLegSquatReport({ patientName, result, interpretation }: Props) {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Single-leg squat test
        </h2>
        <p className="mt-2 text-sm text-muted">
          {patientName ?? "Patient"} · {TARGET_REP_COUNT} reps per side
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SideColumn label="Left-leg squat"  result={result.left} />
        <SideColumn label="Right-leg squat" result={result.right} />
      </div>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
        </div>
      </section>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Reference cutoffs</h3>
        <div className="mt-3 grid gap-3 rounded-card border border-border bg-surface p-5 text-xs leading-relaxed text-muted md:grid-cols-2">
          <div>
            <p>
              <span className="font-medium text-foreground">KFPPA (worst rep):</span>{" "}
              &lt;{KFPPA_GOOD_MAX_DEG}° = good,{" "}
              {KFPPA_GOOD_MAX_DEG}–{KFPPA_BORDERLINE_MAX_DEG}° = borderline,{" "}
              &gt;{KFPPA_BORDERLINE_MAX_DEG}° = dynamic valgus / elevated ACL/PFP risk.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Mean pelvic drop:</span>{" "}
              &gt;{PELVIC_DROP_THRESHOLD_DEG}° = hip abductor insufficiency on the stance side.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Mean trunk lateral lean:</span>{" "}
              &gt;{TRUNK_LEAN_THRESHOLD_DEG}° = compensatory pattern.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">L–R asymmetry:</span>{" "}
              worst-KFPPA delta &gt;{ASYMMETRY_THRESHOLD_DEG}° flags a targeted
              intervention on the worse side.
            </p>
          </div>
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

function SideColumn({
  label,
  result,
}: {
  label: string;
  result: SingleLegSquatSideResult | null;
}) {
  if (!result) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-6 text-center">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <p className="mt-3 text-sm text-muted">No recording for this side.</p>
      </section>
    );
  }

  const classTone = KNEE_CLASSIFICATION_TONE[result.classification];
  const classLabel = KNEE_CLASSIFICATION_LABEL[result.classification];
  const riskTone = RISK_TONE[result.risk_score];
  const riskLabel = RISK_LABEL[result.risk_score];

  const repNumbers = result.reps.map((r) => r.rep_index);
  const kfppaSeries = result.reps.map((r) => r.kfppa_deg);
  const dropSeries  = result.reps.map((r) => (r.pelvic_drop_deg !== null ? Math.abs(r.pelvic_drop_deg) : null));
  const leanSeries  = result.reps.map((r) => (r.trunk_lean_deg  !== null ? Math.abs(r.trunk_lean_deg)  : null));

  return (
    <section className="space-y-4 rounded-card border border-border bg-surface/40 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <div className="flex shrink-0 gap-2">
          <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${classTone}`}>
            KFPPA: {classLabel}
          </span>
          <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${riskTone}`}>
            {riskLabel}
          </span>
        </div>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row label="Worst KFPPA" value={`${result.worst_kfppa_deg.toFixed(1)}°`} />
          <Row label="Mean pelvic drop" value={`${Math.abs(result.mean_pelvic_drop_deg).toFixed(1)}°`} />
          <Row label="Mean trunk lateral lean" value={`${Math.abs(result.mean_trunk_lean_deg).toFixed(1)}°`} />
          <Row label="Mean squat depth" value={`${result.mean_depth_pct.toFixed(0)}% leg-length`} />
          <Row
            label="Reps captured"
            value={`${result.reps.length} / ${TARGET_REP_COUNT}${result.incomplete ? " (incomplete)" : ""}`}
          />
          <Row label="Trial duration" value={`${result.duration_seconds.toFixed(1)} s`} />
          <Row
            label="Termination"
            value={
              result.termination === "completed" ? "5 reps captured"
                : result.termination === "timeout"   ? "30 s timeout"
                : result.termination === "stopped"   ? "Stopped early"
                : "Camera rotated out"
            }
          />
        </tbody>
      </table>

      {result.reps.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Per-rep metrics
          </p>
          <div className="overflow-x-auto rounded-card border border-border bg-surface">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border bg-elevated text-[10px] uppercase tracking-[0.12em] text-subtle">
                <tr>
                  <th className="px-3 py-2 font-medium">Rep</th>
                  <th className="px-3 py-2 text-right font-medium">KFPPA (°)</th>
                  <th className="px-3 py-2 text-right font-medium">Pelvic drop (°)</th>
                  <th className="px-3 py-2 text-right font-medium">Trunk lean (°)</th>
                  <th className="px-3 py-2 text-right font-medium">Depth (% leg)</th>
                </tr>
              </thead>
              <tbody>
                {result.reps.map((r, i) => {
                  const isWorst = result.worst_rep_index === i;
                  return (
                    <tr
                      key={r.rep_index}
                      className={`border-b border-border/50 last:border-b-0 ${isWorst ? "bg-amber-500/5" : ""}`}
                    >
                      <td className="px-3 py-2 text-foreground">
                        {r.rep_index}
                        {isWorst && <span className="ml-1 text-[10px] uppercase text-amber-600">worst</span>}
                      </td>
                      <td className="px-3 py-2 text-right tabular text-foreground">
                        {r.kfppa_deg !== null ? r.kfppa_deg.toFixed(1) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular text-foreground">
                        {r.pelvic_drop_deg !== null ? Math.abs(r.pelvic_drop_deg).toFixed(1) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular text-foreground">
                        {r.trunk_lean_deg !== null ? Math.abs(r.trunk_lean_deg).toFixed(1) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular text-foreground">
                        {r.depth_pct !== null ? r.depth_pct.toFixed(0) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {result.reps.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            KFPPA · pelvic drop · trunk lean across reps
          </p>
          <PlotlyChart
            height={240}
            data={[
              {
                type: "bar", name: "KFPPA (°)",
                x: repNumbers, y: kfppaSeries,
                marker: { color: "#EA580C" },
              },
              {
                type: "bar", name: "Pelvic drop (°)",
                x: repNumbers, y: dropSeries,
                marker: { color: "#2563EB" },
              },
              {
                type: "bar", name: "Trunk lean (°)",
                x: repNumbers, y: leanSeries,
                marker: { color: "#94A3B8" },
              },
            ]}
            layout={{
              barmode: "group",
              xaxis: { title: { text: "Rep" }, dtick: 1 },
              yaxis: { title: { text: "Degrees" } },
              shapes: [
                bandShape(KFPPA_GOOD_MAX_DEG, KFPPA_BORDERLINE_MAX_DEG, "rgba(245,158,11,0.10)"),
                bandShape(KFPPA_BORDERLINE_MAX_DEG, KFPPA_BORDERLINE_MAX_DEG + 25, "rgba(239,68,68,0.10)"),
              ],
              legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
              margin: { l: 56, r: 24, t: 20, b: 44 },
            }}
          />
        </div>
      )}

      {result.worst_rep_screenshot_data_url && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Worst-rep frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.worst_rep_screenshot_data_url}
            alt={`${label} worst-rep frame`}
            className="w-full rounded-md border border-border"
          />
        </div>
      )}
    </section>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr className="border-b border-border/50 last:border-b-0">
      <td className="py-2 text-muted">{label}</td>
      <td className="py-2 text-right tabular text-foreground">{value}</td>
    </tr>
  );
}

function bandShape(low: number, high: number, fill: string) {
  return {
    type: "rect",
    xref: "paper",
    x0: 0,
    x1: 1,
    yref: "y",
    y0: low,
    y1: high,
    fillcolor: fill,
    line: { width: 0 },
    layer: "below",
  };
}
