"use client";
// Forward Lunge (B3) side-by-side report.
//
// Per-side column shows: overall classification badge, big-number
// readout for worst-rep knee depth + KOT ratio + trunk lean, per-rep
// table with all three metrics, grouped-bar chart of the three
// metrics across reps, and the worst-rep annotated frame. Includes a
// plain-language interpretation paragraph + the unified disclaimer.

import dynamic from "next/dynamic";
import {
  DEPTH_VARIATION_FLAG_DEG,
  KNEE_DEPTH_HARD_MAX_DEG,
  KNEE_DEPTH_HARD_MIN_DEG,
  KNEE_TARGET_MAX_DEG,
  KNEE_TARGET_MIN_DEG,
  KOT_FLAG_RATIO,
  LUNGE_CLASSIFICATION_LABEL,
  LUNGE_CLASSIFICATION_TONE,
  TARGET_REP_COUNT,
  TRUNK_LEAN_FLAG_DEG,
  type ForwardLungeFullResult,
  type ForwardLungeSideResult,
} from "@/lib/orthopedic/forwardLunge";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import type { PatientDTO } from "@/lib/patients";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  result: ForwardLungeFullResult;
  interpretation: string;
}

export function ForwardLungeReport({ patientName, patient, result, interpretation }: Props) {
  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle={`Forward Lunge (B3) · ${TARGET_REP_COUNT} reps per side`}
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Forward Lunge test
        </h2>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SideColumn label="Left-leg lunge"  result={result.left} />
        <SideColumn label="Right-leg lunge" result={result.right} />
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
              <span className="font-medium text-foreground">Knee depth at bottom:</span>{" "}
              target {KNEE_TARGET_MIN_DEG}–{KNEE_TARGET_MAX_DEG}°. Outside{" "}
              {KNEE_DEPTH_HARD_MIN_DEG}–{KNEE_DEPTH_HARD_MAX_DEG}° = depth out-of-band flag.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Knee-over-toe (KOT):</span>{" "}
              fraction of leg length the knee passes forward of the foot. Flag at{" "}
              &gt; {(KOT_FLAG_RATIO * 100).toFixed(0)}% (≈ 5 cm on an 85 cm leg) — ankle / quadriceps dominance.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Trunk forward lean:</span>{" "}
              angle of hip → shoulder from vertical. Flag at &gt; {TRUNK_LEAN_FLAG_DEG}° — posterior-chain compensation.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Depth variation:</span>{" "}
              max − min knee depth across the {TARGET_REP_COUNT} reps. Flag at &gt; {DEPTH_VARIATION_FLAG_DEG}° — possible fatigue / inconsistency.
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
  result: ForwardLungeSideResult | null;
}) {
  if (!result) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-6 text-center">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <p className="mt-3 text-sm text-muted">No recording for this side.</p>
      </section>
    );
  }

  const tone  = LUNGE_CLASSIFICATION_TONE[result.classification];
  const cls   = LUNGE_CLASSIFICATION_LABEL[result.classification];

  const repNumbers = result.reps.map((r) => r.rep_index);
  const kneeSeries = result.reps.map((r) => r.knee_angle_at_bottom_deg);
  const kotSeries  = result.reps.map((r) =>
    r.knee_over_toe_ratio !== null ? r.knee_over_toe_ratio * 100 : null,
  );
  const trunkSeries = result.reps.map((r) => r.trunk_lean_deg);

  return (
    <section className="space-y-4 rounded-card border border-border bg-surface p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${tone}`}>
          {cls}
        </span>
      </div>

      {/* Three big-number readouts: worst-rep depth, worst KOT, worst trunk */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <p className="tabular text-2xl font-semibold text-foreground">
            {result.worst_rep_knee_angle_deg.toFixed(0)}°
          </p>
          <p className="text-[11px] text-muted">worst-rep knee</p>
        </div>
        <div>
          <p className="tabular text-2xl font-semibold text-foreground">
            {(result.worst_rep_kot_ratio * 100).toFixed(1)}%
          </p>
          <p className="text-[11px] text-muted">max KOT (leg %)</p>
        </div>
        <div>
          <p className="tabular text-2xl font-semibold text-foreground">
            {result.worst_rep_trunk_lean_deg.toFixed(1)}°
          </p>
          <p className="text-[11px] text-muted">max trunk lean</p>
        </div>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row label="Mean knee depth" value={`${result.mean_knee_angle_deg.toFixed(1)}°`} />
          <Row label="Depth variation" value={`${result.depth_variation_deg.toFixed(1)}°`} />
          <Row
            label="Reps captured"
            value={`${result.reps.length} / ${TARGET_REP_COUNT}${result.incomplete ? " (incomplete)" : ""}`}
          />
          <Row label="Trial duration" value={`${result.duration_seconds.toFixed(1)} s`} />
          <Row
            label="Termination"
            value={
              result.termination === "completed" ? `${TARGET_REP_COUNT} reps captured`
                : result.termination === "timeout"   ? "Trial timeout"
                : "Stopped early"
            }
          />
        </tbody>
      </table>

      {/* Per-flag chips so the doctor sees at a glance which gates fired */}
      <div className="flex flex-wrap gap-2">
        <FlagChip label="Depth in band" ok={!result.depth_out_of_band} />
        <FlagChip label="Knee over toe" ok={!result.kot_flagged} />
        <FlagChip label="Trunk lean"    ok={!result.trunk_lean_flagged} />
        <FlagChip label="Consistent depth" ok={!result.fatigue_flagged} />
      </div>

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
                  <th className="px-3 py-2 text-right font-medium">Knee (°)</th>
                  <th className="px-3 py-2 text-right font-medium">KOT (% leg)</th>
                  <th className="px-3 py-2 text-right font-medium">Trunk lean (°)</th>
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
                        {r.knee_angle_at_bottom_deg !== null ? r.knee_angle_at_bottom_deg.toFixed(1) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular text-foreground">
                        {r.knee_over_toe_ratio !== null ? (r.knee_over_toe_ratio * 100).toFixed(1) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right tabular text-foreground">
                        {r.trunk_lean_deg !== null ? r.trunk_lean_deg.toFixed(1) : "—"}
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
            Metrics across reps
          </p>
          <PlotlyChart
            height={240}
            data={[
              {
                type: "bar", name: "Knee (°)",
                x: repNumbers, y: kneeSeries,
                marker: { color: "#3B82F6" },
              },
              {
                type: "bar", name: "KOT (% leg)",
                x: repNumbers, y: kotSeries,
                marker: { color: "#EA580C" },
              },
              {
                type: "bar", name: "Trunk lean (°)",
                x: repNumbers, y: trunkSeries,
                marker: { color: "#94A3B8" },
              },
            ]}
            layout={{
              barmode: "group",
              xaxis: { title: { text: "Rep" }, dtick: 1 },
              yaxis: { title: { text: "Value" } },
              shapes: [
                // Knee target band (85-95°)
                {
                  type: "rect",
                  xref: "paper", yref: "y",
                  x0: 0, x1: 1,
                  y0: KNEE_TARGET_MIN_DEG, y1: KNEE_TARGET_MAX_DEG,
                  fillcolor: "rgba(16,185,129,0.08)",
                  line: { width: 0 },
                  layer: "below",
                },
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

function FlagChip({ label, ok }: { label: string; ok: boolean }) {
  const tone = ok
    ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
    : "bg-red-500/10 text-red-700 dark:text-red-400";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${tone}`}>
      {label} {ok ? "✓" : "✗"}
    </span>
  );
}
