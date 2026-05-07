"use client";
// 30-Second Chair Stand single-trial report.
//
// Headline: rep count + norm badge (above / below norm) + the
// matched CDC STEADI band label. Per-rep duration trend chart with
// the fatigue-regression line overlaid. Plus depth consistency,
// arm-uncrossing flag, plain-language interpretation, deepest-rep
// frame, and the unified disclaimer.

import dynamic from "next/dynamic";
import { AlertTriangle } from "lucide-react";
import {
  CLASSIFICATION_LABEL,
  CLASSIFICATION_TONE,
  TRIAL_DURATION_SEC,
  type ChairStand30sResult,
} from "@/lib/orthopedic/chairStand30s";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

interface Props {
  patientName: string | null;
  result: ChairStand30sResult;
  interpretation: string;
}

export function ChairStand30sReport({ patientName, result, interpretation }: Props) {
  const classTone = CLASSIFICATION_TONE[result.classification];
  const classLabel = CLASSIFICATION_LABEL[result.classification];

  const repNumbers = result.reps.map((r) => r.rep_index);
  const repDurations = result.reps.map((r) => r.duration_seconds);
  const repDepth = result.reps.map((r) => Math.max(0, 180 - r.min_knee_angle_deg));

  // Fatigue regression line for the chart — y = mx + b across rep
  // indices. Computed from the same slope returned by the math lib
  // so the dashed overlay matches the headline number exactly.
  const slope = result.fatigue_slope_sec_per_rep;
  const meanDur =
    repDurations.length === 0 ? 0 : repDurations.reduce((a, b) => a + b, 0) / repDurations.length;
  const meanX = (repDurations.length - 1) / 2;
  const intercept = meanDur - slope * meanX;
  const trendLine = repNumbers.map((_, i) => slope * i + intercept);

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          30-Second Chair Stand
        </h2>
        <p className="mt-2 text-sm text-muted">
          {patientName ?? "Patient"} · {TRIAL_DURATION_SEC}-second trial
        </p>
      </div>

      <section className="rounded-card border border-border bg-surface/40 p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Rep count
            </p>
            <p className="mt-1 text-3xl font-semibold tabular text-foreground">
              {result.rep_count}
            </p>
          </div>
          <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${classTone}`}>
            {classLabel}
          </span>
        </div>

        <table className="mt-3 w-full text-left text-sm">
          <tbody>
            <Row
              label="CDC threshold"
              value={`≥ ${result.norm_threshold} reps · ${result.norm_band_label}`}
            />
            <Row label="Mean rep duration" value={`${result.mean_rep_duration_sec.toFixed(2)} s`} />
            <Row
              label="Fatigue slope"
              value={`${result.fatigue_slope_sec_per_rep >= 0 ? "+" : ""}${result.fatigue_slope_sec_per_rep.toFixed(2)} s per rep`}
            />
            <Row label="Depth consistency (SD)" value={`${result.depth_sd_deg.toFixed(1)}°`} />
            <Row
              label="Trial duration"
              value={`${result.trial_duration_seconds.toFixed(1)} s`}
            />
            <Row
              label="Termination"
              value={
                result.termination === "completed" ? `${TRIAL_DURATION_SEC}-second timer expired` : "Stopped early"
              }
            />
          </tbody>
        </table>

        {(!result.norm_comparable || result.arm_uncrossed_flag) && (
          <div className="mt-4 space-y-2">
            {!result.norm_comparable && (
              <FlagRow tone="warning">
                Norm comparison limited — patient demographics missing or outside CDC age range (60–94).
              </FlagRow>
            )}
            {result.arm_uncrossed_flag && (
              <FlagRow tone="warning">
                Arms uncrossed during the trial — strength assessment may be inflated.
              </FlagRow>
            )}
          </div>
        )}
      </section>

      {result.reps.length > 0 && (
        <section className="space-y-4">
          <h3 className="text-base font-semibold tracking-tight">Per-rep metrics</h3>
          <div className="overflow-x-auto rounded-card border border-border bg-surface">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border bg-elevated text-[10px] uppercase tracking-[0.12em] text-subtle">
                <tr>
                  <th className="px-3 py-2 font-medium">Rep</th>
                  <th className="px-3 py-2 text-right font-medium">Duration (s)</th>
                  <th className="px-3 py-2 text-right font-medium">Min knee (°)</th>
                  <th className="px-3 py-2 text-right font-medium">Depth (°)</th>
                </tr>
              </thead>
              <tbody>
                {result.reps.map((r) => (
                  <tr key={r.rep_index} className="border-b border-border/50 last:border-b-0">
                    <td className="px-3 py-2 text-foreground">{r.rep_index}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{r.duration_seconds.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{r.min_knee_angle_deg.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{Math.max(0, 180 - r.min_knee_angle_deg).toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
              Per-rep duration · depth · fatigue trend
            </p>
            <PlotlyChart
              height={260}
              data={[
                {
                  type: "bar",
                  name: "Rep duration (s)",
                  x: repNumbers,
                  y: repDurations,
                  marker: { color: "#EA580C" },
                  yaxis: "y",
                },
                {
                  type: "scatter",
                  mode: "lines",
                  name: "Fatigue trend",
                  x: repNumbers,
                  y: trendLine,
                  line: { color: "#EA580C", width: 1.4, dash: "dash" },
                  yaxis: "y",
                  hoverinfo: "skip",
                },
                {
                  type: "scatter",
                  mode: "lines+markers",
                  name: "Depth (180° − min knee)",
                  x: repNumbers,
                  y: repDepth,
                  line: { color: "#2563EB", width: 1.6 },
                  yaxis: "y2",
                },
              ]}
              layout={{
                xaxis: { title: { text: "Rep" }, dtick: 1 },
                yaxis: { title: { text: "Duration (s)" } },
                yaxis2: {
                  title: { text: "Depth (°)" },
                  overlaying: "y",
                  side: "right",
                  showgrid: false,
                },
                legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
                margin: { l: 56, r: 50, t: 20, b: 44 },
              }}
            />
          </div>
        </section>
      )}

      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
        </div>
      </section>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Reference cutoffs</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-xs leading-relaxed text-muted">
          <p>
            <span className="font-medium text-foreground">CDC STEADI 30-Second Chair Stand:</span>{" "}
            patient is at risk for falls when rep count falls below the age- and
            sex-matched threshold. Thresholds rise (looser) with age — e.g. men
            60–64 fail at &lt;14 reps; men 90–94 fail at &lt;7 reps.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Last incomplete rep:</span>{" "}
            a partial rep that hasn&apos;t returned to the seat by the time the
            timer expires does not count.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Arm posture:</span>{" "}
            arms must remain crossed at the chest throughout — wrists dropping
            below the shoulder line flags the trial.
          </p>
        </div>
      </section>

      {result.last_rep_screenshot_data_url && (
        <section>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Deepest-rep frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.last_rep_screenshot_data_url}
            alt="Deepest-rep frame"
            className="w-full max-w-lg rounded-md border border-border"
          />
        </section>
      )}

      <ReportDisclaimer />
    </div>
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

function FlagRow({
  tone,
  children,
}: {
  tone: "warning";
  children: React.ReactNode;
}) {
  const cls =
    tone === "warning"
      ? "border-warning/40 bg-warning/5 text-foreground"
      : "border-border bg-surface text-foreground";
  return (
    <div className={`flex items-start gap-2 rounded-md border px-3 py-2 text-xs ${cls}`}>
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" />
      <span>{children}</span>
    </div>
  );
}
