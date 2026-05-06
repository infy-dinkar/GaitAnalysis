"use client";
// Trendelenburg side-by-side report.
//
// Renders L vs R columns with: classification badge, max/mean drop +
// hold duration table, peak-drop screenshot, pelvic-tilt time-series
// chart. Plus the plain-language interpretation paragraph and the
// unified disclaimer.

import dynamic from "next/dynamic";
import {
  CLASSIFICATION_LABEL,
  CLASSIFICATION_TONE,
  COMPENSATORY_TRUNK_LEAN_DEG,
  COMPENSATED_DROP_MAX_DEG,
  NEGATIVE_DROP_MAX_DEG,
  SHORT_HOLD_THRESHOLD_SEC,
  TARGET_HOLD_SECONDS,
  dropForStance,
  type Side,
  type TrendelenburgFullResult,
  type TrendelenburgSideResult,
} from "@/lib/orthopedic/trendelenburg";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

interface Props {
  patientName: string | null;
  result: TrendelenburgFullResult;
  interpretation: string;
}

export function TrendelenburgReport({ patientName, result, interpretation }: Props) {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Trendelenburg test
        </h2>
        <p className="mt-2 text-sm text-muted">
          {patientName ?? "Patient"} · single-leg stance, {TARGET_HOLD_SECONDS}-second hold per side
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SideColumn label="Left-leg stance"  side="left"  result={result.left} />
        <SideColumn label="Right-leg stance" side="right" result={result.right} />
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
              <span className="font-medium text-foreground">Max pelvic drop:</span>{" "}
              &lt;{NEGATIVE_DROP_MAX_DEG}° = negative,{" "}
              {NEGATIVE_DROP_MAX_DEG}–{COMPENSATED_DROP_MAX_DEG}° = compensated,{" "}
              &gt;{COMPENSATED_DROP_MAX_DEG}° = positive Trendelenburg.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Hold duration:</span>{" "}
              &lt;{SHORT_HOLD_THRESHOLD_SEC}s = additional balance / strength concern.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Compensatory trunk lean:</span>{" "}
              lean &gt;{COMPENSATORY_TRUNK_LEAN_DEG}° toward the stance side flags
              a Trendelenburg gait pattern.
            </p>
            <p className="mt-1">
              Pelvic-drop sign reoriented so that &quot;drop on the lifted side&quot;
              is positive regardless of which leg is the stance leg.
            </p>
          </div>
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Per-side column ─────────────────────────────────────────────
function SideColumn({
  label,
  side,
  result,
}: {
  label: string;
  side: Side;
  result: TrendelenburgSideResult | null;
}) {
  if (!result) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-6 text-center">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">
          {label}
        </h3>
        <p className="mt-3 text-sm text-muted">
          No recording for this side.
        </p>
      </section>
    );
  }

  const tone = CLASSIFICATION_TONE[result.classification];
  const classLabel = CLASSIFICATION_LABEL[result.classification];

  // Plotly traces: pelvic drop curve (re-oriented so positive = drop on
  // lifted side), with reference shading at the spec cutoffs.
  const xs = result.samples.map((s) => s.t_ms / 1000);
  const dropSeries = result.samples.map((s) =>
    s.pelvic_tilt_deg !== null
      ? dropForStance(s.pelvic_tilt_deg, side)
      : null,
  );
  const leanSeries = result.samples.map((s) => s.trunk_lean_deg ?? null);

  return (
    <section className="space-y-4 rounded-card border border-border bg-surface/40 p-5">
      <div className="flex items-baseline justify-between gap-3 border-b border-border pb-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">
          {label}
        </h3>
        <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${tone}`}>
          {classLabel}
        </span>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row label="Max pelvic drop" value={`${result.max_drop_deg.toFixed(1)}°`} />
          <Row label="Mean pelvic drop (after first 2 s)" value={`${result.mean_drop_deg.toFixed(1)}°`} />
          <Row label="Hold duration" value={`${result.hold_seconds.toFixed(1)} s`} />
          <Row
            label="Max compensatory trunk lean"
            value={`${result.max_compensatory_lean_deg.toFixed(1)}°`}
          />
          <Row
            label="Termination"
            value={
              result.termination === "completed"
                ? "Full hold completed"
                : result.termination === "foot_touch"
                  ? "Lifted foot touched down"
                  : "Pelvic tilt spike"
            }
          />
        </tbody>
      </table>

      {(result.short_hold || result.trendelenburg_gait_pattern) && (
        <div className="rounded-md border border-warning/40 bg-warning/5 p-3 text-xs leading-relaxed text-foreground">
          {result.short_hold && (
            <p>Hold ended early ({result.hold_seconds.toFixed(1)} s &lt; {SHORT_HOLD_THRESHOLD_SEC} s) — additional concern.</p>
          )}
          {result.trendelenburg_gait_pattern && (
            <p className="mt-0.5">
              Compensatory trunk lean of {result.max_compensatory_lean_deg.toFixed(1)}° toward
              the stance side — Trendelenburg gait pattern.
            </p>
          )}
        </div>
      )}

      {result.peak_screenshot_data_url && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Peak-drop frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.peak_screenshot_data_url}
            alt={`${label} peak-drop frame`}
            className="w-full rounded-md border border-border"
          />
        </div>
      )}

      {result.samples.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Pelvic drop over the hold
          </p>
          <PlotlyChart
            height={240}
            data={[
              {
                type: "scatter",
                mode: "lines",
                name: "Pelvic drop (°)",
                x: xs,
                y: dropSeries,
                line: { color: "#EA580C", width: 1.6 },
              },
              {
                type: "scatter",
                mode: "lines",
                name: "Trunk lean toward stance (°)",
                x: xs,
                y: leanSeries,
                line: { color: "#2563EB", width: 1.2, dash: "dot" },
                yaxis: "y2",
              },
            ]}
            layout={{
              xaxis: { title: { text: "Time (s)" } },
              yaxis: { title: { text: "Pelvic drop (°)" } },
              yaxis2: {
                title: { text: "Trunk lean (°)" },
                overlaying: "y",
                side: "right",
                showgrid: false,
              },
              shapes: [
                bandShape(NEGATIVE_DROP_MAX_DEG, COMPENSATED_DROP_MAX_DEG, "rgba(245,158,11,0.10)"),
                bandShape(COMPENSATED_DROP_MAX_DEG, COMPENSATED_DROP_MAX_DEG + 20, "rgba(239,68,68,0.10)"),
              ],
              legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
              margin: { l: 60, r: 50, t: 20, b: 44 },
            }}
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
