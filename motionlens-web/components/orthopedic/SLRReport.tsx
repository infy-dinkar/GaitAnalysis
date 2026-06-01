"use client";
// Straight Leg Raise side-by-side report.
//
// Per-side column shows: classification badge + max-valid-raise
// readout, knee-straightness summary, peak-frame annotated
// screenshot, and a per-frame raise-angle line chart. Includes a
// plain-language interpretation paragraph + the unified disclaimer.

import dynamic from "next/dynamic";
import {
  MIN_RAISE_FOR_VALID_TRIAL_DEG,
  NORMAL_MAX_DEG,
  POSITIVE_MAX_DEG,
  SEVERELY_LIMITED_MAX_DEG,
  SLR_CLASSIFICATION_LABEL,
  SLR_CLASSIFICATION_TONE,
  STRAIGHT_THRESHOLD_DEG,
  TRIAL_DURATION_SEC,
  type SLRFullResult,
  type SLRSideResult,
} from "@/lib/orthopedic/slr";
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
  result: SLRFullResult;
  interpretation: string;
}

export function SLRReport({ patientName, patient, result, interpretation }: Props) {
  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Straight Leg Raise test · max valid angle per side"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Straight Leg Raise test
        </h2>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SideColumn label="Left-leg raise"  result={result.left} />
        <SideColumn label="Right-leg raise" result={result.right} />
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
              <span className="font-medium text-foreground">Max valid raise:</span>{" "}
              &lt; {SEVERELY_LIMITED_MAX_DEG}° = severely limited,{" "}
              {SEVERELY_LIMITED_MAX_DEG}–{POSITIVE_MAX_DEG}° = positive SLR (possible nerve tension),{" "}
              {POSITIVE_MAX_DEG}–{NORMAL_MAX_DEG}° = normal,{" "}
              &gt; {NORMAL_MAX_DEG}° = hypermobile.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Knee straightness:</span>{" "}
              measured inner angle at the knee must stay ≥ {STRAIGHT_THRESHOLD_DEG}° for
              the raise to count. Frames where the knee bent are excluded from the max.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Trial window:</span>{" "}
              up to {TRIAL_DURATION_SEC} s per side. The maximum valid raise reached
              during this window is reported.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">L–R asymmetry:</span>{" "}
              a side-to-side delta &gt; 15° flags clinically meaningful asymmetry.
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
  result: SLRSideResult | null;
}) {
  if (!result) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-6 text-center">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <p className="mt-3 text-sm text-muted">No recording for this side.</p>
      </section>
    );
  }

  const classTone = SLR_CLASSIFICATION_TONE[result.classification];
  const classLabel = SLR_CLASSIFICATION_LABEL[result.classification];

  const noRaise = result.max_raise_angle_deg < MIN_RAISE_FOR_VALID_TRIAL_DEG;
  const kneeStraightPct = (result.knee_straight_fraction * 100).toFixed(0);

  const tAxisSec = result.samples.map((s) => s.t_ms / 1000);
  const raiseSeries = result.samples.map((s) =>
    s.knee_straight ? s.raise_angle_deg : null,
  );
  const bentSeries = result.samples.map((s) =>
    !s.knee_straight ? s.raise_angle_deg : null,
  );

  return (
    <section className="space-y-4 rounded-card border border-border bg-surface p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${classTone}`}>
          {classLabel}
        </span>
      </div>

      <div className="flex items-baseline gap-2">
        <p className="tabular text-4xl font-semibold text-foreground">
          {result.max_raise_angle_deg.toFixed(1)}°
        </p>
        <p className="text-xs text-muted">max valid raise</p>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row
            label="Knee straight at peak"
            value={
              result.knee_angle_at_peak_deg !== null
                ? `${result.knee_angle_at_peak_deg.toFixed(1)}°`
                : "—"
            }
          />
          <Row
            label="Knee-straight frames"
            value={`${kneeStraightPct}% of trial`}
          />
          <Row label="Trial duration" value={`${result.duration_seconds.toFixed(1)} s`} />
          <Row
            label="Termination"
            value={
              result.termination === "completed" ? "Trial completed"
                : result.termination === "timeout"   ? `${TRIAL_DURATION_SEC} s timeout`
                : "Stopped early"
            }
          />
        </tbody>
      </table>

      {noRaise && (
        <p className="rounded-md border border-warning/40 bg-warning/5 px-3 py-2 text-xs text-foreground">
          No leg raise detected. Re-record with the patient lifting the leg from
          flat to as high as they can comfortably reach.
        </p>
      )}

      {result.samples.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Raise angle vs time
          </p>
          <PlotlyChart
            height={220}
            data={[
              {
                type: "scatter", mode: "lines", name: "Valid (knee straight)",
                x: tAxisSec, y: raiseSeries,
                line: { color: "#10B981", width: 2 },
                connectgaps: false,
              },
              {
                type: "scatter", mode: "lines", name: "Knee bent",
                x: tAxisSec, y: bentSeries,
                line: { color: "#F59E0B", width: 2, dash: "dot" },
                connectgaps: false,
              },
            ]}
            layout={{
              xaxis: { title: { text: "Time (s)" } },
              yaxis: { title: { text: "Raise angle (°)" }, rangemode: "tozero" },
              shapes: [
                bandShape(0, SEVERELY_LIMITED_MAX_DEG, "rgba(239,68,68,0.10)"),
                bandShape(SEVERELY_LIMITED_MAX_DEG, POSITIVE_MAX_DEG, "rgba(245,158,11,0.10)"),
                bandShape(POSITIVE_MAX_DEG, NORMAL_MAX_DEG, "rgba(16,185,129,0.10)"),
                bandShape(NORMAL_MAX_DEG, NORMAL_MAX_DEG + 30, "rgba(56,189,248,0.10)"),
              ],
              legend: { orientation: "h", y: 1.18, x: 0.5, xanchor: "center" },
              margin: { l: 56, r: 24, t: 20, b: 44 },
            }}
          />
        </div>
      )}

      {result.peak_screenshot_data_url && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Peak-raise frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.peak_screenshot_data_url}
            alt={`${label} peak-raise frame`}
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
