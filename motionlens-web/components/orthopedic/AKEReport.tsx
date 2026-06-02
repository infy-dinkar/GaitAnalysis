"use client";
// Active Knee Extension side-by-side report.
//
// Per-side column shows: classification badge + max-knee-angle readout
// + extension deficit, thigh-held summary, peak-frame annotated
// screenshot, and a per-frame knee-angle line chart split into valid
// (thigh held) vs invalid (thigh drifting) segments. Includes a
// plain-language interpretation paragraph + the unified disclaimer.

import dynamic from "next/dynamic";
import {
  AKE_CLASSIFICATION_LABEL,
  AKE_CLASSIFICATION_TONE,
  MILD_MAX_DEFICIT_DEG,
  MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG,
  MODERATE_MAX_DEFICIT_DEG,
  NORMAL_MAX_DEFICIT_DEG,
  THIGH_HELD_MAX_DEG,
  THIGH_HELD_MIN_DEG,
  TRIAL_DURATION_SEC,
  type AKEFullResult,
  type AKESideResult,
} from "@/lib/orthopedic/ake";
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
  result: AKEFullResult;
  interpretation: string;
}

export function AKEReport({ patientName, patient, result, interpretation }: Props) {
  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Active Knee Extension test · max knee angle + extension deficit per side"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Active Knee Extension test
        </h2>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SideColumn label="Left-leg AKE"  result={result.left} />
        <SideColumn label="Right-leg AKE" result={result.right} />
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
              <span className="font-medium text-foreground">Extension deficit:</span>{" "}
              ≤ {NORMAL_MAX_DEFICIT_DEG}° = normal,{" "}
              {NORMAL_MAX_DEFICIT_DEG + 1}–{MILD_MAX_DEFICIT_DEG}° = mild tightness,{" "}
              {MILD_MAX_DEFICIT_DEG + 1}–{MODERATE_MAX_DEFICIT_DEG}° = moderate,{" "}
              &gt; {MODERATE_MAX_DEFICIT_DEG}° = severe.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Deficit formula:</span>{" "}
              180° − maximum knee angle reached. A fully straight knee = 180° = 0° deficit.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Thigh-held gate:</span>{" "}
              hip-flexion angle must stay between {THIGH_HELD_MIN_DEG}° and {THIGH_HELD_MAX_DEG}°
              for the knee angle to count. Frames outside this band are excluded from the max.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Trial window:</span>{" "}
              up to {TRIAL_DURATION_SEC} s per side. L–R asymmetry &gt; 10° flags meaningful
              side difference.
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
  result: AKESideResult | null;
}) {
  if (!result) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-6 text-center">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <p className="mt-3 text-sm text-muted">No recording for this side.</p>
      </section>
    );
  }

  const classTone = AKE_CLASSIFICATION_TONE[result.classification];
  const classLabel = AKE_CLASSIFICATION_LABEL[result.classification];

  const noExtension = result.max_knee_angle_deg < MIN_MAX_KNEE_FOR_VALID_TRIAL_DEG;
  const heldPct = (result.thigh_held_fraction * 100).toFixed(0);

  const tAxisSec = result.samples.map((s) => s.t_ms / 1000);
  const validSeries = result.samples.map((s) =>
    s.thigh_held ? s.knee_angle_deg : null,
  );
  const invalidSeries = result.samples.map((s) =>
    !s.thigh_held ? s.knee_angle_deg : null,
  );

  return (
    <section className="space-y-4 rounded-card border border-border bg-surface p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${classTone}`}>
          {classLabel}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="tabular text-4xl font-semibold text-foreground">
            {result.max_knee_angle_deg.toFixed(1)}°
          </p>
          <p className="text-xs text-muted">max knee angle</p>
        </div>
        <div>
          <p className="tabular text-4xl font-semibold text-foreground">
            {result.deficit_deg.toFixed(1)}°
          </p>
          <p className="text-xs text-muted">extension deficit</p>
        </div>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row
            label="Hip flex at peak"
            value={
              result.hip_flex_angle_at_peak_deg !== null
                ? `${result.hip_flex_angle_at_peak_deg.toFixed(1)}°`
                : "—"
            }
          />
          <Row
            label="Thigh-held frames"
            value={`${heldPct}% of trial`}
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

      {noExtension && (
        <p className="rounded-md border border-warning/40 bg-warning/5 px-3 py-2 text-xs text-foreground">
          No knee extension detected. Re-record with the patient holding the thigh
          vertical and slowly straightening the knee.
        </p>
      )}

      {result.samples.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Knee angle vs time
          </p>
          <PlotlyChart
            height={220}
            data={[
              {
                type: "scatter", mode: "lines", name: "Valid (thigh held)",
                x: tAxisSec, y: validSeries,
                line: { color: "#10B981", width: 2 },
                connectgaps: false,
              },
              {
                type: "scatter", mode: "lines", name: "Thigh drifting",
                x: tAxisSec, y: invalidSeries,
                line: { color: "#F59E0B", width: 2, dash: "dot" },
                connectgaps: false,
              },
            ]}
            layout={{
              xaxis: { title: { text: "Time (s)" } },
              yaxis: {
                title: { text: "Knee angle (°)" },
                range: [60, 185],
              },
              shapes: [
                // Reference line at 180° (fully straight)
                {
                  type: "line",
                  xref: "paper",
                  x0: 0, x1: 1,
                  yref: "y",
                  y0: 180, y1: 180,
                  line: { color: "rgba(16,185,129,0.5)", width: 1, dash: "dash" },
                  layer: "below",
                },
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
            Peak-extension frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.peak_screenshot_data_url}
            alt={`${label} peak-extension frame`}
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
