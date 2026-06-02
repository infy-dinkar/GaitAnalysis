"use client";
// Modified Thomas Test side-by-side report.
//
// Per-side column shows: hip + knee classification badges, big-number
// readouts for the settled hip and knee angles, low-confidence warning
// banner if the trial timed out without auto-capture, hip+knee vs time
// chart highlighting the stable region, capture-moment screenshot.

import dynamic from "next/dynamic";
import {
  HIP_CLASSIFICATION_LABEL,
  HIP_CLASSIFICATION_TONE,
  HIP_MILD_MIN_DEG,
  HIP_NORMAL_MIN_DEG,
  KNEE_CLASSIFICATION_LABEL,
  KNEE_CLASSIFICATION_TONE,
  KNEE_NORMAL_MAX_DEG,
  TRIAL_DURATION_SEC,
  type ModifiedThomasFullResult,
  type ModifiedThomasSideResult,
} from "@/lib/orthopedic/modifiedThomas";
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
  result: ModifiedThomasFullResult;
  interpretation: string;
}

export function ModifiedThomasReport({ patientName, patient, result, interpretation }: Props) {
  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Modified Thomas Test · hip flexor + rectus femoris length per side"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Modified Thomas Test
        </h2>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <SideColumn label="Left-leg MTT"  result={result.left} />
        <SideColumn label="Right-leg MTT" result={result.right} />
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
              <span className="font-medium text-foreground">Hip angle (hip flexor):</span>{" "}
              ≥ {HIP_NORMAL_MIN_DEG}° = normal,{" "}
              {HIP_MILD_MIN_DEG}–{HIP_NORMAL_MIN_DEG}° = mild tightness,{" "}
              &lt; {HIP_MILD_MIN_DEG}° = significant tightness.
            </p>
            <p className="mt-1">
              Measured at the hip between hip→shoulder and hip→knee. 180° = thigh
              hanging in line with the body; lower values = tighter hip flexor.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Knee angle (rectus femoris):</span>{" "}
              ≤ {KNEE_NORMAL_MAX_DEG}° = relaxed,{" "}
              &gt; {KNEE_NORMAL_MAX_DEG}° = rectus femoris tightness.
            </p>
            <p className="mt-1">
              Measured at the knee between knee→hip and knee→ankle. ~80–90° = shin
              hanging naturally bent; higher values = rectus tightness pulling the
              knee toward extension.
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
  result: ModifiedThomasSideResult | null;
}) {
  if (!result) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-6 text-center">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <p className="mt-3 text-sm text-muted">No recording for this side.</p>
      </section>
    );
  }

  const hipTone  = HIP_CLASSIFICATION_TONE[result.hip_classification];
  const hipLabel = HIP_CLASSIFICATION_LABEL[result.hip_classification];
  const kneeTone  = KNEE_CLASSIFICATION_TONE[result.knee_classification];
  const kneeLabel = KNEE_CLASSIFICATION_LABEL[result.knee_classification];

  const tAxisSec = result.samples.map((s) => s.t_ms / 1000);
  const hipSeries  = result.samples.map((s) => s.hip_angle_deg);
  const kneeSeries = result.samples.map((s) => s.knee_angle_deg);
  // For the stable-region shading: find the start/end times where stable==true.
  let stableStart: number | null = null;
  let stableEnd:   number | null = null;
  for (const s of result.samples) {
    if (s.stable) {
      if (stableStart === null) stableStart = s.t_ms / 1000;
      stableEnd = s.t_ms / 1000;
    }
  }

  const captured = result.hip_angle_deg > 0 && result.knee_angle_deg > 0;

  return (
    <section className="space-y-4 rounded-card border border-border bg-surface p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-subtle">{label}</h3>
        <div className="flex shrink-0 gap-2">
          <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${hipTone}`}>
            Hip: {hipLabel}
          </span>
          <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${kneeTone}`}>
            Knee: {kneeLabel}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="tabular text-4xl font-semibold text-foreground">
            {result.hip_angle_deg.toFixed(1)}°
          </p>
          <p className="text-xs text-muted">hip angle (settled)</p>
        </div>
        <div>
          <p className="tabular text-4xl font-semibold text-foreground">
            {result.knee_angle_deg.toFixed(1)}°
          </p>
          <p className="text-xs text-muted">knee angle (settled)</p>
        </div>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row label="Hip stddev at capture" value={`±${result.hip_angle_stddev_deg.toFixed(2)}°`} />
          <Row label="Knee stddev at capture" value={`±${result.knee_angle_stddev_deg.toFixed(2)}°`} />
          <Row label="Trial duration" value={`${result.duration_seconds.toFixed(1)} s`} />
          <Row
            label="Termination"
            value={
              result.termination === "captured" ? "Auto-captured on settle"
                : result.termination === "timeout" ? `${TRIAL_DURATION_SEC} s timeout`
                : "Stopped early"
            }
          />
        </tbody>
      </table>

      {!captured && (
        <p className="rounded-md border border-warning/40 bg-warning/5 px-3 py-2 text-xs text-foreground">
          No settled position detected. Re-record with the patient lying still in
          the Modified Thomas position for at least 2 seconds.
        </p>
      )}

      {captured && result.low_confidence && (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-foreground">
          Low-confidence capture: the pose didn&apos;t fully settle within the
          stability gate. Values reflect the final position observed, but consider
          re-recording with a longer still hold for a clinically reliable reading.
        </p>
      )}

      {result.samples.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Hip + knee angles vs time
          </p>
          <PlotlyChart
            height={240}
            data={[
              {
                type: "scatter", mode: "lines", name: "Hip (°)",
                x: tAxisSec, y: hipSeries,
                line: { color: "#EC4899", width: 2 },
                connectgaps: false,
              },
              {
                type: "scatter", mode: "lines", name: "Knee (°)",
                x: tAxisSec, y: kneeSeries,
                line: { color: "#0EA5E9", width: 2 },
                connectgaps: false,
              },
            ]}
            layout={{
              xaxis: { title: { text: "Time (s)" } },
              yaxis: {
                title: { text: "Angle (°)" },
                range: [0, 200],
              },
              shapes:
                stableStart !== null && stableEnd !== null
                  ? [{
                      type: "rect",
                      xref: "x", yref: "paper",
                      x0: stableStart, x1: stableEnd,
                      y0: 0, y1: 1,
                      fillcolor: "rgba(16,185,129,0.10)",
                      line: { width: 0 },
                      layer: "below",
                    }]
                  : [],
              legend: { orientation: "h", y: 1.18, x: 0.5, xanchor: "center" },
              margin: { l: 56, r: 24, t: 20, b: 44 },
            }}
          />
        </div>
      )}

      {result.capture_screenshot_data_url && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Capture-moment frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.capture_screenshot_data_url}
            alt={`${label} capture-moment frame`}
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
