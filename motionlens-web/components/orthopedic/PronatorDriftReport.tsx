"use client";
// Pronator Drift (E2) report.
//
// Renders: prominent 2D-limitation banner FIRST (clinical safety —
// must read it before interpreting results), classification badge,
// per-arm drift big-numbers, asymmetry summary, time-series chart
// of left + right wrist drift vs time (with threshold reference
// lines), interpretation paragraph, and the capture-moment frame.

import dynamic from "next/dynamic";
import { AlertTriangle } from "lucide-react";
import {
  BORDERLINE_DRIFT_CM,
  POSITIVE_ASYMMETRY_RATIO,
  POSITIVE_DRIFT_CM,
  PRONATOR_CLASSIFICATION_LABEL,
  PRONATOR_CLASSIFICATION_TONE,
  STABLE_THRESHOLD_CM,
  TARGET_HOLD_DURATION_SEC,
  type PronatorDriftResult,
} from "@/lib/orthopedic/pronatorDrift";
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
  result: PronatorDriftResult;
  interpretation: string;
}

export function PronatorDriftReport({ patientName, patient, result, interpretation }: Props) {
  const tone = PRONATOR_CLASSIFICATION_TONE[result.classification];
  const label = PRONATOR_CLASSIFICATION_LABEL[result.classification];

  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle={`Pronator Drift (E2) · ${result.hold_duration_seconds.toFixed(1)} s eyes-closed hold`}
      />

      {/* CRITICAL 2D-limitation banner — first thing the operator sees. */}
      <div className="rounded-card border-2 border-amber-500/50 bg-amber-500/10 p-5">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-6 w-6 shrink-0 text-amber-600" />
          <div className="space-y-1">
            <p className="text-base font-semibold text-foreground">
              2D measurement — rotation NOT assessed
            </p>
            <p className="text-sm text-foreground/90">
              True clinical pronator drift involves the forearm rotating
              (pronating) as it drops. A monocular 2D camera CANNOT measure
              rotation — this module captures only the vertical drop.
              Clinical judgement is required; treat positive screens as
              indications for a full bedside exam, not as a diagnostic.
            </p>
          </div>
        </div>
      </div>

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Pronator Drift
        </h2>
        <div className="mt-3">
          <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${tone}`}>
            {label}
          </span>
        </div>
      </div>

      {/* Per-arm + asymmetry numbers */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ArmCard
          side="Left arm"
          driftCm={result.left.drift_cm}
          velocityCmPerSec={result.left.drift_velocity_cm_per_sec}
        />
        <ArmCard
          side="Right arm"
          driftCm={result.right.drift_cm}
          velocityCmPerSec={result.right.drift_velocity_cm_per_sec}
        />
        <BigNumber
          value={
            result.asymmetry_ratio >= 999
              ? "—"
              : `${result.asymmetry_ratio.toFixed(1)} : 1`
          }
          label="Asymmetry ratio"
          flagged={result.asymmetry_ratio > POSITIVE_ASYMMETRY_RATIO}
        />
        <BigNumber
          value={fmtCmSigned(result.asymmetry_absolute_cm)}
          label="L-R drift difference"
          flagged={false}
        />
      </section>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
        </div>
      </section>

      {/* Drift vs time chart */}
      {result.samples.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Wrist drift over the hold
          </h3>
          <p className="mt-1 text-xs text-muted">
            Positive Y = wrist dropped (below baseline). The classic positive
            pattern shows one line steadily climbing while the other stays flat.
            Threshold reference lines at +{BORDERLINE_DRIFT_CM} cm (borderline)
            and +{POSITIVE_DRIFT_CM} cm (positive screen).
          </p>
          <div className="mt-3">
            <PlotlyChart
              height={300}
              data={[
                {
                  type: "scatter", mode: "lines", name: "Left wrist",
                  x: result.t_seconds_series, y: result.left.drift_cm_series,
                  line: { color: "#3B82F6", width: 2 },
                  connectgaps: false,
                },
                {
                  type: "scatter", mode: "lines", name: "Right wrist",
                  x: result.t_seconds_series, y: result.right.drift_cm_series,
                  line: { color: "#10B981", width: 2 },
                  connectgaps: false,
                },
              ]}
              layout={{
                xaxis: { title: { text: "Time (s)" } },
                yaxis: { title: { text: "Drift from baseline (cm)" } },
                shapes: [
                  // Positive screen threshold
                  {
                    type: "line",
                    xref: "paper", yref: "y",
                    x0: 0, x1: 1,
                    y0: POSITIVE_DRIFT_CM, y1: POSITIVE_DRIFT_CM,
                    line: { color: "rgba(239,68,68,0.6)", width: 1, dash: "dash" },
                  },
                  // Borderline threshold
                  {
                    type: "line",
                    xref: "paper", yref: "y",
                    x0: 0, x1: 1,
                    y0: BORDERLINE_DRIFT_CM, y1: BORDERLINE_DRIFT_CM,
                    line: { color: "rgba(245,158,11,0.6)", width: 1, dash: "dot" },
                  },
                  // Zero baseline
                  {
                    type: "line",
                    xref: "paper", yref: "y",
                    x0: 0, x1: 1,
                    y0: 0, y1: 0,
                    line: { color: "rgba(148,163,184,0.4)", width: 1 },
                  },
                ],
                legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
                margin: { l: 56, r: 24, t: 20, b: 44 },
              }}
            />
          </div>
        </section>
      )}

      <section>
        <h3 className="text-base font-semibold tracking-tight">Reference cutoffs</h3>
        <div className="mt-3 grid gap-3 rounded-card border border-border bg-surface p-5 text-xs leading-relaxed text-muted md:grid-cols-2">
          <div>
            <p>
              <span className="font-medium text-foreground">Asymmetric drop pattern:</span>{" "}
              drop of &gt; {POSITIVE_DRIFT_CM} cm on one arm AND the other arm
              within {STABLE_THRESHOLD_CM} cm of its baseline = positive screen
              for subtle upper-motor-neuron weakness on the dropping side.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Asymmetry ratio:</span>{" "}
              max drift / min drift &gt; {POSITIVE_ASYMMETRY_RATIO} : 1 (with the
              larger arm above {BORDERLINE_DRIFT_CM} cm) = positive screen.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Hold target:</span>{" "}
              {TARGET_HOLD_DURATION_SEC} s eyes-closed hold. Earlier termination is
              flagged in the result.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Calibration:</span>{" "}
              cm values are normalised via the patient&apos;s shoulder-width
              pixels × 40 cm (adult average). Real shoulder width may differ
              by ~10 % — interpret cm values with that tolerance.
            </p>
          </div>
        </div>
      </section>

      {result.capture_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Capture frame</h3>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.capture_screenshot_data_url}
            alt="Pronator drift capture frame"
            className="mt-3 w-full rounded-md border border-border"
          />
        </section>
      )}

      <ReportDisclaimer />
    </div>
  );
}

function ArmCard({
  side, driftCm, velocityCmPerSec,
}: {
  side: string;
  driftCm: number | null;
  velocityCmPerSec: number | null;
}) {
  const dropped = driftCm !== null && driftCm > BORDERLINE_DRIFT_CM;
  const positive = driftCm !== null && driftCm > POSITIVE_DRIFT_CM;
  const toneClass = positive
    ? "border-red-500/40 bg-red-500/5"
    : dropped
      ? "border-amber-500/40 bg-amber-500/5"
      : "border-border bg-surface";
  return (
    <div className={`rounded-card border p-5 ${toneClass}`}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
        {side}
      </p>
      <p className="mt-2 tabular text-3xl font-semibold text-foreground">
        {fmtCmSigned(driftCm)}
      </p>
      <p className="mt-1 text-[11px] text-muted">
        Velocity: {velocityCmPerSec !== null
          ? `${velocityCmPerSec.toFixed(2)} cm/s`
          : "—"}
      </p>
    </div>
  );
}

function BigNumber({
  value, label, flagged,
}: {
  value: string;
  label: string;
  flagged: boolean;
}) {
  return (
    <div className={`rounded-card border p-5 ${flagged ? "border-red-500/40 bg-red-500/5" : "border-border bg-surface"}`}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
        {label}
      </p>
      <p className="mt-2 tabular text-3xl font-semibold text-foreground">
        {value}
      </p>
    </div>
  );
}

function fmtCmSigned(v: number | null): string {
  if (v === null) return "—";
  if (v > 0.05) return `↓ ${v.toFixed(1)} cm`;
  if (v < -0.05) return `↑ ${Math.abs(v).toFixed(1)} cm`;
  return "0.0 cm";
}
