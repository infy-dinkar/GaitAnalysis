"use client";
// Single-Leg Stance test (Test C5) — report.
//
// Lays out up to four trial cards (L/R × eyes-open/closed) with a
// pass/fail badge against the age-matched CDC threshold, hold time,
// sway metrics, trunk lean, and a top-down sway plot of the
// hip-midpoint trajectory. Surfaces L–R asymmetry callout when the
// > 30% threshold is exceeded, plus the standard interpretation
// paragraph and unified disclaimer.

import dynamic from "next/dynamic";
import { AlertTriangle, Eye, EyeOff } from "lucide-react";
import {
  ASYMMETRY_FLAG_PCT,
  CLASSIFICATION_LABEL,
  CLASSIFICATION_TONE,
  type SessionResult,
  type TrialResult,
} from "@/lib/orthopedic/singleLegStance";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

interface Props {
  patientName: string | null;
  session: SessionResult;
  interpretation: string;
}

export function SingleLegStanceReport({ patientName, session, interpretation }: Props) {
  const t = session.trials;
  const hasEyesClosed = !!t.left_closed || !!t.right_closed;

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Single-Leg Stance test
        </h2>
        <p className="mt-2 text-sm text-muted">
          {patientName ?? "Patient"} · age{" "}
          {session.patient_age !== null ? session.patient_age : "—"}
        </p>
      </div>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Eyes-open trials</h3>
        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          <TrialCard label="Left-leg stance" trial={t.left_open} />
          <TrialCard label="Right-leg stance" trial={t.right_open} />
        </div>
        {session.eyes_open_asymmetry_pct !== null && (
          <p
            className={`mt-3 rounded-md border px-3 py-2 text-xs ${
              session.eyes_open_asymmetry_pct > ASYMMETRY_FLAG_PCT
                ? "border-warning/40 bg-warning/5 text-foreground"
                : "border-border bg-surface text-muted"
            }`}
          >
            {session.eyes_open_asymmetry_pct > ASYMMETRY_FLAG_PCT && (
              <AlertTriangle className="mr-1 inline h-3 w-3 text-warning" />
            )}
            Eyes-open L–R asymmetry: {session.eyes_open_asymmetry_pct.toFixed(0)}%
            {session.eyes_open_asymmetry_pct > ASYMMETRY_FLAG_PCT
              ? ` (> ${ASYMMETRY_FLAG_PCT}% — targeted intervention indicated).`
              : "."}
          </p>
        )}
      </section>

      {hasEyesClosed && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Eyes-closed trials</h3>
          <div className="mt-3 grid gap-4 lg:grid-cols-2">
            <TrialCard label="Left-leg stance" trial={t.left_closed} />
            <TrialCard label="Right-leg stance" trial={t.right_closed} />
          </div>
          {session.eyes_closed_asymmetry_pct !== null && (
            <p
              className={`mt-3 rounded-md border px-3 py-2 text-xs ${
                session.eyes_closed_asymmetry_pct > ASYMMETRY_FLAG_PCT
                  ? "border-warning/40 bg-warning/5 text-foreground"
                  : "border-border bg-surface text-muted"
              }`}
            >
              {session.eyes_closed_asymmetry_pct > ASYMMETRY_FLAG_PCT && (
                <AlertTriangle className="mr-1 inline h-3 w-3 text-warning" />
              )}
              Eyes-closed L–R asymmetry: {session.eyes_closed_asymmetry_pct.toFixed(0)}%
              {session.eyes_closed_asymmetry_pct > ASYMMETRY_FLAG_PCT
                ? ` (> ${ASYMMETRY_FLAG_PCT}% — targeted intervention indicated).`
                : "."}
            </p>
          )}
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
            <span className="font-medium text-foreground">Eyes-open:</span>{" "}
            age &lt; 60 ≥ 10 s, age 60–69 ≥ 7 s, age 70+ ≥ 5 s.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Eyes-closed:</span>{" "}
            roughly half the eyes-open thresholds (age &lt; 60 ≥ 5 s,
            60–69 ≥ 3.5 s, 70+ ≥ 2.5 s).
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Asymmetry:</span>{" "}
            L–R difference &gt; {ASYMMETRY_FLAG_PCT}% indicates targeted intervention.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Sway:</span>{" "}
            path length and 95% ellipse area are in pixels (relative units —
            not calibrated to cm; suitable for trend tracking within the same
            patient).
          </p>
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Per-trial card ─────────────────────────────────────────────

function TrialCard({
  label,
  trial,
}: {
  label: string;
  trial: TrialResult | undefined;
}) {
  if (!trial) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-5 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">{label}</p>
        <p className="mt-2 text-sm text-muted">No recording for this trial.</p>
      </section>
    );
  }

  const tone = CLASSIFICATION_TONE[trial.classification];
  const cls = CLASSIFICATION_LABEL[trial.classification];
  const eyesIcon = trial.condition === "eyes_open"
    ? <Eye className="h-3.5 w-3.5" />
    : <EyeOff className="h-3.5 w-3.5" />;

  return (
    <section className="space-y-3 rounded-card border border-border bg-surface/40 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
        <div>
          <p className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            {eyesIcon}
            {label}
          </p>
          <p className="mt-1 text-2xl font-semibold tabular text-foreground">
            {trial.hold_seconds.toFixed(1)} s
          </p>
        </div>
        <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${tone}`}>
          {cls}
        </span>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row label="Threshold" value={`≥ ${trial.norm_threshold_sec.toFixed(1)} s · ${trial.norm_band_label}`} />
          <Row label="Max hold (cap)" value={`${trial.hold_capped_at} s`} />
          <Row label="Sway path length" value={`${trial.sway_path_px.toFixed(0)} px (relative)`} />
          <Row label="95% sway ellipse" value={`${trial.sway_95_ellipse_px2.toFixed(0)} px² (relative)`} />
          <Row label="Mean trunk lean" value={`${Math.abs(trial.mean_trunk_lean_deg).toFixed(1)}°`} />
          <Row label="Max trunk lean" value={`${trial.max_trunk_lean_deg.toFixed(1)}°`} />
          <Row label="Termination" value={terminationLabel(trial)} />
        </tbody>
      </table>

      {!trial.norm_comparable && (
        <p className="rounded-md border border-warning/40 bg-warning/5 px-3 py-2 text-xs text-foreground">
          <AlertTriangle className="mr-1 inline h-3 w-3 text-warning" />
          Norm comparison limited — patient age was not on the profile;
          the strictest band has been applied.
        </p>
      )}

      {trial.hip_path.length > 1 && (
        <SwayPlot trial={trial} />
      )}

      {trial.screenshot_data_url && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Capture frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={trial.screenshot_data_url}
            alt={`${label} capture`}
            className="w-full rounded-md border border-border"
          />
        </div>
      )}
    </section>
  );
}

function terminationLabel(trial: TrialResult): string {
  if (trial.termination === "max_time") {
    return `Max time reached (${trial.hold_capped_at} s)`;
  }
  if (trial.termination === "stopped") return "Stopped early";
  if (trial.termination === "foot_touchdown") return "Lifted foot touched down";
  if (trial.termination === "arm_grab") return "Arm grab for support";
  if (trial.termination === "hop") return "Hop / stance-foot reposition";
  return "No leg lift detected";
}

function SwayPlot({ trial }: { trial: TrialResult }) {
  const xs = trial.hip_path.map((p) => p.x);
  const ys = trial.hip_path.map((p) => p.y);

  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
        Hip-midpoint sway trace
      </p>
      <PlotlyChart
        height={220}
        data={[
          {
            type: "scatter",
            mode: "lines",
            name: "Sway path",
            x: xs,
            y: ys,
            line: { color: "#EA580C", width: 1.4 },
          },
          {
            type: "scatter",
            mode: "markers",
            name: "Start",
            x: [xs[0]],
            y: [ys[0]],
            marker: { color: "#10B981", size: 8 },
            hoverinfo: "name",
          },
          {
            type: "scatter",
            mode: "markers",
            name: "End",
            x: [xs[xs.length - 1]],
            y: [ys[ys.length - 1]],
            marker: { color: "#DC2626", size: 8 },
            hoverinfo: "name",
          },
        ]}
        layout={{
          xaxis: { title: { text: "X (px)" }, scaleanchor: "y" },
          // Image y-axis is inverted vs Plotly default — reverse so
          // the trace reads spatially correctly relative to the camera.
          yaxis: { title: { text: "Y (px)" }, autorange: "reversed" },
          margin: { l: 56, r: 24, t: 20, b: 44 },
          legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
        }}
      />
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
