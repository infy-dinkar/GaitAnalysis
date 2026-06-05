"use client";
// Functional Reach (C6) report.
//
// Sections:
//   • Best valid reach summary — large number + fall-risk badge
//     (cm if calibrated, "relative units" + "calibration not applied"
//     hint if not).
//   • Calibration banner — shows the pixels-per-cm + sheet orientation
//     when present, or an amber "uncalibrated — relative units only"
//     warning otherwise.
//   • Trial-by-trial validity log.
//   • Wrist displacement trace + baseline + peak markers.
//   • Annotated peak frame screenshot.
//   • Plain-language interpretation paragraph + the standard
//     disclaimer.

import dynamic from "next/dynamic";
import { AlertTriangle, CheckCircle2, Ruler } from "lucide-react";

import {
  FALL_RISK_LABEL,
  FALL_RISK_TONE,
  LOW_FALL_RISK_MIN_CM,
  MODERATE_FALL_RISK_MIN_CM,
  TRIAL_VALIDITY_LABEL,
  TRIAL_VALIDITY_TONE,
  VERY_HIGH_FALL_RISK_MAX_CM,
  type FunctionalReachResult,
  type Trial,
} from "@/lib/orthopedic/functionalReach";
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
  result: FunctionalReachResult;
  interpretation: string;
}

export function FunctionalReachReport({
  patientName,
  patient,
  result,
  interpretation,
}: Props) {
  const calibrated = result.calibration !== null;
  const sideLabel = result.side_tested === "left" ? "Left arm" : "Right arm";

  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle={`Functional Reach test · ${sideLabel}`}
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Functional Reach test
        </h2>
        <p className="mt-2 text-sm text-muted">{sideLabel}</p>
      </div>

      {/* Calibration banner */}
      <CalibrationBanner result={result} />

      {/* Best-valid summary */}
      <BestReachCard result={result} />

      {/* Trial log */}
      <TrialLog result={result} />

      {/* Trace */}
      <ReachTrace result={result} />

      {/* Peak screenshot */}
      {result.peak_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Peak frame</h3>
          <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.peak_screenshot_data_url}
              alt={`Peak reach frame for ${sideLabel}`}
              className="block w-full"
            />
          </div>
        </section>
      )}

      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
          {!calibrated && (
            <p className="mt-3 text-xs text-amber-700 dark:text-amber-400">
              ⚠ This session did not include a scale calibration. Distances
              are reported in pixels (relative units) only — fall-risk cutoffs
              don&apos;t apply. Re-run with calibration for a clinical reading.
            </p>
          )}
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Calibration banner ─────────────────────────────────────────
function CalibrationBanner({ result }: { result: FunctionalReachResult }) {
  if (result.calibration) {
    const c = result.calibration;
    return (
      <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-4 text-sm">
        <div className="flex items-start gap-2">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
          <div>
            <p className="font-medium text-foreground">Calibrated — distances reported in centimetres</p>
            <p className="mt-1 text-xs text-muted">
              Scale ={" "}
              <span className="tabular font-medium">{c.pixels_per_cm.toFixed(2)}</span> px/cm
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
            Uncalibrated — relative units only, distance not calibrated
          </p>
          <p className="mt-1 text-xs text-muted">
            No scale calibration was applied to this recording. Reach is
            reported in raw pixel units; clinical fall-risk cutoffs (≥ {LOW_FALL_RISK_MIN_CM} cm
            low risk, {MODERATE_FALL_RISK_MIN_CM}–{LOW_FALL_RISK_MIN_CM} moderate, &lt; {MODERATE_FALL_RISK_MIN_CM} high,
            &lt; {VERY_HIGH_FALL_RISK_MAX_CM} very high) are NOT applied.
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Best-valid reach summary card ───────────────────────────────
function BestReachCard({ result }: { result: FunctionalReachResult }) {
  const calibrated = result.calibration !== null;
  const validCount = result.trials.filter((t) => t.validity === "valid").length;
  const totalTrials = result.trials.length;

  const reachText = (() => {
    if (result.best_valid_reach_cm !== null) {
      return `${result.best_valid_reach_cm.toFixed(1)} cm`;
    }
    if (result.best_valid_reach_px !== null) {
      return `${result.best_valid_reach_px.toFixed(0)} px`;
    }
    return "—";
  })();

  return (
    <section className="rounded-card border border-border bg-surface p-6">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        Best valid reach
      </p>
      <div className="mt-3 flex flex-wrap items-baseline gap-3">
        <p className="tabular text-4xl font-semibold tracking-tight text-foreground md:text-5xl">
          {reachText}
        </p>
        {!calibrated && result.best_valid_reach_px !== null && (
          <span className="rounded-full bg-amber-500/10 px-2.5 py-1 text-[11px] font-medium text-amber-700 dark:text-amber-400">
            relative units
          </span>
        )}
        {result.classification && (
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-medium ${FALL_RISK_TONE[result.classification]}`}
          >
            {FALL_RISK_LABEL[result.classification]}
          </span>
        )}
      </div>
      <p className="mt-3 text-xs text-muted">
        {validCount} of {totalTrials} trial{totalTrials === 1 ? "" : "s"} valid ·{" "}
        {result.baseline_locked
          ? "baseline locked"
          : "baseline NOT locked — see interpretation"}{" "}
        · {result.duration_seconds.toFixed(1)} s recording.
      </p>
    </section>
  );
}

// ─── Trial-by-trial log ──────────────────────────────────────────
function TrialLog({ result }: { result: FunctionalReachResult }) {
  if (result.trials.length === 0) return null;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Trial log</h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-4 py-3 font-semibold">Trial</th>
              <th className="px-4 py-3 font-semibold">Time</th>
              <th className="px-4 py-3 font-semibold">Reach</th>
              <th className="px-4 py-3 font-semibold">Trunk lean</th>
              <th className="px-4 py-3 font-semibold">Heel rise</th>
              <th className="px-4 py-3 font-semibold">Foot shift</th>
              <th className="px-4 py-3 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {result.trials.map((t) => (
              <TrialRow key={t.trial_index} trial={t} calibrated={result.calibration !== null} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function TrialRow({ trial, calibrated }: { trial: Trial; calibrated: boolean }) {
  const reach = trial.reach_cm !== null
    ? `${trial.reach_cm.toFixed(1)} cm`
    : `${trial.reach_px.toFixed(0)} px`;
  const heel = trial.max_heel_drift_cm !== null
    ? `${trial.max_heel_drift_cm.toFixed(1)} cm`
    : `${trial.max_heel_drift_px.toFixed(0)} px`;
  const ankle = trial.max_ankle_drift_cm !== null
    ? `${trial.max_ankle_drift_cm.toFixed(1)} cm`
    : `${trial.max_ankle_drift_px.toFixed(0)} px`;
  const lean = trial.trunk_angle_at_peak_deg !== null
    ? `${trial.trunk_angle_at_peak_deg.toFixed(0)}°`
    : "—";
  return (
    <tr>
      <td className="px-4 py-3 font-medium text-foreground">Trial {trial.trial_index + 1}</td>
      <td className="px-4 py-3 tabular text-muted">{(trial.peak_t_ms / 1000).toFixed(1)} s</td>
      <td className="px-4 py-3 tabular text-foreground">
        {reach}
        {!calibrated && <span className="ml-1 text-[10px] text-amber-600">rel.</span>}
      </td>
      <td className="px-4 py-3 tabular text-muted">{lean}</td>
      <td className="px-4 py-3 tabular text-muted">{heel}</td>
      <td className="px-4 py-3 tabular text-muted">{ankle}</td>
      <td className="px-4 py-3">
        <span
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${TRIAL_VALIDITY_TONE[trial.validity]}`}
        >
          {TRIAL_VALIDITY_LABEL[trial.validity]}
        </span>
        {trial.invalidity_detail && (
          <p className="mt-1 text-[11px] text-muted">{trial.invalidity_detail}</p>
        )}
      </td>
    </tr>
  );
}

// ─── Reach trace chart ───────────────────────────────────────────
function ReachTrace({ result }: { result: FunctionalReachResult }) {
  if (result.samples.length === 0 || result.baseline_wrist_x_px === null) {
    return null;
  }
  const calibrated = result.calibration !== null;
  const ppc = result.calibration?.pixels_per_cm ?? 1;

  const ts = result.samples.map((s) => s.t_ms / 1000);
  const baseline = result.baseline_wrist_x_px;
  const disp = result.samples.map((s) => {
    if (s.wrist_x_px === null) return null;
    const v = Math.abs(s.wrist_x_px - baseline);
    return calibrated ? v / ppc : v;
  });

  const peakTs = result.trials.map((t) => t.peak_t_ms / 1000);
  const peakVals = result.trials.map((t) =>
    calibrated && t.reach_cm !== null ? t.reach_cm : t.reach_px,
  );
  const peakColors = result.trials.map((t) =>
    t.validity === "valid" ? "#10b981" : "#ef4444",
  );

  const unit = calibrated ? "cm" : "px";
  const thresholds = calibrated
    ? [
        { y: VERY_HIGH_FALL_RISK_MAX_CM, label: "Very high risk threshold (10 cm)", color: "rgba(239,68,68,0.5)" },
        { y: MODERATE_FALL_RISK_MIN_CM, label: "High risk threshold (15 cm)", color: "rgba(249,115,22,0.5)" },
        { y: LOW_FALL_RISK_MIN_CM, label: "Moderate risk threshold (25 cm)", color: "rgba(34,197,94,0.5)" },
      ]
    : [];

  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Wrist displacement trace</h3>
      <p className="mt-1 text-xs text-muted">
        Absolute horizontal displacement of the test-side wrist from the locked
        baseline. Peaks marked green = valid trial · red = voided trial.
      </p>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface p-4">
        <PlotlyChart
          data={[
            {
              x: ts,
              y: disp,
              type: "scatter",
              mode: "lines",
              name: `Reach (${unit})`,
              line: { color: "#3b82f6", width: 2 },
            },
            {
              x: peakTs,
              y: peakVals,
              type: "scatter",
              mode: "markers",
              name: "Trial peak",
              marker: { color: peakColors, size: 12, line: { color: "#ffffff", width: 2 } },
            },
          ]}
          layout={{
            autosize: true,
            height: 280,
            margin: { l: 50, r: 16, t: 16, b: 40 },
            xaxis: { title: { text: "Time (s)" } },
            yaxis: { title: { text: `Reach (${unit})` } },
            showlegend: true,
            legend: { orientation: "h", x: 0, y: -0.25 },
            shapes: thresholds.map((th) => ({
              type: "line",
              xref: "paper",
              yref: "y",
              x0: 0,
              x1: 1,
              y0: th.y,
              y1: th.y,
              line: { color: th.color, width: 1, dash: "dash" },
            })),
          }}
        />
      </div>
      {!calibrated && (
        <p className="mt-2 inline-flex items-center gap-1.5 text-[11px] text-amber-700 dark:text-amber-400">
          <Ruler className="h-3 w-3" />
          Y-axis is in pixels — no scale calibration was applied to this session.
        </p>
      )}
    </section>
  );
}
