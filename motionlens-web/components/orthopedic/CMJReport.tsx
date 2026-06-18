"use client";
// D4 Counter-Movement Jump report.
//
// Sections:
//   • Summary cards — best jump height (cm) + flight time (s) +
//     physics cross-check height.
//   • Calibration banner — cm if a CalibrationResult was applied,
//     otherwise amber "relative units only" warning.
//   • Trial-by-trial log with takeoff/apex/landing times, height,
//     flight time, validity.
//   • Apex screenshot.
//   • Clinical interpretation + disclaimer.

import { AlertTriangle, CheckCircle2 } from "lucide-react";

import {
  physicsJumpHeightCm,
  type CMJResult,
  type CMJTrial,
} from "@/lib/orthopedic/counterMovementJump";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  result: CMJResult;
  interpretation: string;
}

export function CMJReport({
  patientName,
  patient,
  result,
  interpretation,
}: Props) {
  const calibrated = result.calibration !== null;

  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Counter-Movement Jump"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Counter-Movement Jump
        </h2>
        <p className="mt-2 text-sm text-muted">
          Vertical jump height · Flight time · Physics cross-check
        </p>
      </div>

      <CalibrationBanner result={result} />
      <SummaryCards result={result} />
      <TrialLog result={result} calibrated={calibrated} />

      {result.peak_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Apex frame
          </h3>
          <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.peak_screenshot_data_url}
              alt="Apex of best jump"
              className="block w-full"
            />
          </div>
        </section>
      )}

      <section>
        <h3 className="text-base font-semibold tracking-tight">
          Clinical interpretation
        </h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
          {!calibrated && (
            <p className="mt-3 text-xs text-amber-700 dark:text-amber-400">
              ⚠ This session did not include a scale calibration. Jump
              heights are reported in pixels (relative units) only. Flight
              time remains valid; the physics cross-check provides a
              calibration-free centimetre estimate.
            </p>
          )}
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Calibration banner ─────────────────────────────────────────
function CalibrationBanner({ result }: { result: CMJResult }) {
  if (result.calibration) {
    const c = result.calibration;
    return (
      <div className="rounded-card border border-emerald-500/30 bg-emerald-500/5 p-4 text-sm">
        <div className="flex items-start gap-2">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
          <div>
            <p className="font-medium text-foreground">
              Calibrated — jump heights reported in centimetres
            </p>
            <p className="mt-1 text-xs text-muted">
              Scale ={" "}
              <span className="tabular font-medium">
                {c.pixels_per_cm.toFixed(2)}
              </span>{" "}
              px/cm
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
            Uncalibrated — pose-based height in pixels only
          </p>
          <p className="mt-1 text-xs text-muted">
            No scale calibration was applied. Flight time stays valid; the
            physics cross-check (h = g·t²/8) provides a calibration-free
            centimetre estimate.
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Summary cards ─────────────────────────────────────────────
function SummaryCards({ result }: { result: CMJResult }) {
  const calibrated = result.calibration !== null;
  const heightText = (() => {
    if (result.best_valid_jump_cm !== null) {
      return `${result.best_valid_jump_cm.toFixed(1)} cm`;
    }
    if (result.best_valid_jump_px !== null) {
      return `${result.best_valid_jump_px.toFixed(0)} px`;
    }
    return "—";
  })();
  const flightText =
    result.best_valid_flight_sec !== null
      ? `${result.best_valid_flight_sec.toFixed(2)} s`
      : "—";
  const physicsText =
    result.best_valid_flight_sec !== null
      ? `${physicsJumpHeightCm(result.best_valid_flight_sec).toFixed(1)} cm`
      : "—";

  const meanHeight =
    result.mean_valid_jump_cm !== null
      ? `${result.mean_valid_jump_cm.toFixed(1)} cm`
      : null;
  const meanFlight =
    result.mean_valid_flight_sec !== null
      ? `${result.mean_valid_flight_sec.toFixed(2)} s`
      : null;
  const validCount = result.trials.filter((t) => t.valid).length;
  const total = result.trials.length;

  return (
    <section className="grid gap-4 md:grid-cols-3">
      <SummaryCard
        title="Best jump height"
        primary={heightText}
        secondary={
          meanHeight ? `Mean across valid trials: ${meanHeight}` : undefined
        }
        relativeTag={!calibrated && result.best_valid_jump_px !== null}
      />
      <SummaryCard
        title="Flight time"
        primary={flightText}
        secondary={meanFlight ? `Mean: ${meanFlight}` : undefined}
      />
      <SummaryCard
        title="Physics cross-check"
        primary={physicsText}
        secondary={`h = g·t²/8 from flight time · ${validCount} of ${total} trial${total === 1 ? "" : "s"} valid`}
      />
    </section>
  );
}

function SummaryCard({
  title,
  primary,
  secondary,
  relativeTag,
}: {
  title: string;
  primary: string;
  secondary?: string;
  relativeTag?: boolean;
}) {
  return (
    <div className="rounded-card border border-border bg-surface p-6">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {title}
      </p>
      <div className="mt-3 flex flex-wrap items-baseline gap-2">
        <p className="tabular text-3xl font-semibold tracking-tight text-foreground md:text-4xl">
          {primary}
        </p>
        {relativeTag && (
          <span className="rounded-full bg-amber-500/10 px-2 py-1 text-[10px] font-medium text-amber-700 dark:text-amber-400">
            relative
          </span>
        )}
      </div>
      {secondary && (
        <p className="mt-3 text-xs text-muted">{secondary}</p>
      )}
    </div>
  );
}

// ─── Trial log ─────────────────────────────────────────────────
function TrialLog({
  result,
  calibrated,
}: {
  result: CMJResult;
  calibrated: boolean;
}) {
  if (result.trials.length === 0) return null;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Trial log</h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-4 py-3 font-semibold">Trial</th>
              <th className="px-4 py-3 font-semibold">Takeoff</th>
              <th className="px-4 py-3 font-semibold">Flight</th>
              <th className="px-4 py-3 font-semibold">Jump height</th>
              <th className="px-4 py-3 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {result.trials.map((t) => (
              <TrialRow key={t.trial_index} trial={t} calibrated={calibrated} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function TrialRow({
  trial,
  calibrated,
}: {
  trial: CMJTrial;
  calibrated: boolean;
}) {
  const height =
    trial.jump_height_cm !== null
      ? `${trial.jump_height_cm.toFixed(1)} cm`
      : `${trial.jump_height_px.toFixed(0)} px`;
  return (
    <tr>
      <td className="px-4 py-3 font-medium text-foreground">
        Trial {trial.trial_index}
      </td>
      <td className="px-4 py-3 tabular text-muted">
        {(trial.takeoff_t_ms / 1000).toFixed(2)} s
      </td>
      <td className="px-4 py-3 tabular text-muted">
        {trial.flight_time_sec.toFixed(2)} s
      </td>
      <td className="px-4 py-3 tabular text-foreground">
        {height}
        {!calibrated && (
          <span className="ml-1 text-[10px] text-amber-600">rel.</span>
        )}
        <span className="ml-2 text-[10px] text-muted">
          (physics {trial.physics_height_cm.toFixed(1)} cm)
        </span>
      </td>
      <td className="px-4 py-3">
        {trial.valid ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:text-emerald-300">
            Valid
          </span>
        ) : (
          <>
            <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-xs font-medium text-rose-700 dark:text-rose-300">
              Invalid
            </span>
            {trial.invalidation_reason && (
              <p className="mt-1 text-[11px] text-muted">
                {trial.invalidation_reason}
              </p>
            )}
          </>
        )}
      </td>
    </tr>
  );
}

export function buildCMJInterpretation(result: CMJResult): string {
  if (result.interpretation && result.interpretation.length > 0) {
    return result.interpretation;
  }
  const validCount = result.trials.filter((t) => t.valid).length;
  if (validCount === 0) {
    return "No valid jumps captured.";
  }
  return `${validCount} of ${result.trials.length} trial(s) valid.`;
}
