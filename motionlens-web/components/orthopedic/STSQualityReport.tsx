"use client";
// Sit-to-Stand QUALITY (B4) report.
//
// Single trial, 3 reps, no L/R split — different shape from
// SLR/AKE/MTT/FL. Renders: overall classification, big-number
// medians (sit-to-stand ms, pause, stand-to-sit, trunk lean at
// seat-off, knee at seat-off, smoothness), per-rep table, hip-Y vs
// time chart with phase-boundary markers, capture-moment screenshot.

import dynamic from "next/dynamic";
import {
  SMOOTHNESS_SMOOTH_MIN,
  STS_QUALITY_CLASSIFICATION_LABEL,
  STS_QUALITY_CLASSIFICATION_TONE,
  TARGET_REP_COUNT,
  TRUNK_LEAN_EFFICIENT_MAX_DEG,
  TRUNK_LEAN_EFFICIENT_MIN_DEG,
  TRUNK_LEAN_MOMENTUM_DEG,
  type STSQualityResult,
} from "@/lib/orthopedic/stsQuality";
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
  result: STSQualityResult;
  interpretation: string;
}

export function STSQualityReport({ patientName, patient, result, interpretation }: Props) {
  const tone = STS_QUALITY_CLASSIFICATION_TONE[result.classification];
  const label = STS_QUALITY_CLASSIFICATION_LABEL[result.classification];

  const tAxisSec = result.samples.map((s) => s.t_ms / 1000);
  const hipYSeries = result.samples.map((s) => s.hip_y);

  // Phase boundary markers from the per-rep timing.
  const phaseLines: Array<{ t: number; color: string; label: string }> = [];
  for (const r of result.reps) {
    if (r.seat_off_t_ms !== null) {
      phaseLines.push({
        t: r.seat_off_t_ms / 1000,
        color: "rgba(16,185,129,0.5)",
        label: `seat-off ${r.rep_index}`,
      });
    }
    phaseLines.push({
      t: r.top_of_stand_t_ms / 1000,
      color: "rgba(59,130,246,0.5)",
      label: `top ${r.rep_index}`,
    });
    if (r.re_seated_t_ms !== null) {
      phaseLines.push({
        t: r.re_seated_t_ms / 1000,
        color: "rgba(244,63,94,0.5)",
        label: `re-seated ${r.rep_index}`,
      });
    }
  }

  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle={`Sit-to-Stand Quality (B4) · ${TARGET_REP_COUNT} reps, median across reps`}
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Sit-to-Stand Quality
        </h2>
        <div className="mt-3">
          <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${tone}`}>
            {label}
          </span>
        </div>
      </div>

      {/* Median readouts */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <BigNumber
          value={fmtMs(result.median_sit_to_stand_ms)}
          unit=""
          label="Median sit-to-stand"
        />
        <BigNumber
          value={fmtMs(result.median_pause_ms)}
          unit=""
          label="Median pause at top"
        />
        <BigNumber
          value={fmtMs(result.median_stand_to_sit_ms)}
          unit=""
          label="Median stand-to-sit"
        />
        <BigNumber
          value={fmtDeg(result.median_trunk_lean_deg)}
          unit=""
          label="Median trunk lean at seat-off"
        />
        <BigNumber
          value={fmtDeg(result.median_knee_angle_deg)}
          unit=""
          label="Median knee angle at seat-off"
        />
        <BigNumber
          value={result.median_smoothness_score !== null
            ? result.median_smoothness_score.toFixed(2)
            : "—"}
          unit=""
          label="Median smoothness score"
        />
      </section>

      {/* Flag chips */}
      <section className="flex flex-wrap gap-2">
        <FlagChip
          ok={!result.any_hand_use}
          label={
            result.any_hand_use
              ? `Hand use detected (${result.hand_use_count}/${result.reps.length})`
              : "No hand use"
          }
        />
        <FlagChip
          ok={
            result.median_trunk_lean_deg === null ||
            (result.median_trunk_lean_deg >= TRUNK_LEAN_EFFICIENT_MIN_DEG &&
              result.median_trunk_lean_deg <= TRUNK_LEAN_MOMENTUM_DEG)
          }
          label={
            result.median_trunk_lean_deg === null
              ? "Trunk lean —"
              : result.median_trunk_lean_deg > TRUNK_LEAN_MOMENTUM_DEG
                ? `Trunk lean > ${TRUNK_LEAN_MOMENTUM_DEG}°`
                : `Trunk lean ${result.median_trunk_lean_deg.toFixed(0)}°`
          }
        />
        <FlagChip
          ok={
            result.median_smoothness_score === null ||
            result.median_smoothness_score >= SMOOTHNESS_SMOOTH_MIN
          }
          label={
            result.median_smoothness_score === null
              ? "Smoothness —"
              : result.median_smoothness_score >= SMOOTHNESS_SMOOTH_MIN
                ? "Smooth rise"
                : "Jerky / hesitant rise"
          }
        />
      </section>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
        </div>
      </section>

      {/* Per-rep table */}
      {result.reps.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Per-rep metrics</h3>
          <div className="mt-3 overflow-x-auto rounded-card border border-border bg-surface">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border bg-elevated text-[10px] uppercase tracking-[0.12em] text-subtle">
                <tr>
                  <th className="px-3 py-2 font-medium">Rep</th>
                  <th className="px-3 py-2 text-right font-medium">Sit→stand (ms)</th>
                  <th className="px-3 py-2 text-right font-medium">Pause (ms)</th>
                  <th className="px-3 py-2 text-right font-medium">Stand→sit (ms)</th>
                  <th className="px-3 py-2 text-right font-medium">Trunk @SO (°)</th>
                  <th className="px-3 py-2 text-right font-medium">Knee @SO (°)</th>
                  <th className="px-3 py-2 text-right font-medium">Smooth</th>
                  <th className="px-3 py-2 text-right font-medium">Hands</th>
                </tr>
              </thead>
              <tbody>
                {result.reps.map((r) => (
                  <tr key={r.rep_index} className="border-b border-border/50 last:border-b-0">
                    <td className="px-3 py-2 text-foreground">{r.rep_index}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{fmtMs(r.sit_to_stand_ms)}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{fmtMs(r.pause_ms)}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{fmtMs(r.stand_to_sit_ms)}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{fmtDeg(r.trunk_lean_at_seat_off_deg)}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{fmtDeg(r.knee_angle_at_seat_off_deg)}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">
                      {r.smoothness_score !== null ? r.smoothness_score.toFixed(2) : "—"}
                    </td>
                    <td className={`px-3 py-2 text-right tabular ${r.hand_use_detected ? "text-red-600" : "text-emerald-600"}`}>
                      {r.hand_use_detected ? "✗ push-off" : "✓ crossed"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Hip Y vs time chart with phase markers */}
      {result.samples.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Hip vertical trajectory (image y-down)
          </h3>
          <p className="mt-1 text-xs text-muted">
            Standing moments = local minima in hip Y. Vertical markers show seat-off (green),
            top of stand (blue), and re-seated (red) for each rep.
          </p>
          <div className="mt-3">
            <PlotlyChart
              height={260}
              data={[
                {
                  type: "scatter", mode: "lines", name: "Hip Y (px)",
                  x: tAxisSec, y: hipYSeries,
                  line: { color: "#3B82F6", width: 2 },
                  connectgaps: false,
                },
              ]}
              layout={{
                xaxis: { title: { text: "Time (s)" } },
                yaxis: {
                  title: { text: "Hip Y (px, image y-down)" },
                  autorange: "reversed", // so "standing" (low y) is up
                },
                shapes: phaseLines.map((p) => ({
                  type: "line",
                  xref: "x", yref: "paper",
                  x0: p.t, x1: p.t,
                  y0: 0, y1: 1,
                  line: { color: p.color, width: 1, dash: "dot" },
                  layer: "below",
                })),
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
              <span className="font-medium text-foreground">Trunk lean at seat-off:</span>{" "}
              {TRUNK_LEAN_EFFICIENT_MIN_DEG}–{TRUNK_LEAN_EFFICIENT_MAX_DEG}° = efficient pattern.
              &gt; {TRUNK_LEAN_MOMENTUM_DEG}° = momentum-dependent (hip/quad weakness).
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Smoothness score:</span>{" "}
              ≥ {SMOOTHNESS_SMOOTH_MIN} = smooth rise. Below = hesitant/jerky.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Hand use:</span>{" "}
              auto-detected when the camera-side wrist drops more than 10% of leg
              length below the shoulder line during the rising phase. Push-off
              indicates significant lower-extremity weakness.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Chair seat height:</span>{" "}
              {result.chair_seat_height_cm !== null ? `${result.chair_seat_height_cm} cm` : "not recorded"}.
              Recorded for context — affects mechanics.
            </p>
          </div>
        </div>
      </section>

      {result.worst_rep_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Capture-moment frame</h3>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.worst_rep_screenshot_data_url}
            alt="Sit-to-stand quality capture frame"
            className="mt-3 w-full rounded-md border border-border"
          />
        </section>
      )}

      <ReportDisclaimer />
    </div>
  );
}

function BigNumber({ value, unit, label }: { value: string; unit: string; label: string }) {
  return (
    <div className="rounded-card border border-border bg-surface p-5">
      <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">{label}</p>
      <p className="mt-2 tabular text-3xl font-semibold text-foreground">
        {value}
        {unit && <span className="text-base text-muted"> {unit}</span>}
      </p>
    </div>
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

function fmtMs(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(0)} ms`;
}

function fmtDeg(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(1)}°`;
}
