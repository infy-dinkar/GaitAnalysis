"use client";
// Timed Up and Go (TUG) — full report.
//
// Renders all 10 sections per the MotionLens spec, in order:
//   1. Patient header
//   2. Primary outcome card (total time + classification)
//   3. Phase decomposition (horizontal stacked bar)
//   4. Walking metrics table (Walk-Out vs Walk-Back)
//   5. Turn analysis + flags
//   6. Sub-phase timings table (% of total)
//   7. Annotated key frames (5 screenshots)
//   8. Plain-language interpretation
//   9. Age-matched norms comparison
//  10. Unified disclaimer

import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

import { PatientHeader } from "@/components/dashboard/PatientHeader";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import type { PatientDTO } from "@/lib/patients";
import {
  TUG_CLASSIFICATION_LABEL,
  TUG_CLASSIFICATION_TONE,
  TUG_KEY_FRAME_LABEL,
  TUG_PHASE_COLOR,
  TUG_PHASE_LABEL,
  type TUGFlag,
  type TUGPhase,
  type TUGPhaseName,
  type TUGResult,
} from "@/lib/orthopedic/tug";

interface Props {
  result: TUGResult;
  patient?: PatientDTO | null;
  patientName?: string | null;
}

const ORDERED_PHASES: TUGPhaseName[] = [
  "sit_to_stand",
  "walk_out",
  "turn",
  "walk_back",
  "stand_to_sit",
];

function phaseOf(result: TUGResult, name: TUGPhaseName): TUGPhase {
  return result[name];
}

function symmetryPct(a: number | null, b: number | null): number | null {
  if (a === null || b === null || a + b === 0) return null;
  return Math.round((1 - Math.abs(a - b) / ((a + b) / 2)) * 100);
}

export function TUGReport({ result, patient, patientName }: Props) {
  const tone = TUG_CLASSIFICATION_TONE[result.classification];
  const classLabel = TUG_CLASSIFICATION_LABEL[result.classification];

  const phases = ORDERED_PHASES.map((p) => phaseOf(result, p));
  const total = result.total_time_sec || phases.reduce((s, p) => s + p.duration_sec, 0);

  return (
    <div className="space-y-10">
      {/* 1. Patient header */}
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Timed Up and Go (TUG) · 3 m walk + turn"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Timed Up and Go (TUG)
        </h2>
      </div>

      {/* 2. Primary outcome */}
      <section className={`rounded-card border-0 p-6 ${tone}`}>
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] opacity-80">
              Total TUG time
            </p>
            <p className="mt-1 tabular text-5xl font-semibold tracking-tight">
              {result.total_time_sec.toFixed(1)} s
            </p>
            <p className="mt-1 text-sm font-medium opacity-90">{classLabel}</p>
          </div>
          {result.age_norm_threshold_sec !== null && (
            <div className="text-right">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] opacity-80">
                Age-matched norm
              </p>
              <p className="mt-1 tabular text-2xl font-semibold">
                ≤ {result.age_norm_threshold_sec.toFixed(1)} s
              </p>
              <p className="mt-1 inline-flex items-center gap-1 text-sm opacity-90">
                {result.age_norm_passed ? (
                  <>
                    <CheckCircle2 className="h-4 w-4" />
                    Within norm
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4" />
                    Above norm
                  </>
                )}
              </p>
            </div>
          )}
        </div>
      </section>

      {/* 3. Phase decomposition bar */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Phase decomposition</h3>
        <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface p-5">
          <div className="flex h-10 overflow-hidden rounded-md">
            {phases.map((p) => {
              const widthPct = total > 0 ? (p.duration_sec / total) * 100 : 20;
              return (
                <div
                  key={p.phase}
                  className="flex items-center justify-center text-[10px] font-semibold uppercase tracking-[0.08em] text-white"
                  style={{
                    width: `${widthPct}%`,
                    backgroundColor: TUG_PHASE_COLOR[p.phase],
                    minWidth: 30,
                  }}
                  title={`${TUG_PHASE_LABEL[p.phase]} — ${p.duration_sec.toFixed(2)} s`}
                >
                  {widthPct > 12 ? `${p.duration_sec.toFixed(1)}s` : ""}
                </div>
              );
            })}
          </div>
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs">
            {phases.map((p) => (
              <span key={p.phase} className="inline-flex items-center gap-1.5 text-muted">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: TUG_PHASE_COLOR[p.phase] }}
                />
                {TUG_PHASE_LABEL[p.phase]}
              </span>
            ))}
          </div>
          <p className="mt-3 text-right text-sm font-medium tabular text-foreground">
            Total: {total.toFixed(2)} s
          </p>
        </div>
      </section>

      {/* 4. Walking metrics table */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Walking metrics</h3>
        <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
              <tr>
                <th className="px-5 py-3 font-medium">Metric</th>
                <th className="px-5 py-3 text-right font-medium">Walk-out</th>
                <th className="px-5 py-3 text-right font-medium">Walk-back</th>
              </tr>
            </thead>
            <tbody>
              <WalkRow label="Duration" v1={result.walk_out.duration_sec} v2={result.walk_back.duration_sec} unit=" s" digits={2} />
              <WalkRow label="Step count" v1={result.walk_out.step_count} v2={result.walk_back.step_count} unit="" digits={0} />
              <WalkRow label="Walking speed" v1={result.walk_out.walking_speed_mps} v2={result.walk_back.walking_speed_mps} unit=" m/s" digits={2} />
              <WalkRow label="Cadence" v1={result.walk_out.cadence_steps_per_min} v2={result.walk_back.cadence_steps_per_min} unit=" steps/min" digits={1} />
              <WalkRow label="Step length (L)" v1={result.walk_out.step_length_l_px} v2={result.walk_back.step_length_l_px} unit=" px" digits={0} />
              <WalkRow label="Step length (R)" v1={result.walk_out.step_length_r_px} v2={result.walk_back.step_length_r_px} unit=" px" digits={0} />
              <SymmetryRow
                label="L-R step-length symmetry"
                v1={symmetryPct(result.walk_out.step_length_l_px, result.walk_out.step_length_r_px)}
                v2={symmetryPct(result.walk_back.step_length_l_px, result.walk_back.step_length_r_px)}
              />
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">
          Step lengths reported in pixels (relative units — no calibration object).
          Walking speed assumes a 3.0 m operator-confirmed path.
        </p>
      </section>

      {/* 5. Turn analysis + flags */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Turn analysis</h3>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Turn duration
            </p>
            <p className="mt-1 tabular text-2xl font-semibold text-foreground">
              {result.turn.duration_sec.toFixed(2)} s
            </p>
            <p className="mt-1 text-xs text-muted">
              Threshold for balance impairment flag: &gt; 4 s.
            </p>
          </div>
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Turn step count
            </p>
            <p className="mt-1 tabular text-2xl font-semibold text-foreground">
              {result.turn.step_count ?? "—"}
            </p>
            <p className="mt-1 text-xs text-muted">
              Threshold for turning instability flag: &gt; 5 shuffle steps.
            </p>
          </div>
        </div>
        {result.flags.length > 0 && (
          <div className="mt-3 space-y-2">
            {result.flags.map((f, i) => (
              <FlagBanner key={i} flag={f} />
            ))}
          </div>
        )}
      </section>

      {/* 6. Sub-phase timings */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Sub-phase timings</h3>
        <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
              <tr>
                <th className="px-5 py-3 font-medium">Phase</th>
                <th className="px-5 py-3 text-right font-medium">Duration</th>
                <th className="px-5 py-3 text-right font-medium">% of total</th>
              </tr>
            </thead>
            <tbody>
              {phases.map((p) => (
                <tr key={p.phase} className="border-b border-border/50 last:border-b-0">
                  <td className="px-5 py-3 text-foreground">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: TUG_PHASE_COLOR[p.phase] }}
                      />
                      {TUG_PHASE_LABEL[p.phase]}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-right tabular text-foreground">
                    {p.duration_sec.toFixed(2)} s
                  </td>
                  <td className="px-5 py-3 text-right tabular text-muted">
                    {total > 0 ? ((p.duration_sec / total) * 100).toFixed(1) : "—"}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 7. Annotated key frames */}
      {result.key_frames.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Key frames</h3>
          <div className="mt-3 grid gap-3 md:grid-cols-3 lg:grid-cols-5">
            {result.key_frames.map((kf) => (
              <figure key={kf.frame_index} className="overflow-hidden rounded-card border border-border bg-surface">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={kf.image_data_url}
                  alt={TUG_KEY_FRAME_LABEL[kf.label]}
                  className="block w-full"
                />
                <figcaption className="border-t border-border px-3 py-2 text-[11px] text-muted">
                  {TUG_KEY_FRAME_LABEL[kf.label]}
                </figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}

      {/* 8. Interpretation */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {result.interpretation}
        </div>
      </section>

      {/* 9. Age-matched norms */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Reference cutoffs</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-xs leading-relaxed text-muted">
          <p>
            <span className="font-medium text-foreground">Total time:</span>{" "}
            &lt; 10 s normal · 10-13.5 s mild fall risk · 13.5-20 s elevated
            fall risk · &gt; 20 s significant impairment.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Age-matched norms:</span>{" "}
            under 60 ≤ 10 s · 60-69 ≤ 12 s · 70-79 ≤ 13 s · 80+ ≤ 14 s.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Independent flags:</span>{" "}
            turn &gt; 4 s = balance impairment · &gt; 5 shuffle steps on the
            turn = turning instability — raised regardless of total time.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Step length:</span>{" "}
            reported in pixels (relative). Walking speed assumes 3.0 m
            operator-confirmed path.
          </p>
        </div>
      </section>

      {/* 10. Disclaimer */}
      <ReportDisclaimer />
    </div>
  );
}

// ─── Helpers ────────────────────────────────────────────────────

function WalkRow({
  label,
  v1,
  v2,
  unit,
  digits,
}: {
  label: string;
  v1: number | null;
  v2: number | null;
  unit: string;
  digits: number;
}) {
  const fmt = (n: number | null) =>
    n === null || !isFinite(n) ? "—" : `${n.toFixed(digits)}${unit}`;
  return (
    <tr className="border-b border-border/50 last:border-b-0">
      <td className="px-5 py-3 text-foreground">{label}</td>
      <td className="px-5 py-3 text-right tabular text-foreground">{fmt(v1)}</td>
      <td className="px-5 py-3 text-right tabular text-foreground">{fmt(v2)}</td>
    </tr>
  );
}

function SymmetryRow({
  label,
  v1,
  v2,
}: {
  label: string;
  v1: number | null;
  v2: number | null;
}) {
  const fmt = (n: number | null) =>
    n === null ? "—" : `${n}%`;
  return (
    <tr className="border-b border-border/50 last:border-b-0">
      <td className="px-5 py-3 text-foreground">{label}</td>
      <td className="px-5 py-3 text-right tabular text-foreground">{fmt(v1)}</td>
      <td className="px-5 py-3 text-right tabular text-foreground">{fmt(v2)}</td>
    </tr>
  );
}

function FlagBanner({ flag }: { flag: TUGFlag }) {
  const tone =
    flag.severity === "concern"
      ? "border-red-500/40 bg-red-500/5 text-foreground"
      : flag.severity === "warning"
        ? "border-warning/40 bg-warning/5 text-foreground"
        : "border-border bg-elevated text-muted";
  const Icon = flag.severity === "concern" || flag.severity === "warning"
    ? AlertTriangle
    : CheckCircle2;
  return (
    <div className={`flex items-start gap-3 rounded-md border p-3 text-xs ${tone}`}>
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span>{flag.message}</span>
    </div>
  );
}
