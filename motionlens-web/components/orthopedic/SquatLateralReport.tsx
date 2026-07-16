"use client";
// Squat (Lateral) — result report.
//
// Sections:
//   • Classification pill (good / moderate / poor / insufficient_data)
//   • Summary cards — the six metrics from the deepest rep
//   • Per-rep table
//   • Inline-SVG angle trace (knee + hip vs time) with bottom-frame
//     markers — PDF-safe, no Plotly, no <canvas>
//   • Deepest-rep screenshot
//   • Interpretation + caveats block (frontal / in-plane / far-side)

import { CheckCircle2, MinusCircle, XCircle } from "lucide-react";

import {
  type SquatLateralCaveat,
  type SquatLateralClassification,
  type SquatLateralRep,
  type SquatLateralResult,
} from "@/lib/orthopedic/squatLateral";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  result: SquatLateralResult;
  interpretation: string;
}

export function SquatLateralReport({
  patientName,
  patient,
  result,
  interpretation,
}: Props) {
  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle={`Squat — Lateral (${result.side === "left" ? "Left" : "Right"} leg)`}
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Squat (Lateral)
        </h2>
        <p className="mt-2 text-sm text-muted">
          Sagittal-plane squat screen · {result.side === "left" ? "Left" : "Right"} leg (near-side)
        </p>
      </div>

      <ClassificationBanner result={result} />
      <SummaryCards result={result} />
      <AngleTraceChart result={result} />
      <RepTable reps={result.reps} deepestIndex={result.deepest_rep_index} />

      {result.peak_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Deepest-rep frame
          </h3>
          <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={result.peak_screenshot_data_url}
              alt="Bottom of the deepest squat rep"
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
        </div>
      </section>

      <CaveatsBlock caveats={result.caveats} />

      <ReportDisclaimer />
    </div>
  );
}

// ─── Classification banner ─────────────────────────────────────
function ClassificationBanner({ result }: { result: SquatLateralResult }) {
  const tone = classificationTone(result.classification);
  const label = classificationLabel(result.classification);
  const durationSec = result.duration_seconds.toFixed(1);
  return (
    <div className={`rounded-card border p-5 ${tone.bg}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-subtle">
            Classification
          </p>
          <div className="mt-2 flex flex-wrap items-baseline gap-3">
            <span
              className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ${tone.pill}`}
            >
              {label}
            </span>
            <span className="text-sm text-muted">
              {result.rep_count} rep{result.rep_count === 1 ? "" : "s"} · {durationSec}s
            </span>
          </div>
          <p className="mt-3 text-xs text-muted">
            Depth (peak knee flexion) + trunk lean + heel rise drive the
            classification. Frontal-plane valgus is honestly not assessed —
            see caveats.
          </p>
        </div>
      </div>
    </div>
  );
}

function classificationLabel(c: SquatLateralClassification): string {
  if (c === "good") return "Good";
  if (c === "moderate") return "Moderate";
  if (c === "poor") return "Poor";
  return "Insufficient data";
}
function classificationTone(c: SquatLateralClassification): {
  bg: string;
  pill: string;
} {
  if (c === "good") {
    return {
      bg: "border-emerald-500/30 bg-emerald-500/5",
      pill: "bg-emerald-500/15 text-emerald-700 ring-emerald-500/40 dark:text-emerald-300",
    };
  }
  if (c === "moderate") {
    return {
      bg: "border-amber-500/40 bg-amber-500/5",
      pill: "bg-amber-500/15 text-amber-700 ring-amber-500/40 dark:text-amber-300",
    };
  }
  if (c === "poor") {
    return {
      bg: "border-rose-500/40 bg-rose-500/5",
      pill: "bg-rose-500/15 text-rose-700 ring-rose-500/40 dark:text-rose-300",
    };
  }
  return {
    bg: "border-slate-500/30 bg-slate-500/5",
    pill: "bg-slate-500/15 text-slate-700 ring-slate-500/40 dark:text-slate-300",
  };
}

// ─── Summary cards — the six metrics ───────────────────────────
function SummaryCards({ result }: { result: SquatLateralResult }) {
  const cards: Array<{
    title: string;
    primary: string;
    secondary?: string;
  }> = [
    {
      title: "Peak knee flexion (deepest rep)",
      primary: fmtDeg(result.peak_knee_flexion_deg),
      secondary:
        result.mean_peak_knee_flexion_deg !== null
          ? `Mean across ${result.rep_count} reps: ${fmtDeg(result.mean_peak_knee_flexion_deg)}`
          : undefined,
    },
    {
      title: "Peak hip flexion (deepest rep)",
      primary: fmtDeg(result.peak_hip_flexion_deg),
    },
    {
      title: "Trunk lean (from vertical)",
      primary: fmtDeg(result.trunk_lean_deg),
      secondary:
        result.mean_trunk_lean_deg !== null
          ? `Mean across reps: ${fmtDeg(result.mean_trunk_lean_deg)}`
          : undefined,
    },
    {
      title: "Hip : knee ratio",
      primary:
        result.hip_knee_ratio !== null
          ? result.hip_knee_ratio.toFixed(2)
          : "—",
      secondary:
        result.hip_knee_ratio !== null
          ? result.hip_knee_ratio >= 1.05
            ? "Hip-dominant (posterior-chain)"
            : result.hip_knee_ratio <= 0.9
              ? "Knee-dominant (quad)"
              : "Balanced"
          : undefined,
    },
    {
      title: "Heel rise",
      primary: result.any_heel_rise ? "Detected" : "None",
      secondary: result.any_heel_rise
        ? "Heel lifted on at least one rep — check ankle mobility"
        : "Heels stayed grounded on all reps",
    },
  ];

  return (
    <section className="grid gap-4 md:grid-cols-3">
      {cards.map((c) => (
        <div
          key={c.title}
          className="rounded-card border border-border bg-surface p-6"
        >
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            {c.title}
          </p>
          <p className="tabular mt-3 text-3xl font-semibold tracking-tight text-foreground md:text-4xl">
            {c.primary}
          </p>
          {c.secondary && (
            <p className="mt-3 text-xs text-muted">{c.secondary}</p>
          )}
        </div>
      ))}
    </section>
  );
}

function fmtDeg(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return `${v.toFixed(1)}°`;
}

// ─── Inline-SVG angle trace (knee + hip vs time) ───────────────
function AngleTraceChart({ result }: { result: SquatLateralResult }) {
  const t = result.angle_trace.t_ms;
  const knee = result.angle_trace.knee;
  const hip = result.angle_trace.hip;
  const bottoms = result.angle_trace.bottom_t_ms;
  if (t.length < 2) return null;

  const width = 720;
  const height = 220;
  const padLeft = 46;
  const padRight = 20;
  const padTop = 24;
  const padBottom = 34;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  const tMin = t[0];
  const tMax = t[t.length - 1];
  const tSpan = Math.max(1, tMax - tMin);
  const kneeVals = knee.filter((v): v is number => v !== null);
  const hipVals = hip.filter((v): v is number => v !== null);
  const vAll = [...kneeVals, ...hipVals];
  const yMax = vAll.length > 0 ? Math.max(...vAll, 10) : 90;
  const yPadTop = yMax * 0.1;
  const yLo = 0;
  const yHi = yMax + yPadTop;
  const yRange = Math.max(1, yHi - yLo);

  const toX = (ms: number) => padLeft + ((ms - tMin) / tSpan) * innerW;
  const toY = (v: number) =>
    padTop + innerH - ((v - yLo) / yRange) * innerH;

  const buildPath = (vals: (number | null)[]): string => {
    let path = "";
    let inSeg = false;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (v === null || !Number.isFinite(v)) {
        inSeg = false;
        continue;
      }
      const x = toX(t[i]).toFixed(1);
      const y = toY(v).toFixed(1);
      path += `${inSeg ? "L" : "M"} ${x} ${y} `;
      inSeg = true;
    }
    return path.trim();
  };

  const kneePath = buildPath(knee);
  const hipPath = buildPath(hip);

  const yTicks = niceTicks(yLo, yHi, 4);

  return (
    <section className="rounded-card border border-border bg-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Angle trace</p>
          <h3 className="mt-1 text-base font-semibold tracking-tight">
            Knee + hip flexion over time · squat bottoms marked
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
          <LegendChip color="#0EA5E9" label="Knee flexion" />
          <LegendChip color="#F97316" label="Hip flexion" />
          <LegendChip color="#94A3B8" label="Rep bottom" dashed />
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Squat angle trace — knee and hip flexion over time"
          className="block w-full"
          preserveAspectRatio="xMinYMid meet"
        >
          {/* Y gridlines + labels */}
          {yTicks.map((v) => (
            <g key={v}>
              <line
                x1={padLeft}
                x2={width - padRight}
                y1={toY(v)}
                y2={toY(v)}
                stroke="rgba(148,163,184,0.25)"
                strokeWidth={0.8}
              />
              <text
                x={padLeft - 6}
                y={toY(v) + 3.5}
                fontSize={10}
                fill="rgba(148,163,184,0.9)"
                textAnchor="end"
              >
                {v.toFixed(0)}°
              </text>
            </g>
          ))}

          {/* X axis endpoints */}
          <text
            x={padLeft}
            y={height - 14}
            fontSize={10}
            fill="rgba(148,163,184,0.9)"
          >
            0.0s
          </text>
          <text
            x={width - padRight}
            y={height - 14}
            fontSize={10}
            fill="rgba(148,163,184,0.9)"
            textAnchor="end"
          >
            {(tMax / 1000).toFixed(1)}s
          </text>

          {/* Rep-bottom vertical markers */}
          {bottoms.map((tb, i) => (
            <line
              key={`b-${i}`}
              x1={toX(tb)}
              x2={toX(tb)}
              y1={padTop}
              y2={padTop + innerH}
              stroke="rgba(148,163,184,0.7)"
              strokeWidth={1}
              strokeDasharray="3 4"
            />
          ))}

          {/* Traces */}
          {hipPath && (
            <path
              d={hipPath}
              fill="none"
              stroke="#F97316"
              strokeWidth={1.8}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          {kneePath && (
            <path
              d={kneePath}
              fill="none"
              stroke="#0EA5E9"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
        </svg>
      </div>
    </section>
  );
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const step = niceStep(span / count);
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    out.push(Math.round(v));
  }
  return out;
}
function niceStep(raw: number): number {
  const pow10 = Math.pow(10, Math.floor(Math.log10(raw)));
  const rel = raw / pow10;
  const nice = rel < 1.5 ? 1 : rel < 3 ? 2 : rel < 7 ? 5 : 10;
  return nice * pow10;
}

function LegendChip({
  color,
  label,
  dashed,
}: {
  color: string;
  label: string;
  dashed?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width="18" height="10" aria-hidden="true">
        {dashed ? (
          <line
            x1={0}
            x2={18}
            y1={5}
            y2={5}
            stroke={color}
            strokeWidth={2}
            strokeDasharray="4 3"
          />
        ) : (
          <line
            x1={0}
            x2={18}
            y1={5}
            y2={5}
            stroke={color}
            strokeWidth={2.2}
            strokeLinecap="round"
          />
        )}
      </svg>
      <span>{label}</span>
    </span>
  );
}

// ─── Per-rep table ─────────────────────────────────────────────
function RepTable({
  reps,
  deepestIndex,
}: {
  reps: SquatLateralRep[];
  deepestIndex: number;
}) {
  if (reps.length === 0) return null;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Per-rep breakdown
      </h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
              <tr>
                <th className="w-14 px-4 py-3 font-semibold">Rep</th>
                <th className="px-4 py-3 font-semibold">Knee</th>
                <th className="px-4 py-3 font-semibold">Hip</th>
                <th className="px-4 py-3 font-semibold">Trunk</th>
                <th className="px-4 py-3 font-semibold">H : K</th>
                <th className="px-4 py-3 font-semibold">Heel rise</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {reps.map((r) => {
                const isDeepest = r.rep_index === deepestIndex;
                return (
                  <tr
                    key={r.rep_index}
                    className={isDeepest ? "bg-emerald-500/5" : ""}
                  >
                    <td className="px-4 py-3 tabular font-medium text-foreground">
                      #{r.rep_index + 1}
                      {isDeepest && (
                        <span className="ml-2 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:text-emerald-300">
                          deepest
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 tabular text-foreground">
                      {fmtDeg(r.peak_knee_flexion_deg)}
                    </td>
                    <td className="px-4 py-3 tabular text-foreground">
                      {fmtDeg(r.peak_hip_flexion_deg)}
                    </td>
                    <td className="px-4 py-3 tabular text-muted">
                      {fmtDeg(r.trunk_lean_deg)}
                    </td>
                    <td className="px-4 py-3 tabular text-muted">
                      {r.hip_knee_ratio !== null
                        ? r.hip_knee_ratio.toFixed(2)
                        : "—"}
                    </td>
                    <td className="px-4 py-3">
                      {r.heel_rise ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-xs font-medium text-rose-700 dark:text-rose-300">
                          <XCircle className="h-3.5 w-3.5" />
                          Lifted
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:text-emerald-300">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          Grounded
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

// ─── Caveats block ─────────────────────────────────────────────
function CaveatsBlock({ caveats }: { caveats: SquatLateralCaveat[] }) {
  if (caveats.length === 0) return null;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Assessment caveats
      </h3>
      <div className="mt-3 space-y-2">
        {caveats.map((c) => (
          <div
            key={c.code}
            className="flex items-start gap-2 rounded-card border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-foreground"
          >
            <MinusCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
            <div>
              <p className="font-medium">{c.label}</p>
              {c.detail && (
                <p className="mt-0.5 text-muted">{c.detail}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ─── Interpretation builder (used by capture + saved-report) ───
export function buildSquatLateralInterpretation(
  result: SquatLateralResult,
): string {
  if (result.interpretation && result.interpretation.length > 0) {
    return result.interpretation;
  }
  if (result.classification === "insufficient_data") {
    return (
      "Not enough usable video to score this squat session. "
      + "Re-record side-on with a still upright stance and 3-6 slow reps."
    );
  }
  const label = classificationLabel(result.classification);
  const notes: string[] = [];
  if (result.peak_knee_flexion_deg !== null && result.peak_knee_flexion_deg < 90) {
    notes.push(
      `Peak knee flexion ${result.peak_knee_flexion_deg.toFixed(0)}° — below `
      + "parallel; check ankle mobility and glute recruitment.",
    );
  } else if (result.peak_knee_flexion_deg !== null) {
    notes.push(
      `Depth reached ${result.peak_knee_flexion_deg.toFixed(0)}° peak knee flexion.`,
    );
  }
  if (result.trunk_lean_deg !== null && result.trunk_lean_deg > 40) {
    notes.push(
      `Trunk lean ${result.trunk_lean_deg.toFixed(0)}° at bottom — consider `
      + "hip-hinge mechanics and thoracic mobility.",
    );
  }
  if (result.any_heel_rise) {
    notes.push(
      "Heel lifted on at least one rep — restricted ankle dorsiflexion or "
      + "over-forward weight bias.",
    );
  }
  if (result.hip_knee_ratio !== null) {
    if (result.hip_knee_ratio >= 1.05) {
      notes.push("Hip-dominant pattern (posterior chain).");
    } else if (result.hip_knee_ratio <= 0.9) {
      notes.push("Knee-dominant pattern (quad-forward).");
    }
  }
  if (notes.length === 0) {
    notes.push(`${result.rep_count} rep${result.rep_count === 1 ? "" : "s"} scored.`);
  }
  return `${label}. ${notes.join(" ")} Frontal-plane valgus not assessed here (see caveats).`;
}
