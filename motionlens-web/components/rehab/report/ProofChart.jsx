"use client";
// ProofChart — pure inline SVG chart rendering ONE improvement graph:
// baseline (earliest assessment) → daily rehab trend → re-assessment.
// All datapoints on the same Y axis in the same units, per the PRD
// closed-loop USP.
//
// Renders as SVG + Tailwind — no <canvas>, no external chart lib —
// so it survives html2canvas-pro in the PDF export path just like
// the other components/rehab/report primitives.
//
// Layout:
//   • Header row: exercise label + delta pill ("15° → 8°, improved 7°")
//                 or "Re-assessment pending" chip
//   • SVG (viewBox scaled) with axes, gridlines, baseline marker,
//     trend line + points, re-assessment marker
//   • Caveat text below when the sharedMetric has one

import { ArrowDownRight, ArrowUpRight, Clock } from "lucide-react";

/**
 * @typedef {import("@/lib/rehab/proofArtifact").ProofArtifact} ProofArtifact
 */

/**
 * @param {{ artifact: ProofArtifact }} props
 */
export function ProofChart({ artifact }) {
  if (!artifact) return null;
  const {
    slug,
    unit,
    label,
    betterDirection,
    caveat,
    baseline,
    trend,
    reassessment,
    improvementDelta,
    assessmentsFound,
    sessionsFound,
  } = artifact;

  // Assemble the full point list for axis scaling, in time order.
  const allPoints = [];
  if (baseline) allPoints.push({ ...baseline, kind: "baseline" });
  for (const p of trend) allPoints.push({ ...p, kind: "trend" });
  if (reassessment) allPoints.push({ ...reassessment, kind: "reassessment" });

  const width = 640;
  const height = 220;
  const padTop = 24;
  const padBottom = 40;
  const padLeft = 44;
  const padRight = 20;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  // Time / value domains — handle single-point and empty cases.
  const ts = allPoints.map((p) => new Date(p.date).getTime()).filter((t) => Number.isFinite(t));
  const vs = allPoints.map((p) => p.value).filter((v) => Number.isFinite(v));
  const hasData = ts.length > 0 && vs.length > 0;
  const tMin = hasData ? Math.min(...ts) : 0;
  const tMax = hasData ? Math.max(...ts) : 1;
  const tRange = Math.max(1, tMax - tMin);
  const vMinRaw = hasData ? Math.min(...vs) : 0;
  const vMaxRaw = hasData ? Math.max(...vs) : 1;
  const vSpan = Math.max(1e-3, vMaxRaw - vMinRaw);
  const vPad = vSpan * 0.15;
  const vMin = Math.max(0, vMinRaw - vPad);
  const vMax = vMaxRaw + vPad;
  const vRange = Math.max(1e-3, vMax - vMin);

  const toX = (dateIso) => {
    if (!hasData || ts.length === 1) return padLeft + innerW / 2;
    const t = new Date(dateIso).getTime();
    return padLeft + ((t - tMin) / tRange) * innerW;
  };
  const toY = (value) => {
    if (!hasData) return padTop + innerH / 2;
    return padTop + innerH - ((value - vMin) / vRange) * innerH;
  };

  const baselineX = baseline ? toX(baseline.date) : null;
  const baselineY = baseline ? toY(baseline.value) : null;
  const reX = reassessment ? toX(reassessment.date) : null;
  const reY = reassessment ? toY(reassessment.value) : null;

  const trendPath = trend
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.date).toFixed(1)} ${toY(p.value).toFixed(1)}`)
    .join(" ");

  const improved =
    improvementDelta != null && improvementDelta > 0
      ? "improved"
      : improvementDelta != null && improvementDelta < 0
        ? "regressed"
        : null;

  return (
    <section className="rounded-card border border-border bg-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Proof of progress</p>
          <h3 className="mt-1 text-base font-semibold tracking-tight">
            {humanizeSlug(slug)} · {label}
          </h3>
          <p className="mt-1 text-[11px] text-muted">
            {sessionsFound} rehab session{sessionsFound === 1 ? "" : "s"} · {assessmentsFound} assessment{assessmentsFound === 1 ? "" : "s"}
            {" · Y-axis in "}
            {unit}
            {" · lower / higher better: "}
            {betterDirection}
          </p>
        </div>
        <DeltaPill
          baseline={baseline}
          reassessment={reassessment}
          improvementDelta={improvementDelta}
          improved={improved}
          unit={unit}
        />
      </div>

      {hasData ? (
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`${humanizeSlug(slug)} improvement chart`}
          className="mt-4 w-full"
          preserveAspectRatio="xMinYMid meet"
        >
          {/* Y gridlines */}
          {[0.25, 0.5, 0.75, 1].map((frac) => {
            const y = padTop + innerH * (1 - frac);
            const val = vMin + vRange * frac;
            return (
              <g key={frac}>
                <line
                  x1={padLeft}
                  x2={width - padRight}
                  y1={y}
                  y2={y}
                  stroke="rgba(255,255,255,0.06)"
                  strokeWidth={1}
                />
                <text
                  x={padLeft - 6}
                  y={y + 3}
                  fontSize={10}
                  fill="rgba(255,255,255,0.55)"
                  textAnchor="end"
                >
                  {val.toFixed(0)}
                </text>
              </g>
            );
          })}
          {/* X axis endpoints */}
          <text
            x={padLeft}
            y={height - 12}
            fontSize={10}
            fill="rgba(255,255,255,0.55)"
          >
            {formatDate(new Date(tMin))}
          </text>
          <text
            x={width - padRight}
            y={height - 12}
            fontSize={10}
            fill="rgba(255,255,255,0.55)"
            textAnchor="end"
          >
            {formatDate(new Date(tMax))}
          </text>

          {/* Baseline horizontal reference line (subtle) so the trend
              is comparable at a glance even if the assessment date is
              off to the far left. */}
          {baseline && (
            <line
              x1={padLeft}
              x2={width - padRight}
              y1={baselineY}
              y2={baselineY}
              stroke="rgba(148,163,184,0.35)"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
          )}

          {/* Trend line */}
          {trend.length >= 2 && (
            <path
              d={trendPath}
              fill="none"
              stroke="rgb(234,88,12)"
              strokeWidth={2.2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          {/* Trend points */}
          {trend.map((p, i) => (
            <circle
              key={p.reportId ?? i}
              cx={toX(p.date)}
              cy={toY(p.value)}
              r={4}
              fill="rgb(234,88,12)"
              stroke="white"
              strokeWidth={1.2}
            />
          ))}

          {/* Baseline marker (blue triangle) */}
          {baseline && (
            <g>
              <polygon
                points={`${baselineX - 6},${baselineY + 6} ${baselineX + 6},${baselineY + 6} ${baselineX},${baselineY - 6}`}
                fill="rgb(56,189,248)"
                stroke="white"
                strokeWidth={1.5}
              />
              <text
                x={baselineX + 8}
                y={baselineY - 8}
                fontSize={10}
                fill="rgb(56,189,248)"
                fontWeight={600}
              >
                Baseline {formatVal(baseline.value)}{unit}
              </text>
            </g>
          )}

          {/* Re-assessment marker (green diamond) OR pending marker */}
          {reassessment ? (
            <g>
              <polygon
                points={`${reX},${reY - 7} ${reX + 7},${reY} ${reX},${reY + 7} ${reX - 7},${reY}`}
                fill="rgb(34,197,94)"
                stroke="white"
                strokeWidth={1.5}
              />
              <text
                x={reX - 8}
                y={reY - 10}
                fontSize={10}
                fill="rgb(34,197,94)"
                fontWeight={600}
                textAnchor="end"
              >
                Re-assessment {formatVal(reassessment.value)}{unit}
              </text>
            </g>
          ) : (
            <g>
              <line
                x1={width - padRight - 24}
                x2={width - padRight - 4}
                y1={padTop + 12}
                y2={padTop + 12}
                stroke="rgba(148,163,184,0.7)"
                strokeDasharray="3 3"
                strokeWidth={1.5}
              />
              <text
                x={width - padRight}
                y={padTop + 14}
                fontSize={10}
                fill="rgba(148,163,184,0.85)"
                textAnchor="end"
                fontWeight={600}
              >
                Re-assessment pending
              </text>
            </g>
          )}
        </svg>
      ) : (
        <div className="mt-4 rounded-md border border-dashed border-border bg-background p-6 text-sm text-muted">
          No baseline or trend datapoints available yet.
        </div>
      )}

      {caveat && (
        <p className="mt-3 rounded-md border border-border bg-background px-3 py-2 text-[11px] text-muted">
          {caveat}
        </p>
      )}
    </section>
  );
}

function DeltaPill({ baseline, reassessment, improvementDelta, improved, unit }) {
  if (baseline && reassessment && improvementDelta != null) {
    const Arrow = improvementDelta > 0 ? ArrowUpRight : ArrowDownRight;
    const tone =
      improvementDelta > 0
        ? "bg-emerald-500/15 text-emerald-700 ring-emerald-500/40 dark:text-emerald-300"
        : improvementDelta < 0
          ? "bg-rose-500/15 text-rose-700 ring-rose-500/40 dark:text-rose-300"
          : "bg-slate-500/15 text-slate-600 ring-slate-500/40 dark:text-slate-300";
    return (
      <span
        className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ring-1 ${tone}`}
      >
        <Arrow className="h-3.5 w-3.5" />
        {formatVal(baseline.value)}{unit} → {formatVal(reassessment.value)}{unit}{" · "}
        {improved} {formatVal(Math.abs(improvementDelta))}{unit}
      </span>
    );
  }
  if (reassessment == null) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-500/15 px-3 py-1 text-xs font-semibold text-slate-600 ring-1 ring-slate-500/40 dark:text-slate-300">
        <Clock className="h-3.5 w-3.5" />
        Re-assessment pending
      </span>
    );
  }
  return null;
}

function humanizeSlug(slug) {
  return slug
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function formatVal(v) {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 100 || Number.isInteger(v)) return v.toFixed(0);
  return v.toFixed(1);
}

function formatDate(d) {
  if (!(d instanceof Date) || Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
