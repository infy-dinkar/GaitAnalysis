"use client";
// Gait Cycle % — inline SVG + Tailwind, PDF-safe.
//
// Renders two horizontal stacked bars (left / right):
//   Stance % (blue) + Swing % (green) = 100%
// with a vertical orange "Expected" marker line at x = 60 %.
// Below the chart, two summary lines and a Cadence tile row.
//
// Pure presentational — takes a pre-derived GaitCycleBlockData and
// draws it. All arithmetic is in `lib/gait/gaitCycle.ts:getGaitCycleBlock`.
// Uses inline SVG (no Plotly) so it survives the html2canvas-pro PDF
// export path that the gait report goes through
// (app/dashboard/reports/[id]/page.tsx → exportReportPdf).

import type { GaitCycleBlockData } from "@/lib/gait/gaitCycle";

const STANCE_COLOR = "#0EA5E9";       // sky-500
const SWING_COLOR = "#10B981";        // emerald-500
const EXPECTED_COLOR = "#F59E0B";     // amber-500
const GRID_COLOR = "rgba(148,163,184,0.35)";  // slate-400/35
const AXIS_TEXT_COLOR = "rgba(148,163,184,0.9)";

interface Props {
  data: GaitCycleBlockData;
}

export function GaitCyclePercentBlock({ data }: Props) {
  const hasAny =
    data.left.stancePct !== null
    || data.right.stancePct !== null
    || data.cadence !== null
    || data.variabilityPct !== null;
  if (!hasAny) return null;

  // Phase-timing (stance/swing) requires ≥ 2 gait cycles AND a
  // physiologically-plausible mask sum (see the sanity guard in
  // engines/gait_engine.py:_gait_cycle_percentages). When those
  // don't hold the engine returns null for all 5 phase keys, but
  // we still surface cadence + variability + the physics-independent
  // Y-axis so the block isn't useless — just muted.
  const phaseUnavailable =
    data.left.stancePct === null && data.right.stancePct === null;

  // Chart layout (viewBox-based so it scales at any width).
  const width = 720;
  const height = 220;
  const padLeft = 60;
  const padRight = 24;
  const padTop = 30;
  const padBottom = 50;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const rowGap = 24;
  const barH = (plotH - rowGap) / 2;

  const xAt = (pct: number) => padLeft + (pct / 100) * plotW;

  return (
    <section className="rounded-card border border-border bg-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Gait Cycle %</p>
          <h3 className="mt-1 text-base font-semibold tracking-tight">
            Stance vs Swing per side · expected marker at 60 %
          </h3>
        </div>
        <Legend />
      </div>

      {phaseUnavailable ? (
        <div className="mt-4 rounded-md border border-border bg-elevated/40 px-4 py-3 text-sm text-muted">
          Stance / swing not computed — insufficient validated
          walking data. Cadence + variability below are reported
          from the same clip.
        </div>
      ) : (
      <div className="mt-4 overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Gait cycle percentage stacked bars for left and right sides"
          className="block w-full"
          preserveAspectRatio="xMinYMid meet"
        >
          {/* X-axis gridlines every 10 % */}
          {Array.from({ length: 11 }, (_, i) => i * 10).map((pct) => {
            const x = xAt(pct);
            return (
              <g key={pct}>
                <line
                  x1={x}
                  x2={x}
                  y1={padTop}
                  y2={padTop + plotH}
                  stroke={GRID_COLOR}
                  strokeWidth={pct === 0 || pct === 100 ? 1.2 : 0.8}
                />
                <text
                  x={x}
                  y={padTop + plotH + 16}
                  fontSize={11}
                  fill={AXIS_TEXT_COLOR}
                  textAnchor="middle"
                >
                  {pct}
                </text>
              </g>
            );
          })}
          <text
            x={padLeft + plotW / 2}
            y={padTop + plotH + 34}
            fontSize={11}
            fill={AXIS_TEXT_COLOR}
            textAnchor="middle"
          >
            % of gait cycle
          </text>

          {/* Left side row */}
          <SideRow
            label="Left"
            stance={data.left.stancePct}
            swing={data.left.swingPct}
            x0={padLeft}
            y={padTop}
            plotW={plotW}
            barH={barH}
          />
          {/* Right side row */}
          <SideRow
            label="Right"
            stance={data.right.stancePct}
            swing={data.right.swingPct}
            x0={padLeft}
            y={padTop + barH + rowGap}
            plotW={plotW}
            barH={barH}
          />

          {/* Expected marker line at 60 % — drawn LAST so it sits on
              top of the bars. Full plot height, orange, with a small
              "Expected 60%" chip at the top. */}
          <line
            x1={xAt(60)}
            x2={xAt(60)}
            y1={padTop - 8}
            y2={padTop + plotH + 4}
            stroke={EXPECTED_COLOR}
            strokeWidth={2}
            strokeDasharray="4 4"
          />
          <rect
            x={xAt(60) - 40}
            y={padTop - 22}
            width={80}
            height={16}
            rx={8}
            fill={EXPECTED_COLOR}
          />
          <text
            x={xAt(60)}
            y={padTop - 10}
            fontSize={10}
            fill="white"
            fontWeight={600}
            textAnchor="middle"
          >
            Expected 60%
          </text>
        </svg>
      </div>
      )}

      {/* Text lines under chart */}
      <div className="mt-3 space-y-1 text-center text-sm text-foreground">
        {data.variabilityPct !== null && (
          <p>
            Gait cycle variability:{" "}
            <span className="tabular font-semibold">
              {data.variabilityPct.toFixed(1)}
            </span>{" "}
            %
          </p>
        )}
        {data.optimumDeviationPct !== null && (
          <p>
            Gait cycle optimum deviation:{" "}
            <span className="tabular font-semibold">
              {data.optimumDeviationPct.toFixed(1)}
            </span>{" "}
            %
          </p>
        )}
        {data.doubleSupport !== null && (
          <p className="text-xs text-muted">
            Double-support phase (both feet grounded):{" "}
            <span className="tabular">{data.doubleSupport.toFixed(1)} %</span>
          </p>
        )}
      </div>

      {/* Cadence tile row — amber "Cadence steps/min" + orange "value [expected]" */}
      <div className="mt-4 flex flex-wrap items-stretch justify-center gap-0 overflow-hidden rounded-lg">
        <div className="flex items-center bg-amber-500/15 px-4 py-2 text-sm font-semibold text-amber-800 dark:text-amber-300">
          Cadence steps/min
        </div>
        <div className="flex items-center bg-orange-500 px-4 py-2 text-sm font-semibold text-white">
          <span className="tabular">
            {data.cadence !== null ? data.cadence.toFixed(1) : "—"}
          </span>
          {data.cadenceExpected !== null && (
            <span className="ml-2 opacity-90">
              [{data.cadenceExpected}]
            </span>
          )}
        </div>
      </div>
    </section>
  );
}

// ─── One side row (stance + swing stacked bar) ─────────────────
function SideRow({
  label,
  stance,
  swing,
  x0,
  y,
  plotW,
  barH,
}: {
  label: string;
  stance: number | null;
  swing: number | null;
  x0: number;
  y: number;
  plotW: number;
  barH: number;
}) {
  const missing = stance === null && swing === null;

  return (
    <g>
      <text
        x={x0 - 10}
        y={y + barH / 2 + 4}
        fontSize={12}
        fill={AXIS_TEXT_COLOR}
        textAnchor="end"
        fontWeight={600}
      >
        {label}
      </text>

      {missing ? (
        <>
          <rect
            x={x0}
            y={y}
            width={plotW}
            height={barH}
            rx={4}
            fill="rgba(148,163,184,0.18)"
            stroke={GRID_COLOR}
            strokeWidth={1}
          />
          <text
            x={x0 + plotW / 2}
            y={y + barH / 2 + 5}
            fontSize={12}
            fill={AXIS_TEXT_COLOR}
            textAnchor="middle"
          >
            n/a
          </text>
        </>
      ) : (
        <>
          {/* Stance segment */}
          {stance !== null && stance > 0 && (
            <>
              <rect
                x={x0}
                y={y}
                width={(stance / 100) * plotW}
                height={barH}
                rx={4}
                fill={STANCE_COLOR}
              />
              {stance >= 6 && (
                <text
                  x={x0 + ((stance / 100) * plotW) / 2}
                  y={y + barH / 2 + 5}
                  fontSize={13}
                  fill="white"
                  fontWeight={700}
                  textAnchor="middle"
                >
                  {stance.toFixed(1)}%
                </text>
              )}
            </>
          )}
          {/* Swing segment */}
          {swing !== null && swing > 0 && (
            <>
              <rect
                x={x0 + ((stance ?? 0) / 100) * plotW}
                y={y}
                width={(swing / 100) * plotW}
                height={barH}
                rx={4}
                fill={SWING_COLOR}
              />
              {swing >= 6 && (
                <text
                  x={x0 + ((stance ?? 0) / 100) * plotW + ((swing / 100) * plotW) / 2}
                  y={y + barH / 2 + 5}
                  fontSize={13}
                  fill="white"
                  fontWeight={700}
                  textAnchor="middle"
                >
                  {swing.toFixed(1)}%
                </text>
              )}
            </>
          )}
        </>
      )}
    </g>
  );
}

// ─── Legend chips ──────────────────────────────────────────────
function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
      <LegendChip color={STANCE_COLOR} label="Stance" />
      <LegendChip color={SWING_COLOR} label="Swing" />
      <LegendChip color={EXPECTED_COLOR} label="Expected 60%" dashed />
    </div>
  );
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
          <rect x={0} y={1} width={18} height={8} rx={2} fill={color} />
        )}
      </svg>
      <span>{label}</span>
    </span>
  );
}
