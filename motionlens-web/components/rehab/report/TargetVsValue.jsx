"use client";
// TargetVsValue — mini SVG number-line showing a shaded target band
// with the patient's value plotted as a marker. Replaces the plain
// "In band / Outside band" text pill used on the clinical metric
// card. Reusable across every mechanic that saves signal +
// target_band. Pure inline SVG — PDF-safe under html2canvas-pro.

/**
 * @typedef {object} TargetVsValueProps
 * @property {string} label          e.g. "Knee angle (°)"
 * @property {number | null} value   the peak / achieved value; null → "Not captured"
 * @property {string} [unit]         e.g. "°", "%"
 * @property {{min: number, max: number}} [band]  Target band; if omitted, marker plotted on a bare axis
 * @property {number} [axisMin]      Optional axis floor; defaults to smart pick around value+band
 * @property {number} [axisMax]      Optional axis ceiling
 * @property {string} [caption]      Optional subtext under the axis
 */

/**
 * @param {TargetVsValueProps} props
 */
export function TargetVsValue({
  label,
  value,
  unit = "",
  band,
  axisMin,
  axisMax,
  caption,
}) {
  const width = 320;
  const height = 42;
  const trackY = 24;
  const trackH = 8;
  const padX = 16;
  const usableW = width - padX * 2;

  const min =
    axisMin != null
      ? axisMin
      : Math.min(
          Number.isFinite(value) ? value : Infinity,
          band?.min ?? Infinity,
        ) - 5;
  const max =
    axisMax != null
      ? axisMax
      : Math.max(
          Number.isFinite(value) ? value : -Infinity,
          band?.max ?? -Infinity,
        ) + 5;
  const range = Math.max(1e-3, max - min);
  const toX = (v) => padX + ((v - min) / range) * usableW;

  const hasValue = Number.isFinite(value);
  const inBand =
    hasValue
      && band != null
      && value >= band.min
      && value <= band.max;
  const markerColor = !hasValue
    ? "rgba(255,255,255,0.25)"
    : band == null
      ? "rgb(56,189,248)"
      : inBand
        ? "rgb(34,197,94)"
        : "rgb(245,158,11)";
  const bandColor = "rgba(34,197,94,0.28)";
  const bandStroke = "rgba(34,197,94,0.55)";
  const markerX = toX(Number.isFinite(value) ? value : min);

  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="flex items-baseline justify-between">
        <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
          {label}
        </p>
        <p className="tabular text-xs text-muted">
          {Number.isFinite(value) ? formatVal(value) : "—"}
          {unit ? ` ${unit}` : ""}
          {band && (
            <span className="ml-2 text-subtle">
              target {formatVal(band.min)}–{formatVal(band.max)}
              {unit ? ` ${unit}` : ""}
            </span>
          )}
        </p>
      </div>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        className="mt-2"
        aria-hidden="true"
      >
        {/* Axis */}
        <line
          x1={padX}
          x2={width - padX}
          y1={trackY + trackH / 2}
          y2={trackY + trackH / 2}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={1}
        />
        {/* Target band */}
        {band && (
          <rect
            x={toX(band.min)}
            y={trackY}
            width={Math.max(0, toX(band.max) - toX(band.min))}
            height={trackH}
            rx={trackH / 2}
            ry={trackH / 2}
            fill={bandColor}
            stroke={bandStroke}
            strokeWidth={1}
          />
        )}
        {/* Marker */}
        <circle
          cx={markerX}
          cy={trackY + trackH / 2}
          r={6}
          fill={markerColor}
          stroke="white"
          strokeWidth={1.5}
        />
        {/* Axis extents */}
        <text
          x={padX}
          y={height - 4}
          fontSize={9}
          fill="rgba(255,255,255,0.55)"
        >
          {formatVal(min)}
        </text>
        <text
          x={width - padX}
          y={height - 4}
          fontSize={9}
          fill="rgba(255,255,255,0.55)"
          textAnchor="end"
        >
          {formatVal(max)}
        </text>
      </svg>
      {(caption || band) && (
        <p className="mt-1 text-[11px] text-muted">
          {caption
            ? caption
            : !hasValue
              ? "Not captured."
              : inBand
                ? "In target band."
                : "Outside target band."}
        </p>
      )}
    </div>
  );
}

function formatVal(v) {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 100 || Number.isInteger(v)) return v.toFixed(0);
  return v.toFixed(1);
}
