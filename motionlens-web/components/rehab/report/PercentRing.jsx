"use client";
// PercentRing — inline SVG donut showing 0..100 %. PDF-safe under
// html2canvas-pro (SVG renders cleanly). No external chart lib.

/**
 * @typedef {object} PercentRingProps
 * @property {string} label            e.g. "In zone", "Accuracy"
 * @property {number} value            0..100
 * @property {"emerald"|"amber"|"sky"|"accent"|"rose"} [tone]
 * @property {string} [subtext]        e.g. "12.4 s / 30 s"
 * @property {number} [size]           px, defaults to 92
 * @property {string} [centerLabel]    Override the center text; defaults to `${pct}%`
 */

const TONE = {
  emerald: "rgb(34, 197, 94)",
  amber: "rgb(245, 158, 11)",
  sky: "rgb(14, 165, 233)",
  accent: "rgb(234, 88, 12)",
  rose: "rgb(244, 63, 94)",
};

/**
 * @param {PercentRingProps} props
 */
export function PercentRing({
  label,
  value,
  tone = "emerald",
  subtext,
  size = 92,
  centerLabel,
}) {
  const clamped = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  const stroke = 8;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dashOffset = c * (1 - clamped / 100);
  const color = TONE[tone] ?? TONE.emerald;
  const displayLabel = centerLabel ?? `${Math.round(clamped)}%`;
  return (
    <div className="flex items-center gap-4 rounded-card border border-border bg-surface p-4">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        aria-label={`${label} ${Math.round(clamped)} percent`}
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.10)"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={dashOffset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <text
          x="50%"
          y="50%"
          dominantBaseline="central"
          textAnchor="middle"
          fill="currentColor"
          className="tabular fill-foreground"
          style={{ fontSize: size * 0.22, fontWeight: 700 }}
        >
          {displayLabel}
        </text>
      </svg>
      <div className="min-w-0 flex-1">
        <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
          {label}
        </p>
        {subtext && (
          <p className="mt-1 tabular text-sm text-foreground">{subtext}</p>
        )}
      </div>
    </div>
  );
}
