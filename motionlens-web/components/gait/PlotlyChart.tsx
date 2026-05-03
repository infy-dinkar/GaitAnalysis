"use client";
import dynamic from "next/dynamic";
import { useMemo } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface PlotlyChartProps {
  data: any[];                                // Plotly traces
  layout?: Record<string, any>;               // overrides; merged on top of dark defaults
  height?: number;
  className?: string;
}

const BASE_LAYOUT: Record<string, any> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#475569", family: "Inter, system-ui, sans-serif", size: 12 },
  margin: { l: 56, r: 24, t: 36, b: 44 },
  legend: {
    orientation: "h",
    y: -0.25,
    font: { color: "#475569", size: 11 },
  },
};

const BASE_AXIS = {
  gridcolor: "#E2E8F0",
  zerolinecolor: "#CBD5E1",
  linecolor: "#CBD5E1",
};

/**
 * Low-level Plotly wrapper that merges sensible dark-theme defaults with
 * caller-supplied layout overrides. Use it for charts that need shapes,
 * annotations, multi-trace overlays, or custom y-ranges.
 */
export function PlotlyChart({ data, layout, height = 320, className }: PlotlyChartProps) {
  const merged = useMemo(() => {
    const out: Record<string, any> = { ...BASE_LAYOUT, ...(layout || {}) };
    out.xaxis = { ...BASE_AXIS, ...(BASE_LAYOUT.xaxis || {}), ...(layout?.xaxis || {}) };
    out.yaxis = { ...BASE_AXIS, ...(BASE_LAYOUT.yaxis || {}), ...(layout?.yaxis || {}) };
    if (layout?.legend) {
      out.legend = { ...BASE_LAYOUT.legend, ...layout.legend };
    }
    return out;
  }, [layout]);

  return (
    <div className={className}>
      <Plot
        data={data}
        layout={merged}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height }}
      />
    </div>
  );
}

// Helper: pass-segment background shading shapes
export function passShapes(passes: { core_start_sec: number; core_end_sec: number }[]) {
  return passes.map((p) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: p.core_start_sec,
    x1: p.core_end_sec,
    y0: 0,
    y1: 1,
    fillcolor: "rgba(234,88,12,0.08)",
    line: { width: 0 },
    layer: "below",
  }));
}

// Helper: horizontal band (axhspan)
export function bandShape(low: number, high: number, fill = "rgba(161,161,170,0.10)") {
  return {
    type: "rect",
    xref: "paper",
    x0: 0,
    x1: 1,
    yref: "y",
    y0: low,
    y1: high,
    fillcolor: fill,
    line: { width: 0 },
    layer: "below",
  };
}

// Helper: horizontal reference line (axhline)
export function hLineShape(y: number, color = "#EA580C", dash = "dash") {
  return {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    yref: "y",
    y0: y,
    y1: y,
    line: { color, width: 1.4, dash },
  };
}

// Helper: vertical reference line (axvline)
export function vLineShape(x: number, color = "#A1A1AA", dash = "dash") {
  return {
    type: "line",
    xref: "x",
    x0: x,
    x1: x,
    yref: "paper",
    y0: 0,
    y1: 1,
    line: { color, width: 1, dash },
  };
}
