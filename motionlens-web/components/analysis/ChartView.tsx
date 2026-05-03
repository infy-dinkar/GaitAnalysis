"use client";
import dynamic from "next/dynamic";
import { useMemo } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export interface ChartSeries {
  name: string;
  x: number[];
  y: number[];
  color?: string;
}

interface ChartViewProps {
  title?: string;
  yLabel?: string;
  xLabel?: string;
  series: ChartSeries[];
  band?: { from: number; to: number; label?: string };
  height?: number;
}

const PALETTE = ["#EA580C", "#2563EB", "#D97706", "#DC2626"];

export function ChartView({
  title,
  yLabel = "Angle (°)",
  xLabel = "Time (s)",
  series,
  band,
  height = 320,
}: ChartViewProps) {
  const data = useMemo(
    () =>
      series.map((s, i) => ({
        type: "scatter" as const,
        mode: "lines" as const,
        name: s.name,
        x: s.x,
        y: s.y,
        line: { color: s.color ?? PALETTE[i % PALETTE.length], width: 2 },
      })),
    [series],
  );

  const layout = useMemo(
    () => ({
      title: title ? { text: title, font: { color: "#0F172A", size: 14 } } : undefined,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#475569", family: "Inter, system-ui, sans-serif", size: 12 },
      margin: { l: 56, r: 24, t: title ? 36 : 16, b: 44 },
      xaxis: {
        title: { text: xLabel },
        gridcolor: "#E2E8F0",
        zerolinecolor: "#CBD5E1",
        linecolor: "#CBD5E1",
      },
      yaxis: {
        title: { text: yLabel },
        gridcolor: "#E2E8F0",
        zerolinecolor: "#CBD5E1",
        linecolor: "#CBD5E1",
      },
      legend: {
        orientation: "h" as const,
        y: -0.25,
        font: { color: "#475569", size: 11 },
      },
      shapes: band
        ? [
            {
              type: "rect" as const,
              xref: "paper" as const,
              x0: 0,
              x1: 1,
              yref: "y" as const,
              y0: band.from,
              y1: band.to,
              fillcolor: "rgba(234,88,12,0.10)",
              line: { width: 0 },
              layer: "below" as const,
            },
          ]
        : [],
      annotations:
        band && band.label
          ? [
              {
                xref: "paper" as const,
                x: 1,
                y: band.to,
                xanchor: "right" as const,
                yanchor: "bottom" as const,
                text: band.label,
                showarrow: false,
                font: { color: "#EA580C", size: 10 },
              },
            ]
          : [],
    }),
    [title, xLabel, yLabel, band],
  );

  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <Plot
        data={data}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height }}
      />
    </div>
  );
}
