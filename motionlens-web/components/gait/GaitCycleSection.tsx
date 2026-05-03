"use client";
import { useMemo } from "react";
import { PlotlyChart } from "@/components/gait/PlotlyChart";
import type {
  GaitCycleDataDTO,
  GaitCycleCurveDTO,
  NormalRefCurveDTO,
} from "@/lib/api";

const COLOR_LEFT = "#2563EB";
const COLOR_RIGHT = "#DC2626";
const COLOR_REF = "#94A3B8";
const COLOR_BG_STANCE = "rgba(15,23,42,0.05)";

const X = Array.from({ length: 101 }, (_, i) => i);

type Joint = "hip" | "knee" | "ankle";

const JOINT_TITLES: Record<Joint, string> = {
  hip: "HIP ANGLE",
  knee: "KNEE ANGLE",
  ankle: "ANKLE ANGLE",
};

interface RowProps {
  joint: Joint;
  left: GaitCycleCurveDTO;
  right: GaitCycleCurveDTO;
  reference: NormalRefCurveDTO;
  stanceEndPct: number;
}

function GaitCycleRow({ joint, left, right, reference, stanceEndPct }: RowProps) {
  const data = useMemo(() => {
    const traces: any[] = [];

    // Reference band: filled area between lower and upper.
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Normal range (upper)",
      x: X,
      y: reference.upper_band,
      line: { color: COLOR_REF, width: 0 },
      showlegend: false,
      hoverinfo: "skip",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Normal adult range",
      x: X,
      y: reference.lower_band,
      line: { color: COLOR_REF, width: 0 },
      fill: "tonexty",
      fillcolor: "rgba(161,161,170,0.18)",
      hoverinfo: "skip",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Normal mean",
      x: X,
      y: reference.mean_curve,
      line: { color: COLOR_REF, width: 1.2, dash: "dash" },
      hoverinfo: "skip",
      showlegend: false,
    });

    // Left leg: SD band first (so the line draws on top)
    if (left.K > 0 && left.mean_curve.length === 101) {
      const lo = left.mean_curve.map((m, i) =>
        m === null || left.std_curve[i] === null
          ? null
          : (m as number) - (left.std_curve[i] as number),
      );
      const hi = left.mean_curve.map((m, i) =>
        m === null || left.std_curve[i] === null
          ? null
          : (m as number) + (left.std_curve[i] as number),
      );
      traces.push({
        type: "scatter",
        mode: "lines",
        x: X,
        y: hi,
        line: { color: COLOR_LEFT, width: 0 },
        showlegend: false,
        hoverinfo: "skip",
      });
      traces.push({
        type: "scatter",
        mode: "lines",
        x: X,
        y: lo,
        line: { color: COLOR_LEFT, width: 0 },
        fill: "tonexty",
        fillcolor: "rgba(79,195,247,0.20)",
        showlegend: false,
        hoverinfo: "skip",
      });
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `Left (K=${left.K})`,
        x: X,
        y: left.mean_curve,
        line: { color: COLOR_LEFT, width: 2 },
      });
    }

    // Right leg
    if (right.K > 0 && right.mean_curve.length === 101) {
      const lo = right.mean_curve.map((m, i) =>
        m === null || right.std_curve[i] === null
          ? null
          : (m as number) - (right.std_curve[i] as number),
      );
      const hi = right.mean_curve.map((m, i) =>
        m === null || right.std_curve[i] === null
          ? null
          : (m as number) + (right.std_curve[i] as number),
      );
      traces.push({
        type: "scatter",
        mode: "lines",
        x: X,
        y: hi,
        line: { color: COLOR_RIGHT, width: 0 },
        showlegend: false,
        hoverinfo: "skip",
      });
      traces.push({
        type: "scatter",
        mode: "lines",
        x: X,
        y: lo,
        line: { color: COLOR_RIGHT, width: 0 },
        fill: "tonexty",
        fillcolor: "rgba(229,115,115,0.20)",
        showlegend: false,
        hoverinfo: "skip",
      });
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `Right (K=${right.K})`,
        x: X,
        y: right.mean_curve,
        line: { color: COLOR_RIGHT, width: 2 },
      });
    }

    return traces;
  }, [left, right, reference]);

  // Compute a sensible y-range that contains both reference and subject curves.
  const yRange = useMemo(() => {
    const all: number[] = [];
    const push = (a: (number | null)[]) =>
      a.forEach((v) => {
        if (v !== null && Number.isFinite(v)) all.push(v as number);
      });
    push(reference.lower_band);
    push(reference.upper_band);
    if (left.K > 0) {
      push(left.mean_curve);
      push(left.std_curve);
    }
    if (right.K > 0) {
      push(right.mean_curve);
      push(right.std_curve);
    }
    if (!all.length) return undefined;
    const ymin = Math.floor(Math.min(...all) / 5) * 5 - 5;
    const ymax = Math.ceil(Math.max(...all) / 5) * 5 + 5;
    return [ymin, ymax];
  }, [left, right, reference]);

  const layout = {
    title: { text: `${JOINT_TITLES[joint]} — mean ± 1 SD`, font: { color: "#0F172A", size: 13 } },
    height: 280,
    xaxis: {
      title: { text: "Gait Cycle (%)" },
      range: [0, 100],
    },
    yaxis: {
      title: { text: "Angle (°)" },
      range: yRange,
    },
    shapes: [
      {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: 0,
        x1: stanceEndPct,
        y0: 0,
        y1: 1,
        fillcolor: COLOR_BG_STANCE,
        line: { width: 0 },
        layer: "below",
      },
      {
        type: "line",
        xref: "x",
        x0: stanceEndPct,
        x1: stanceEndPct,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color: "#94A3B8", width: 1, dash: "dash" },
      },
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: 0,
        y1: 0,
        line: { color: "#94A3B8", width: 0.6 },
      },
    ],
  };

  return <PlotlyChart data={data} layout={layout} height={280} />;
}

interface SectionProps {
  data: GaitCycleDataDTO;
}

export function GaitCycleSection({ data }: SectionProps) {
  return (
    <div className="space-y-4">
      <GaitCycleRow
        joint="hip"
        left={data.left.hip}
        right={data.right.hip}
        reference={data.normal_reference.hip}
        stanceEndPct={data.stance_phase_pct}
      />
      <GaitCycleRow
        joint="knee"
        left={data.left.knee}
        right={data.right.knee}
        reference={data.normal_reference.knee}
        stanceEndPct={data.stance_phase_pct}
      />
      <GaitCycleRow
        joint="ankle"
        left={data.left.ankle}
        right={data.right.ankle}
        reference={data.normal_reference.ankle}
        stanceEndPct={data.stance_phase_pct}
      />

      {/* Cycle detection metadata */}
      <div className="rounded-card border border-border bg-surface p-5 text-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Cycle detection
        </p>
        <p className="mt-2 text-muted">
          <span className="text-foreground">{data.right.cycles_accepted}</span> right strides
          accepted, <span className="text-foreground">{data.right.cycles_rejected_amplitude}</span>{" "}
          rejected due to low heel-clearance amplitude.{" "}
          <span className="text-foreground">{data.left.cycles_accepted}</span> left strides
          accepted, <span className="text-foreground">{data.left.cycles_rejected_amplitude}</span>{" "}
          rejected.
        </p>
        <p className="mt-2 text-muted">
          <span className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Cycle quality:
          </span>{" "}
          <span className="text-foreground">{data.left.cycles_kept}</span> left cycles kept (
          <span className="text-foreground">{data.left.cycles_rejected_too_long}</span> rejected as
          too long, <span className="text-foreground">{data.left.cycles_rejected_too_short}</span>{" "}
          as too short);{" "}
          <span className="text-foreground">{data.right.cycles_kept}</span> right cycles kept (
          <span className="text-foreground">{data.right.cycles_rejected_too_long}</span> rejected
          as too long,{" "}
          <span className="text-foreground">{data.right.cycles_rejected_too_short}</span> as too
          short).
        </p>
      </div>
    </div>
  );
}
