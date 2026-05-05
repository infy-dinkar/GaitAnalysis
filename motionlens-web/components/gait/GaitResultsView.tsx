"use client";
// Reusable gait results view — same charts + tabs the live results page
// renders, but as a self-contained component so the saved-report viewer
// can render identical UI from a stored GaitDataDTO.

import { useState } from "react";
import { CalibrationHeader } from "@/components/gait/CalibrationHeader";
import { MetricsSection } from "@/components/gait/MetricsSection";
import { JointTabs, TabDef } from "@/components/gait/JointTabs";
import { InfoBox } from "@/components/gait/InfoBox";
import { GaitCycleSection } from "@/components/gait/GaitCycleSection";
import {
  PlotlyChart,
  passShapes,
  bandShape,
  hLineShape,
} from "@/components/gait/PlotlyChart";
import { fmt } from "@/lib/utils";
import type { GaitDataDTO, JointDetailDTO, PassSegmentDTO } from "@/lib/api";

const COLOR_LEFT = "#2563EB";
const COLOR_RIGHT = "#DC2626";
const COLOR_AMBER = "#EA580C";
const COLOR_REF = "#94A3B8";
const COLOR_ACCENT = "#EA580C";

type TabId =
  | "overview"
  | "knee"
  | "heel"
  | "step_length"
  | "timing"
  | "torso"
  | "ankle"
  | "cycle";

const TABS: TabDef<TabId>[] = [
  { id: "overview",    label: "Overview" },
  { id: "knee",        label: "Knee Angles" },
  { id: "heel",        label: "Heel Position" },
  { id: "step_length", label: "Step Length" },
  { id: "timing",      label: "Timing" },
  { id: "torso",       label: "Torso Lean" },
  { id: "ankle",       label: "Ankle Trajectory" },
  { id: "cycle",       label: "Gait Cycle" },
];

interface Props {
  data: GaitDataDTO;
  /** Optional override for the patient name shown in the calibration header. */
  patientNameOverride?: string | null;
}

export function GaitResultsView({ data, patientNameOverride }: Props) {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  return (
    <div className="space-y-12">
      <div>
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          {patientNameOverride || data.patient_info.name || "Anonymous patient"}
          <span className="text-accent">.</span>
        </h2>
        <div className="mt-5">
          <CalibrationHeader
            videoInfo={data.video_info}
            heightCm={data.patient_info.height_cm}
          />
        </div>
      </div>

      <div className="space-y-12">
        <MetricsSection variant="total" metrics={data.metrics_total} />
        <MetricsSection
          variant="clean"
          metrics={data.metrics_clean}
          walkingDirection={data.walking_direction}
        />
      </div>

      <div>
        <h3 className="text-2xl font-semibold tracking-tight md:text-3xl">
          Per-joint analysis<span className="text-accent">.</span>
        </h3>
        <p className="mt-2 text-sm text-muted">
          Detailed per-joint and per-cycle views. Validated walking passes are highlighted on
          every time-domain chart in lime — turning, accel and decel frames are unhighlighted.
        </p>

        <div className="mt-8">
          <JointTabs tabs={TABS} active={activeTab} onChange={setActiveTab}>
            {activeTab === "overview"    && <OverviewTab data={data} />}
            {activeTab === "knee"        && <KneeTab data={data} />}
            {activeTab === "heel"        && <HeelTab data={data} />}
            {activeTab === "step_length" && <StepLengthTab data={data} />}
            {activeTab === "timing"      && <TimingTab data={data} />}
            {activeTab === "torso"       && <TorsoTab data={data} />}
            {activeTab === "ankle"       && <AnkleTab data={data} />}
            {activeTab === "cycle"       && <CycleTab data={data} />}
          </JointTabs>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════
// TAB CONTENTS — direct port from /gait/results/page.tsx
// ══════════════════════════════════════════════════════════════════════

function OverviewTab({ data }: { data: GaitDataDTO }) {
  const o = data.normalized_overview;
  const passes = data.tabs_data.pass_segments;

  const rows: {
    title: string;
    yLabel: string;
    left: (number | null)[];
    right: (number | null)[];
    leftLabel: string;
    rightLabel: string;
    leftColor: string;
    rightColor: string;
  }[] = [
    {
      title: "Leg Angle Analysis (Normalized)",
      yLabel: "Leg Angle (degrees)",
      left: o.leg_angle.left,
      right: o.leg_angle.right,
      leftLabel: "Left Leg",
      rightLabel: "Right Leg",
      leftColor: "#A78BFA",
      rightColor: "#FB923C",
    },
    {
      title: "Knee Flexion Analysis (Normalized)",
      yLabel: "Knee Flexion (degrees)",
      left: o.knee_flexion.left,
      right: o.knee_flexion.right,
      leftLabel: "Left Knee",
      rightLabel: "Right Knee",
      leftColor: "#4FC3F7",
      rightColor: "#FB923C",
    },
    {
      title: "Hip Flexion Analysis (Normalized)",
      yLabel: "Hip Flexion (degrees)",
      left: o.hip_flexion.left,
      right: o.hip_flexion.right,
      leftLabel: "Left Hip",
      rightLabel: "Right Hip",
      leftColor: "#EA580C",
      rightColor: "#EF4444",
    },
    {
      title: "Ankle Deflection Analysis (Normalized)",
      yLabel: "Ankle Deflection (degrees)",
      left: o.ankle_deflection.left,
      right: o.ankle_deflection.right,
      leftLabel: "Left Ankle",
      rightLabel: "Right Ankle",
      leftColor: "#22D3EE",
      rightColor: "#FBBF24",
    },
  ];

  return (
    <div className="space-y-10">
      <div className="text-center">
        <h3 className="text-2xl font-semibold tracking-tight md:text-3xl">
          Normalized Gait Analysis Overview
        </h3>
        <p className="mx-auto mt-2 max-w-2xl text-sm italic text-muted">
          Scroll through each joint below. For deeper inspection with zoom and pan controls,
          switch to the individual joint tabs above.
        </p>
      </div>

      {rows.map((row) => (
        <div key={row.title} className="space-y-2">
          <h4 className="text-base font-semibold tracking-tight text-foreground">
            {row.title}
          </h4>
          <PlotlyChart
            height={280}
            data={[
              {
                type: "scatter",
                mode: "lines+markers",
                name: row.leftLabel,
                x: o.time_axis,
                y: row.left,
                line: { color: row.leftColor, width: 1.2 },
                marker: { color: row.leftColor, size: 3 },
              },
              {
                type: "scatter",
                mode: "lines+markers",
                name: row.rightLabel,
                x: o.time_axis,
                y: row.right,
                line: { color: row.rightColor, width: 1.2 },
                marker: { color: row.rightColor, size: 3 },
              },
            ]}
            layout={{
              xaxis: { title: { text: "Time (seconds)" } },
              yaxis: { title: { text: row.yLabel } },
              shapes: passShapes(passes),
              legend: {
                orientation: "h",
                y: 1.08,
                x: 0.5,
                xanchor: "center",
                font: { color: "#475569", size: 11 },
              },
              margin: { l: 64, r: 24, t: 36, b: 48 },
            }}
          />
        </div>
      ))}
    </div>
  );
}

function KneeTab({ data }: { data: GaitDataDTO }) {
  const passes = data.tabs_data.pass_segments;
  const ts = data.normalized_overview.time_axis;
  const left = data.joint_angles.left_knee;
  const right = data.joint_angles.right_knee;

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold tracking-tight">
        Knee Flexion Angles Over Time
      </h3>

      <div className="grid gap-6 md:grid-cols-2">
        <SideChart
          title={`Left Knee — Mean ${fmtJoint(left.mean)}`}
          color={COLOR_LEFT}
          time={ts}
          values={left.time_series}
          mean={left.mean}
          peak={left.peak}
          band={[0, 70]}
          bandLabel="Normal Swing Range (0°–70°)"
          passes={passes}
        />
        <SideChart
          title={`Right Knee — Mean ${fmtJoint(right.mean)}`}
          color={COLOR_RIGHT}
          time={ts}
          values={right.time_series}
          mean={right.mean}
          peak={right.peak}
          band={[0, 70]}
          bandLabel="Normal Swing Range (0°–70°)"
          passes={passes}
        />
      </div>

      <InfoBox>
        Knee flexion is the angle at the knee joint relative to full extension. During normal
        walking the knee should bend to roughly 0–70° in the swing phase to clear the foot. The
        green dashed line is your <em>mean</em> over the validated passes; the orange dotted line
        is your <em>peak</em>. The grey band shows the typical adult swing range for reference.
      </InfoBox>
    </div>
  );
}

function HeelTab({ data }: { data: GaitDataDTO }) {
  const h = data.tabs_data.heel_position;
  const passes = data.tabs_data.pass_segments;

  const buildChart = (
    side: "left" | "right",
    color: string,
  ) => (
    <PlotlyChart
      height={300}
      data={[
        {
          type: "scatter", mode: "lines",
          name: `${side === "left" ? "Left" : "Right"} heel`,
          x: h.time_axis,
          y: side === "left" ? h.left_x : h.right_x,
          line: { color, width: 1.4 },
        },
        {
          type: "scatter", mode: "markers",
          name: `Heel strikes (${side === "left" ? h.left_count : h.right_count})`,
          x: side === "left" ? h.left_strikes_t : h.right_strikes_t,
          y: side === "left" ? h.left_strikes_x : h.right_strikes_x,
          marker: { color: COLOR_ACCENT, size: 8 },
        },
      ]}
      layout={{
        title: {
          text: `${side === "left" ? "Left" : "Right"} Heel — ${
            side === "left" ? h.left_count : h.right_count
          } steps`,
          font: { color: "#0F172A", size: 13 },
        },
        xaxis: { title: { text: "Time (s)" } },
        yaxis: { title: { text: "Heel X (image-normalized)" } },
        shapes: passShapes(passes),
      }}
    />
  );

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold tracking-tight">Heel Position Trajectory</h3>
      <div className="grid gap-6 md:grid-cols-2">
        {buildChart("left", COLOR_LEFT)}
        {buildChart("right", COLOR_RIGHT)}
      </div>
      <InfoBox>
        Each line is the horizontal position of one heel over time (normalised to image width).
        Green dots mark detected heel strikes — these are the gait events used to compute cadence
        and step length. Validated walking passes are highlighted in lime; turning frames and
        accel/decel sections are excluded.
      </InfoBox>
    </div>
  );
}

function StepLengthTab({ data }: { data: GaitDataDTO }) {
  const sl = data.tabs_data.step_length;
  const isMeters = sl.unit === "m";
  const yLabel = `Step Length (${sl.unit})`;

  const buildChart = (side: "left" | "right", color: string) => {
    const values = side === "left" ? sl.left_values : sl.right_values;
    const mean = side === "left" ? sl.left_mean : sl.right_mean;
    const xs = values.map((_, i) => i + 1);
    const shapes: Record<string, unknown>[] = [];
    if (isMeters) {
      shapes.push(bandShape(0.55, 0.8, "rgba(161,161,170,0.12)"));
    }
    if (mean > 0) {
      shapes.push(hLineShape(mean, COLOR_ACCENT));
    }
    return (
      <PlotlyChart
        height={300}
        data={[
          {
            type: "bar",
            name: side === "left" ? "Left step length" : "Right step length",
            x: xs,
            y: values,
            marker: {
              color,
              line: { color: "#E2E8F0", width: 0.6 },
            },
          },
        ]}
        layout={{
          title: {
            text: `${side === "left" ? "Left" : "Right"} — Mean ${fmt(mean, 2)} ${sl.unit}`,
            font: { color: "#0F172A", size: 13 },
          },
          xaxis: { title: { text: "Step Index" }, dtick: 1 },
          yaxis: { title: { text: yLabel } },
          shapes,
        }}
      />
    );
  };

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold tracking-tight">
        Step Length per Step{" "}
        <span className="text-sm font-normal text-muted">
          ({isMeters ? "metres, anatomically scaled" : "pixels — set height in setup to scale"})
        </span>
      </h3>
      <div className="grid gap-6 md:grid-cols-2">
        {buildChart("left", COLOR_LEFT)}
        {buildChart("right", COLOR_RIGHT)}
      </div>
      <InfoBox>
        Each bar is one step. The green dashed line is the per-side mean; the grey band shows the
        typical adult range (0.55–0.80 m) when calibrated. Symmetric bars between Left and Right
        indicate a balanced gait.
      </InfoBox>
    </div>
  );
}

function TimingTab({ data }: { data: GaitDataDTO }) {
  const t = data.tabs_data.timing;

  const buildChart = (side: "left" | "right", color: string) => {
    const values = side === "left" ? t.left_intervals : t.right_intervals;
    const mean = side === "left" ? t.left_mean : t.right_mean;
    const xs = values.map((_, i) => i + 1);
    const shapes: Record<string, unknown>[] = [];
    if (mean > 0) {
      shapes.push(bandShape(mean * 0.9, mean * 1.1, "rgba(161,161,170,0.12)"));
      shapes.push(hLineShape(mean, COLOR_ACCENT));
    }
    return (
      <PlotlyChart
        height={300}
        data={[
          {
            type: "bar",
            name: side === "left" ? "Left timing" : "Right timing",
            x: xs,
            y: values,
            marker: { color },
          },
        ]}
        layout={{
          title: {
            text: `${side === "left" ? "Left" : "Right"} — Mean ${fmt(mean, 3)} s`,
            font: { color: "#0F172A", size: 13 },
          },
          xaxis: { title: { text: "Step Interval Index" }, dtick: 1 },
          yaxis: { title: { text: "Time (s)" } },
          shapes,
        }}
      />
    );
  };

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold tracking-tight">Step & Stride Timing</h3>
      <div className="grid gap-6 md:grid-cols-2">
        {buildChart("left", COLOR_LEFT)}
        {buildChart("right", COLOR_RIGHT)}
      </div>
      <InfoBox>
        Each bar is one inter-strike interval (same leg). Tighter clustering around the mean (lime
        line) means more consistent rhythm; the grey band marks ±10% of mean. Variability above
        ±10% can indicate gait instability or coordination issues.
      </InfoBox>
    </div>
  );
}

function TorsoTab({ data }: { data: GaitDataDTO }) {
  const t = data.tabs_data.torso_lean;
  const passes = data.tabs_data.pass_segments;

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold tracking-tight">
        Torso Lean Over Time{" "}
        <span className="text-sm font-normal text-muted">
          (+ = forward in walking direction)
        </span>
      </h3>
      <PlotlyChart
        height={320}
        data={[
          {
            type: "scatter", mode: "lines", name: "Torso lean",
            x: t.time_axis, y: t.angles,
            line: { color: COLOR_AMBER, width: 1.6 },
          },
        ]}
        layout={{
          xaxis: { title: { text: "Time (s)" } },
          yaxis: { title: { text: "Lean (°)" } },
          shapes: [
            ...passShapes(passes),
            bandShape(-5, 5, "rgba(161,161,170,0.12)"),
            hLineShape(t.mean, COLOR_RIGHT),
          ],
        }}
      />
      <InfoBox>
        Torso lean is the trunk&apos;s sagittal-plane angle, signed so positive values mean the torso
        is leaning <em>forward in the walking direction</em>. The grey band marks ±5° (typical
        adult range). Persistent lean beyond ±5° suggests trunk instability or compensation. Your
        mean: <span className="tabular text-foreground">{fmt(t.mean, 1)}°</span> (σ ={" "}
        <span className="tabular text-foreground">{fmt(t.std, 1)}°</span>).
      </InfoBox>
    </div>
  );
}

function AnkleTab({ data }: { data: GaitDataDTO }) {
  const a = data.tabs_data.ankle_trajectory;
  const passes = data.tabs_data.pass_segments;

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold tracking-tight">Ankle X-Trajectory Over Time</h3>
      <PlotlyChart
        height={320}
        data={[
          {
            type: "scatter", mode: "lines", name: "Left ankle",
            x: a.time_axis, y: a.left_x,
            line: { color: COLOR_LEFT, width: 1.4 },
          },
          {
            type: "scatter", mode: "lines", name: "Right ankle",
            x: a.time_axis, y: a.right_x,
            line: { color: COLOR_RIGHT, width: 1.4 },
          },
        ]}
        layout={{
          xaxis: { title: { text: "Time (s)" } },
          yaxis: { title: { text: "Ankle X (image-normalized)" } },
          shapes: passShapes(passes),
        }}
      />
      <InfoBox>
        Both ankles&apos; horizontal positions over time. Symmetric, mirrored sweeps between Left and
        Right indicate balanced gait. The validated walking passes are highlighted in lime —
        anything outside is excluded from the metrics.
      </InfoBox>
    </div>
  );
}

function CycleTab({ data }: { data: GaitDataDTO }) {
  if (!data.gait_cycle_data) {
    return (
      <InfoBox>
        Not enough clean cycles to display gait-cycle-normalized curves. Try a longer clip with
        more continuous walking and at least 3 strides per leg inside a validated pass.
      </InfoBox>
    );
  }
  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold tracking-tight">
        Gait-Cycle-Normalized Joint Angles
      </h3>
      <p className="text-sm text-muted">
        Each curve is the average of all clean cycles for that leg, with ±1 SD shaded. The grey
        band is the healthy-adult reference. Stance phase (0–60%) is shaded; toe-off is the
        dashed vertical line at 60%.
      </p>
      <GaitCycleSection data={data.gait_cycle_data} />
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

interface SideChartProps {
  title: string;
  color: string;
  time: number[];
  values: (number | null)[];
  mean: number | null;
  peak: number | null;
  band?: [number, number];
  bandLabel?: string;
  passes: PassSegmentDTO[];
}

function SideChart({
  title,
  color,
  time,
  values,
  mean,
  peak,
  band,
  bandLabel,
  passes,
}: SideChartProps) {
  const shapes: Record<string, unknown>[] = [...passShapes(passes)];
  if (band) shapes.push(bandShape(band[0], band[1], "rgba(161,161,170,0.12)"));
  if (mean !== null) shapes.push(hLineShape(mean, COLOR_ACCENT));
  if (peak !== null) shapes.push(hLineShape(peak, COLOR_AMBER, "dot"));

  const annotations: Record<string, unknown>[] = [];
  if (mean !== null) {
    annotations.push({
      x: 1, xref: "paper", y: mean, yref: "y",
      xanchor: "right", yanchor: "bottom",
      text: `Mean ${fmt(mean, 1)}°`,
      showarrow: false,
      font: { color: COLOR_ACCENT, size: 10 },
    });
  }
  if (peak !== null) {
    annotations.push({
      x: 1, xref: "paper", y: peak, yref: "y",
      xanchor: "right", yanchor: "bottom",
      text: `Peak ${fmt(peak, 1)}°`,
      showarrow: false,
      font: { color: COLOR_AMBER, size: 10 },
    });
  }
  if (band && bandLabel) {
    annotations.push({
      x: 0, xref: "paper", y: band[1], yref: "y",
      xanchor: "left", yanchor: "bottom",
      text: bandLabel,
      showarrow: false,
      font: { color: COLOR_REF, size: 10 },
    });
  }

  return (
    <PlotlyChart
      height={320}
      data={[
        {
          type: "scatter", mode: "lines", name: title.split(" — ")[0],
          x: time, y: values,
          line: { color, width: 1.6 },
        },
      ]}
      layout={{
        title: { text: title, font: { color: "#0F172A", size: 13 } },
        xaxis: { title: { text: "Time (s)" } },
        yaxis: { title: { text: "Flexion (°)" } },
        shapes,
        annotations,
      }}
    />
  );
}

function fmtJoint(v: number | null) {
  return v === null ? "—" : `${fmt(v, 1)}°`;
}

// Re-export to satisfy existing usages; not actually used here, but keep
// the import surface stable so this file is self-contained.
export type { JointDetailDTO };
