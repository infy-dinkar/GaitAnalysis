"use client";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Download } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
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
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { REPORT_DISCLAIMER } from "@/lib/disclaimer";
import { fmt } from "@/lib/utils";
import type { GaitDataDTO, JointDetailDTO, PassSegmentDTO } from "@/lib/api";

const STORAGE_KEY = "motionlens.gait_api_result";

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

export default function GaitResultsPage() {
  const [data, setData] = useState<GaitDataDTO | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) setData(JSON.parse(raw) as GaitDataDTO);
    } catch {
      // ignore
    }
  }, []);

  function downloadPDF() {
    if (!data) return;
    import("jspdf").then(({ default: jsPDF }) => {
      const doc = new jsPDF({ unit: "pt", format: "a4" });
      const margin = 48;
      let y = margin;
      const writeRow = (k: string, v: string) => {
        doc.text(k, margin, y);
        doc.text(v, margin + 220, y);
        y += 14;
      };
      doc.setFontSize(20);
      doc.text("MotionLens — Gait Report", margin, y);
      y += 26;
      doc.setFontSize(10);
      doc.setTextColor(120);
      const dateStr = new Date().toLocaleDateString();
      doc.text(`Generated ${dateStr}`, margin, y);
      y += 22;

      doc.setTextColor(0);
      doc.setFontSize(12);
      doc.text("Patient", margin, y); y += 16;
      doc.setFontSize(10);
      writeRow("Name", data.patient_info.name || "—");
      writeRow("Height", `${fmt(data.patient_info.height_cm, 0)} cm`);
      y += 8;

      doc.setFontSize(12);
      doc.text("Video", margin, y); y += 16;
      doc.setFontSize(10);
      writeRow("Duration",      `${fmt(data.video_info.duration_sec, 1)} s`);
      writeRow("FPS",           fmt(data.video_info.fps, 0));
      writeRow("Frames used",   `${data.video_info.frames_used} / ${data.video_info.total_frames}`);
      writeRow("Calibration",
        data.video_info.calibration_mm_per_px !== null
          ? `${fmt(data.video_info.calibration_mm_per_px, 3)} mm/px`
          : "uncalibrated");
      writeRow("Direction", data.walking_direction);
      y += 8;

      doc.setFontSize(12);
      doc.text("Clean metrics (steady-state)", margin, y); y += 16;
      doc.setFontSize(10);
      const c = data.metrics_clean;
      writeRow("Step count",  String(c.step_count));
      writeRow("Cadence",     c.cadence !== null ? `${fmt(c.cadence, 0)} steps/min` : "—");
      writeRow("Symmetry",    c.symmetry !== null ? `${fmt(c.symmetry * 100, 0)}%` : "—");
      writeRow("Knee peak",   c.knee_peak !== null ? `${fmt(c.knee_peak, 1)}°` : "—");
      writeRow("Stride CV",   c.stride_cv !== null ? `${fmt(c.stride_cv, 1)}%` : "—");
      writeRow("Step length", c.step_length !== null ? `${fmt(c.step_length, 2)} ${c.step_length_unit}` : "—");
      writeRow("Torso lean",  c.torso_lean !== null ? `${fmt(c.torso_lean, 1)}°` : "—");
      writeRow("Step time",   c.step_time !== null ? `${fmt(c.step_time, 2)} s` : "—");

      // ── Disclaimer footer (always at the bottom of the last page) ──
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const wrapWidth = pageWidth - margin * 2;
      const lineHeight = 11;

      doc.setFontSize(9);
      doc.setFont("helvetica", "italic");
      const wrapped = doc.splitTextToSize(REPORT_DISCLAIMER, wrapWidth);
      const blockHeight = wrapped.length * lineHeight + 12; // +12 for separator + gap

      // If the disclaimer wouldn't fit on the current page, push it to a
      // new one so it's never truncated.
      if (y + blockHeight > pageHeight - margin) {
        doc.addPage();
        y = margin;
      } else {
        y += 16;
      }

      // Thin separator line above
      doc.setDrawColor(180);
      doc.setLineWidth(0.5);
      doc.line(margin, y, pageWidth - margin, y);
      y += 10;

      doc.setTextColor(90);
      doc.text(wrapped, margin, y);
      // Reset font state in case caller adds anything else after
      doc.setFont("helvetica", "normal");
      doc.setTextColor(0);

      doc.save(`motionlens-gait-${dateStr.replaceAll("/", "-")}.pdf`);
    });
  }

  if (!data) {
    return (
      <>
        <Nav />
        <main className="flex flex-col">
          <Section className="pt-32 md:pt-40">
            <div className="max-w-xl">
              <Badge>No data</Badge>
              <h1 className="mt-5 text-3xl font-semibold tracking-tight md:text-4xl">
                Nothing to display yet
              </h1>
              <p className="mt-4 text-muted">
                Upload a walking clip first — results from your last analysis appear here.
              </p>
              <div className="mt-8">
                <Link href="/gait/upload">
                  <Button>Upload a video</Button>
                </Link>
              </div>
            </div>
          </Section>
        </main>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div>
              <Badge>Gait results</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-5xl">
                {data.patient_info.name || "Anonymous patient"}
                <span className="text-accent">.</span>
              </h1>
              <div className="mt-5">
                <CalibrationHeader
                  videoInfo={data.video_info}
                  heightCm={data.patient_info.height_cm}
                />
              </div>
            </div>
            <Button onClick={downloadPDF}>
              <Download className="h-4 w-4" />
              Download PDF
            </Button>
          </div>

          {/* ── Save to patient history (only when ?patientId in URL) ── */}
          <div className="mt-8">
            <SaveToPatientButton
              buildPayload={() => ({
                module: "gait",
                metrics: {
                  metrics_total: data.metrics_total,
                  metrics_clean: data.metrics_clean,
                  video_info: data.video_info,
                  walking_direction: data.walking_direction,
                  joint_angles: data.joint_angles,
                  normalized_overview: data.normalized_overview,
                  tabs_data: data.tabs_data,
                  gait_cycle_data: data.gait_cycle_data,
                },
                observations: data.observations as unknown as Record<string, unknown>,
                video_filename:
                  (data as unknown as { _video_filename?: string })._video_filename,
                video_size_bytes:
                  (data as unknown as { _video_size_bytes?: number })._video_size_bytes,
              })}
            />
          </div>

          {/* ── Total + Clean metric sections ───────────────────────── */}
          <div className="mt-12 space-y-12">
            <MetricsSection variant="total" metrics={data.metrics_total} />
            <MetricsSection
              variant="clean"
              metrics={data.metrics_clean}
              walkingDirection={data.walking_direction}
            />
          </div>

          {/* ── Per-joint tabs ──────────────────────────────────────── */}
          <div className="mt-16">
            <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
              Per-joint analysis<span className="text-accent">.</span>
            </h2>
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

          <div className="mt-12 flex justify-between">
            <Link href="/gait/upload">
              <Button variant="ghost">Analyse another clip</Button>
            </Link>
          </div>

          {/* ── Unified report disclaimer ──────────────────────────── */}
          <ReportDisclaimer />
        </Section>
      </main>
      <Footer />
    </>
  );
}

// ══════════════════════════════════════════════════════════════════════
// TAB CONTENTS
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
      leftColor: "#A78BFA",   // purple
      rightColor: "#FB923C",  // orange
    },
    {
      title: "Knee Flexion Analysis (Normalized)",
      yLabel: "Knee Flexion (degrees)",
      left: o.knee_flexion.left,
      right: o.knee_flexion.right,
      leftLabel: "Left Knee",
      rightLabel: "Right Knee",
      leftColor: "#4FC3F7",   // blue
      rightColor: "#FB923C",  // orange
    },
    {
      title: "Hip Flexion Analysis (Normalized)",
      yLabel: "Hip Flexion (degrees)",
      left: o.hip_flexion.left,
      right: o.hip_flexion.right,
      leftLabel: "Left Hip",
      rightLabel: "Right Hip",
      leftColor: "#EA580C",   // orange (was green)
      rightColor: "#EF4444",  // red
    },
    {
      title: "Ankle Deflection Analysis (Normalized)",
      yLabel: "Ankle Deflection (degrees)",
      left: o.ankle_deflection.left,
      right: o.ankle_deflection.right,
      leftLabel: "Left Ankle",
      rightLabel: "Right Ankle",
      leftColor: "#22D3EE",   // cyan
      rightColor: "#FBBF24",  // amber
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
    const shapes: any[] = [];
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
    const shapes: any[] = [];
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
        Torso lean is the trunk's sagittal-plane angle, signed so positive values mean the torso
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
        Both ankles' horizontal positions over time. Symmetric, mirrored sweeps between Left and
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
  const shapes: any[] = [...passShapes(passes)];
  if (band) shapes.push(bandShape(band[0], band[1], "rgba(161,161,170,0.12)"));
  if (mean !== null) shapes.push(hLineShape(mean, COLOR_ACCENT));
  if (peak !== null) shapes.push(hLineShape(peak, COLOR_AMBER, "dot"));

  const annotations: any[] = [];
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

