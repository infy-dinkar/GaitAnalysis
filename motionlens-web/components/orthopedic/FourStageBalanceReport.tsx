"use client";
// 4-Stage Balance Test (Test C4) — report.
//
// Top: 4 traffic-lights for stage progression + final classification
// banner. Below: per-stage cards with hold time, failure mode (when
// failed), sway metrics, hip-mid sway plot, and capture screenshot.
// Closes with the standard interpretation paragraph + reference
// cutoffs + unified disclaimer.

import dynamic from "next/dynamic";
import { CheckCircle2, XCircle } from "lucide-react";
import {
  CLASSIFICATION_LABEL,
  CLASSIFICATION_TONE,
  STAGE_HOLD_SEC,
  STAGE_LABEL,
  type SessionResult,
  type StageIndex,
  type StageResult,
} from "@/lib/orthopedic/fourStageBalance";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

const STAGES: readonly StageIndex[] = [1, 2, 3, 4];

interface Props {
  patientName: string | null;
  session: SessionResult;
  interpretation: string;
}

export function FourStageBalanceReport({
  patientName,
  session,
  interpretation,
}: Props) {
  const tone = CLASSIFICATION_TONE[session.classification];
  const cls = CLASSIFICATION_LABEL[session.classification];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          4-Stage Balance Test
        </h2>
        <p className="mt-2 text-sm text-muted">
          {patientName ?? "Patient"} · age{" "}
          {session.patient_age !== null ? session.patient_age : "—"}
        </p>
      </div>

      {/* Final classification banner */}
      <section className={`rounded-card border-0 p-5 ${tone}`}>
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] opacity-80">
              CDC fall-risk classification
            </p>
            <p className="mt-1 text-2xl font-semibold tracking-tight">
              {cls}
            </p>
            <p className="mt-1 text-sm opacity-80">{session.norm_band_label}</p>
          </div>
          <div className="text-right">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] opacity-80">
              Final stage
            </p>
            <p className="mt-1 text-2xl font-semibold tabular">
              {session.final_stage_completed} / 4
            </p>
            <p className="mt-1 text-sm opacity-80">
              {session.final_stage_completed === 4
                ? "All stages held"
                : session.first_failed_stage !== null
                ? `Failed at stage ${session.first_failed_stage}`
                : "Test stopped early"}
            </p>
          </div>
        </div>
      </section>

      {/* 4 traffic lights — stage progression visual */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">
          Stage progression
        </h3>
        <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
          {STAGES.map((s) => (
            <ProgressTile key={s} stage={s} result={session.stages[s]} />
          ))}
        </div>
      </section>

      {/* Per-stage detailed cards */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Per-stage detail</h3>
        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          {STAGES.map((s) => (
            <StageCard key={s} stage={s} result={session.stages[s]} />
          ))}
        </div>
      </section>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
        </div>
      </section>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Reference cutoffs</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-xs leading-relaxed text-muted">
          <p>
            <span className="font-medium text-foreground">Stage 3 (tandem):</span>{" "}
            unable to hold for {STAGE_HOLD_SEC} s = significantly elevated fall risk.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Stage 4 (single-leg):</span>{" "}
            &lt; 5 s for age &gt; 60 = high fall risk (PDF C4 sub-criterion).
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Sway:</span>{" "}
            path length and 95% ellipse area are in pixels (relative
            units — not calibrated to cm; suitable for trend tracking
            within the same patient).
          </p>
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Progression tile ───────────────────────────────────────────

function ProgressTile({
  stage,
  result,
}: {
  stage: StageIndex;
  result: StageResult | undefined;
}) {
  let tone: string;
  let icon: React.ReactNode;
  let status: string;
  if (result?.outcome === "pass") {
    tone = "border-emerald-500/40 bg-emerald-500/5";
    icon = <CheckCircle2 className="h-4 w-4 text-emerald-600" />;
    status = "Held full 10 s";
  } else if (result?.outcome === "fail") {
    tone = "border-red-500/40 bg-red-500/5";
    icon = <XCircle className="h-4 w-4 text-red-600" />;
    status = `Failed at ${result.duration_seconds.toFixed(1)} s`;
  } else {
    tone = "border-border bg-surface";
    icon = <span className="h-2 w-2 rounded-full bg-border" />;
    status = "Not attempted";
  }
  return (
    <div className={`rounded-card border p-4 ${tone}`}>
      <div className="flex items-center gap-2">
        {icon}
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-foreground">
          Stage {stage}
        </p>
      </div>
      <p className="mt-1 text-sm font-medium text-foreground">
        {STAGE_LABEL[stage].split(" · ")[1]}
      </p>
      <p className="mt-1 text-xs text-muted">{status}</p>
    </div>
  );
}

// ─── Per-stage card ─────────────────────────────────────────────

function StageCard({
  stage,
  result,
}: {
  stage: StageIndex;
  result: StageResult | undefined;
}) {
  if (!result) {
    return (
      <section className="rounded-card border border-dashed border-border bg-surface/50 p-5 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          {STAGE_LABEL[stage]}
        </p>
        <p className="mt-2 text-sm text-muted">
          Not attempted — test stopped before this stage.
        </p>
      </section>
    );
  }

  const passed = result.outcome === "pass";
  const badge = passed
    ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
    : "bg-red-500/10 text-red-700 dark:text-red-400";
  const badgeText = passed ? "Pass" : "Fail";

  return (
    <section className="space-y-3 rounded-card border border-border bg-surface/40 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-border pb-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            {STAGE_LABEL[stage]}
          </p>
          <p className="mt-1 text-2xl font-semibold tabular text-foreground">
            {result.duration_seconds.toFixed(1)} s
          </p>
        </div>
        <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${badge}`}>
          {badgeText}
        </span>
      </div>

      <table className="w-full text-left text-sm">
        <tbody>
          <Row label="Hold time (capped)" value={`${result.hold_seconds.toFixed(1)} / ${STAGE_HOLD_SEC} s`} />
          <Row label="Sway path length" value={`${result.sway_path_px.toFixed(0)} px (relative)`} />
          <Row label="95% sway ellipse" value={`${result.sway_95_ellipse_px2.toFixed(0)} px² (relative)`} />
          {!passed && (
            <Row label="Failure mode" value={failureLabel(result.failure_mode)} />
          )}
        </tbody>
      </table>

      {result.hip_path.length > 1 && <SwayPlot result={result} />}

      {result.screenshot_data_url && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
            Capture frame
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.screenshot_data_url}
            alt={`${STAGE_LABEL[stage]} capture`}
            className="w-full rounded-md border border-border"
          />
        </div>
      )}
    </section>
  );
}

function failureLabel(mode: StageResult["failure_mode"]): string {
  if (mode === "foot_touchdown") return "Lifted foot touched down";
  if (mode === "arm_grab") return "Arm grab for support";
  if (mode === "position_lost") return "Stance position drifted out of tolerance";
  if (mode === "stopped") return "Operator stopped the trial";
  return "—";
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr className="border-b border-border/50 last:border-b-0">
      <td className="py-2 text-muted">{label}</td>
      <td className="py-2 text-right tabular text-foreground">{value}</td>
    </tr>
  );
}

function SwayPlot({ result }: { result: StageResult }) {
  const xs = result.hip_path.map((p) => p.x);
  const ys = result.hip_path.map((p) => p.y);

  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
        Hip-midpoint sway trace
      </p>
      <PlotlyChart
        height={220}
        data={[
          {
            type: "scatter",
            mode: "lines",
            name: "Sway path",
            x: xs,
            y: ys,
            line: { color: "#EA580C", width: 1.4 },
          },
          {
            type: "scatter",
            mode: "markers",
            name: "Start",
            x: [xs[0]],
            y: [ys[0]],
            marker: { color: "#10B981", size: 8 },
            hoverinfo: "name",
          },
          {
            type: "scatter",
            mode: "markers",
            name: "End",
            x: [xs[xs.length - 1]],
            y: [ys[ys.length - 1]],
            marker: { color: "#DC2626", size: 8 },
            hoverinfo: "name",
          },
        ]}
        layout={{
          xaxis: { title: { text: "X (px)" }, scaleanchor: "y" },
          yaxis: { title: { text: "Y (px)" }, autorange: "reversed" },
          margin: { l: 56, r: 24, t: 20, b: 44 },
          legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
        }}
      />
    </div>
  );
}

