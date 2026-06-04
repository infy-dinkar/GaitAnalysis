"use client";
// Tandem Walk (E1) report.
//
// Single trial — renders: classification badge, big-number aggregates
// (missteps, arm grabs, mean lateral deviation, step-time CV, trunk
// sway), per-step table, top-down line trace plotting each footstrike
// relative to the fitted walking line, capture-moment screenshot.

import dynamic from "next/dynamic";
import {
  ABNORMAL_MEAN_DEVIATION_CM,
  ARM_ABDUCTION_DEG,
  DEVIATION_TOLERANCE_CM,
  MISSTEP_DEVIATION_CM,
  POSITIVE_SCREEN_MISSTEP_COUNT,
  TANDEM_CLASSIFICATION_LABEL,
  TANDEM_CLASSIFICATION_TONE,
  TARGET_STEP_COUNT,
  type TandemWalkResult,
} from "@/lib/orthopedic/tandemWalk";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import type { PatientDTO } from "@/lib/patients";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  result: TandemWalkResult;
  interpretation: string;
}

export function TandemWalkReport({ patientName, patient, result, interpretation }: Props) {
  const tone = TANDEM_CLASSIFICATION_TONE[result.classification];
  const label = TANDEM_CLASSIFICATION_LABEL[result.classification];

  // Build the line-trace chart. Plot each footstrike as a point at
  // its (foot_x, foot_y) in pixel space; overlay the fitted walking
  // line. Image y-down means the camera-near (largest y) end is the
  // "end" of the walk — we flip the y axis so it reads bottom-to-top
  // as the patient walks toward the camera.
  const stepXs = result.steps.map((s) => s.foot_x);
  const stepYs = result.steps.map((s) => s.foot_y);
  const stepLabels = result.steps.map((s) =>
    `Step ${s.step_index} (${s.side}) — ${s.deviation_cm !== null ? s.deviation_cm.toFixed(1) + " cm" : "—"}`,
  );
  const stepColors = result.steps.map((s) =>
    s.is_misstep ? "#EF4444" : (s.side === "left" ? "#3B82F6" : "#10B981"),
  );

  // Walking-line endpoints — sample at the y-min and y-max of the
  // captured hip-mid trail so the line spans the actual walked region.
  let lineTraceX: number[] = [];
  let lineTraceY: number[] = [];
  if (result.walking_line) {
    const ys = result.samples
      .map((s) => s.hip_mid_y)
      .filter((y): y is number => y !== null);
    if (ys.length >= 2) {
      const yMin = Math.min(...ys);
      const yMax = Math.max(...ys);
      lineTraceX = [
        result.walking_line.a * yMin + result.walking_line.b,
        result.walking_line.a * yMax + result.walking_line.b,
      ];
      lineTraceY = [yMin, yMax];
    }
  }

  return (
    <div className="space-y-8">
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle={`Tandem Walk (E1) · ${TARGET_STEP_COUNT} heel-to-toe steps`}
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Tandem Walk
        </h2>
        <div className="mt-3">
          <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${tone}`}>
            {label}
          </span>
        </div>
      </div>

      {/* Aggregate readouts */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <BigNumber
          value={`${result.misstep_count} / ${result.steps.length}`}
          unit=""
          label="Missteps (off-line steps)"
        />
        <BigNumber
          value={`${result.arm_grab_count}`}
          unit=""
          label="Arm-grab events"
        />
        <BigNumber
          value={fmtCm(result.mean_deviation_cm)}
          unit=""
          label="Mean lateral deviation"
        />
        <BigNumber
          value={fmtCm(result.max_deviation_cm)}
          unit=""
          label="Worst-step deviation"
        />
        <BigNumber
          value={result.step_time_cv > 0
            ? result.step_time_cv.toFixed(2)
            : "—"}
          unit=""
          label="Step-time CV"
        />
        <BigNumber
          value={fmtCm(result.trunk_sway_range_cm)}
          unit=""
          label="Trunk sway range"
        />
      </section>

      {/* Flag chips */}
      <section className="flex flex-wrap gap-2">
        <FlagChip
          ok={result.misstep_count < POSITIVE_SCREEN_MISSTEP_COUNT}
          label={
            result.misstep_count >= POSITIVE_SCREEN_MISSTEP_COUNT
              ? `≥ ${POSITIVE_SCREEN_MISSTEP_COUNT} missteps`
              : `< ${POSITIVE_SCREEN_MISSTEP_COUNT} missteps`
          }
        />
        <FlagChip
          ok={result.mean_deviation_cm <= ABNORMAL_MEAN_DEVIATION_CM}
          label={
            result.mean_deviation_cm > ABNORMAL_MEAN_DEVIATION_CM
              ? `Mean deviation > ${ABNORMAL_MEAN_DEVIATION_CM} cm`
              : `Mean deviation ≤ ${ABNORMAL_MEAN_DEVIATION_CM} cm`
          }
        />
        <FlagChip
          ok={result.arm_grab_count === 0}
          label={
            result.arm_grab_count > 0
              ? `${result.arm_grab_count} arm-grab events`
              : "No arm-grab events"
          }
        />
      </section>

      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {interpretation}
        </div>
      </section>

      {/* Per-step table */}
      {result.steps.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Per-step metrics</h3>
          <div className="mt-3 overflow-x-auto rounded-card border border-border bg-surface">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border bg-elevated text-[10px] uppercase tracking-[0.12em] text-subtle">
                <tr>
                  <th className="px-3 py-2 font-medium">Step</th>
                  <th className="px-3 py-2 font-medium">Foot</th>
                  <th className="px-3 py-2 text-right font-medium">t (ms)</th>
                  <th className="px-3 py-2 text-right font-medium">Raw (cm)</th>
                  <th className="px-3 py-2 text-right font-medium">Effective (cm)</th>
                  <th className="px-3 py-2 text-right font-medium">Misstep</th>
                </tr>
              </thead>
              <tbody>
                {result.steps.map((s) => (
                  <tr
                    key={s.step_index}
                    className={`border-b border-border/50 last:border-b-0 ${s.is_misstep ? "bg-red-500/5" : ""}`}
                  >
                    <td className="px-3 py-2 text-foreground">{s.step_index}</td>
                    <td className="px-3 py-2 text-foreground capitalize">{s.side}</td>
                    <td className="px-3 py-2 text-right tabular text-foreground">{s.t_ms.toFixed(0)}</td>
                    <td className="px-3 py-2 text-right tabular text-muted">
                      {s.raw_deviation_cm !== null ? s.raw_deviation_cm.toFixed(1) : "—"}
                    </td>
                    <td className="px-3 py-2 text-right tabular text-foreground">
                      {s.deviation_cm !== null ? s.deviation_cm.toFixed(1) : "—"}
                    </td>
                    <td className={`px-3 py-2 text-right tabular ${s.is_misstep ? "text-red-600" : "text-emerald-600"}`}>
                      {s.is_misstep ? "✗ off-line" : "✓"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Top-down trace */}
      {result.steps.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Footstrike trace (camera frame)
          </h3>
          <p className="mt-1 text-xs text-muted">
            Each dot is a footstrike at its image-pixel location. The dashed line
            is the fitted walking-line (hip-midpoint least-squares). Y axis is flipped
            so the camera-near end (where the patient finishes) is at the top.
            Blue = left foot, green = right foot, red = misstep.
          </p>
          <div className="mt-3">
            <PlotlyChart
              height={320}
              data={[
                ...(lineTraceX.length > 0 ? [{
                  type: "scatter", mode: "lines", name: "Walking line",
                  x: lineTraceX, y: lineTraceY,
                  line: { color: "#94A3B8", width: 1.5, dash: "dash" },
                }] : []),
                {
                  type: "scatter", mode: "markers+text",
                  name: "Footstrikes",
                  x: stepXs, y: stepYs,
                  text: result.steps.map((s) => String(s.step_index)),
                  textposition: "top right",
                  marker: { color: stepColors, size: 12 },
                  hovertext: stepLabels,
                  hoverinfo: "text",
                },
              ]}
              layout={{
                xaxis: { title: { text: "X (px)" } },
                yaxis: {
                  title: { text: "Y (px, image y-down — flipped)" },
                  autorange: "reversed",
                },
                showlegend: true,
                legend: { orientation: "h", y: 1.12, x: 0.5, xanchor: "center" },
                margin: { l: 56, r: 24, t: 20, b: 44 },
              }}
            />
          </div>
        </section>
      )}

      <section>
        <h3 className="text-base font-semibold tracking-tight">Reference cutoffs</h3>
        <div className="mt-3 grid gap-3 rounded-card border border-border bg-surface p-5 text-xs leading-relaxed text-muted md:grid-cols-2">
          <div>
            <p>
              <span className="font-medium text-foreground">Positive screen:</span>{" "}
              ≥ {POSITIVE_SCREEN_MISSTEP_COUNT} missteps in {TARGET_STEP_COUNT} steps —
              suggests cerebellar / vestibular dysfunction.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Misstep:</span>{" "}
              a single step with effective deviation &gt; {MISSTEP_DEVIATION_CM} cm
              from the fitted walking line.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Noise floor:</span>{" "}
              a tolerance of {DEVIATION_TOLERANCE_CM} cm (≈ 6 % of shoulder width)
              is subtracted from every raw step deviation before any flag fires.
              This absorbs pose-detection jitter and natural hip-midpoint sway
              during straight walking — so a perfect tandem walk doesn&apos;t
              register false missteps.
            </p>
          </div>
          <div>
            <p>
              <span className="font-medium text-foreground">Mean placement:</span>{" "}
              effective-mean deviation &gt; {ABNORMAL_MEAN_DEVIATION_CM} cm = abnormal foot placement.
            </p>
            <p className="mt-1">
              <span className="font-medium text-foreground">Arm grab:</span>{" "}
              wrist abducted &gt; {ARM_ABDUCTION_DEG}° from the body axis — patient
              throwing an arm out for balance.
            </p>
          </div>
        </div>
        {result.patient_age !== null && (
          <p className="mt-2 text-xs text-muted">
            Patient age on record: {result.patient_age} years. (Age-norm comparison
            isn&apos;t yet in the codebase&apos;s norm table — placeholder for future
            interpretation.)
          </p>
        )}
      </section>

      {result.capture_screenshot_data_url && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Capture frame</h3>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.capture_screenshot_data_url}
            alt="Tandem walk capture frame"
            className="mt-3 w-full rounded-md border border-border"
          />
        </section>
      )}

      <ReportDisclaimer />
    </div>
  );
}

function BigNumber({ value, unit, label }: { value: string; unit: string; label: string }) {
  return (
    <div className="rounded-card border border-border bg-surface p-5">
      <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">{label}</p>
      <p className="mt-2 tabular text-3xl font-semibold text-foreground">
        {value}
        {unit && <span className="text-base text-muted"> {unit}</span>}
      </p>
    </div>
  );
}

function FlagChip({ label, ok }: { label: string; ok: boolean }) {
  const tone = ok
    ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
    : "bg-red-500/10 text-red-700 dark:text-red-400";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${tone}`}>
      {label} {ok ? "✓" : "✗"}
    </span>
  );
}

function fmtCm(v: number): string {
  return `${v.toFixed(1)} cm`;
}
