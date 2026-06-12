"use client";
import { useEffect, useMemo, useState } from "react";
import { PlotlyChart } from "@/components/gait/PlotlyChart";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import { fmt } from "@/lib/utils";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { PatientDTO } from "@/lib/patients";
import type { BiomechCompensationDTO } from "@/lib/api";

interface KeyFrame {
  label: string;
  frame_index: number;
  image_data_url: string;
}

interface Props {
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle";
  /** Just the movement name — e.g. "Lateral Flexion", "Flexion".
   *  For merged movements this is the PRIMARY direction's label
   *  (e.g. "External Rotation"); secondaryMovementName carries the
   *  other direction. */
  movementName: string;
  movementId: string;
  measured: number;
  target: [number, number];
  side?: "left" | "right";
  /** When rendering a saved report, override the live patient header. */
  patientNameOverride?: string | null;
  patientIdOverride?: string | null;
  dateOverride?: string;
  /** Full patient object — preferred over patientNameOverride. When
   *  provided, the PatientHeader renders age / gender / height / weight
   *  / contact / notes; falls back to the legacy minimal strip when not. */
  patientOverride?: PatientDTO | null;
  /** Annotated screenshots of the test's key moments. Optional;
   *  empty / undefined just hides the section. Returned by the
   *  backend MediaPipe path (ankle today). */
  keyFrames?: KeyFrame[];
  /** Secondary-direction label for merged movements (e.g.
   *  "Internal Rotation"). When set, the report renders an extra
   *  results row, an extra chart bar, and an extra interpretation
   *  sentence for the secondary direction. */
  secondaryMovementName?: string;
  secondaryMeasured?: number;
  secondaryTarget?: [number, number];
  /** Compensatory-movement findings from the recording. Renders as
   *  a "Compensations Detected" section between the Plotly chart
   *  and Key frames. When undefined or empty, the section is hidden
   *  entirely. When set, ALL entries are shown — flagged ones as
   *  colored cards, unflagged ones contribute to a green "no
   *  compensations detected" line if EVERY entry is unflagged. */
  compensations?: BiomechCompensationDTO[];
}

// Educational text — direct port of biomech_flow.py SHOULDER_EDU / NECK_EDU.
const SHOULDER_EDU =
  "The shoulder is a ball-and-socket joint with the greatest range of " +
  "motion of any joint in the body. Reduced range of motion in shoulder " +
  "movements can indicate rotator cuff issues, adhesive capsulitis " +
  "('frozen shoulder'), impingement syndromes, or post-injury stiffness. " +
  "Bilateral asymmetry — significantly different ROM on left vs right — " +
  "is often more clinically informative than absolute values, since " +
  "individual baselines vary widely.";

const NECK_EDU =
  "The cervical spine allows the head to move in all directions. " +
  "Reduced cervical range of motion is commonly associated with muscle " +
  "strain, cervical spondylosis, disc herniation, or whiplash injury. " +
  "Approximately 50% of cervical rotation occurs at the atlanto-axial " +
  "(C1–C2) joint; reductions in rotation specifically may point to " +
  "upper-cervical involvement, while flexion/extension deficits more " +
  "often reflect lower-cervical pathology.";

const KNEE_EDU =
  "The knee is the largest hinge joint in the body, supporting full body " +
  "weight while allowing flexion/extension. Reduced knee flexion can " +
  "indicate post-surgical stiffness, meniscal injury, patellofemoral " +
  "pain, or arthritis; deficits in full extension (extension lag) are " +
  "common after ACL reconstruction or quadriceps weakness. Symmetry " +
  "between sides is a key indicator of recovery progress.";

const HIP_EDU =
  "The hip is a ball-and-socket joint with high stability and a wide " +
  "range of motion. Reduced hip flexion is associated with hip flexor " +
  "tightness, osteoarthritis, or femoroacetabular impingement; reduced " +
  "rotation often points to capsular tightness or labral pathology. " +
  "Hip extension limits frequently reflect anterior hip stiffness or " +
  "psoas dysfunction. Bilateral comparison is essential for screening.";

const ANKLE_EDU =
  "The ankle's dorsiflexion is critical for normal gait, squatting, " +
  "and stair descent. Reduced dorsiflexion (less than 10–15°) is a " +
  "common cause of compensatory knee, hip, and lower-back pain, and is " +
  "often seen after ankle sprains, achilles tightness, or post-cast " +
  "stiffness. Plantarflexion limits typically reflect anterior ankle " +
  "impingement or weakness in the calf complex.";

const PATIENT_ID_KEY = "motionlens.biomech_patient_id";

function getOrCreatePatientId(): string {
  if (typeof window === "undefined") return "";
  try {
    let id = sessionStorage.getItem(PATIENT_ID_KEY);
    if (!id) {
      const d = new Date();
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, "0");
      const dd = String(d.getDate()).padStart(2, "0");
      const rand = Math.random().toString(16).slice(2, 8).toUpperCase();
      id = `BIO-${yyyy}${mm}${dd}-${rand}`;
      sessionStorage.setItem(PATIENT_ID_KEY, id);
    }
    return id;
  } catch {
    return "";
  }
}

/** Asymmetric range-aware classification used across body parts.
 *
 *  Rules:
 *    • In range                        → good
 *    • Below range by ≤ 30% of width   → fair  (mild restriction)
 *    • Below range by > 30% of width   → poor  (notable restriction)
 *    • Above range by ≤ 30% of width   → good  (normal variation;
 *                                                exceeding the spec
 *                                                upper bound by a
 *                                                little is not an
 *                                                impairment)
 *    • Above range by 30-100% of width → fair  (notable hypermobility;
 *                                                worth flagging but
 *                                                rarely problematic)
 *    • Above range by > 100% of width  → poor  (anatomically suspect;
 *                                                likely a sign-flip /
 *                                                keypoint artefact)
 *
 *  Why asymmetric: ROM screening's clinical worry is RESTRICTION
 *  (below range), not having more range than average (above range).
 *  A patient with 70° shoulder extension when the spec range is
 *  [45, 60] has BETTER mobility than the population norm — calling
 *  that "poor" inverts the clinical interpretation. Below-range gets
 *  the strict thresholds; above-range gets graceful handling that
 *  only flips to poor when the value is so far past the spec it's
 *  almost certainly a measurement artefact.
 *
 *  `pct` stays the simple measured/upper-bound legacy percentage so
 *  the interpretation sentence reads consistently with prior reports. */
function classify(measured: number, target: [number, number]) {
  const [lo, hi] = target;
  if (hi <= 0 && lo <= 0) return { status: "poor" as const, pct: 0 };
  const denom = hi > 0 ? hi : 1;
  const legacyPct = Math.round((measured / denom) * 100);
  if (measured >= lo && measured <= hi) {
    return { status: "good" as const, pct: legacyPct };
  }
  const rangeWidth = Math.max(1, hi - lo);
  if (measured < lo) {
    const distFrac = (lo - measured) / rangeWidth;
    return distFrac <= 0.30
      ? { status: "fair" as const, pct: legacyPct }
      : { status: "poor" as const, pct: legacyPct };
  }
  // measured > hi
  const distFrac = (measured - hi) / rangeWidth;
  if (distFrac <= 0.30) return { status: "good" as const, pct: legacyPct };
  if (distFrac <= 1.00) return { status: "fair" as const, pct: legacyPct };
  return { status: "poor" as const, pct: legacyPct };
}

export function AssessmentReport({
  bodyPart,
  movementName,
  movementId,
  measured,
  target,
  side,
  patientNameOverride,
  patientIdOverride,
  dateOverride,
  patientOverride,
  keyFrames,
  secondaryMovementName,
  secondaryMeasured,
  secondaryTarget,
  compensations,
}: Props) {
  // Merged tests always render BOTH direction rows (so the operator
  // gets a clear "Internal Rotation: Not detected" line when only
  // external was performed, instead of a silently single-row report
  // titled with the combined name). Driven by secondaryMovementName +
  // secondaryTarget being set; the measured value can be missing.
  //
  // A measured value of 0 is a legitimate reading (e.g. knee
  // extension peak = 0° means perfect knee straightening), so the
  // "captured vs missing" check is type-based, not value-based. The
  // upstream component is responsible for passing `undefined` when
  // the secondary direction wasn't actually captured.
  const hasSecondary = !!secondaryMovementName && !!secondaryTarget;
  const hasSecondaryValue =
    hasSecondary && typeof secondaryMeasured === "number";
  // Patient identity comes from the doctor-flow context when the
  // assessment is launched from /dashboard/patients/{id}/analyze.
  // Outside the doctor flow we just render the report with no patient
  // header (or a session-scoped anonymous ID).
  const { patient: doctorPatient } = usePatientContext();
  const [sessionPatientId, setSessionPatientId] = useState("");

  useEffect(() => {
    if (patientIdOverride === undefined) {
      setSessionPatientId(getOrCreatePatientId());
    }
  }, [patientIdOverride]);

  // Resolve full patient object for the rich header. Preference order:
  //   1. patientOverride (saved-report viewer / explicit prop)
  //   2. doctor-flow context (live capture)
  //   3. null (anonymous)
  const fullPatient: PatientDTO | null =
    patientOverride !== undefined
      ? patientOverride
      : (doctorPatient as PatientDTO | null);

  const patient = patientNameOverride !== undefined
    ? (patientNameOverride ? { name: patientNameOverride, height_cm: 0 } : null)
    : doctorPatient
      ? { name: doctorPatient.name, height_cm: doctorPatient.height_cm }
      : null;

  const patientId = patientIdOverride !== undefined
    ? (patientIdOverride ?? "")
    : sessionPatientId;

  const { status, pct } = useMemo(() => classify(measured, target), [measured, target]);
  const secondaryStatus = useMemo(
    () =>
      hasSecondaryValue && secondaryTarget && typeof secondaryMeasured === "number"
        ? classify(secondaryMeasured, secondaryTarget)
        : null,
    [hasSecondaryValue, secondaryMeasured, secondaryTarget],
  );

  const today = new Date();
  const dateStr = dateOverride
    ?? `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

  const sideLabel = side ? side.charAt(0).toUpperCase() + side.slice(1) : "—";
  const rangeStr =
    target[0] === target[1] ? `${target[0]}°` : `${target[0]}°–${target[1]}°`;
  const secondaryRangeStr =
    secondaryTarget
      ? secondaryTarget[0] === secondaryTarget[1]
        ? `${secondaryTarget[0]}°`
        : `${secondaryTarget[0]}°–${secondaryTarget[1]}°`
      : "";
  const bodyPartCap = bodyPart.charAt(0).toUpperCase() + bodyPart.slice(1);
  const eduText =
    bodyPart === "shoulder"
      ? SHOULDER_EDU
      : bodyPart === "neck"
        ? NECK_EDU
        : bodyPart === "knee"
          ? KNEE_EDU
          : bodyPart === "hip"
            ? HIP_EDU
            : ANKLE_EDU;

  // ── Plotly chart: Normal range band + Measured bar (matches biomech_flow.py) ──
  // For merged movements (secondaryMeasured present) we render a second
  // pair of bars on a second y-row so both directions are visible side
  // by side, each compared to its own normal range band.
  const chartData = useMemo(() => {
    const low = target[0];
    const high = target[1];
    const collapsed = low === high;
    const bandStart = collapsed ? Math.max(0, low - 1) : low;
    const bandLength = collapsed ? 2 : high - low;

    const yLabels = hasSecondary && secondaryMovementName
      ? [secondaryMovementName, movementName]
      : [movementName];
    // For merged tests, render the secondary bar at 0 when the
    // direction wasn't detected — keeps the chart's two-row layout
    // and aligns visually with the "Not detected" row in the table.
    const measuredXs = hasSecondary
      ? [hasSecondaryValue && typeof secondaryMeasured === "number" ? secondaryMeasured : 0, measured]
      : [measured];

    const lowB = secondaryTarget?.[0] ?? 0;
    const highB = secondaryTarget?.[1] ?? 0;
    const collapsedB = lowB === highB;
    const bandStartB = collapsedB ? Math.max(0, lowB - 1) : lowB;
    const bandLengthB = collapsedB ? 2 : highB - lowB;

    const rangeBands = hasSecondary
      ? [bandLengthB, bandLength]
      : [bandLength];
    const rangeBases = hasSecondary
      ? [bandStartB, bandStart]
      : [bandStart];
    const rangeTexts = hasSecondary
      ? [collapsedB ? `target ${lowB}°` : secondaryRangeStr,
         collapsed ? `target ${low}°` : rangeStr]
      : [collapsed ? `target ${low}°` : rangeStr];

    return [
      {
        type: "bar" as const,
        orientation: "h" as const,
        name: "Normal range",
        y: yLabels,
        x: rangeBands,
        base: rangeBases,
        marker: {
          color: "rgba(234,88,12,0.30)",
          line: { color: "#EA580C", width: 1.5 },
        },
        text: rangeTexts,
        textposition: "inside" as const,
        textfont: { color: "#0F172A", size: 12 },
        hovertemplate: `<b>%{y}</b><br>Normal: %{text}<extra></extra>`,
      },
      {
        type: "bar" as const,
        orientation: "h" as const,
        name: "Measured",
        y: yLabels,
        x: measuredXs,
        marker: { color: "#2563EB" },
        text: measuredXs.map((v) => `${v.toFixed(1)}°`),
        textposition: "outside" as const,
        textfont: { color: "#0F172A", size: 12 },
        hovertemplate: `<b>%{y}</b><br>Measured: %{x:.1f}°<extra></extra>`,
      },
    ];
  }, [
    movementName, target, measured, rangeStr,
    hasSecondary, secondaryMovementName, secondaryMeasured, secondaryTarget, secondaryRangeStr,
  ]);

  const xMax =
    Math.max(
      measured,
      target[1],
      hasSecondary && typeof secondaryMeasured === "number" ? secondaryMeasured : 0,
      hasSecondary && secondaryTarget ? secondaryTarget[1] : 0,
      1,
    ) * 1.15;

  const chartLayout = {
    barmode: "group" as const,
    margin: { l: 140, r: 50, t: 60, b: 60 },
    legend: {
      orientation: "h" as const,
      y: 1.18,
      x: 0.5,
      xanchor: "center" as const,
      bgcolor: "rgba(0,0,0,0)",
      font: { color: "#CBD5E1", size: 12 },
    },
    xaxis: {
      title: { text: "Angle (°)", standoff: 10 },
      range: [0, xMax],
    },
    yaxis: {
      autorange: "reversed" as const,
    },
  };

  const interpSentence =
    `${movementName}${side ? ` (${sideLabel})` : ""} measured ${measured.toFixed(1)}°, ` +
    `which is ${Math.round(pct)}% of the ${rangeStr} normal range — ${status}.`;
  const interpSentenceB = hasSecondary && secondaryMovementName
    ? hasSecondaryValue &&
      secondaryStatus &&
      typeof secondaryMeasured === "number"
      ? `${secondaryMovementName}${side ? ` (${sideLabel})` : ""} measured ${secondaryMeasured.toFixed(1)}°, ` +
        `which is ${Math.round(secondaryStatus.pct)}% of the ${secondaryRangeStr} normal range — ${secondaryStatus.status}.`
      : `${secondaryMovementName}${side ? ` (${sideLabel})` : ""} was not detected in this trial — perform that direction and re-record to capture it.`
    : null;

  const interpStyles =
    status === "good"
      ? "border-accent/40 bg-accent/5"
      : status === "fair"
        ? "border-warning/40 bg-warning/5"
        : "border-error/40 bg-error/5";

  return (
    <div className="space-y-8">
      <h2 className="text-center text-3xl font-semibold tracking-tight md:text-4xl">
        Assessment Report
      </h2>

      {/* ── Patient header ─────────────────────────────────────── */}
      {fullPatient ? (
        <PatientHeader
          patient={fullPatient}
          subtitle={`ID: ${patientId} · Body part: ${bodyPartCap}`}
          date={dateStr}
        />
      ) : (
        <div className="rounded-card border border-border bg-surface px-5 py-4 text-sm">
          <span className="font-semibold text-foreground">
            {patient?.name?.trim() || "Anonymous patient"}
          </span>
          <span className="text-muted"> · ID: </span>
          <span className="tabular text-foreground">{patientId}</span>
          <span className="text-muted"> · Date: </span>
          <span className="tabular text-foreground">{dateStr}</span>
          <span className="text-muted"> · Body part: </span>
          <span className="font-semibold text-foreground">{bodyPartCap}</span>
        </div>
      )}

      {/* ── Results table ──────────────────────────────────────── */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Results</h3>
        <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
              <tr>
                <th className="px-5 py-3 font-medium">Movement</th>
                <th className="px-5 py-3 text-center font-medium">Side</th>
                <th className="px-5 py-3 text-right font-medium">Measured</th>
                <th className="px-5 py-3 text-right font-medium">Normal</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-5 py-4 text-foreground">{movementName}</td>
                <td className="px-5 py-4 text-center text-muted">{sideLabel}</td>
                <td className="px-5 py-4 text-right tabular text-foreground">
                  {fmt(measured, 1)}°
                </td>
                <td className="px-5 py-4 text-right tabular text-muted">{rangeStr}</td>
              </tr>
              {hasSecondary && secondaryMovementName && (
                <tr className="border-t border-border/60">
                  <td className="px-5 py-4 text-foreground">{secondaryMovementName}</td>
                  <td className="px-5 py-4 text-center text-muted">{sideLabel}</td>
                  <td className="px-5 py-4 text-right tabular text-foreground">
                    {hasSecondaryValue && typeof secondaryMeasured === "number"
                      ? `${fmt(secondaryMeasured, 1)}°`
                      : <span className="italic text-subtle">Not detected</span>}
                  </td>
                  <td className="px-5 py-4 text-right tabular text-muted">
                    {secondaryRangeStr}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Plotly bar chart ───────────────────────────────────── */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Measured vs Normal</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-4">
          <PlotlyChart data={chartData} layout={chartLayout} height={220} />
        </div>
      </section>

      {/* ── Compensations Detected ─────────────────────────────── */}
      {/* Always renders all 3 compensation entries when present —
          flagged ones as colored cards with severity tag, unflagged
          ones as muted "within range" rows. Both styles include the
          peak measured value (in `details`) so clinicians can see how
          close to the threshold the trial landed, and sub-threshold
          movement is visible rather than silently hidden. */}
      {compensations && compensations.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Compensations Detected
          </h3>
          <div className="mt-3 space-y-3">
            {compensations.map((c) => {
              if (c.flagged) {
                const styles =
                  c.severity === "high"
                    ? "border-error/40 bg-error/5"
                    : c.severity === "medium"
                      ? "border-warning/40 bg-warning/5"
                      : "border-border bg-surface";
                const dot =
                  c.severity === "high"
                    ? "bg-error"
                    : c.severity === "medium"
                      ? "bg-warning"
                      : "bg-muted";
                const tag =
                  c.severity === "high"
                    ? "HIGH"
                    : c.severity === "medium"
                      ? "MEDIUM"
                      : "LOW";
                return (
                  <div
                    key={c.type}
                    className={`rounded-card border p-4 ${styles}`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${dot}`} />
                        <span className="text-sm font-semibold text-foreground">
                          {c.label}
                        </span>
                      </div>
                      <span className="rounded-full bg-foreground/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-foreground">
                        {tag}
                      </span>
                    </div>
                    {c.details && (
                      <p className="mt-2 text-xs text-muted">{c.details}</p>
                    )}
                  </div>
                );
              }
              // Unflagged — muted row showing the measured peak value.
              return (
                <div
                  key={c.type}
                  className="rounded-card border border-border bg-surface p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-accent" />
                      <span className="text-sm font-semibold text-foreground">
                        {c.label}
                      </span>
                    </div>
                    <span className="rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-accent">
                      Within range
                    </span>
                  </div>
                  {c.details && (
                    <p className="mt-2 text-xs text-muted">{c.details}</p>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── Annotated key frames (images only, no captions) ─────── */}
      {keyFrames && keyFrames.length > 0 && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">Key frames</h3>
          <div className="mt-3 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {keyFrames.map((kf) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={kf.frame_index}
                src={kf.image_data_url}
                alt={kf.label}
                className="block w-full overflow-hidden rounded-card border border-border"
              />
            ))}
          </div>
        </section>
      )}

      {/* ── Clinical interpretation ────────────────────────────── */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className={`mt-3 rounded-card border p-5 ${interpStyles}`}>
          <p className="text-sm leading-relaxed text-foreground">{interpSentence}</p>
          {interpSentenceB && (
            <p className="mt-2 border-t border-border/40 pt-2 text-sm leading-relaxed text-foreground">
              {interpSentenceB}
            </p>
          )}
        </div>
      </section>

      {/* ── Educational block ──────────────────────────────────── */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">
          About {bodyPart} biomechanics
        </h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5">
          <p className="text-sm leading-relaxed text-muted">{eduText}</p>
        </div>
      </section>

      {/* ── Unified report disclaimer ──────────────────────────── */}
      <ReportDisclaimer />
    </div>
  );
}
