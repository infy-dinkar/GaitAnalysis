"use client";
import { useEffect, useMemo, useState } from "react";
import { PlotlyChart } from "@/components/gait/PlotlyChart";
import { fmt } from "@/lib/utils";
import { usePatientContext } from "@/hooks/usePatientContext";

interface Props {
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle";
  /** Just the movement name — e.g. "Lateral Flexion", "Flexion". */
  movementName: string;
  movementId: string;
  measured: number;
  target: [number, number];
  side?: "left" | "right";
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

function isRotationMovement(
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle",
  id: string,
) {
  if (bodyPart === "shoulder" || bodyPart === "hip")
    return id === "external_rotation" || id === "internal_rotation";
  if (bodyPart === "neck") return id === "rotation";
  return false;
}

/** Mirrors biomech_flow._classify in the Streamlit app. */
function classify(measured: number, target: [number, number]) {
  // Use the upper bound as the canonical target when the range is wide,
  // matching SHOULDER_NORMAL_RANGES["flexion"]["target"] = 180 type values.
  const t = target[1];
  if (t <= 0) return { status: "poor" as const, pct: 0 };
  const pct = (measured / t) * 100;
  if (pct >= 90) return { status: "good" as const, pct };
  if (pct >= 75) return { status: "fair" as const, pct };
  return { status: "poor" as const, pct };
}

export function AssessmentReport({
  bodyPart,
  movementName,
  movementId,
  measured,
  target,
  side,
}: Props) {
  // Patient identity comes from the doctor-flow context when the
  // assessment is launched from /dashboard/patients/{id}/analyze.
  // Outside the doctor flow we just render the report with no patient
  // header (or a session-scoped anonymous ID).
  const { patient: doctorPatient } = usePatientContext();
  const [patientId, setPatientId] = useState("");

  useEffect(() => {
    setPatientId(getOrCreatePatientId());
  }, []);

  const patient = doctorPatient
    ? { name: doctorPatient.name, height_cm: doctorPatient.height_cm }
    : null;

  const { status, pct } = useMemo(() => classify(measured, target), [measured, target]);
  const isRotation = isRotationMovement(bodyPart, movementId);

  const today = new Date();
  const dateStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

  const sideLabel = side ? side.charAt(0).toUpperCase() + side.slice(1) : "—";
  const rangeStr =
    target[0] === target[1] ? `${target[0]}°` : `${target[0]}°–${target[1]}°`;
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
  const chartData = useMemo(() => {
    const low = target[0];
    const high = target[1];
    const collapsed = low === high;
    const bandStart = collapsed ? Math.max(0, low - 1) : low;
    const bandLength = collapsed ? 2 : high - low;

    return [
      {
        type: "bar" as const,
        orientation: "h" as const,
        name: "Normal range",
        y: [movementName],
        x: [bandLength],
        base: [bandStart],
        marker: {
          color: "rgba(234,88,12,0.30)",
          line: { color: "#EA580C", width: 1.5 },
        },
        text: [collapsed ? `target ${low}°` : rangeStr],
        textposition: "inside" as const,
        textfont: { color: "#0F172A", size: 12 },
        hovertemplate: `<b>${movementName}</b><br>Normal: ${rangeStr}<extra></extra>`,
      },
      {
        type: "bar" as const,
        orientation: "h" as const,
        name: "Measured",
        y: [movementName],
        x: [measured],
        marker: { color: "#2563EB" },
        text: [`${measured.toFixed(1)}°`],
        textposition: "outside" as const,
        textfont: { color: "#0F172A", size: 12 },
        hovertemplate: `<b>${movementName}</b><br>Measured: %{x:.1f}°<extra></extra>`,
      },
    ];
  }, [movementName, target, measured, rangeStr]);

  const xMax = Math.max(measured, target[1], 1) * 1.15;

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

      {/* ── Patient header strip ───────────────────────────────── */}
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

      {/* ── Clinical interpretation ────────────────────────────── */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className={`mt-3 rounded-card border p-5 ${interpStyles}`}>
          <p className="text-sm leading-relaxed text-foreground">{interpSentence}</p>
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

      {/* ── Subtle disclaimer footer ───────────────────────────── */}
      <p className="border-t border-border/60 pt-4 text-center text-[11px] leading-relaxed text-subtle/80">
        Measurements are estimated from a single 2D camera and are intended for
        movement tracking and self-screening
        {isRotation ? " — rotation values are approximate. " : ". "}
        For clinical-grade range-of-motion assessment, please consult a qualified
        practitioner.
      </p>
    </div>
  );
}
