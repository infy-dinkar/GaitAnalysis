// Shared patient-detail header rendered at the top of every report
// (live capture, saved-report viewer, downloaded PDF). Surfaces the
// full set of fields captured at registration so the doctor can see
// the patient context without leaving the report.
//
// Used by all 9 report modules — Trendelenburg, Single-Leg Squat,
// 5xSTS, 30s Chair Stand, Single-Leg Stance, 4-Stage Balance,
// Biomechanics ROM, Posture, Gait. Each report accepts a
// `patient?: PatientDTO | null` prop and renders this header above
// its module-specific content.
//
// When `patient` is null (anonymous / non-doctor flow), the header
// gracefully degrades to just `fallbackName` so existing screens
// without patient context still look polished.

import type { PatientDTO } from "@/lib/patients";

interface Props {
  patient: PatientDTO | null;
  /** Used as the heading when no patient object is supplied. */
  fallbackName?: string | null;
  /** Optional subtitle shown after the demographics line — e.g.
   *  "5x Sit-to-Stand · 5 reps". */
  subtitle?: string | null;
  /** When provided, renders this date next to the demographics; falls
   *  back to today's date in the local timezone. */
  date?: Date | string | null;
}

const GENDER_LABEL: Record<PatientDTO["gender"], string> = {
  male: "Male",
  female: "Female",
  other: "Other",
};

export function PatientHeader({
  patient,
  fallbackName,
  subtitle,
  date,
}: Props) {
  const dateStr = formatDate(date ?? new Date());

  if (!patient) {
    if (!fallbackName?.trim()) return null;
    return (
      <section className="rounded-card border border-border bg-surface px-5 py-4">
        <p className="text-lg font-semibold text-foreground">{fallbackName}</p>
        <p className="mt-1 text-xs text-muted">{dateStr}</p>
        {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
      </section>
    );
  }

  // Build the demographics line: "28 y · Female · 165 cm · 58 kg".
  const demographics: string[] = [];
  demographics.push(`${patient.age} y`);
  demographics.push(GENDER_LABEL[patient.gender]);
  demographics.push(`${patient.height_cm} cm`);
  if (patient.weight_kg !== null && patient.weight_kg !== undefined) {
    demographics.push(`${patient.weight_kg} kg`);
  }

  return (
    <section className="rounded-card border border-border bg-surface px-5 py-4">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <p className="text-lg font-semibold text-foreground">{patient.name}</p>
          <p className="mt-1 text-sm text-muted">
            {demographics.join(" · ")}
          </p>
        </div>
        <p className="text-xs text-subtle tabular">{dateStr}</p>
      </div>

      {subtitle && (
        <p className="mt-2 text-xs text-muted">{subtitle}</p>
      )}

      {(patient.contact || patient.medical_notes) && (
        <dl className="mt-3 space-y-1 border-t border-border pt-3 text-xs">
          {patient.contact && (
            <Row label="Contact" value={patient.contact} />
          )}
          {patient.medical_notes && (
            <Row label="Medical notes" value={patient.medical_notes} />
          )}
        </dl>
      )}
    </section>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-wrap gap-2">
      <dt className="font-semibold uppercase tracking-[0.12em] text-subtle">
        {label}:
      </dt>
      <dd className="text-foreground">{value}</dd>
    </div>
  );
}

function formatDate(d: Date | string): string {
  const date = typeof d === "string" ? new Date(d) : d;
  if (isNaN(date.getTime())) return "";
  // YYYY-MM-DD in local timezone — matches the existing AssessmentReport
  // header convention and reads cleanly on both screen + PDF.
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}
