"use client";
// Patient create/edit form. Used by:
//   /dashboard/patients/new  — create (no `initial`)
//   /dashboard/patients/[id] — edit (with `initial`)

import { useState } from "react";
import { Loader2, Save, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import type { PatientCreatePayload, PatientDTO } from "@/lib/patients";

interface Props {
  initial?: PatientDTO;
  onSubmit: (payload: PatientCreatePayload) => Promise<void>;
  submitLabel?: string;
}

export function PatientForm({ initial, onSubmit, submitLabel = "Save patient" }: Props) {
  const [name, setName] = useState(initial?.name ?? "");
  const [age, setAge] = useState<string>(initial?.age?.toString() ?? "");
  const [gender, setGender] = useState<"male" | "female" | "other">(
    (initial?.gender as "male" | "female" | "other") ?? "female",
  );
  const [heightCm, setHeightCm] = useState<string>(initial?.height_cm?.toString() ?? "");
  const [weightKg, setWeightKg] = useState<string>(
    initial?.weight_kg ? initial.weight_kg.toString() : "",
  );
  const [contact, setContact] = useState(initial?.contact ?? "");
  const [notes, setNotes] = useState(initial?.medical_notes ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handle(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await onSubmit({
        name: name.trim(),
        age: parseInt(age, 10),
        gender,
        height_cm: parseFloat(heightCm),
        weight_kg: weightKg ? parseFloat(weightKg) : null,
        contact: contact.trim() || null,
        medical_notes: notes.trim() || null,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setBusy(false);
    }
  }

  const valid = name && age && heightCm;

  return (
    <form onSubmit={handle} className="space-y-6">
      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        {/* Name */}
        <Field label="Full name" required>
          <input
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Priya Sharma"
            className={inputCls}
          />
        </Field>

        {/* Age */}
        <Field label="Age" required>
          <input
            type="number"
            required
            min={0}
            max={150}
            value={age}
            onChange={(e) => setAge(e.target.value)}
            placeholder="e.g. 32"
            className={inputCls}
          />
        </Field>

        {/* Gender */}
        <Field label="Gender" required>
          <select
            value={gender}
            onChange={(e) => setGender(e.target.value as "male" | "female" | "other")}
            className={inputCls}
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </Field>

        {/* Height */}
        <Field label="Height (cm)" required>
          <input
            type="number"
            required
            step="0.1"
            min={30}
            max={250}
            value={heightCm}
            onChange={(e) => setHeightCm(e.target.value)}
            placeholder="e.g. 165"
            className={inputCls}
          />
        </Field>

        {/* Weight (optional) */}
        <Field label="Weight (kg) — optional">
          <input
            type="number"
            step="0.1"
            min={2}
            max={400}
            value={weightKg}
            onChange={(e) => setWeightKg(e.target.value)}
            placeholder="e.g. 62"
            className={inputCls}
          />
        </Field>

        {/* Contact (optional) */}
        <Field label="Contact — optional">
          <input
            type="text"
            value={contact}
            onChange={(e) => setContact(e.target.value)}
            placeholder="e.g. +91-9876543210"
            className={inputCls}
          />
        </Field>
      </div>

      {/* Medical notes (full-width) */}
      <Field label="Medical notes — optional">
        <textarea
          rows={4}
          maxLength={2000}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Brief medical history, current concerns, prior interventions, etc."
          className={`${inputCls} resize-none`}
        />
      </Field>

      {/* Submit */}
      <div className="flex justify-end">
        <Button type="submit" disabled={busy || !valid}>
          {busy ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Saving…
            </>
          ) : (
            <>
              <Save className="h-4 w-4" />
              {submitLabel}
            </>
          )}
        </Button>
      </div>
    </form>
  );
}

const inputCls =
  "w-full rounded-card border border-border bg-surface px-4 py-2.5 text-sm text-foreground transition focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20";

function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {label} {required && <span className="text-error">*</span>}
      </label>
      {children}
    </div>
  );
}
