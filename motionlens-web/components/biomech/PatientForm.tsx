"use client";
import { useEffect, useState } from "react";

export interface PatientInfo {
  name: string;
  age: string;
  gender: string;
}

const STORAGE_KEY = "motionlens.patient";

export function loadPatient(): PatientInfo | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as PatientInfo) : null;
  } catch {
    return null;
  }
}

export function savePatient(p: PatientInfo) {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(p));
}

interface PatientFormProps {
  onSubmit?: (p: PatientInfo) => void;
}

const inputClass =
  "h-11 w-full rounded-lg border border-border bg-surface px-3 text-sm text-foreground placeholder:text-subtle transition focus:border-accent focus:outline-none";

export function PatientForm({ onSubmit }: PatientFormProps) {
  const [info, setInfo] = useState<PatientInfo>({ name: "", age: "", gender: "" });

  useEffect(() => {
    const existing = loadPatient();
    if (existing) setInfo(existing);
  }, []);

  function handle(k: keyof PatientInfo, v: string) {
    const next = { ...info, [k]: v };
    setInfo(next);
    savePatient(next);
    onSubmit?.(next);
  }

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Field label="Patient name">
        <input
          value={info.name}
          onChange={(e) => handle("name", e.target.value)}
          placeholder="e.g. Priya Sharma"
          className={inputClass}
        />
      </Field>
      <Field label="Age">
        <input
          type="number"
          min={0}
          max={120}
          value={info.age}
          onChange={(e) => handle("age", e.target.value)}
          placeholder="34"
          className={inputClass}
        />
      </Field>
      <Field label="Gender">
        <select
          value={info.gender}
          onChange={(e) => handle("gender", e.target.value)}
          className={inputClass}
        >
          <option value="">—</option>
          <option value="female">Female</option>
          <option value="male">Male</option>
          <option value="other">Other</option>
        </select>
      </Field>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">{label}</span>
      <div className="mt-2">{children}</div>
    </label>
  );
}
