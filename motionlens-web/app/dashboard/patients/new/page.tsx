"use client";
// /dashboard/patients/new — add a new patient

import { useRouter } from "next/navigation";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { PatientForm } from "@/components/dashboard/PatientForm";
import { createPatient } from "@/lib/patients";

export default function NewPatientPage() {
  return (
    <AuthGuard>
      <DashboardShell backHref="/dashboard/patients" backLabel="Patients" title="Add patient">
        <NewPatientContent />
      </DashboardShell>
    </AuthGuard>
  );
}

function NewPatientContent() {
  const router = useRouter();

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
          New patient
        </h1>
        <p className="mt-1 text-sm text-muted">
          Add a patient record. You can edit any field later.
        </p>
      </div>

      <div className="rounded-card border border-border bg-surface p-6 md:p-8">
        <PatientForm
          submitLabel="Create patient"
          onSubmit={async (payload) => {
            const created = await createPatient(payload);
            router.replace(`/dashboard/patients/${created.id}`);
          }}
        />
      </div>
    </div>
  );
}
