"use client";
// /dashboard/patients — full patient list

import { useEffect, useState } from "react";
import Link from "next/link";
import { UserPlus, Users, Search } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { PatientCard } from "@/components/dashboard/PatientCard";
import { Button } from "@/components/ui/Button";
import { listPatients, type PatientDTO } from "@/lib/patients";

export default function PatientsListPage() {
  return (
    <AuthGuard>
      <DashboardShell title="Patients">
        <ListContent />
      </DashboardShell>
    </AuthGuard>
  );
}

function ListContent() {
  const [patients, setPatients] = useState<PatientDTO[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    listPatients()
      .then((r) => !cancelled && setPatients(r.data))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered =
    patients === null
      ? null
      : patients.filter((p) => {
          if (!query) return true;
          const q = query.toLowerCase();
          return (
            p.name.toLowerCase().includes(q) ||
            p.contact?.toLowerCase().includes(q) ||
            p.medical_notes?.toLowerCase().includes(q)
          );
        });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">
            All patients
          </h1>
          <p className="mt-1 text-sm text-muted">
            {patients ? `${patients.length} total` : "Loading…"}
          </p>
        </div>
        <Link href="/dashboard/patients/new">
          <Button>
            <UserPlus className="h-4 w-4" />
            Add patient
          </Button>
        </Link>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by name, contact, or notes…"
          className="w-full rounded-card border border-border bg-surface px-10 py-2.5 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
        />
      </div>

      {error && (
        <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
          {error}
        </div>
      )}

      {/* Grid */}
      {filtered === null ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="h-28 animate-pulse rounded-card border border-border bg-surface"
            />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="rounded-card border border-dashed border-border bg-surface/50 px-6 py-12 text-center">
          <Users className="mx-auto h-8 w-8 text-muted" />
          <h3 className="mt-3 text-base font-semibold text-foreground">
            {query ? "No matching patients" : "No patients yet"}
          </h3>
          <p className="mt-1 text-sm text-muted">
            {query
              ? "Try a different search term."
              : "Add your first patient to start running assessments."}
          </p>
          {!query && (
            <Link href="/dashboard/patients/new" className="mt-4 inline-block">
              <Button>
                <UserPlus className="h-4 w-4" />
                Add first patient
              </Button>
            </Link>
          )}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filtered.map((p) => (
            <PatientCard key={p.id} patient={p} />
          ))}
        </div>
      )}
    </div>
  );
}
