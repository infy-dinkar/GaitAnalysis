"use client";
// /dashboard/patients — full patient list
//
// Two entry paths land here:
//
//   1. Normal dashboard browsing — the doctor clicks "Patients" in the
//      sidebar. Picking a patient opens that patient's profile, which
//      is the long-standing behaviour we don't want to break.
//
//   2. Landing-page module card — the doctor clicked e.g. "Gait
//      analysis" on the marketing landing page. ProductShowcase stashes
//      the module route in sessionStorage under INTENDED_MODULE_KEY and
//      sends them here so they can attach the upcoming assessment to a
//      patient. In that case picking a patient must forward straight
//      to /<module>?patientId=<id> — NOT the profile page — so the
//      doctor lands on the module they actually chose.
//
// We read sessionStorage exactly once on mount, copy it into local
// state, and clear sessionStorage immediately so a stale value can't
// hijack a later, unrelated visit. A null intendedModule means we
// fall back to the legacy patient-profile navigation in PatientCard.

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { UserPlus, Users, Search } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { PatientCard } from "@/components/dashboard/PatientCard";
import { Button } from "@/components/ui/Button";
import { listPatients, type PatientDTO } from "@/lib/patients";

// Must stay in sync with the key written by ProductShowcase.tsx.
const INTENDED_MODULE_KEY = "motionlens.intendedModule";

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
  const router = useRouter();
  const [patients, setPatients] = useState<PatientDTO[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  // Captured on mount, never refreshed. Stays valid for exactly one
  // patient pick from this page load.
  const [intendedModule, setIntendedModule] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listPatients()
      .then((r) => !cancelled && setPatients(r.data))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  // Read+consume the intended-module marker on mount. Clearing sessionStorage
  // here means a later sidebar visit to /dashboard/patients reverts to the
  // legacy patient-profile flow with no residue from a prior landing click.
  useEffect(() => {
    try {
      const stashed = sessionStorage.getItem(INTENDED_MODULE_KEY);
      if (stashed) {
        setIntendedModule(stashed);
        sessionStorage.removeItem(INTENDED_MODULE_KEY);
      }
    } catch {
      // sessionStorage unavailable (private mode etc.) — fall back to
      // the legacy patient-profile flow.
    }
  }, []);

  // Capture-phase click handler on the grid wrapper. React fires
  // capture handlers top-down BEFORE descendant bubble handlers, so we
  // can short-circuit the PatientCard's <Link> navigation without
  // having to touch PatientCard itself. When no intendedModule is set
  // we leave the event alone — the Link runs normally and routes to
  // the patient profile, preserving the dashboard flow.
  function handleGridClickCapture(e: React.MouseEvent<HTMLDivElement>) {
    if (!intendedModule) return;
    // Resolve the patient card the click originated from. PatientCard
    // renders <Link href="/dashboard/patients/<id>"> as its root, so
    // walking up to the nearest anchor gives us the id without needing
    // any wiring on the card itself.
    const target = e.target as HTMLElement | null;
    const anchor = target?.closest?.("a[href^='/dashboard/patients/']");
    if (!(anchor instanceof HTMLAnchorElement)) return;

    const href = anchor.getAttribute("href") ?? "";
    // Pull just the id segment so we don't accidentally forward a
    // /dashboard/patients/new click or anything else.
    const match = href.match(/^\/dashboard\/patients\/([^/?#]+)$/);
    if (!match) return;
    const patientId = match[1];
    if (patientId === "new") return;

    e.preventDefault();
    e.stopPropagation();
    // Consume on first use so a double-click doesn't fire twice and a
    // future visit can't accidentally inherit this value.
    setIntendedModule(null);
    router.push(`${intendedModule}?patientId=${patientId}`);
  }

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

      {/* Inline notice when we're going to forward to a module after
          patient pick. Without this the doctor would be surprised that
          clicking a patient skips the profile. */}
      {intendedModule && (
        <div className="rounded-card border border-accent/30 bg-accent/5 px-4 py-3 text-sm">
          <p className="text-foreground">
            Pick a patient to launch{" "}
            <span className="font-medium text-accent">{intendedModule}</span>{" "}
            for them.
          </p>
        </div>
      )}

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
        <div
          className="grid gap-4 md:grid-cols-2 lg:grid-cols-3"
          onClickCapture={handleGridClickCapture}
        >
          {filtered.map((p) => (
            <PatientCard key={p.id} patient={p} />
          ))}
        </div>
      )}
    </div>
  );
}
