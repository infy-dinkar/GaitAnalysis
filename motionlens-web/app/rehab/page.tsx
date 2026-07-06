"use client";
// Public rehab catalogue — 24 exercises grouped by joint family
// (Knee / Hip / Back / Shoulder). Card definitions live in the
// shared lib/rehab/exerciseCatalog.js so both this page and the
// doctor-flow launcher stay in sync.
//
// Logged-in clinicians can pick a patient at the top of the page —
// picking one appends ?patientId=xxx to every exercise link so the
// full doctor-flow (Save, Level chip, Compensation flags, Progress)
// activates without having to go through /dashboard first.

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { ArrowUpRight, UserRound, LineChart } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";
import { groupExercisesByJoint } from "@/lib/rehab/exerciseCatalog";
import { useAuth } from "@/contexts/AuthContext";
import { listPatients, type PatientDTO } from "@/lib/patients";

export default function RehabCataloguePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const groups = groupExercisesByJoint();
  const { doctor } = useAuth();
  const [patients, setPatients] = useState<PatientDTO[] | null>(null);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("");

  useEffect(() => {
    if (!doctor) {
      setPatients(null);
      setSelectedPatientId("");
      return;
    }
    let cancelled = false;
    listPatients()
      .then((res) => {
        if (!cancelled) setPatients(res.data);
      })
      .catch(() => {
        if (!cancelled) setPatients([]);
      });
    return () => {
      cancelled = true;
    };
  }, [doctor]);

  const selectedPatient =
    patients?.find((p) => p.id === selectedPatientId) ?? null;
  const patientQuery = selectedPatientId
    ? `?patientId=${selectedPatientId}`
    : "";

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div className="max-w-2xl">
              <Badge>Rehab catalogue</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Game-based therapy<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Twenty-four wired exercises across the knee, hip,
                back, and shoulder families — every one powered by
                the same BlazePose pipeline that drives the
                assessments. Each game wraps a pure scoring engine
                (hold-in-zone, rep-count gate, target-reach, trace,
                weight-shift, match-pose, metronome) plugged into
                the joint movement clinicians actually prescribe.
              </p>
            </div>
            <Link href="/">
              <Button variant="ghost" size="sm">← Home</Button>
            </Link>
          </div>

          {/* Clinician-only patient picker. Not shown to logged-out
              visitors — the catalogue works fine without saving. */}
          {doctor && (
            <div className="mt-10 rounded-card border border-border bg-surface p-5">
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-card bg-accent/10 text-accent">
                  <UserRound className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-foreground">
                    Playing on behalf of…
                  </p>
                  <p className="mt-0.5 text-xs text-muted">
                    Pick a patient to unlock save, level progression,
                    compensation flags, and the progress dashboard.
                    Leave blank for a solo demo run.
                  </p>
                </div>
                <select
                  value={selectedPatientId}
                  onChange={(e) => setSelectedPatientId(e.target.value)}
                  className="min-w-[200px] rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-accent"
                >
                  <option value="">— Solo (no patient) —</option>
                  {(patients ?? []).map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
              {selectedPatient && (
                <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-border pt-4">
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-emerald-500/30 dark:text-emerald-300">
                    Saving to {selectedPatient.name}
                  </span>
                  <Link
                    href={`/dashboard/patients/${selectedPatient.id}/rehab`}
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-accent hover:underline"
                  >
                    <LineChart className="h-3.5 w-3.5" />
                    View progress dashboard
                  </Link>
                </div>
              )}
            </div>
          )}

          <div className="mt-16 space-y-16">
            {groups.map(({ joint, meta, items }) => (
              <section key={joint}>
                <div className="flex items-baseline justify-between">
                  <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
                    {meta.label}
                  </h2>
                  <p className="text-sm text-muted">{meta.subtitle}</p>
                </div>
                <div className="mt-6 grid gap-5 md:grid-cols-3">
                  {items.map((m) => {
                    const Icon = m.icon;
                    const imageUrl = REHAB_EXERCISE_IMAGES[m.slug];
                    const href = `/rehab/${m.slug}${patientQuery}`;
                    const eyebrow = `${m.code} · ${m.title}`;
                    const sharedClass = `group relative flex flex-col overflow-hidden rounded-hero border border-border bg-gradient-to-br ${m.tone} p-6 transition md:p-8 hover:border-accent hover:shadow-glow-sm`;
                    return (
                      <Link key={m.slug} href={href} className={sharedClass}>
                        {imageUrl && (
                          <div className="mb-3 w-full overflow-hidden rounded-md bg-white">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={imageUrl}
                              alt=""
                              aria-hidden="true"
                              loading="lazy"
                              className="block h-28 w-full object-contain"
                            />
                          </div>
                        )}
                        <div className="flex items-center justify-between">
                          <Icon className={`h-7 w-7 ${m.iconTone}`} />
                          <ArrowUpRight className="h-5 w-5 text-muted transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-accent" />
                        </div>
                        <div className="mt-8">
                          <p className="eyebrow">{eyebrow}</p>
                          <h3 className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">
                            {m.title}
                          </h3>
                          <p className="mt-3 text-sm leading-relaxed text-muted">
                            {m.publicBody}
                          </p>
                        </div>
                      </Link>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">For clinicians</p>
            <p className="mt-2">
              Rehab games share the patient context the assessment
              modules already use — open this catalogue from a
              patient&apos;s record and any session played here saves
              against that patient&apos;s history once an exercise is
              wired into a mechanic.
            </p>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
