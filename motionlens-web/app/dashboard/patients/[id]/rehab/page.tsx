"use client";
// Dashboard rehab launcher — parallel to /analyze/page.tsx. Shows
// the rehab exercise catalogue with the patient context already
// attached, so any session played from here saves against the
// patient's record.
//
// Exercise definitions live in the shared lib/rehab/exerciseCatalog
// so this page and the public /rehab catalogue stay in sync.
//
// Auto-recommendations: on mount we call useRecommendations(patientId)
// which reads getPrescribedSet — today that returns the auto ranked
// set, later it will prefer a doctor-saved prescription without any
// UI change here.

import { useEffect, useMemo, useState, use as usePromise } from "react";
import Link from "next/link";
import { ArrowUpRight, LineChart, Pencil, Sparkles } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Button } from "@/components/ui/Button";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";
import { groupExercisesByJoint } from "@/lib/rehab/exerciseCatalog";
import { useRecommendations } from "@/lib/rehab/useRecommendations";
import { PrescriptionEditor } from "@/components/rehab/PrescriptionEditor";
import { getPatient, type PatientDTO } from "@/lib/patients";

export default function PatientRehabPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = usePromise(params);
  return (
    <AuthGuard>
      <DashboardShell
        backHref={`/dashboard/patients/${id}`}
        backLabel="Patient"
      >
        <Content patientId={id} />
      </DashboardShell>
    </AuthGuard>
  );
}

function Content({ patientId }: { patientId: string }) {
  const groups = useMemo(() => groupExercisesByJoint(), []);
  const recs = useRecommendations(patientId);

  const [patient, setPatient] = useState<PatientDTO | null>(null);
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    let cancelled = false;
    getPatient(patientId)
      .then((p) => {
        if (!cancelled) setPatient(p);
      })
      .catch(() => {
        // Non-fatal — recommender strip degrades to generic wording.
      });
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Choose rehab game</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            Which mechanic are we playing today?
          </h1>
          <p className="mt-2 max-w-xl text-sm text-muted">
            Pick a game. Session scores save automatically against this
            patient&apos;s record.
          </p>
        </div>
        <Link href={`/dashboard/patients/${patientId}/rehab/progress`}>
          <Button variant="secondary" size="sm">
            <LineChart className="h-4 w-4" />
            Progress
          </Button>
        </Link>
      </div>

      <RecommendedStrip
        recs={recs}
        patientName={patient?.name ?? null}
        onEdit={() => setEditing(true)}
      />

      {editing && (
        <PrescriptionEditor
          patientName={patient?.name ?? null}
          currentSlugs={recs.slugs}
          source={recs.source}
          reasonsBySlug={recs.bySlug}
          saving={recs.saving}
          onSave={recs.save}
          onReset={recs.reset}
          onClose={() => setEditing(false)}
        />
      )}

      <div className="space-y-12">
        {groups.map(({ joint, meta, items }) => {
          // Recommended-first sort within each joint section. Ranking
          // uses the recommendation score so top-scored exercises float
          // to the head of the list.
          const sortedItems = [...items].sort((a, b) => {
            const scoreA = recs.bySlug.get(a.slug)?.score ?? -Infinity;
            const scoreB = recs.bySlug.get(b.slug)?.score ?? -Infinity;
            if (scoreA === scoreB) return 0;
            return scoreB - scoreA;
          });
          return (
            <section key={joint}>
              <div className="flex items-baseline justify-between">
                <h2 className="text-xl font-semibold tracking-tight md:text-2xl">
                  {meta.label}
                </h2>
                <p className="text-xs text-muted">{meta.subtitle}</p>
              </div>
              <div className="mt-4 grid gap-5 md:grid-cols-3">
                {sortedItems.map((m) => {
                  const Icon = m.icon;
                  const href = `/rehab/${m.slug}?patientId=${patientId}`;
                  const imageUrl = REHAB_EXERCISE_IMAGES[m.slug];
                  const eyebrow = `${m.code} · ${m.title}`;
                  const rec = recs.bySlug.get(m.slug);
                  const isRecommended = Boolean(rec);
                  const sharedClass = `group relative flex flex-col overflow-hidden rounded-hero border ${
                    isRecommended
                      ? "border-accent/60 shadow-glow-sm ring-1 ring-accent/20"
                      : "border-border"
                  } bg-gradient-to-br ${m.tone} p-6 transition md:p-8 hover:border-accent hover:shadow-glow-sm`;
                  return (
                    <Link key={m.slug} href={href} className={sharedClass}>
                      {isRecommended && (
                        <span className="absolute right-3 top-3 z-10 inline-flex items-center gap-1 rounded-full bg-accent/15 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-accent ring-1 ring-accent/30">
                          <Sparkles className="h-3 w-3" />
                          Recommended
                        </span>
                      )}
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
                          {m.patientBody}
                        </p>
                        {rec && rec.reasons.length > 0 && (
                          <p className="mt-3 text-xs leading-relaxed text-accent/90">
                            Why: {rec.reasons[0].reason}
                          </p>
                        )}
                      </div>
                    </Link>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

// ─── Recommended-for-this-patient strip ─────────────────────────
// Pure auto — the doctor sees a curated list derived from the latest
// assessment reports. The strip is structured so that an "Edit"
// button can slot in the future (see FUTURE DOCTOR-EDIT note in
// lib/rehab/recommendation.js:getPrescribedSet).

function RecommendedStrip({
  recs,
  patientName,
  onEdit,
}: {
  recs: ReturnType<typeof useRecommendations>;
  patientName: string | null;
  onEdit: () => void;
}) {
  const nameLabel = patientName?.trim() ? patientName : "this patient";

  if (recs.status === "loading") {
    return (
      <div className="rounded-card border border-border bg-surface p-5">
        <p className="text-xs uppercase tracking-[0.14em] text-subtle">
          Recommended for {nameLabel}
        </p>
        <p className="mt-2 text-sm text-muted">
          Reading recent assessments…
        </p>
      </div>
    );
  }

  if (recs.status === "error") {
    return (
      <div className="rounded-card border border-warning/40 bg-warning/5 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.14em] text-warning">
              Recommended for {nameLabel}
            </p>
            <p className="mt-2 text-sm text-foreground">
              Could not load assessments — showing full catalogue instead.
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={onEdit}>
            <Pencil className="h-4 w-4" />
            Edit prescription
          </Button>
        </div>
      </div>
    );
  }

  if (recs.status === "empty" || recs.recommended.length === 0) {
    return (
      <div className="rounded-card border border-dashed border-border bg-surface p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.14em] text-subtle">
              Recommended for {nameLabel}
            </p>
            <p className="mt-2 text-sm text-muted">
              {recs.assessmentsUsed > 0
                ? `Reviewed ${recs.assessmentsUsed} assessment${recs.assessmentsUsed === 1 ? "" : "s"} — no clear deficits flagged yet.`
                : "No assessment reports on file yet."}
              {" "}
              Prescribe manually to get started.
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={onEdit}>
            <Pencil className="h-4 w-4" />
            Edit prescription
          </Button>
        </div>
      </div>
    );
  }

  const top = recs.recommended.slice(0, 6);
  const sourceLabel =
    recs.source === "doctor" ? "clinician-prescribed" : "auto-derived";
  return (
    <div className="rounded-card border border-accent/30 bg-accent/5 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.14em] text-accent">
            Prescribed for {nameLabel}
          </p>
          <p className="mt-1 text-sm text-foreground">
            {recs.recommended.length} exercise{recs.recommended.length === 1 ? "" : "s"}
            {" · "}
            {sourceLabel}
            {recs.source === "auto"
              ? ` from ${recs.assessmentsUsed} recent assessment${recs.assessmentsUsed === 1 ? "" : "s"}`
              : ""}
            .
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={onEdit}>
          <Pencil className="h-4 w-4" />
          Edit
        </Button>
      </div>
      <ul className="mt-4 grid gap-2 md:grid-cols-2 lg:grid-cols-3">
        {top.map((rec) => {
          const firstReason = rec.reasons[0]?.reason ?? rec.note;
          return (
            <li
              key={rec.slug}
              className="rounded-md border border-border bg-background p-3"
            >
              <p className="text-sm font-semibold text-foreground">
                {humanizeSlug(rec.slug)}
              </p>
              {firstReason && (
                <p className="mt-0.5 text-xs text-muted">
                  {firstReason}
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function humanizeSlug(slug: string): string {
  return slug
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
