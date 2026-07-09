"use client";
// PrescriptionEditor — modal for the doctor to edit the prescribed
// exercise list. Reads the current prescribed set from the parent's
// useRecommendations() and pushes changes back via its save/reset.
//
// Layout: joint-grouped checklist (Knee / Hip / Back / Shoulder). Each
// row shows the exercise code + title + first "why recommended"
// reason (if the auto recommender scored it). Save persists the
// selected slugs; Reset removes the doctor prescription so the auto
// recommender wins again.

import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2, RotateCcw, Save, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import {
  groupExercisesByJoint,
  type RehabExerciseEntry,
} from "@/lib/rehab/exerciseCatalog";

interface PrescriptionEditorProps {
  patientName: string | null;
  /** Currently prescribed slugs from useRecommendations().slugs. */
  currentSlugs: Set<string>;
  /** "auto" or "doctor" — controls the reset button visibility. */
  source: "auto" | "doctor";
  /** Recommender reasons keyed by slug (bySlug) for the "why" hint. */
  reasonsBySlug: Map<string, { reasons: { reason: string }[]; note?: string }>;
  saving: boolean;
  onSave: (slugs: string[]) => Promise<void>;
  onReset: () => Promise<void>;
  onClose: () => void;
}

// The catalog metadata is JS with a JSDoc typedef — TS reads it as
// `any` through the shared shape. We narrow to a minimal shape here.
type ExerciseCard = Pick<
  RehabExerciseEntry,
  "slug" | "code" | "title" | "joint"
>;

interface JointGroupLite {
  joint: string;
  meta: { label: string; subtitle?: string };
  items: ExerciseCard[];
}

export function PrescriptionEditor({
  patientName,
  currentSlugs,
  source,
  reasonsBySlug,
  saving,
  onSave,
  onReset,
  onClose,
}: PrescriptionEditorProps) {
  // Local draft — user toggles first, saves on commit.
  const [draft, setDraft] = useState<Set<string>>(new Set(currentSlugs));
  const [error, setError] = useState<string | null>(null);

  // If the parent's slugs change while the modal is open (rare — e.g.
  // a background refetch), keep the draft in sync until the user
  // starts editing. Simpler: always reset on prop change.
  useEffect(() => {
    setDraft(new Set(currentSlugs));
  }, [currentSlugs]);

  const groups: JointGroupLite[] = useMemo(
    () => groupExercisesByJoint() as JointGroupLite[],
    [],
  );

  const toggle = (slug: string) => {
    setDraft((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) next.delete(slug);
      else next.add(slug);
      return next;
    });
  };

  const handleSave = async () => {
    setError(null);
    try {
      await onSave(Array.from(draft));
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  const handleReset = async () => {
    setError(null);
    try {
      await onReset();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reset failed");
    }
  };

  const draftCount = draft.size;
  const noChanges =
    draft.size === currentSlugs.size &&
    Array.from(draft).every((s) => currentSlugs.has(s));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur"
      role="dialog"
      aria-modal="true"
    >
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-hero border border-border bg-background shadow-2xl">
        {/* Header */}
        <div className="flex shrink-0 items-start justify-between gap-3 border-b border-border bg-surface px-5 py-4">
          <div>
            <p className="text-xs uppercase tracking-[0.14em] text-accent">
              Edit prescription
            </p>
            <h2 className="mt-1 text-lg font-semibold tracking-tight text-foreground">
              {patientName?.trim() || "This patient"}
            </h2>
            <p className="mt-0.5 text-xs text-muted">
              Toggle exercises the patient should focus on. Reset to let
              the auto recommender take over again.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1.5 text-muted transition hover:bg-surface hover:text-foreground"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Body */}
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <div className="space-y-6">
            {groups.map(({ joint, meta, items }) => (
              <section key={joint}>
                <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-subtle">
                  {meta.label}
                </h3>
                <ul className="mt-2 grid gap-2 sm:grid-cols-2">
                  {items.map((m) => {
                    const checked = draft.has(m.slug);
                    const rec = reasonsBySlug.get(m.slug);
                    const reason = rec?.reasons?.[0]?.reason ?? rec?.note ?? "";
                    return (
                      <li key={m.slug}>
                        <label
                          className={`flex cursor-pointer items-start gap-3 rounded-md border p-3 transition ${
                            checked
                              ? "border-accent/60 bg-accent/5"
                              : "border-border bg-surface hover:border-accent/40"
                          }`}
                        >
                          <input
                            type="checkbox"
                            className="mt-0.5 h-4 w-4 accent-orange-500"
                            checked={checked}
                            onChange={() => toggle(m.slug)}
                          />
                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-semibold text-foreground">
                              <span className="text-muted">{m.code}</span>{" "}
                              · {m.title}
                            </p>
                            {reason && (
                              <p className="mt-0.5 line-clamp-2 text-xs text-muted">
                                {reason}
                              </p>
                            )}
                          </div>
                        </label>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="shrink-0 border-t border-border bg-surface px-5 py-4">
          {error && (
            <div className="mb-3 flex items-start gap-2 rounded-md border border-error/40 bg-error/5 px-3 py-2 text-xs">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">{error}</p>
            </div>
          )}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted">
              {draftCount} exercise{draftCount === 1 ? "" : "s"} selected
              {source === "doctor" ? " · clinician-prescribed" : " · auto"}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              {source === "doctor" && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleReset}
                  disabled={saving}
                >
                  <RotateCcw className="h-4 w-4" />
                  Reset to auto
                </Button>
              )}
              <Button
                variant="secondary"
                size="sm"
                onClick={onClose}
                disabled={saving}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleSave}
                disabled={saving || noChanges}
              >
                {saving ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Saving…
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Save prescription
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
