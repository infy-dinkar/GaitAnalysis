"use client";
// Biomech Auto Mode — configurator.
//
// Doctor / patient picks which joints + movements + sides to run,
// then clicks Start. The queue is encoded into the URL and handed to
// /biomech/auto/run which mounts LiveAssessment per step with a
// countdown timer that auto-advances the sequence.
//
// Additive: no biomech engine / LiveAssessment / backend changes.

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Play, RotateCcw, Timer } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  JOINT_META,
  MOVEMENTS_BY_JOINT,
  buildAutoQueue,
  encodeQueue,
  pickKey,
  type AutoSelection,
  type Joint,
  type Side,
} from "@/lib/biomech/autoModeCatalog";

const BOTH_SIDES = (): Set<Side> => new Set<Side>(["left", "right"]);

export default function BiomechAutoPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <Suspense fallback={null}>
            <Inner />
          </Suspense>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function Inner() {
  const router = useRouter();
  const params = useSearchParams();
  const patientId = params.get("patientId");
  const qs = patientId ? `?patientId=${patientId}` : "";

  // selection: pickKey(joint, movementId) -> set of sides chosen for
  // THAT movement. Presence of the key = movement picked. For
  // bilateral joints (neck) the side set is unused.
  const [selection, setSelection] = useState<AutoSelection>(new Map());

  const totalSelected = selection.size;

  const queue = useMemo(() => buildAutoQueue(selection), [selection]);

  // Toggle a movement on/off. Turning ON a side-requiring movement
  // seeds it with BOTH sides; the per-movement chips below refine it.
  const toggleMovement = (joint: Joint, moveId: string, hasSide: boolean) => {
    const key = pickKey(joint, moveId);
    setSelection((prev) => {
      const next = new Map(prev);
      if (next.has(key)) next.delete(key);
      else next.set(key, hasSide ? BOTH_SIDES() : new Set<Side>());
      return next;
    });
  };

  // Toggle one side for an already-picked movement. If both sides end
  // up off, the movement stays picked but contributes no queue steps
  // (the row shows a "pick a side" hint) — the operator can flip a
  // side back on without re-checking the box.
  const toggleMovementSide = (joint: Joint, moveId: string, s: Side) => {
    const key = pickKey(joint, moveId);
    setSelection((prev) => {
      const cur = prev.get(key);
      if (!cur) return prev; // not picked — ignore
      const next = new Map(prev);
      const sides = new Set(cur);
      if (sides.has(s)) sides.delete(s);
      else sides.add(s);
      next.set(key, sides);
      return next;
    });
  };

  const selectAllForJoint = (joint: Joint, hasSide: boolean) => {
    setSelection((prev) => {
      const next = new Map(prev);
      const moves = MOVEMENTS_BY_JOINT[joint];
      const allPicked = moves.every((m) => next.has(pickKey(joint, m.id)));
      if (allPicked) {
        for (const m of moves) next.delete(pickKey(joint, m.id));
      } else {
        for (const m of moves) {
          if (!next.has(pickKey(joint, m.id))) {
            next.set(pickKey(joint, m.id), hasSide ? BOTH_SIDES() : new Set<Side>());
          }
        }
      }
      return next;
    });
  };

  const selectEverything = () => {
    const next: AutoSelection = new Map();
    for (const meta of JOINT_META) {
      for (const m of MOVEMENTS_BY_JOINT[meta.id]) {
        next.set(pickKey(meta.id, m.id), meta.hasSide ? BOTH_SIDES() : new Set<Side>());
      }
    }
    setSelection(next);
  };

  const clearAll = () => {
    setSelection(new Map());
  };

  const start = () => {
    if (queue.length === 0) return;
    const encoded = encodeQueue(queue);
    const url = new URL("/biomech/auto/run", window.location.origin);
    url.searchParams.set("q", encoded);
    if (patientId) url.searchParams.set("patientId", patientId);
    router.push(url.pathname + url.search);
  };

  return (
    <>
      <div className="flex items-start justify-between gap-4">
        <div className="max-w-2xl">
          <Badge>Biomechanics · Auto Mode</Badge>
          <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
            Screen it all in one go<span className="text-accent">.</span>
          </h1>
          <p className="mt-5 text-lg text-muted">
            Pick joints and movements — each test runs until the
            patient completes 5 reps, auto-saves, and switches to the
            next. Great for a full-body ROM screening without hopping
            between pages.
          </p>
        </div>
        <Link href={`/biomech${qs}`}>
          <Button variant="ghost" size="sm">← Biomech</Button>
        </Link>
      </div>

      <div className="mt-10 grid gap-6 lg:grid-cols-[1fr_320px]">
        {/* ── Left column — joint × movement grid ─────────────── */}
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <Button size="sm" variant="secondary" onClick={selectEverything}>
              Select everything
            </Button>
            <Button size="sm" variant="ghost" onClick={clearAll}>
              <RotateCcw className="h-4 w-4" />
              Clear
            </Button>
            <span className="ml-auto text-xs text-muted">
              {totalSelected} movement{totalSelected === 1 ? "" : "s"} picked
            </span>
          </div>

          {JOINT_META.map((meta) => {
            const moves = MOVEMENTS_BY_JOINT[meta.id];
            const allPicked = moves.every((m) =>
              selection.has(pickKey(meta.id, m.id)),
            );
            return (
              <section
                key={meta.id}
                className="rounded-card border border-border bg-surface p-5"
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold tracking-tight">
                      {meta.label}
                    </h2>
                    <p className="mt-0.5 text-xs text-muted">
                      {meta.hasSide
                        ? "Per-side test — pick Left / Right for each movement."
                        : "Bilateral test — one run per movement."}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => selectAllForJoint(meta.id, meta.hasSide)}
                    className="rounded-full border border-border px-3 py-1 text-xs font-medium text-muted transition hover:border-accent hover:text-foreground"
                  >
                    {allPicked ? "Deselect all" : "Select all"}
                  </button>
                </div>
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {moves.map((m) => {
                    const key = pickKey(meta.id, m.id);
                    const sel = selection.get(key);
                    const picked = sel !== undefined;
                    const noSide = meta.hasSide && picked && sel!.size === 0;
                    return (
                      <div
                        key={m.id}
                        className={`rounded-md border p-3 transition ${
                          picked
                            ? noSide
                              ? "border-warning/50 bg-warning/5"
                              : "border-accent bg-accent/5"
                            : "border-border bg-background hover:border-accent/40"
                        }`}
                      >
                        <label className="flex cursor-pointer items-start gap-3">
                          <input
                            type="checkbox"
                            checked={picked}
                            onChange={() => toggleMovement(meta.id, m.id, meta.hasSide)}
                            className="mt-0.5 h-4 w-4 shrink-0 rounded border-border accent-orange-500"
                          />
                          <span className="min-w-0">
                            <span className="block text-sm font-medium text-foreground">
                              {m.label}
                            </span>
                            <span className="mt-0.5 block text-xs text-muted">
                              {m.description}
                            </span>
                          </span>
                        </label>

                        {/* Per-movement Left / Right — only for
                            side-requiring joints, only once picked. */}
                        {meta.hasSide && picked && (
                          <div className="mt-3 flex items-center gap-2 pl-7">
                            {(["left", "right"] as Side[]).map((s) => {
                              const on = sel!.has(s);
                              return (
                                <button
                                  key={s}
                                  type="button"
                                  onClick={() => toggleMovementSide(meta.id, m.id, s)}
                                  className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
                                    on
                                      ? "border-accent bg-accent/15 text-accent"
                                      : "border-border bg-background text-muted hover:border-accent/40"
                                  }`}
                                >
                                  {s === "left" ? "Left" : "Right"}
                                </button>
                              );
                            })}
                            {noSide && (
                              <span className="text-[11px] text-warning">
                                pick a side
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>

        {/* ── Right column — summary + start ───────────────────── */}
        <div className="space-y-4 lg:sticky lg:top-24 lg:self-start">
          <div className="rounded-card border border-border bg-surface p-5">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-subtle">
              <Timer className="h-4 w-4" />
              Summary
            </div>
            <div className="mt-3 space-y-1.5 text-sm">
              <p>
                <span className="tabular text-lg font-semibold text-foreground">
                  {queue.length}
                </span>
                <span className="ml-1 text-muted">test{queue.length === 1 ? "" : "s"} queued</span>
              </p>
              <p className="text-muted">
                Each movement picks its own Left / Right. Every test
                runs until the patient completes 5 reps, then
                auto-saves and moves to the next — no fixed timer.
              </p>
            </div>
            <Button
              className="mt-4 w-full"
              disabled={queue.length === 0}
              onClick={start}
            >
              <Play className="h-4 w-4" />
              Start auto sequence
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
