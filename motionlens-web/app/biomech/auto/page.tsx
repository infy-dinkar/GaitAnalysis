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
  type Joint,
  type Side,
} from "@/lib/biomech/autoModeCatalog";

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

  // picks: joint -> set of movement ids selected under it
  const [picks, setPicks] = useState<Map<Joint, Set<string>>>(new Map());
  const [sides, setSides] = useState<Set<Side>>(new Set(["left", "right"]));

  const totalSelected = useMemo(() => {
    let n = 0;
    for (const s of picks.values()) n += s.size;
    return n;
  }, [picks]);

  const queue = useMemo(
    () => buildAutoQueue(picks, sides),
    [picks, sides],
  );

  const toggleMovement = (joint: Joint, moveId: string) => {
    setPicks((prev) => {
      const next = new Map(prev);
      const cur = new Set(next.get(joint) ?? []);
      if (cur.has(moveId)) cur.delete(moveId);
      else cur.add(moveId);
      if (cur.size === 0) next.delete(joint);
      else next.set(joint, cur);
      return next;
    });
  };

  const selectAllForJoint = (joint: Joint) => {
    setPicks((prev) => {
      const next = new Map(prev);
      const all = MOVEMENTS_BY_JOINT[joint].map((m) => m.id);
      const cur = next.get(joint);
      // Toggle: if all already selected, clear; else select all.
      if (cur && cur.size === all.length) {
        next.delete(joint);
      } else {
        next.set(joint, new Set(all));
      }
      return next;
    });
  };

  const selectEverything = () => {
    const next = new Map<Joint, Set<string>>();
    for (const meta of JOINT_META) {
      next.set(
        meta.id,
        new Set(MOVEMENTS_BY_JOINT[meta.id].map((m) => m.id)),
      );
    }
    setPicks(next);
    setSides(new Set(["left", "right"]));
  };

  const clearAll = () => {
    setPicks(new Map());
  };

  const toggleSide = (s: Side) => {
    setSides((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
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
            const chosen = picks.get(meta.id) ?? new Set<string>();
            const allCount = MOVEMENTS_BY_JOINT[meta.id].length;
            const allPicked = chosen.size === allCount;
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
                        ? "Per-side test — will run once per selected side."
                        : "Bilateral test — one run per movement."}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => selectAllForJoint(meta.id)}
                    className="rounded-full border border-border px-3 py-1 text-xs font-medium text-muted transition hover:border-accent hover:text-foreground"
                  >
                    {allPicked ? "Deselect all" : "Select all"}
                  </button>
                </div>
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {MOVEMENTS_BY_JOINT[meta.id].map((m) => {
                    const picked = chosen.has(m.id);
                    return (
                      <label
                        key={m.id}
                        className={`flex cursor-pointer items-start gap-3 rounded-md border p-3 transition ${
                          picked
                            ? "border-accent bg-accent/5"
                            : "border-border bg-background hover:border-accent/40"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={picked}
                          onChange={() => toggleMovement(meta.id, m.id)}
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
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>

        {/* ── Right column — sides + duration + start ──────────── */}
        <div className="space-y-4 lg:sticky lg:top-24 lg:self-start">
          <div className="rounded-card border border-border bg-surface p-5">
            <h3 className="text-sm font-semibold tracking-tight">
              Sides to test
            </h3>
            <p className="mt-1 text-xs text-muted">
              Ignored for bilateral tests (neck).
            </p>
            <div className="mt-3 flex gap-2">
              {(["left", "right"] as Side[]).map((s) => {
                const on = sides.has(s);
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() => toggleSide(s)}
                    className={`flex-1 rounded-md border px-3 py-2 text-sm font-medium transition ${
                      on
                        ? "border-accent bg-accent/10 text-accent"
                        : "border-border bg-background text-muted hover:border-accent/40"
                    }`}
                  >
                    {s === "left" ? "Left" : "Right"}
                  </button>
                );
              })}
            </div>
          </div>

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
                Each test runs until the patient completes 5 reps,
                then auto-saves and moves to the next — no fixed
                timer.
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
