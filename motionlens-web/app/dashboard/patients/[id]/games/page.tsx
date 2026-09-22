"use client";
// Games launcher — the dashboard half of the games module.
//
// Same shape as app/dashboard/patients/[id]/rehab/page.tsx: AuthGuard +
// DashboardShell, and every card links out to an activity page under
// /games/<slug>?patientId=<id>. Only Fruit Harvest exists in step 1.

import { useEffect, useState, use as usePromise } from "react";
import Link from "next/link";
import { ArrowUpRight, Gamepad2 } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Button } from "@/components/ui/Button";
import { getPatient, type PatientDTO } from "@/lib/patients";

const INTRO_MS = 1500;

export default function PatientGamesPage({
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
  const [patient, setPatient] = useState<PatientDTO | null>(null);
  const [introDone, setIntroDone] = useState(false);

  useEffect(() => {
    const t = window.setTimeout(() => setIntroDone(true), INTRO_MS);
    return () => window.clearTimeout(t);
  }, []);

  useEffect(() => {
    let cancelled = false;
    getPatient(patientId)
      .then((p) => {
        if (!cancelled) setPatient(p);
      })
      .catch(() => {
        // Non-fatal: the list works without the patient's name.
      });
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  if (!introDone) return <Intro />;

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Games</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight md:text-3xl">
          Camera games
        </h1>
        <p className="mt-2 max-w-2xl text-muted">
          Reach-and-collect games played standing about 2 m from the camera.
          {patient ? ` Session for ${patient.name}.` : ""}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Link
          href={`/games/fruit-harvest?patientId=${patientId}`}
          className="group rounded-card border border-border bg-elevated p-5 shadow-glow-sm transition hover:border-accent/50"
        >
          <span className="flex h-12 w-12 items-center justify-center rounded-card bg-stone-500/10 text-stone-700 dark:text-stone-400">
            <Gamepad2 className="h-6 w-6" />
          </span>
          <h2 className="mt-4 flex items-center gap-1 text-lg font-semibold">
            Fruit Harvest
            <ArrowUpRight className="h-4 w-4 opacity-0 transition group-hover:opacity-100" />
          </h2>
          <p className="mt-1 text-sm text-muted">
            Reach out and collect fruit with one hand, including across the
            midline. 60 seconds.
          </p>
          <Button className="mt-4" size="sm">
            Play
          </Button>
        </Link>
      </div>

      <p className="text-sm text-muted">
        More games arrive as they are built.
      </p>
    </div>
  );
}

/** ~1.5 s intro — three fruit settling into place, then the list. */
function Intro() {
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center">
      <div className="flex gap-4 text-6xl">
        {["🍎", "🍊", "🍐"].map((f, i) => (
          <span
            key={f}
            className="animate-bounce"
            style={{ animationDelay: `${i * 140}ms`, animationDuration: "900ms" }}
            role="img"
            aria-hidden
          >
            {f}
          </span>
        ))}
      </div>
      <p className="mt-6 text-lg font-medium tracking-tight">Camera games</p>
    </div>
  );
}
