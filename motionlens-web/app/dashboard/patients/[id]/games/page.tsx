"use client";
// Games launcher — the dashboard half of the games module.
//
// Same shape as app/dashboard/patients/[id]/rehab/page.tsx: AuthGuard +
// DashboardShell, and every card links out to an activity page under
// /games/<slug>?patientId=<id>. Only Fruit Harvest exists in step 1.

import { useEffect, useState, use as usePromise } from "react";
import Link from "next/link";
import { ArrowUpRight, CloudLightning, Gamepad2, Wind } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Button } from "@/components/ui/Button";
import { getPatient, type PatientDTO } from "@/lib/patients";

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
        {GAMES.map((g) => (
          <GameCard key={g.slug} game={g} patientId={patientId} />
        ))}
      </div>

      <p className="text-sm text-muted">
        More games arrive as they are built.
      </p>
    </div>
  );
}

interface GameCard {
  slug: string;
  title: string;
  blurb: string;
  /** What a round of this game actually measures. */
  measures: string;
  /** The presentations it is meant for. Guidance for choosing between
   *  the games, not a claim that any of them is diagnostic. */
  usefulFor: string;
  icon: typeof Gamepad2;
  tone: string;
}

/** One entry per activity page under /games. Order is the order the
 *  cards appear in. */
const GAMES: GameCard[] = [
  {
    slug: "fruit-harvest",
    title: "Fruit Harvest",
    blurb:
      "Reach out and collect fruit with one hand, including across the "
      + "midline. 60 seconds.",
    measures:
      "shoulder reach up and out, reach across the midline, accuracy and "
      + "speed of reaching.",
    usefulFor:
      "post-stroke arm weakness, shoulder rehabilitation, recovery after "
      + "shoulder surgery.",
    icon: Gamepad2,
    tone: "bg-stone-500/10 text-stone-700 dark:text-stone-400",
  },
  {
    slug: "cloudburst",
    title: "Cloudburst",
    blurb:
      "Catch falling water drops and leave the lightning alone. Speed rises "
      + "through the round. 60 seconds.",
    measures:
      "reaction time, catch accuracy, ability to withhold a movement "
      + "(lightning), dodging a sudden strike.",
    usefulFor:
      "cognitive slowing, post-concussion, frontal/attention problems, "
      + "reactive movement training.",
    icon: CloudLightning,
    tone: "bg-sky-500/10 text-sky-700 dark:text-sky-400",
  },
  {
    slug: "kite-flying",
    title: "Kite Flying",
    blurb:
      "Keep a kite inside a drifting ribbon of wind. Measures how "
      + "steadily the hand holds a moving line. 60 seconds.",
    measures:
      "movement smoothness and steadiness — how steadily the hand holds a "
      + "moving path.",
    usefulFor:
      "Parkinson's disease, ataxia, tremor, cerebellar involvement.",
    icon: Wind,
    tone: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  },
];

function GameCard({ game, patientId }: { game: GameCard; patientId: string }) {
  const Icon = game.icon;
  return (
    <Link
      href={`/games/${game.slug}?patientId=${patientId}`}
      className="group rounded-card border border-border bg-elevated p-5 shadow-glow-sm transition hover:border-accent/50"
    >
      <span
        className={`flex h-12 w-12 items-center justify-center rounded-card ${game.tone}`}
      >
        <Icon className="h-6 w-6" />
      </span>
      <h2 className="mt-4 flex items-center gap-1 text-lg font-semibold">
        {game.title}
        <ArrowUpRight className="h-4 w-4 opacity-0 transition group-hover:opacity-100" />
      </h2>
      <p className="mt-1 text-sm text-muted">{game.blurb}</p>
      {/* The two lines a clinician chooses between games on. Each is
          one paragraph with a bold lead-in rather than a label beside
          the text, so a long line wraps under its own label instead of
          forming a ragged second column. */}
      <p className="mt-3 text-sm text-muted">
        <span className="font-semibold text-foreground">Measures: </span>
        {game.measures}
      </p>
      <p className="mt-1 text-sm text-muted">
        <span className="font-semibold text-foreground">Useful for: </span>
        {game.usefulFor}
      </p>
      <Button className="mt-4" size="sm">
        Play
      </Button>
    </Link>
  );
}
