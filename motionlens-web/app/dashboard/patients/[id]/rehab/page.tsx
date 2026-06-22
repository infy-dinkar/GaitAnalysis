"use client";
// Dashboard rehab launcher — parallel to /analyze/page.tsx. Shows
// the rehab mechanic catalogue with the patient context already
// attached so any session played from here will save against the
// patient's record once an exercise is wired into a mechanic.
//
// Identical visual structure to the analyze page (DashboardShell +
// REHAB_MODULES grid). Cards link to /rehab/<game-id>?patientId=…
// where available; otherwise they show a "Coming soon" pill.

import { use as usePromise } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  Dumbbell,
  Footprints,
  Heart,
  Music,
  Spline,
  Sparkles,
  Target,
  Timer,
  type LucideIcon,
} from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";

interface RehabModule {
  id: string;
  /** When null = coming soon. Otherwise the route slug, e.g.
   *  "rehab/hold-in-zone" — patientId is appended at click time. */
  href: string | null;
  eyebrow: string;
  title: string;
  body: string;
  icon: LucideIcon;
  tone: string;
  iconTone: string;
}

const MODULES: RehabModule[] = [
  {
    id: "squat",
    // Dashboard page builds the URL as `/${m.href}?patientId=…` —
    // NO leading slash here, so the result is /rehab/squat?patientId=…
    href: "rehab/squat",
    eyebrow: "K1 · Controlled Squat",
    title: "Controlled Squat",
    body:
      "Quality-gated squat rep counter — depth 110°, amplitude 50°, knee-interior signal. Powered by the Rep-Count mechanic. Side picker before recording.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    id: "hold_in_zone",
    href: null,
    eyebrow: "Hold-in-Zone",
    title: "Hold & control",
    body:
      "Isometric holds turned into a game — patient drives a marker into a target band and keeps it there.",
    icon: Timer,
    tone: "from-teal-500/15 to-teal-500/5",
    iconTone: "text-teal-600",
  },
  {
    id: "rep_count",
    href: null,
    eyebrow: "Rep-Count Gate",
    title: "Quality reps",
    body:
      "Reps with built-in depth + amplitude + jerk gates. Shallow / rushed reps are flagged transparently.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    id: "target_reach",
    href: null,
    eyebrow: "Target-Reach",
    title: "Reach further",
    body:
      "Targets spawn around the play area; the patient drives a body-anchored cursor onto them. Tracks max excursion.",
    icon: Target,
    tone: "from-cyan-500/15 to-cyan-500/5",
    iconTone: "text-cyan-600",
  },
  {
    id: "trace",
    href: null,
    eyebrow: "Trace",
    title: "Smooth trace",
    body:
      "Lead target glides along a path; cursor stays glued to it. Per-sample accuracy AND smoothness scoring.",
    icon: Spline,
    tone: "from-purple-500/15 to-purple-500/5",
    iconTone: "text-purple-600",
  },
  {
    id: "weight_shift",
    href: null,
    eyebrow: "Weight-Shift",
    title: "Honest weight shift",
    body:
      "Lateral weight-shift with anti-cheat — step detected → dwell auto-pauses. Real medio-lateral COM control.",
    icon: Footprints,
    tone: "from-teal-500/15 to-teal-500/5",
    iconTone: "text-teal-600",
  },
  {
    id: "match_pose",
    href: null,
    eyebrow: "Match-Pose",
    title: "Hold the pose",
    body:
      "Per-joint angle targets with weighted aggregate. Linear tolerance fall-off + required hold-time gate.",
    icon: Sparkles,
    tone: "from-pink-500/15 to-pink-500/5",
    iconTone: "text-pink-600",
  },
  {
    id: "metronome",
    href: null,
    eyebrow: "Metronome",
    title: "Move to the beat",
    body:
      "Cadence training with audio + visual beat. Perfect / good / miss grading. Ideal for Parkinson's cueing.",
    icon: Music,
    tone: "from-fuchsia-500/15 to-fuchsia-500/5",
    iconTone: "text-fuchsia-600",
  },
  {
    id: "cardio",
    href: null,
    eyebrow: "Cardio rehab",
    title: "Submax cadence",
    body:
      "Stationary cardio drills with rep-counted intervals + on-tempo rhythm cues. Phase II / III progression.",
    icon: Heart,
    tone: "from-rose-500/15 to-rose-500/5",
    iconTone: "text-rose-600",
  },
];

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
  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Choose rehab game</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
          Which mechanic are we playing today?
        </h1>
        <p className="mt-2 max-w-xl text-sm text-muted">
          Pick a game. Once an exercise is plugged into the mechanic,
          the session score saves automatically against this
          patient&apos;s record.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-3">
        {MODULES.map((m) => {
          const Icon = m.icon;
          const comingSoon = m.href === null;
          const href = comingSoon
            ? null
            : `/${m.href}?patientId=${patientId}`;
          const sharedClass = `group relative flex flex-col overflow-hidden rounded-hero border border-border bg-gradient-to-br ${m.tone} p-6 transition md:p-8`;
          const interactive = comingSoon
            ? "cursor-default opacity-80"
            : "hover:border-accent hover:shadow-glow-sm";

          const content = (
            <>
              <div className="flex items-center justify-between">
                <Icon className={`h-7 w-7 ${m.iconTone}`} />
                {comingSoon ? (
                  <span className="inline-flex items-center gap-1 rounded-full bg-zinc-800/80 px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.12em] text-zinc-200 ring-1 ring-zinc-600">
                    Coming soon
                  </span>
                ) : (
                  <ArrowUpRight className="h-5 w-5 text-muted transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-accent" />
                )}
              </div>

              <div className="mt-8">
                <p className="eyebrow">{m.eyebrow}</p>
                <h2 className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">
                  {m.title}
                </h2>
                <p className="mt-3 text-sm leading-relaxed text-muted">
                  {m.body}
                </p>
              </div>
            </>
          );

          if (comingSoon || !href) {
            return (
              <div
                key={m.id}
                aria-disabled
                className={`${sharedClass} ${interactive}`}
              >
                {content}
              </div>
            );
          }
          return (
            <Link
              key={m.id}
              href={href}
              className={`${sharedClass} ${interactive}`}
            >
              {content}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
