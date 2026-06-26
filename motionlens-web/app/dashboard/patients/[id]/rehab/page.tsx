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
    id: "wall_sit",
    // Dashboard page builds the URL as `/${m.href}?patientId=…` —
    // NO leading slash here, so the result is /rehab/wall-sit?patientId=…
    href: "rehab/wall-sit",
    eyebrow: "K5 · Wall Sit",
    title: "Wall Sit",
    body:
      "Isometric hold at 80°–100° knee flexion. Timer accumulates inside the band, pauses outside. 30 s target. Powered by the Hold-in-Zone mechanic.",
    icon: Timer,
    tone: "from-teal-500/15 to-teal-500/5",
    iconTone: "text-teal-600",
  },
  {
    id: "pelvic_hold",
    href: "rehab/pelvic-hold",
    eyebrow: "H1 · Pelvic-Level Hold",
    title: "Pelvic-Level Hold",
    body:
      "Trendelenburg retraining — single-leg stance with pelvis level (±5° band). Hip drop pauses the timer. 25 s cumulative target. Hold-in-Zone mechanic.",
    icon: Timer,
    tone: "from-teal-500/15 to-teal-500/5",
    iconTone: "text-teal-600",
  },
  {
    id: "wall-slide",
    href: "rehab/wall-slide",
    eyebrow: "S4 · Wall Slide",
    title: "Wall Slide",
    body:
      "Overhead-reach hold — back-to-wall, slide working arm up to the 140°–160° shoulder flexion band. 20 s cumulative target. Hold-in-Zone mechanic.",
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
    id: "shoulder_raise",
    // Dashboard page builds the URL as `/${m.href}?patientId=…` —
    // NO leading slash here, so the result is /rehab/shoulder-raise?patientId=…
    href: "rehab/shoulder-raise",
    eyebrow: "S1 · Shoulder Raise",
    title: "Shoulder Raise",
    body:
      "Active shoulder abduction to target. Cursor.y is the shared shoulder-elevation angle — patient raises arm to hit higher targets. Target-Reach mechanic.",
    icon: Target,
    tone: "from-cyan-500/15 to-cyan-500/5",
    iconTone: "text-cyan-600",
  },
  {
    id: "knee-extension",
    href: "rehab/knee-extension",
    eyebrow: "K3 · Terminal Knee Extension",
    title: "Terminal Knee Extension",
    body:
      "Active terminal-extension drill — cursor.y is the shared knee extension angle. Top targets target the last 0–27° (post-op terminal band). Target-Reach mechanic.",
    icon: Target,
    tone: "from-cyan-500/15 to-cyan-500/5",
    iconTone: "text-cyan-600",
  },
  {
    id: "hip-abduction",
    href: "rehab/hip-abduction",
    eyebrow: "H2 · Hip Abduction",
    title: "Hip Abduction",
    body:
      "Standing hip abduction to target — cursor.y is the shared hip abduction angle. Top targets at ~38° (upper end of active ROM). Frontal view. Target-Reach mechanic.",
    icon: Target,
    tone: "from-cyan-500/15 to-cyan-500/5",
    iconTone: "text-cyan-600",
  },
  {
    id: "wall-clock",
    href: "rehab/wall-clock",
    eyebrow: "S2 · Wall-Clock Reach",
    title: "Wall-Clock Reach",
    body:
      "Multidirectional shoulder reach — hand is the cursor. Targets spawn at clock-like positions, forcing reach in every direction. Frontal view. Target-Reach mechanic.",
    icon: Target,
    tone: "from-cyan-500/15 to-cyan-500/5",
    iconTone: "text-cyan-600",
  },
  {
    id: "pendulum",
    href: "rehab/pendulum",
    eyebrow: "S3 · Pendulum / Circle Trace",
    title: "Pendulum / Circle Trace",
    body:
      "Gentle shoulder mobility — wrist follows a slow circle. Per-sample accuracy + smoothness scoring. Trace mechanic.",
    icon: Spline,
    tone: "from-purple-500/15 to-purple-500/5",
    iconTone: "text-purple-600",
  },
  {
    id: "weight_shift",
    href: "rehab/weight-shift",
    eyebrow: "H3 · Weight-Shift Balance",
    title: "Weight-Shift Balance",
    body:
      "Limits-of-stability training — feet fixed, shift the hip-mid to capture four lateral zones at ±0.4 and ±0.8. Step detection auto-pauses dwell. Weight-Shift mechanic.",
    icon: Footprints,
    tone: "from-teal-500/15 to-teal-500/5",
    iconTone: "text-teal-600",
  },
  {
    id: "bridge",
    href: "rehab/bridge",
    eyebrow: "H4 · Bridge",
    title: "Bridge",
    body:
      "Supine glute bridge — hip interior angle drives the Rep-Count engine. Lateral view, depth + amplitude gates. Rep-Count mechanic.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    id: "step-up",
    href: "rehab/step-up",
    eyebrow: "K4 · Step-Up Control",
    title: "Step-Up Control",
    body:
      "Stepping-leg knee interior drives the Rep-Count engine. Patient steps up to full extension, lowers under control. Depth + amplitude gates. Rep-Count mechanic.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    id: "lateral-step",
    href: "rehab/lateral-step",
    eyebrow: "H6 · Lateral Step",
    title: "Lateral Step",
    body:
      "Side-stepping drill in a quarter-squat. Working knee interior drives the Rep-Count engine. Tighter amplitude gate (30°) for the shallower ROM. Rep-Count mechanic.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    id: "single-leg-squat",
    href: "rehab/single-leg-squat",
    eyebrow: "K6 · Single-Leg Squat",
    title: "Single-Leg Squat",
    body:
      "Unipedal squat — same Rep-Count engine as K1, tighter amplitude (35°) and 8-rep target reflect the reduced ROM and higher balance load. Lateral view. Rep-Count mechanic.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    id: "external-rotation",
    href: "rehab/external-rotation",
    eyebrow: "S5 · External Rotation (trend)",
    title: "External Rotation",
    body:
      "Elbow-at-side ER rep counter using a 2-D forearm-position proxy. Trend only — not absolute ER. Frontal view. Rep-Count mechanic.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    id: "scapular-set",
    href: "rehab/scapular-set",
    eyebrow: "S6 · Scapular Set (coarse)",
    title: "Scapular Set",
    body:
      "Scapular retraction rep counter — proxy from shoulder-width narrowing. Coarse coaching cue only — not a precise scapular measurement. Frontal view, auto-calibrated baseline. Rep-Count mechanic.",
    icon: Dumbbell,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
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
