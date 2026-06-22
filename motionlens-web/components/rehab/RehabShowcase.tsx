"use client";
// Landing-page Rehab section — mirror of ProductShowcase.tsx adapted
// for rehabilitation game categories. Clicking a card stashes the
// chosen route in sessionStorage and redirects through the patient
// list so doctor-flow patient context attaches automatically — same
// mechanism ProductShowcase uses.
//
// Distinct visual tone (teal / violet / cyan / indigo) from the
// assessment ProductShowcase (orange / amber / red) so the two
// sections read as separate pillars on the landing page.

import { useRouter } from "next/navigation";
import {
  ArrowUpRight,
  Footprints,
  Heart,
  Music,
  Spline,
  Sparkles,
  Target,
  Timer,
  Dumbbell,
  type LucideIcon,
} from "lucide-react";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { useAuth } from "@/contexts/AuthContext";

interface RehabCard {
  /** Route the doctor is launching into. Stashed in sessionStorage
   *  so the patient list can forward there with ?patientId=<id>. */
  targetRoute: string;
  eyebrow: string;
  title: string;
  body: string;
  gradient: string;
  icon: LucideIcon;
  iconTone: string;
}

// Same key the patient list page reads in
// app/dashboard/patients/page.tsx — mirrors ProductShowcase.
const INTENDED_MODULE_KEY = "motionlens.intendedModule";
const PATIENT_LIST_PATH = "/dashboard/patients";

const REHAB_CARDS: RehabCard[] = [
  {
    targetRoute: "/rehab",
    eyebrow: "Hold-in-Zone mechanic",
    title: "Hold & control.",
    body:
      "Isometric holds turned into a game. Patient drives a marker into a target band and keeps it there — knee-extension holds, glute bridges, wall sits. Hysteresis-debounced scoring so noisy pose doesn't break the streak.",
    gradient:
      "linear-gradient(135deg, rgba(20,184,166,0.20) 0%, rgba(45,212,191,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Timer,
    iconTone: "text-teal-500",
  },
  {
    targetRoute: "/rehab",
    eyebrow: "Rep-Count Gate mechanic",
    title: "Count what matters.",
    body:
      "Reps with built-in quality control — depth gate, amplitude check, jerk-flag. Five-phase state machine catches shallow or rushed reps and reports them transparently to the patient + clinician.",
    gradient:
      "linear-gradient(135deg, rgba(99,102,241,0.20) 0%, rgba(129,140,248,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
  },
  {
    targetRoute: "/rehab",
    eyebrow: "Target-Reach mechanic",
    title: "Reach further, every session.",
    body:
      "Targets spawn around the play area; the patient drives a body-anchored cursor onto them. Tracks excursion (how far the cursor reaches from centre) — a clean clinical signal for functional range gains week-over-week.",
    gradient:
      "linear-gradient(135deg, rgba(34,211,238,0.20) 0%, rgba(103,232,249,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Target,
    iconTone: "text-cyan-500",
  },
  {
    targetRoute: "/rehab",
    eyebrow: "Trace mechanic",
    title: "Smooth, accurate movement.",
    body:
      "A lead target glides along a path; the patient's cursor stays glued to it. Scores per-sample accuracy AND smoothness (jerk magnitude) — so an ataxic or guarded movement registers differently from a clean one.",
    gradient:
      "linear-gradient(135deg, rgba(168,85,247,0.20) 0%, rgba(192,132,252,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Spline,
    iconTone: "text-purple-500",
  },
  {
    targetRoute: "/rehab",
    eyebrow: "Weight-Shift mechanic",
    title: "Shift, don't step.",
    body:
      "Lateral weight-shift training with anti-cheat: the moment a step is detected, the game auto-pauses dwell accumulation. Forces honest medio-lateral COM control. Built for vestibular + post-arthroplasty balance work.",
    gradient:
      "linear-gradient(135deg, rgba(20,184,166,0.20) 0%, rgba(45,212,191,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Footprints,
    iconTone: "text-teal-500",
  },
  {
    targetRoute: "/rehab",
    eyebrow: "Match-Pose mechanic",
    title: "Match the silhouette.",
    body:
      "Per-joint angle targets with weighted aggregate match %. Linear tolerance fall-off so 'close' gets partial credit. Required hold-time gate captures the static portion of a yoga / PNF position.",
    gradient:
      "linear-gradient(135deg, rgba(244,114,182,0.20) 0%, rgba(249,168,212,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Sparkles,
    iconTone: "text-pink-500",
  },
  {
    targetRoute: "/rehab",
    eyebrow: "Metronome mechanic",
    title: "Move to the beat.",
    body:
      "Cadence training with audible + visual beat. Patient's event timestamps are snapped to the nearest scheduled beat and graded perfect / good / miss with deviation histogram. Ideal for Parkinson's cueing + gait retraining.",
    gradient:
      "linear-gradient(135deg, rgba(217,70,239,0.20) 0%, rgba(232,121,249,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Music,
    iconTone: "text-fuchsia-500",
  },
  {
    targetRoute: "/rehab",
    eyebrow: "Cardio rehab",
    title: "Submax cadence training.",
    body:
      "Stationary cardio drills with rep-counted intervals + on-tempo rhythm cues. Built for cardiac rehab phase II / III progression. Coming soon.",
    gradient:
      "linear-gradient(135deg, rgba(239,68,68,0.20) 0%, rgba(248,113,113,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Heart,
    iconTone: "text-rose-500",
  },
];

export function RehabShowcase() {
  const router = useRouter();
  const { doctor, loading } = useAuth();

  function handleCardClick(targetRoute: string) {
    if (loading) return;
    try {
      sessionStorage.setItem(INTENDED_MODULE_KEY, targetRoute);
    } catch {
      // sessionStorage can throw in private-mode contexts. Fallback
      // is the legacy "land on patient list" behaviour.
    }
    if (doctor === null) {
      router.push(`/auth/signin?next=${encodeURIComponent(PATIENT_LIST_PATH)}`);
      return;
    }
    router.push(PATIENT_LIST_PATH);
  }

  return (
    <Section id="rehab" className="bg-dots">
      <div className="max-w-2xl">
        <Badge>Rehab module</Badge>
        <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">
          Therapy turned<br />into a game.
        </h2>
        <p className="mt-5 text-lg text-muted">
          Seven reusable game mechanics built on the same BlazePose
          pipeline that powers the assessments. Plug in any joint /
          target / pose — the engine handles scoring, gating, and
          per-rep feedback. Designed for post-injury motor learning,
          balance + vestibular work, neuro cueing.
        </p>
      </div>

      <div className="mt-16 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {REHAB_CARDS.map((p) => {
          const Icon = p.icon;
          return (
            <button
              key={p.title}
              type="button"
              onClick={() => handleCardClick(p.targetRoute)}
              disabled={loading}
              className="group relative m-[2px] flex min-h-[436px] flex-col justify-between overflow-hidden rounded-hero border border-border bg-elevated p-8 text-left transition-all duration-300 hover:border-accent hover:shadow-glow-sm disabled:cursor-wait md:min-h-[476px] md:p-10"
            >
              <div
                className="pointer-events-none absolute inset-0 opacity-90 transition-opacity duration-500 group-hover:opacity-100"
                style={{ background: p.gradient }}
                aria-hidden
              />
              <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" aria-hidden />

              <div className="relative">
                <div className="flex items-center justify-between">
                  <Icon className={`h-7 w-7 ${p.iconTone}`} aria-hidden />
                  <ArrowUpRight className="h-5 w-5 text-muted transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-accent" />
                </div>

                <div className="mt-8">
                  <Badge>{p.eyebrow}</Badge>
                  <h3 className="mt-4 text-2xl font-semibold tracking-tight md:text-3xl">
                    {p.title}
                  </h3>
                  <p className="mt-4 max-w-md text-sm leading-relaxed text-muted">
                    {p.body}
                  </p>
                </div>
              </div>

              <div className="relative mt-8 inline-flex items-center gap-1 text-sm text-accent">
                Open rehab catalogue
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </div>
            </button>
          );
        })}
      </div>
    </Section>
  );
}
