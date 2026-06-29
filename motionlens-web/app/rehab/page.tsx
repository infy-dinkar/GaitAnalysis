"use client";
// Public rehab catalogue — lists every game-mechanic the rehab
// module exposes. Mirrors the dashboard /analyze grid pattern so
// the visual language matches the assessment catalogue.
//
// Routes for fully wired exercises link straight through; mechanics
// that don't have a specific exercise plugged in yet show a graceful
// "Coming soon" badge instead of a broken link — clicking them is a
// no-op (the card stays disabled). No 404s, no dead routes.

import Link from "next/link";
import { Suspense } from "react";
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
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

interface RehabModule {
  /** Stable id for the URL slug. */
  id: string;
  eyebrow: string;
  title: string;
  body: string;
  icon: LucideIcon;
  iconTone: string;
  tone: string;
  /** When null → "Coming soon" — the card renders disabled.
   *  When set → Link target. */
  href: string | null;
}

const MODULES: RehabModule[] = [
  {
    id: "squat",
    eyebrow: "K1 · Controlled Squat",
    title: "Controlled Squat",
    body:
      "Quality-gated squat rep counter. Each rep checked against depth (110° interior knee), amplitude (50° excursion), and starting position. Shallow reps flagged transparently. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    // Catalogue page uses href directly → LEADING slash required.
    href: "/rehab/squat",
  },
  {
    id: "wall_sit",
    eyebrow: "K5 · Wall Sit",
    title: "Wall Sit",
    body:
      "Isometric wall-sit hold at 80°–100° knee flexion. The in-zone timer accumulates as long as the knee stays inside the band; drift out and it pauses. Powered by the Hold-in-Zone mechanic. Target 30 s cumulative.",
    icon: Timer,
    iconTone: "text-teal-500",
    tone: "from-teal-500/15 to-teal-500/5",
    // Catalogue page uses href directly → LEADING slash required.
    href: "/rehab/wall-sit",
  },
  {
    id: "pelvic_hold",
    eyebrow: "H1 · Pelvic-Level Hold",
    title: "Pelvic-Level Hold",
    body:
      "Trendelenburg retraining — single-leg stance with the pelvis held level (tilt within ±5° of horizontal). Symmetric ±5° band drives the in-zone timer; a hip drop pauses it. 25 s cumulative target. Hold-in-Zone mechanic.",
    icon: Timer,
    iconTone: "text-teal-500",
    tone: "from-teal-500/15 to-teal-500/5",
    href: "/rehab/pelvic-hold",
  },
  {
    id: "wall-slide",
    eyebrow: "S4 · Wall Slide",
    title: "Wall Slide",
    body:
      "Overhead-reach hold against a wall. Patient stands back-to-wall and slides the working arm up the wall, holding at the 140°–160° shoulder flexion band. 20 s cumulative target. Hold-in-Zone mechanic.",
    icon: Timer,
    iconTone: "text-teal-500",
    tone: "from-teal-500/15 to-teal-500/5",
    href: "/rehab/wall-slide",
  },
  {
    id: "rep_count",
    eyebrow: "Rep-Count Gate mechanic",
    title: "Quality reps",
    body:
      "Counts reps with built-in depth + amplitude + jerk gates. Catches shallow / rushed reps and reports them to clinician + patient.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: null,
  },
  {
    id: "shoulder_raise",
    eyebrow: "S1 · Shoulder Raise",
    title: "Shoulder Raise",
    body:
      "Active shoulder abduction to target. Raises the test arm to drive a cursor onto spawning targets — cursor height is the shared shoulder elevation angle, so the game control IS the clinical metric. Powered by the Target-Reach mechanic.",
    icon: Target,
    iconTone: "text-cyan-500",
    tone: "from-cyan-500/15 to-cyan-500/5",
    // Catalogue page uses href directly → LEADING slash required.
    href: "/rehab/shoulder-raise",
  },
  {
    id: "knee-extension",
    eyebrow: "K3 · Terminal Knee Extension",
    title: "Terminal Knee Extension",
    body:
      "Active terminal-extension drill — patient sitting with thigh supported, actively straightens the knee to drive a cursor onto spawning targets. Cursor.y is the shared knee extension angle; top targets at ≥153° extension target the post-op terminal band. Powered by the Target-Reach mechanic.",
    icon: Target,
    iconTone: "text-cyan-500",
    tone: "from-cyan-500/15 to-cyan-500/5",
    href: "/rehab/knee-extension",
  },
  {
    id: "hip-abduction",
    eyebrow: "H2 · Hip Abduction",
    title: "Hip Abduction",
    body:
      "Standing hip abduction to target — patient stands with light support, lifts the working leg sideways. Cursor.y is the shared hip abduction angle; top targets at ~38° abduction sit at the upper end of typical active ROM. Frontal view. Powered by the Target-Reach mechanic.",
    icon: Target,
    iconTone: "text-cyan-500",
    tone: "from-cyan-500/15 to-cyan-500/5",
    href: "/rehab/hip-abduction",
  },
  {
    id: "wall-clock",
    eyebrow: "S2 · Wall-Clock Reach",
    title: "Wall-Clock Reach",
    body:
      "Multidirectional shoulder reach — patient stands frontal and the hand IS the cursor. Targets spawn around the play area (12, 3, 6, 9 o’clock + diagonals), training full-hemisphere shoulder ROM + reach coordination. Powered by the Target-Reach mechanic.",
    icon: Target,
    iconTone: "text-cyan-500",
    tone: "from-cyan-500/15 to-cyan-500/5",
    href: "/rehab/wall-clock",
  },
  {
    id: "pendulum",
    eyebrow: "S3 · Pendulum / Circle Trace",
    title: "Pendulum / Circle Trace",
    body:
      "Gentle shoulder mobility — patient leans forward, lets the test arm hang, and traces a slow circle following a moving lead target. Cursor is the wrist position directly; score = accuracy + smoothness. Powered by the Trace mechanic.",
    icon: Spline,
    iconTone: "text-purple-500",
    tone: "from-purple-500/15 to-purple-500/5",
    href: "/rehab/pendulum",
  },
  {
    id: "weight_shift",
    eyebrow: "H3 · Weight-Shift Balance",
    title: "Weight-Shift Balance",
    body:
      "Limits-of-stability training. Patient stands feet fixed, shifts weight medio-laterally to drive a cursor through four target zones at ±0.4 and ±0.8. Step detection auto-pauses dwell — only honest weight shifts count. Powered by the Weight-Shift mechanic.",
    icon: Footprints,
    iconTone: "text-teal-500",
    tone: "from-teal-500/15 to-teal-500/5",
    href: "/rehab/weight-shift",
  },
  {
    id: "bridge",
    eyebrow: "H4 · Bridge",
    title: "Bridge",
    body:
      "Supine glute bridge with quality-gated rep counting. Patient lifts hips toward a straight shoulder-hip-knee line; the hip interior angle drives the same Rep-Count engine K1 Squat uses. Lateral view, depth + amplitude gates. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/bridge",
  },
  {
    id: "step-up",
    eyebrow: "K4 · Step-Up Control",
    title: "Step-Up Control",
    body:
      "Quality-gated step-up rep counter. Patient steps onto a low platform with the working leg, reaches full extension at the top, lowers under control. Stepping-leg knee interior drives the same Rep-Count engine K1 uses. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/step-up",
  },
  {
    id: "lateral-step",
    eyebrow: "H6 · Lateral Step",
    title: "Lateral Step",
    body:
      "Side-stepping drill in a maintained quarter-squat. Working-leg knee interior drives the Rep-Count engine — tighter amplitude gate (30°) reflects the shallower ROM. Frontal view, knee-tracking focus. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/lateral-step",
  },
  {
    id: "single-leg-squat",
    eyebrow: "K6 · Single-Leg Squat",
    title: "Single-Leg Squat",
    body:
      "Unipedal squat — patient stands on the working leg, descends in a controlled single-leg squat, returns. Same Rep-Count engine as K1; tightened amplitude gate (35°) and rep target (8) reflect the reduced ROM and higher demand. Lateral view. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/single-leg-squat",
  },
  {
    id: "external-rotation",
    eyebrow: "S5 · External Rotation (trend)",
    title: "External Rotation",
    body:
      "Elbow-at-side external rotation rep counter — patient tucks the upper arm and rotates the forearm outward. Uses the same Rep-Count engine as K1. Note: 2-D forearm proxy, NOT absolute ER measurement — report as within-patient trend only. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/external-rotation",
  },
  {
    id: "scapular-set",
    eyebrow: "S6 · Scapular Set (coarse)",
    title: "Scapular Set",
    body:
      "Scapular retraction / row-pattern rep counter — patient squeezes the shoulder blades together. Same Rep-Count engine as K1. Note: scapula is not landmarked; the cue is inferred from small medial shoulder displacement — a coarse coaching signal, not a precise scapular measurement. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/scapular-set",
  },
  {
    id: "mini-squat",
    eyebrow: "K2 · Mini-Squat",
    title: "Mini-Squat",
    body:
      "Shallow partial squat — lower intensity than K1. Looser depth gate (interior 140°) and smaller amplitude (25°) make it suitable for early-stage / deconditioned patients. Same Rep-Count engine, higher rep target (12). Lateral view. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/mini-squat",
  },
  {
    id: "back-extension",
    eyebrow: "B2 · Back Extension",
    title: "Back Extension",
    body:
      "Standing or prone back-extension rep counter. Patient arches the trunk gently backward through a small controlled range. Same Rep-Count engine as K1 with a new trunk-tilt proxy. Lateral view. PRD: small pain-free range only. Powered by the Rep-Count mechanic.",
    icon: Dumbbell,
    iconTone: "text-indigo-500",
    tone: "from-indigo-500/15 to-indigo-500/5",
    href: "/rehab/back-extension",
  },
  {
    id: "posture-hold",
    eyebrow: "B1 · Posture Hold",
    title: "Posture Hold",
    body:
      "Forward-head reset — patient sits or stands lateral to the camera and holds the ear stacked above the shoulder. Forward-head offset drives an in-zone band timer. Drift > 12° → timer pauses; return to alignment → resumes. 20 s cumulative target. Powered by the Hold-in-Zone mechanic.",
    icon: Timer,
    iconTone: "text-teal-500",
    tone: "from-teal-500/15 to-teal-500/5",
    href: "/rehab/posture-hold",
  },
  {
    id: "side-bend",
    eyebrow: "B3 · Side Bend",
    title: "Side Bend",
    body:
      "Lateral trunk-flexion drill — patient stands frontal and bends sideways to drive a cursor. Cursor.x is the signed lateral-flexion angle (left bends ⇒ left, right ⇒ right); cursor.y rises with magnitude, so top targets demand near-max ROM. Frontal view. Powered by the Target-Reach mechanic.",
    icon: Target,
    iconTone: "text-cyan-500",
    tone: "from-cyan-500/15 to-cyan-500/5",
    href: "/rehab/side-bend",
  },
  {
    id: "match_pose",
    eyebrow: "Match-Pose mechanic",
    title: "Hold the pose",
    body:
      "Per-joint angle targets with weighted aggregate match %. Linear tolerance fall-off gives partial credit; required hold-time gate captures the static portion.",
    icon: Sparkles,
    iconTone: "text-pink-500",
    tone: "from-pink-500/15 to-pink-500/5",
    href: null,
  },
  {
    id: "metronome",
    eyebrow: "Metronome mechanic",
    title: "Move to the beat",
    body:
      "Cadence training with audio + visual beat. Patient events are graded perfect / good / miss with a deviation histogram. Ideal for Parkinson's cueing + gait retraining.",
    icon: Music,
    iconTone: "text-fuchsia-500",
    tone: "from-fuchsia-500/15 to-fuchsia-500/5",
    href: null,
  },
  {
    id: "cardio",
    eyebrow: "Cardio rehab",
    title: "Submax cadence",
    body:
      "Stationary cardio drills with rep-counted intervals + on-tempo rhythm cues. Built for cardiac rehab phase II / III progression.",
    icon: Heart,
    iconTone: "text-rose-500",
    tone: "from-rose-500/15 to-rose-500/5",
    href: null,
  },
];

export default function RehabCataloguePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
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
                Seven reusable mechanics built on the same BlazePose
                pipeline that powers the assessments. Each game wraps a
                pure scoring engine — hold-in-zone, rep-count gate,
                target-reach, trace, weight-shift, match-pose, metronome
                — and plugs into any joint / movement the clinician
                chooses. Specific exercise wirings ship in subsequent
                releases.
              </p>
            </div>
            <Link href="/">
              <Button variant="ghost" size="sm">← Home</Button>
            </Link>
          </div>

          <div className="mt-16 grid gap-5 md:grid-cols-3">
            {MODULES.map((m) => {
              const Icon = m.icon;
              const comingSoon = m.href === null;
              // Card ids use mixed conventions (early entries use
              // underscores e.g. "wall_sit"; newer ones use hyphens
              // e.g. "wall-clock"). The image map keys on the
              // hyphen-slug form, matching the route folder names.
              const imageUrl =
                REHAB_EXERCISE_IMAGES[m.id.replace(/_/g, "-")];
              const sharedClass = `group relative flex flex-col overflow-hidden rounded-hero border border-border bg-gradient-to-br ${m.tone} p-6 transition md:p-8`;
              const interactive = comingSoon
                ? "cursor-default opacity-80"
                : "hover:border-accent hover:shadow-glow-sm";

              const content = (
                <>
                  {/* Reference thumbnail — mirrors biomech's
                      MovementGrid tile (h-28, white bg, rounded,
                      object-contain, lazy, decorative). Conditional
                      so placeholder cards stay text-only. */}
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
                    {comingSoon ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-zinc-800/80 px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.12em] text-zinc-300 ring-1 ring-zinc-600">
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

              if (comingSoon) {
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
                  href={m.href!}
                  className={`${sharedClass} ${interactive}`}
                >
                  {content}
                </Link>
              );
            })}
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">For clinicians</p>
            <p className="mt-2">
              Rehab games are designed to share the patient context the
              assessment modules already use — open this catalogue from a
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
