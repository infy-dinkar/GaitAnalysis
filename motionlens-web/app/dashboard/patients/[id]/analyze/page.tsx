"use client";
// /dashboard/patients/[id]/analyze — pick assessment module

import Link from "next/link";
import { use as usePromise } from "react";
import {
  ArmchairIcon,
  ArrowUpRight,
  Award,
  Clock,
  Footprints,
  Activity,
  Layers,
  Move3d,
  MoveUp,
  MoveDiagonal,
  Hourglass,
  ChevronsRight,
  ArrowUpDown,
  Ruler,
  Hand,
  MoveRight,
  PersonStanding,
  Scale,
  StretchHorizontal,
  TimerIcon,
} from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";

const MODULES = [
  {
    href: "gait/upload",
    eyebrow: "Walking video",
    title: "Gait analysis",
    body: "Upload a side-view walking clip. We extract cadence, stride symmetry, joint kinematics, and cycle-locked curves.",
    icon: Footprints,
    tone: "from-cyan-500/15 to-cyan-500/5",
    iconTone: "text-cyan-600",
  },
  {
    href: "biomech",
    eyebrow: "Range of motion",
    title: "Biomechanics",
    body: "Assess shoulder, neck, knee, hip, or ankle ROM. Live camera or video upload — choose any movement.",
    icon: Activity,
    tone: "from-amber-500/15 to-amber-500/5",
    iconTone: "text-amber-600",
  },
  {
    href: "posture",
    eyebrow: "Static posture",
    title: "Posture screening",
    body: "Upload front and side photos. Get tilts, plumb-line shifts, and an annotated report.",
    icon: PersonStanding,
    tone: "from-emerald-500/15 to-emerald-500/5",
    iconTone: "text-emerald-600",
  },
  {
    href: "orthopedic/trendelenburg",
    eyebrow: "Orthopedic test",
    title: "Trendelenburg",
    body: "Live single-leg-stance test. 30-second hold per side, automatic pelvic-tilt + trunk-lean tracking, side-by-side report.",
    icon: StretchHorizontal,
    tone: "from-violet-500/15 to-violet-500/5",
    iconTone: "text-violet-600",
  },
  {
    href: "orthopedic/single-leg-squat",
    eyebrow: "Orthopedic test",
    title: "Single-leg squat",
    body: "Live frontal-plane knee-control screen. 5 squats per side, automatic KFPPA + pelvic drop + trunk lean, composite injury-risk score.",
    icon: Move3d,
    tone: "from-fuchsia-500/15 to-fuchsia-500/5",
    iconTone: "text-fuchsia-600",
  },
  {
    href: "orthopedic/sit-to-stand",
    eyebrow: "Geriatric screen",
    title: "5x Sit-to-Stand",
    body: "Lower-extremity strength + fall-risk indicator. 5 timed sit-to-stand cycles, auto rep detection, fatigue + arm-uncrossing flags.",
    icon: ArmchairIcon,
    tone: "from-rose-500/15 to-rose-500/5",
    iconTone: "text-rose-600",
  },
  {
    href: "orthopedic/30-second-chair-stand",
    eyebrow: "Geriatric screen",
    title: "30-Second Chair Stand",
    body: "CDC STEADI fall-risk screen. Max sit-to-stand reps in 30 s, age- and sex-matched norm comparison, fatigue trend.",
    icon: TimerIcon,
    tone: "from-sky-500/15 to-sky-500/5",
    iconTone: "text-sky-600",
  },
  {
    href: "orthopedic/single-leg-stance",
    eyebrow: "Balance test",
    title: "Single-Leg Stance",
    body: "Standalone balance assessment. Eyes-open + optional eyes-closed trials per side, hold time, sway path + 95% ellipse, age-matched thresholds.",
    icon: Scale,
    tone: "from-teal-500/15 to-teal-500/5",
    iconTone: "text-teal-600",
  },
  {
    href: "orthopedic/4-stage-balance",
    eyebrow: "Balance test",
    title: "4-Stage Balance Test",
    body: "CDC fall-risk progression. 4 progressively harder static stances held for 10 s each, sway path + 95% ellipse per stage, classification on stage reached.",
    icon: Layers,
    tone: "from-indigo-500/15 to-indigo-500/5",
    iconTone: "text-indigo-600",
  },
  {
    href: "orthopedic/tug",
    eyebrow: "Geriatric / fall-risk",
    title: "Timed Up and Go (TUG)",
    body: "Side-view video upload. Backend MediaPipe analysis decomposes the 3 m walk into 5 sub-phases — sit-to-stand, walk-out, turn, walk-back, stand-to-sit — with auto-flags for balance impairment.",
    icon: Clock,
    tone: "from-yellow-500/15 to-yellow-500/5",
    iconTone: "text-yellow-600",
  },
  {
    href: "orthopedic/sppb",
    eyebrow: "Geriatric flagship · composite",
    title: "SPPB",
    body: "Three-test composite: balance stages + 4 m gait speed + 5x sit-to-stand. One 0-12 score predicts falls, hospitalisation, and disability.",
    icon: Award,
    tone: "from-purple-500/15 to-purple-500/5",
    iconTone: "text-purple-600",
  },
  {
    href: "orthopedic/slr",
    eyebrow: "Orthopedic test",
    title: "Straight Leg Raise",
    body: "Passive hip-flexion screen. Supine patient raises one straight leg per side; we capture the maximum angle reached while the knee stayed straight.",
    icon: MoveUp,
    tone: "from-lime-500/15 to-lime-500/5",
    iconTone: "text-lime-600",
  },
  {
    href: "orthopedic/ake",
    eyebrow: "Orthopedic test",
    title: "Active Knee Extension",
    body: "Hamstring-length screen via the 90/90 test. Supine patient holds the thigh vertical and slowly extends the knee; we capture max knee angle + extension deficit per side.",
    icon: MoveDiagonal,
    tone: "from-orange-500/15 to-orange-500/5",
    iconTone: "text-orange-600",
  },
  {
    href: "orthopedic/modified-thomas",
    eyebrow: "Orthopedic test",
    title: "Modified Thomas Test",
    body: "Hip-flexor and rectus-femoris length screen. Supine patient holds one knee to chest and lets the other leg hang; we auto-capture settled hip + knee angles per side.",
    icon: Hourglass,
    tone: "from-pink-500/15 to-pink-500/5",
    iconTone: "text-pink-600",
  },
  {
    href: "orthopedic/forward-lunge",
    eyebrow: "Functional movement",
    title: "Forward Lunge",
    body: "Lateral-view 5-rep lunge screen. Auto-segments each rep and reports front-knee depth, knee-over-toe, trunk forward lean, and depth consistency per side.",
    icon: ChevronsRight,
    tone: "from-blue-500/15 to-blue-500/5",
    iconTone: "text-blue-600",
  },
  {
    href: "orthopedic/sts-quality",
    eyebrow: "Functional movement",
    title: "Sit-to-Stand Quality",
    body: "Lateral 3-rep quality assessment (post-TKR/THR, geriatric rehab). Phase timing, trunk lean + knee at seat-off, smoothness score, hand-use compensation flag. Separate from the 5xSTS speed test.",
    icon: ArrowUpDown,
    tone: "from-green-500/15 to-green-500/5",
    iconTone: "text-green-600",
  },
  {
    href: "orthopedic/tandem-walk",
    eyebrow: "Balance / vestibular",
    title: "Tandem Walk",
    body: "Heel-to-toe gait screen on a taped line, 10 steps toward the camera. Auto-counts missteps and arm-grabs, measures lateral deviation, step-time variability, and trunk sway.",
    icon: Ruler,
    tone: "from-slate-500/15 to-slate-500/5",
    iconTone: "text-slate-600",
  },
  {
    href: "orthopedic/pronator-drift",
    eyebrow: "Neurological screen",
    title: "Pronator Drift",
    body: "Bedside upper-motor-neuron screen. 20-second eyes-closed hold with both arms extended; we track each wrist's vertical drift and flag asymmetric drops. 2D camera measures vertical drop only — rotation NOT assessed.",
    icon: Hand,
    tone: "from-zinc-500/15 to-zinc-500/5",
    iconTone: "text-zinc-600",
  },
  {
    href: "orthopedic/functional-reach",
    eyebrow: "Balance / fall-risk screen",
    title: "Functional Reach",
    body: "Lateral-view bedside fall-risk screen. Patient's height calibrates scale (centimetres), then reaches forward three times; we auto-detect the peak, void trials with heel-rise or stepping, and classify fall risk on the best valid reach.",
    icon: MoveRight,
    tone: "from-red-500/15 to-red-500/5",
    iconTone: "text-red-600",
  },
  {
    href: "orthopedic/single-leg-hop",
    eyebrow: "Functional hop test",
    title: "Single-Leg Hop",
    body: "Forward hop for distance. Patient stands on the test leg, hops forward, and lands on the same leg — three trials per leg. Scale is calibrated from the patient's standing height; the Limb Symmetry Index (≥ 90 % = cleared, standard ACL convention) flags side-to-side asymmetry.",
    icon: MoveRight,
    tone: "from-orange-500/15 to-orange-500/5",
    iconTone: "text-orange-600",
  },
  {
    href: "orthopedic/counter-movement-jump",
    eyebrow: "Vertical jump · power",
    title: "Counter-Movement Jump",
    body: "Vertical jump for power. Patient dips into a brief squat then jumps straight up as high as possible — up to 3 trials per recording. Primary outcome is jump height (cm) from the hip-midpoint apex; secondary is flight time plus a gravity-based physics cross-check.",
    icon: MoveRight,
    tone: "from-violet-500/15 to-violet-500/5",
    iconTone: "text-violet-600",
  },
  {
    href: "orthopedic/tuck-jump",
    eyebrow: "Injury-risk screen · frontal",
    title: "Tuck Jump Assessment",
    body: "Myer's TJA — continuous ~10 s tuck-jump session (frontal). Scored against Myer's 10-item checklist and classified good / moderate / poor. 8 items measurable single-camera; items 5 (foot yaw) and 7 (contact noise) honestly marked not assessed.",
    icon: MoveRight,
    tone: "from-fuchsia-500/15 to-fuchsia-500/5",
    iconTone: "text-fuchsia-600",
  },
  {
    href: "orthopedic/overhead-squat",
    eyebrow: "Movement screen · frontal",
    title: "Overhead Squat Assessment",
    body: "NASM/FMS-style overhead-squat screen — 3-5 slow reps with arms overhead (frontal). Scored against a 7-item checklist: knee valgus, pelvic drop, foot placement, arm drop, depth proxy. Items 6 (torso lean) and 7 (heel rise) need sagittal/feet close-up so honestly not assessed.",
    icon: MoveRight,
    tone: "from-sky-500/15 to-sky-500/5",
    iconTone: "text-sky-600",
  },
  {
    href: "orthopedic/squat-lateral",
    eyebrow: "Physio squat · lateral",
    title: "Squat (Lateral)",
    body: "Sagittal-plane squat — side-on single camera. Six metrics at the deepest rep: peak knee flexion, peak hip flexion, ankle dorsiflexion (shank tilt), trunk lean, hip:knee ratio, heel rise. Near-side leg only. Frontal-plane valgus honestly not assessed — use Overhead Squat or Single-Leg Squat for that.",
    icon: MoveRight,
    tone: "from-teal-500/15 to-teal-500/5",
    iconTone: "text-teal-600",
  },
];

export default function AnalyzePage({
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
        <p className="eyebrow">Choose assessment</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
          What do you want to measure?
        </h1>
        <p className="mt-2 max-w-xl text-sm text-muted">
          Pick a module. The result will be saved automatically against this patient&apos;s record.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-3">
        {MODULES.map((m) => {
          const Icon = m.icon;
          // patientId is forwarded as a query param so the existing analysis
          // pages can detect "doctor flow" and auto-save the report when done.
          const href = `/${m.href}?patientId=${patientId}`;
          return (
            <Link
              key={m.href}
              href={href}
              className={`group relative flex flex-col overflow-hidden rounded-hero border border-border bg-gradient-to-br ${m.tone} p-6 transition hover:border-accent hover:shadow-glow-sm md:p-8`}
            >
              <div className="flex items-center justify-between">
                <Icon className={`h-7 w-7 ${m.iconTone}`} />
                <ArrowUpRight className="h-5 w-5 text-muted transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-accent" />
              </div>

              <div className="mt-8">
                <p className="eyebrow">{m.eyebrow}</p>
                <h2 className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">
                  {m.title}
                </h2>
                <p className="mt-3 text-sm leading-relaxed text-muted">{m.body}</p>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
