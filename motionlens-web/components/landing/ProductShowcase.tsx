"use client";
// Fifteen landing-page module cards.
//
// Why card clicks no longer link directly to bare module routes:
//   A bare URL like /orthopedic/trendelenburg carries no ?patientId=,
//   so usePatientContext returns isDoctorFlow=false and the entire
//   save / banner / history surface stays hidden — even for a
//   logged-in doctor. The audit showed this was the *only* difference
//   between the dashboard path and the landing path.
//
// The fix: route the visitor through the existing dashboard flow so
// the patientId is attached to the URL the same way it always is.
// Each card stashes its *intended module route* in sessionStorage
// before redirecting; the patient list page reads that key on the
// next patient-card click and forwards straight to
//   <module>?patientId=<id>
// — skipping the patient-profile page so the doctor lands on the
// module they actually picked. If sessionStorage is empty (normal
// dashboard usage), the patient list keeps its existing
// behaviour (open the profile).
//
//   - Not signed in  → /auth/signin?next=/dashboard/patients
//                      (SignInForm already honours `next` and routes
//                       there after login; intendedModule survives
//                       the round-trip since it's per-tab.)
//   - Signed in      → /dashboard/patients  → patient click
//                      → /<module>?patientId=<id>
//
// No other file is touched — usePatientContext, SaveToPatientButton,
// the capture components, the patient profile, and every backend
// endpoint are all unchanged.

import { useRouter } from "next/navigation";
import {
  Activity,
  ArmchairIcon,
  ArrowUpRight,
  Award,
  Clock,
  Footprints,
  Layers,
  Move3d,
  MoveUp,
  MoveDiagonal,
  Hourglass,
  ChevronsRight,
  PersonStanding,
  Scale,
  StretchHorizontal,
  TimerIcon,
  type LucideIcon,
} from "lucide-react";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { useAuth } from "@/contexts/AuthContext";

interface ProductCard {
  /** Module route the doctor is launching. Stored in sessionStorage
   *  under INTENDED_MODULE_KEY so the patient list page can forward
   *  here with ?patientId=<id> appended. */
  targetRoute: string;
  eyebrow: string;
  title: string;
  body: string;
  gradient: string;
  icon: LucideIcon;
  iconTone: string;
}

// Same key the patient list page reads in app/dashboard/patients/page.tsx.
// Keep the two in sync if either side is ever renamed.
const INTENDED_MODULE_KEY = "motionlens.intendedModule";
const PATIENT_LIST_PATH = "/dashboard/patients";

const PRODUCTS: ProductCard[] = [
  {
    targetRoute: "/gait/upload",
    eyebrow: "Gait module",
    title: "Walk in. Walk out with data.",
    body:
      "Cadence, stride time, step symmetry, joint excursion — extracted from a single video. Cycle-locked charts and reference normal bands.",
    gradient:
      "linear-gradient(135deg, rgba(234,88,12,0.20) 0%, rgba(79,195,247,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Footprints,
    iconTone: "text-cyan-500",
  },
  {
    targetRoute: "/biomech",
    eyebrow: "Biomechanics module",
    title: "Range of motion, objectively.",
    body:
      "Shoulder, neck, knee, hip and ankle ROM — peak angles, target ranges, instant good/fair/poor classification. Live or upload.",
    gradient:
      "linear-gradient(135deg, rgba(255,183,77,0.18) 0%, rgba(234,88,12,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Activity,
    iconTone: "text-amber-500",
  },
  {
    targetRoute: "/posture",
    eyebrow: "Posture module",
    title: "Static posture, instantly.",
    body:
      "Front and side photo screening — head/shoulder/hip tilt, forward-head shift, plumb-line alignment, knee tracking. All client-side.",
    gradient:
      "linear-gradient(135deg, rgba(168,85,247,0.18) 0%, rgba(34,197,94,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: PersonStanding,
    iconTone: "text-emerald-500",
  },
  {
    targetRoute: "/orthopedic/trendelenburg",
    eyebrow: "Orthopedic test",
    title: "Trendelenburg, in 30 seconds.",
    body:
      "Live single-leg-stance assessment for hip-abductor strength — 30-second hold per side, automatic pelvic tilt + trunk lean tracking, side-by-side report.",
    gradient:
      "linear-gradient(135deg, rgba(139,92,246,0.20) 0%, rgba(236,72,153,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: StretchHorizontal,
    iconTone: "text-violet-500",
  },
  {
    targetRoute: "/orthopedic/single-leg-squat",
    eyebrow: "Orthopedic test",
    title: "Single-leg squat.",
    body:
      "Live frontal-plane knee-control screen. 5 squats per side, automatic KFPPA + pelvic drop + trunk lean, composite injury-risk score.",
    gradient:
      "linear-gradient(135deg, rgba(217,70,239,0.20) 0%, rgba(236,72,153,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Move3d,
    iconTone: "text-fuchsia-500",
  },
  {
    targetRoute: "/orthopedic/sit-to-stand",
    eyebrow: "Geriatric screen",
    title: "5x Sit-to-Stand.",
    body:
      "Lower-extremity strength + fall-risk indicator. 5 timed sit-to-stand cycles, auto rep detection, fatigue + arm-uncrossing flags.",
    gradient:
      "linear-gradient(135deg, rgba(244,63,94,0.20) 0%, rgba(251,113,133,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: ArmchairIcon,
    iconTone: "text-rose-500",
  },
  {
    targetRoute: "/orthopedic/30-second-chair-stand",
    eyebrow: "Geriatric screen",
    title: "30-Second Chair Stand.",
    body:
      "CDC STEADI fall-risk screen. Max sit-to-stand reps in 30 s, age- and sex-matched norm comparison, fatigue trend.",
    gradient:
      "linear-gradient(135deg, rgba(14,165,233,0.20) 0%, rgba(56,189,248,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: TimerIcon,
    iconTone: "text-sky-500",
  },
  {
    targetRoute: "/orthopedic/single-leg-stance",
    eyebrow: "Balance test",
    title: "Single-Leg Stance.",
    body:
      "Standalone balance assessment. Eyes-open + optional eyes-closed trials per side, hold time, sway path + 95% ellipse, age-matched thresholds.",
    gradient:
      "linear-gradient(135deg, rgba(20,184,166,0.20) 0%, rgba(45,212,191,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Scale,
    iconTone: "text-teal-500",
  },
  {
    targetRoute: "/orthopedic/4-stage-balance",
    eyebrow: "Balance test",
    title: "4-Stage Balance Test.",
    body:
      "CDC fall-risk progression. 4 progressively harder static stances held for 10 s each, sway path + 95% ellipse per stage, classification on stage reached.",
    gradient:
      "linear-gradient(135deg, rgba(99,102,241,0.20) 0%, rgba(129,140,248,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Layers,
    iconTone: "text-indigo-500",
  },
  {
    targetRoute: "/orthopedic/tug",
    eyebrow: "Geriatric / fall-risk",
    title: "Timed Up and Go (TUG).",
    body:
      "Side-view 3 m walk. Backend MediaPipe decomposes the test into 5 sub-phases — sit-to-stand, walk-out, turn, walk-back, stand-to-sit — and flags balance impairment independently of total time.",
    gradient:
      "linear-gradient(135deg, rgba(234,179,8,0.20) 0%, rgba(250,204,21,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Clock,
    iconTone: "text-yellow-500",
  },
  {
    targetRoute: "/orthopedic/sppb",
    eyebrow: "Geriatric flagship · composite",
    title: "SPPB.",
    body:
      "Three-test composite session: balance stages + 4 m gait speed + 5x sit-to-stand → one 0-12 score that predicts falls, hospitalisation, and disability.",
    gradient:
      "linear-gradient(135deg, rgba(168,85,247,0.20) 0%, rgba(192,132,252,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Award,
    iconTone: "text-purple-500",
  },
  {
    targetRoute: "/orthopedic/slr",
    eyebrow: "Orthopedic test",
    title: "Straight Leg Raise.",
    body:
      "Passive hip-flexion screen. Supine patient raises one straight leg at a time; we capture the maximum angle reached while the knee stayed straight, per side.",
    gradient:
      "linear-gradient(135deg, rgba(132,204,22,0.20) 0%, rgba(163,230,53,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: MoveUp,
    iconTone: "text-lime-500",
  },
  {
    targetRoute: "/orthopedic/ake",
    eyebrow: "Orthopedic test",
    title: "Active Knee Extension.",
    body:
      "Hamstring-length screen via the 90/90 test. Supine patient holds the thigh vertical and slowly extends the knee; we capture the maximum knee angle and extension deficit, per side.",
    gradient:
      "linear-gradient(135deg, rgba(249,115,22,0.20) 0%, rgba(251,146,60,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: MoveDiagonal,
    iconTone: "text-orange-500",
  },
  {
    targetRoute: "/orthopedic/modified-thomas",
    eyebrow: "Orthopedic test",
    title: "Modified Thomas Test.",
    body:
      "Hip-flexor and rectus-femoris length screen. Patient lies on the edge of a table with one knee to chest; the other leg hangs naturally. We auto-capture settled hip + knee angles per side.",
    gradient:
      "linear-gradient(135deg, rgba(236,72,153,0.20) 0%, rgba(244,114,182,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: Hourglass,
    iconTone: "text-pink-500",
  },
  {
    targetRoute: "/orthopedic/forward-lunge",
    eyebrow: "Functional movement",
    title: "Forward Lunge.",
    body:
      "Lateral-view 5-rep lunge screen. Auto-segments each rep and reports front-knee depth, knee-over-toe excursion, trunk forward lean, and depth consistency per side.",
    gradient:
      "linear-gradient(135deg, rgba(59,130,246,0.20) 0%, rgba(96,165,250,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: ChevronsRight,
    iconTone: "text-blue-500",
  },
];

export function ProductShowcase() {
  const router = useRouter();
  const { doctor, loading } = useAuth();

  function handleCardClick(targetRoute: string) {
    if (loading) {
      // Auth status not yet known — wait for the next click rather than
      // sending the doctor somewhere wrong. The button stays visually
      // active so the click registers and they can retry instantly.
      return;
    }
    // Stash the picked module so the patient list can forward there
    // after the doctor selects a patient. sessionStorage is per-tab and
    // survives the sign-in round-trip below.
    try {
      sessionStorage.setItem(INTENDED_MODULE_KEY, targetRoute);
    } catch {
      // sessionStorage can throw in private-mode contexts. The fallback
      // is the legacy behaviour: doctor lands on the patient list, picks
      // a patient, lands on the profile — exactly what worked before.
    }
    if (doctor === null) {
      router.push(`/auth/signin?next=${encodeURIComponent(PATIENT_LIST_PATH)}`);
      return;
    }
    router.push(PATIENT_LIST_PATH);
  }

  return (
    <Section id="modules" className="bg-dots">
      <div className="max-w-2xl">
        <Badge>Fifteen modules</Badge>
        <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">
          Movement, measured<br />fifteen ways.
        </h2>
      </div>

      <div className="mt-16 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {PRODUCTS.map((p) => {
          const Icon = p.icon;
          return (
            <button
              key={p.targetRoute}
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

              {/* Top group — icon row, then badge + title + body */}
              <div className="relative">
                {/* Icon at top-left, ArrowUpRight at top-right — same
                    convention as the dashboard analyze-page tiles. */}
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

              {/* Bottom group — Launch CTA, pinned to the very bottom of
                  the card via justify-between on the parent. */}
              <div className="relative mt-8 inline-flex items-center gap-1 text-sm text-accent">
                Launch assessment
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </div>
            </button>
          );
        })}
      </div>
    </Section>
  );
}
