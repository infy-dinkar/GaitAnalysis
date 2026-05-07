import Link from "next/link";
import {
  Activity,
  ArmchairIcon,
  ArrowUpRight,
  Footprints,
  Move3d,
  PersonStanding,
  StretchHorizontal,
  TimerIcon,
  type LucideIcon,
} from "lucide-react";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";

interface ProductCard {
  href: string;
  eyebrow: string;
  title: string;
  body: string;
  gradient: string;
  icon: LucideIcon;
  iconTone: string;
}

const PRODUCTS: ProductCard[] = [
  {
    href: "/gait",
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
    href: "/biomech",
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
    href: "/posture",
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
    href: "/orthopedic/trendelenburg",
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
    href: "/orthopedic/single-leg-squat",
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
    href: "/orthopedic/sit-to-stand",
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
    href: "/orthopedic/30-second-chair-stand",
    eyebrow: "Geriatric screen",
    title: "30-Second Chair Stand.",
    body:
      "CDC STEADI fall-risk screen. Max sit-to-stand reps in 30 s, age- and sex-matched norm comparison, fatigue trend.",
    gradient:
      "linear-gradient(135deg, rgba(14,165,233,0.20) 0%, rgba(56,189,248,0.10) 50%, rgba(28,28,33,0.0) 100%)",
    icon: TimerIcon,
    iconTone: "text-sky-500",
  },
];

export function ProductShowcase() {
  return (
    <Section id="modules" className="bg-dots">
      <div className="max-w-2xl">
        <Badge>Seven modules</Badge>
        <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">
          Movement, measured<br />seven ways.
        </h2>
      </div>

      <div className="mt-16 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {PRODUCTS.map((p) => {
          const Icon = p.icon;
          return (
            <Link
              key={p.href}
              href={p.href}
              className="group relative m-[2px] flex min-h-[436px] flex-col justify-between overflow-hidden rounded-hero border border-border bg-elevated p-8 transition-all duration-300 hover:border-accent hover:shadow-glow-sm md:min-h-[476px] md:p-10"
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

              {/* Bottom group — Learn more, pinned to the very bottom of
                  the card via justify-between on the parent. */}
              <div className="relative mt-8 inline-flex items-center gap-1 text-sm text-accent">
                Learn more
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </div>
            </Link>
          );
        })}
      </div>
    </Section>
  );
}
