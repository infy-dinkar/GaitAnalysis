import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";

const PRODUCTS = [
  {
    href: "/gait",
    eyebrow: "Gait module",
    title: "Walk in. Walk out with data.",
    body:
      "Cadence, stride time, step symmetry, joint excursion — extracted from a single video. Cycle-locked charts and reference normal bands.",
    gradient:
      "linear-gradient(135deg, rgba(234,88,12,0.20) 0%, rgba(79,195,247,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    href: "/biomech",
    eyebrow: "Biomechanics module",
    title: "Range of motion, objectively.",
    body:
      "Shoulder, neck, knee, hip and ankle ROM — peak angles, target ranges, instant good/fair/poor classification. Live or upload.",
    gradient:
      "linear-gradient(135deg, rgba(255,183,77,0.18) 0%, rgba(234,88,12,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    href: "/posture",
    eyebrow: "Posture module",
    title: "Static posture, instantly.",
    body:
      "Front and side photo screening — head/shoulder/hip tilt, forward-head shift, plumb-line alignment, knee tracking. All client-side.",
    gradient:
      "linear-gradient(135deg, rgba(168,85,247,0.18) 0%, rgba(34,197,94,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
];

export function ProductShowcase() {
  return (
    <Section id="modules" className="bg-dots">
      <div className="max-w-2xl">
        <Badge>Three modules</Badge>
        <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">
          Movement, measured<br />three ways.
        </h2>
      </div>

      <div className="mt-16 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {PRODUCTS.map((p) => (
          <Link
            key={p.href}
            href={p.href}
            className="group relative flex aspect-[4/3] flex-col justify-between overflow-hidden rounded-hero border border-border bg-elevated p-8 transition-all duration-300 hover:border-accent hover:shadow-glow-sm md:p-10"
          >
            <div
              className="pointer-events-none absolute inset-0 opacity-90 transition-opacity duration-500 group-hover:opacity-100"
              style={{ background: p.gradient }}
              aria-hidden
            />
            <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" aria-hidden />

            <div className="relative">
              <Badge>{p.eyebrow}</Badge>
            </div>

            <div className="relative">
              <h3 className="text-2xl font-semibold tracking-tight md:text-3xl">{p.title}</h3>
              <p className="mt-4 max-w-md text-sm leading-relaxed text-muted">{p.body}</p>
              <div className="mt-6 inline-flex items-center gap-1 text-sm text-accent">
                Learn more
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </div>
            </div>
          </Link>
        ))}
      </div>
    </Section>
  );
}
