"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SingleLegHopCapture } from "@/components/orthopedic/SingleLegHopCapture";

export default function SingleLegHopPage() {
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
              <Badge>Functional hop test</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Single-Leg Hop<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Forward hop for distance — patient stands side-on to the
                camera on the test leg, hops forward as far as comfortable,
                and lands on the SAME leg. Up to 3 trials per leg, both
                legs tested. The primary outcome is hop distance in cm; the
                Limb Symmetry Index (LSI = weaker ÷ stronger × 100) flags
                side-to-side asymmetry. ACL clearance convention:
                LSI ≥ 90 %.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Scale calibration:</strong> the system measures the
                patient&apos;s body pixel height during the standing window
                and combines it with the entered standing height to
                convert pixel distances to centimetres. Without
                calibration, hops are reported in relative pixel units
                only and the LSI classification is not applied.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <SingleLegHopCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Lateral view, full body in frame.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip-to-shoulder height, ~2.5 m away, perpendicular
                to the patient&apos;s direction of motion.
              </li>
              <li>
                Patient stands SIDE-ON — facing the side they&apos;ll hop
                toward.
              </li>
              <li>
                Enough clear floor space ahead for a comfortable hop
                (~1.5–2 m of clear floor visible).
              </li>
              <li>
                Frame must show the full body: head, hips, knees, and BOTH
                feet (the test-side foot for hop-distance, the contralateral
                foot for the single-leg validity gate).
              </li>
              <li>
                Hold a ~1 s static stance on the test leg BEFORE the first
                hop — this is when the engine locks the baseline.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
