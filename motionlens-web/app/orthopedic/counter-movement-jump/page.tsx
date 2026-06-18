"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { CMJCapture } from "@/components/orthopedic/CMJCapture";

export default function CounterMovementJumpPage() {
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
              <Badge>Vertical jump · power</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Counter-Movement Jump<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Lateral-view vertical jump for power. Patient dips into a
                brief squat then jumps straight up as high as possible and
                lands on both feet — up to 3 trials per recording. Primary
                outcome is jump height (cm) from the hip-midpoint apex;
                secondary is flight time (s) plus a gravity-based physics
                cross-check (h = g · t² / 8) that stays useful even when
                the pose-based cm conversion fails.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Scale calibration:</strong> the system measures
                body pixel height during the standing window and combines
                it with the entered standing height to convert pixel
                heights to centimetres. Without calibration the pose-
                based height is in pixels only — flight time and the
                physics estimate remain valid.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <CMJCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Lateral view, full body in frame.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip-to-shoulder height, ~2.5 m away,
                perpendicular to the patient.
              </li>
              <li>
                Patient stands SIDE-ON, feet shoulder-width, arms relaxed.
              </li>
              <li>
                Hold a ~1 s static standing pose BEFORE the first jump —
                this is when the engine locks the baseline.
              </li>
              <li>
                Frame must show the full body: head, hips, knees and BOTH
                feet (the engine uses hip Y for jump-height apex and both
                ankles for takeoff/landing event detection).
              </li>
              <li>
                Land on both feet each time — the validity gate requires
                both ankles airborne together and both back on the ground
                together.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
