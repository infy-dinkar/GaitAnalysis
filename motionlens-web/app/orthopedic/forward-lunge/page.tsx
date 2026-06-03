"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ForwardLungeCapture } from "@/components/orthopedic/ForwardLungeCapture";

export default function ForwardLungePage() {
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
              <Badge>Functional movement</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Forward Lunge<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Functional lower-extremity screen. Patient performs 5
                forward lunges per side from a lateral camera. We
                auto-segment each rep and report front-knee depth,
                knee-over-toe excursion, trunk forward lean, and depth
                consistency across the set.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <ForwardLungeCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Lateral view, on the side of the front (test) leg.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Open space ahead of the patient so they can step forward into the lunge.</li>
              <li>Camera at hip height, ~6 feet (2 m) away, perpendicular to the body — the full body should be in frame.</li>
              <li>Place the camera on the SAME side as the leg that will step forward (the test leg).</li>
              <li>Patient stands tall, hands on hips or at the sides. Steps forward into the lunge with the test leg.</li>
              <li>Lower until the back knee approaches the floor (front knee close to 90°), hold ~1 s, push back to standing.</li>
              <li>Perform 5 reps, then swap to test the other leg (swap camera side too).</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
