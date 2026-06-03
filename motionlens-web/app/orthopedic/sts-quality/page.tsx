"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { STSQualityCapture } from "@/components/orthopedic/STSQualityCapture";

export default function STSQualityPage() {
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
                Sit-to-Stand Quality<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Movement-quality assessment of the sit-to-stand cycle — used
                post-TKR/THR and in geriatric rehab. Patient performs 3 reps at
                a self-selected pace. We auto-segment each rep into sit-to-stand,
                pause, and stand-to-sit phases, capture trunk lean and knee angle
                at seat-off, score smoothness, and flag hand push-off
                compensation.
              </p>
              <p className="mt-3 text-xs text-muted">
                This is separate from the <strong>5x Sit-to-Stand (C2)</strong>
                {" "}geriatric speed test — that module measures how fast the
                patient completes 5 reps, this one measures how well.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <STSQualityCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Lateral view (side of patient).
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Standard chair, ~45 cm seat height, NO armrests.</li>
              <li>Patient seated, feet flat on the floor, arms CROSSED at the chest.</li>
              <li>Camera at hip height, ~6 feet (2 m) away, perpendicular to the body.</li>
              <li>Full body in frame, side-on. The side facing the camera is what we measure.</li>
              <li>Patient stands up at a comfortable pace, pauses briefly, sits back down with control.</li>
              <li>Repeat 3 times. The system auto-finishes after the 3rd stand.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
