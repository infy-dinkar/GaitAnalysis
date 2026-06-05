"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { FunctionalReachCapture } from "@/components/orthopedic/FunctionalReachCapture";

export default function FunctionalReachPage() {
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
              <Badge>Balance / fall-risk screen</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Functional Reach<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Bedside fall-risk screen — patient stands side-on to the
                camera, raises the near arm to shoulder height, then reaches
                forward as far as comfortable WITHOUT stepping or lifting the
                heels. Three trials in one 30 s recording; the best valid
                trial is the reach distance. Reach &lt; 25 cm flags moderate
                fall risk, &lt; 15 cm flags high, &lt; 10 cm flags very high.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Scale calibration:</strong> a height-based calibration
                is on its way (Step 2). Until then the report shows reach in
                relative pixel units only and fall-risk cutoffs do not apply.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <FunctionalReachCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Lateral view, hip-to-shoulder height.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Camera at hip-to-shoulder height, ~2 m away, perpendicular to the patient.</li>
              <li>Patient stands SIDE-ON — one shoulder toward the camera.</li>
              <li>Feet flat, shoulder-width apart, behind a visible tape line.</li>
              <li>The test arm (near the camera) raises to 90° (shoulder height), fist closed; the patient reaches forward 3 times.</li>
              <li>Frame must show the full body: shoulder, wrist, hip, ankle and heel of the test side.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
