"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SingleLegStanceCapture } from "@/components/orthopedic/SingleLegStanceCapture";

export default function SingleLegStancePage() {
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
              <Badge>Balance test</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Single-Leg Stance<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Standalone single-leg balance assessment. Patient lifts one
                leg to roughly 90° hip flexion and holds as long as
                possible. Eyes-open trial is mandatory; an optional
                eyes-closed pass uses an audio cue. Hold time, sway path,
                and 95% sway ellipse area are auto-tracked and graded
                against age-matched thresholds.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <SingleLegStanceCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Patient stands facing the camera, full body in frame.</li>
              <li>Camera at hip height, 6+ feet (2 m) away.</li>
              <li>Plain background, even lighting, fitted clothing where possible.</li>
              <li>Patient barefoot, hands on hips or at sides.</li>
              <li>Lift one leg to about 90° hip flexion and hold as long as possible.</li>
              <li>Eyes-open: max 60 s. Eyes-closed: max 30 s, preceded by an audio cue.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
