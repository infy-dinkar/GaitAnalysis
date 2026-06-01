"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SLRCapture } from "@/components/orthopedic/SLRCapture";

export default function SLRPage() {
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
              <Badge>Orthopedic test</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Straight Leg Raise<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Passive-range hip flexion test for hamstring length and lumbar
                nerve-root irritability. Patient lies supine; one straight leg
                is raised at a time. We capture the maximum angle reached while
                the knee stayed straight, per side.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <SLRCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Lateral view (side of patient).
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Patient lies supine (face up) on a flat surface; arms by the sides.</li>
              <li>Camera at hip height, ~6 feet (2 m) away, side-on.</li>
              <li>Frame the full body — head, torso, hip, knee, and ankle visible.</li>
              <li>Test one leg at a time. Place the camera on the SAME side as the leg being raised.</li>
              <li>Patient raises the leg slowly with the knee fully straight, lifting as high as comfortable.</li>
              <li>Lower the leg, swap the camera to the other side, and repeat for the other leg.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
