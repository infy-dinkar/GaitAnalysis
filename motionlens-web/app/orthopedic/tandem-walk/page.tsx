"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { TandemWalkCapture } from "@/components/orthopedic/TandemWalkCapture";

export default function TandemWalkPage() {
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
              <Badge>Balance / vestibular</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Tandem Walk<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Heel-to-toe gait screen for cerebellar and vestibular
                dysfunction. Patient walks 10 steps along a taped line toward
                the camera. We auto-detect each footstrike, measure the
                lateral deviation from the patient&apos;s own walking-line fit,
                count missteps and arm-grab compensations, and report
                step-time variability and trunk sway.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <TandemWalkCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Frontal view (patient walks toward the camera).
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Mark a straight tape line on the floor, ~3 m long.</li>
              <li>Camera at the END of the line, at hip height, on a stable surface.</li>
              <li>Patient stands at the FAR end of the line, eyes open, hands relaxed at sides.</li>
              <li>Patient walks heel-to-toe — each new step plants the advancing heel touching the previous foot&apos;s toe.</li>
              <li>10 steps total. The trial auto-finishes on the 10th footstrike.</li>
              <li>Stop early if the patient cannot continue or steps off the line repeatedly.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
