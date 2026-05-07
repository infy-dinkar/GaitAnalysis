"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SitToStandCapture } from "@/components/orthopedic/SitToStandCapture";

export default function SitToStandPage() {
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
                5x Sit-to-Stand<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Lower-extremity strength and fall-risk indicator (validated
                geriatric outcome). Five sit-to-stand cycles, auto-detected,
                with per-rep timing, depth tracking, fatigue flag, and
                arm-uncrossing detection.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <SitToStandCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Use a standard chair without armrests.</li>
              <li>Patient seated in profile to the camera, full body in frame.</li>
              <li>Patient back against backrest, feet flat, arms crossed at chest.</li>
              <li>Camera at hip-knee height, 6+ feet (2 m) away.</li>
              <li>On Start, patient performs five sit-to-stand cycles as fast as safely possible.</li>
              <li>Arms must remain crossed throughout — uncrossing flags the trial.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
