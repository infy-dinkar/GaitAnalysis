"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ChairStand30sCapture } from "@/components/orthopedic/ChairStand30sCapture";

export default function ChairStand30sPage() {
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
              <Badge>Geriatric screen</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                30-Second Chair Stand<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                CDC STEADI fall-risk screen. Patient performs as many full
                sit-to-stand cycles as possible in 30 seconds; the system
                auto-counts valid reps and grades the result against
                age- and sex-matched CDC norms.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <ChairStand30sCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Use a standard chair without armrests.</li>
              <li>Patient seated in profile to the camera, full body in frame.</li>
              <li>Patient back against backrest, feet flat, arms crossed at chest.</li>
              <li>Camera at hip-knee height, 6+ feet (2 m) away.</li>
              <li>On Start, patient performs as many full sit-to-stand cycles as possible in 30 s.</li>
              <li>Arms must remain crossed throughout — uncrossing flags the trial.</li>
              <li>Patient&apos;s age and sex must be on the profile for accurate CDC norm comparison.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
