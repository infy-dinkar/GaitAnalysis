"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { TrendelenburgCapture } from "@/components/orthopedic/TrendelenburgCapture";

export default function TrendelenburgPage() {
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
                Trendelenburg test<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Single-leg-stance assessment for gluteus medius (hip
                abductor) strength via pelvic stability. Frontal-camera
                capture, 30-second hold per side, with auto-detection of
                the stance leg from foot landmarks.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <TrendelenburgCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Patient stands facing the camera, both hips visible end-to-end.</li>
              <li>Camera at hip height, 6+ feet (2 m) away, full body in frame.</li>
              <li>Plain background, even lighting, fitted clothing where possible.</li>
              <li>Patient barefoot, feet hip-width apart, arms relaxed or crossed.</li>
              <li>Lift one leg by flexing hip + knee to ~90°. Hold for 30 s.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
