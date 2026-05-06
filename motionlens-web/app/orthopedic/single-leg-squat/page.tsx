"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SingleLegSquatCapture } from "@/components/orthopedic/SingleLegSquatCapture";

export default function SingleLegSquatPage() {
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
                Single-leg squat<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Frontal-plane knee-control screen. Patient performs five
                single-leg squats per side; the system auto-detects each rep
                and tracks knee-frontal-plane projection angle (KFPPA),
                pelvic drop, and trunk lean. Composite injury-risk score
                surfaced per side.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <SingleLegSquatCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Patient stands facing the camera squarely — both shoulders level in frame.</li>
              <li>Camera at hip height, 6+ feet (2 m) away, full body visible.</li>
              <li>Plain background, even lighting, fitted clothing where possible.</li>
              <li>Patient barefoot, arms folded across chest or out for balance.</li>
              <li>Lift the contralateral knee to the side — keep it off the stance hip.</li>
              <li>Squat to a comfortable depth at a steady tempo (~2 s down, 2 s up).</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
