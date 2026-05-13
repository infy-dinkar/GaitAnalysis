"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SPPBCapture } from "@/components/orthopedic/SPPBCapture";

export default function SPPBPage() {
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
              <Badge>Geriatric flagship · composite</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                SPPB — Short Physical Performance Battery<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Three-component geriatric screen run in one session — balance
                stages, 4 m gait speed, and 5x sit-to-stand. Combines into a
                single 0-12 composite score that predicts falls,
                hospitalisation, and disability over the following 12-24 months.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <SPPBCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup overview</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                <span className="font-medium text-foreground">Component 1 (Balance):</span>{" "}
                <em>front view</em> — patient faces the camera, full body visible.
              </li>
              <li>
                <span className="font-medium text-foreground">Component 2 (Gait speed):</span>{" "}
                <em>side view</em> — reposition the camera so the whole 4 m walking path is in frame.
              </li>
              <li>
                <span className="font-medium text-foreground">Component 3 (Chair stand):</span>{" "}
                <em>side view</em> — same lateral position as Component 2; the patient sits in a standard ~45 cm chair without armrests.
              </li>
              <li>The orchestrator prompts the operator to reposition the camera between Components 1 and 2.</li>
              <li>30-60 s rest between components is optional — the operator clicks Continue when both are ready.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
