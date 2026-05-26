"use client";
// /biomech/batch — multi-joint upload session. Doctor selects which
// movements they want, uploads one video per selection, gets a
// combined stacked report at the bottom with Save All + Download PDF
// buttons. Everything reuses existing single-joint analysis +
// reporting components — see BatchSession.tsx for the orchestrator.

import Link from "next/link";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { BatchSession } from "@/components/biomech/BatchSession";

function BatchInner() {
  // patientId is read inside BatchSession via usePatientContext —
  // this thin wrapper exists so the page renders inside <Suspense>
  // (a Next.js 14+ requirement when a child uses useSearchParams).
  // Touching useSearchParams here guarantees the page is dynamically
  // rendered.
  useSearchParams();
  return <BatchSession />;
}

export default function BiomechBatchPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div>
              <Badge>Biomechanics · batch upload</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Multi-joint assessment
              </h1>
              <p className="mt-2 max-w-2xl text-sm text-muted">
                Run several biomechanics tests in one session. Pick the
                movements, upload a video per movement, and get a
                combined report you can save to the patient&apos;s history
                or download as a single PDF.
              </p>
            </div>
            <Link href="/biomech">
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
          </div>
          <div className="mt-10">
            <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
              <BatchInner />
            </Suspense>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
