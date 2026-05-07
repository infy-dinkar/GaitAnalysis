"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { FourStageBalanceCapture } from "@/components/orthopedic/FourStageBalanceCapture";

export default function FourStageBalancePage() {
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
                4-Stage Balance Test<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                CDC fall-risk progression. Patient holds 4 progressively
                harder static stances — side-by-side, semi-tandem, tandem,
                and single-leg — for 10 s each. The test stops at the
                first stage they cannot complete; sway path, 95% sway
                ellipse, and final-stage classification are auto-computed.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <FourStageBalanceCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Patient stands facing the camera, full body in frame.</li>
              <li>Camera at hip height, ~2 m away.</li>
              <li>Plain background, even lighting, fitted clothing where possible.</li>
              <li>Patient barefoot, level surface, no support within arm&apos;s reach.</li>
              <li>For stages 1 and 2, ensure both ankles are visible and not overlapping in the frame.</li>
              <li>Each stage holds for 10 s. Test progresses only on successful holds.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
