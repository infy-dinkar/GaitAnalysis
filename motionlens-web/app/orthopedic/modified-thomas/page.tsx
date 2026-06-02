"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ModifiedThomasCapture } from "@/components/orthopedic/ModifiedThomasCapture";

export default function ModifiedThomasPage() {
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
                Modified Thomas Test<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Hip-flexor and rectus-femoris length screen. Patient lies on the
                edge of a table with one knee held to chest and the other leg
                hanging naturally. We auto-capture two settled angles per side —
                the hip angle (hip-flexor length) and the knee angle (rectus
                femoris length).
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <ModifiedThomasCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Lateral view (side of patient) — TALL frame.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Patient sits on the edge of a flat table, then lies back so the
                upper body rests on the table and both legs hang off the edge.
              </li>
              <li>
                Camera at hip height, ~6 feet (2 m) away, side-on. Use a
                portrait/tall frame so the whole body — shoulder at the top,
                hanging ankle at the bottom — fits in view.
              </li>
              <li>
                Test one leg at a time. The OPPOSITE knee is pulled to the
                chest and held there (this stabilises the pelvis — it is NOT
                measured).
              </li>
              <li>
                The TEST leg hangs naturally off the table edge. Patient should
                relax and let gravity do the work.
              </li>
              <li>
                Hold the settled position for at least 2 seconds. The system
                auto-captures the moment the pose stops moving.
              </li>
              <li>
                Swap legs (and the knee pulled to chest) and repeat for the
                other side.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
