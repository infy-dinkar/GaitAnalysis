"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { OverheadSquatCapture } from "@/components/orthopedic/OverheadSquatCapture";

export default function OverheadSquatPage() {
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
              <Badge>Movement screen · frontal</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Overhead Squat Assessment<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                NASM/FMS-style overhead-squat movement screen — frontal-
                view single camera. Patient stands feet shoulder-width
                apart, arms straight overhead (biceps by ears), performs
                3-5 slow squats to about parallel depth, returning fully
                to standing between reps. Scored against a 7-item frontal-
                plane checklist and classified good / moderate / poor.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Honest frontal-view coverage:</strong> 5 of the 7
                items are scorable from a single frontal camera (knee
                valgus at bottom, hip / pelvic drop, foot placement, arm
                drop from overhead, depth proxy). Items 6 (excessive torso
                forward lean) and 7 (heel rise) are marked <em>not
                assessed</em> — the first needs a sagittal camera, the
                second needs a feet close-up view.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Scale calibration:</strong> the system measures
                body pixel height during the standing window and combines
                it with the entered standing height to convert pixel
                measurements to centimetres. Without calibration,
                classification remains valid — the thresholds are
                fraction-of-leg-length, not centimetre.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <OverheadSquatCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              FRONTAL view, full body + arms in frame.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2.5-3 m away, perpendicular to the
                patient. Patient faces the camera directly.
              </li>
              <li>
                Ensure the top of the frame is high enough that arms
                overhead do not clip out — otherwise the arm-drop item
                degrades to <em>not assessed</em>.
              </li>
              <li>
                Feet shoulder-width apart, toes forward. Arms held
                straight overhead (biceps by ears) throughout.
              </li>
              <li>
                Hold a ~1 s static standing pose BEFORE the first squat —
                this is when the engine locks the baseline hip Y,
                shoulder Y, and wrist Y (overhead reference).
              </li>
              <li>
                Perform 3-5 slow overhead squats to about parallel depth.
                Return fully to standing between reps; keep arms overhead
                throughout every rep.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
