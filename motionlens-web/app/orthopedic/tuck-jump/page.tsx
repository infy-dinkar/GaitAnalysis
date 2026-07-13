"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { TuckJumpCapture } from "@/components/orthopedic/TuckJumpCapture";

export default function TuckJumpPage() {
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
              <Badge>Injury-risk screen · frontal</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Tuck Jump Assessment<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Myer&apos;s Tuck Jump Assessment (TJA) — frontal-view screen
                for lower-extremity injury-risk. Patient performs continuous
                tuck jumps (knees to chest at apex, land on both feet,
                immediately re-jump) for ~10 seconds. Session is scored
                against Myer&apos;s 10-item checklist and classified good /
                moderate / poor.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Honest single-camera coverage:</strong> 8 of the 10
                items are scorable from a single frontal camera (valgus at
                landing, thigh height, L/R symmetry, foot spread, contact
                timing, pause between jumps, technique-declines-with-fatigue,
                footprint drift). Items 5 (foot yaw) and 7 (excessive
                contact noise) are marked <em>not assessed</em> — foot yaw
                is unreliable at MediaPipe&apos;s resolution and contact
                noise needs force-plate or audio input.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Scale calibration:</strong> the system measures body
                pixel height during the standing window and combines it
                with the entered standing height to convert pixel heights
                to centimetres. Without calibration, Myer&apos;s classification
                remains valid — the scoring thresholds are fraction-of-
                leg-length, not centimetre.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <TuckJumpCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              FRONTAL view, full body in frame.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip-to-shoulder height, ~2.5 m away, perpendicular
                to the patient. Patient faces the camera directly.
              </li>
              <li>
                Feet shoulder-width apart at the start; hands loose at
                sides (Myer records with hands on head, but the pose model
                works either way).
              </li>
              <li>
                Hold a ~1 s static standing pose BEFORE the first jump —
                this is when the engine locks the baseline hip Y, ankle
                positions, and shoulder-hip span.
              </li>
              <li>
                Frame must show the full body: head, shoulders, hips,
                knees and BOTH feet — the engine scores valgus (hip-knee-
                ankle) and footprint (ankle X drift) from these landmarks.
              </li>
              <li>
                Perform continuous tuck jumps for ~10 seconds — knees pull
                up to chest at apex, land on both feet, immediately
                re-jump with no long pauses.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
