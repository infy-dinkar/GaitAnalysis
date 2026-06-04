"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { PronatorDriftCapture } from "@/components/orthopedic/PronatorDriftCapture";

export default function PronatorDriftPage() {
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
              <Badge>Neurological screen</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Pronator Drift<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Bedside screen for subtle upper-motor-neuron weakness.
                Patient holds both arms extended forward at shoulder height
                with palms up, eyes closed, for a 20-second hold. We track
                each wrist&apos;s vertical drop from baseline and flag
                asymmetric drift patterns. Audio cues mark the start and
                end of the hold (eyes are closed, no visual cue lands).
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>2D limitation:</strong> the test&apos;s third
                classical sign — forearm rotation/pronation as the arm
                drops — is NOT assessable by a single-lens camera. This
                module measures vertical drop only; clinical judgement is
                required.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <PronatorDriftCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Frontal view, chest height.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Camera at chest height, ~6 ft (2 m) away, perpendicular to the patient.</li>
              <li>Patient stands or sits facing the camera squarely.</li>
              <li>Both arms extended FORWARD at shoulder height (90° shoulder flexion), elbows straight, palms UP.</li>
              <li>Frame must show both shoulders AND both extended wrists together.</li>
              <li>Patient closes eyes on the START beep. Holds for 20 seconds. Opens eyes on the END beep.</li>
              <li>Audio cues handle the timing — the patient should not need to look at anything.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
