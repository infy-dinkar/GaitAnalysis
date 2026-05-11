"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { TUGCapture } from "@/components/orthopedic/TUGCapture";

export default function TUGPage() {
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
              <Badge>Geriatric / fall-risk test</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Timed Up and Go (TUG)<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Validated geriatric mobility screen. Patient stands from a
                chair, walks 3 metres, turns, walks back, and sits. Backend
                MediaPipe analysis decomposes the test into 5 sub-phases —
                sit-to-stand, walk-out, turn, walk-back, stand-to-sit — and
                flags balance impairment independently of total time.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <TUGCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              Stand sideways to the camera (side view).
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>Standard chair (~45 cm seat) at one end of a 3 m path.</li>
              <li>Visible marker (cone or tape) at exactly 3 metres from the chair.</li>
              <li>Camera perpendicular to the walk path — entire 3 m visible in frame.</li>
              <li>Patient barefoot or in low-heeled shoes; back against the chair backrest.</li>
              <li>
                <span className="font-medium text-foreground">Record mode:</span>{" "}
                recording starts with the operator&apos;s &quot;Go&quot; cue and stops when the patient is fully seated again.
              </li>
              <li>
                <span className="font-medium text-foreground">Upload mode:</span>{" "}
                pre-recorded video must be trimmed to start at the &quot;Go&quot; cue and end at full seat contact, 5–60 seconds, ≤ 100 MB, side view of the entire 3 m path. Supported formats: MP4, WebM, MOV, MKV.
              </li>
              <li>Video is processed on the server (MediaPipe 33-keypoint BlazePose Full).</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
