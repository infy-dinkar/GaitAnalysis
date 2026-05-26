"use client";
// Ankle real-time live-camera mode. Uses the same LiveAssessment +
// LiveBiomechCamera stack as shoulder / neck / knee / hip — runs
// browser-side BlazePose Full (33-keypoint WASM) and computes the
// per-frame ankle angle via lib/biomech/ankle-live.ts.
//
// Why this works now (it didn't before): the legacy MoveNet detector
// only emitted 17 keypoints with no foot landmarks, so live-side
// ankle math could only read shin-from-vertical and ankle ROM had to
// go to a backend-MediaPipe video-upload endpoint. BlazePose Full
// includes LEFT_HEEL / RIGHT_HEEL / LEFT_FOOT_INDEX /
// RIGHT_FOOT_INDEX (indices 29-32), so we no longer need a
// record-and-upload detour for live ankle assessment.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { LiveAssessment } from "@/components/biomech/LiveAssessment";
import {
  ANKLE_MOVEMENTS,
  type AnkleMovementId,
} from "@/lib/biomech/ankle-live";

function AnkleLiveInner() {
  const params = useSearchParams();
  const movementId = (params.get("movement") as AnkleMovementId) || "flexion";
  const sideParam = params.get("side");
  const side: "left" | "right" = sideParam === "left" ? "left" : "right";
  const movement =
    ANKLE_MOVEMENTS.find((m) => m.id === movementId) ?? ANKLE_MOVEMENTS[0];

  return (
    <LiveAssessment
      bodyPart="ankle"
      movementId={movement.id}
      movementLabel={`Ankle · ${movement.label}`}
      description={movement.description}
      target={movement.target}
      side={side}
    />
  );
}

export default function AnkleLivePage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="relative">
            <Link
              href="/biomech/ankle"
              className="absolute right-0 top-0 z-10"
            >
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
            <div className="text-center">
              <Badge>Live assessment</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Ankle — live capture
              </h1>
            </div>
          </div>
          <div className="mt-10">
            <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
              <AnkleLiveInner />
            </Suspense>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
