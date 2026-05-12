"use client";
// Ankle live-camera recording mode. The recording is sent to the same
// backend MediaPipe pipeline as the upload page — browser-side pose
// detection (MoveNet, 17 keypoints) does not have foot landmarks, so
// the heavy work cannot happen client-side regardless of capture mode.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { AnkleCapture } from "@/components/biomech/AnkleCapture";
import { ANKLE_MOVEMENTS, type AnkleMovementId } from "@/lib/biomech/ankle";

function AnkleLiveInner() {
  const params = useSearchParams();
  const movementId = (params.get("movement") as AnkleMovementId) || "flexion";
  const sideParam = params.get("side");
  const side: "left" | "right" = sideParam === "left" ? "left" : "right";
  const movement =
    ANKLE_MOVEMENTS.find((m) => m.id === movementId) ?? ANKLE_MOVEMENTS[0];

  return (
    <AnkleCapture
      movementId={movement.id}
      movementLabel={`Ankle · ${movement.label}`}
      description={movement.description}
      target={movement.target}
      side={side}
      initialMode="record"
    />
  );
}

export default function AnkleLivePage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div>
              <Badge>Live recording</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Ankle — live recording
              </h1>
              <p className="mt-2 text-sm text-muted">
                Capture a short side-view clip from your camera. Server-side
                MediaPipe BlazePose Full (33-keypoint model with foot
                landmarks) computes the ankle joint angle from the
                shin-foot vector. The video is processed once and
                discarded — only the landmarks + peak angle are persisted.
              </p>
            </div>
            <Link href="/biomech/ankle">
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
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
