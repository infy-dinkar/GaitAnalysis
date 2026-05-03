"use client";
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
  HIP_MOVEMENTS,
  type HipMovementId,
} from "@/lib/biomech/hip";

function HipLiveInner() {
  const params = useSearchParams();
  const movementId = (params.get("movement") as HipMovementId) || "flexion";
  const sideParam = params.get("side");
  const side: "left" | "right" = sideParam === "left" ? "left" : "right";
  const movement = HIP_MOVEMENTS.find((m) => m.id === movementId) ?? HIP_MOVEMENTS[0];

  return (
    <LiveAssessment
      bodyPart="hip"
      movementId={movement.id}
      movementLabel={`Hip · ${movement.label}`}
      description={movement.description}
      target={movement.target}
      side={side}
    />
  );
}

export default function HipLivePage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="relative">
            <Link
              href="/biomech/hip"
              className="absolute right-0 top-0 z-10"
            >
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
            <div className="text-center">
              <Badge>Live assessment</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Hip — live capture
              </h1>
            </div>
          </div>
          <div className="mt-10">
            <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
              <HipLiveInner />
            </Suspense>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
