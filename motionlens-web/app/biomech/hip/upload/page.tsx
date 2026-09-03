"use client";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ApiUploadAssessment } from "@/components/biomech/ApiUploadAssessment";
import { HIP_MOVEMENTS, type HipMovementId } from "@/lib/biomech/hip";
import { resolveLegacyMovement } from "@/lib/biomech/movements";

function HipUploadInner() {
  const params = useSearchParams();
  const movementId =
    (params.get("movement") as HipMovementId) || "flexion_extension";
  const sideParam = params.get("side");
  const side: "left" | "right" = sideParam === "left" ? "left" : "right";
  // A deprecated single-direction id can still arrive from an old
  // bookmark or a hand-typed ?movement=. There is no backend branch
  // for these, so send the operator to the merged test that replaced
  // them instead of failing the analysis. Silent replace() — no
  // history entry, no error screen.
  const router = useRouter();
  const legacyTarget = resolveLegacyMovement("hip", movementId);
  useEffect(() => {
    if (!legacyTarget) return;
    const next = new URLSearchParams(params.toString());
    next.set("movement", legacyTarget);
    router.replace(`/biomech/hip/upload?${next.toString()}`);
  }, [legacyTarget, params, router]);

  const movement =
    HIP_MOVEMENTS.find((m) => m.id === movementId) ?? HIP_MOVEMENTS[0];

  if (legacyTarget) return null;

  return (
    <ApiUploadAssessment
      bodyPart="hip"
      movementId={movement.id}
      movementLabel={`Hip · ${movement.label}`}
      description={movement.description}
      target={movement.target}
      side={side}
    />
  );
}

export default function HipUploadPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div>
              <Badge>Video upload</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Hip — video analysis
              </h1>
              <p className="mt-2 text-sm text-muted">
                Pose detection runs entirely in your browser — your video never leaves
                this device. The peak hip ROM angle is computed locally.
              </p>
            </div>
            <Link href="/biomech/hip">
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
          </div>
          <div className="mt-10">
            <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
              <HipUploadInner />
            </Suspense>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
