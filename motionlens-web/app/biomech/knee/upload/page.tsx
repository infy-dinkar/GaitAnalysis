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
import { KNEE_MOVEMENTS } from "@/lib/biomech/knee";
import { resolveLegacyMovement } from "@/lib/biomech/movements";

function KneeUploadInner() {
  const params = useSearchParams();
  const movementId = params.get("movement") || "flexion_extension";
  const sideParam = params.get("side");
  const side: "left" | "right" = sideParam === "left" ? "left" : "right";

  // A deprecated single-direction id can still arrive from an old
  // bookmark or a hand-typed ?movement=. There is no backend branch
  // for these, so send the operator to the merged test that replaced
  // them instead of failing the analysis. Silent replace() — no
  // history entry, no error screen.
  const router = useRouter();
  const legacyTarget = resolveLegacyMovement("knee", movementId);
  useEffect(() => {
    if (!legacyTarget) return;
    const next = new URLSearchParams(params.toString());
    next.set("movement", legacyTarget);
    router.replace(`/biomech/knee/upload?${next.toString()}`);
  }, [legacyTarget, params, router]);
  const movement =
    KNEE_MOVEMENTS.find((m) => m.id === movementId) ?? KNEE_MOVEMENTS[0];

  if (legacyTarget) return null;

  return (
    <ApiUploadAssessment
      bodyPart="knee"
      movementId={movement.id}
      movementLabel={`Knee · ${movement.label}`}
      description={movement.description}
      target={movement.target}
      side={side}
    />
  );
}

export default function KneeUploadPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div>
              <Badge>Video upload</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Knee — video analysis
              </h1>
              <p className="mt-2 text-sm text-muted">
                Pose detection runs entirely in your browser — your video never leaves
                this device. The peak knee ROM angle is computed locally.
              </p>
            </div>
            <Link href="/biomech/knee">
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
          </div>
          <div className="mt-10">
            <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
              <KneeUploadInner />
            </Suspense>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
