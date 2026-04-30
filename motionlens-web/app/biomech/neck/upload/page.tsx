"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ApiUploadAssessment } from "@/components/biomech/ApiUploadAssessment";
import { NECK_MOVEMENTS, type NeckMovementId } from "@/lib/biomech/neck";

function NeckUploadInner() {
  const params = useSearchParams();
  const movementId = (params.get("movement") as NeckMovementId) || "flexion";
  const movement =
    NECK_MOVEMENTS.find((m) => m.id === movementId) ?? NECK_MOVEMENTS[0];

  return (
    <ApiUploadAssessment
      bodyPart="neck"
      movementId={movement.id}
      movementLabel={`Neck · ${movement.label}`}
      description={movement.description}
      target={movement.target}
    />
  );
}

export default function NeckUploadPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div>
              <Badge>Video upload</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Neck — server analysis
              </h1>
              <p className="mt-2 text-sm text-muted">
                Video is uploaded to the MotionLens API; the same Python engine your
                Streamlit app uses computes the peak cervical ROM angle.
              </p>
            </div>
            <Link href="/biomech/neck">
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
          </div>
          <div className="mt-10">
            <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
              <NeckUploadInner />
            </Suspense>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
