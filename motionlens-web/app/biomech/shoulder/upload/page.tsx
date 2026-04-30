"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { ApiUploadAssessment } from "@/components/biomech/ApiUploadAssessment";
import {
  SHOULDER_MOVEMENTS,
  type ShoulderMovementId,
} from "@/lib/biomech/shoulder";

function ShoulderUploadInner() {
  const params = useSearchParams();
  const movementId = (params.get("movement") as ShoulderMovementId) || "flexion";
  const movement =
    SHOULDER_MOVEMENTS.find((m) => m.id === movementId) ?? SHOULDER_MOVEMENTS[0];

  const [side, setSide] = useState<"left" | "right">("right");

  return (
    <div className="space-y-8">
      <div className="rounded-card border border-border bg-surface p-5">
        <p className="text-xs uppercase tracking-[0.12em] text-subtle">
          Side to assess
        </p>
        <div className="mt-3 flex gap-2">
          {(["right", "left"] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSide(s)}
              className={`flex-1 rounded-lg border px-4 py-2 text-sm font-medium transition ${
                side === s
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-border bg-elevated text-foreground hover:border-accent/60"
              }`}
            >
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <ApiUploadAssessment
        bodyPart="shoulder"
        movementId={movement.id}
        movementLabel={`Shoulder · ${movement.label}`}
        description={movement.description}
        target={movement.target}
        side={side}
      />
    </div>
  );
}

export default function ShoulderUploadPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div>
              <Badge>Video upload</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Shoulder — server analysis
              </h1>
              <p className="mt-2 text-sm text-muted">
                Video is uploaded to the MotionLens API; the same Python engine your
                Streamlit app uses computes the peak ROM angle.
              </p>
            </div>
            <Link href="/biomech/shoulder">
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
          </div>
          <div className="mt-10">
            <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
              <ShoulderUploadInner />
            </Suspense>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
