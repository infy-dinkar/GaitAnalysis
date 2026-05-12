"use client";
import Link from "next/link";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Camera, Upload } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { MovementGrid } from "@/components/biomech/MovementGrid";
import { SideSelector } from "@/components/biomech/SideSelector";
import { ANKLE_MOVEMENTS } from "@/lib/biomech/ankle";

export default function AnkleSetupPage() {
  return (
    <Suspense fallback={null}>
      <AnkleSetupInner />
    </Suspense>
  );
}

function AnkleSetupInner() {
  const params = useSearchParams();
  const patientId = params.get("patientId");
  const patientQS = patientId ? `&patientId=${patientId}` : "";

  const [movement, setMovement] = useState<string | null>(null);
  const [side, setSide] = useState<"left" | "right" | null>(null);
  const canProceed = !!movement && !!side;

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="max-w-2xl">
            <Badge>Ankle assessment</Badge>
            <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
              Configure the assessment<span className="text-accent">.</span>
            </h1>
            <p className="mt-5 text-lg text-muted">
              Ankle dorsiflexion (toes up) and plantarflexion (toes down).
              Server-side MediaPipe analysis with foot landmarks for an
              accurate joint angle. Capture live from the camera or
              upload a pre-recorded clip.
            </p>
          </div>

          <div className="mt-12 space-y-12">
            <section>
              <h2 className="text-sm font-bold uppercase tracking-[0.12em] text-foreground">
                Movement
              </h2>
              <div className="mt-4">
                <MovementGrid
                  options={ANKLE_MOVEMENTS}
                  selected={movement}
                  onSelect={setMovement}
                />
              </div>
            </section>

            <section>
              <h2 className="text-sm font-bold uppercase tracking-[0.12em] text-foreground">
                Side
              </h2>
              <div className="mt-4">
                <SideSelector selected={side} onSelect={setSide} />
              </div>
            </section>

            <section>
              <h2 className="text-sm font-bold uppercase tracking-[0.12em] text-foreground">
                Capture mode
              </h2>
              {/* Both modes route to the SAME backend MediaPipe pipeline.
                  The browser cannot do ankle math itself (MoveNet, 17 kp,
                  has no foot_index landmark) — but it CAN record video
                  via getUserMedia + MediaRecorder and ship the bytes to
                  the server. */}
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <Link
                  href={`/biomech/ankle/live?movement=${movement ?? ""}&side=${side ?? ""}${patientQS}`}
                  className={`rounded-card border p-6 transition ${
                    canProceed
                      ? "border-border bg-surface hover:border-accent"
                      : "pointer-events-none border-border/50 bg-surface/40 opacity-50"
                  }`}
                >
                  <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-accent/10 text-accent">
                    <Camera className="h-5 w-5" />
                  </span>
                  <h3 className="mt-4 text-lg font-semibold">Live recording</h3>
                  <p className="mt-2 text-sm text-muted">
                    Record a short clip from your device camera. The video
                    is uploaded once and analysed server-side; nothing is
                    stored after the result is returned.
                  </p>
                </Link>

                <Link
                  href={`/biomech/ankle/upload?movement=${movement ?? ""}&side=${side ?? ""}${patientQS}`}
                  className={`rounded-card border p-6 transition ${
                    canProceed
                      ? "border-border bg-surface hover:border-accent"
                      : "pointer-events-none border-border/50 bg-surface/40 opacity-50"
                  }`}
                >
                  <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-accent/10 text-accent">
                    <Upload className="h-5 w-5" />
                  </span>
                  <h3 className="mt-4 text-lg font-semibold">Video upload</h3>
                  <p className="mt-2 text-sm text-muted">
                    Drop a pre-recorded video. Backend pose analysis with
                    foot landmarks gives an accurate joint-angle reading.
                  </p>
                </Link>
              </div>
              {!canProceed && (
                <p className="mt-3 text-xs text-subtle">Select a movement and side to continue.</p>
              )}
            </section>
          </div>

          <div className="mt-16 flex justify-between">
            <Link href="/biomech">
              <Button variant="ghost">← Back</Button>
            </Link>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
