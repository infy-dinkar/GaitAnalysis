"use client";
import Link from "next/link";
import { useState } from "react";
import { Camera, Upload } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { PatientForm } from "@/components/biomech/PatientForm";
import { MovementGrid } from "@/components/biomech/MovementGrid";
import { SideSelector } from "@/components/biomech/SideSelector";
import { SHOULDER_MOVEMENTS } from "@/lib/biomech/shoulder";

export default function ShoulderSetupPage() {
  const [movement, setMovement] = useState<string | null>(null);
  const [side, setSide] = useState<"left" | "right" | null>(null);
  const canProceed = !!movement && !!side;

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="max-w-2xl">
            <Badge>Shoulder assessment</Badge>
            <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
              Configure the assessment<span className="text-accent">.</span>
            </h1>
            <p className="mt-5 text-lg text-muted">
              Enter patient details, choose a movement, then pick live capture or video upload.
            </p>
          </div>

          <div className="mt-12 space-y-12">
            <section>
              <h2 className="text-sm font-bold uppercase tracking-[0.12em] text-foreground">
                Patient details
              </h2>
              <div className="mt-4">
                <PatientForm />
              </div>
            </section>

            <section>
              <h2 className="text-sm font-bold uppercase tracking-[0.12em] text-foreground">
                Movement
              </h2>
              <div className="mt-4">
                <MovementGrid
                  options={SHOULDER_MOVEMENTS}
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
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <Link
                  href={`/biomech/shoulder/live?movement=${movement ?? ""}&side=${side ?? ""}`}
                  className={`rounded-card border p-6 transition ${
                    canProceed
                      ? "border-border bg-surface hover:border-accent"
                      : "pointer-events-none border-border/50 bg-surface/40 opacity-50"
                  }`}
                >
                  <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-accent/10 text-accent">
                    <Camera className="h-5 w-5" />
                  </span>
                  <h3 className="mt-4 text-lg font-semibold">Live camera</h3>
                  <p className="mt-2 text-sm text-muted">
                    Use your webcam. 10-second recording window with peak angle tracking.
                  </p>
                </Link>
                <Link
                  href={`/biomech/shoulder/upload?movement=${movement ?? ""}&side=${side ?? ""}`}
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
                    Drop a pre-recorded video. Frame-by-frame analysis with full angle chart.
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
