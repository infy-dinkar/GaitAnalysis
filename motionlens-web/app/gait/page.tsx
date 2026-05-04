"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight, Footprints } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

const HEIGHT_KEY = "motionlens.height_cm";

export default function GaitPage() {
  const [height, setHeight] = useState("");

  useEffect(() => {
    const v = sessionStorage.getItem(HEIGHT_KEY);
    if (v) setHeight(v);
  }, []);

  function handleHeight(v: string) {
    setHeight(v);
    if (v) sessionStorage.setItem(HEIGHT_KEY, v);
    else sessionStorage.removeItem(HEIGHT_KEY);
  }

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="grid gap-12 md:grid-cols-[3fr_2fr] md:items-start">
            <div>
              <Badge>Gait analysis</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
                Walk in. Walk<br />out with data<span className="text-accent">.</span>
              </h1>
              <p className="mt-6 max-w-xl text-lg text-muted">
                Record a 10-15 second clip of the patient walking left to right (or right to
                left) at a comfortable pace. The full body should be visible end to end.
              </p>

              <div className="mt-12 space-y-12">
                <section>
                  <h2 className="text-sm font-bold uppercase tracking-[0.12em] text-foreground">
                    Height (for distance calibration)
                  </h2>
                  <div className="mt-4 max-w-xs">
                    <label className="block">
                      <span className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
                        Height (cm)
                      </span>
                      <input
                        type="number"
                        min={50}
                        max={250}
                        value={height}
                        onChange={(e) => handleHeight(e.target.value)}
                        placeholder="170"
                        className="mt-2 h-11 w-full rounded-lg border border-border bg-surface px-3 text-sm text-foreground placeholder:text-subtle transition focus:border-accent focus:outline-none"
                      />
                    </label>
                    <p className="mt-2 text-xs text-muted">
                      Used to convert pixels into meters for walking speed. Leave blank to skip.
                    </p>
                  </div>
                </section>

                <section>
                  <h2 className="text-sm font-bold uppercase tracking-[0.12em] text-foreground">
                    Capture mode
                  </h2>
                  <div className="mt-4">
                    <Link href="/gait/upload">
                      <Button size="lg">
                        Upload video
                        <ArrowRight className="h-4 w-4" />
                      </Button>
                    </Link>
                  </div>
                </section>
              </div>
            </div>

            <aside className="rounded-card border border-border bg-surface p-6">
              <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-accent/10 text-accent">
                <Footprints className="h-5 w-5" />
              </span>
              <h3 className="mt-5 text-lg font-semibold">Recording tips</h3>
              <ul className="mt-4 space-y-3 text-sm text-muted">
                <li>• Patient walks parallel to the camera (side-on view).</li>
                <li>• Capture at least 4 full gait cycles (~10 seconds).</li>
                <li>• Plain background helps detection; minimal occlusion.</li>
                <li>• Phone camera is fine — landscape orientation, 30 fps or higher.</li>
                <li>• Patient wears form-fitting clothing where possible.</li>
              </ul>
            </aside>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
