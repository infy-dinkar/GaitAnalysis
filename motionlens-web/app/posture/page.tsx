"use client";
import { useState } from "react";
import Link from "next/link";
import { Upload, Video } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { PostureCapture } from "@/components/posture/PostureCapture";
import { PostureLiveCapture } from "@/components/posture/PostureLiveCapture";

type Mode = "upload" | "live";

export default function PosturePage() {
  // Upload mode preserved as the DEFAULT for backward-compat.
  const [mode, setMode] = useState<Mode>("upload");
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div className="max-w-2xl">
              <Badge>Posture screening</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Static posture analysis<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Upload photos or capture live — front, back, and both
                sides. We extract joint landmarks, measure tilts and
                shifts in the frontal and sagittal planes, and generate
                an annotated report.
              </p>
            </div>
            <Link href="/">
              <Button variant="ghost" size="sm">← Home</Button>
            </Link>
          </div>

          <div className="mt-8 inline-flex rounded-card border border-border bg-surface p-1">
            <button
              type="button"
              onClick={() => setMode("upload")}
              className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
                mode === "upload"
                  ? "bg-accent text-white shadow-sm"
                  : "text-muted hover:text-foreground"
              }`}
            >
              <Upload className="h-4 w-4" /> Upload photos
            </button>
            <button
              type="button"
              onClick={() => setMode("live")}
              className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
                mode === "live"
                  ? "bg-accent text-white shadow-sm"
                  : "text-muted hover:text-foreground"
              }`}
            >
              <Video className="h-4 w-4" /> Live capture
            </button>
          </div>

          <div className="mt-6">
            {mode === "upload" ? <PostureCapture /> : <PostureLiveCapture />}
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Photo guidelines</p>
            <ul className="mt-3 space-y-1.5 list-disc pl-5">
              <li>Patient stands relaxed with arms hanging naturally at sides.</li>
              <li>Plain background, even lighting, no shadows across the body.</li>
              <li>Clothing snug enough to see joint landmarks (shorts and a fitted top work well).</li>
              <li>Camera at hip height, at least 6 feet (2 metres) away, full body in frame.</li>
              <li>Front: face the camera squarely. Side: turn 90° so the camera sees one full side. Back: turn all the way around.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
