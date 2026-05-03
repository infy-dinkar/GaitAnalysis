"use client";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { PostureCapture } from "@/components/posture/PostureCapture";

export default function PosturePage() {
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
                Upload a front-view and a side-view photo of the patient. We&apos;ll
                extract joint landmarks, measure tilts and shifts in the frontal
                and sagittal planes, and generate an annotated report — all in
                your browser.
              </p>
            </div>
            <Link href="/">
              <Button variant="ghost" size="sm">← Home</Button>
            </Link>
          </div>

          <div className="mt-10">
            <PostureCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Photo guidelines</p>
            <ul className="mt-3 space-y-1.5 list-disc pl-5">
              <li>Patient stands relaxed with arms hanging naturally at sides.</li>
              <li>Plain background, even lighting, no shadows across the body.</li>
              <li>Clothing snug enough to see joint landmarks (shorts and a fitted top work well).</li>
              <li>Camera at hip height, at least 6 feet (2 metres) away, full body in frame.</li>
              <li>Front: face the camera squarely. Side: turn 90° so the camera sees one full side.</li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
