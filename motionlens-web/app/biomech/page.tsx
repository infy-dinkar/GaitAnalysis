"use client";
import { Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ArrowUpRight } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";

const JOINTS = [
  {
    href: "/biomech/shoulder",
    eyebrow: "Upper limb",
    title: "Shoulder",
    body:
      "Six movements — flexion, extension, abduction, adduction, internal & external rotation. Peak angle, target range, status.",
    gradient:
      "linear-gradient(135deg, rgba(234,88,12,0.18) 0%, rgba(79,195,247,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    href: "/biomech/neck",
    eyebrow: "Cervical spine",
    title: "Neck",
    body:
      "Four movements — flexion, extension, lateral flexion, rotation. Cervical ROM with normative ranges.",
    gradient:
      "linear-gradient(135deg, rgba(255,183,77,0.18) 0%, rgba(234,88,12,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    href: "/biomech/knee",
    eyebrow: "Lower limb",
    title: "Knee",
    body:
      "Two movements — flexion and extension. Hip-knee-ankle angle measurement for ROM tracking and rehab.",
    gradient:
      "linear-gradient(135deg, rgba(79,195,247,0.20) 0%, rgba(56,189,248,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    href: "/biomech/hip",
    eyebrow: "Lower limb",
    title: "Hip",
    body:
      "Four movements — flexion, extension, internal & external rotation. Trunk-thigh and rotational tests.",
    gradient:
      "linear-gradient(135deg, rgba(168,85,247,0.18) 0%, rgba(234,88,12,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
  {
    href: "/biomech/ankle",
    eyebrow: "Lower limb",
    title: "Ankle",
    body:
      "Two movements — dorsiflexion (toes up) and plantarflexion (toes down). Knee-to-wall style screening.",
    gradient:
      "linear-gradient(135deg, rgba(34,197,94,0.18) 0%, rgba(79,195,247,0.10) 50%, rgba(28,28,33,0.0) 100%)",
  },
];

export default function BiomechPage() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <Suspense fallback={null}>
            <BiomechInner />
          </Suspense>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function BiomechInner() {
  const params = useSearchParams();
  const patientId = params.get("patientId");
  // Forward patientId through every joint card so the doctor flow
  // stays connected end-to-end (joint chooser → setup → live/upload).
  const qs = patientId ? `?patientId=${patientId}` : "";

  return (
    <>
      <div className="max-w-2xl">
        <Badge>Biomechanics module</Badge>
        <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-6xl">
          Choose a joint<span className="text-accent">.</span>
        </h1>
        <p className="mt-5 text-lg text-muted">
          Pick the body region you want to assess. Each module guides you through patient
          setup, movement selection, and live or upload-based capture.
        </p>
      </div>

      <div className="mt-16 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {JOINTS.map((j) => (
              <Link
                key={j.href}
                href={`${j.href}${qs}`}
                className="group relative flex aspect-[4/3] flex-col justify-between overflow-hidden rounded-hero border border-border bg-elevated p-6 transition-all duration-300 hover:border-accent hover:shadow-glow-sm md:p-8"
              >
                <div
                  className="pointer-events-none absolute inset-0 opacity-90 transition-opacity duration-500 group-hover:opacity-100"
                  style={{ background: j.gradient }}
                  aria-hidden
                />
                <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" aria-hidden />

                <div className="relative">
                  <Badge>{j.eyebrow}</Badge>
                </div>
                <div className="relative">
                  <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">{j.title}</h2>
                  <p className="mt-4 max-w-md text-sm leading-relaxed text-muted">{j.body}</p>
                  <div className="mt-6 inline-flex items-center gap-1 text-sm text-accent">
                    Start assessment
                    <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
    </>
  );
}
