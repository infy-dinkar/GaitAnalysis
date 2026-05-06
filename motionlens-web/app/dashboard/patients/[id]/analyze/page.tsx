"use client";
// /dashboard/patients/[id]/analyze — pick assessment module

import Link from "next/link";
import { use as usePromise } from "react";
import {
  ArrowUpRight,
  Footprints,
  Activity,
  PersonStanding,
  StretchHorizontal,
} from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";

const MODULES = [
  {
    href: "gait/upload",
    eyebrow: "Walking video",
    title: "Gait analysis",
    body: "Upload a side-view walking clip. We extract cadence, stride symmetry, joint kinematics, and cycle-locked curves.",
    icon: Footprints,
    tone: "from-cyan-500/15 to-cyan-500/5",
    iconTone: "text-cyan-600",
  },
  {
    href: "biomech",
    eyebrow: "Range of motion",
    title: "Biomechanics",
    body: "Assess shoulder, neck, knee, hip, or ankle ROM. Live camera or video upload — choose any movement.",
    icon: Activity,
    tone: "from-amber-500/15 to-amber-500/5",
    iconTone: "text-amber-600",
  },
  {
    href: "posture",
    eyebrow: "Static posture",
    title: "Posture screening",
    body: "Upload front and side photos. Get tilts, plumb-line shifts, and an annotated report.",
    icon: PersonStanding,
    tone: "from-emerald-500/15 to-emerald-500/5",
    iconTone: "text-emerald-600",
  },
  {
    href: "orthopedic/trendelenburg",
    eyebrow: "Orthopedic test",
    title: "Trendelenburg",
    body: "Live single-leg-stance test. 30-second hold per side, automatic pelvic-tilt + trunk-lean tracking, side-by-side report.",
    icon: StretchHorizontal,
    tone: "from-violet-500/15 to-violet-500/5",
    iconTone: "text-violet-600",
  },
];

export default function AnalyzePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = usePromise(params);
  return (
    <AuthGuard>
      <DashboardShell
        backHref={`/dashboard/patients/${id}`}
        backLabel="Patient"
      >
        <Content patientId={id} />
      </DashboardShell>
    </AuthGuard>
  );
}

function Content({ patientId }: { patientId: string }) {
  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Choose assessment</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
          What do you want to measure?
        </h1>
        <p className="mt-2 max-w-xl text-sm text-muted">
          Pick a module. The result will be saved automatically against this patient&apos;s record.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-3">
        {MODULES.map((m) => {
          const Icon = m.icon;
          // patientId is forwarded as a query param so the existing analysis
          // pages can detect "doctor flow" and auto-save the report when done.
          const href = `/${m.href}?patientId=${patientId}`;
          return (
            <Link
              key={m.href}
              href={href}
              className={`group relative flex flex-col overflow-hidden rounded-hero border border-border bg-gradient-to-br ${m.tone} p-6 transition hover:border-accent hover:shadow-glow-sm md:p-8`}
            >
              <div className="flex items-center justify-between">
                <Icon className={`h-7 w-7 ${m.iconTone}`} />
                <ArrowUpRight className="h-5 w-5 text-muted transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-accent" />
              </div>

              <div className="mt-8">
                <p className="eyebrow">{m.eyebrow}</p>
                <h2 className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">
                  {m.title}
                </h2>
                <p className="mt-3 text-sm leading-relaxed text-muted">{m.body}</p>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
