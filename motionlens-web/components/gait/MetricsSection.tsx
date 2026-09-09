import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { MetricTile, Status } from "@/components/gait/MetricTile";
import { fmt } from "@/lib/utils";
import type { MetricsBlockDTO } from "@/lib/api";

const NORMAL = {
  cadence: [100, 120] as [number, number],
  symmetryPct: [95, 100] as [number, number],
  kneePeak: [25, 45] as [number, number],
  strideCv: [0, 3] as [number, number],
  stepLength: [0.55, 0.8] as [number, number],
  torsoLeanAbs: [0, 5] as [number, number],
  stepTime: [0.45, 0.7] as [number, number],
};

function classify(value: number | null, range: [number, number], invert = false): Status {
  if (value === null || value === undefined) return "neutral";
  if (invert) {
    if (value <= range[1]) return "good";
    if (value <= range[1] * 2) return "fair";
    return "poor";
  }
  if (value >= range[0] && value <= range[1]) return "good";
  if (value >= range[0] * 0.85 && value <= range[1] * 1.15) return "fair";
  return "poor";
}

interface SectionHeaderProps {
  eyebrow: string;
  title: string;
  description: string;
  gradient: string;
}

function SectionHeader({ eyebrow, title, description, gradient }: SectionHeaderProps) {
  return (
    <div className="flex gap-4">
      <div
        className={cn("w-1 shrink-0 rounded-full", gradient)}
        aria-hidden
      />
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2 className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">{title}</h2>
        <p className="mt-1 text-sm text-muted">{description}</p>
      </div>
    </div>
  );
}

interface MetricsSectionProps {
  variant: "total" | "clean";
  metrics: MetricsBlockDTO;
  walkingDirection?: string;
  children?: ReactNode;
}

export function MetricsSection({
  variant,
  metrics,
  walkingDirection,
}: MetricsSectionProps) {
  const isClean = variant === "clean";

  const tiles: { label: string; value: string; hint?: string; status: Status }[] = [
    {
      label: "Step count",
      value: String(metrics.step_count),
      hint: "strikes",
      status: "neutral",
    },
    {
      label: "Cadence",
      value: metrics.cadence !== null ? fmt(metrics.cadence, 0) : "—",
      hint: "steps/min",
      status: classify(metrics.cadence, NORMAL.cadence),
    },
    {
      label: "Symmetry",
      value: metrics.symmetry !== null ? `${fmt(metrics.symmetry * 100, 0)}%` : "—",
      hint: "% L/R rhythm",
      status:
        metrics.symmetry !== null
          ? classify(metrics.symmetry * 100, NORMAL.symmetryPct)
          : "neutral",
    },
    {
      label: "Knee peak",
      value: metrics.knee_peak !== null ? `${fmt(metrics.knee_peak, 1)}°` : "—",
      hint: "swing flexion",
      status: classify(metrics.knee_peak, NORMAL.kneePeak),
    },
    {
      label: "Stride CV",
      value: metrics.stride_cv !== null ? `${fmt(metrics.stride_cv, 1)}%` : "—",
      hint: "lower = better",
      status: classify(metrics.stride_cv, NORMAL.strideCv, true),
    },
    {
      label: "Step length",
      value:
        metrics.step_length !== null
          ? `${fmt(metrics.step_length, 2)} ${metrics.step_length_unit}`
          : "—",
      hint: metrics.step_length_unit === "m" ? "metres" : "pixels (no calibration)",
      status:
        metrics.step_length_unit === "m"
          ? classify(metrics.step_length, NORMAL.stepLength)
          : "neutral",
    },
    {
      label: "Torso lean",
      value:
        metrics.torso_lean !== null ? `${fmt(metrics.torso_lean, 1)}°` : "—",
      hint: "+ fwd / − back",
      status:
        metrics.torso_lean !== null
          ? classify(Math.abs(metrics.torso_lean), NORMAL.torsoLeanAbs, true)
          : "neutral",
    },
    {
      label: "Step time",
      value: metrics.step_time !== null ? `${fmt(metrics.step_time, 2)}s` : "—",
      hint: "avg interval",
      status: classify(metrics.step_time, NORMAL.stepTime),
    },
  ];

  return (
    <section className="space-y-5">
      <SectionHeader
        eyebrow={isClean ? "Clean metrics" : "Total metrics"}
        title={isClean ? "Steady-state only" : "Entire video"}
        description={
          isClean
            ? "Used for all observations & suggestions — turning, accel and decel frames excluded."
            : "Informational — every frame included, including turns / accel / decel."
        }
        gradient={
          isClean
            ? "bg-gradient-to-b from-[#a78bfa] via-accent to-[#22d3ee]"
            : "bg-gradient-to-b from-[#a78bfa] via-[#f472b6] to-[#fb7185]"
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {tiles.map((t) => (
          <MetricTile
            key={t.label}
            label={t.label}
            value={t.value}
            hint={t.hint}
            status={t.status}
          />
        ))}
      </div>

      <div className="text-xs text-muted">
        <p>
          <span className="text-subtle">Window:</span> {metrics.window_description}
        </p>
        {isClean && walkingDirection && (
          <p className="mt-1">
            <span className="text-subtle">Walking direction:</span>{" "}
            <span className="text-foreground">{walkingDirection}</span>
          </p>
        )}
      </div>
    </section>
  );
}
