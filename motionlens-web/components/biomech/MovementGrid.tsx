"use client";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export interface MovementOption {
  id: string;
  label: string;
  description: string;
  target: [number, number];
  /** Optional reference illustration shown as a thumbnail at the
   *  top of the tile. One image per exercise (no left/right
   *  variants); the side selector lives in a separate UI section.
   *  Asset path convention: /images/biomech/<joint>/<joint>_<id>.png */
  imageUrl?: string;
}

interface MovementGridProps<T extends MovementOption> {
  options: T[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export function MovementGrid<T extends MovementOption>({
  options,
  selected,
  onSelect,
}: MovementGridProps<T>) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {options.map((opt) => {
        const active = selected === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => onSelect(opt.id)}
            className={cn(
              "group relative flex flex-col items-start gap-2 rounded-card border p-5 text-left transition-all duration-200",
              active
                ? "border-accent bg-accent/5 shadow-glow-sm"
                : "border-border bg-surface hover:border-accent/60",
            )}
          >
            {opt.imageUrl && (
              <div className="mb-1 w-full overflow-hidden rounded-md bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={opt.imageUrl}
                  alt=""
                  aria-hidden="true"
                  loading="lazy"
                  className="block h-28 w-full object-contain"
                />
              </div>
            )}
            <div className="flex w-full items-start justify-between gap-3">
              <span className="text-sm font-semibold text-foreground">{opt.label}</span>
              {active && (
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent text-background">
                  <Check className="h-3 w-3" />
                </span>
              )}
            </div>
            <span className="text-xs leading-relaxed text-muted">{opt.description}</span>
            <span className="mt-1 text-xs uppercase tracking-[0.12em] text-subtle tabular">
              Target {opt.target[0]}°–{opt.target[1]}°
            </span>
          </button>
        );
      })}
    </div>
  );
}
