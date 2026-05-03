"use client";
// Two-button left/right side picker. Used by joint setup pages
// (shoulder, knee, hip, ankle) where the assessment is per-side.

interface Props {
  selected: "left" | "right" | null;
  onSelect: (side: "left" | "right") => void;
}

export function SideSelector({ selected, onSelect }: Props) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {(["left", "right"] as const).map((s) => (
        <button
          key={s}
          type="button"
          onClick={() => onSelect(s)}
          className={`rounded-card border p-5 text-left transition ${
            selected === s
              ? "border-accent bg-accent/5 ring-1 ring-accent/30"
              : "border-border bg-surface hover:border-accent/60"
          }`}
        >
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Side
          </p>
          <p className="mt-1 text-lg font-semibold text-foreground">
            {s === "left" ? "Left" : "Right"}
          </p>
          <p className="mt-1 text-xs text-muted">
            Patient&apos;s {s} side — face the camera so this is the side being assessed.
          </p>
        </button>
      ))}
    </div>
  );
}
