"use client";
// Rehab Start Card — placeholder rendered in the mechanic-shell
// column BEFORE the patient explicitly starts the exercise.
//
// Rationale: several mechanic shells (TargetReachShell,
// HoldInZoneShell, TraceShell, MetronomeShell) begin scoring /
// spawning / playing audio the moment they mount. If a shell mounts
// as soon as the patient picks a side, the score / timer / music
// starts before the patient is even in position and before the
// camera has had a chance to detect them.
//
// Fix: gate the shell mount at the page level with a `started` flag.
// While `!started`, this card renders in the shell's slot with a
// clear "Start exercise" button. Clicking the button flips `started`
// and the real shell mounts on the next render.
//
// Additive: does not modify any mechanic shell or engine. Each
// affected page opts in individually.

import { Play } from "lucide-react";
import { Button } from "@/components/ui/Button";

/**
 * @typedef {object} RehabStartCardProps
 * @property {() => void} onStart
 * @property {string} [title]
 * @property {string} [hint]
 */

/**
 * @param {RehabStartCardProps} props
 */
export function RehabStartCard({
  onStart,
  title = "Ready when you are",
  hint = "Position yourself in front of the camera, then press Start. The exercise timer and score only begin after you press Start.",
}) {
  return (
    <div className="flex h-full min-h-[280px] flex-col items-center justify-center gap-4 rounded-card border border-dashed border-border bg-surface p-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-accent/10 ring-2 ring-accent/30">
        <Play className="h-6 w-6 text-accent" />
      </div>
      <div className="max-w-sm">
        <h3 className="text-lg font-semibold tracking-tight text-foreground">
          {title}
        </h3>
        <p className="mt-1.5 text-sm text-muted">{hint}</p>
      </div>
      <Button onClick={onStart}>
        <Play className="h-4 w-4" />
        Start exercise
      </Button>
    </div>
  );
}
