"use client";
// Two-slot posture capture flow: front-view photo + side-view photo.
// Each photo is analyzed independently in the browser using the same
// MoveNet detector the live + biomech features use. After both views
// are picked the user clicks "Run analysis" and gets a combined report.

import { useCallback, useState } from "react";
import { Image as ImageIcon, Play, RotateCcw, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import {
  analyzePostureImage,
  type PostureAnalysisResult,
} from "@/lib/posture/analyzer";
import { PostureReport } from "@/components/posture/PostureReport";

export function PostureCapture() {
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<"idle" | "analysing" | "done" | "error">("idle");
  const [front, setFront] = useState<PostureAnalysisResult | null>(null);
  const [side, setSide] = useState<PostureAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onPick = useCallback(
    (which: "front" | "side", f: File | null) => {
      setError(null);
      if (which === "front") setFrontFile(f);
      else setSideFile(f);
    },
    [],
  );

  async function run() {
    if (!frontFile || !sideFile) {
      setError("Both a front-view and a side-view photo are required.");
      return;
    }
    setBusy(true);
    setError(null);
    setPhase("analysing");
    setFront(null);
    setSide(null);
    try {
      const frontResult = await analyzePostureImage(frontFile, "front");
      const sideResult = await analyzePostureImage(sideFile, "side");
      setFront(frontResult);
      setSide(sideResult);
      setPhase("done");
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    if (front) URL.revokeObjectURL(front.imageUrl);
    if (side) URL.revokeObjectURL(side.imageUrl);
    setFrontFile(null);
    setSideFile(null);
    setFront(null);
    setSide(null);
    setError(null);
    setPhase("idle");
  }

  if (phase === "done" && (front || side)) {
    return (
      <div className="space-y-8">
        <PostureReport front={front} side={side} />
        <div className="flex justify-center border-t border-border pt-6">
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
            Analyse another set
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="grid gap-5 md:grid-cols-2">
        <PhotoSlot
          label="Front view"
          file={frontFile}
          onPick={(f) => onPick("front", f)}
          hint="Patient stands facing the camera, full body in frame, arms relaxed at sides."
        />
        <PhotoSlot
          label="Side view"
          file={sideFile}
          onPick={(f) => onPick("side", f)}
          hint="Patient stands sideways to the camera (left or right side), full body in frame."
        />
      </div>

      {phase === "analysing" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-xs uppercase tracking-[0.12em] text-subtle">
            Analysing photos in browser…
          </p>
          <p className="mt-3 text-[11px] text-subtle">
            Your photos never leave this device — pose detection runs locally.
          </p>
        </div>
      )}

      {phase === "error" && error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-5 text-sm">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-error" />
          <div>
            <p className="font-medium text-foreground">Analysis failed</p>
            <p className="mt-1 text-muted">{error}</p>
          </div>
        </div>
      )}

      <div className="flex flex-col gap-2 md:flex-row md:items-center">
        <Button
          onClick={run}
          disabled={busy || !frontFile || !sideFile}
          loading={busy}
          className="w-full md:w-auto"
        >
          <Play className="h-4 w-4" />
          Run analysis
        </Button>
        {(!frontFile || !sideFile) && (
          <p className="text-xs text-subtle">
            Both front-view and side-view photos are required to run analysis.
          </p>
        )}
      </div>
    </div>
  );
}

function PhotoSlot({
  label,
  file,
  onPick,
  hint,
}: {
  label: string;
  file: File | null;
  onPick: (f: File | null) => void;
  hint: string;
}) {
  return (
    <label className="flex cursor-pointer flex-col gap-3 rounded-card border border-dashed border-border bg-surface p-5 transition hover:border-accent/60">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {label} <span className="text-error">*</span>
      </p>
      {file ? (
        <div className="flex items-center gap-3 rounded-md bg-elevated p-3 text-sm">
          <ImageIcon className="h-5 w-5 text-accent" />
          <span className="truncate font-medium text-foreground">{file.name}</span>
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              onPick(null);
            }}
            className="ml-auto text-xs text-muted hover:text-error"
          >
            remove
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center gap-2 py-6 text-center">
          <ImageIcon className="h-8 w-8 text-muted" />
          <p className="text-sm text-foreground">Click to upload a photo</p>
          <p className="text-[11px] text-subtle">{hint}</p>
        </div>
      )}
      <input
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => onPick(e.target.files?.[0] ?? null)}
      />
    </label>
  );
}
