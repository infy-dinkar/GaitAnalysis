"use client";
import { useCallback, useState } from "react";
import { Play, RotateCcw, AlertCircle } from "lucide-react";
import { VideoUpload } from "@/components/analysis/VideoUpload";
import { Button } from "@/components/ui/Button";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import {
  analyzeNeck,
  analyzeShoulder,
  type BiomechDataDTO,
} from "@/lib/api";
import { loadPatient } from "@/components/biomech/PatientForm";

interface Props {
  bodyPart: "shoulder" | "neck";
  movementId: string;
  movementLabel: string;
  /** Bare movement name for the report, e.g. "Lateral Flexion". */
  movementName?: string;
  description: string;
  target: [number, number];
  side?: "left" | "right";
}

export function ApiUploadAssessment({
  bodyPart,
  movementId,
  movementLabel,
  movementName,
  description,
  target,
  side,
}: Props) {
  const reportName =
    movementName ?? movementLabel.split(" · ").pop() ?? movementLabel;
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);     // 0..1 upload progress
  const [phase, setPhase] = useState<"idle" | "uploading" | "analysing" | "done" | "error">("idle");
  const [result, setResult] = useState<BiomechDataDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onSelect = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    setProgress(0);
    setPhase("idle");
  }, []);

  async function run() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setPhase("uploading");

    const patient = loadPatient();
    const patientName = patient?.name?.trim() || null;

    const onProgress = (loaded: number, total: number) => {
      setProgress(total > 0 ? loaded / total : 0);
      if (loaded >= total) setPhase("analysing");
    };

    try {
      const res =
        bodyPart === "shoulder"
          ? await analyzeShoulder(
              {
                video: file,
                movement: movementId,
                side: side ?? "right",
                patientName,
              },
              onProgress,
            )
          : await analyzeNeck(
              { video: file, movement: movementId, patientName },
              onProgress,
            );

      if (!res.success || !res.data) {
        setPhase("error");
        setError(res.error || "Analysis failed");
        return;
      }
      setResult(res.data);
      setPhase("done");
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setPhase("idle");
  }

  // ── DONE: full Assessment Report ────────────────────────────
  if (result && phase === "done") {
    return (
      <div className="space-y-8">
        <AssessmentReport
          bodyPart={bodyPart}
          movementName={reportName}
          movementId={movementId}
          measured={result.peak_magnitude}
          target={[result.reference_range[0], result.reference_range[1]]}
          side={side}
        />

        <div className="flex flex-wrap items-center justify-center gap-3 border-t border-border pt-6 text-xs text-muted">
          <span>
            Valid frames:{" "}
            <span className="tabular text-foreground">
              {result.valid_frames}/{result.total_frames}
            </span>
          </span>
          <span>·</span>
          <span>
            FPS: <span className="tabular text-foreground">{result.fps.toFixed(0)}</span>
          </span>
        </div>

        <div className="flex justify-center">
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
            Analyse another video
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Movement</p>
        <h2 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">
          {movementLabel}
        </h2>
        <p className="mt-2 max-w-xl text-sm text-muted">{description}</p>
        <p className="mt-2 text-xs text-subtle">
          Reference range{" "}
          <span className="tabular text-foreground">
            {target[0]}°–{target[1]}°
          </span>
          {side && (
            <>
              {" · "}side <span className="tabular text-foreground">{side}</span>
            </>
          )}
        </p>
      </div>

      <VideoUpload onSelect={onSelect} />

      {file && phase === "idle" && (
        <Button onClick={run} disabled={busy} loading={busy}>
          <Play className="h-4 w-4" />
          Analyse on server
        </Button>
      )}

      {(phase === "uploading" || phase === "analysing") && (
        <div className="rounded-card border border-border bg-surface p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-[0.12em] text-subtle">
              {phase === "uploading" ? "Uploading video" : "Analysing on server"}
            </span>
            <span className="tabular text-sm text-foreground">
              {phase === "uploading"
                ? `${Math.round(progress * 100)}%`
                : "running pose model…"}
            </span>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
            <div
              className={`h-full bg-accent transition-all ${
                phase === "analysing" ? "animate-pulse" : "duration-200"
              }`}
              style={{
                width:
                  phase === "uploading" ? `${progress * 100}%` : "100%",
              }}
            />
          </div>
        </div>
      )}

      {phase === "error" && error && (
        <div className="rounded-card border border-error/40 bg-error/5 p-5">
          <div className="flex items-start gap-3">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-error" />
            <div className="text-sm">
              <p className="font-medium text-foreground">Analysis failed</p>
              <p className="mt-1 text-muted">{error}</p>
              <Button variant="secondary" size="sm" className="mt-4" onClick={reset}>
                <RotateCcw className="h-4 w-4" />
                Try another video
              </Button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
