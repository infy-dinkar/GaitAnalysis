"use client";
import { useCallback, useState } from "react";
import { Play, RotateCcw, AlertCircle } from "lucide-react";
import { VideoUpload } from "@/components/analysis/VideoUpload";
import { Button } from "@/components/ui/Button";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { BiomechDataDTO } from "@/lib/api";
import { analyzeBiomechVideo } from "@/lib/biomech/uploadAnalyze";

interface Props {
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle";
  movementId: string;
  movementLabel: string;
  /** Bare movement name for the report, e.g. "Lateral Flexion". */
  movementName?: string;
  description: string;
  target: [number, number];
  side?: "left" | "right";
}

/** Returns true when the (bodyPart, movementId) pair posts its
 *  upload to a FastAPI /api/analyze-… endpoint rather than running
 *  locally in the browser. Keep in sync with the dispatch table at
 *  the top of analyzeBiomechVideo in lib/biomech/uploadAnalyze.ts —
 *  this drives the loading-stage label ("Uploading video… →
 *  Analysing movement…") and the privacy note ("your video is not
 *  uploaded") in the UI. */
function isBackendRouted(bodyPart: Props["bodyPart"], movementId: string): boolean {
  if (bodyPart === "ankle") return true;
  if (bodyPart === "shoulder") {
    return (
      movementId === "flexion_extension" ||
      movementId === "abduction_adduction" ||
      movementId === "rotation"
    );
  }
  if (bodyPart === "knee") {
    return movementId === "flexion_extension";
  }
  if (bodyPart === "neck") {
    // Both merged neck tests route to /api/analyze-neck; rotation
    // still uses the browser path.
    return (
      movementId === "flexion_extension" ||
      movementId === "lateral_flexion"
    );
  }
  return false;
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
  const [progress, setProgress] = useState(0);     // 0..1 analysis progress
  const [phase, setPhase] = useState<"idle" | "analysing" | "done" | "error">("idle");
  const [result, setResult] = useState<BiomechDataDTO | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Doctor-flow context — analysis result will be saved on explicit
  // user click via the SaveToPatientButton in the report view.
  const { isDoctorFlow, patient } = usePatientContext();

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
    setPhase("analysing");

    try {
      // Fully client-side: video never leaves the browser. Same MoveNet
      // detector and same TS angle math the live mode uses.
      const data = await analyzeBiomechVideo({
        file,
        bodyPart,
        movement: movementId,
        side,
        onProgress: (frac) => setProgress(frac),
      });
      setResult(data);
      setPhase("done");
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : "Analysis failed");
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
    // Merged movements (shoulder rotation / abduction_adduction)
    // return two peaks per recording. The DTO carries label + target
    // for the secondary direction whenever the test is merged — even
    // if the patient didn't actually perform that direction, the
    // analyser sets primary_label / secondary_label / secondary_*
    // metadata so the report can render two rows ("Not detected" in
    // the missing one) rather than silently collapsing to a single
    // row with the combined merged label.
    const isMerged =
      !!result.primary_label &&
      !!result.secondary_label &&
      !!result.secondary_reference_range;
    // A peak of 0 is a legitimate reading (knee extension perfectly
    // straightened, for example). Treat "captured" as type-based,
    // not value-based — the analyser omits the field entirely when
    // the direction wasn't recorded.
    const hasSecondaryValue =
      isMerged && typeof result.secondary_peak_magnitude === "number";

    return (
      <div className="space-y-8">
        <AssessmentReport
          bodyPart={bodyPart}
          movementName={result.primary_label ?? reportName}
          movementId={movementId}
          measured={result.peak_magnitude}
          target={[result.reference_range[0], result.reference_range[1]]}
          side={side}
          keyFrames={result.key_frames}
          secondaryMovementName={isMerged ? result.secondary_label : undefined}
          secondaryMeasured={
            hasSecondaryValue ? result.secondary_peak_magnitude : undefined
          }
          secondaryTarget={isMerged ? result.secondary_reference_range : undefined}
        />

        {/* Explicit "Save to patient history" — only shows in doctor flow */}
        <SaveToPatientButton
          buildPayload={() => ({
            module: "biomech",
            body_part: bodyPart,
            movement: movementId,
            side,
            metrics: {
              peak_angle: result.peak_angle,
              peak_magnitude: result.peak_magnitude,
              reference_range: result.reference_range,
              target: result.target,
              percentage: result.percentage,
              status: result.status,
              valid_frames: result.valid_frames,
              total_frames: result.total_frames,
              fps: result.fps,
              // For merged tests, persist the secondary direction
              // (label + range always; measured value may be missing
              // when the patient didn't perform that direction) so
              // saved reports can render both rows when re-opened.
              ...(isMerged
                ? {
                    secondary_peak_angle: result.secondary_peak_angle,
                    secondary_peak_magnitude: result.secondary_peak_magnitude,
                    secondary_reference_range: result.secondary_reference_range,
                    primary_label: result.primary_label,
                    secondary_label: result.secondary_label,
                  }
                : {}),
              // Persist annotated key-frame screenshots so the
              // saved-report viewer can show the same thumbnail
              // strip the live result page shows.
              ...(result.key_frames && result.key_frames.length > 0
                ? { key_frames: result.key_frames }
                : {}),
            },
            observations: { interpretation: result.interpretation },
            video_filename: file?.name,
            video_size_bytes: file?.size,
          })}
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
      {isDoctorFlow && phase !== "done" && (
        <SaveStatusBanner patient={patient} saveStatus={null} />
      )}

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
          Analyse video
        </Button>
      )}

      {phase === "analysing" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-[0.12em] text-subtle">
              {(() => {
                // Backend-routed tests run on the FastAPI BlazePose
                // pipeline (video uploads to /api/analyze-…); other
                // tests run locally in the browser via MoveNet. The
                // stage label + the "not uploaded" privacy note
                // both branch on this. Keep this list in sync with
                // analyzeBiomechVideo's dispatch in
                // lib/biomech/uploadAnalyze.ts.
                if (!isBackendRouted(bodyPart, movementId)) {
                  return "Analysing in browser";
                }
                if (progress < 0.25) return "Uploading video…";
                if (progress < 0.95) return "Analysing movement…";
                return "Done";
              })()}
            </span>
            <span className="tabular text-sm text-foreground">
              {`${Math.round(progress * 100)}%`}
            </span>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
            <div
              className="h-full bg-accent transition-all duration-200"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
          {!isBackendRouted(bodyPart, movementId) && (
            <p className="mt-3 text-[11px] text-subtle">
              Pose detection runs locally on your device — your video is not uploaded.
            </p>
          )}
        </div>
      )}

      {phase === "error" && error && (
        <div className="rounded-card border border-error/40 bg-error/5 p-5">
          <div className="flex items-start gap-3">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-error" />
            <div className="text-sm">
              <p className="font-medium text-foreground">Analysis failed</p>
              <p className="mt-1 text-muted">{error}</p>
              <div className="mt-4 flex flex-wrap gap-2">
                {/* For transient failures (network / 500-ish), let the
                    operator retry the SAME video without re-picking
                    the file. Otherwise show the file-picker reset. */}
                {file && /check connection|Analysis failed\.?$/.test(error) && (
                  <Button size="sm" onClick={run} disabled={busy} loading={busy}>
                    <Play className="h-4 w-4" />
                    Retry
                  </Button>
                )}
                <Button variant="secondary" size="sm" onClick={reset}>
                  <RotateCcw className="h-4 w-4" />
                  Try another video
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
