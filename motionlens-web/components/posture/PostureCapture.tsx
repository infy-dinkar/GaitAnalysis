"use client";
// Four-slot posture upload flow: front + side (both REQUIRED) plus
// back + left-side + right-side (all OPTIONAL). All uploaded photos
// are analysed in ONE POST to /api/analyze-posture via the
// analyzePostureMultiView client — the endpoint is 4-view-aware.
//
// Backwards-compatible: leave the 3 optional slots empty and the
// endpoint / response / save payload behave EXACTLY like the old
// 2-view flow. Only when the operator picks the extra photos do the
// new views populate.

import { useCallback, useEffect, useRef, useState } from "react";
import { Image as ImageIcon, Play, RotateCcw, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import {
  analyzePostureMultiView,
  isPostureViewError,
  type PostureAnalysisResult,
  type PostureBackResult,
  type PostureExplicitSideResult,
  type PostureMultiViewResult,
  type PostureViewError,
} from "@/lib/posture/analyzer";
import { PostureReport } from "@/components/posture/PostureReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  buildFrontFindings,
  buildSideFindings,
} from "@/lib/posture/measurements";
import type { KeypointDTO } from "@/lib/reports";

// Compressed image data URL paired with its decoded pixel dimensions.
// Stored alongside per-view metrics so the saved report can re-render
// the annotated overlay without keeping the source file on disk.
interface PersistedView {
  dataUrl: string;
  width: number;
  height: number;
}

// Slot identity — matches the FormData field names + save-payload
// keys the backend/dispatch page already understands.
type SlotKey = "front" | "side" | "back" | "left_side" | "right_side";

// Resize-and-encode a source image File to a JPEG data URL at most
// `maxWidth` pixels wide. Keeps the saved-report payload manageable
// (a 4000-px phone photo would otherwise serialise to ~3-5 MB of
// base64 in Mongo); 800 px / 0.8 quality lands at ~50-150 KB per view.
async function compressFileToDataUrl(
  file: File,
  maxWidth = 800,
  quality = 0.8,
): Promise<PersistedView> {
  const imgUrl = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("Failed to load image for compression"));
      el.src = imgUrl;
    });
    const scale = Math.min(1, maxWidth / img.naturalWidth);
    const w = Math.round(img.naturalWidth * scale);
    const h = Math.round(img.naturalHeight * scale);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    ctx.drawImage(img, 0, 0, w, h);
    const dataUrl = canvas.toDataURL("image/jpeg", quality);
    return { dataUrl, width: w, height: h };
  } finally {
    URL.revokeObjectURL(imgUrl);
  }
}

export function PostureCapture() {
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [backFile, setBackFile] = useState<File | null>(null);
  const [leftSideFile, setLeftSideFile] = useState<File | null>(null);
  const [rightSideFile, setRightSideFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<"idle" | "analysing" | "done" | "error">("idle");
  const [result, setResult] = useState<PostureMultiViewResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Compressed image data URLs cached for the save payload — one entry
  // per slot. Only the ones the operator actually uploaded get set.
  const persistedRef = useRef<Record<SlotKey, PersistedView | null>>({
    front: null,
    side: null,
    back: null,
    left_side: null,
    right_side: null,
  });

  const { isDoctorFlow, patient } = usePatientContext();

  useEffect(() => {
    // Reset cached payload images whenever a file is removed / swapped.
    if (!frontFile) persistedRef.current.front = null;
    if (!sideFile) persistedRef.current.side = null;
    if (!backFile) persistedRef.current.back = null;
    if (!leftSideFile) persistedRef.current.left_side = null;
    if (!rightSideFile) persistedRef.current.right_side = null;
  }, [frontFile, sideFile, backFile, leftSideFile, rightSideFile]);

  const onPick = useCallback((which: SlotKey, f: File | null) => {
    setError(null);
    switch (which) {
      case "front":      setFrontFile(f); break;
      case "side":       setSideFile(f); break;
      case "back":       setBackFile(f); break;
      case "left_side":  setLeftSideFile(f); break;
      case "right_side": setRightSideFile(f); break;
    }
  }, []);

  async function run() {
    if (!frontFile || !sideFile) {
      setError("Both a front-view and a side-view photo are required.");
      return;
    }
    setBusy(true);
    setError(null);
    setPhase("analysing");
    setResult(null);
    try {
      // Single POST to /api/analyze-posture. The multi-view client
      // sends whichever of the 3 optional file fields are present;
      // the response shape stays back-compat when only front+side
      // are uploaded (see analyzer.ts contract).
      const compressFront = compressFileToDataUrl(frontFile);
      const compressSide  = compressFileToDataUrl(sideFile);
      const compressBack  = backFile ? compressFileToDataUrl(backFile) : Promise.resolve(null);
      const compressLeft  = leftSideFile ? compressFileToDataUrl(leftSideFile) : Promise.resolve(null);
      const compressRight = rightSideFile ? compressFileToDataUrl(rightSideFile) : Promise.resolve(null);
      const [
        multiResult, frontPersist, sidePersist,
        backPersist, leftPersist, rightPersist,
      ] = await Promise.all([
        analyzePostureMultiView({
          frontFile,
          sideFile,
          backFile: backFile ?? undefined,
          leftSideFile: leftSideFile ?? undefined,
          rightSideFile: rightSideFile ?? undefined,
        }),
        compressFront,
        compressSide,
        compressBack,
        compressLeft,
        compressRight,
      ]);
      persistedRef.current.front = frontPersist;
      persistedRef.current.side = sidePersist;
      persistedRef.current.back = backPersist;
      persistedRef.current.left_side = leftPersist;
      persistedRef.current.right_side = rightPersist;
      setResult(multiResult);
      setPhase("done");
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    if (result) {
      URL.revokeObjectURL(result.front.imageUrl);
      URL.revokeObjectURL(result.side.imageUrl);
      if (result.back && !isPostureViewError(result.back)) {
        URL.revokeObjectURL(result.back.imageUrl);
      }
      if (result.left_side && !isPostureViewError(result.left_side)) {
        URL.revokeObjectURL(result.left_side.imageUrl);
      }
      if (result.right_side && !isPostureViewError(result.right_side)) {
        URL.revokeObjectURL(result.right_side.imageUrl);
      }
    }
    setFrontFile(null);
    setSideFile(null);
    setBackFile(null);
    setLeftSideFile(null);
    setRightSideFile(null);
    setResult(null);
    setError(null);
    setPhase("idle");
  }

  if (phase === "done" && result) {
    return (
      <div className="space-y-8">
        <PostureReport
          front={result.front}
          side={result.side}
          patient={patient ?? null}
          patientName={patient?.name ?? null}
          back={result.back ?? null}
          leftSide={result.left_side ?? null}
          rightSide={result.right_side ?? null}
        />

        {/* Explicit save button — only renders in doctor flow */}
        <SaveToPatientButton
          buildPayload={() => buildSavePayload(result, persistedRef.current)}
        />

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
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <div className="grid gap-5 md:grid-cols-2">
        <PhotoSlot
          label="Front view"
          required
          file={frontFile}
          onPick={(f) => onPick("front", f)}
          hint="Patient stands facing the camera, full body in frame, arms relaxed at sides."
        />
        <PhotoSlot
          label="Side view"
          required
          file={sideFile}
          onPick={(f) => onPick("side", f)}
          hint="Patient stands sideways to the camera (left or right side), full body in frame."
        />
        <PhotoSlot
          label="Back view"
          file={backFile}
          onPick={(f) => onPick("back", f)}
          hint="Optional. Patient turns around — back to camera, full body in frame."
        />
        <PhotoSlot
          label="Left-side view"
          file={leftSideFile}
          onPick={(f) => onPick("left_side", f)}
          hint="Optional. Patient turns so LEFT side faces the camera. Analysed with pickedSide forced left."
        />
        <PhotoSlot
          label="Right-side view"
          file={rightSideFile}
          onPick={(f) => onPick("right_side", f)}
          hint="Optional. Patient turns so RIGHT side faces the camera. Analysed with pickedSide forced right."
        />
      </div>

      {phase === "analysing" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-xs uppercase tracking-[0.12em] text-subtle">
            Uploading and analysing photos…
          </p>
          <p className="mt-3 text-[11px] text-subtle">
            Pose detection runs on the server. Photos are processed in memory
            and deleted immediately after analysis — only metrics and
            keypoints are saved with the patient&apos;s report.
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
            Back / left / right views are optional.
          </p>
        )}
      </div>
    </div>
  );
}

// ─── Save payload builder ────────────────────────────────────────
// Mirrors buildPayload in PostureLiveCapture 1:1 so both flows land
// the same shape in Mongo and the dispatch page renders both
// identically. Keypoints are scaled from the analyzer's returned
// coordinate space (matches the source photo's dimensions) into the
// compressed image's coordinate space so the saved-report overlay
// draws dots at the right positions.
function buildSavePayload(
  result: PostureMultiViewResult,
  persisted: Record<SlotKey, PersistedView | null>,
) {
  const frontFindings = result.front.front
    ? buildFrontFindings(result.front.front)
    : [];
  const sideFindings = result.side.side
    ? buildSideFindings(result.side.side)
    : [];

  const scaleKp = (
    kps: unknown,
    fromW: number | undefined,
    toW: number | undefined,
  ): KeypointDTO[] | null => {
    if (!Array.isArray(kps) || !fromW || !toW || fromW === 0) {
      return (kps as KeypointDTO[]) ?? null;
    }
    const s = toW / fromW;
    return (kps as Array<{ x: number; y: number; score?: number; name?: string }>).map(
      (kp) => ({ ...kp, x: kp.x * s, y: kp.y * s }),
    ) as KeypointDTO[];
  };

  const backSuccess: PostureBackResult | null =
    result.back && !isPostureViewError(result.back) ? result.back : null;
  const leftSideSuccess: PostureExplicitSideResult | null =
    result.left_side && !isPostureViewError(result.left_side) ? result.left_side : null;
  const rightSideSuccess: PostureExplicitSideResult | null =
    result.right_side && !isPostureViewError(result.right_side) ? result.right_side : null;

  const asImg = (
    p: PersistedView | null,
  ): { data_url: string; width: number; height: number } | null =>
    p ? { data_url: p.dataUrl, width: p.width, height: p.height } : null;

  const frontKp = scaleKp(
    result.front.keypoints, result.front.imageWidth,
    persisted.front?.width,
  );
  const sideKp = scaleKp(
    result.side.keypoints, result.side.imageWidth,
    persisted.side?.width,
  );
  const backKp = backSuccess
    ? scaleKp(backSuccess.keypoints, backSuccess.imageWidth, persisted.back?.width)
    : null;
  const leftKp = leftSideSuccess
    ? scaleKp(leftSideSuccess.keypoints, leftSideSuccess.imageWidth, persisted.left_side?.width)
    : null;
  const rightKp = rightSideSuccess
    ? scaleKp(rightSideSuccess.keypoints, rightSideSuccess.imageWidth, persisted.right_side?.width)
    : null;

  const keypoints: Record<string, KeypointDTO[] | null> = {
    front: frontKp,
    side: sideKp,
  };
  if (backKp) keypoints.back = backKp;
  if (leftKp) keypoints.left_side = leftKp;
  if (rightKp) keypoints.right_side = rightKp;

  return {
    module: "posture" as const,
    metrics: {
      front: result.front.front ?? {},
      side: result.side.side ?? {},
      // Existing image keys — read by the dispatch page PostureBody.
      front_image: asImg(persisted.front),
      side_image: asImg(persisted.side),
      // Additive new-view blobs.
      back: backSuccess?.back ?? null,
      left_side: leftSideSuccess?.side ?? null,
      right_side: rightSideSuccess?.side ?? null,
      back_image: asImg(persisted.back),
      left_side_image: asImg(persisted.left_side),
      right_side_image: asImg(persisted.right_side),
      // Back-view extras — SavedPostureReport surfaces these as the
      // honest not_assessed list + the L/R swap chip.
      back_not_assessed: backSuccess?.not_assessed ?? null,
      back_lr_swap_applied: backSuccess?.lr_swap_applied ?? null,
    } as Record<string, unknown>,
    observations: {
      // snake_case matches PostureBody at reports/[id]/page.tsx.
      front_findings: frontFindings,
      side_findings: sideFindings,
      back_findings: backSuccess?.findings ?? [],
      left_side_findings: leftSideSuccess?.findings ?? [],
      right_side_findings: rightSideSuccess?.findings ?? [],
    } as Record<string, unknown>,
    keypoints,
  };
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
type _EnsureAnalysisResultShapeStillUsed = PostureAnalysisResult;
// Live-render error shape reserved for the same not-currently-rendered
// error banner PostureLiveCapture uses. Kept for symmetry.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
type _EnsureViewErrorShapeStillUsed = PostureViewError;

// ─── Photo slot ────────────────────────────────────────────────
function PhotoSlot({
  label,
  file,
  onPick,
  hint,
  required = false,
}: {
  label: string;
  file: File | null;
  onPick: (f: File | null) => void;
  hint: string;
  required?: boolean;
}) {
  return (
    <label className="flex cursor-pointer flex-col gap-3 rounded-card border border-dashed border-border bg-surface p-5 transition hover:border-accent/60">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {label}{" "}
        {required ? (
          <span className="text-error">*</span>
        ) : (
          <span className="text-subtle">(optional)</span>
        )}
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
