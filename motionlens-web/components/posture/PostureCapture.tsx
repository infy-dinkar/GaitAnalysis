"use client";
// Two-slot posture capture flow: front-view photo + side-view photo.
// Each photo is analyzed independently in the browser using the same
// MoveNet detector the live + biomech features use. After both views
// are picked the user clicks "Run analysis" and gets a combined report.

import { useCallback, useEffect, useRef, useState } from "react";
import { Image as ImageIcon, Play, RotateCcw, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import {
  analyzePostureImage,
  type PostureAnalysisResult,
} from "@/lib/posture/analyzer";
import { PostureReport } from "@/components/posture/PostureReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  buildFrontFindings,
  buildSideFindings,
} from "@/lib/posture/measurements";

// Compressed image data URL paired with its decoded pixel dimensions.
// Stored alongside per-view metrics so the saved report can re-render
// the annotated overlay without keeping the source file on disk.
interface PersistedView {
  dataUrl: string;
  width: number;
  height: number;
}

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
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<"idle" | "analysing" | "done" | "error">("idle");
  const [front, setFront] = useState<PostureAnalysisResult | null>(null);
  const [side, setSide] = useState<PostureAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Compressed image data URLs cached for the save payload — same
  // image the analyzer ran on, but downscaled + JPEG-encoded so it's
  // small enough to serialise into MongoDB.
  const persistedRef = useRef<{ front: PersistedView | null; side: PersistedView | null }>({
    front: null,
    side: null,
  });

  // Doctor-flow context — save is explicit via SaveToPatientButton.
  const { isDoctorFlow, patient } = usePatientContext();

  useEffect(() => {
    // Reset cached payload images whenever a file is removed / swapped.
    if (!frontFile) persistedRef.current.front = null;
    if (!sideFile) persistedRef.current.side = null;
  }, [frontFile, sideFile]);

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
      // Compress the source images in parallel with the analysis
      // results so the save payload has them ready by the time the
      // doctor clicks Save.
      const [frontPersist, sidePersist] = await Promise.all([
        compressFileToDataUrl(frontFile),
        compressFileToDataUrl(sideFile),
      ]);
      persistedRef.current.front = frontPersist;
      persistedRef.current.side = sidePersist;
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
        <PostureReport
          front={front}
          side={side}
          patient={patient ?? null}
          patientName={patient?.name ?? null}
        />

        {/* Explicit save button — only renders in doctor flow */}
        <SaveToPatientButton
          buildPayload={() => {
            const frontFindings = front?.front ? buildFrontFindings(front.front) : [];
            const sideFindings = side?.side ? buildSideFindings(side.side) : [];
            const fImg = persistedRef.current.front;
            const sImg = persistedRef.current.side;

            // Keypoints come out of the analyzer in ORIGINAL-image
            // pixel space (e.g. a 4032×3024 phone photo's coordinates).
            // The compressed source we persist alongside is ~800 px
            // wide, so we scale the keypoints into the compressed
            // image's coordinate space before saving — otherwise the
            // SavedPostureReport overlay draws dots / lines / badges
            // at the wrong positions when re-rendering.
            const scaleKp = (
              kps: typeof front extends infer T ? T extends { keypoints: infer K } ? K : null : null,
              fromW: number | undefined,
              toW: number | undefined,
            ) => {
              if (!kps || !fromW || !toW || fromW === 0) return kps;
              const s = toW / fromW;
              return (kps as Array<{ x: number; y: number; score?: number; name?: string }>).map((kp) => ({
                ...kp,
                x: kp.x * s,
                y: kp.y * s,
              }));
            };
            const frontKpScaled = scaleKp(front?.keypoints ?? null, front?.imageWidth, fImg?.width);
            const sideKpScaled  = scaleKp(side?.keypoints  ?? null, side?.imageWidth,  sImg?.width);

            return {
              module: "posture",
              metrics: {
                front: front?.front ?? {},
                side: side?.side ?? {},
                // Compressed source images so the saved-report viewer
                // can re-render the annotated overlay. JPEG-encoded at
                // ~800 px max width.
                front_image: fImg
                  ? {
                      data_url: fImg.dataUrl,
                      width: fImg.width,
                      height: fImg.height,
                    }
                  : null,
                side_image: sImg
                  ? {
                      data_url: sImg.dataUrl,
                      width: sImg.width,
                      height: sImg.height,
                    }
                  : null,
              },
              observations: {
                front_findings: frontFindings,
                side_findings: sideFindings,
              },
              // Spec Section 2 (a): persist the raw landmark stream as
              // JSON, but in the COMPRESSED image's coordinate space
              // so it aligns with the saved photo at render time.
              keypoints: {
                front: frontKpScaled,
                side: sideKpScaled,
              },
            };
          }}
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
