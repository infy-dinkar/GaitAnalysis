"use client";
// PostureLiveCapture — 4-view still-image capture flow.
//
// Wizard: front → back → left_side → right_side. Front is required
// (endpoint contract); the other three are optional. Each step
// shows the AssessmentCameraShell (video + skeleton overlay), a
// stance instruction, a "Capture" button, a preview thumb, and
// Retake / Skip controls.
//
// IMPORTANT — MIRROR TRAP (do NOT "fix" this):
//   AssessmentCameraShell applies CSS `-scale-x-100` to the <video>
//   element, so the OPERATOR SEES a mirrored selfie preview. CSS
//   transforms do NOT affect `ctx.drawImage(video, ...)` — the
//   canvas receives the RAW, UNMIRRORED sensor frame. That raw
//   frame is exactly what the posture analysis needs (EXIF-agnostic
//   left/right, camera-native geometry).
//   Therefore: NEVER apply scale(-1,1) / ctx.translate + ctx.scale
//   / .style.transform when drawing the capture canvas. If someone
//   thinks the saved frame "looks flipped" — they're comparing it
//   to the mirrored PREVIEW, not to reality. The analysis is correct.
//
// No MediaRecorder anywhere. Just canvas.drawImage → toBlob → File
// → same /api/analyze-posture endpoint via analyzePostureMultiView.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Camera as CameraIcon,
  CheckCircle2,
  Loader2,
  Play,
  RotateCcw,
  SkipForward,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { AssessmentCameraShell } from "@/components/orthopedic/AssessmentCameraShell";
import { PostureReport } from "@/components/posture/PostureReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  analyzePostureMultiView,
  isPostureViewError,
  type PostureBackResult,
  type PostureExplicitSideResult,
  type PostureMultiViewResult,
  type PostureViewError,
} from "@/lib/posture/analyzer";
import type { KeypointDTO } from "@/lib/reports";
import {
  buildFrontFindings,
  buildSideFindings,
} from "@/lib/posture/measurements";

// ── Wizard step configuration ────────────────────────────────
type StepKey = "front" | "back" | "left_side" | "right_side";

interface StepDef {
  key: StepKey;
  title: string;
  stance: string;
  required: boolean;
  fileFieldName:
    | "frontFile"
    | "backFile"
    | "leftSideFile"
    | "rightSideFile";
}

const STEPS: StepDef[] = [
  {
    key: "front",
    title: "Front view",
    stance:
      "Patient stands facing the camera, feet shoulder-width apart, arms relaxed at sides. Full body in frame.",
    required: true,
    fileFieldName: "frontFile",
  },
  {
    key: "back",
    title: "Back view",
    stance:
      "Patient turns around, back to the camera. Same relaxed stance. Full body in frame.",
    required: false,
    fileFieldName: "backFile",
  },
  {
    key: "left_side",
    title: "Left-side view",
    stance:
      "Patient turns 90° so the LEFT side faces the camera. Arms relaxed. Full body in frame.",
    required: false,
    fileFieldName: "leftSideFile",
  },
  {
    key: "right_side",
    title: "Right-side view",
    stance:
      "Patient turns 90° again so the RIGHT side faces the camera. Arms relaxed. Full body in frame.",
    required: false,
    fileFieldName: "rightSideFile",
  },
];

// Endpoint's boxed "side" field maps to whichever of left_side /
// right_side the operator captured first. If both are captured, we
// use the LEFT clip for the boxed `side` field so the existing
// front+side pipeline still fires; the explicit left_side/right_side
// keys carry the same data with pickedSide forced.
async function captureFrameFromVideo(
  video: HTMLVideoElement,
  fileName: string,
): Promise<{
  file: File;
  persisted: { dataUrl: string; width: number; height: number };
  captureWidth: number;
  captureHeight: number;
} | null> {
  // *** MIRROR TRAP — see file header. Draw the RAW frame; do not
  // *** apply scale(-1,1) or any transform. CSS -scale-x-100 on the
  // *** video only affects display, not ctx.drawImage.
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return null;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, w, h);
  // Full-resolution JPEG File for the upload analysis endpoint.
  const blob: Blob | null = await new Promise((resolve) => {
    canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92);
  });
  if (!blob) return null;
  const file = new File([blob], fileName, { type: "image/jpeg" });

  // Compressed 800-px-wide data URL for save-to-report persistence.
  // Same 800 / 0.8 encoding PostureCapture uses so the SavedPostureReport
  // overlay renders identically for upload-mode and live-mode saves.
  const persisted = await compressToPersistedDataUrl(canvas, 800, 0.8);
  return { file, persisted, captureWidth: w, captureHeight: h };
}

async function compressToPersistedDataUrl(
  sourceCanvas: HTMLCanvasElement,
  maxWidth: number,
  quality: number,
): Promise<{ dataUrl: string; width: number; height: number }> {
  const scale = Math.min(1, maxWidth / sourceCanvas.width);
  const w = Math.round(sourceCanvas.width * scale);
  const h = Math.round(sourceCanvas.height * scale);
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.drawImage(sourceCanvas, 0, 0, w, h);
  const dataUrl = out.toDataURL("image/jpeg", quality);
  return { dataUrl, width: w, height: h };
}

interface CapturedView {
  file: File;
  thumbUrl: string;
  // Compressed source photo persisted into the saved-report
  // payload — matches the shape PostureCapture stores.
  persisted: { dataUrl: string; width: number; height: number };
  // Raw capture dims so we can scale analyzed keypoints into the
  // compressed image's coordinate space before saving.
  captureWidth: number;
  captureHeight: number;
}

type Phase = "capturing" | "analyzing" | "done" | "error";

export function PostureLiveCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [stepIndex, setStepIndex] = useState<number>(0);
  const [captures, setCaptures] = useState<
    Partial<Record<StepKey, CapturedView>>
  >({});
  const [phase, setPhase] = useState<Phase>("capturing");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [result, setResult] = useState<PostureMultiViewResult | null>(null);

  const videoElRef = useRef<HTMLVideoElement | null>(null);

  // AssessmentCameraShell owns the <video> element. Capture a
  // reference by finding it in the DOM the same way the orthopedic
  // captures do for MediaRecorder — but here we just need the
  // element to drawImage from, not the stream.
  function grabVideoEl(): HTMLVideoElement | null {
    const vid = document.querySelector(
      "video[playsinline]",
    ) as HTMLVideoElement | null;
    videoElRef.current = vid;
    return vid;
  }

  // Revoke object URLs when re-captured / unmounted.
  useEffect(() => {
    return () => {
      Object.values(captures).forEach((c) => {
        if (c?.thumbUrl) URL.revokeObjectURL(c.thumbUrl);
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const currentStep = STEPS[stepIndex];
  const currentCapture = currentStep ? captures[currentStep.key] : null;

  const canAnalyse = Boolean(captures.front);

  const handleFrame = useCallback(
    (_kp: Keypoint[], _video: HTMLVideoElement) => {
      // No live coaching — capture is manual via the Capture button.
    },
    [],
  );

  async function onCapture() {
    setError(null);
    const step = STEPS[stepIndex];
    if (!step) return;
    const video = grabVideoEl();
    if (!video) {
      setError("Camera not ready — click Start camera on the preview above.");
      return;
    }
    setBusy(true);
    try {
      const grabbed = await captureFrameFromVideo(
        video,
        `posture_${step.key}.jpg`,
      );
      if (!grabbed) {
        setError("Could not grab a frame. Try again once the video is playing.");
        return;
      }
      const thumbUrl = URL.createObjectURL(grabbed.file);
      // Revoke the previous thumb if we're overwriting.
      const prev = captures[step.key];
      if (prev) URL.revokeObjectURL(prev.thumbUrl);
      setCaptures((c) => ({
        ...c,
        [step.key]: {
          file: grabbed.file,
          thumbUrl,
          persisted: grabbed.persisted,
          captureWidth: grabbed.captureWidth,
          captureHeight: grabbed.captureHeight,
        },
      }));
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Frame capture failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  function onRetake() {
    setError(null);
    const step = STEPS[stepIndex];
    if (!step) return;
    const prev = captures[step.key];
    if (prev) URL.revokeObjectURL(prev.thumbUrl);
    setCaptures((c) => {
      const next = { ...c };
      delete next[step.key];
      return next;
    });
  }

  function onNext() {
    if (stepIndex < STEPS.length - 1) {
      setStepIndex(stepIndex + 1);
    }
  }
  function onPrev() {
    if (stepIndex > 0) setStepIndex(stepIndex - 1);
  }
  function onSkip() {
    if (!currentStep) return;
    if (currentStep.required) return;
    if (stepIndex < STEPS.length - 1) setStepIndex(stepIndex + 1);
  }

  async function onAnalyse() {
    if (!captures.front) {
      setError("Front-view capture is required to start analysis.");
      return;
    }
    // The endpoint requires BOTH front + side. If the operator didn't
    // capture a "side" via the explicit L/R steps, we fall back to
    // reusing the left_side (if present) or right_side (if present)
    // as the boxed `side` field. If neither exists we refuse.
    const sideCandidate = captures.left_side || captures.right_side;
    if (!sideCandidate) {
      setError(
        "At least one side view (left or right) is required — the "
        + "endpoint's boxed 'side' field falls back to whichever was "
        + "captured first.",
      );
      return;
    }
    setPhase("analyzing");
    setError(null);
    try {
      const r = await analyzePostureMultiView({
        frontFile: captures.front.file,
        sideFile: sideCandidate.file,
        backFile: captures.back?.file,
        leftSideFile: captures.left_side?.file,
        rightSideFile: captures.right_side?.file,
      });
      setResult(r);
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed.");
      setPhase("error");
    }
  }

  function onReset() {
    Object.values(captures).forEach((c) => {
      if (c?.thumbUrl) URL.revokeObjectURL(c.thumbUrl);
    });
    setCaptures({});
    setStepIndex(0);
    setPhase("capturing");
    setError(null);
    setResult(null);
  }

  // ── Render: done ──────────────────────────────────────────
  if (phase === "done" && result) {
    return (
      <DoneView
        result={result}
        captures={captures}
        onReset={onReset}
        patientName={patient?.name ?? null}
      />
    );
  }

  // ── Render: capturing / analyzing / error ─────────────────
  return (
    <div className="space-y-6">
      {isDoctorFlow && (
        <SaveStatusBanner patient={patient} saveStatus={null} />
      )}

      <div>
        <p className="eyebrow">Live capture · 4-view wizard</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">
          Step {stepIndex + 1} of {STEPS.length}: {currentStep?.title}
        </h2>
        <p className="mt-2 max-w-2xl text-sm text-muted">
          {currentStep?.stance}
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex flex-wrap gap-2">
        {STEPS.map((s, i) => {
          const captured = Boolean(captures[s.key]);
          const isCurrent = i === stepIndex;
          return (
            <button
              key={s.key}
              type="button"
              onClick={() => setStepIndex(i)}
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ring-1 transition ${
                isCurrent
                  ? "bg-accent/15 text-accent ring-accent/40"
                  : captured
                    ? "bg-emerald-500/15 text-emerald-700 ring-emerald-500/40 dark:text-emerald-300"
                    : "bg-surface text-muted ring-border"
              }`}
            >
              {captured && <CheckCircle2 className="h-3 w-3" />}
              {s.title}
              {s.required && !captured && <span className="text-error">*</span>}
            </button>
          );
        })}
      </div>

      <div className="grid gap-6 md:grid-cols-[2fr_1fr]">
        <div>
          <AssessmentCameraShell onFrame={handleFrame} hideControls={false} />
        </div>
        <div className="space-y-4">
          <div className="rounded-card border border-border bg-surface p-4">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-subtle">
              Captured preview
            </p>
            {currentCapture ? (
              <div className="mt-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={currentCapture.thumbUrl}
                  alt={`${currentStep?.title} capture`}
                  className="block w-full rounded-md border border-border"
                />
                <p className="mt-2 text-[11px] text-muted">
                  Saved frame is <strong>unmirrored</strong> (raw sensor
                  frame). Mirror trap: the live preview shows a
                  selfie-mirror; the analysis uses the raw frame.
                </p>
              </div>
            ) : (
              <p className="mt-2 text-xs text-muted">
                No frame captured yet.
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              {currentCapture ? (
                <Button variant="secondary" onClick={onRetake} disabled={busy}>
                  <RotateCcw className="h-4 w-4" />
                  Retake
                </Button>
              ) : (
                <Button onClick={onCapture} disabled={busy}>
                  {busy ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <CameraIcon className="h-4 w-4" />
                  )}
                  Capture
                </Button>
              )}
              {currentStep && !currentStep.required && !currentCapture && (
                <Button variant="ghost" onClick={onSkip} disabled={busy}>
                  <SkipForward className="h-4 w-4" />
                  Skip
                </Button>
              )}
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              onClick={onPrev}
              disabled={stepIndex === 0 || busy}
            >
              ← Back
            </Button>
            {stepIndex < STEPS.length - 1 ? (
              <Button onClick={onNext} disabled={busy}>
                Next
                <ArrowRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={onAnalyse}
                disabled={!canAnalyse || phase === "analyzing"}
              >
                {phase === "analyzing" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                Analyse
              </Button>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
          {error}
        </div>
      )}
    </div>
  );
}

// ─── Done view ─────────────────────────────────────────────
function DoneView({
  result,
  captures,
  onReset,
  patientName,
}: {
  result: PostureMultiViewResult;
  captures: Partial<Record<StepKey, CapturedView>>;
  onReset: () => void;
  patientName: string | null;
}) {
  const { isDoctorFlow, patient } = usePatientContext();

  // Build the save payload — mirrors the PostureCapture (upload-mode)
  // shape 1:1 so the dispatch page and SavedPostureReport render both
  // flows identically. Two-step scale-then-persist:
  //   • Analyzer runs on the FULL-RESOLUTION frame — keypoints come
  //     back in that coordinate space.
  //   • We persist a compressed ~800-px-wide JPEG data URL alongside
  //     each view, plus a KEYPOINT ARRAY SCALED into that compressed
  //     image's coordinate space so the SavedPostureReport overlay
  //     draws dots and lines at the right positions.
  function buildPayload() {
    const frontFindings = result.front.front
      ? buildFrontFindings(result.front.front)
      : [];
    const sideFindings = result.side.side
      ? buildSideFindings(result.side.side)
      : [];

    type ScaledKp = { x: number; y: number; score?: number; name?: string };
    const scaleKp = (
      kps: unknown,
      fromW: number | undefined,
      toW: number | undefined,
    ): KeypointDTO[] | null => {
      if (!Array.isArray(kps) || !fromW || !toW || fromW === 0) {
        return (kps as KeypointDTO[]) ?? null;
      }
      const s = toW / fromW;
      return (kps as ScaledKp[]).map((kp) => ({
        ...kp,
        x: kp.x * s,
        y: kp.y * s,
      })) as KeypointDTO[];
    };

    // Front + side: `capture` widths come from the live capture at
    // captureWidth, but the analyzer's returned imageWidth is what
    // the BACKEND actually decoded (should match; use analyzer's
    // number to be safe).
    const fCap = captures.front;
    const sCap = captures.left_side || captures.right_side;
    const bCap = captures.back;
    const lCap = captures.left_side;
    const rCap = captures.right_side;

    const frontKpScaled = scaleKp(
      result.front.keypoints,
      result.front.imageWidth,
      fCap?.persisted.width,
    );
    const sideView = result.left_side && !isPostureViewError(result.left_side)
      ? result.left_side
      : result.right_side && !isPostureViewError(result.right_side)
        ? result.right_side
        : null;
    const sideKpScaled = scaleKp(
      result.side.keypoints,
      result.side.imageWidth,
      sCap?.persisted.width,
    );

    // Extract per-view data ONLY when the view succeeded — errors
    // shouldn't corrupt the saved metrics.
    const backSuccess =
      result.back && !isPostureViewError(result.back) ? result.back : null;
    const leftSideSuccess =
      result.left_side && !isPostureViewError(result.left_side)
        ? result.left_side
        : null;
    const rightSideSuccess =
      result.right_side && !isPostureViewError(result.right_side)
        ? result.right_side
        : null;

    const backKpScaled = backSuccess
      ? scaleKp(
          backSuccess.keypoints,
          backSuccess.imageWidth,
          bCap?.persisted.width,
        )
      : null;
    const leftSideKpScaled = leftSideSuccess
      ? scaleKp(
          leftSideSuccess.keypoints,
          leftSideSuccess.imageWidth,
          lCap?.persisted.width,
        )
      : null;
    const rightSideKpScaled = rightSideSuccess
      ? scaleKp(
          rightSideSuccess.keypoints,
          rightSideSuccess.imageWidth,
          rCap?.persisted.width,
        )
      : null;

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const _sideView = sideView; // touched so lints don't nag on shape audit

    const keypoints: Record<string, KeypointDTO[] | null> = {
      front: frontKpScaled,
      side: sideKpScaled,
    };
    if (backKpScaled) keypoints.back = backKpScaled;
    if (leftSideKpScaled) keypoints.left_side = leftSideKpScaled;
    if (rightSideKpScaled) keypoints.right_side = rightSideKpScaled;

    const asImg = (
      cap: CapturedView | undefined,
    ): { data_url: string; width: number; height: number } | null =>
      cap
        ? {
            data_url: cap.persisted.dataUrl,
            width: cap.persisted.width,
            height: cap.persisted.height,
          }
        : null;

    return {
      module: "posture" as const,
      metrics: {
        front: result.front.front,
        side: result.side.side,
        // ── Existing PostureCapture image keys — used by the
        // dispatch page's PostureBody at reports/[id]/page.tsx.
        front_image: asImg(fCap),
        side_image: asImg(sCap),
        // ── Additive new-view metric blobs
        back: pickBack(result.back),
        left_side: pickSide(result.left_side),
        right_side: pickSide(result.right_side),
        // ── Additive new-view images
        back_image: asImg(bCap),
        left_side_image: asImg(lCap),
        right_side_image: asImg(rCap),
        // ── Back-view extras (SavedPostureReport reads these as
        // `backNotAssessed` + `backLrSwapApplied` props).
        back_not_assessed: backSuccess?.not_assessed ?? null,
        back_lr_swap_applied: backSuccess?.lr_swap_applied ?? null,
      } as Record<string, unknown>,
      observations: {
        // snake_case to match the dispatch page's PostureBody
        // (o.front_findings / o.side_findings). CamelCase versions
        // would silently disappear on save.
        front_findings: frontFindings,
        side_findings: sideFindings,
        back_findings: pickFindings(result.back),
        left_side_findings: pickFindings(result.left_side),
        right_side_findings: pickFindings(result.right_side),
      } as Record<string, unknown>,
      keypoints,
    };
  }

  return (
    <div className="space-y-8">
      <PostureReport
        front={result.front}
        side={result.side}
        patient={patient ?? null}
        patientName={patientName}
        back={result.back ?? null}
        leftSide={result.left_side ?? null}
        rightSide={result.right_side ?? null}
      />

      {/* Auto-save — fires as soon as the report renders (doctor
          flow only; AutoSaveToast no-ops otherwise), big banner
          with a 10s Undo. */}
      {isDoctorFlow && <AutoSaveToast buildPayload={buildPayload} />}

      <div className="no-pdf flex flex-wrap items-center justify-between gap-3 border-t border-border pt-6">
        <Button variant="secondary" onClick={onReset}>
          <RotateCcw className="h-4 w-4" />
          New session
        </Button>
      </div>
    </div>
  );
}

type MaybeBack = PostureBackResult | PostureViewError | undefined;
type MaybeSide = PostureExplicitSideResult | PostureViewError | undefined;

function pickBack(v: MaybeBack) {
  if (!v || isPostureViewError(v)) return null;
  return v.back ?? null;
}
function pickSide(v: MaybeSide) {
  if (!v || isPostureViewError(v)) return null;
  return v.side ?? null;
}
function pickFindings(v: MaybeBack | MaybeSide) {
  if (!v || isPostureViewError(v)) return [];
  return v.findings ?? [];
}
function pickKeypoints(v: MaybeBack | MaybeSide): KeypointDTO[] | null {
  if (!v || isPostureViewError(v)) return null;
  return v.keypoints as unknown as KeypointDTO[];
}
