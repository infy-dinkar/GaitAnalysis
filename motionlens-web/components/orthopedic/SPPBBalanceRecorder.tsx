"use client";
// SPPB Component 1 (Balance) — recording-only capture component.
//
// Unlike the live-pose path used by the standalone C4 4-Stage Balance
// test, this component does NOT run MoveNet in the browser. It only:
//   1. shows a camera preview,
//   2. records via MediaRecorder for as long as the operator wants,
//   3. uploads the resulting Blob to /api/sppb/balance for backend
//      MediaPipe analysis,
//   4. surfaces the per-stage result to the caller.
//
// Why no live pose detection: SPPB's foot-precise stage geometry needs
// MediaPipe's heel + foot_index landmarks. Doing that in-browser would
// require a different model than the rest of the app already uses. The
// backend path keeps the model selection consistent with TUG + gait +
// ankle (all already on MediaPipe BlazePose Full server-side).
//
// Compatible with the existing SPPB scoring pipeline: returns the
// same {1?: StageResult, 2?: StageResult, 3?: StageResult} shape the
// orchestrator's `buildBalanceComponent()` already consumes.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  Camera,
  CameraOff,
  FileVideo,
  Loader2,
  Play,
  RotateCcw,
  Square,
  Upload,
  Video,
} from "lucide-react";

import { Button } from "@/components/ui/Button";
import {
  analyzeSPPBBalance,
  normaliseBalanceStages,
  type SPPBBalanceDiagnostics,
} from "@/lib/orthopedic/sppbBalance";
import type { StageResult } from "@/lib/orthopedic/fourStageBalance";

type Phase = "idle" | "recording" | "uploading" | "done" | "error";
type Mode = "record" | "upload";

const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = [
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/x-matroska",
];

interface Props {
  /** Called once with the per-stage result map AND backend
   *  diagnostics when the analysis is finished. SPPBCapture feeds
   *  the stages into buildBalanceComponent() and keeps the
   *  diagnostics around to surface in the result panel when stage
   *  detection didn't go well. */
  onComplete: (
    stages: { 1?: StageResult; 2?: StageResult; 3?: StageResult },
    diagnostics: SPPBBalanceDiagnostics | null,
  ) => void;
}

function pickRecorderMime(): string {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const m of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) {
      return m;
    }
  }
  return "video/webm";
}

export function SPPBBalanceRecorder({ onComplete }: Props) {
  const [mode, setMode] = useState<Mode>("record");
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [recordingMs, setRecordingMs] = useState(0);

  // Upload-mode state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const uploadVideoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const tickRef = useRef<number | null>(null);

  // ── Camera lifecycle ────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          // "environment" = rear camera. The standard clinical setup
          // for SPPB Balance is the device propped up with the rear
          // camera pointing at the patient; the operator stands beside
          // the device. Front camera was returning very low
          // resolutions on some devices (body height < 100 px in the
          // captured frame, which broke pose normalization).
          facingMode: { ideal: "environment" },
          // `min` constraints force a usable resolution. If the
          // device can't satisfy them getUserMedia rejects and the
          // user sees an actionable error rather than a silent
          // analysis failure.
          width: { min: 640, ideal: 1280 },
          height: { min: 480, ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setCameraActive(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Camera access denied");
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // Tear down the camera when the operator flips to upload mode —
  // the preview is irrelevant there and releasing the device kills
  // the camera-active light immediately.
  useEffect(() => {
    if (mode === "upload" && cameraActive) {
      stopCamera();
    }
  }, [mode, cameraActive, stopCamera]);

  // Clean up the blob URL we generated for the upload preview.
  useEffect(() => {
    return () => {
      if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    };
  }, [uploadPreviewUrl]);

  // ── Upload mode handlers ────────────────────────────────────
  function handleFilePick(file: File | null) {
    setError(null);
    if (uploadPreviewUrl) {
      URL.revokeObjectURL(uploadPreviewUrl);
      setUploadPreviewUrl(null);
    }
    if (!file) {
      setUploadFile(null);
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(
        `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_MB} MB.`,
      );
      setUploadFile(null);
      return;
    }
    if (file.type && !ACCEPTED_VIDEO_TYPES.includes(file.type)) {
      setError(
        `Unsupported file type (${file.type}). Use MP4, WebM, MOV, or MKV.`,
      );
      setUploadFile(null);
      return;
    }
    setUploadFile(file);
    setUploadPreviewUrl(URL.createObjectURL(file));
  }

  async function analyzeUpload() {
    if (!uploadFile) return;
    setPhase("uploading");
    // Uploaded files have proper container metadata — no client
    // recording duration needed.
    await uploadAndAnalyze(uploadFile, null);
  }

  // ── Recording control ───────────────────────────────────────
  function startRecording() {
    const stream = streamRef.current;
    if (!stream) {
      setError("Start the camera first.");
      return;
    }
    chunksRef.current = [];
    const mime = pickRecorderMime();
    const rec = new MediaRecorder(stream, { mimeType: mime });
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    rec.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: mime });
      const durationMs = Date.now() - startedAtRef.current;
      await uploadAndAnalyze(blob, durationMs);
    };
    // No timeslice — produces a single clean WebM blob with a proper
    // header at stop(), same pattern used by TUGCapture / AnkleCapture.
    rec.start();
    recorderRef.current = rec;
    startedAtRef.current = Date.now();
    setRecordingMs(0);
    setPhase("recording");
    tickRef.current = window.setInterval(() => {
      setRecordingMs(Date.now() - startedAtRef.current);
    }, 100);
  }

  function stopRecording() {
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
    setPhase("uploading");
  }

  async function uploadAndAnalyze(blob: Blob, recordingDurationMs: number | null) {
    setError(null);
    try {
      const res = await analyzeSPPBBalance(blob, recordingDurationMs);
      if (!res.success || !res.data) {
        setError(res.error ?? "Analysis failed");
        setPhase("error");
        return;
      }
      onComplete(
        normaliseBalanceStages(res.data.stages),
        res.data.diagnostics ?? null,
      );
      // After a successful upload the parent will transition the
      // orchestrator to the next phase; release the camera now so
      // the user doesn't see a stale preview.
      stopCamera();
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setPhase("error");
    }
  }

  function reset() {
    setError(null);
    setRecordingMs(0);
    setPhase("idle");
    if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    setUploadFile(null);
    setUploadPreviewUrl(null);
  }

  // ── Render ──────────────────────────────────────────────────
  const fmtTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const cs = Math.floor((ms % 1000) / 10);
    return `${s.toString().padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
  };

  const busy = phase === "recording" || phase === "uploading";

  return (
    <div className="space-y-5">
      {/* Framing requirement — surface this BEFORE the mode toggle
          because feet-not-visible is the #1 cause of analysis
          failures, and the error only becomes visible after the
          upload completes. */}
      <div className="rounded-card border border-amber-500/40 bg-amber-500/5 p-4">
        <p className="text-sm font-semibold text-foreground">
          ⚠ Camera must capture the patient&apos;s FULL BODY — head to toes
        </p>
        <p className="mt-1 text-xs text-muted">
          The backend tracks the patient&apos;s heel and toe positions to
          detect each stage. Place the camera far enough back that the
          patient&apos;s feet stay in the frame for the entire test.
          A face-only or chest-only recording will be rejected.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="inline-flex rounded-card border border-border bg-surface p-1">
        <button
          type="button"
          onClick={() => setMode("record")}
          disabled={busy}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "record"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${busy ? "opacity-50" : ""}`}
        >
          <Camera className="h-4 w-4" />
          Record live
        </button>
        <button
          type="button"
          onClick={() => setMode("upload")}
          disabled={busy}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "upload"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${busy ? "opacity-50" : ""}`}
        >
          <Upload className="h-4 w-4" />
          Upload video
        </button>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[3fr_2fr]">
        {/* Preview pane — camera in record mode, video preview in upload mode */}
        <div className="relative aspect-video overflow-hidden rounded-card border border-border bg-gradient-to-br from-[#0A0A0B] via-[#0d0d10] to-[#15151a]">
          {mode === "record" ? (
            <>
              <video
                ref={videoRef}
                playsInline
                muted
                className="block h-full w-full object-cover"
              />
              {!cameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <Camera className="mb-3 h-10 w-10 text-white/40" />
                  <p className="text-sm text-white/60">Camera is off</p>
                </div>
              )}
              {phase === "recording" && (
                <div className="absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-full bg-black/65 px-3 py-1 text-xs font-medium text-white">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                  REC · {fmtTime(recordingMs)}s
                </div>
              )}
            </>
          ) : (
            <>
              {uploadPreviewUrl ? (
                <video
                  ref={uploadVideoRef}
                  src={uploadPreviewUrl}
                  controls
                  playsInline
                  className="block h-full w-full object-contain"
                />
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <FileVideo className="mb-3 h-10 w-10 text-white/40" />
                  <p className="text-sm text-white/60">No video selected</p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Stage instructions — visible throughout the recording so
            the operator can prompt the patient between stages. */}
        <div className="rounded-card border border-border bg-surface p-4 text-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            All three stages — perform in sequence
          </p>
          <ol className="mt-3 space-y-3">
            <li>
              <p className="font-medium text-foreground">
                Stage 1 — Side-by-side · 10 s
              </p>
              <p className="text-xs text-muted">
                Both feet next to each other, toes and heels in line. Stand still.
              </p>
            </li>
            <li>
              <p className="font-medium text-foreground">
                Stage 2 — Semi-tandem · 10 s
              </p>
              <p className="text-xs text-muted">
                One foot half-step forward — heel of the moved foot beside the
                big toe of the other foot.
              </p>
            </li>
            <li>
              <p className="font-medium text-foreground">
                Stage 3 — Tandem (heel-to-toe) · 10 s
              </p>
              <p className="text-xs text-muted">
                One foot directly in front of the other in a single line. Heel
                of front foot touches toes of back foot.
              </p>
            </li>
          </ol>
          <p className="mt-3 text-[11px] text-subtle">
            The backend detects each stage from the recorded video. Stop the
            recording after the patient finishes Stage 3 (or at the first
            failure — SPPB stops at the first stage that fails).
          </p>
        </div>
      </div>

      {/* RECORD MODE — recording controls */}
      {mode === "record" && phase === "idle" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Recording controls
          </p>
          {!cameraActive ? (
            <div className="mt-3 space-y-3">
              <p className="text-sm text-muted">
                Step 1 — start the camera and frame the patient&apos;s full
                body in front view (head to toes visible).
              </p>
              <Button onClick={startCamera}>
                <Camera className="h-4 w-4" />
                Start camera
              </Button>
            </div>
          ) : (
            <div className="mt-3 space-y-3">
              <p className="text-sm text-muted">
                Patient performs all three stages in sequence — 10 s each.
                Start the recording on operator&apos;s &quot;Go&quot;.
              </p>
              <div className="flex gap-2">
                <Button onClick={startRecording}>
                  <Play className="h-4 w-4" />
                  Start recording
                </Button>
                <Button variant="secondary" onClick={stopCamera}>
                  <CameraOff className="h-4 w-4" />
                  Stop camera
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* UPLOAD MODE — file picker + analyze button */}
      {mode === "upload" && phase === "idle" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Upload video
          </p>

          {!uploadFile && (
            <label className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-card border border-dashed border-border bg-elevated p-8 text-center transition hover:border-accent/60">
              <FileVideo className="h-10 w-10 text-muted" />
              <p className="text-sm font-medium text-foreground">
                Click to choose a video file
              </p>
              <p className="text-xs text-muted">
                MP4, WebM, MOV, or MKV · max {MAX_FILE_MB} MB · all three
                stages performed in sequence
              </p>
              <input
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
                className="hidden"
                onChange={(e) => handleFilePick(e.target.files?.[0] ?? null)}
              />
            </label>
          )}

          {uploadFile && (
            <div className="mt-3 space-y-3">
              <div className="flex items-center gap-3 rounded-md bg-elevated p-3 text-sm">
                <Video className="h-5 w-5 shrink-0 text-accent" />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-foreground">
                    {uploadFile.name}
                  </p>
                  <p className="text-xs text-muted">
                    {(uploadFile.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => handleFilePick(null)}
                  className="text-xs text-muted hover:text-error"
                >
                  remove
                </button>
              </div>
              <div className="flex gap-2">
                <Button onClick={analyzeUpload}>
                  <Upload className="h-4 w-4" />
                  Analyse video
                </Button>
                <Button variant="secondary" onClick={() => handleFilePick(null)}>
                  Choose different file
                </Button>
              </div>
              <p className="text-[11px] text-subtle">
                Video should show the patient performing all 3 balance stages
                in front view, from setup through Stage 3 finish (or first
                failure).
              </p>
            </div>
          )}
        </div>
      )}

      {phase === "recording" && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <div className="flex items-center justify-between">
            <p className="inline-flex items-center gap-2 text-sm font-medium text-error">
              <span className="h-2 w-2 animate-pulse rounded-full bg-error" />
              Recording
            </p>
            <p className="tabular text-3xl font-semibold text-accent">
              {fmtTime(recordingMs)}s
            </p>
          </div>
          <p className="mt-2 text-xs text-muted">
            Coach the patient through Stage 1 → 2 → 3. Stop as soon as Stage
            3 ends (or the patient loses balance). The backend handles stage
            detection automatically.
          </p>
          <div className="mt-3">
            <Button onClick={stopRecording}>
              <Square className="h-4 w-4" />
              Stop recording
            </Button>
          </div>
          {recordingMs / 1000 > 60 && (
            <p className="mt-2 text-xs text-warning">
              Recording has exceeded 60 seconds — backend will reject clips
              longer than 60 s. Stop and try again.
            </p>
          )}
        </div>
      )}

      {phase === "uploading" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-accent" />
            <p className="text-sm text-foreground">
              Uploading and analyzing balance test — this usually takes 20-40
              seconds.
            </p>
          </div>
        </div>
      )}

      {phase === "error" && (
        <div className="rounded-card border border-border bg-surface p-5">
          <div className="flex items-start gap-3 rounded-md border border-error/40 bg-error/5 p-3 text-sm">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
            <p className="text-foreground">{error ?? "Analysis failed"}</p>
          </div>
          <div className="mt-3">
            <Button variant="secondary" onClick={reset}>
              <RotateCcw className="h-4 w-4" />
              Try again
            </Button>
          </div>
        </div>
      )}

      {error && phase !== "error" && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}
    </div>
  );
}
