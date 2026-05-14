"use client";
// Ankle dorsi/plantarflexion capture flow.
//
// Two modes, both hitting POST /api/analyze-ankle (backend MediaPipe
// BlazePose Full — needed because MoveNet's 17-keypoint set lacks
// foot_index, which is required for accurate ankle joint-angle math):
//
//   1. RECORD — getUserMedia + MediaRecorder. The recorded Blob is
//      sent to the same backend endpoint with `recording_duration_ms`
//      as a sidecar so the WebM-header-broken duration problem is
//      handled by tug_engine._ensure_decodable_video.
//
//   2. UPLOAD — operator picks a pre-recorded clip. Same endpoint.
//
// On success the response feeds the existing AssessmentReport
// renderer so the report UI matches every other biomech module.

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
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { BiomechDataDTO } from "@/lib/api";
import { analyzeAnkleBlob } from "@/lib/biomech/uploadAnalyze";

type Phase = "idle" | "recording" | "uploading" | "done" | "error";
type Mode = "record" | "upload";

const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = [
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/x-matroska",
];

function pickRecorderMime(): string {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const m of candidates) {
    if (
      typeof MediaRecorder !== "undefined" &&
      MediaRecorder.isTypeSupported(m)
    ) {
      return m;
    }
  }
  return "video/webm";
}

interface Props {
  movementId: "flexion" | "extension";
  movementLabel: string;        // e.g. "Ankle · Dorsiflexion"
  description: string;
  target: [number, number];
  side: "left" | "right";
  /** Which mode the toggle is initialised in (still user-switchable). */
  initialMode?: Mode;
}

export function AnkleCapture({
  movementId,
  movementLabel,
  description,
  target,
  side,
  initialMode = "record",
}: Props) {
  const { isDoctorFlow, patient } = usePatientContext();
  const reportName = movementLabel.split(" · ").pop() ?? movementLabel;

  const [mode, setMode] = useState<Mode>(initialMode);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);

  // Record-mode state
  const [cameraActive, setCameraActive] = useState(false);
  const [recordingMs, setRecordingMs] = useState(0);

  // Upload-mode state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState<string | null>(null);

  // Result shared by both modes
  const [result, setResult] = useState<BiomechDataDTO | null>(null);

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
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
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

  useEffect(() => {
    if (mode === "upload" && cameraActive) {
      stopCamera();
    }
  }, [mode, cameraActive, stopCamera]);

  useEffect(() => {
    return () => {
      if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    };
  }, [uploadPreviewUrl]);

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
    // No timeslice — single-blob encode produces a clean WebM header.
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
    if (rec && rec.state !== "inactive") {
      rec.stop();
    }
    setPhase("uploading");
  }

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
    await uploadAndAnalyze(uploadFile, null);
  }

  async function uploadAndAnalyze(
    blob: Blob,
    recordingDurationMs: number | null,
  ) {
    setError(null);
    try {
      const data = await analyzeAnkleBlob(
        blob,
        movementId,
        side,
        recordingDurationMs,
      );
      setResult(data);
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
      setPhase("error");
    }
  }

  function reset() {
    setResult(null);
    setError(null);
    setRecordingMs(0);
    setPhase("idle");
    stopCamera();
    if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    setUploadFile(null);
    setUploadPreviewUrl(null);
  }

  // ── Done view ───────────────────────────────────────────────
  if (phase === "done" && result) {
    return (
      <div className="space-y-8">
        <AssessmentReport
          bodyPart="ankle"
          movementName={reportName}
          movementId={movementId}
          measured={result.peak_magnitude}
          target={[result.reference_range[0], result.reference_range[1]]}
          side={side}
          keyFrames={result.key_frames}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "biomech",
            body_part: "ankle",
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
              // Persist key-frame thumbnails so the saved-report
              // viewer can render them later (mirrors how TUG saves
              // its annotated screenshots).
              key_frames: result.key_frames ?? [],
            },
            observations: { interpretation: result.interpretation },
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
            FPS:{" "}
            <span className="tabular text-foreground">
              {result.fps.toFixed(0)}
            </span>
          </span>
        </div>

        <div className="flex justify-center">
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
            Run again
          </Button>
        </div>
      </div>
    );
  }

  // ── Capture view ────────────────────────────────────────────
  const busy = phase === "recording" || phase === "uploading";
  const fmtTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const cs = Math.floor((ms % 1000) / 10);
    return `${s.toString().padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-10">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

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
          {" · "}side{" "}
          <span className="tabular text-foreground">{side}</span>
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

      <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
        {/* LEFT — instructions + controls */}
        <div className="space-y-5">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Setup checklist
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Patient sits comfortably with the test leg fully extended forward.",
                "Camera SIDEWAYS to the test leg — entire shin + foot visible.",
                "Remove socks/shoes so the foot landmarks are clearly visible.",
                movementId === "extension"
                  ? "On 'Start', tell the patient to point the toes DOWN (gas-pedal motion), hold for ~1 s, return to neutral."
                  : "On 'Start', tell the patient to pull the toes UP toward the shin, hold for ~1 s, return to neutral.",
                mode === "record"
                  ? "Stop recording once the patient returns to the neutral position."
                  : "Trim to 3-15 seconds, starting at neutral and ending after the patient returns to neutral.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          {mode === "record" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Recording controls
              </p>

              {phase === "idle" && (
                <div className="mt-3 space-y-3">
                  <p className="text-sm text-muted">
                    Step 1 — start the camera and frame the patient&apos;s
                    shin and foot in profile.
                  </p>
                  {!cameraActive ? (
                    <Button onClick={startCamera}>
                      <Camera className="h-4 w-4" />
                      Start camera
                    </Button>
                  ) : (
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
                  )}
                </div>
              )}

              {phase === "recording" && (
                <div className="mt-3 space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="inline-flex items-center gap-2 text-sm font-medium text-error">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-error" />
                      Recording
                    </p>
                    <p className="tabular text-3xl font-semibold text-accent">
                      {fmtTime(recordingMs)}s
                    </p>
                  </div>
                  <p className="text-xs text-muted">
                    Stop once the patient is back at neutral.
                  </p>
                  <Button onClick={stopRecording}>
                    <Square className="h-4 w-4" />
                    Stop recording
                  </Button>
                  {recordingMs / 1000 > 60 && (
                    <p className="text-xs text-warning">
                      Recording has exceeded 60 seconds — the backend
                      will reject clips longer than 60 s. Stop and try
                      again.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {mode === "upload" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Upload video
              </p>

              {phase === "idle" && !uploadFile && (
                <label className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-card border border-dashed border-border bg-elevated p-8 text-center transition hover:border-accent/60">
                  <FileVideo className="h-10 w-10 text-muted" />
                  <p className="text-sm font-medium text-foreground">
                    Click to choose a video file
                  </p>
                  <p className="text-xs text-muted">
                    MP4, WebM, MOV, or MKV · max {MAX_FILE_MB} MB
                  </p>
                  <input
                    type="file"
                    accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
                    className="hidden"
                    onChange={(e) =>
                      handleFilePick(e.target.files?.[0] ?? null)
                    }
                  />
                </label>
              )}

              {phase === "idle" && uploadFile && (
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
                    <Button
                      variant="secondary"
                      onClick={() => handleFilePick(null)}
                    >
                      Choose different file
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}

          {phase === "uploading" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-accent" />
                <p className="text-sm text-foreground">
                  Uploading and analysing — this can take 10-30 seconds.
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
                  Reset
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT — preview */}
        <div className="lg:sticky lg:top-28">
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
          <p className="mt-3 text-xs text-subtle">
            {mode === "record"
              ? "Live preview only — pose analysis runs on the backend after upload. Frame the patient in profile, full shin + foot visible."
              : "Preview your selected video. Pose analysis runs on the backend after you click Analyse."}
          </p>
        </div>
      </div>

      {error && phase !== "error" && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}
    </div>
  );
}
