"use client";
// Timed Up and Go (TUG) capture flow.
//
// Two entry modes, both hitting the same POST /api/analyze-tug endpoint:
//
//   1. RECORD — browser captures video via getUserMedia + MediaRecorder.
//      Live preview only, no per-frame pose analysis on the client.
//
//   2. UPLOAD — operator selects a pre-recorded video file from disk.
//      Same backend pipeline (MediaPipe BlazePose Full, 33 keypoints).
//
// In both modes the backend runs the heavy work; the frontend is only
// responsible for getting a video Blob to the server, surfacing the
// upload status, and rendering the TUGReport on return.
//
// State machine (shared across modes):
//
//   idle → recording (record mode only) → uploading → done | error
//   idle → (file selected) ───────────────→ uploading → done | error

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
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import { TUGReport } from "@/components/orthopedic/TUGReport";
import { analyzeTUG, type TUGResult } from "@/lib/orthopedic/tug";

type Phase = "idle" | "recording" | "uploading" | "done" | "error";
type Mode = "record" | "upload";

// Hard ceiling enforced server-side too — keep client + server aligned.
const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = ["video/mp4", "video/webm", "video/quicktime", "video/x-matroska"];

// Pick the best video MIME type the browser supports for live recording.
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

export function TUGCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("record");
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [markerConfirmed, setMarkerConfirmed] = useState(false);

  // Record-mode state
  const [cameraActive, setCameraActive] = useState(false);
  const [recordingMs, setRecordingMs] = useState(0);

  // Upload-mode state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState<string | null>(null);

  // Result + server response shared by both modes
  const [result, setResult] = useState<TUGResult | null>(null);
  const [serverWarning, setServerWarning] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const uploadVideoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const recordingTickRef = useRef<number | null>(null);

  // ── Camera lifecycle ────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
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

  // Tear down camera when switching to upload mode (it's not needed
  // there and free up the camera light immediately).
  useEffect(() => {
    if (mode === "upload" && cameraActive) {
      stopCamera();
    }
  }, [mode, cameraActive, stopCamera]);

  // Cleanup the blob URL we generated for the upload preview.
  useEffect(() => {
    return () => {
      if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    };
  }, [uploadPreviewUrl]);

  // ── Recording control ─────────────────────────────────────
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
      // Recording-mode WebM containers from MediaRecorder often have
      // broken / missing duration metadata in their headers, which
      // causes OpenCV's CAP_PROP_FPS probe to return 0 and the
      // server's validation gate to reject the file as "frame rate
      // could not be determined". The fix: send the wall-clock
      // recording duration as a sidecar so the server can compute
      // FPS = frame_count / duration when the container metadata is
      // unreliable.
      const durationMs = Date.now() - startedAtRef.current;
      await uploadAndAnalyze(blob, durationMs);
    };
    // IMPORTANT: don't pass a timeslice argument. Passing 250 ms tells
    // MediaRecorder to fire ondataavailable every 250 ms, which
    // produces fragmented WebM chunks. The header of the concatenated
    // file then has duration=0 and OpenCV cannot decode FPS. Calling
    // start() with no argument makes MediaRecorder write one clean
    // blob with a proper header at stop().
    rec.start();
    recorderRef.current = rec;
    startedAtRef.current = Date.now();
    setRecordingMs(0);
    setPhase("recording");
    recordingTickRef.current = window.setInterval(() => {
      setRecordingMs(Date.now() - startedAtRef.current);
    }, 100);
  }

  function stopRecording() {
    if (recordingTickRef.current !== null) {
      window.clearInterval(recordingTickRef.current);
      recordingTickRef.current = null;
    }
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.stop();
    }
    setPhase("uploading");
  }

  // ── Upload mode handlers ──────────────────────────────────
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
    // Client-side validation — mirrors the backend gate so the user
    // gets immediate feedback rather than waiting for the upload to
    // start before learning the file is rejected.
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_MB} MB.`);
      setUploadFile(null);
      return;
    }
    if (file.type && !ACCEPTED_VIDEO_TYPES.includes(file.type)) {
      setError(`Unsupported file type (${file.type}). Use MP4, WebM, MOV, or MKV.`);
      setUploadFile(null);
      return;
    }
    setUploadFile(file);
    setUploadPreviewUrl(URL.createObjectURL(file));
  }

  async function analyzeUpload() {
    if (!uploadFile) return;
    setPhase("uploading");
    // No client-known duration for uploaded files — the container's
    // metadata is the source of truth.
    await uploadAndAnalyze(uploadFile, null);
  }

  async function uploadAndAnalyze(blob: Blob, recordingDurationMs: number | null) {
    setError(null);
    setServerWarning(null);
    try {
      const res = await analyzeTUG(blob, patient?.age ?? null, recordingDurationMs);
      if (!res.success || !res.data) {
        setError(res.error ?? "Analysis failed");
        setPhase("error");
        return;
      }
      setResult(res.data);
      if (res.fps_warning || res.duration_warning) {
        setServerWarning([res.fps_warning, res.duration_warning].filter(Boolean).join(" "));
      }
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setPhase("error");
    }
  }

  function reset() {
    setResult(null);
    setError(null);
    setServerWarning(null);
    setRecordingMs(0);
    setMarkerConfirmed(false);
    setPhase("idle");
    stopCamera();
    if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    setUploadFile(null);
    setUploadPreviewUrl(null);
  }

  // ── Done view ─────────────────────────────────────────────
  if (phase === "done" && result) {
    return (
      <div className="space-y-8">
        {serverWarning && (
          <div className="flex items-start gap-3 rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
            <p className="text-foreground">{serverWarning}</p>
          </div>
        )}
        <TUGReport
          patient={patient ?? null}
          patientName={patient?.name ?? null}
          result={result}
        />
        <SaveToPatientButton
          buildPayload={() => ({
            module: "tug",
            metrics: { result },
            observations: { interpretation: result.interpretation },
          })}
        />
        <div className="flex justify-center border-t border-border pt-6">
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
            Run again
          </Button>
        </div>
      </div>
    );
  }

  // ── Capture view ──────────────────────────────────────────
  const recordingDisabled = phase === "recording" || phase === "uploading";
  const fmtTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const cs = Math.floor((ms % 1000) / 10);
    return `${s.toString().padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-10">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      {/* Mode toggle — record vs upload */}
      <div className="inline-flex rounded-card border border-border bg-surface p-1">
        <button
          type="button"
          onClick={() => setMode("record")}
          disabled={recordingDisabled}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "record"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${recordingDisabled ? "opacity-50" : ""}`}
        >
          <Camera className="h-4 w-4" />
          Record live
        </button>
        <button
          type="button"
          onClick={() => setMode("upload")}
          disabled={recordingDisabled}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "upload"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${recordingDisabled ? "opacity-50" : ""}`}
        >
          <Upload className="h-4 w-4" />
          Upload video
        </button>
      </div>

      <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
        {/* LEFT — setup + controls (shared between modes, varies by phase) */}
        <div className="space-y-5">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Setup checklist
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Place a sturdy chair (~45 cm seat height) at one end of a 3-metre path.",
                "Mark the 3-metre point with a cone, tape, or any visible object.",
                "Position the camera SIDEWAYS to the walk path — side view, so the entire 3 m is in the frame.",
                "Patient sits on the chair, back against the backrest, feet flat, arms relaxed.",
                mode === "record"
                  ? "On 'Start', tell the patient: stand up, walk to the marker, turn around, walk back, sit down."
                  : "Record the patient performing the TUG (sit→walk→turn→walk→sit). Trim to start with 'Go' cue and end at seat contact.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Confirm setup
            </p>
            <label className="mt-3 flex cursor-pointer items-start gap-3 text-sm">
              <input
                type="checkbox"
                className="mt-0.5 h-4 w-4 shrink-0 rounded border-border accent-accent"
                checked={markerConfirmed}
                onChange={(e) => setMarkerConfirmed(e.target.checked)}
                disabled={recordingDisabled}
              />
              <span className="text-foreground">
                I have confirmed that the marker is placed exactly{" "}
                <span className="font-semibold">3 metres</span> from the chair.
              </span>
            </label>
            <p className="mt-2 text-xs text-muted">
              Walking speed is calculated using this 3 m distance — incorrect
              placement will skew the metrics.
            </p>
          </div>

          {/* RECORD MODE — recording controls */}
          {mode === "record" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Recording controls
              </p>

              {phase === "idle" && (
                <div className="mt-3 space-y-3">
                  <p className="text-sm text-muted">
                    Step 1 — start the camera and frame the full 3 m path.
                  </p>
                  {!cameraActive ? (
                    <Button onClick={startCamera}>
                      <Camera className="h-4 w-4" />
                      Start camera
                    </Button>
                  ) : (
                    <div className="flex gap-2">
                      <Button onClick={startRecording} disabled={!markerConfirmed}>
                        <Play className="h-4 w-4" />
                        Start recording
                      </Button>
                      <Button variant="secondary" onClick={stopCamera}>
                        <CameraOff className="h-4 w-4" />
                        Stop camera
                      </Button>
                    </div>
                  )}
                  {!markerConfirmed && cameraActive && (
                    <p className="text-xs text-warning">
                      Confirm the 3 m marker placement before recording.
                    </p>
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
                    Click <em>Stop</em> as soon as the patient is fully seated again.
                  </p>
                  <Button onClick={stopRecording}>
                    <Square className="h-4 w-4" />
                    Stop recording
                  </Button>
                  {recordingMs / 1000 > 60 && (
                    <p className="text-xs text-warning">
                      Recording has exceeded 60 seconds — the backend will reject
                      clips longer than 60 s. Stop and try again.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* UPLOAD MODE — file picker + analyze */}
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
                    MP4, WebM, MOV, or MKV · max {MAX_FILE_MB} MB · 5–60 s
                  </p>
                  <input
                    type="file"
                    accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
                    className="hidden"
                    onChange={(e) => handleFilePick(e.target.files?.[0] ?? null)}
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
                    <Button onClick={analyzeUpload} disabled={!markerConfirmed}>
                      <Upload className="h-4 w-4" />
                      Analyse video
                    </Button>
                    <Button variant="secondary" onClick={() => handleFilePick(null)}>
                      Choose different file
                    </Button>
                  </div>
                  {!markerConfirmed && (
                    <p className="text-xs text-warning">
                      Confirm the 3 m marker placement before analysing.
                    </p>
                  )}
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

          <p className="text-xs text-muted">
            Cutoffs (TUG): &lt; 10 s normal · 10-13.5 s mild fall risk ·
            13.5-20 s elevated fall risk · &gt; 20 s significant impairment.
            Turn &gt; 4 s OR turn-step-count &gt; 5 raise independent flags.
            3-metre path is operator-confirmed; walking speed = 3.0 m / phase
            duration.
          </p>
        </div>

        {/* RIGHT — preview pane (varies by mode) */}
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
              ? "Live preview only — pose analysis runs on the backend after upload. Frame the full 3 m walking path end-to-end before starting."
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
