"use client";
// Timed Up and Go (TUG) capture flow.
//
// Two entry modes, both hitting the same POST /api/analyze-tug endpoint:
//
//   1. RECORD — fullscreen less-click auto-flow. One click ("Start
//      Assessment") opens the fullscreen LiveModeLayout, the camera
//      auto-starts, a 3-2-1 countdown runs, and MediaRecorder starts
//      by itself. The operator clicks Stop at seat contact; the blob
//      uploads to the backend and the report auto-saves (doctor flow).
//      Live preview only, no per-frame pose analysis on the client.
//
//   2. UPLOAD — operator selects a pre-recorded video file from disk.
//      Same backend pipeline (MediaPipe BlazePose Full, 33 keypoints).
//      Unchanged boxed form + manual save.
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
  FileVideo,
  Loader2,
  RotateCcw,
  Square,
  Upload,
  Video,
} from "lucide-react";

import { Button } from "@/components/ui/Button";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
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

// Camera surface for the fullscreen shell. Accepts the `fill` prop
// LiveModeLayout injects into its camera slot (plain DOM nodes would
// receive it as an unknown attribute otherwise).
function TUGCameraSurface({
  videoRef,
  fill: _fill,
  children,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  fill?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className="relative min-h-0 w-full flex-1 overflow-hidden rounded-card border border-border bg-gradient-to-br from-[#0A0A0B] via-[#0d0d10] to-[#15151a]">
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 h-full w-full object-cover"
      />
      <div className="pointer-events-none absolute inset-0">{children}</div>
    </div>
  );
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

  // Fullscreen auto-flow shell (record mode). One click opens the
  // shell; the camera auto-starts; once frames flow a 3-2-1 countdown
  // runs and MediaRecorder starts without another click.
  const [liveFullscreen, setLiveFullscreen] = useState(false);

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

  // Auto-start the camera exactly once each time the fullscreen shell
  // opens (StrictMode-safe; enterLive already ran inside a user
  // gesture so getUserMedia succeeds).
  const liveAutoStartRef = useRef(false);
  useEffect(() => {
    if (!liveFullscreen) {
      liveAutoStartRef.current = false;
      return;
    }
    if (liveAutoStartRef.current) return;
    liveAutoStartRef.current = true;
    void startCamera();
  }, [liveFullscreen, startCamera]);

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

  // ── Fullscreen auto-flow (record mode) ─────────────────────
  // Countdown starts only once the camera stream is actually live —
  // otherwise the 3-2-1 would eat the getUserMedia permission delay.
  const {
    phase: flowPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(
    mode === "record" && liveFullscreen && cameraActive && phase === "idle",
    () => {
      startRecording();
    },
  );

  // Enter the fullscreen shell — the single click of record mode.
  function enterLive() {
    setError(null);
    setPhase("idle");
    setLiveFullscreen(true);
  }

  // Exit the fullscreen shell. If a recording is in flight this is a
  // deliberate abort — detach onstop so the partial blob is NOT
  // uploaded to the backend.
  function exitLive() {
    if (recordingTickRef.current !== null) {
      window.clearInterval(recordingTickRef.current);
      recordingTickRef.current = null;
    }
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.onstop = null;
      try {
        rec.stop();
      } catch {
        // ignore
      }
    }
    recorderRef.current = null;
    chunksRef.current = [];
    stopCamera();
    setRecordingMs(0);
    setError(null);
    setPhase("idle");
    setLiveFullscreen(false);
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
        if (mode === "record") {
          stopCamera();
          setLiveFullscreen(false);
        }
        return;
      }
      setResult(res.data);
      if (res.fps_warning || res.duration_warning) {
        setServerWarning([res.fps_warning, res.duration_warning].filter(Boolean).join(" "));
      }
      setPhase("done");
      if (mode === "record") {
        stopCamera();
        setLiveFullscreen(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setPhase("error");
      if (mode === "record") {
        stopCamera();
        setLiveFullscreen(false);
      }
    }
  }

  function reset() {
    setResult(null);
    setError(null);
    setServerWarning(null);
    setRecordingMs(0);
    setMarkerConfirmed(false);
    setPhase("idle");
    setLiveFullscreen(false);
    stopCamera();
    if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    setUploadFile(null);
    setUploadPreviewUrl(null);
  }

  const fmtTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const cs = Math.floor((ms % 1000) / 10);
    return `${s.toString().padStart(2, "0")}.${cs.toString().padStart(2, "0")}`;
  };

  // ── Done view ─────────────────────────────────────────────
  if (phase === "done" && result) {
    const buildPayload = () => ({
      module: "tug" as const,
      metrics: { result },
      observations: { interpretation: result.interpretation },
    });
    return (
      <div className="space-y-8">
        {/* Results auto-save in the doctor flow (toast with a 10s
            undo) for both record and upload runs. */}
        <AutoSaveToast buildPayload={buildPayload} />
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
        <div className="flex justify-center border-t border-border pt-6">
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
            Run again
          </Button>
        </div>
      </div>
    );
  }

  // ── Fullscreen auto-flow shell (record mode) ──────────────
  if (mode === "record" && liveFullscreen && phase !== "error") {
    const overSixty = recordingMs / 1000 > 60;
    return (
      <LiveModeLayout
        title="Timed Up & Go (TUG)"
        subtitle={
          phase === "recording"
            ? "Recording — stop as soon as the patient is fully seated again"
            : phase === "uploading"
              ? "Analysing on the server…"
              : "Camera sideways to the 3 m walk path, patient seated"
        }
        onExit={exitLive}
        camera={(
          <TUGCameraSurface videoRef={videoRef}>
            {!cameraActive && (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="rounded-full bg-black/60 px-4 py-2 text-sm text-white/80">
                  Starting camera…
                </p>
              </div>
            )}
            {flowPhase === "countdown" && countdown !== null && (
              <AutoFlowCountdownOverlay countdown={countdown} label="Recording starts in" />
            )}
            {phase === "recording" && (
              <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                <p className="text-[10px] uppercase tracking-[0.14em] text-rose-300">
                  ● Recording
                </p>
                <p className="tabular text-2xl font-semibold text-white">
                  {fmtTime(recordingMs)}s
                </p>
                <p className="text-[10px] text-white/70">
                  Stop at seat contact after the walk
                </p>
              </div>
            )}
            {phase === "uploading" && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
                <div className="flex items-center gap-2 rounded-lg border border-white/15 bg-black/70 px-4 py-3 text-sm text-white">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading and analysing — this can take 10-30 seconds.
                </div>
              </div>
            )}
          </TUGCameraSurface>
        )}
        sidebar={(
          <>
            {flowPhase === "countdown" && countdown !== null && (
              <AutoFlowCountdownCard
                countdown={countdown}
                onSkip={skipCountdown}
                hint="Patient seated, back against the backrest, full 3 m path in frame."
              />
            )}

            {phase === "recording" && (
              <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="inline-flex items-center gap-2 text-sm font-medium text-foreground">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-error" />
                    Recording
                  </p>
                  <p className="tabular text-2xl font-semibold text-accent">
                    {fmtTime(recordingMs)}s
                  </p>
                </div>
                <p className="text-xs text-muted">
                  Tell the patient: stand up, walk to the marker, turn around,
                  walk back, sit down. Click <em>Stop</em> as soon as the
                  patient is fully seated again.
                </p>
                {overSixty && (
                  <p className="text-xs text-warning">
                    Recording has exceeded 60 seconds — the backend will reject
                    clips longer than 60 s. Stop and try again.
                  </p>
                )}
                <Button onClick={stopRecording}>
                  <Square className="h-4 w-4" />
                  Stop recording
                </Button>
              </div>
            )}

            {phase === "uploading" && (
              <div className="rounded-card border border-blue-500/30 bg-blue-500/10 p-3 text-sm">
                <p className="flex items-center gap-2 font-medium text-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading and analysing — this can take 10-30 seconds.
                </p>
              </div>
            )}

            <div className="rounded-card border border-border bg-surface p-3 text-xs text-muted">
              <p className="font-semibold text-foreground">Session brief</p>
              <ol className="mt-2 list-decimal space-y-1 pl-4">
                <li>Patient seated, side-on camera, full 3 m path in frame.</li>
                <li>After the 3-2-1, cue: stand, walk, turn, walk back, sit.</li>
                <li>Stop the recording at seat contact — analysis runs on the server.</li>
              </ol>
            </div>

            {error && (
              <div className="rounded-card border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-foreground">
                <AlertTriangle className="mr-2 inline h-4 w-4 text-rose-500" />
                {error}
              </div>
            )}

            <div className="mt-auto flex flex-wrap gap-2">
              <Button variant="ghost" size="sm" onClick={exitLive}>
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            </div>
          </>
        )}
      />
    );
  }

  // ── Capture view ──────────────────────────────────────────
  const recordingDisabled = phase === "recording" || phase === "uploading";

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
                  ? "After the 3-2-1 countdown, tell the patient: stand up, walk to the marker, turn around, walk back, sit down."
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

          {/* RECORD MODE — one-click fullscreen auto-flow entry */}
          {mode === "record" && phase !== "error" && (
            <div className="rounded-card border border-border bg-surface p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Start assessment
              </p>
              <p className="mt-3 text-sm text-muted">
                One click — the camera opens fullscreen, a 3-2-1 countdown
                runs, and recording starts by itself. Stop the recording as
                soon as the patient is fully seated again; the analysis runs
                on the server and the report saves to the patient record.
              </p>
              <div className="mt-4">
                <Button onClick={enterLive} disabled={!markerConfirmed}>
                  <Camera className="h-4 w-4" />
                  Start Assessment
                </Button>
              </div>
              {!markerConfirmed && (
                <p className="mt-2 text-xs text-warning">
                  Confirm the 3 m marker placement before starting.
                </p>
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
              <div className="absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
                <Camera className="mb-3 h-10 w-10 text-white/40" />
                <p className="text-sm text-white/60">
                  The camera opens fullscreen when you click Start Assessment.
                </p>
              </div>
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
              ? "Live preview only — pose analysis runs on the backend after upload. Frame the full 3 m walking path end-to-end before the countdown ends."
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
