"use client";
// Gait record-then-upload component.
//
// Live camera preview with a full-body BlazePose Lite 33-landmark
// skeleton drawn on a canvas overlay for visual framing feedback.
// The RECORDED video is the raw camera stream ONLY — the skeleton
// canvas is a separate DOM element that sits visually on top of the
// <video> but is NOT part of the MediaStream MediaRecorder consumes.
// This is critical: the backend re-runs MediaPipe BlazePose Full on
// the raw video for authoritative metrics, so any baked-in skeleton
// pixels would break the pose detection.
//
// Architecture (mirrors components/biomech/LiveBiomechCamera.tsx +
// components/biomech/AnkleCapture.tsx, kept as a separate component
// in components/gait/ rather than reaching into biomech):
//
//   • useCamera hook  → getUserMedia stream + start/stop
//   • usePoseDetectionLive hook → singleton MediaPipe BlazePose Lite
//     detector running browser-side via @mediapipe/pose
//   • rAF loop: read frame from <video>, run detect(), draw skeleton
//     on overlay <canvas>. Skipped frames (concurrent detect calls)
//     are handled gracefully — last drawn skeleton stays until the
//     next successful frame.
//   • MediaRecorder(streamRef.current, ...) — records the camera
//     stream directly, no canvas mixing. pickRecorderMime() matches
//     AnkleCapture's vp9 → vp8 → webm → mp4 fallback ladder.
//   • Single-blob encode (no MediaRecorder timeslice) → clean
//     header in the common case. The backend repair fallback handles
//     the failure mode where the WebM duration metadata is missing.
//   • On stop, the recorded Blob is wrapped as a File and the
//     parent's onRecorded callback drives the existing analyzeGait
//     upload + navigation flow — the gait pipeline downstream is
//     untouched.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  CameraOff,
  Loader2,
  Square,
  Video,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { useCamera } from "@/hooks/useCamera";
import { useSecondaryCamera } from "@/hooks/useSecondaryCamera";
import { usePoseDetectionLive } from "@/hooks/usePoseDetectionLive";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

// MoveNet / BlazePose visibility floor below which a landmark is
// likely hallucinated (out-of-frame or fully occluded). Same value
// LiveBiomechCamera uses for the overlay.
const OVERLAY_VIS_THRESHOLD = 0.35;
const LINE_COLOR = "#FFFFFF";
const DOT_COLOR = "#EF4444";

// 33-landmark BlazePose body skeleton — face dots + arms + torso +
// legs. Drawn as visual feedback only; the analysis re-runs on the
// server with BlazePose Full and doesn't consume these.
const FACE_DOTS = [
  LM.NOSE,
  LM.LEFT_EYE, LM.RIGHT_EYE,
  LM.LEFT_EAR, LM.RIGHT_EAR,
];

const BODY_DOTS: number[] = [
  ...FACE_DOTS,
  LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  LM.LEFT_ELBOW,    LM.RIGHT_ELBOW,
  LM.LEFT_WRIST,    LM.RIGHT_WRIST,
  LM.LEFT_HIP,      LM.RIGHT_HIP,
  LM.LEFT_KNEE,     LM.RIGHT_KNEE,
  LM.LEFT_ANKLE,    LM.RIGHT_ANKLE,
];

const BODY_EDGES: Array<[number, number]> = [
  // Arms
  [LM.LEFT_SHOULDER,  LM.LEFT_ELBOW],
  [LM.LEFT_ELBOW,     LM.LEFT_WRIST],
  [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW],
  [LM.RIGHT_ELBOW,    LM.RIGHT_WRIST],
  // Torso
  [LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER],
  [LM.LEFT_SHOULDER,  LM.LEFT_HIP],
  [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
  [LM.LEFT_HIP,       LM.RIGHT_HIP],
  // Legs
  [LM.LEFT_HIP,       LM.LEFT_KNEE],
  [LM.LEFT_KNEE,      LM.LEFT_ANKLE],
  [LM.RIGHT_HIP,      LM.RIGHT_KNEE],
  [LM.RIGHT_KNEE,     LM.RIGHT_ANKLE],
];

/** MediaRecorder codec preference — matches AnkleCapture so the
 *  backend's existing WebM-repair path covers both. */
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
  /** Fired when the patient stops recording. The File is built from
   *  the recorded raw camera blob (NOT the skeleton-overlaid view) so
   *  the backend gets clean footage to re-run BlazePose Full on. The
   *  durationMs is wall-clock between start() and stop() and lets the
   *  backend repair WebMs with broken duration headers. */
  onRecorded: (file: File, durationMs: number) => void;
  /** When true (e.g. parent is currently uploading) disables the
   *  controls so the operator can't start a second recording mid-
   *  flight. */
  disabled?: boolean;
}

type Phase = "idle" | "recording";

export function GaitRecordCapture({ onRecorded, disabled }: Props) {
  const { videoRef, streamRef, active, error: camError, start: startCamera, stop: stopCamera } =
    useCamera();
  const { ready: poseReady, error: poseError, detect } = usePoseDetectionLive();

  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef<number>(0);
  const tickRef = useRef<number | null>(null);

  const [phase, setPhase] = useState<Phase>("idle");
  const [recordingMs, setRecordingMs] = useState(0);
  const [localError, setLocalError] = useState<string | null>(null);

  // ── Secondary camera (live preview only — no recording) ──────
  // useSecondaryCamera mirrors useCamera's shape but is deviceId-
  // driven and never feeds MediaRecorder. The recorded blob still
  // pulls from streamRef.current (primary only), so the upload
  // path stays byte-identical.
  const secondaryVideoRef = useRef<HTMLVideoElement | null>(null);
  const {
    active: secActive,
    error: secError,
    start: startSecondary,
    stop: stopSecondary,
  } = useSecondaryCamera(secondaryVideoRef);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [primaryDeviceId, setPrimaryDeviceId] = useState<string>("");
  const [secondaryDeviceId, setSecondaryDeviceId] = useState<string>("");

  // ── Skeleton overlay rAF loop ─────────────────────────────────
  // Runs only while the camera is active AND the detector is ready.
  // Reads each frame from the <video>, runs MediaPipe BlazePose,
  // draws the skeleton on the overlay canvas. The MediaRecorder
  // (when running) consumes streamRef.current independently — the
  // canvas is purely visual.
  useEffect(() => {
    if (!active || !poseReady) return;
    cancelledRef.current = false;

    const loop = async () => {
      if (cancelledRef.current) return;
      const v = videoRef.current;
      const c = overlayRef.current;
      if (!v || !c || v.videoWidth === 0 || v.videoHeight === 0) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      // Keep overlay canvas pixel dims in sync with the video so the
      // pose keypoints (returned in input-pixel space by the
      // detector) draw correctly without scaling.
      if (c.width !== v.videoWidth) c.width = v.videoWidth;
      if (c.height !== v.videoHeight) c.height = v.videoHeight;

      try {
        const result = await detect(v);
        const ctx = c.getContext("2d");
        if (!ctx) {
          rafRef.current = requestAnimationFrame(loop);
          return;
        }
        ctx.clearRect(0, 0, c.width, c.height);

        if (result && result.keypoints.length > 0) {
          const kp = result.keypoints;
          // Edges
          ctx.strokeStyle = LINE_COLOR;
          ctx.lineWidth = 3;
          for (const [a, b] of BODY_EDGES) {
            const pa = kp[a];
            const pb = kp[b];
            if (!pa || !pb) continue;
            if ((pa.score ?? 0) < OVERLAY_VIS_THRESHOLD) continue;
            if ((pb.score ?? 0) < OVERLAY_VIS_THRESHOLD) continue;
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.stroke();
          }
          // Dots
          ctx.fillStyle = DOT_COLOR;
          for (const i of BODY_DOTS) {
            const p = kp[i];
            if (!p) continue;
            if ((p.score ?? 0) < OVERLAY_VIS_THRESHOLD) continue;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      } catch {
        // Drop frame on detect failure; loop continues.
      }
      rafRef.current = requestAnimationFrame(loop);
    };

    loop();
    return () => {
      cancelledRef.current = true;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [active, poseReady, detect, videoRef]);

  // ── Device enumeration + primary auto-detect ──────────────────
  // Populate enumerated videoinput devices once the primary camera
  // is active (browsers blank `label` until at least one
  // getUserMedia call has succeeded). Also detect which device the
  // browser actually picked for the primary so the dropdown
  // reflects reality before the operator interacts with it.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    (async () => {
      try {
        const all = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;
        setDevices(all.filter((d) => d.kind === "videoinput"));
      } catch {
        /* enumeration failure isn't fatal */
      }
      if (cancelled) return;
      const track = streamRef.current?.getVideoTracks()[0];
      const settings = track?.getSettings();
      if (settings?.deviceId && !primaryDeviceId) {
        setPrimaryDeviceId(settings.deviceId);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [active, streamRef, primaryDeviceId]);

  // ── Primary-camera device swap ────────────────────────────────
  // When the operator picks a different primary camera, we mutate
  // the exposed streamRef + videoRef that useCamera() returns —
  // useCamera.ts itself is untouched (10 other capture screens
  // depend on it). The MediaRecorder reads streamRef.current at
  // the moment of rec.start(), and the primary dropdown is
  // disabled while phase === "recording" (see JSX below), so this
  // swap can only happen between recordings; the recorded blob
  // stays clean.
  const overridePrimary = useCallback(
    async (deviceId: string) => {
      if (!streamRef.current) return;
      const current = streamRef.current
        .getVideoTracks()[0]
        ?.getSettings()?.deviceId;
      if (current === deviceId) return;
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      try {
        const newStream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: deviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        streamRef.current = newStream;
        if (videoRef.current) {
          videoRef.current.srcObject = newStream;
          await videoRef.current.play().catch(() => {});
        }
      } catch (e) {
        setLocalError(
          e instanceof Error
            ? e.message
            : "Could not switch primary camera.",
        );
      }
    },
    [streamRef, videoRef],
  );

  useEffect(() => {
    if (!active || !primaryDeviceId) return;
    overridePrimary(primaryDeviceId);
  }, [primaryDeviceId, active, overridePrimary]);

  // ── Secondary lifecycle (lock-step with primary) ──────────────
  // Start when primary is active AND a valid (different-from-
  // primary) deviceId is picked. Stop on any of: primary stops,
  // deviceId cleared, deviceId equals primary. One Start/Stop
  // button thereby controls both cameras.
  useEffect(() => {
    if (
      !active ||
      !secondaryDeviceId ||
      secondaryDeviceId === primaryDeviceId
    ) {
      stopSecondary();
      return;
    }
    startSecondary(secondaryDeviceId);
  }, [
    active,
    secondaryDeviceId,
    primaryDeviceId,
    startSecondary,
    stopSecondary,
  ]);

  // ── Recording control ─────────────────────────────────────────
  const startRecording = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) {
      setLocalError("Start the camera first.");
      return;
    }
    setLocalError(null);
    chunksRef.current = [];
    const mime = pickRecorderMime();
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, { mimeType: mime });
    } catch (e) {
      setLocalError(
        e instanceof Error
          ? `Could not start recorder: ${e.message}`
          : "Could not start recorder.",
      );
      return;
    }
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mime });
      const durationMs = Date.now() - startedAtRef.current;
      const ext = mime.startsWith("video/mp4") ? "mp4" : "webm";
      // File wraps the raw camera Blob — no canvas pixels included.
      const file = new File([blob], `gait-${Date.now()}.${ext}`, { type: mime });
      // Stop the camera once the recording is captured so the
      // detector loop tears down promptly while the parent uploads.
      stopCamera();
      setPhase("idle");
      onRecorded(file, durationMs);
    };
    // No timeslice — single-blob encode produces a clean WebM header
    // when the browser/codec supports it.
    rec.start();
    recorderRef.current = rec;
    startedAtRef.current = Date.now();
    setRecordingMs(0);
    setPhase("recording");
    tickRef.current = window.setInterval(() => {
      setRecordingMs(Date.now() - startedAtRef.current);
    }, 100);
  }, [streamRef, onRecorded, stopCamera]);

  const stopRecording = useCallback(() => {
    if (tickRef.current !== null) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
    // phase flip happens inside rec.onstop after the blob is captured.
  }, []);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      if (tickRef.current !== null) {
        clearInterval(tickRef.current);
        tickRef.current = null;
      }
      if (
        recorderRef.current &&
        recorderRef.current.state === "recording"
      ) {
        try {
          recorderRef.current.stop();
        } catch {
          /* noop */
        }
      }
    };
  }, []);

  const detectorError = poseError;
  const visibleError = localError ?? camError ?? detectorError;
  const seconds = (recordingMs / 1000).toFixed(1);

  return (
    <div className="space-y-4">
      <div className="rounded-card border border-border bg-surface p-4">
        {/* Two-column camera grid. LEFT = primary (skeleton-on-
            black, unchanged behaviour and recorded). RIGHT =
            secondary (plain reference preview, no recording, no
            skeleton). max-w-4xl keeps each tile around ~370 px wide
            at md+ so the controls below stay above the fold. */}
        <div className="mx-auto grid w-full max-w-4xl grid-cols-1 gap-4 md:grid-cols-2">
          {/* ── Primary tile ─────────────────────────────────── */}
          <div className="space-y-2">
            <span className="inline-flex items-center gap-1.5 rounded-md bg-elevated px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-subtle">
              Primary · side view
            </span>
            <select
              value={primaryDeviceId}
              onChange={(e) => setPrimaryDeviceId(e.target.value)}
              disabled={!active || phase === "recording" || devices.length === 0}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              {devices.length === 0 && (
                <option value="">No cameras enumerated yet</option>
              )}
              {devices.map((d) => (
                <option
                  key={d.deviceId}
                  value={d.deviceId}
                  disabled={d.deviceId === secondaryDeviceId && d.deviceId !== ""}
                >
                  {d.label || `Camera ${d.deviceId.slice(0, 8)}…`}
                </option>
              ))}
            </select>
            <div className="relative aspect-video w-full overflow-hidden rounded-md bg-black">
              {/* The <video> renders the raw camera stream
                  internally (so usePoseDetectionLive.detect(v) can
                  read videoWidth/videoHeight + sample pixels every
                  frame), but is visually hidden via `opacity-0`.
                  The skeleton canvas above is the only thing the
                  operator sees. DISPLAY-ONLY: MediaRecorder reads
                  the RAW MediaStream from streamRef.current —
                  opacity has zero effect on the recorded blob. */}
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <video
                ref={videoRef}
                playsInline
                muted
                className="absolute inset-0 h-full w-full object-contain opacity-0"
              />
              <canvas
                ref={overlayRef}
                className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              />
              {!active && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-subtle">
                  <CameraOff className="h-8 w-8 text-subtle" />
                  <p>Camera off — click Start camera below.</p>
                </div>
              )}
              {active && !poseReady && (
                <div className="absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-md bg-background/80 px-2 py-1 text-[10px] text-subtle">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Loading skeleton…
                </div>
              )}
              {phase === "recording" && (
                <div className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-md bg-error/90 px-2 py-1 text-[11px] font-semibold text-white">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
                  REC · {seconds}s
                </div>
              )}
            </div>
          </div>

          {/* ── Secondary tile (preview only — no recording) ──── */}
          <div className="space-y-2">
            <span className="inline-flex items-center gap-1.5 rounded-md bg-elevated px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-subtle">
              Reference · front / back
            </span>
            <select
              value={secondaryDeviceId}
              onChange={(e) => setSecondaryDeviceId(e.target.value)}
              disabled={!active || phase === "recording" || devices.length < 2}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            >
              <option value="">— None (no reference) —</option>
              {devices.map((d) => (
                <option
                  key={d.deviceId}
                  value={d.deviceId}
                  disabled={d.deviceId === primaryDeviceId}
                >
                  {d.label || `Camera ${d.deviceId.slice(0, 8)}…`}
                </option>
              ))}
            </select>
            <div className="relative aspect-video w-full overflow-hidden rounded-md bg-black">
              {/* Plain visible preview. No opacity-0, no canvas
                  overlay, no skeleton. Stream is held in
                  useSecondaryCamera's streamRef and is NEVER
                  passed to MediaRecorder — secondary produces no
                  blob, touches no backend. autoPlay deliberately
                  omitted; useSecondaryCamera.start() handles the
                  explicit play() once srcObject is attached. */}
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <video
                ref={secondaryVideoRef}
                playsInline
                muted
                className="absolute inset-0 h-full w-full object-contain"
              />
              {(!active || !secondaryDeviceId) && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-subtle">
                  <CameraOff className="h-8 w-8 text-subtle" />
                  <p>
                    {!active
                      ? "Reference preview activates with primary."
                      : devices.length < 2
                        ? "Only one camera detected."
                        : "Pick a reference camera above."}
                  </p>
                </div>
              )}
              {active && secondaryDeviceId && secError && (
                <div className="absolute inset-0 flex items-center justify-center bg-error/10 p-3 text-center text-xs text-error">
                  <span>{secError}</span>
                </div>
              )}
              {active && secActive && secondaryDeviceId && !secError && (
                <div className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-md bg-background/80 px-2 py-1 text-[10px] text-subtle">
                  LIVE
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          {!active ? (
            <Button onClick={startCamera} disabled={disabled} size="sm">
              <Camera className="h-4 w-4" />
              Start camera
            </Button>
          ) : (
            <Button
              onClick={stopCamera}
              variant="secondary"
              disabled={disabled || phase === "recording"}
              size="sm"
            >
              <CameraOff className="h-4 w-4" />
              Stop camera
            </Button>
          )}
          {active && phase === "idle" && (
            <Button
              onClick={startRecording}
              disabled={disabled || !poseReady}
              size="sm"
            >
              <Video className="h-4 w-4" />
              Start recording
            </Button>
          )}
          {phase === "recording" && (
            <Button
              onClick={stopRecording}
              variant="secondary"
              disabled={disabled}
              size="sm"
            >
              <Square className="h-4 w-4" />
              Stop & upload
            </Button>
          )}
        </div>

        <p className="mt-3 text-xs text-muted">
          Stand fully in frame, side-on to the primary camera, and walk back-
          and-forth for 4–6 cycles (~10 s). The skeleton overlay and
          reference preview are for framing only — only the primary clip is
          uploaded and analysed.
        </p>
      </div>

      {visibleError && (
        <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <div className="flex items-start gap-2">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
            <p className="text-foreground">{visibleError}</p>
          </div>
        </div>
      )}
    </div>
  );
}
