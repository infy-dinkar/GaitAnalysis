"use client";
// Shared full-screen live-camera shell for orthopedic + biomech
// assessments. Layout is a direct copy of the rehab
// `RehabCameraShell` (video fills container, transparent skeleton
// canvas over the body, no PIP, `bg-black`) so assessment LIVE mode
// gets the same immersive Kemtai-style look.
//
// Palette + skeleton style stay ASSESSMENT-native (white bones +
// red dots) — not the side-coded rehab palette — so operators can
// tell "you're being assessed" from "you're playing a game" at a
// glance. If a future assessment wants side-coded / centerline /
// angle-arc overlays, those can be added via the `children` slot.
//
// Contract mirrors RehabCameraShell:
//   • Props: onFrame, onError?, children?, fill?
//   • `fill=true` drops the intrinsic aspect-video ratio and
//     stretches to the parent height — LiveModeLayout injects this
//     via cloneElement so consumers don't set it manually.
//
// Camera / pose plumbing (`useCamera`, `usePoseDetectionLive`,
// `LM_LIVE`, `SKELETON_EDGES_LIVE`) is imported AS-IS — no changes
// to the detection stack.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  CameraOff,
  Loader2,
} from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { usePoseDetectionLive as usePoseDetection } from "@/hooks/usePoseDetectionLive";
import { Button } from "@/components/ui/Button";
import {
  LM_LIVE as LM,
  SKELETON_EDGES_LIVE as SKELETON_EDGES,
} from "@/lib/pose/landmarks-live";
import type { Keypoint } from "@tensorflow-models/pose-detection";

const OVERLAY_VIS_THRESHOLD = 0.35;

const FULL_BODY_DOTS: number[] = [
  LM.NOSE, LM.LEFT_EYE, LM.RIGHT_EYE, LM.LEFT_EAR, LM.RIGHT_EAR,
  LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  LM.LEFT_ELBOW,    LM.RIGHT_ELBOW,
  LM.LEFT_WRIST,    LM.RIGHT_WRIST,
  LM.LEFT_HIP,      LM.RIGHT_HIP,
  LM.LEFT_KNEE,     LM.RIGHT_KNEE,
  LM.LEFT_ANKLE,    LM.RIGHT_ANKLE,
  LM.LEFT_HEEL,     LM.RIGHT_HEEL,
  LM.LEFT_FOOT_INDEX, LM.RIGHT_FOOT_INDEX,
];

interface Norm {
  x: number;
  y: number;
  visibility: number;
}

export interface AssessmentCameraShellProps {
  /** Called every frame with the latest 33-keypoint BlazePose
   *  result. The shell does NOT compute any assessment metric —
   *  it exposes raw landmarks so the mounting capture component
   *  can drive its own state machine / analysis. */
  onFrame: (keypoints: Keypoint[], video: HTMLVideoElement) => void;
  /** Optional overlays drawn ON TOP of the skeleton — recording
   *  chip, timer, live-checklist HUD, etc. */
  children?: React.ReactNode;
  onError?: (msg: string) => void;
  /** Fill the parent container instead of using a fixed 16:9
   *  aspect. LiveModeLayout injects `fill=true` via cloneElement. */
  fill?: boolean;
  /** Hide the built-in Start/Stop camera buttons. Some capture
   *  components run their own recording controls in the sidebar
   *  and only want the camera + skeleton visual. */
  hideControls?: boolean;
  /** Start the camera automatically on mount. Use when the parent
   *  has already collected a user click (e.g. a "Start camera"
   *  button in a pre-camera setup step) — that click carries the
   *  user-activation grant into the mount effect so the browser
   *  allows the getUserMedia request. Pair with `hideControls`. */
  autoStart?: boolean;
}

export function AssessmentCameraShell({
  onFrame,
  onError,
  children,
  fill = false,
  hideControls = false,
  autoStart = false,
}: AssessmentCameraShellProps) {
  const { videoRef, active, error: camError, start, stop } = useCamera();
  // Auto-start on mount when requested. Guarded by `active` so a
  // re-render after the stream is already up doesn't retrigger start.
  useEffect(() => {
    if (!autoStart || active) return;
    void start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart]);
  const { ready: detectorReady, error: detectorError, detect } =
    usePoseDetection();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const rafRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const lastNormRef = useRef<Norm[] | null>(null);
  const onFrameRef = useRef(onFrame);

  const [busy, setBusy] = useState(false);

  useEffect(() => { onFrameRef.current = onFrame; }, [onFrame]);
  useEffect(() => {
    if (camError) onError?.(camError);
    if (detectorError) onError?.(detectorError);
  }, [camError, detectorError, onError]);

  const drawSkeleton = useCallback((landmarks: Norm[] | null) => {
    const overlay = overlayRef.current;
    const container = containerRef.current;
    if (!overlay || !container) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (overlay.width !== w * dpr || overlay.height !== h * dpr) {
      overlay.width = w * dpr;
      overlay.height = h * dpr;
      overlay.style.width = `${w}px`;
      overlay.style.height = `${h}px`;
    }
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!landmarks || landmarks.length === 0) return;

    const px = (n: Norm) => ({ x: n.x * w, y: n.y * h });

    // Bones — white with a soft dark halo so they read on ANY
    // background (light shirt, dark bg, etc.).
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = Math.max(2, w * 0.0035);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 3;
    for (const [a, b] of SKELETON_EDGES) {
      const p = landmarks[a];
      const q = landmarks[b];
      if (
        !p || !q ||
        p.visibility < OVERLAY_VIS_THRESHOLD ||
        q.visibility < OVERLAY_VIS_THRESHOLD
      ) continue;
      const A = px(p); const B = px(q);
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
      ctx.stroke();
    }

    // Joint dots — red, slightly larger than bone width.
    ctx.fillStyle = "#EF4444";
    ctx.lineWidth = 2;
    for (const i of FULL_BODY_DOTS) {
      const p = landmarks[i];
      if (!p || p.visibility < OVERLAY_VIS_THRESHOLD) continue;
      const r = Math.max(4, w * 0.005);
      const A = px(p);
      ctx.beginPath();
      ctx.arc(A.x, A.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }, []);

  useEffect(() => {
    if (!active || !detectorReady) {
      drawSkeleton(null);
      lastNormRef.current = null;
      return;
    }
    cancelledRef.current = false;

    const tick = async () => {
      if (cancelledRef.current) return;
      const video = videoRef.current;
      if (!video || video.readyState < 2 || !video.videoWidth) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      try {
        const pose = await detect(video);
        if (cancelledRef.current) return;

        const kp = pose?.keypoints ?? null;
        if (!kp) {
          drawSkeleton(null);
          lastNormRef.current = null;
        } else {
          const sw = video.videoWidth;
          const sh = video.videoHeight;
          const norm: Norm[] = kp.map((p) => ({
            x: 1 - p.x / sw,
            y: p.y / sh,
            visibility: p.score ?? 0,
          }));
          drawSkeleton(norm);
          lastNormRef.current = norm;
          onFrameRef.current(kp, video);
        }
      } catch {
        // ignore
      }
      if (!cancelledRef.current) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelledRef.current = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [active, detectorReady, detect, drawSkeleton, videoRef]);

  useEffect(() => () => stop(), [stop]);

  async function handleStart() {
    setBusy(true);
    try { await start(); } finally { setBusy(false); }
  }

  return (
    <div className={fill ? "flex h-full w-full flex-col" : undefined}>
      <div
        ref={containerRef}
        className={
          fill
            ? "relative min-h-0 w-full flex-1 overflow-hidden rounded-card border border-border bg-gradient-to-br from-[#0A0A0B] via-[#0d0d10] to-[#15151a]"
            : "relative aspect-video overflow-hidden rounded-card border border-border bg-gradient-to-br from-[#0A0A0B] via-[#0d0d10] to-[#15151a]"
        }
      >
        {/* Skeleton canvas is the primary layer — bones + dots draw
            over the dark gradient card so the visual reads as a
            "wireframe on studio-dark" (matches CMJ / SLH / other
            assessment captures). The raw video is only shown as a
            tiny selfie-preview PIP in the bottom-right corner. */}
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0" />
        {/* HUD overlay slot — capture components render recording
            chips, timers, live checklists, etc. on top of the
            skeleton via children. */}
        <div className="pointer-events-none absolute inset-0">{children}</div>
        {/* Selfie-preview PIP — mirrored so the operator sees the
            patient in natural orientation. `pointer-events-none` so
            it never intercepts overlay HUD clicks. */}
        <div
          className={`pointer-events-none absolute bottom-3 right-3 z-10 overflow-hidden rounded-lg border border-white/15 bg-black/80 shadow-2xl transition-opacity duration-200 ${
            active ? "opacity-100" : "opacity-0"
          }`}
        >
          <video
            ref={videoRef}
            playsInline
            muted
            className="block h-16 -scale-x-100 object-cover md:h-20 lg:h-24"
          />
          <span className="absolute left-1.5 top-1 rounded-full bg-black/55 px-1.5 py-[1px] text-[8px] uppercase tracking-[0.14em] text-white/70 backdrop-blur">
            Live
          </span>
        </div>
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            {!detectorReady ? (
              <>
                <Loader2 className="mb-3 h-6 w-6 animate-spin text-white/60" />
                <p className="text-sm text-white/60">Loading pose model…</p>
              </>
            ) : (
              <>
                <Camera className="mb-3 h-10 w-10 text-white/40" />
                <p className="text-sm text-white/60">Camera is off</p>
              </>
            )}
          </div>
        )}
        {active && detectorError && (
          <div className="absolute inset-x-3 top-3 mx-auto flex max-w-md items-start gap-2 rounded-md bg-error/90 px-3 py-2 text-xs text-white">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>Pose model failed: {detectorError}</span>
          </div>
        )}
      </div>
      {camError && (
        <p className={fill ? "mt-1 text-xs text-error" : "mt-3 text-xs text-error"}>
          {camError}
        </p>
      )}
      {!hideControls && (
        <div className={fill ? "mt-2 flex shrink-0 flex-wrap gap-2" : "mt-4 flex flex-wrap gap-3"}>
          {!active ? (
            <Button onClick={handleStart} disabled={busy}>
              {busy
                ? <Loader2 className="h-4 w-4 animate-spin" />
                : <Camera className="h-4 w-4" />}
              Start camera
            </Button>
          ) : (
            <Button variant="secondary" onClick={stop}>
              <CameraOff className="h-4 w-4" />
              Stop
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
