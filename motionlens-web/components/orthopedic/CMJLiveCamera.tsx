"use client";
// Live BlazePose capture for the D4 Counter-Movement Jump test.
// Lateral view, both legs in frame. Clones SingleLegHopLiveCamera
// structurally — same dark canvas + skeleton overlay + PiP toggle —
// but installs its own window-level screenshot helper
// (__cmjCapture) so the apex-frame screenshot is independent of
// other tests on the page.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  CameraOff,
  Eye,
  EyeOff,
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

interface Props {
  onFrame: (keypoints: Keypoint[], video: HTMLVideoElement) => void;
  onError?: (msg: string) => void;
  /** Auto-request the camera on mount (parent already gated behind a
   *  user gesture, e.g. "Start Assessment"). Same pattern as
   *  RehabCameraShell.autoStart. */
  autoStart?: boolean;
  /** Hide the Start/Stop/PiP button row — parent owns session control. */
  hideControls?: boolean;
  /** Fill the parent container (LiveModeLayout camera slot) instead of
   *  the fixed 16:9 aspect card. */
  fill?: boolean;
  /** Notified when the camera stream turns on/off — lets the parent
   *  hold its countdown until frames are actually flowing. */
  onActiveChange?: (active: boolean) => void;
  /** Overlay slot rendered above the skeleton canvas (countdown,
   *  complete banner, HUD chips). */
  children?: React.ReactNode;
}

export function CMJLiveCamera({
  onFrame,
  onError,
  autoStart = false,
  hideControls = false,
  fill = false,
  onActiveChange,
  children,
}: Props) {
  const { videoRef, active, error: camError, start, stop } = useCamera();
  const { ready: detectorReady, error: detectorError, detect } = usePoseDetection();

  // Mirror stream state up to the parent.
  const onActiveChangeRef = useRef(onActiveChange);
  useEffect(() => { onActiveChangeRef.current = onActiveChange; });
  useEffect(() => { onActiveChangeRef.current?.(active); }, [active]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const compositeCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const rafRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const lastVideoRef = useRef<HTMLVideoElement | null>(null);
  const lastNormRef = useRef<Norm[] | null>(null);
  const onFrameRef = useRef(onFrame);

  const [busy, setBusy] = useState(false);
  const [showPip, setShowPip] = useState(true);

  // Auto-start exactly once on mount when opted in (StrictMode-safe;
  // parent already gated this behind a user gesture).
  const autoStartFiredRef = useRef(false);
  useEffect(() => {
    if (!autoStart || autoStartFiredRef.current) return;
    autoStartFiredRef.current = true;
    setBusy(true);
    start().catch(() => {}).finally(() => setBusy(false));
  }, [autoStart, start]);

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

    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = Math.max(2, w * 0.0035);
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
      lastVideoRef.current = video;
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

  const captureFrame = useCallback((): string | null => {
    const video = lastVideoRef.current ?? videoRef.current;
    if (!video || !video.videoWidth) return null;
    const targetW = Math.min(480, video.videoWidth);
    const scale = targetW / video.videoWidth;
    const cw = Math.round(video.videoWidth * scale);
    const ch = Math.round(video.videoHeight * scale);
    let comp = compositeCanvasRef.current;
    if (!comp) {
      comp = document.createElement("canvas");
      compositeCanvasRef.current = comp;
    }
    comp.width = cw;
    comp.height = ch;
    const ctx = comp.getContext("2d");
    if (!ctx) return null;
    ctx.save();
    ctx.translate(cw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, cw, ch);
    ctx.restore();
    const norm = lastNormRef.current;
    if (norm) {
      const px = (n: Norm) => ({ x: n.x * cw, y: n.y * ch });
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = Math.max(1.5, cw * 0.0035);
      ctx.shadowColor = "rgba(0,0,0,0.55)";
      ctx.shadowBlur = 2;
      for (const [a, b] of SKELETON_EDGES) {
        const p = norm[a];
        const q = norm[b];
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
      ctx.fillStyle = "#EF4444";
      for (const i of FULL_BODY_DOTS) {
        const p = norm[i];
        if (!p || p.visibility < OVERLAY_VIS_THRESHOLD) continue;
        const r = Math.max(3, cw * 0.005);
        const A = px(p);
        ctx.beginPath();
        ctx.arc(A.x, A.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
    return comp.toDataURL("image/jpeg", 0.78);
  }, [videoRef]);

  useEffect(() => {
    (window as unknown as { __cmjCapture?: () => string | null })
      .__cmjCapture = captureFrame;
    return () => {
      delete (window as unknown as { __cmjCapture?: () => string | null })
        .__cmjCapture;
    };
  }, [captureFrame]);

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
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0" />
        {/* Parent overlay slot (countdown / complete banner / HUD). */}
        <div className="pointer-events-none absolute inset-0">{children}</div>
        <div
          className={`absolute bottom-3 right-3 z-10 overflow-hidden rounded-lg border border-white/15 bg-black/80 shadow-2xl transition-opacity duration-200 ${
            active && showPip ? "opacity-100" : "pointer-events-none opacity-0"
          }`}
        >
          <video
            ref={videoRef}
            playsInline
            muted
            className="block h-12 -scale-x-100 object-cover md:h-14 lg:h-16"
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
      {camError && <p className={fill ? "mt-1 text-xs text-error" : "mt-3 text-xs text-error"}>{camError}</p>}
      {!hideControls && (
        <div className="mt-4 flex flex-wrap gap-3">
          {!active ? (
            <Button onClick={handleStart} disabled={busy}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
              Start camera
            </Button>
          ) : (
            <Button variant="secondary" onClick={stop}>
              <CameraOff className="h-4 w-4" />
              Stop
            </Button>
          )}
          {active && (
            <Button
              variant="secondary"
              onClick={() => setShowPip((v) => !v)}
              aria-pressed={showPip}
            >
              {showPip ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              {showPip ? "Hide preview" : "Show preview"}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
