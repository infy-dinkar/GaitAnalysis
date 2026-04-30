"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff } from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { Button } from "@/components/ui/Button";
import { analyzeLiveFrame, type LiveBiomechFrameDataDTO } from "@/lib/api";
import { LM } from "@/lib/pose/landmarks";

interface Props {
  bodyPart: "shoulder" | "neck";
  movement: string;
  side?: "left" | "right";
  onResult: (data: LiveBiomechFrameDataDTO | null) => void;
  onError?: (msg: string) => void;
  /** Target frames-per-second sent to the backend. Default 24.
   *  Self-throttled by the in-flight guard if the CPU can't keep up. */
  targetFps?: number;
}

// Bold-white skeleton lines + bold-red joint dots — high contrast on
// any background, mirrors clinical mocap convention.
const LINE_COLOR = "#FFFFFF";
const DOT_COLOR = "#EF4444";
// Threshold for the OVERLAY only — independent of the angle math
// (which lives in the Python engines now).
const OVERLAY_VIS_THRESHOLD = 0.4;

// Per-body-part overlay filters — only landmarks the engine uses get drawn.
const RELEVANT_DOTS: Record<"shoulder" | "neck", number[]> = {
  shoulder: [
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_WRIST, LM.RIGHT_WRIST,
    LM.LEFT_HIP, LM.RIGHT_HIP,
  ],
  neck: [
    LM.NOSE,
    LM.LEFT_EAR, LM.RIGHT_EAR,
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  ],
};

const RELEVANT_EDGES: Record<"shoulder" | "neck", Array<[number, number]>> = {
  shoulder: [
    [LM.LEFT_SHOULDER,  LM.LEFT_ELBOW],
    [LM.LEFT_ELBOW,     LM.LEFT_WRIST],
    [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW],
    [LM.RIGHT_ELBOW,    LM.RIGHT_WRIST],
    [LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER],
    [LM.LEFT_SHOULDER,  LM.LEFT_HIP],
    [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
    [LM.LEFT_HIP,       LM.RIGHT_HIP],
  ],
  neck: [
    [LM.LEFT_EAR,       LM.RIGHT_EAR],
    [LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER],
  ],
};

/**
 * Live biomech camera — server-side pose detection (Python MediaPipe).
 * Captures frames at `targetFps`, sends each as a JPEG to /api/live/
 * biomech-frame, draws the returned skeleton + invokes onResult with
 * angle + status. Same accuracy as Streamlit's biomech analyzer.
 */
export function LiveBiomechCamera({
  bodyPart,
  movement,
  side,
  onResult,
  onError,
  targetFps = 24,
}: Props) {
  const { videoRef, streamRef, active, error, start, stop } = useCamera();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const pipVideoRef = useRef<HTMLVideoElement | null>(null);
  const inFlightRef = useRef(false);
  const cancelledRef = useRef(false);
  const onResultRef = useRef(onResult);
  const onErrorRef = useRef(onError);
  const [networkError, setNetworkError] = useState<string | null>(null);

  // ── rAF interpolation snapshots ─────────────────────────────────
  // API gives ~20 FPS but the human eye perceives motion best at 60.
  // Two snapshots (prev + curr) are stored each time a new API response
  // arrives; an rAF loop redraws between them every browser frame so
  // the lines move continuously instead of stepping every 50 ms.
  const prevSnapshotRef = useRef<{
    landmarks: LiveBiomechFrameDataDTO["landmarks"];
    time: number;
  } | null>(null);
  const currSnapshotRef = useRef<{
    landmarks: LiveBiomechFrameDataDTO["landmarks"];
    time: number;
  } | null>(null);
  const rafRef = useRef<number | null>(null);

  const ingestSnapshot = useCallback(
    (landmarks: LiveBiomechFrameDataDTO["landmarks"]) => {
      const now = performance.now();
      prevSnapshotRef.current = currSnapshotRef.current;
      currSnapshotRef.current = { landmarks, time: now };
    },
    [],
  );

  useEffect(() => { onResultRef.current = onResult; }, [onResult]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);

  // Mirror the camera stream onto the PIP <video>.
  useEffect(() => {
    const pip = pipVideoRef.current;
    if (!pip) return;
    if (!active) {
      pip.srcObject = null;
      return;
    }
    const apply = () => {
      const stream = streamRef.current;
      if (stream && pip.srcObject !== stream) {
        pip.srcObject = stream;
        pip.play().catch(() => {});
      }
    };
    apply();
    const id = setTimeout(apply, 150);
    return () => clearTimeout(id);
  }, [active, streamRef]);

  // Skeleton drawing.
  const drawSkeleton = useCallback(
    (landmarks: LiveBiomechFrameDataDTO["landmarks"]) => {
      const overlay = overlayRef.current;
      const container = containerRef.current;
      if (!overlay || !container) return;

      const w = container.clientWidth;
      const h = container.clientHeight;
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

      const xy = (i: number) => ({
        x: landmarks[i].x * w,
        y: landmarks[i].y * h,
      });

      const edges = RELEVANT_EDGES[bodyPart];
      const dots = RELEVANT_DOTS[bodyPart];

      ctx.strokeStyle = LINE_COLOR;
      ctx.lineWidth = 3.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 6;
      ctx.shadowColor = "rgba(0, 0, 0, 0.55)";
      for (const [a, b] of edges) {
        const la = landmarks[a];
        const lb = landmarks[b];
        if (!la || !lb) continue;
        if (
          la.visibility < OVERLAY_VIS_THRESHOLD ||
          lb.visibility < OVERLAY_VIS_THRESHOLD
        ) continue;
        const pa = xy(a);
        const pb = xy(b);
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }

      ctx.shadowBlur = 0;
      ctx.fillStyle = DOT_COLOR;
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 1.8;
      for (const j of dots) {
        const lm = landmarks[j];
        if (!lm || lm.visibility < OVERLAY_VIS_THRESHOLD) continue;
        const p = xy(j);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    },
    [bodyPart],
  );

  // ── rAF redraw loop ────────────────────────────────────────────
  // Runs at the browser's natural ~60 Hz. Lerps between the two most
  // recent API snapshots based on time-since-prev / api-interval. When
  // we run out of forward data (rAF ahead of next API response) we
  // hold at the current snapshot — no extrapolation, no overshoot.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;

    const tick = () => {
      if (cancelled) return;
      const prev = prevSnapshotRef.current;
      const curr = currSnapshotRef.current;
      if (curr) {
        if (
          !prev ||
          prev.landmarks.length !== curr.landmarks.length ||
          curr.landmarks.length === 0
        ) {
          drawSkeleton(curr.landmarks);
        } else {
          const dt = curr.time - prev.time;
          const progress = dt > 0
            ? Math.min(1, (performance.now() - prev.time) / dt)
            : 1;
          const lerped = curr.landmarks.map((c, i) => {
            const p = prev.landmarks[i];
            if (!p) return c;
            return {
              x: p.x + progress * (c.x - p.x),
              y: p.y + progress * (c.y - p.y),
              visibility: c.visibility,
            };
          });
          drawSkeleton(lerped);
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [active, drawSkeleton]);

  // Streaming loop — capture → POST → ingestSnapshot.
  useEffect(() => {
    if (!active) {
      drawSkeleton([]);
      prevSnapshotRef.current = null;
      currSnapshotRef.current = null;
      return;
    }
    cancelledRef.current = false;
    const intervalMs = Math.max(1000 / targetFps, 30);

    const tick = async () => {
      if (cancelledRef.current) return;
      if (inFlightRef.current) return;

      const video = videoRef.current;
      if (!video || video.readyState < 2 || !video.videoWidth) return;

      inFlightRef.current = true;
      try {
        let canvas = offscreenRef.current;
        if (!canvas) {
          canvas = document.createElement("canvas");
          offscreenRef.current = canvas;
        }
        // 360 px is plenty for shoulder/neck — MediaPipe Pose Lite
        // resizes any input to 256² internally, so dropping below 480 px
        // mainly saves encode + upload time without hurting accuracy.
        const srcW = video.videoWidth;
        const srcH = video.videoHeight;
        const targetW = Math.min(srcW, 360);
        const targetH = Math.round((targetW / srcW) * srcH);
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        // Mirror the captured frame so MediaPipe sees the selfie-style
        // orientation that matches the visual mirror on screen.
        ctx.save();
        ctx.translate(targetW, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, targetW, targetH);
        ctx.restore();

        // JPEG quality 0.5 — pose detection is robust to compression noise.
        const blob = await new Promise<Blob | null>((resolve) =>
          canvas!.toBlob((b) => resolve(b), "image/jpeg", 0.5),
        );
        if (!blob || cancelledRef.current) return;

        const res = await analyzeLiveFrame({
          frame: blob,
          bodyPart,
          movement,
          side,
        });

        if (cancelledRef.current) return;

        if (res.success && res.data) {
          setNetworkError(null);
          // Feed the rAF interpolator instead of drawing directly — the
          // visual loop will lerp between this snapshot and the
          // previous one at 60 FPS for silky-smooth lines.
          ingestSnapshot(res.data.landmarks);
          onResultRef.current(res.data);
        } else {
          setNetworkError(res.error || "API error");
          onErrorRef.current?.(res.error || "API error");
          onResultRef.current(null);
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setNetworkError(msg);
        onErrorRef.current?.(msg);
        onResultRef.current(null);
      } finally {
        inFlightRef.current = false;
      }
    };

    const id = setInterval(tick, intervalMs);
    tick();
    return () => {
      cancelledRef.current = true;
      clearInterval(id);
    };
  }, [active, bodyPart, movement, side, targetFps, videoRef, drawSkeleton]);

  return (
    <div>
      <div
        ref={containerRef}
        className="relative aspect-video overflow-hidden rounded-card border border-border bg-[#0A0A0B]"
      >
        <video
          ref={videoRef}
          playsInline
          muted
          className="invisible h-full w-full object-cover"
        />
        <canvas
          ref={overlayRef}
          className="pointer-events-none absolute inset-0"
        />

        {active && (
          <div className="absolute bottom-3 right-3 aspect-video w-20 overflow-hidden rounded-lg border-2 border-white/30 shadow-lg sm:w-24 md:w-28">
            <video
              ref={pipVideoRef}
              playsInline
              muted
              autoPlay
              className="h-full w-full -scale-x-100 object-cover"
            />
          </div>
        )}

        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0A0A0B]/95 text-center">
            <Camera className="mb-3 h-10 w-10 text-white/40" />
            <p className="text-sm text-white/60">Camera is off</p>
          </div>
        )}
      </div>

      {(error || networkError) && (
        <p className="mt-3 text-xs text-error">{error ?? networkError}</p>
      )}

      <div className="mt-4 flex gap-3">
        {!active ? (
          <Button onClick={start}>
            <Camera className="h-4 w-4" />
            Start camera
          </Button>
        ) : (
          <Button variant="secondary" onClick={stop}>
            <CameraOff className="h-4 w-4" />
            Stop
          </Button>
        )}
      </div>
    </div>
  );
}
