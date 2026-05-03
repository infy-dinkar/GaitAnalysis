"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Loader2, AlertCircle, Eye, EyeOff } from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { usePoseDetection } from "@/hooks/usePoseDetection";
import { Button } from "@/components/ui/Button";
import type { LiveBiomechFrameDataDTO } from "@/lib/api";
import { LM } from "@/lib/pose/landmarks";
import {
  computeShoulderAngle,
  type ShoulderMovementId,
} from "@/lib/biomech/shoulder";
import { computeNeckAngle, type NeckMovementId } from "@/lib/biomech/neck";
import { computeKneeAngle, type KneeMovementId } from "@/lib/biomech/knee";
import { computeHipAngle, type HipMovementId } from "@/lib/biomech/hip";
import { computeAnkleAngle, type AnkleMovementId } from "@/lib/biomech/ankle";

export type BiomechBodyPart = "shoulder" | "neck" | "knee" | "hip" | "ankle";

interface Props {
  bodyPart: BiomechBodyPart;
  movement: string;
  side?: "left" | "right";
  onResult: (data: LiveBiomechFrameDataDTO | null) => void;
  onError?: (msg: string) => void;
}

const LINE_COLOR = "#FFFFFF";
const DOT_COLOR = "#EF4444";
// BlazePose-tfjs `score` for a confidently visible joint is typically
// 0.4-0.99, but partially-occluded joints can drop to 0.1-0.2. We draw
// any keypoint the model is at least minimally confident about so the
// skeleton doesn't blink in/out. The math layer applies its own (lower)
// threshold for angle compute.
const OVERLAY_VIS_THRESHOLD = 0.1;

// Face points always shown for visual completeness — even when the
// movement math doesn't strictly need them, they help the user see
// the full upper-body skeleton. Dots only, no edges — connecting
// face landmarks to anything else creates awkward angles because
// these points sit at similar heights and the head naturally floats
// just above the shoulders without needing visual tethering.
const FACE_DOTS = [
  LM.NOSE,
  LM.LEFT_EYE, LM.RIGHT_EYE,
  LM.LEFT_EAR, LM.RIGHT_EAR,
];

const FACE_EDGES: Array<[number, number]> = [];

const LOWER_BODY_DOTS = [
  LM.LEFT_HIP, LM.RIGHT_HIP,
  LM.LEFT_KNEE, LM.RIGHT_KNEE,
  LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
];

const LOWER_BODY_EDGES: Array<[number, number]> = [
  [LM.LEFT_HIP,   LM.RIGHT_HIP],
  [LM.LEFT_HIP,   LM.LEFT_KNEE],
  [LM.LEFT_KNEE,  LM.LEFT_ANKLE],
  [LM.RIGHT_HIP,  LM.RIGHT_KNEE],
  [LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
];

const RELEVANT_DOTS: Record<BiomechBodyPart, number[]> = {
  shoulder: [
    ...FACE_DOTS,
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_WRIST, LM.RIGHT_WRIST,
    LM.LEFT_HIP, LM.RIGHT_HIP,
  ],
  neck: [
    ...FACE_DOTS,
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  ],
  knee: [
    ...LOWER_BODY_DOTS,
  ],
  hip: [
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    ...LOWER_BODY_DOTS,
  ],
  ankle: [
    ...LOWER_BODY_DOTS,
  ],
};

const RELEVANT_EDGES: Record<BiomechBodyPart, Array<[number, number]>> = {
  shoulder: [
    ...FACE_EDGES,
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
    ...FACE_EDGES,
    [LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER],
  ],
  knee: [...LOWER_BODY_EDGES],
  hip: [
    [LM.LEFT_SHOULDER, LM.LEFT_HIP],
    [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
    [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER],
    ...LOWER_BODY_EDGES,
  ],
  ankle: [...LOWER_BODY_EDGES],
};

interface Norm {
  x: number;
  y: number;
  visibility: number;
}

/**
 * Live biomech camera — fully client-side pose detection.
 *
 * Pipeline:
 *   webcam stream → BlazePose-tfjs (WebGL) → angle math → overlay
 *   Runs at the browser's natural rAF rate (~30-60 FPS depending on
 *   hardware). No HTTP per frame. No backend involvement during live
 *   capture — only `Show Analysis` records the locally-tracked peak.
 *
 * Layout:
 *   • <video> element shown as the primary view (CSS-mirrored selfie)
 *   • <canvas> overlay (also CSS-mirrored) draws the skeleton on top
 *   • Math operates on raw keypoints — formulas are scale-invariant
 *
 * Server-side endpoint /api/live/biomech-frame stays untouched as a
 * fallback if you ever want to switch back; it is simply not called.
 */
export function LiveBiomechCamera({
  bodyPart,
  movement,
  side,
  onResult,
  onError,
}: Props) {
  const { videoRef, active, error, start, stop } = useCamera();
  const { ready, error: poseError, detect } = usePoseDetection();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const onResultRef = useRef(onResult);
  const onErrorRef = useRef(onError);

  // Per-keypoint EMA buffer to smooth jitter. Lower alpha = heavier
  // smoothing (more lag), higher = lighter (more jitter). 0.4 is a
  // sweet spot at 30-60 FPS — visibly smooths without feeling laggy.
  const smoothedKpRef = useRef<{ x: number; y: number; score: number }[] | null>(null);

  const [showPip, setShowPip] = useState(true);

  useEffect(() => { onResultRef.current = onResult; }, [onResult]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);

  // Skeleton drawing — direct canvas, no React state churn.
  const drawSkeleton = useCallback(
    (landmarks: Norm[]) => {
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

      const xy = (i: number) => ({
        x: landmarks[i].x * w,
        y: landmarks[i].y * h,
      });

      const edges = RELEVANT_EDGES[bodyPart];
      const dots = RELEVANT_DOTS[bodyPart];

      // edges — soft white glow on dark bg
      ctx.strokeStyle = LINE_COLOR;
      ctx.lineWidth = 3.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 10;
      ctx.shadowColor = "rgba(255, 255, 255, 0.35)";
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
        if (!isFinite(pa.x) || !isFinite(pb.x)) continue;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }

      // dots — accent red with subtle glow + white ring
      ctx.shadowBlur = 8;
      ctx.shadowColor = "rgba(239, 68, 68, 0.55)";
      ctx.fillStyle = DOT_COLOR;
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 2;
      for (const j of dots) {
        const lm = landmarks[j];
        if (!lm || lm.visibility < OVERLAY_VIS_THRESHOLD) continue;
        const p = xy(j);
        if (!isFinite(p.x) || !isFinite(p.y)) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
    },
    [bodyPart],
  );

  // ── rAF detection loop ─────────────────────────────────────────
  useEffect(() => {
    if (!active || !ready) {
      drawSkeleton([]);
      smoothedKpRef.current = null;
      return;
    }
    cancelledRef.current = false;
    smoothedKpRef.current = null;

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

        if (!pose) {
          drawSkeleton([]);
          smoothedKpRef.current = null;
          onResultRef.current({
            status: "no_landmarks",
            landmarks: [],
            current_angle: null,
            current_magnitude: 0,
          });
        } else {
          // ── EMA smoothing on raw pixel keypoints to kill jitter ──
          // alpha = 0.4: fresh frame contributes 40%, history 60%.
          // Visibly smooth at 30-60 FPS without feeling laggy.
          const ALPHA = 0.4;
          const raw = pose.keypoints;
          const prev = smoothedKpRef.current;
          const smoothed =
            prev && prev.length === raw.length
              ? raw.map((kp, i) => ({
                  x: ALPHA * kp.x + (1 - ALPHA) * prev[i].x,
                  y: ALPHA * kp.y + (1 - ALPHA) * prev[i].y,
                  score: kp.score ?? 0,
                }))
              : raw.map((kp) => ({
                  x: kp.x,
                  y: kp.y,
                  score: kp.score ?? 0,
                }));
          smoothedKpRef.current = smoothed;

          // Pose-detection Keypoint shape with smoothed coords —
          // angle math reads x/y/score so this is a drop-in replacement.
          const smoothedKps = smoothed.map((s) => ({
            x: s.x,
            y: s.y,
            score: s.score,
          }));

          const sw = video.videoWidth;
          const sh = video.videoHeight;
          // Normalised landmarks for overlay (resolution-independent)
          const norm: Norm[] = smoothed.map((kp) => ({
            x: kp.x / sw,
            y: kp.y / sh,
            visibility: kp.score,
          }));

          // Math uses raw pixel coords — formulas are scale-invariant.
          let angle: number | null = null;
          const sideOrRight = side ?? "right";
          switch (bodyPart) {
            case "shoulder":
              angle = computeShoulderAngle(
                movement as ShoulderMovementId,
                smoothedKps,
                sideOrRight,
              );
              break;
            case "neck":
              angle = computeNeckAngle(
                movement as NeckMovementId,
                smoothedKps,
              );
              break;
            case "knee":
              angle = computeKneeAngle(
                movement as KneeMovementId,
                smoothedKps,
                sideOrRight,
              );
              break;
            case "hip":
              angle = computeHipAngle(
                movement as HipMovementId,
                smoothedKps,
                sideOrRight,
              );
              break;
            case "ankle":
              angle = computeAnkleAngle(
                movement as AnkleMovementId,
                smoothedKps,
                sideOrRight,
              );
              break;
          }

          drawSkeleton(norm);

          onResultRef.current({
            status: angle !== null ? "good" : "low_visibility",
            landmarks: norm,
            current_angle: angle,
            current_magnitude: angle !== null ? Math.abs(angle) : 0,
          });
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        onErrorRef.current?.(msg);
      }
      if (!cancelledRef.current) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      cancelledRef.current = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [active, ready, detect, bodyPart, movement, side, drawSkeleton, videoRef]);

  return (
    <div>
      <div
        ref={containerRef}
        className="relative aspect-video overflow-hidden rounded-card border border-border bg-gradient-to-br from-[#0A0A0B] via-[#0d0d10] to-[#15151a]"
      >
        {/* Skeleton canvas — primary view on dark bg. CSS-mirrored so
            movement direction matches the PiP selfie preview. */}
        <canvas
          ref={overlayRef}
          className="pointer-events-none absolute inset-0 -scale-x-100"
        />

        {/* PiP camera preview — small mirrored selfie inset (bottom-right
            corner). Video element is always in the DOM so the ref stays
            stable; we fade the wrapper when the camera is off OR when
            the user has toggled the preview off. */}
        <div
          className={`absolute bottom-3 right-3 z-10 overflow-hidden rounded-lg border border-white/15 bg-black/80 shadow-2xl transition-opacity duration-200 ${
            active && showPip ? "opacity-100" : "pointer-events-none opacity-0"
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

        {/* Camera off placeholder */}
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <Camera className="mb-3 h-10 w-10 text-white/40" />
            <p className="text-sm text-white/60">Camera is off</p>
          </div>
        )}

        {/* TF.js model loading */}
        {active && !ready && !poseError && (
          <div className="absolute inset-x-0 top-3 mx-auto flex w-fit items-center gap-2 rounded-full bg-black/75 px-4 py-1.5 text-xs text-white/85 backdrop-blur">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading pose model…
          </div>
        )}

        {/* Pose model load error */}
        {active && poseError && (
          <div className="absolute inset-x-3 top-3 mx-auto flex max-w-md items-start gap-2 rounded-md bg-error/90 px-3 py-2 text-xs text-white">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>Pose model failed: {poseError}</span>
          </div>
        )}

      </div>

      {error && <p className="mt-3 text-xs text-error">{error}</p>}

      <div className="mt-4 flex flex-wrap gap-3">
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
    </div>
  );
}
