"use client";
// Generic live-camera + skeleton overlay shell for the rehab module.
// Clones the structure of SingleLegHopLiveCamera.tsx — same useCamera
// + usePoseDetectionLive singleton, same rAF loop, same skeleton-on-
// dark-canvas presentation — but is EXERCISE-AGNOSTIC. It just
// exposes per-frame BlazePose landmarks via the `onFrame` callback.
//
// The future exercise components plug in by:
//   1. Mounting this shell
//   2. Reading landmarks in their onFrame callback
//   3. Computing the input signal (e.g. knee angle via
//      lib/biomech/knee-live.ts:computeKneeAngle — IMPORTED, NOT
//      MODIFIED)
//   4. Passing the signal into one of the 7 mechanic shells as a prop
//
// Zero modifications to the existing pose pipeline — usePoseDetectionLive,
// useCamera, and the *-live.ts math files are all imported as-is.

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
import {
  drawAngleArc,
  drawCenterline,
  drawSpineSegment,
  type AngleArcConfig,
} from "@/lib/pose/skeletonExtras";
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

// Face landmarks — rendered smaller than the body joints so the
// nose + eyes + ears cluster reads as a light head anchor rather
// than a busy blob of big dots. Kept the same colour + glow as the
// body joints so the palette stays consistent.
const FACE_DOTS = new Set<number>([
  LM.NOSE, LM.LEFT_EYE, LM.RIGHT_EYE, LM.LEFT_EAR, LM.RIGHT_EAR,
]);

// Torso side edges (shoulder→hip verticals) — skipped in the bone
// loop so the shoulders + hips no longer close into a rectangle.
// The horizontal shoulder-shoulder and hip-hip bars stay; the
// vertebra chain drawn by drawSpineSegment fills the middle.
const TORSO_SIDE_EDGES = new Set<string>([
  `${LM.LEFT_SHOULDER}-${LM.LEFT_HIP}`,
  `${LM.RIGHT_SHOULDER}-${LM.RIGHT_HIP}`,
]);

// Side-coded palette matching the clinician's reference:
//   • patient's LEFT limb  → cyan   (on-screen left in mirror view)
//   • patient's RIGHT limb → pink   (on-screen right)
//   • centre column        → orange (shoulder + hip crossbars, spine)
// Face landmarks are handled separately below — kept small + orange
// per the "face ko chod ke" instruction so the head cluster stays
// unchanged from the prior turn.
const LEFT_COLOR = "#38BDF8";              // sky-400
const LEFT_GLOW = "rgba(56, 189, 248, 0.55)";
const RIGHT_COLOR = "#F87171";             // red-400
const RIGHT_GLOW = "rgba(248, 113, 113, 0.55)";
const CENTER_COLOR = "#F97316";            // orange-500
const CENTER_GLOW = "rgba(249, 115, 22, 0.55)";
// Face palette (unchanged from prior turn — small orange dots).
const FACE_COLOR = "#F97316";
const FACE_GLOW = "rgba(249, 115, 22, 0.55)";

// Side classifier from a landmark index.
const LEFT_BODY_LMS = new Set<number>([
  LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST,
  LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE,
  LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX,
]);
const RIGHT_BODY_LMS = new Set<number>([
  LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,
  LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE,
  LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX,
]);
function sideOf(idx: number): "left" | "right" | "face" | "center" {
  if (LEFT_BODY_LMS.has(idx)) return "left";
  if (RIGHT_BODY_LMS.has(idx)) return "right";
  return "center"; // face is handled through its own dot set
}
function boneStyle(a: number, b: number): { stroke: string; glow: string } {
  const sa = sideOf(a);
  const sb = sideOf(b);
  if (sa === "left" && sb === "left") return { stroke: LEFT_COLOR, glow: LEFT_GLOW };
  if (sa === "right" && sb === "right") return { stroke: RIGHT_COLOR, glow: RIGHT_GLOW };
  return { stroke: CENTER_COLOR, glow: CENTER_GLOW };
}

interface Norm {
  x: number;
  y: number;
  visibility: number;
}

export interface RehabCameraShellProps {
  /** Called every frame with the latest 33-keypoint BlazePose
   *  result. The shell does NOT compute any exercise metric — it
   *  just exposes the raw landmarks so the mounting component can
   *  derive whatever signal the chosen mechanic needs. */
  onFrame: (keypoints: Keypoint[], video: HTMLVideoElement) => void;
  /** Optional overlay drawn ON TOP of the skeleton (e.g. a target
   *  band, a cursor, a path) — rendered as absolutely-positioned
   *  children. */
  children?: React.ReactNode;
  onError?: (msg: string) => void;
  /** Optional angle-arc overlay drawn on the canvas at a joint
   *  vertex. When present, an arc + labelled degree chip is
   *  rendered AFTER the skeleton, centerline, and spine — reads
   *  the vertex + arm indices from LM_LIVE, and the currentDeg
   *  from the page's per-frame handler (the page still owns the
   *  math; the shell just renders). Omit to keep the existing
   *  overlay unchanged. */
  angleArc?: AngleArcConfig;
  /** Fill the parent container instead of using a fixed 16:9 aspect.
   *  Used inside LiveModeLayout where the shell must expand to the
   *  entire left half of the viewport. */
  fill?: boolean;
  /** Auto-request the camera on mount instead of waiting for the
   *  operator to click "Start camera". Hides the Start button while
   *  the browser resolves the permission prompt. Used by the reduced-
   *  click doctor flow — parent has already gated this behind a
   *  user gesture (e.g. side pick) so autoplay policies pass. */
  autoStart?: boolean;
  /** Hide the Start/Stop button row entirely. Useful when the parent
   *  owns session control (auto-start + no manual stop). */
  hideControls?: boolean;
}

export function RehabCameraShell({
  onFrame,
  onError,
  children,
  angleArc,
  fill = false,
  autoStart = false,
  hideControls = false,
}: RehabCameraShellProps) {
  const { videoRef, active, error: camError, start, stop } = useCamera();
  const { ready: detectorReady, error: detectorError, detect } =
    usePoseDetection();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const rafRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const lastNormRef = useRef<Norm[] | null>(null);
  const onFrameRef = useRef(onFrame);
  // Latest angleArc mirrored into a ref so the memoised
  // drawSkeleton closure (with [] deps) picks up new values
  // without needing to rebind on every prop change.
  const angleArcRef = useRef<AngleArcConfig | undefined>(angleArc);

  const [busy, setBusy] = useState(false);

  useEffect(() => { onFrameRef.current = onFrame; }, [onFrame]);
  useEffect(() => { angleArcRef.current = angleArc; }, [angleArc]);
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

    // ── Bones — side-coded strokes with matching glow. Torso sides
    //    still skipped so the shoulders + hips don't close into a
    //    rectangle across the trunk.
    ctx.lineWidth = Math.max(3, w * 0.005);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowBlur = 12;
    for (const [a, b] of SKELETON_EDGES) {
      const key = a < b ? `${a}-${b}` : `${b}-${a}`;
      if (TORSO_SIDE_EDGES.has(key)) continue;
      const p = landmarks[a];
      const q = landmarks[b];
      if (
        !p || !q
        || p.visibility < OVERLAY_VIS_THRESHOLD
        || q.visibility < OVERLAY_VIS_THRESHOLD
      ) continue;
      const style = boneStyle(a, b);
      ctx.strokeStyle = style.stroke;
      ctx.shadowColor = style.glow;
      const A = px(p); const B = px(q);
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
      ctx.stroke();
    }
    // ── Joint dots — side-coded fills with matching glow. Face
    //    landmarks (nose/eyes/ears) stay small orange per the
    //    "face ko chod ke" instruction.
    ctx.shadowBlur = 10;
    const bodyDotR = Math.max(6, w * 0.009);
    const faceDotR = Math.max(3, w * 0.004);
    for (const i of FULL_BODY_DOTS) {
      const p = landmarks[i];
      if (!p || p.visibility < OVERLAY_VIS_THRESHOLD) continue;
      const A = px(p);
      if (FACE_DOTS.has(i)) {
        ctx.fillStyle = FACE_COLOR;
        ctx.shadowColor = FACE_GLOW;
        ctx.beginPath();
        ctx.arc(A.x, A.y, faceDotR, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      const s = sideOf(i);
      if (s === "left") {
        ctx.fillStyle = LEFT_COLOR;
        ctx.shadowColor = LEFT_GLOW;
      } else if (s === "right") {
        ctx.fillStyle = RIGHT_COLOR;
        ctx.shadowColor = RIGHT_GLOW;
      } else {
        ctx.fillStyle = CENTER_COLOR;
        ctx.shadowColor = CENTER_GLOW;
      }
      ctx.beginPath();
      ctx.arc(A.x, A.y, bodyDotR, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;

    // ── ViFive-style overlay extras (additive) ───────────────────
    // Rendered AFTER the classic skeleton so guides sit on top as
    // annotation rather than under the bones. Each helper is
    // visibility-gated internally and no-ops on degenerate frames,
    // so absence of any required landmark simply skips that guide.
    drawCenterline(ctx, landmarks, w, h, {
      visibilityThreshold: OVERLAY_VIS_THRESHOLD,
    });
    drawSpineSegment(ctx, landmarks, w, h, {
      visibilityThreshold: OVERLAY_VIS_THRESHOLD,
    });
    const arc = angleArcRef.current;
    if (arc) {
      drawAngleArc(ctx, landmarks, w, h, {
        ...arc,
        visibilityThreshold:
          arc.visibilityThreshold ?? OVERLAY_VIS_THRESHOLD,
      });
    }
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

  // Auto-start: fires exactly once on mount when the parent opted in.
  // Guarded by a ref so a re-render (or React 18 strict-mode double
  // effect) never re-triggers the camera request after the operator
  // has manually stopped it.
  const autoStartFiredRef = useRef(false);
  useEffect(() => {
    if (!autoStart || autoStartFiredRef.current) return;
    autoStartFiredRef.current = true;
    setBusy(true);
    start()
      .catch(() => {
        // useCamera surfaces its own error via camError → onError; no
        // extra handling needed here beyond dropping the busy flag.
      })
      .finally(() => setBusy(false));
  }, [autoStart, start]);

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
            ? "relative min-h-0 w-full flex-1 overflow-hidden rounded-card border border-border bg-black"
            : "relative aspect-video overflow-hidden rounded-card border border-border bg-black"
        }
      >
        {/* Live camera fills the tile. Mirrored horizontally so the
            selfie orientation matches the mirrored landmark space
            (`1 - p.x / sw`). object-cover keeps aspect + fills. */}
        <video
          ref={videoRef}
          playsInline
          muted
          className={`absolute inset-0 h-full w-full -scale-x-100 object-cover transition-opacity duration-200 ${
            active ? "opacity-100" : "opacity-0"
          }`}
        />
        {/* Skeleton canvas overlays the video. Transparent — the
            body shows through everywhere the skeleton isn't drawn. */}
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0" />
        {/* Game overlay slot — exercises render their target zones,
            cursors, paths, etc. on top of the skeleton via children. */}
        <div className="pointer-events-none absolute inset-0">{children}</div>
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
