"use client";
// HeightCalibrationStep — reusable height-based calibration UI.
//
// Drop this component into any test capture flow that needs px↔cm
// conversion (C6 Functional Reach today; D3 Single-Leg Hop, D4 CMJ
// to follow). The component:
//
//   1. Pre-fills the patient's height in cm (from the patient
//      record where available; otherwise a blank input).
//   2. Renders its own camera preview + skeleton overlay so it's
//      drop-in: no need for the consumer to wire a camera.
//   3. Continuously computes body pixel height. When the patient
//      is fully in frame AND the reading has been stable for ~1 s,
//      derives pixels_per_cm and surfaces a "Calibrated (height)"
//      lock-in banner.
//   4. Emits the resulting CalibrationResult through onCalibrated.
//   5. Always offers a "Skip" button — the test still runs in
//      relative-units mode if calibration is skipped (the explicit
//      graceful-degradation path the spec asks for).
//
// Important: this component owns its OWN camera lifecycle so the
// caller doesn't have to. Step 2 of the C6 flow runs this, then
// hands off to the test's own LiveCamera for the recording phase.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  CameraOff,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Ruler,
  SkipForward,
} from "lucide-react";

import { Button } from "@/components/ui/Button";
import { useCamera } from "@/hooks/useCamera";
import { usePoseDetectionLive as usePoseDetection } from "@/hooks/usePoseDetectionLive";
import {
  LM_LIVE as LM,
  SKELETON_EDGES_LIVE as SKELETON_EDGES,
} from "@/lib/pose/landmarks-live";
import {
  STABLE_FRAMES_REQUIRED,
  MIN_HEIGHT_CM,
  MAX_HEIGHT_CM,
  type BodyHeightReading,
  areReadingsStable,
  buildHeightCalibration,
  checkBodyInFrame,
  computeBodyPixelHeight,
} from "@/lib/calibration/heightCalibration";
import type { CalibrationResult } from "@/lib/calibration/types";
import type { Keypoint } from "@tensorflow-models/pose-detection";

const OVERLAY_VIS_THRESHOLD = 0.35;

interface Props {
  /** Pre-fill from patient record where available. */
  defaultHeightCm: number | null;
  /** Called when calibration is accepted (with a CalibrationResult)
   *  OR skipped (null = uncalibrated, relative units only). */
  onCalibrated: (result: CalibrationResult | null) => void;
  /** Render a "Skip" button. Default true. */
  allowSkip?: boolean;
}

interface Norm {
  x: number;
  y: number;
  visibility: number;
}

export function HeightCalibrationStep({
  defaultHeightCm,
  onCalibrated,
  allowSkip = true,
}: Props) {
  const { videoRef, active, error: camError, start, stop } = useCamera();
  const { ready: detectorReady, error: detectorError, detect } = usePoseDetection();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const recentReadingsRef = useRef<BodyHeightReading[]>([]);
  const lockedResultRef = useRef<CalibrationResult | null>(null);

  const [busy, setBusy] = useState(false);
  const [heightInput, setHeightInput] = useState<string>(
    defaultHeightCm && defaultHeightCm > 0
      ? defaultHeightCm.toFixed(0)
      : "",
  );

  // Sync when the parent patient record hydrates after mount — the
  // useState initializer captures whatever defaultHeightCm was at
  // mount time (often null while usePatientContext is still loading).
  // Only overwrite while the field is still empty so operator edits
  // are never clobbered.
  useEffect(() => {
    if (defaultHeightCm && defaultHeightCm > 0) {
      setHeightInput((prev) =>
        prev === "" ? defaultHeightCm.toFixed(0) : prev,
      );
    }
  }, [defaultHeightCm]);
  const [latestReading, setLatestReading] = useState<BodyHeightReading | null>(null);
  const [frameReason, setFrameReason] = useState<string>("torso_missing");
  const [stableCount, setStableCount] = useState<number>(0);
  const [locked, setLocked] = useState<boolean>(false);

  const heightCm = Number.parseFloat(heightInput);
  const heightCmValid =
    Number.isFinite(heightCm) &&
    heightCm >= MIN_HEIGHT_CM &&
    heightCm <= MAX_HEIGHT_CM;

  // Stop the detection loop when leaving the component.
  useEffect(() => {
    cancelledRef.current = false;
    return () => {
      cancelledRef.current = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Release the camera on unmount.
  useEffect(() => () => stop(), [stop]);

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
      const p = landmarks[a]; const q = landmarks[b];
      if (!p || !q ||
          p.visibility < OVERLAY_VIS_THRESHOLD ||
          q.visibility < OVERLAY_VIS_THRESHOLD) continue;
      const A = px(p); const B = px(q);
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
      ctx.stroke();
    }
    ctx.fillStyle = "#22c55e";
    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      if (!p || p.visibility < OVERLAY_VIS_THRESHOLD) continue;
      const r = Math.max(3, w * 0.004);
      const A = px(p);
      ctx.beginPath();
      ctx.arc(A.x, A.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }, []);

  // Per-frame: compute body pixel height, check stability, lock in if all gates pass.
  useEffect(() => {
    if (!active || !detectorReady) {
      drawSkeleton(null);
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
        if (kp) {
          const sw = video.videoWidth;
          const sh = video.videoHeight;
          const norm: Norm[] = kp.map((p) => ({
            x: 1 - p.x / sw,
            y: p.y / sh,
            visibility: p.score ?? 0,
          }));
          drawSkeleton(norm);

          const reading = computeBodyPixelHeight(kp);
          setLatestReading(reading);
          const frameCheck = checkBodyInFrame(kp, reading, sh);
          setFrameReason(frameCheck.ok ? "" : frameCheck.reason);

          if (locked) {
            // already done — no further sampling needed
          } else if (frameCheck.ok && reading && heightCmValid) {
            // Stability gate
            const recent = recentReadingsRef.current;
            if (recent.length > 0) {
              const last = recent[recent.length - 1];
              if (!areReadingsStable(last, reading)) {
                recent.length = 0;
              }
            }
            recent.push(reading);
            if (recent.length > STABLE_FRAMES_REQUIRED) recent.shift();
            setStableCount(recent.length);

            if (recent.length >= STABLE_FRAMES_REQUIRED) {
              // Lock in: median pixel height across the window
              const sorted = [...recent]
                .map((r) => r.body_pixel_height_px)
                .sort((a, b) => a - b);
              const median = sorted[Math.floor(sorted.length / 2)];
              const calibration = buildHeightCalibration(
                median,
                heightCm,
                { width: sw, height: sh },
              );
              if (calibration) {
                lockedResultRef.current = calibration;
                setLocked(true);
              }
            }
          } else {
            recentReadingsRef.current = [];
            setStableCount(0);
          }
        } else {
          drawSkeleton(null);
        }
      } catch {
        // ignore individual detect errors
      }
      if (!cancelledRef.current) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelledRef.current = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [active, detectorReady, detect, drawSkeleton, videoRef, heightCmValid, heightCm, locked]);

  async function handleStartCamera() {
    setBusy(true);
    try { await start(); } finally { setBusy(false); }
  }

  function handleRetake() {
    recentReadingsRef.current = [];
    setStableCount(0);
    setLocked(false);
    lockedResultRef.current = null;
  }

  function handleConfirm() {
    const r = lockedResultRef.current;
    if (!r) return;
    stop();
    onCalibrated(r);
  }

  function handleSkip() {
    stop();
    onCalibrated(null);
  }

  // ── Coaching message ─────────────────────────────────────────
  const coachingMessage = (() => {
    if (locked) return null;
    if (!active) return "Start the camera. Patient should stand straight, full body in frame, facing the camera.";
    if (!heightCmValid) return `Enter the patient's height in cm (${MIN_HEIGHT_CM}–${MAX_HEIGHT_CM}).`;
    switch (frameReason) {
      case "torso_missing": return "Body not yet tracked — make sure the patient's torso is in frame.";
      case "head_missing":  return "Head not visible — raise the camera or step the patient closer.";
      case "feet_missing":  return "Feet not visible — lower the camera or step the patient back.";
      case "body_partial":  return "Patient not fully in frame yet.";
      case "head_at_frame_edge": return "Head is cropped at the top — patient should step back or camera should be raised.";
      case "feet_at_frame_edge": return "Feet are cropped at the bottom — patient should step back or camera should be lowered.";
      default:               return `Stand straight, full body in frame — holding for stable reading (${stableCount}/${STABLE_FRAMES_REQUIRED}).`;
    }
  })();

  return (
    <div className="space-y-5">
      <div className="rounded-card border border-border bg-surface p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Calibration · height-based
        </p>
        <h2 className="mt-2 text-lg font-semibold tracking-tight text-foreground">
          Patient height scale calibration
        </h2>
        <p className="mt-2 text-sm text-muted">
          Enter the patient&apos;s standing height (cm). Patient stands
          straight with their full body in frame; the system measures pixel
          height and computes a pixels-per-cm scale so reach is reported in
          centimetres. Skipping is allowed — the test still runs but
          distances are reported in relative pixel units only.
        </p>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[3fr_2fr]">
        {/* Camera preview */}
        <div>
          <div
            ref={containerRef}
            className="relative aspect-video overflow-hidden rounded-card border border-border bg-gradient-to-br from-[#0A0A0B] via-[#0d0d10] to-[#15151a]"
          >
            <video
              ref={videoRef}
              playsInline
              muted
              className={`block h-full w-full -scale-x-100 object-cover transition-opacity duration-200 ${active ? "opacity-100" : "opacity-0"}`}
            />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0" />
            {!active && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                <Ruler className="mb-3 h-10 w-10 text-white/40" />
                <p className="text-sm text-white/60">Camera is off — start it to begin</p>
              </div>
            )}
            {locked && (
              <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-emerald-500/95 px-3 py-1.5 text-xs font-medium text-white shadow-lg">
                <CheckCircle2 className="h-4 w-4" />
                Calibrated (height)
              </div>
            )}
            {active && detectorError && (
              <div className="absolute inset-x-3 top-3 mx-auto flex max-w-md items-start gap-2 rounded-md bg-error/90 px-3 py-2 text-xs text-white">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>Pose model failed: {detectorError}</span>
              </div>
            )}
          </div>

          <div className="mt-4 flex flex-wrap gap-3">
            {!active && !locked && (
              <Button onClick={handleStartCamera} disabled={busy}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
                Start camera
              </Button>
            )}
            {active && !locked && (
              <Button variant="secondary" onClick={() => { stop(); }}>
                <CameraOff className="h-4 w-4" />
                Stop camera
              </Button>
            )}
            {locked && (
              <>
                <Button onClick={handleConfirm}>
                  <CheckCircle2 className="h-4 w-4" />
                  Use this calibration
                </Button>
                <Button variant="secondary" onClick={handleRetake}>
                  <RefreshCw className="h-4 w-4" />
                  Re-take
                </Button>
              </>
            )}
            {allowSkip && !locked && (
              <Button variant="ghost" onClick={handleSkip}>
                <SkipForward className="h-4 w-4" />
                Skip calibration (use relative units)
              </Button>
            )}
          </div>
          {camError && <p className="mt-3 text-xs text-error">{camError}</p>}
        </div>

        {/* Right column — height input + status */}
        <div className="space-y-4">
          <div className="rounded-card border border-border bg-surface p-4">
            <label className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Patient height (cm)
            </label>
            <input
              type="number"
              min={MIN_HEIGHT_CM}
              max={MAX_HEIGHT_CM}
              step={0.5}
              value={heightInput}
              onChange={(e) => setHeightInput(e.target.value)}
              placeholder="e.g. 170"
              disabled={locked}
              className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-base tabular text-foreground outline-none focus:border-accent disabled:opacity-60"
            />
            {!heightCmValid && heightInput.length > 0 && (
              <p className="mt-1 text-xs text-error">
                Enter a value between {MIN_HEIGHT_CM} and {MAX_HEIGHT_CM} cm.
              </p>
            )}
            {defaultHeightCm && defaultHeightCm > 0 && (
              <p className="mt-1 text-xs text-muted">
                Pre-filled from patient record: {defaultHeightCm.toFixed(0)} cm.
              </p>
            )}
          </div>

          <div
            className={`rounded-card border p-4 text-sm ${
              locked
                ? "border-emerald-500/30 bg-emerald-500/5"
                : "border-border bg-surface"
            }`}
          >
            <div className="flex items-start gap-2">
              {locked ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
              ) : (
                <Loader2
                  className={`mt-0.5 h-4 w-4 shrink-0 ${
                    active && heightCmValid ? "animate-spin text-amber-600" : "text-muted"
                  }`}
                />
              )}
              <div>
                <p className="font-medium text-foreground">
                  {locked
                    ? "Calibration locked"
                    : "Acquiring stable reading…"}
                </p>
                {coachingMessage && (
                  <p className="mt-1 text-xs text-muted">{coachingMessage}</p>
                )}
              </div>
            </div>
            {!locked && (
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                <div
                  className="h-full bg-accent transition-all"
                  style={{ width: `${(stableCount / STABLE_FRAMES_REQUIRED) * 100}%` }}
                />
              </div>
            )}
          </div>

          {(latestReading || lockedResultRef.current) && (
            <div className="rounded-card border border-border bg-surface p-4 text-sm">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Scale
              </p>
              {lockedResultRef.current ? (
                <>
                  <p className="mt-2 tabular text-foreground">
                    <span className="font-semibold">{lockedResultRef.current.pixels_per_cm.toFixed(2)}</span>{" "}
                    px per cm
                  </p>
                  <p className="mt-1 text-xs text-muted">
                    Body pixel height ={" "}
                    {lockedResultRef.current.body_pixel_height_px?.toFixed(0)} px ·
                    height = {lockedResultRef.current.patient_height_cm?.toFixed(0)} cm
                  </p>
                </>
              ) : latestReading ? (
                <p className="mt-2 tabular text-muted">
                  Current body pixel height ={" "}
                  {latestReading.body_pixel_height_px.toFixed(0)} px (not yet stable)
                </p>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
