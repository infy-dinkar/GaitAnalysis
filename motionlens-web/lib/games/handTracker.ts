// Single-hand tracking state shared between the React pose loop and the
// Phaser scene.
//
// Deliberately a MUTABLE object rather than React state: the pose loop
// runs at camera rate and Phaser renders on its own rAF, so routing a
// position through setState would re-render the tree ~30x/s for nothing.
// React reads it only for low-frequency UI (the readiness ticks).

import { LM_LIVE } from "@/lib/pose/landmarks-live";

export type Hand = "left" | "right";

/**
 * Which BlazePose landmark drives the cursor, per hand choice.
 *
 * These indices are ANATOMICAL — landmark 16 is the patient's own right
 * wrist regardless of how the picture is drawn. The live shells mirror
 * the frame for display (`x = 1 - p.x / videoWidth`, see
 * RehabCameraShell.tsx:358), which is a selfie view: the patient's right
 * hand ends up on the RIGHT of the screen, exactly as it would in a
 * mirror. Mirroring changes only where the point is drawn, never which
 * index it came from — so "Right" maps to RIGHT_WRIST and nothing else
 * needs to flip.
 */
export const WRIST_INDEX: Record<Hand, number> = {
  left: LM_LIVE.LEFT_WRIST, // 15
  right: LM_LIVE.RIGHT_WRIST, // 16
};

/** Same-side elbow, used to project the wrist forward to the palm. */
export const ELBOW_INDEX: Record<Hand, number> = {
  left: LM_LIVE.LEFT_ELBOW, // 13
  right: LM_LIVE.RIGHT_ELBOW, // 14
};

/**
 * How far past the wrist the palm sits, as a fraction of the forearm.
 *
 * BlazePose gives a wrist, but the ✋ cursor reads as a palm, so drawing
 * it on the wrist puts it visibly short of the hand. Extending along
 * the forearm by this much lands it near the middle of the palm.
 *
 * Computed in CANVAS pixels, not normalised space: normalised space is
 * anisotropic whenever the video is not square (dispW != dispH), so
 * extrapolating a direction there would skew it.
 */
export const PALM_REACH = 0.3;

/** Below this visibility a landmark is treated as not present. Matches
 *  the value the production camera shells use for their overlay. */
export const VIS_FLOOR = 0.35;

/** Half-frame slack allowed before a landmark counts as out of frame.
 *  MediaPipe extrapolates joints it cannot see and still reports a
 *  passing visibility, so `live` alone is not enough to trust a point. */
export const FRAME_SLACK = 0.05;

export interface HandState {
  /** Normalised MIRRORED video space, 0..1 across the frame. */
  nx: number;
  ny: number;
  /** WRIST in canvas-space pixels, after object-cover compensation.
   *  Calibration and the reach box are defined in wrist terms, so this
   *  keeps its original meaning. */
  x: number;
  y: number;
  /** PALM in canvas-space pixels — the wrist projected along the
   *  forearm. This is what the cursor is drawn at and what the hit test
   *  uses, so what the patient sees is what collects the fruit.
   *  Falls back to the wrist when the elbow is not usable. */
  palmX: number;
  palmY: number;
  /** True when palmX/palmY came from a live, in-frame elbow. */
  palmFromElbow: boolean;
  score: number;
  /** Visibility passed the floor. */
  live: boolean;
  /** Inside the frame (plus slack). */
  inFrame: boolean;
  /** live && inFrame — the only condition under which x/y may be used. */
  usable: boolean;

  /** Setup-check readiness flags. */
  shouldersOk: boolean;
  hipsOk: boolean;
  wristOk: boolean;
  ready: boolean;

  /** Body midline in mirrored normalised x, from the shoulders. Used to
   *  split the reach box into same-side and across-midline halves. */
  midX: number;
  midXValid: boolean;

  /** Object-cover geometry used for the last mapping. */
  cover: { dispW: number; dispH: number; offX: number; offY: number };

  /** True once any pose has arrived. */
  hasPose: boolean;
  /** Timestamp of the last usable wrist sample, ms (performance.now). */
  lastUsableMs: number;
  /** Detector callbacks per second, smoothed. Written by the pose loop.
   *  The cursor can never be fresher than this, so it sets the floor on
   *  how much lag any amount of filtering can remove. */
  poseHz: number;
}

export function createHandState(): HandState {
  return {
    nx: 0.5,
    ny: 0.5,
    x: 0,
    y: 0,
    palmX: 0,
    palmY: 0,
    palmFromElbow: false,
    score: 0,
    live: false,
    inFrame: false,
    usable: false,
    shouldersOk: false,
    hipsOk: false,
    wristOk: false,
    ready: false,
    midX: 0.5,
    midXValid: false,
    cover: { dispW: 0, dispH: 0, offX: 0, offY: 0 },
    hasPose: false,
    lastUsableMs: 0,
    poseHz: 0,
  };
}

/**
 * Reproduce the browser's `object-cover` geometry, exactly as
 * RehabCameraShell does (lines 215-234): scale by the LARGER ratio so
 * the video COVERS the box, then centre the overflow. When the
 * container and the video share an aspect ratio this collapses to
 * `nx * containerW, ny * containerH`.
 */
export function coverGeometry(
  containerW: number,
  containerH: number,
  videoW: number,
  videoH: number,
) {
  const scale =
    videoW > 0 && videoH > 0
      ? Math.max(containerW / videoW, containerH / videoH)
      : 0;
  const dispW = scale > 0 ? videoW * scale : containerW;
  const dispH = scale > 0 ? videoH * scale : containerH;
  return {
    dispW,
    dispH,
    offX: (containerW - dispW) / 2,
    offY: (containerH - dispH) / 2,
  };
}

function inFrameN(nx: number, ny: number): boolean {
  return (
    nx >= -FRAME_SLACK
    && nx <= 1 + FRAME_SLACK
    && ny >= -FRAME_SLACK
    && ny <= 1 + FRAME_SLACK
  );
}

/**
 * Fill a HandState from one detector result.
 *
 * `kp` is the raw keypoint array in VIDEO PIXEL space — exactly what
 * usePoseDetectionLive returns. Mirroring matches the production shells
 * (`1 - x / videoW`) so the patient sees their hand on the side they
 * expect. The OTHER wrist is never read.
 */
export function updateHandState(
  s: HandState,
  kp: { x: number; y: number; score: number }[] | null,
  hand: Hand,
  videoW: number,
  videoH: number,
  containerW: number,
  containerH: number,
  nowMs: number,
): void {
  if (!kp || videoW <= 0 || videoH <= 0) {
    s.live = false;
    s.inFrame = false;
    s.usable = false;
    s.shouldersOk = false;
    s.hipsOk = false;
    s.wristOk = false;
    s.ready = false;
    return;
  }

  s.hasPose = true;
  const cover = coverGeometry(containerW, containerH, videoW, videoH);
  s.cover = cover;

  const read = (idx: number) => {
    const p = kp[idx];
    if (!p) return null;
    const nx = 1 - p.x / videoW;
    const ny = p.y / videoH;
    const score = p.score ?? 0;
    return {
      nx,
      ny,
      score,
      ok: score >= VIS_FLOOR && inFrameN(nx, ny),
    };
  };

  const wrist = read(WRIST_INDEX[hand]);
  if (wrist) {
    s.nx = wrist.nx;
    s.ny = wrist.ny;
    s.score = wrist.score;
    s.live = wrist.score >= VIS_FLOOR;
    s.inFrame = inFrameN(wrist.nx, wrist.ny);
    s.usable = wrist.ok;
    s.x = cover.offX + wrist.nx * cover.dispW;
    s.y = cover.offY + wrist.ny * cover.dispH;
    if (wrist.ok) s.lastUsableMs = nowMs;

    // Palm = wrist projected along the forearm, in canvas pixels.
    const elbow = read(ELBOW_INDEX[hand]);
    if (elbow?.ok && wrist.ok) {
      const ex = cover.offX + elbow.nx * cover.dispW;
      const ey = cover.offY + elbow.ny * cover.dispH;
      s.palmX = s.x + (s.x - ex) * PALM_REACH;
      s.palmY = s.y + (s.y - ey) * PALM_REACH;
      s.palmFromElbow = true;
    } else {
      // No usable elbow — sit on the wrist rather than guess a
      // direction from a landmark MediaPipe extrapolated.
      s.palmX = s.x;
      s.palmY = s.y;
      s.palmFromElbow = false;
    }
  } else {
    s.live = false;
    s.inFrame = false;
    s.usable = false;
  }

  const lSh = read(LM_LIVE.LEFT_SHOULDER);
  const rSh = read(LM_LIVE.RIGHT_SHOULDER);
  const lHip = read(LM_LIVE.LEFT_HIP);
  const rHip = read(LM_LIVE.RIGHT_HIP);

  s.shouldersOk = !!(lSh?.ok && rSh?.ok);
  s.hipsOk = !!(lHip?.ok && rHip?.ok);
  s.wristOk = s.usable;
  s.ready = s.shouldersOk && s.hipsOk && s.wristOk;

  if (lSh?.ok && rSh?.ok) {
    s.midX = (lSh.nx + rSh.nx) / 2;
    s.midXValid = true;
  }
}
