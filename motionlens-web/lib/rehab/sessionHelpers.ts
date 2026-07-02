// Shared pure helpers for rehab session save wiring.
//
// Every rehab exercise page follows the same shape when it comes
// to harvesting session state:
//   1. Grab each raw @tensorflow-models/pose-detection Keypoint[]
//      frame → serialize to a compact rounded snapshot.
//   2. Keep the "last known good" snapshot every frame so a save
//      always has SOMETHING to redraw, even if no "best" moment
//      ever triggered.
//   3. Promote to a "best" snapshot on the mechanic-specific
//      trigger (new min interior, new max reach, new bestDwell
//      midpoint, on hit, etc).
//   4. On save-click, assemble a `skeleton_pose` object combining
//      best (preferred) OR last (fallback) with side + label.
//
// Extracting these into pure functions keeps every page's edit
// small and identical in shape. Zero React, zero shell knowledge,
// zero engine references — additive across the rehab module.

import type { Keypoint } from "@tensorflow-models/pose-detection";

export interface CompactLandmark {
  x: number;
  y: number;
  score: number;
  name?: string;
}

export interface PoseSnapshot {
  landmarks: CompactLandmark[];
  source_frame: { width: number; height: number };
}

export interface BestPoseSnapshot extends PoseSnapshot {
  angle: number;
  capturedAtMs: number;
}

/** Compact + rounded keypoint snapshot. Returns null on degenerate
 *  frames (video not ready / empty keypoint array). Rounding to 2
 *  decimals on x/y and 3 on score keeps the payload lean (~1.5 KB
 *  for 33 landmarks). */
export function kpToPoseSnapshot(
  kp: Keypoint[],
  videoWidth: number,
  videoHeight: number,
): PoseSnapshot | null {
  if (videoWidth <= 0 || videoHeight <= 0) return null;
  if (!kp || kp.length === 0) return null;
  return {
    landmarks: kp.map((p) => {
      const entry: CompactLandmark = {
        x: Math.round(p.x * 100) / 100,
        y: Math.round(p.y * 100) / 100,
        score: Math.round((p.score ?? 0) * 1000) / 1000,
      };
      if (p.name) entry.name = p.name;
      return entry;
    }),
    source_frame: { width: videoWidth, height: videoHeight },
  };
}

/** Build the final skeleton_pose object for a save payload. Prefers
 *  the best-frame snapshot; falls back to the last-known-good with
 *  the caller-supplied fallback angle. Returns null when neither
 *  is available. */
export function buildSkeletonPosePayload(
  best: BestPoseSnapshot | null,
  fallback: PoseSnapshot | null,
  fallbackAngle: number,
  side: string | null,
  label: string,
): {
  landmarks: CompactLandmark[];
  source_frame: { width: number; height: number };
  angle: number;
  captured_at_ms: number;
  side: string | null;
  label: string;
} | null {
  if (best) {
    return {
      landmarks: best.landmarks,
      source_frame: best.source_frame,
      angle: best.angle,
      captured_at_ms: best.capturedAtMs,
      side,
      label,
    };
  }
  if (fallback) {
    return {
      landmarks: fallback.landmarks,
      source_frame: fallback.source_frame,
      angle: fallbackAngle,
      captured_at_ms: Date.now(),
      side,
      label,
    };
  }
  return null;
}

/** Common denominator seconds-since-start helper. */
export function elapsedSecondsSince(startMs: number): number {
  return Math.max(0, (performance.now() - startMs) / 1000);
}
