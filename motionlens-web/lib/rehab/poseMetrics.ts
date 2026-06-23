// Pure pose-derived metrics for the rehab module.
//
// Mirrors the style of lib/biomech/*-live.ts helpers
// (computeShoulderLineAngleFromHorizontal etc.) but lives under
// lib/rehab/ so it never collides with — or risks modification of —
// existing biomech code. New rehab-specific pose metrics that don't
// have an equivalent in biomech go here.

import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";

const VIS_THRESHOLD = 0.15;

/** Signed pelvic tilt (degrees) of the line from LEFT_HIP to
 *  RIGHT_HIP relative to horizontal. Image coordinates are y-down,
 *  so:
 *    • 0°     = pelvis perfectly level
 *    • +5°    = right hip dropped relative to left
 *    • -5°    = left hip dropped relative to right
 *
 *  Returns null when either hip landmark fails visibility, or when
 *  the two hips project to coincident image points (degenerate).
 *  Pure — no state, no side effects.
 */
export function computePelvicTiltDeg(
  keypoints: Keypoint[],
): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!lHip || !rHip) return null;
  if ((lHip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rHip.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = rHip.x - lHip.x;
  const dy = rHip.y - lHip.y;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

/** Pixel-space midpoint x of LEFT_HIP and RIGHT_HIP — a cheap
 *  medio-lateral centre-of-mass proxy. Returns null when either
 *  hip fails visibility. Pure helper for the weight-shift game.
 *
 *  Note: no equivalent exists in lib/biomech/*-live.ts —
 *  computeHipShoulderMidX in hip-live.ts is the SHOULDER midpoint,
 *  not the hip midpoint. Added here rather than in biomech to keep
 *  the "rehab module never modifies biomech" invariant.
 */
export function computeHipMidX(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!lHip || !rHip) return null;
  if ((lHip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rHip.score ?? 0) < VIS_THRESHOLD) return null;
  return (lHip.x + rHip.x) / 2;
}

/** Shoulder pixel-width — |rSh.x - lSh.x|. Used as a body-scale
 *  reference (e.g. mapping medio-lateral hip shifts onto a
 *  normalised cursor). Duplicates the helper in
 *  lib/biomech/neck-live.ts intentionally so the rehab module
 *  carries no compile-time dependency on a biomech file that might
 *  evolve for assessment-specific reasons. */
export function computeShoulderWidth(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!lSh || !rSh) return null;
  if ((lSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rSh.score ?? 0) < VIS_THRESHOLD) return null;
  return Math.abs(rSh.x - lSh.x);
}
