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

/** LOW-CONFIDENCE proxy for shoulder external rotation in degrees,
 *  derived from the lateral excursion of the wrist relative to the
 *  elbow in a 2-D frontal image. This is a TREND-ONLY signal, NOT
 *  an absolute clinical ER measurement.
 *
 *  Why a proxy is the best we can do here — true gleno-humeral
 *  rotation is axial (humerus spinning about its long axis) and is
 *  invisible in a single 2-D frontal view. The PRD explicitly
 *  warns the S5 reading should be reported as a within-patient
 *  trend, not an absolute angle. Callers MUST surface this caveat
 *  in the UI.
 *
 *  Method — with the elbow tucked at 90° flexion (forearm forward,
 *  pointing toward camera), external rotation swings the forearm
 *  out laterally in the image plane:
 *    • Neutral, forearm-forward: wrist projects near the elbow's
 *      x → lateral component ≈ 0 → proxy ≈ 0°
 *    • Half rotated: lateral ≈ half forearm length → proxy ≈ 30°
 *    • Full ER, forearm-lateral: lateral ≈ full forearm length →
 *      proxy ≈ 90°
 *
 *  Assumes the elbow stays at ~90° flexion AND remains tucked to
 *  the side throughout the rep. Either elbow extension OR drift
 *  of the elbow away from the side will distort the reading —
 *  caller should detect and coach against drift separately.
 */
export function computeForearmRotationProxyDeg(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const elbowIdx = side === "left" ? LM.LEFT_ELBOW : LM.RIGHT_ELBOW;
  const wristIdx = side === "left" ? LM.LEFT_WRIST : LM.RIGHT_WRIST;
  const elbow = keypoints[elbowIdx];
  const wrist = keypoints[wristIdx];
  if (!elbow || !wrist) return null;
  if ((elbow.score ?? 0) < VIS_THRESHOLD) return null;
  if ((wrist.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = wrist.x - elbow.x;
  const dy = wrist.y - elbow.y;
  const forearmLen = Math.hypot(dx, dy);
  if (forearmLen < 1e-4) return null;
  // sin(rotation) = |lateral component| / |forearm|.
  // Unsigned magnitude — side picker handles left-vs-right.
  const ratio = Math.min(1, Math.max(0, Math.abs(dx) / forearmLen));
  return (Math.asin(ratio) * 180) / Math.PI;
}

/** Unsigned hip abduction angle in degrees — angle between the
 *  thigh vector (hip → knee) and vertical-down, measured in image
 *  space (frontal-view assumption).
 *
 *  Reference cases:
 *    • 0°   = leg hanging straight down (standing neutral)
 *    • 30°  = mid-range active abduction
 *    • 45°  = upper end of typical clinical active ROM
 *    • 90°  = leg held horizontal (rare; usually passive only)
 *
 *  NOTE — no equivalent in lib/biomech/hip-live.ts. That file
 *  models sagittal flexion / extension only; abduction is not
 *  surfaced anywhere in biomech. Added here so the rehab module
 *  can drive cursor + score without modifying biomech.
 *
 *  CAVEAT — this is a 2-D image-plane projection. If the patient
 *  also flexes the hip (lifts leg forward), the apparent
 *  abduction angle inflates because the knee moves upward in the
 *  frontal image. The exercise's setup guidance instructs the
 *  patient to keep the leg purely lateral.
 */
export function computeHipAbductionDeg(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const hipIdx = side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP;
  const kneeIdx = side === "left" ? LM.LEFT_KNEE : LM.RIGHT_KNEE;
  const hip = keypoints[hipIdx];
  const knee = keypoints[kneeIdx];
  if (!hip || !knee) return null;
  if ((hip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((knee.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = knee.x - hip.x;
  const dy = knee.y - hip.y;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  // Unsigned magnitude — atan2 of |horizontal| over |vertical|.
  // Side picker handles left-vs-right; we don't care which way the
  // thigh swings.
  return (Math.atan2(Math.abs(dx), Math.abs(dy)) * 180) / Math.PI;
}
