// UI metadata + browser-side angle math for the shoulder biomech flow.
// The compute function ports shoulder_engine.py exactly so live and
// final measurements use the same formulas. Used by the live mode
// (client-side TF.js BlazePose + this math). Server-side upload mode
// still uses the Python engine of record for the authoritative report.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

export type ShoulderMovementId =
  | "flexion"
  | "extension"
  | "abduction"
  | "adduction"
  | "external_rotation"
  | "internal_rotation";

export interface ShoulderMovement {
  id: ShoulderMovementId;
  label: string;
  description: string;
  target: [number, number];
}

export const SHOULDER_MOVEMENTS: ShoulderMovement[] = [
  { id: "flexion",            label: "Flexion",            description: "Lift the arm forward and overhead",          target: [150, 180] },
  { id: "extension",          label: "Extension",          description: "Move the arm backward behind the body",       target: [45, 60] },
  { id: "abduction",          label: "Abduction",          description: "Lift the arm sideways away from the body",    target: [150, 180] },
  { id: "adduction",          label: "Adduction",          description: "Bring the arm across the chest",              target: [30, 50] },
  // Rotation targets adjusted to clinical reality. The previous 80-90 /
  // 70-90 ranges were the theoretical anatomical maxima; in practice
  // most healthy adults peak at 70-85° (external) and 60-75° (internal)
  // with the elbow tucked at 90°. The calibrated arcsin formula now
  // returns true projected rotation, so the previous targets would
  // false-flag healthy patients as "below normal".
  { id: "external_rotation",  label: "External Rotation",  description: "Rotate the forearm outward (elbow at 90°)",  target: [70, 90] },
  { id: "internal_rotation",  label: "Internal Rotation",  description: "Rotate the forearm inward (elbow at 90°)",   target: [60, 80] },
];

// BlazePose-tfjs scores are lower than MediaPipe's `visibility` field —
// typical visible joints score 0.2-0.4. 0.15 lets math run on most
// in-frame joints without false-passing fully occluded ones.
const VIS_THRESHOLD = 0.15;

// Signed angle from v1 to v2 in degrees (image-coord 2D), in (-180, 180].
function angleBetween(
  v1x: number, v1y: number,
  v2x: number, v2y: number,
): number {
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(cross, dot) * 180) / Math.PI;
}

function shoulderFlexionExtension(
  s: Keypoint, e: Keypoint, h: Keypoint,
): number {
  const tx = h.x - s.x, ty = h.y - s.y;       // trunk_down
  const ax = e.x - s.x, ay = e.y - s.y;       // arm
  return -angleBetween(tx, ty, ax, ay);
}

function shoulderAbductionAdduction(
  s: Keypoint, e: Keypoint, h: Keypoint,
): number {
  const tx = h.x - s.x, ty = h.y - s.y;
  const ax = e.x - s.x, ay = e.y - s.y;
  return Math.abs(angleBetween(tx, ty, ax, ay));
}

function shoulderRotation(
  _s: Keypoint, e: Keypoint, w: Keypoint,
): number {
  const fx = w.x - e.x, fy = w.y - e.y;       // forearm
  // reference = (1, 0)
  return Math.abs(angleBetween(1, 0, fx, fy));
}

const SIDE_INDICES = {
  left:  { shoulder: LM.LEFT_SHOULDER,  elbow: LM.LEFT_ELBOW,  wrist: LM.LEFT_WRIST,  hip: LM.LEFT_HIP  },
  right: { shoulder: LM.RIGHT_SHOULDER, elbow: LM.RIGHT_ELBOW, wrist: LM.RIGHT_WRIST, hip: LM.RIGHT_HIP },
} as const;

export function computeShoulderAngle(
  movement: ShoulderMovementId,
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const needed: number[] = [idx.shoulder, idx.elbow, idx.hip];
  if (movement === "external_rotation" || movement === "internal_rotation") {
    needed.push(idx.wrist);
  }
  for (const k of needed) {
    const kp = keypoints[k];
    if (!kp || (kp.score ?? 0) < VIS_THRESHOLD) return null;
  }
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  const h = keypoints[idx.hip];
  if (movement === "flexion" || movement === "extension") {
    return shoulderFlexionExtension(s, e, h);
  }
  if (movement === "abduction" || movement === "adduction") {
    return shoulderAbductionAdduction(s, e, h);
  }
  const w = keypoints[idx.wrist];
  return shoulderRotation(s, e, w);
}

// ─── Calibrated rotation (baseline-locked) ──────────────────────
//
// Geometry — same idea as the neck-rotation fix, applied to the
// shoulder. With the elbow tucked at the side and bent 90°, the
// forearm sweeps in a horizontal plane around the shoulder axis.
// At the patient's neutral position (forearm pointing forward at
// the camera), the forearm projection in the 2D image is near zero.
// As the patient rotates externally or internally, the forearm
// projection grows. Mathematically:
//
//     forearmProjectedLength = R_forearmLength × sin(θ)
//     ⇒ θ = arcsin(forearmProjectedLength / R_forearmLength)
//
// We can't measure R_forearmLength directly at neutral (projection ≈
// 0), so we use the upper-arm length as a proxy — anatomically
// forearm length ≈ upper-arm length for most adults (within ±10%).
// The upper arm is fully visible in 2D when tucked at the side
// (vertical pose), so its pixel length is a reliable scale.
//
// Sign is intentionally NOT returned — the operator already selects
// "external" or "internal" before recording, so direction is known
// from the test selection. We just report the magnitude (always ≥ 0).
// This also matches the report-card display, which renders peak
// magnitude vs target range without a +/- distinction.

export interface ShoulderRotationCalibration {
  /** Upper-arm pixel length at calibration (shoulder→elbow). Used
   *  as a proxy for forearm length in the arcsin formula. */
  R_upperArmLength: number;
  /** Side this baseline was captured for — guards against accidental
   *  cross-side application if the operator switches mid-flow. */
  side: "left" | "right";
}

/** Maximum forearm-projection-to-upper-arm ratio for "patient is at
 *  neutral rotation" (forearm pointing at the camera). 0.20 = forearm
 *  projects to no more than 20% of the upper-arm length. Generous
 *  enough to tolerate slight angle off-camera but tight enough that
 *  the patient can't be holding the forearm fully sideways. */
const SHOULDER_NEUTRAL_FOREARM_RATIO_MAX = 0.20;

/** Minimum upper-arm pixel length to accept calibration. Below this
 *  the patient is too far from the camera, MoveNet's keypoints are
 *  too noisy, and the rotation readings would be unreliable. */
const MIN_UPPER_ARM_PX = 30;

/** True when MoveNet sees shoulder + elbow + wrist for the given
 *  side AND the forearm is approximately pointing at the camera (low
 *  2D projection). Used by the live-capture flow to decide when to
 *  auto-lock the shoulder-rotation baseline. */
export function isShoulderRotationNeutral(
  keypoints: Keypoint[],
  side: "left" | "right",
): boolean {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  const w = keypoints[idx.wrist];
  for (const k of [s, e, w]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return false;
  }
  const upperArmLen = Math.hypot(e.x - s.x, e.y - s.y);
  if (upperArmLen < MIN_UPPER_ARM_PX) return false;
  const forearmLen = Math.hypot(w.x - e.x, w.y - e.y);
  return forearmLen / upperArmLen <= SHOULDER_NEUTRAL_FOREARM_RATIO_MAX;
}

/** Snapshot the patient's neutral-position upper-arm length as the
 *  rotation reference. Called once after isShoulderRotationNeutral
 *  has held true for the calibration-stable window. */
export function captureShoulderRotationBaseline(
  keypoints: Keypoint[],
  side: "left" | "right",
): ShoulderRotationCalibration | null {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  if (!s || !e) return null;
  if ((s.score ?? 0) < VIS_THRESHOLD || (e.score ?? 0) < VIS_THRESHOLD) return null;
  const len = Math.hypot(e.x - s.x, e.y - s.y);
  if (len < MIN_UPPER_ARM_PX) return null;
  return { R_upperArmLength: len, side };
}

/** Calibrated rotation magnitude in degrees, capped at 90°.
 *  Returns null when wrist or elbow visibility drops (typical at the
 *  extreme of rotation when the wrist crosses behind the body). The
 *  upstream peak-tracker holds the last good value in that case. */
export function computeShoulderRotationFromBaseline(
  keypoints: Keypoint[],
  side: "left" | "right",
  baseline: ShoulderRotationCalibration,
): number | null {
  if (baseline.side !== side) return null; // wrong-side baseline guard
  const idx = SIDE_INDICES[side];
  const e = keypoints[idx.elbow];
  const w = keypoints[idx.wrist];
  if (!e || !w) return null;
  if ((e.score ?? 0) < VIS_THRESHOLD || (w.score ?? 0) < VIS_THRESHOLD) return null;
  const forearmProjLen = Math.hypot(w.x - e.x, w.y - e.y);
  // arcsin saturates at 1; clamp the ratio. If the patient over-rotates
  // (somehow projecting longer than the upper-arm-length proxy
  // suggests, e.g. anatomical forearm-longer-than-upper-arm), we cap
  // at 90° rather than returning NaN.
  const ratio = Math.max(0, Math.min(1, forearmProjLen / baseline.R_upperArmLength));
  return (Math.asin(ratio) * 180) / Math.PI;
}
