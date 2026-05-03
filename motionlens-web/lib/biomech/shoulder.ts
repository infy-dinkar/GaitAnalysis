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
  { id: "external_rotation",  label: "External Rotation",  description: "Rotate the forearm outward (elbow at 90°)",  target: [80, 90] },
  { id: "internal_rotation",  label: "Internal Rotation",  description: "Rotate the forearm inward (elbow at 90°)",   target: [70, 90] },
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
