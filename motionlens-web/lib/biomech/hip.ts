// UI metadata + browser-side angle math for the hip biomech flow.
// Flexion / extension use the trunk → thigh angle in the sagittal plane.
// Internal / external rotation use the shin direction (knee → ankle)
// projected to the camera plane; this is a 2D approximation suitable
// for screening — not for clinical-grade rotation measurement.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

export type HipMovementId =
  | "flexion"
  | "extension"
  | "internal_rotation"
  | "external_rotation";

export interface HipMovement {
  id: HipMovementId;
  label: string;
  description: string;
  target: [number, number];
}

export const HIP_MOVEMENTS: HipMovement[] = [
  {
    id: "flexion",
    label: "Flexion",
    description: "Lift the leg forward — bringing the thigh toward the chest",
    target: [110, 130],
  },
  {
    id: "extension",
    label: "Extension",
    description: "Move the leg backward behind the body",
    target: [10, 30],
  },
  {
    id: "internal_rotation",
    label: "Internal Rotation",
    description: "Rotate the thigh inward (knee bent at 90°)",
    target: [30, 45],
  },
  {
    id: "external_rotation",
    label: "External Rotation",
    description: "Rotate the thigh outward (knee bent at 90°)",
    target: [40, 60],
  },
];

const VIS_THRESHOLD = 0.15;

function signedAngleBetween(
  v1x: number, v1y: number,
  v2x: number, v2y: number,
): number {
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(cross, dot) * 180) / Math.PI;
}

const SIDE_INDICES = {
  left:  {
    shoulder: LM.LEFT_SHOULDER,
    hip:      LM.LEFT_HIP,
    knee:     LM.LEFT_KNEE,
    ankle:    LM.LEFT_ANKLE,
  },
  right: {
    shoulder: LM.RIGHT_SHOULDER,
    hip:      LM.RIGHT_HIP,
    knee:     LM.RIGHT_KNEE,
    ankle:    LM.RIGHT_ANKLE,
  },
} as const;

export function computeHipAngle(
  movement: HipMovementId,
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const shoulder = keypoints[idx.shoulder];
  const hip      = keypoints[idx.hip];
  const knee     = keypoints[idx.knee];
  const ankle    = keypoints[idx.ankle];

  const needed: Keypoint[] = [hip, knee];
  if (movement === "flexion" || movement === "extension") needed.push(shoulder);
  if (movement === "internal_rotation" || movement === "external_rotation") {
    needed.push(ankle);
  }
  for (const k of needed) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }

  if (movement === "flexion" || movement === "extension") {
    // Trunk vector (hip → shoulder, pointing UP) vs thigh (hip → knee,
    // pointing DOWN at neutral). Standing neutral angle ≈ 180°.
    // Flexion lifts thigh toward trunk → angle decreases.
    const trunkX = shoulder.x - hip.x;
    const trunkY = shoulder.y - hip.y;
    const thighX = knee.x - hip.x;
    const thighY = knee.y - hip.y;
    const interior =
      (Math.atan2(
        Math.abs(trunkX * thighY - trunkY * thighX),
        trunkX * thighX + trunkY * thighY,
      ) *
        180) /
      Math.PI;
    // Convert: 180° (standing straight) → 0°, leg lifted forward → larger.
    return 180 - interior;
  }

  // Rotation: 2D approximation using shin direction (knee → ankle)
  // relative to vertical. With the patient supine + knee at 90°, foot
  // pointing at the camera, shin should be ≈ vertical at neutral; any
  // sideways tilt = rotation.
  const shinX = ankle.x - knee.x;
  const shinY = ankle.y - knee.y;
  // Image vertical axis points DOWN, so neutral shin (foot toward
  // camera, supine) is approximately along (0, +1).
  return Math.abs(signedAngleBetween(0, 1, shinX, shinY));
}
