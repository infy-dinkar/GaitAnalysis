// UI metadata + browser-side angle math for the knee biomech flow.
// Both flexion and extension use the same hip-knee-ankle interior angle;
// the difference is only the target range and clinical interpretation.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

export type KneeMovementId = "flexion" | "extension";

export interface KneeMovement {
  id: KneeMovementId;
  label: string;
  description: string;
  target: [number, number];
}

export const KNEE_MOVEMENTS: KneeMovement[] = [
  {
    id: "flexion",
    label: "Flexion",
    description: "Bend the knee — bringing the heel toward the buttock",
    target: [125, 145],
  },
  {
    id: "extension",
    label: "Extension",
    description: "Straighten a bent knee back to a fully extended leg",
    target: [125, 145],
  },
];

const VIS_THRESHOLD = 0.15;

function angleAt(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number,
): number {
  // Interior angle at vertex B for points A-B-C.
  const v1x = ax - bx, v1y = ay - by;
  const v2x = cx - bx, v2y = cy - by;
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
}

const SIDE_INDICES = {
  left:  { hip: LM.LEFT_HIP,  knee: LM.LEFT_KNEE,  ankle: LM.LEFT_ANKLE  },
  right: { hip: LM.RIGHT_HIP, knee: LM.RIGHT_KNEE, ankle: LM.RIGHT_ANKLE },
} as const;

export function computeKneeAngle(
  _movement: KneeMovementId,
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const hip   = keypoints[idx.hip];
  const knee  = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  for (const k of [hip, knee, ankle]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }
  const interior = angleAt(hip.x, hip.y, knee.x, knee.y, ankle.x, ankle.y);
  // Convert interior angle (180° = straight) → flexion-from-straight.
  // 0° flexion = perfectly straight, ~140° = fully bent.
  return 180 - interior;
}
