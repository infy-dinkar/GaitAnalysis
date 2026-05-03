// UI metadata + browser-side angle math for the ankle biomech flow.
// MoveNet does not give heel or foot_index keypoints, so we measure
// ankle flexion via the shin (knee → ankle) angle from vertical, as
// in the standing knee-to-wall test for dorsiflexion. This is the
// best 2D approximation possible without dedicated foot landmarks.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

export type AnkleMovementId = "flexion" | "extension";

export interface AnkleMovement {
  id: AnkleMovementId;
  label: string;
  description: string;
  target: [number, number];
}

export const ANKLE_MOVEMENTS: AnkleMovement[] = [
  {
    id: "flexion",
    label: "Dorsiflexion",
    description: "Pull the foot upward (toes toward shin) — knee-to-wall lean",
    target: [15, 25],
  },
  {
    id: "extension",
    label: "Plantarflexion",
    description: "Point the foot down, like pressing a gas pedal",
    target: [40, 55],
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
  left:  { knee: LM.LEFT_KNEE,  ankle: LM.LEFT_ANKLE  },
  right: { knee: LM.RIGHT_KNEE, ankle: LM.RIGHT_ANKLE },
} as const;

export function computeAnkleAngle(
  _movement: AnkleMovementId,
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const knee  = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  for (const k of [knee, ankle]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }
  // Shin vector (knee → ankle). Image y-axis points down, so a vertical
  // shin (standing neutral) is approximately (0, +1).
  const shinX = ankle.x - knee.x;
  const shinY = ankle.y - knee.y;
  return Math.abs(signedAngleBetween(0, 1, shinX, shinY));
}
