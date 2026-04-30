// UI metadata + browser-side angle math for the neck biomech flow.
// Ports neck_engine.py exactly. The neck engine has no per-side
// parameter — it uses both ears + the shoulder midline regardless of
// which way the patient tilts/turns.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

export type NeckMovementId =
  | "flexion"
  | "extension"
  | "lateral_flexion"
  | "rotation";

export interface NeckMovement {
  id: NeckMovementId;
  label: string;
  description: string;
  target: [number, number];
}

export const NECK_MOVEMENTS: NeckMovement[] = [
  { id: "flexion",         label: "Flexion",         description: "Tilt the head forward, chin to chest",   target: [45, 80] },
  { id: "extension",       label: "Extension",       description: "Tilt the head backward",                  target: [50, 70] },
  { id: "lateral_flexion", label: "Lateral Flexion", description: "Tilt the ear toward either shoulder",   target: [20, 45] },
  { id: "rotation",        label: "Rotation",        description: "Turn the head to either side",            target: [70, 90] },
];

const VIS_THRESHOLD = 0.3;

function angleBetween(
  v1x: number, v1y: number,
  v2x: number, v2y: number,
): number {
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(cross, dot) * 180) / Math.PI;
}

export function computeNeckAngle(
  movement: NeckMovementId,
  keypoints: Keypoint[],
): number | null {
  const nose = keypoints[LM.NOSE];
  const lEar = keypoints[LM.LEFT_EAR];
  const rEar = keypoints[LM.RIGHT_EAR];
  const lSh  = keypoints[LM.LEFT_SHOULDER];
  const rSh  = keypoints[LM.RIGHT_SHOULDER];

  for (const k of [nose, lEar, rEar, lSh, rSh]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }

  const earMidX = (lEar.x + rEar.x) / 2;
  const earMidY = (lEar.y + rEar.y) / 2;
  const shldrMidX = (lSh.x + rSh.x) / 2;
  const shldrMidY = (lSh.y + rSh.y) / 2;

  if (
    movement === "flexion" ||
    movement === "extension" ||
    movement === "lateral_flexion"
  ) {
    // angle of (shoulder_mid → ear_mid) from upward vertical
    const neckVecX = earMidX - shldrMidX;
    const neckVecY = earMidY - shldrMidY;
    const verticalX = 0;
    const verticalY = -1;
    const raw = angleBetween(verticalX, verticalY, neckVecX, neckVecY);
    return movement === "lateral_flexion" ? Math.abs(raw) : raw;
  }

  // rotation — nose offset relative to ear midline, ratio × 90°
  const earWidth = Math.abs(lEar.x - rEar.x);
  if (earWidth < 1e-3) return 0;
  const offsetRatio = (nose.x - earMidX) / (earWidth / 2);
  return Math.min(Math.abs(offsetRatio * 90), 120);
}
