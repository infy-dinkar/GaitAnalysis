// UI metadata + browser-side angle math for the ankle biomech flow.
//
// BlazePose Full provides foot landmarks (HEEL + FOOT_INDEX, kp 29-32)
// that the legacy MoveNet detector did not have, so this live version
// can compute the true ankle joint angle via the interior angle at
// the ankle vertex between the shin (knee→ankle) and the foot
// (ankle→foot_index) — the same approach the backend
// ankle_engine.py uses for uploaded videos.
//
// Convention:
//   • At anatomical neutral (foot perpendicular to shin) the interior
//     angle ≈ 90°.
//   • Dorsiflexion (toes toward the shin) CLOSES the angle below 90°.
//   • Plantarflexion (toes away from the shin) OPENS it above 90°.
//   • Direction-aware signed return: for the "flexion" (dorsi) test
//     we return (90 − interior) — positive when the foot is past
//     neutral toward the shin, negative when it's still in the plantar
//     half of the ROM. The "extension" (plantar) test mirrors this
//     with (interior − 90). The signed reading lets the live UI
//     respond to ANY foot motion (so the doctor sees the angle change
//     while the patient is transitioning from rest), and the peak
//     tracker — which `Math.max`-tracks the signed value across
//     frames — naturally locks onto the deepest excursion in the
//     test direction once the patient crosses past neutral.

import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

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

function angleAt(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number,
): number {
  // Interior (unsigned) angle at vertex B for points A-B-C, in [0°, 180°].
  const v1x = ax - bx, v1y = ay - by;
  const v2x = cx - bx, v2y = cy - by;
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
}

const SIDE_INDICES = {
  left:  {
    knee:  LM.LEFT_KNEE,
    ankle: LM.LEFT_ANKLE,
    foot:  LM.LEFT_FOOT_INDEX,
  },
  right: {
    knee:  LM.RIGHT_KNEE,
    ankle: LM.RIGHT_ANKLE,
    foot:  LM.RIGHT_FOOT_INDEX,
  },
} as const;

export function computeAnkleAngle(
  movement: AnkleMovementId,
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const knee  = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  const foot  = keypoints[idx.foot];
  for (const k of [knee, ankle, foot]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }
  // Interior angle at the ANKLE vertex between the shin and foot
  // vectors. With foot ⟂ shin (neutral) the angle ≈ 90°; dorsiflexion
  // closes it below 90°, plantarflexion opens it above 90°.
  const interior = angleAt(knee.x, knee.y, ankle.x, ankle.y, foot.x, foot.y);
  if (movement === "flexion") {
    // Dorsiflexion test: + when foot is past neutral toward shin
    // (true dorsi), − while the foot is still on the plantar side.
    // Live UI reflects motion either way; peak tracker locks the
    // maximum signed value seen during the recording.
    return 90 - interior;
  }
  // Plantarflexion test: + when foot points away from shin past
  // neutral, − while it's on the dorsi side.
  return interior - 90;
}
