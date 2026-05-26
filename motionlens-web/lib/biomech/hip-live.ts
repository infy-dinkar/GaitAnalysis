// UI metadata + browser-side angle math for the hip biomech flow.
// Flexion / extension use the trunk → thigh angle in the sagittal plane.
// Internal / external rotation use the shin direction (knee → ankle)
// projected to the camera plane; this is a 2D approximation suitable
// for screening — not for clinical-grade rotation measurement.

import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

export type HipMovementId =
  | "flexion"
  | "extension"
  | "rotation"
  | "internal_rotation"
  | "external_rotation";

export interface HipMovement {
  id: HipMovementId;
  label: string;
  description: string;
  target: [number, number];
  /** Merged tests carry a secondary direction with its own target.
   *  Only set on dual-direction merged entries (currently `rotation`). */
  merged?: boolean;
  primaryLabel?: string;
  secondaryLabel?: string;
  secondaryTarget?: [number, number];
  /** When true the chooser hides the entry but the lookup table still
   *  resolves it — kept so saved reports referencing legacy single-
   *  direction IDs (`internal_rotation` / `external_rotation`) still
   *  resolve their labels + targets. */
  hidden?: boolean;
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
  // Merged Internal + External rotation. Patient supine, knee bent at
  // 90° with the lower leg pointing toward the camera. One recording
  // captures BOTH directions (lower leg falls laterally for internal
  // rotation at the hip, medially for external — counter-intuitive but
  // anatomically correct, since rotation happens at the hip and the
  // ankle is at the far end of the tibia lever). The backend pipeline
  // mirrors the shoulder rotation flow (calibrated foreshortening with
  // a baseline locked at the neutral pose).
  {
    id: "rotation",
    label: "Rotation (Internal + External)",
    description:
      "Lying on the back with the knee bent at 90° and the lower leg pointing at the camera, rotate the thigh inward then outward. One session captures both peaks.",
    target: [30, 45],
    merged: true,
    primaryLabel: "Internal Rotation",
    secondaryLabel: "External Rotation",
    secondaryTarget: [30, 45],
  },
  // Legacy single-direction entries — kept in the lookup table so
  // saved reports that referenced these IDs still resolve labels +
  // targets. Hidden from the chooser since the merged entry above is
  // the new default.
  {
    id: "internal_rotation",
    label: "Internal Rotation",
    description: "Rotate the thigh inward (knee bent at 90°)",
    target: [30, 45],
    hidden: true,
  },
  {
    id: "external_rotation",
    label: "External Rotation",
    description: "Rotate the thigh outward (knee bent at 90°)",
    target: [40, 60],
    hidden: true,
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
  if (
    movement === "internal_rotation" ||
    movement === "external_rotation" ||
    movement === "rotation"
  ) {
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
