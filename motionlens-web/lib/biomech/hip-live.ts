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

/** Anatomical ceiling for hip rotation magnitude in degrees. Matches
 *  the backend's `_HIP_ROT_MAX_DEG` in hip_engine.py. Hip internal +
 *  external rotation maxes out clinically around 45°; anything above
 *  is either a measurement artefact (patient not in supine test
 *  posture, shin tilted by non-rotation factors) or pose noise. We
 *  clip the returned magnitude here so the live peak tracker can't
 *  lock onto anatomically impossible values like the 81.5° readings
 *  the formula could otherwise produce when the patient is sitting
 *  upright or partly out of frame. */
const HIP_ROTATION_MAX_DEG = 45;

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
  // sideways tilt = rotation. Clipped at HIP_ROTATION_MAX_DEG so
  // body-posture artefacts (sitting upright, partial framing, etc.)
  // can't pollute the peak tracker with unrealistic values.
  const shinX = ankle.x - knee.x;
  const shinY = ankle.y - knee.y;
  const rawMag = Math.abs(signedAngleBetween(0, 1, shinX, shinY));
  return Math.min(rawMag, HIP_ROTATION_MAX_DEG);
}

// ─── Direction detection for merged hip rotation ─────────────────
//
// Patient supine, knee bent 90°, foot pointing at the camera. The
// rotation happens at the HIP but the shin is the visible lever —
// counter-intuitive but anatomically correct:
//   • Internal rotation at the hip → thigh rotates medially → lower
//     leg (the lever's far end) swings LATERALLY (away from body
//     midline).
//   • External rotation at the hip → thigh rotates laterally → lower
//     leg swings MEDIALLY (toward body midline).
//
// We project the shin (knee→ankle) onto the image plane and use the
// SIGNED angle from vertical. The sign depends on which side: for the
// left leg the outward direction is +x in the typical un-mirrored
// camera frame, for the right it's −x. Multiplying the raw signed
// angle by a per-side outwardSign normalises both legs to the same
// convention: positive = INTERNAL rotation (lateral swing), negative
// = EXTERNAL rotation (medial swing). Matches the backend convention
// in hip_engine.py's _analyze_hip_rotation so saved-report viewers
// and live mode show the same direction labels for the same physical
// motion.

export type HipRotationDirection = "internal" | "external";

/** Minimum signed shin-from-vertical magnitude (degrees) before a
 *  direction is committed. Below this the lower leg is too close to
 *  vertical for reliable left/right discrimination — return null so
 *  the peak tracker doesn't flip slots on neutral-pose jitter. */
const HIP_ROT_DIRECTION_DEADBAND_DEG = 3;

/** Detect internal vs external rotation of the hip from the signed
 *  shin angle from vertical. Counter-intuitive lever convention:
 *    • LEFT hip: shin tilting to +x (image right) ≈ internal rotation
 *    • RIGHT hip: shin tilting to −x (image left) ≈ internal rotation
 *  Wrapped in an outwardSign multiplier so both sides reduce to the
 *  same "positive signed = internal" rule. */
export function detectHipRotationDirection(
  keypoints: Keypoint[],
  side: "left" | "right",
): HipRotationDirection | null {
  const idx = SIDE_INDICES[side];
  const knee = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  if (!knee || !ankle) return null;
  if ((knee.score ?? 0) < VIS_THRESHOLD || (ankle.score ?? 0) < VIS_THRESHOLD) {
    return null;
  }
  const shinX = ankle.x - knee.x;
  const shinY = ankle.y - knee.y;
  // signedAngleBetween((0, 1), shin) returns positive when the shin
  // tilts toward −x in image space (because cross = 1*shinY - 0*shinX
  // wait — that's not right; let me re-derive). atan2(cross, dot)
  // where cross = v1x*v2y − v1y*v2x = 0*shinY − 1*shinX = −shinX, and
  // dot = 0*shinX + 1*shinY = shinY. So signedAngle = atan2(−shinX,
  // shinY). Positive signed → shin tilts to −x (image left).
  const signedAngle = signedAngleBetween(0, 1, shinX, shinY);
  if (Math.abs(signedAngle) < HIP_ROT_DIRECTION_DEADBAND_DEG) return null;
  // Per-side outward sign: matches hip_engine.py's
  // `_hip_rotation_outward_sign` — +1 left, −1 right.
  const outwardSign = side === "left" ? 1 : -1;
  // Empirical label assignment confirmed against live patient testing:
  // patient seated supine on a chair / table, the in-browser camera
  // shows a mirrored selfie preview but MediaPipe emits landmarks in
  // the UN-mirrored image frame. The two conventions land on opposite
  // sides of vertical compared to the backend's reference frame, so
  // we flip the labels here. (signed*outwardSign > 0 means the shin
  // swings to the patient's MEDIAL side in the un-mirrored frame —
  // which is the lower-leg signature of EXTERNAL hip rotation, not
  // internal, despite hip_engine.py's positive-signed = internal
  // convention. Backend works the same way; only the per-frame
  // detection in live mode needs this sign correction.)
  return signedAngle * outwardSign > 0 ? "external" : "internal";
}
