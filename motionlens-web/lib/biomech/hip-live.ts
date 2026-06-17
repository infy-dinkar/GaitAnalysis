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
  /** Optional reference illustration. See MovementGrid's
   *  MovementOption.imageUrl for the path convention. */
  imageUrl?: string;
}

export const HIP_MOVEMENTS: HipMovement[] = [
  {
    id: "flexion",
    label: "Flexion",
    description: "Lift the leg forward — bringing the thigh toward the chest",
    target: [110, 130],
    imageUrl: "/images/biomech/hip/hip_flexion.png",
  },
  {
    id: "extension",
    label: "Extension",
    description: "Move the leg backward behind the body",
    target: [10, 30],
    imageUrl: "/images/biomech/hip/hip_extension.png",
  },
  // Merged Internal + External rotation. Seated heel-fixed test:
  // patient sits upright on a chair with both feet flat on the
  // ground, then rotates ONE leg at the hip while keeping the heel
  // planted as the pivot. The toes (foot_index landmark) swing
  // laterally around the heel — outward = external rotation,
  // inward = internal rotation. One recording captures both peaks.
  {
    id: "rotation",
    label: "Rotation (Internal + External)",
    description:
      "Sit upright on a chair with both feet flat on the ground. Keeping the heel planted, rotate one foot outward then inward — the movement should come from the hip. One session captures both peaks.",
    target: [30, 45],
    merged: true,
    primaryLabel: "Internal Rotation",
    secondaryLabel: "External Rotation",
    secondaryTarget: [30, 45],
    imageUrl: "/images/biomech/hip/hip_rotation.png",
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
    shoulder:  LM.LEFT_SHOULDER,
    hip:       LM.LEFT_HIP,
    knee:      LM.LEFT_KNEE,
    ankle:     LM.LEFT_ANKLE,
    heel:      LM.LEFT_HEEL,
    footIndex: LM.LEFT_FOOT_INDEX,
  },
  right: {
    shoulder:  LM.RIGHT_SHOULDER,
    hip:       LM.RIGHT_HIP,
    knee:      LM.RIGHT_KNEE,
    ankle:     LM.RIGHT_ANKLE,
    heel:      LM.RIGHT_HEEL,
    footIndex: LM.RIGHT_FOOT_INDEX,
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
  const heel     = keypoints[idx.heel];
  const footIdx  = keypoints[idx.footIndex];

  const needed: Keypoint[] = [hip, knee];
  if (movement === "flexion" || movement === "extension") needed.push(shoulder);
  if (
    movement === "internal_rotation" ||
    movement === "external_rotation" ||
    movement === "rotation"
  ) {
    // Seated heel-fixed rotation needs the foot landmarks
    // (heel = pivot, foot_index = swinging end) instead of the
    // ankle that the legacy supine formula relied on.
    needed.push(heel);
    needed.push(footIdx);
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

  // Rotation — seated heel-fixed test. The patient sits upright with
  // the heel planted as a pivot; the toes (foot_index) swing
  // laterally as the hip rotates. We measure the angle of the
  // (heel → foot_index) vector from straight-down in the image:
  //   • At neutral (foot pointing forward at the camera) the vector
  //     projects to ≈ vertical (small lateral component) → angle ≈ 0°.
  //   • As the foot swings outward or inward, the vector tilts in
  //     the image plane → angle grows.
  // Clipped at HIP_ROTATION_MAX_DEG (45°, the anatomical ceiling) so
  // off-test postures or partial framing can't push the peak into
  // implausible territory.
  const footX = footIdx.x - heel.x;
  const footY = footIdx.y - heel.y;
  const rawMag = Math.abs(signedAngleBetween(0, 1, footX, footY));
  return Math.min(rawMag, HIP_ROTATION_MAX_DEG);
}

// ─── Direction detection for merged hip rotation ─────────────────
//
// Seated heel-fixed test: patient sits upright with the heel
// planted and the foot pivots at the hip. The toes (foot_index)
// swing laterally in the image:
//   • External rotation → toes swing OUTWARD (away from body midline)
//   • Internal rotation → toes swing INWARD (toward midline / opposite leg)
//
// We project (heel → foot_index) onto the image plane and use the
// SIGNED angle from vertical-down. The sign depends on which leg:
// the patient's LEFT foot's "outward" is +x in the typical un-
// mirrored MediaPipe frame, the RIGHT foot's "outward" is −x.
// Multiplying the raw signed angle by a per-side outwardSign
// normalises both legs to the same rule. Empirical label
// assignment confirmed against live patient testing.

export type HipRotationDirection = "internal" | "external";

/** Minimum signed foot-from-vertical magnitude (degrees) before a
 *  direction is committed. Below this the foot is too close to
 *  pointing-at-camera neutral for reliable left/right discrimination
 *  — return null so the peak tracker doesn't flip slots on neutral-
 *  pose jitter. */
const HIP_ROT_DIRECTION_DEADBAND_DEG = 3;

/** Detect internal vs external rotation of the hip in the seated
 *  heel-fixed test, from the signed foot-vector angle (heel →
 *  foot_index) from vertical-down in the image. Per-side outwardSign
 *  flips for left vs right; the inner conditional then maps "toes
 *  swung outward" to "external" and "toes swung inward" to
 *  "internal" — same label/sign convention the supine version used
 *  after empirical correction, just with the new pivot/lever pair. */
export function detectHipRotationDirection(
  keypoints: Keypoint[],
  side: "left" | "right",
): HipRotationDirection | null {
  const idx = SIDE_INDICES[side];
  const heel = keypoints[idx.heel];
  const footIdx = keypoints[idx.footIndex];
  if (!heel || !footIdx) return null;
  if ((heel.score ?? 0) < VIS_THRESHOLD || (footIdx.score ?? 0) < VIS_THRESHOLD) {
    return null;
  }
  const footX = footIdx.x - heel.x;
  const footY = footIdx.y - heel.y;
  const signedAngle = signedAngleBetween(0, 1, footX, footY);
  if (Math.abs(signedAngle) < HIP_ROT_DIRECTION_DEADBAND_DEG) return null;
  // Per-side outward sign: +1 left, −1 right (un-mirrored MediaPipe
  // image: patient's LEFT side appears on image-right, so "outward"
  // for the left foot lives in +x territory).
  const outwardSign = side === "left" ? 1 : -1;
  return signedAngle * outwardSign > 0 ? "external" : "internal";
}

// ─── Compensation detection (Hip flex / ext / rotation) — mirror ─

export type CompensationSeverity = "high" | "medium" | "low";

export interface Compensation {
  type:
    | "pelvic_tilt_posterior"
    | "pelvic_tilt_anterior"
    | "hip_trunk_lean"
    | "hip_trunk_rotation"
    | "knee_tilt";
  label: string;
  severity: CompensationSeverity;
  flagged: boolean;
  details?: string;
}

const HIP_PELVIC_TILT_THRESHOLD_DEG = 8;
const HIP_TRUNK_LEAN_THRESHOLD_FRAC = 0.15;
const HIP_TRUNK_ROTATION_THRESHOLD_FRAC = 0.15;
const HIP_KNEE_TILT_THRESHOLD_FRAC = 0.12;
const HIP_COMPENSATION_BASELINE_FRAME_COUNT = 10;

export function computeHipTrunkAngleDeg(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!lSh || !rSh || !lHip || !rHip) return null;
  if ((lSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((lHip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rHip.score ?? 0) < VIS_THRESHOLD) return null;
  const shMidX = (lSh.x + rSh.x) / 2;
  const shMidY = (lSh.y + rSh.y) / 2;
  const hpMidX = (lHip.x + rHip.x) / 2;
  const hpMidY = (lHip.y + rHip.y) / 2;
  const dx = shMidX - hpMidX;
  const dy = -(shMidY - hpMidY);
  if (Math.hypot(dx, dy) < 1e-4) return null;
  return (Math.atan2(dx, dy) * 180) / Math.PI;
}

export function computeHipShoulderMidX(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!lSh || !rSh) return null;
  if ((lSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rSh.score ?? 0) < VIS_THRESHOLD) return null;
  return (lSh.x + rSh.x) / 2;
}

export function computeHipShoulderWidth(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!lSh || !rSh) return null;
  if ((lSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rSh.score ?? 0) < VIS_THRESHOLD) return null;
  return Math.abs(rSh.x - lSh.x);
}

function hipCompMean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

abstract class HipBaseTracker {
  protected frameCounter = 0;
  protected primaryPeakFrame: number | null = null;
  protected secondaryPeakFrame: number | null = null;

  markPrimaryPeak(): void {
    this.primaryPeakFrame = Math.max(0, this.frameCounter - 1);
  }
  markSecondaryPeak(): void {
    this.secondaryPeakFrame = Math.max(0, this.frameCounter - 1);
  }
}

export class HipFlexionCompensationTracker extends HipBaseTracker {
  private trunkSamples: number[] = [];
  private shoulderXSamples: number[] = [];
  private widthSamples: number[] = [];
  private baselineTrunk: number | null = null;
  private baselineShoulderX: number | null = null;
  private baselineWidth: number | null = null;
  private pelvicPeakDev = 0;
  private leanPeakFrac = 0;
  private pelvicFlagged = false;
  private leanFlagged = false;
  private currentPelvicActive = false;
  private currentLeanActive = false;

  feed(keypoints: Keypoint[]): void {
    this.currentPelvicActive = false;
    this.currentLeanActive = false;
    const trunk = computeHipTrunkAngleDeg(keypoints);
    const shX = computeHipShoulderMidX(keypoints);
    const width = computeHipShoulderWidth(keypoints);

    if (trunk !== null && this.trunkSamples.length < HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.trunkSamples.push(trunk);
      if (this.trunkSamples.length === HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineTrunk = hipCompMean(this.trunkSamples);
      }
    }
    if (shX !== null && width !== null
        && this.shoulderXSamples.length < HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.shoulderXSamples.push(shX);
      this.widthSamples.push(width);
      if (this.shoulderXSamples.length === HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineShoulderX = hipCompMean(this.shoulderXSamples);
        this.baselineWidth = hipCompMean(this.widthSamples);
      }
    }

    if (trunk !== null && this.baselineTrunk !== null) {
      const dev = Math.abs(trunk - this.baselineTrunk);
      if (dev > this.pelvicPeakDev) this.pelvicPeakDev = dev;
      if (dev > HIP_PELVIC_TILT_THRESHOLD_DEG) {
        this.pelvicFlagged = true;
        this.currentPelvicActive = true;
      }
    }
    if (shX !== null && this.baselineShoulderX !== null
        && this.baselineWidth !== null && this.baselineWidth > 1e-3) {
      const dev = Math.abs(shX - this.baselineShoulderX);
      const frac = dev / this.baselineWidth;
      if (frac > this.leanPeakFrac) this.leanPeakFrac = frac;
      if (frac > HIP_TRUNK_LEAN_THRESHOLD_FRAC) {
        this.leanFlagged = true;
        this.currentLeanActive = true;
      }
    }

    this.frameCounter += 1;
  }

  currentFlags(): Compensation[] {
    const out: Compensation[] = [];
    if (this.currentPelvicActive) {
      out.push({ type: "pelvic_tilt_posterior", label: "Posterior Pelvic Tilt", severity: "high", flagged: true });
    }
    if (this.currentLeanActive) {
      out.push({ type: "hip_trunk_lean", label: "Trunk Lean", severity: "medium", flagged: true });
    }
    return out;
  }

  finish(): Compensation[] {
    return [
      {
        type: "pelvic_tilt_posterior",
        label: "Posterior Pelvic Tilt",
        severity: "high",
        flagged: this.pelvicFlagged,
        details: `Peak trunk-angle deviation ${this.pelvicPeakDev.toFixed(1)}° from baseline (threshold ${HIP_PELVIC_TILT_THRESHOLD_DEG}°)`,
      },
      {
        type: "hip_trunk_lean",
        label: "Trunk Lean",
        severity: "medium",
        flagged: this.leanFlagged,
        details: `Peak shoulder displacement ${(this.leanPeakFrac * 100).toFixed(1)} % of shoulder-width (threshold ${(HIP_TRUNK_LEAN_THRESHOLD_FRAC * 100).toFixed(0)} %)`,
      },
    ];
  }
}

export class HipExtensionCompensationTracker extends HipBaseTracker {
  private trunkSamples: number[] = [];
  private shoulderXSamples: number[] = [];
  private widthSamples: number[] = [];
  private baselineTrunk: number | null = null;
  private baselineShoulderX: number | null = null;
  private baselineWidth: number | null = null;
  private pelvicPeakDev = 0;
  private leanPeakFrac = 0;
  private pelvicFlagged = false;
  private leanFlagged = false;
  private currentPelvicActive = false;
  private currentLeanActive = false;

  feed(keypoints: Keypoint[]): void {
    this.currentPelvicActive = false;
    this.currentLeanActive = false;
    const trunk = computeHipTrunkAngleDeg(keypoints);
    const shX = computeHipShoulderMidX(keypoints);
    const width = computeHipShoulderWidth(keypoints);

    if (trunk !== null && this.trunkSamples.length < HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.trunkSamples.push(trunk);
      if (this.trunkSamples.length === HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineTrunk = hipCompMean(this.trunkSamples);
      }
    }
    if (shX !== null && width !== null
        && this.shoulderXSamples.length < HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.shoulderXSamples.push(shX);
      this.widthSamples.push(width);
      if (this.shoulderXSamples.length === HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineShoulderX = hipCompMean(this.shoulderXSamples);
        this.baselineWidth = hipCompMean(this.widthSamples);
      }
    }

    if (trunk !== null && this.baselineTrunk !== null) {
      const dev = Math.abs(trunk - this.baselineTrunk);
      if (dev > this.pelvicPeakDev) this.pelvicPeakDev = dev;
      if (dev > HIP_PELVIC_TILT_THRESHOLD_DEG) {
        this.pelvicFlagged = true;
        this.currentPelvicActive = true;
      }
    }
    if (shX !== null && this.baselineShoulderX !== null
        && this.baselineWidth !== null && this.baselineWidth > 1e-3) {
      const dev = Math.abs(shX - this.baselineShoulderX);
      const frac = dev / this.baselineWidth;
      if (frac > this.leanPeakFrac) this.leanPeakFrac = frac;
      if (frac > HIP_TRUNK_LEAN_THRESHOLD_FRAC) {
        this.leanFlagged = true;
        this.currentLeanActive = true;
      }
    }

    this.frameCounter += 1;
  }

  currentFlags(): Compensation[] {
    const out: Compensation[] = [];
    if (this.currentPelvicActive) {
      out.push({ type: "pelvic_tilt_anterior", label: "Anterior Pelvic Tilt", severity: "high", flagged: true });
    }
    if (this.currentLeanActive) {
      out.push({ type: "hip_trunk_lean", label: "Trunk Lean", severity: "medium", flagged: true });
    }
    return out;
  }

  finish(): Compensation[] {
    return [
      {
        type: "pelvic_tilt_anterior",
        label: "Anterior Pelvic Tilt",
        severity: "high",
        flagged: this.pelvicFlagged,
        details: `Peak trunk-angle deviation ${this.pelvicPeakDev.toFixed(1)}° from baseline (threshold ${HIP_PELVIC_TILT_THRESHOLD_DEG}°)`,
      },
      {
        type: "hip_trunk_lean",
        label: "Trunk Lean",
        severity: "medium",
        flagged: this.leanFlagged,
        details: `Peak shoulder displacement ${(this.leanPeakFrac * 100).toFixed(1)} % of shoulder-width (threshold ${(HIP_TRUNK_LEAN_THRESHOLD_FRAC * 100).toFixed(0)} %)`,
      },
    ];
  }
}

export class HipRotationCompensationTracker extends HipBaseTracker {
  private readonly side: "left" | "right";
  private widthSamples: number[] = [];
  private baselineWidth: number | null = null;
  private rotationPeakFrac = 0;
  private rotationFlagged = false;
  private currentRotationActive = false;
  private kneeXSamples: number[] = [];
  private baselineKneeX: number | null = null;
  private kneeTiltPeakFrac = 0;
  private kneeFlagged = false;
  private currentKneeActive = false;

  constructor(side: "left" | "right") {
    super();
    this.side = side;
  }

  feed(keypoints: Keypoint[]): void {
    this.currentRotationActive = false;
    this.currentKneeActive = false;

    const width = computeHipShoulderWidth(keypoints);
    const idx = SIDE_INDICES[this.side];
    const knee = keypoints[idx.knee];
    const kneeX = knee && (knee.score ?? 0) >= VIS_THRESHOLD ? knee.x : null;

    if (width !== null
        && this.widthSamples.length < HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.widthSamples.push(width);
      if (this.widthSamples.length === HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineWidth = hipCompMean(this.widthSamples);
      }
    }
    if (kneeX !== null
        && this.kneeXSamples.length < HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.kneeXSamples.push(kneeX);
      if (this.kneeXSamples.length === HIP_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineKneeX = hipCompMean(this.kneeXSamples);
      }
    }

    if (width !== null && this.baselineWidth !== null && this.baselineWidth > 1e-3) {
      const shrink = this.baselineWidth - width;
      const frac = shrink / this.baselineWidth;
      if (frac > this.rotationPeakFrac) this.rotationPeakFrac = frac;
      if (frac > HIP_TRUNK_ROTATION_THRESHOLD_FRAC) {
        this.rotationFlagged = true;
        this.currentRotationActive = true;
      }
    }
    if (kneeX !== null && this.baselineKneeX !== null
        && this.baselineWidth !== null && this.baselineWidth > 1e-3) {
      const dev = Math.abs(kneeX - this.baselineKneeX);
      const frac = dev / this.baselineWidth;
      if (frac > this.kneeTiltPeakFrac) this.kneeTiltPeakFrac = frac;
      if (frac > HIP_KNEE_TILT_THRESHOLD_FRAC) {
        this.kneeFlagged = true;
        this.currentKneeActive = true;
      }
    }

    this.frameCounter += 1;
  }

  currentFlags(): Compensation[] {
    const out: Compensation[] = [];
    if (this.currentRotationActive) {
      out.push({ type: "hip_trunk_rotation", label: "Trunk Rotation", severity: "high", flagged: true });
    }
    if (this.currentKneeActive) {
      out.push({ type: "knee_tilt", label: "Knee Tilt", severity: "medium", flagged: true });
    }
    return out;
  }

  finish(): Compensation[] {
    return [
      {
        type: "hip_trunk_rotation",
        label: "Trunk Rotation",
        severity: "high",
        flagged: this.rotationFlagged,
        details: `Peak shoulder-width shrink ${(this.rotationPeakFrac * 100).toFixed(1)} % of baseline (threshold ${(HIP_TRUNK_ROTATION_THRESHOLD_FRAC * 100).toFixed(0)} %)`,
      },
      {
        type: "knee_tilt",
        label: "Knee Tilt",
        severity: "medium",
        flagged: this.kneeFlagged,
        details: `Peak knee lateral deviation ${(this.kneeTiltPeakFrac * 100).toFixed(1)} % of shoulder-width (threshold ${(HIP_KNEE_TILT_THRESHOLD_FRAC * 100).toFixed(0)} %)`,
      },
    ];
  }
}
