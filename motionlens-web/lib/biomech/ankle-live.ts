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

export type AnkleMovementId = "flexion" | "extension" | "flexion_extension";

export interface AnkleMovement {
  id: AnkleMovementId;
  label: string;
  description: string;
  target: [number, number];
  /** Merged two-direction entries carry a secondary direction with its
   *  own target (mirrors knee flexion_extension / hip rotation). */
  merged?: boolean;
  primaryLabel?: string;
  secondaryLabel?: string;
  secondaryTarget?: [number, number];
  /** Hidden from the movement chooser. Legacy single-direction entries
   *  are kept so saved reports referencing them still resolve labels /
   *  targets, but they no longer appear when starting a new trial. */
  hidden?: boolean;
  /** Optional reference illustration. See MovementGrid's
   *  MovementOption.imageUrl for the path convention. */
  imageUrl?: string;
}

export const ANKLE_MOVEMENTS: AnkleMovement[] = [
  // Combined Dorsiflexion + Plantarflexion. One recording captures
  // both peaks. Measurement = angle between the FOOT SOLE LINE
  // (heel → toe) and the SHIN LINE (knee–ankle). Neutral is the
  // geometric 90° crossing — no timed neutral capture needed:
  //   angle < 90° (foot pulled toward shin)  → dorsiflexion  = 90 − θ
  //   angle > 90° (foot pointed away)        → plantarflexion = θ − 90
  {
    id: "flexion_extension",
    label: "Dorsiflexion + Plantarflexion",
    description:
      "Seated, leg extended. Pull the foot up toward the shin (dorsiflexion), then point the toes down like a gas pedal (plantarflexion). One session captures both ends of the ROM.",
    target: [15, 25],
    merged: true,
    primaryLabel: "Dorsiflexion",
    secondaryLabel: "Plantarflexion",
    secondaryTarget: [40, 55],
    imageUrl: "/images/biomech/ankle/ankle_dorsi_plantar.png",
  },
  // Legacy single-direction entries — kept so saved reports referring
  // to "flexion" or "extension" alone still resolve a label/target.
  // Hidden from the chooser since the merged entry is the new default.
  {
    id: "flexion",
    label: "Dorsiflexion",
    description: "Pull the foot upward (toes toward shin) — knee-to-wall lean",
    target: [15, 25],
    hidden: true,
    imageUrl: "/images/biomech/ankle/ankle_dorsiflexion.png",
  },
  {
    id: "extension",
    label: "Plantarflexion",
    description: "Point the foot down, like pressing a gas pedal",
    target: [40, 55],
    hidden: true,
    imageUrl: "/images/biomech/ankle/ankle_plantarflexion.png",
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
    heel:  LM.LEFT_HEEL,
    foot:  LM.LEFT_FOOT_INDEX,
  },
  right: {
    knee:  LM.RIGHT_KNEE,
    ankle: LM.RIGHT_ANKLE,
    heel:  LM.RIGHT_HEEL,
    foot:  LM.RIGHT_FOOT_INDEX,
  },
} as const;

// Foot landmarks (heel + foot_index) are intrinsically noisier than
// the anchor joints — accept a lower confidence for them so the peak
// frames (where the foot is at its extreme) aren't dropped.
const FOOT_VIS_THRESHOLD = 0.1;

export function computeAnkleAngle(
  movement: AnkleMovementId,
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const knee  = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  const heel  = keypoints[idx.heel];
  const foot  = keypoints[idx.foot];

  if (movement === "flexion_extension") {
    // Merged dorsi + plantar. Angle between the FOOT SOLE LINE
    // (heel → toe) and the SHIN LINE (ankle → knee, pointing up the
    // leg). Neutral (foot ⟂ shin) reads 90°:
    //   • dorsiflexion  (foot toward shin) → θ < 90 → signed = +(90−θ)
    //   • plantarflexion (foot away)       → θ > 90 → signed = −(θ−90)
    // The SIGNED value routes the live merged slots (positive =
    // primary/dorsi, negative = secondary/plantar); each slot's peak
    // magnitude is |signed|.
    for (const k of [knee, ankle]) {
      if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
    }
    for (const k of [heel, foot]) {
      if (!k || (k.score ?? 0) < FOOT_VIS_THRESHOLD) return null;
    }
    const sx = knee.x - ankle.x;   // shin: ankle → knee
    const sy = knee.y - ankle.y;
    const fx = foot.x - heel.x;    // sole: heel → toe
    const fy = foot.y - heel.y;
    if (Math.hypot(sx, sy) < 1e-6 || Math.hypot(fx, fy) < 1e-6) return null;
    const dot = sx * fx + sy * fy;
    const cross = sx * fy - sy * fx;
    const theta = (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
    return 90 - theta;
  }

  for (const k of [knee, ankle, foot]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }
  // Legacy single-direction math — interior angle at the ANKLE vertex
  // between the shin and foot vectors. Kept for saved-report parity;
  // the chooser no longer offers these movements.
  const interior = angleAt(knee.x, knee.y, ankle.x, ankle.y, foot.x, foot.y);
  if (movement === "flexion") {
    return 90 - interior;
  }
  return interior - 90;
}

// ─── Compensation detection (Ankle dorsi / plantar) — mirror ─────
// Same math, same thresholds, BlazePose-tfjs LiveKeypoint source.

export type CompensationSeverity = "high" | "medium" | "low";

export interface Compensation {
  type: "ankle_knee_movement" | "ankle_leg_lift";
  label: string;
  severity: CompensationSeverity;
  flagged: boolean;
  details?: string;
}

const ANKLE_KNEE_MOVE_DORSI_THRESHOLD_DEG = 40;
const ANKLE_KNEE_MOVE_PLANTAR_THRESHOLD_DEG = 15;
const ANKLE_LEG_LIFT_THRESHOLD_FRAC = 0.10;
const ANKLE_COMPENSATION_BASELINE_FRAME_COUNT = 10;

const COMP_SIDE_INDICES = {
  left:  { hip: LM.LEFT_HIP,  knee: LM.LEFT_KNEE,  ankle: LM.LEFT_ANKLE  },
  right: { hip: LM.RIGHT_HIP, knee: LM.RIGHT_KNEE, ankle: LM.RIGHT_ANKLE },
} as const;

function interiorAngleAt(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number,
): number {
  const v1x = ax - bx, v1y = ay - by;
  const v2x = cx - bx, v2y = cy - by;
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
}

export function computeAnkleKneeInteriorAngle(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = COMP_SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  if (!hip || !knee || !ankle) return null;
  if ((hip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((knee.score ?? 0) < VIS_THRESHOLD) return null;
  if ((ankle.score ?? 0) < VIS_THRESHOLD) return null;
  return interiorAngleAt(hip.x, hip.y, knee.x, knee.y, ankle.x, ankle.y);
}

export function computeAnkleShoulderWidth(
  keypoints: Keypoint[],
): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!lSh || !rSh) return null;
  if ((lSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rSh.score ?? 0) < VIS_THRESHOLD) return null;
  return Math.abs(rSh.x - lSh.x);
}

function ankleCompMean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

abstract class AnkleBaseTracker {
  protected readonly side: "left" | "right";
  protected readonly kneeThresholdDeg: number;

  protected kneeSamples: number[] = [];
  protected legYSamples: number[] = [];
  protected widthSamples: number[] = [];
  protected baselineKnee: number | null = null;
  protected baselineLegY: number | null = null;
  protected baselineWidth: number | null = null;
  protected kneePeakDev = 0;
  protected legLiftPeakFrac = 0;
  protected kneeFlagged = false;
  protected legLiftFlagged = false;
  protected currentKneeActive = false;
  protected currentLegLiftActive = false;
  protected frameCounter = 0;
  protected primaryPeakFrame: number | null = null;
  protected secondaryPeakFrame: number | null = null;

  constructor(side: "left" | "right", kneeThresholdDeg: number) {
    this.side = side;
    this.kneeThresholdDeg = kneeThresholdDeg;
  }

  feed(keypoints: Keypoint[]): void {
    this.currentKneeActive = false;
    this.currentLegLiftActive = false;

    const kneeInterior = computeAnkleKneeInteriorAngle(keypoints, this.side);
    const idx = COMP_SIDE_INDICES[this.side];
    const hip = keypoints[idx.hip];
    const knee = keypoints[idx.knee];
    const legY = (hip && (hip.score ?? 0) >= VIS_THRESHOLD
                  && knee && (knee.score ?? 0) >= VIS_THRESHOLD)
                   ? (hip.y + knee.y) / 2
                   : null;
    const width = computeAnkleShoulderWidth(keypoints);

    if (kneeInterior !== null
        && this.kneeSamples.length < ANKLE_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.kneeSamples.push(kneeInterior);
      if (this.kneeSamples.length === ANKLE_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineKnee = ankleCompMean(this.kneeSamples);
      }
    }
    if (legY !== null && width !== null
        && this.legYSamples.length < ANKLE_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.legYSamples.push(legY);
      this.widthSamples.push(width);
      if (this.legYSamples.length === ANKLE_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineLegY = ankleCompMean(this.legYSamples);
        this.baselineWidth = ankleCompMean(this.widthSamples);
      }
    }

    if (kneeInterior !== null && this.baselineKnee !== null) {
      const dev = Math.abs(kneeInterior - this.baselineKnee);
      if (dev > this.kneePeakDev) this.kneePeakDev = dev;
      if (dev > this.kneeThresholdDeg) {
        this.kneeFlagged = true;
        this.currentKneeActive = true;
      }
    }
    if (legY !== null && this.baselineLegY !== null
        && this.baselineWidth !== null && this.baselineWidth > 1e-3) {
      const dev = Math.abs(legY - this.baselineLegY);
      const frac = dev / this.baselineWidth;
      if (frac > this.legLiftPeakFrac) this.legLiftPeakFrac = frac;
      if (frac > ANKLE_LEG_LIFT_THRESHOLD_FRAC) {
        this.legLiftFlagged = true;
        this.currentLegLiftActive = true;
      }
    }

    this.frameCounter += 1;
  }

  markPrimaryPeak(): void {
    this.primaryPeakFrame = Math.max(0, this.frameCounter - 1);
  }
  markSecondaryPeak(): void {
    this.secondaryPeakFrame = Math.max(0, this.frameCounter - 1);
  }

  currentFlags(): Compensation[] {
    const out: Compensation[] = [];
    if (this.currentKneeActive) {
      out.push({ type: "ankle_knee_movement", label: "Knee Movement", severity: "high", flagged: true });
    }
    if (this.currentLegLiftActive) {
      out.push({ type: "ankle_leg_lift", label: "Leg Lift", severity: "medium", flagged: true });
    }
    return out;
  }

  finish(): Compensation[] {
    return [
      {
        type: "ankle_knee_movement",
        label: "Knee Movement",
        severity: "high",
        flagged: this.kneeFlagged,
        details: `Peak knee-angle deviation ${this.kneePeakDev.toFixed(1)}° from baseline (threshold ${this.kneeThresholdDeg}°)`,
      },
      {
        type: "ankle_leg_lift",
        label: "Leg Lift",
        severity: "medium",
        flagged: this.legLiftFlagged,
        details: `Peak hip+knee vertical shift ${(this.legLiftPeakFrac * 100).toFixed(1)} % of shoulder-width (threshold ${(ANKLE_LEG_LIFT_THRESHOLD_FRAC * 100).toFixed(0)} %)`,
      },
    ];
  }
}

export class AnkleFlexionCompensationTracker extends AnkleBaseTracker {
  constructor(side: "left" | "right") {
    super(side, ANKLE_KNEE_MOVE_DORSI_THRESHOLD_DEG);
  }
}

export class AnkleExtensionCompensationTracker extends AnkleBaseTracker {
  constructor(side: "left" | "right") {
    super(side, ANKLE_KNEE_MOVE_PLANTAR_THRESHOLD_DEG);
  }
}
