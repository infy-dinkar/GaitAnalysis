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
  /** Optional reference illustration. See MovementGrid's
   *  MovementOption.imageUrl for the path convention. */
  imageUrl?: string;
}

export const ANKLE_MOVEMENTS: AnkleMovement[] = [
  {
    id: "flexion",
    label: "Dorsiflexion",
    description: "Pull the foot upward (toes toward shin) — knee-to-wall lean",
    target: [15, 25],
    imageUrl: "/images/biomech/ankle/ankle_dorsiflexion.png",
  },
  {
    id: "extension",
    label: "Plantarflexion",
    description: "Point the foot down, like pressing a gas pedal",
    target: [40, 55],
    imageUrl: "/images/biomech/ankle/ankle_plantarflexion.png",
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

// ─── Compensation detection (Ankle dorsi / plantar) ─────────────
//
// Two compensations, both baseline-relative whole-trial. Defined
// GENERICALLY (no foot landmarks needed) so this works in MoveNet
// LIVE mode where heel + foot_index aren't available:
//
//   1. Ankle Knee Movement (HIGH) — interior knee angle change
//      from first-10-frame baseline. Per-exercise thresholds:
//        • dorsi (knee-to-wall): 40° (the knee SHOULD bend
//          forward; only flag excessive movement)
//        • plantar (seated leg-out): 15° (leg should stay
//          straight; tighter clamp)
//   2. Ankle Leg Lift (MEDIUM) — mean of hip.y + knee.y shift
//      from baseline / baseline shoulder-width. Patient lifting
//      the whole leg / shifting the thigh instead of pivoting at
//      the ankle.

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
  protected legYSamples: number[] = [];     // (hip.y + knee.y) / 2
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
