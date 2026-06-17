// UI metadata + browser-side angle math for the knee biomech flow.
// Both flexion and extension use the same hip-knee-ankle interior angle;
// the difference is only the target range and clinical interpretation.

import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

export type KneeMovementId = "flexion" | "extension" | "flexion_extension";

export interface KneeMovement {
  id: string;
  label: string;
  description: string;
  /** Primary direction's normal range. For the merged
   *  flexion_extension test this is the flexion range (peak bend);
   *  for the legacy single tests it's just the test's normal range. */
  target: [number, number];
  /** True for the merged test that captures BOTH flexion and
   *  extension in a single recording. Live and upload modes
   *  track min + max knee angle simultaneously when this is set. */
  merged?: boolean;
  /** Display label for the primary direction in the merged dual
   *  readout / report. */
  primaryLabel?: string;
  /** Display label for the secondary direction. */
  secondaryLabel?: string;
  /** Normal range for the secondary direction. Required when merged.
   *  For knee extension this is the "residual flexion at full
   *  extension" range — lower is better (0° = perfectly straight).
   *  AssessmentReport.classify treats in-range as "good" regardless
   *  of which way is anatomically better. */
  secondaryTarget?: [number, number];
  /** Hidden from the movement chooser. Legacy single-direction entries
   *  are kept in the metadata table so saved reports referencing them
   *  still resolve labels / targets without breaking, but they no
   *  longer appear when starting a new trial. */
  hidden?: boolean;
  /** Optional reference illustration. See MovementGrid's
   *  MovementOption.imageUrl for the path convention. */
  imageUrl?: string;
}

export const KNEE_MOVEMENTS: KneeMovement[] = [
  // Combined Flexion + Extension. One recording captures both peaks:
  // the maximum knee bend (flexion) and the residual flexion at the
  // patient's straightest position (extension deficit).
  {
    id: "flexion_extension",
    label: "Flexion + Extension",
    description:
      "Bend the knee fully (peak flexion), then straighten it back to fully extended (peak extension). One session captures both ends of the ROM.",
    target: [125, 145],
    merged: true,
    primaryLabel: "Flexion",
    secondaryLabel: "Extension",
    // Extension target is the residual-flexion range when the knee is
    // "fully extended" — 0° means perfectly straight, lower is better.
    // Range widened from the strict clinical [0, 5] to [0, 10] to
    // absorb the inherent precision limit of 2D pose estimation:
    // MoveNet's hip/knee/ankle keypoint placement carries ~5-10° of
    // systematic noise even when the patient's knee is anatomically
    // straight (the knee keypoint sits near the patella rather than
    // the true joint axis, the hip keypoint near the greater
    // trochanter, etc.). [0, 10] flags genuine extension deficits
    // while not penalising healthy patients for the camera's
    // measurement uncertainty.
    secondaryTarget: [0, 10],
    imageUrl: "/images/biomech/knee/knee_flexion_extension.png",
  },
  // Legacy single-direction entries — kept so saved reports referring
  // to "flexion" or "extension" alone still resolve a label/target
  // from this table. Hidden from the chooser since the merged entry
  // above is the new default.
  {
    id: "flexion",
    label: "Flexion",
    description: "Bend the knee — bringing the heel toward the buttock",
    target: [125, 145],
    hidden: true,
  },
  {
    id: "extension",
    label: "Extension",
    description: "Straighten a bent knee back to a fully extended leg",
    target: [125, 145],
    hidden: true,
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

// ─── Compensation detection (Knee FE) — mirror of knee.ts ────────
//
// Same math, same thresholds, same tracker class. Only difference is
// the keypoint source (BlazePose-tfjs LiveKeypoint).

export type CompensationSeverity = "high" | "medium" | "low";

export interface Compensation {
  type: "thigh_movement" | "knee_trunk_lean";
  label: string;
  severity: CompensationSeverity;
  flagged: boolean;
  details?: string;
}

const KNEE_THIGH_MOVEMENT_THRESHOLD_DEG = 15;
const KNEE_TRUNK_LEAN_THRESHOLD_DEG = 12;
const KNEE_COMPENSATION_BASELINE_FRAME_COUNT = 10;

export function computeKneeThighAngleDeg(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const hip = keypoints[idx.hip];
  const knee = keypoints[idx.knee];
  if (!hip || !knee) return null;
  if ((hip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((knee.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = knee.x - hip.x;
  const dy = knee.y - hip.y;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  return (Math.atan2(dx, dy) * 180) / Math.PI;
}

export function computeKneeTrunkAngleDeg(
  keypoints: Keypoint[],
): number | null {
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

function kneeCompMean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

export class KneeFlexExtCompensationTracker {
  private readonly side: "left" | "right";
  private thighSamples: number[] = [];
  private trunkSamples: number[] = [];
  private baselineThigh: number | null = null;
  private baselineTrunk: number | null = null;
  private thighPeakDev = 0;
  private trunkPeakDev = 0;
  private thighFlagged = false;
  private trunkFlagged = false;
  private currentThighActive = false;
  private currentTrunkActive = false;
  private frameCounter = 0;
  private primaryPeakFrame: number | null = null;
  private secondaryPeakFrame: number | null = null;

  constructor(side: "left" | "right") {
    this.side = side;
  }

  feed(keypoints: Keypoint[]): void {
    this.currentThighActive = false;
    this.currentTrunkActive = false;

    const thigh = computeKneeThighAngleDeg(keypoints, this.side);
    const trunk = computeKneeTrunkAngleDeg(keypoints);

    if (thigh !== null
        && this.thighSamples.length < KNEE_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.thighSamples.push(thigh);
      if (this.thighSamples.length === KNEE_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineThigh = kneeCompMean(this.thighSamples);
      }
    }
    if (trunk !== null
        && this.trunkSamples.length < KNEE_COMPENSATION_BASELINE_FRAME_COUNT) {
      this.trunkSamples.push(trunk);
      if (this.trunkSamples.length === KNEE_COMPENSATION_BASELINE_FRAME_COUNT) {
        this.baselineTrunk = kneeCompMean(this.trunkSamples);
      }
    }

    if (thigh !== null && this.baselineThigh !== null) {
      const dev = Math.abs(thigh - this.baselineThigh);
      if (dev > this.thighPeakDev) this.thighPeakDev = dev;
      if (dev > KNEE_THIGH_MOVEMENT_THRESHOLD_DEG) {
        this.thighFlagged = true;
        this.currentThighActive = true;
      }
    }
    if (trunk !== null && this.baselineTrunk !== null) {
      const dev = Math.abs(trunk - this.baselineTrunk);
      if (dev > this.trunkPeakDev) this.trunkPeakDev = dev;
      if (dev > KNEE_TRUNK_LEAN_THRESHOLD_DEG) {
        this.trunkFlagged = true;
        this.currentTrunkActive = true;
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
    if (this.currentThighActive) {
      out.push({ type: "thigh_movement", label: "Thigh Movement", severity: "high", flagged: true });
    }
    if (this.currentTrunkActive) {
      out.push({ type: "knee_trunk_lean", label: "Trunk Lean", severity: "medium", flagged: true });
    }
    return out;
  }

  finish(): Compensation[] {
    return [
      {
        type: "thigh_movement",
        label: "Thigh Movement",
        severity: "high",
        flagged: this.thighFlagged,
        details: `Peak thigh-angle deviation ${this.thighPeakDev.toFixed(1)}° from baseline (threshold ${KNEE_THIGH_MOVEMENT_THRESHOLD_DEG}°)`,
      },
      {
        type: "knee_trunk_lean",
        label: "Trunk Lean",
        severity: "medium",
        flagged: this.trunkFlagged,
        details: `Peak trunk-angle deviation ${this.trunkPeakDev.toFixed(1)}° from baseline (threshold ${KNEE_TRUNK_LEAN_THRESHOLD_DEG}°)`,
      },
    ];
  }
}
