// UI metadata + browser-side angle math for the shoulder biomech flow.
// The compute function ports shoulder_engine.py exactly so live and
// final measurements use the same formulas. Used by the live mode
// (client-side TF.js BlazePose + this math). Server-side upload mode
// still uses the Python engine of record for the authoritative report.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

export type ShoulderMovementId =
  | "flexion"
  | "extension"
  | "abduction"
  | "adduction"
  | "external_rotation"
  | "internal_rotation";

export interface ShoulderMovement {
  id: string;
  label: string;
  description: string;
  /** Primary direction's normal range (for merged movements this is
   *  the "A" direction — external for rotation, abduction for
   *  abduction/adduction). For single-direction movements this is
   *  simply the movement's normal range. */
  target: [number, number];
  /** True when this movement bundles two directions captured in a
   *  single recording session (rotation = external + internal,
   *  abduction_adduction = abduction + adduction). LiveAssessment
   *  switches to a dual-peak state machine when this is set. */
  merged?: boolean;
  /** Display label for the primary direction in the dual-readout UI.
   *  Only meaningful when merged === true. */
  primaryLabel?: string;
  /** Display label for the secondary direction in the dual-readout UI. */
  secondaryLabel?: string;
  /** Normal range for the secondary direction. Required when merged. */
  secondaryTarget?: [number, number];
  /** Hidden from the movement chooser. Kept in the lookup table so
   *  legacy saved reports referencing the old (now-merged) directional
   *  IDs can still resolve labels / targets without breaking. */
  hidden?: boolean;
}

export const SHOULDER_MOVEMENTS: ShoulderMovement[] = [
  { id: "flexion",   label: "Flexion",   description: "Lift the arm forward and overhead",     target: [150, 180] },
  { id: "extension", label: "Extension", description: "Move the arm backward behind the body", target: [45, 60] },
  // Combined Abduction + Adduction. One recording captures both
  // directions; the live engine detects which way the elbow is
  // travelling and tracks a separate peak for each.
  {
    id: "abduction_adduction",
    label: "Abduction + Adduction",
    description:
      "Lift the arm sideways away from the body (abduction), then bring it across the chest (adduction). One session captures both peaks.",
    target: [150, 180],
    merged: true,
    primaryLabel: "Abduction",
    secondaryLabel: "Adduction",
    secondaryTarget: [30, 50],
  },
  // Combined External + Internal Rotation. Patient rotates outward,
  // returns to neutral, then rotates inward. Both peaks are captured
  // in the same trial.
  {
    id: "rotation",
    label: "Rotation (External + Internal)",
    description:
      "With elbow tucked at the side and bent 90°, rotate the forearm outward (external) and inward (internal). One session captures both peaks.",
    target: [70, 90],
    merged: true,
    primaryLabel: "External Rotation",
    secondaryLabel: "Internal Rotation",
    secondaryTarget: [60, 80],
  },
  // Legacy single-direction entries — kept in the lookup table so
  // saved reports that referenced these IDs still resolve their label
  // / target, but hidden from the chooser since the merged versions
  // above replace them in the UI flow.
  { id: "abduction",         label: "Abduction",         description: "Lift the arm sideways away from the body", target: [150, 180], hidden: true },
  { id: "adduction",         label: "Adduction",         description: "Bring the arm across the chest",            target: [30, 50],  hidden: true },
  { id: "external_rotation", label: "External Rotation", description: "Rotate the forearm outward (elbow at 90°)", target: [70, 90],  hidden: true },
  { id: "internal_rotation", label: "Internal Rotation", description: "Rotate the forearm inward (elbow at 90°)",  target: [60, 80],  hidden: true },
];

// BlazePose-tfjs scores are lower than MediaPipe's `visibility` field —
// typical visible joints score 0.2-0.4. 0.15 lets math run on most
// in-frame joints without false-passing fully occluded ones.
const VIS_THRESHOLD = 0.15;

// Signed angle from v1 to v2 in degrees (image-coord 2D), in (-180, 180].
function angleBetween(
  v1x: number, v1y: number,
  v2x: number, v2y: number,
): number {
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(cross, dot) * 180) / Math.PI;
}

function shoulderFlexionExtension(
  s: Keypoint, e: Keypoint, h: Keypoint,
): number {
  const tx = h.x - s.x, ty = h.y - s.y;       // trunk_down
  const ax = e.x - s.x, ay = e.y - s.y;       // arm
  return -angleBetween(tx, ty, ax, ay);
}

function shoulderAbductionAdduction(
  s: Keypoint, e: Keypoint, h: Keypoint,
): number {
  const tx = h.x - s.x, ty = h.y - s.y;
  const ax = e.x - s.x, ay = e.y - s.y;
  return Math.abs(angleBetween(tx, ty, ax, ay));
}

function shoulderRotation(
  _s: Keypoint, e: Keypoint, w: Keypoint,
): number {
  const fx = w.x - e.x, fy = w.y - e.y;       // forearm
  // reference = (1, 0)
  return Math.abs(angleBetween(1, 0, fx, fy));
}

const SIDE_INDICES = {
  left:  { shoulder: LM.LEFT_SHOULDER,  elbow: LM.LEFT_ELBOW,  wrist: LM.LEFT_WRIST,  hip: LM.LEFT_HIP  },
  right: { shoulder: LM.RIGHT_SHOULDER, elbow: LM.RIGHT_ELBOW, wrist: LM.RIGHT_WRIST, hip: LM.RIGHT_HIP },
} as const;

export function computeShoulderAngle(
  movement: ShoulderMovementId,
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const needed: number[] = [idx.shoulder, idx.elbow, idx.hip];
  if (movement === "external_rotation" || movement === "internal_rotation") {
    needed.push(idx.wrist);
  }
  for (const k of needed) {
    const kp = keypoints[k];
    if (!kp || (kp.score ?? 0) < VIS_THRESHOLD) return null;
  }
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  const h = keypoints[idx.hip];
  if (movement === "flexion" || movement === "extension") {
    return shoulderFlexionExtension(s, e, h);
  }
  if (movement === "abduction" || movement === "adduction") {
    return shoulderAbductionAdduction(s, e, h);
  }
  const w = keypoints[idx.wrist];
  return shoulderRotation(s, e, w);
}

// ─── Calibrated rotation (baseline-locked) ──────────────────────
//
// Geometry — same idea as the neck-rotation fix, applied to the
// shoulder. With the elbow tucked at the side and bent 90°, the
// forearm sweeps in a horizontal plane around the shoulder axis.
// At the patient's neutral position (forearm pointing forward at
// the camera), the forearm projection in the 2D image is near zero.
// As the patient rotates externally or internally, the forearm
// projection grows. Mathematically:
//
//     forearmProjectedLength = R_forearmLength × sin(θ)
//     ⇒ θ = arcsin(forearmProjectedLength / R_forearmLength)
//
// We can't measure R_forearmLength directly at neutral (projection ≈
// 0), so we use the upper-arm length as a proxy — anatomically
// forearm length ≈ upper-arm length for most adults (within ±10%).
// The upper arm is fully visible in 2D when tucked at the side
// (vertical pose), so its pixel length is a reliable scale.
//
// Sign is intentionally NOT returned — the operator already selects
// "external" or "internal" before recording, so direction is known
// from the test selection. We just report the magnitude (always ≥ 0).
// This also matches the report-card display, which renders peak
// magnitude vs target range without a +/- distinction.

export interface ShoulderRotationCalibration {
  /** Upper-arm pixel length at calibration (shoulder→elbow). Used
   *  as a proxy for forearm length in the arcsin formula. */
  R_upperArmLength: number;
  /** Side this baseline was captured for — guards against accidental
   *  cross-side application if the operator switches mid-flow. */
  side: "left" | "right";
}

/** Maximum forearm-projection-to-upper-arm ratio for "patient is at
 *  neutral rotation" (forearm pointing at the camera). 0.20 = forearm
 *  projects to no more than 20% of the upper-arm length. Generous
 *  enough to tolerate slight angle off-camera but tight enough that
 *  the patient can't be holding the forearm fully sideways. */
const SHOULDER_NEUTRAL_FOREARM_RATIO_MAX = 0.20;

/** Minimum upper-arm pixel length to accept calibration. Below this
 *  the patient is too far from the camera, MoveNet's keypoints are
 *  too noisy, and the rotation readings would be unreliable. */
const MIN_UPPER_ARM_PX = 30;

/** True when MoveNet sees shoulder + elbow + wrist for the given
 *  side AND the forearm is approximately pointing at the camera (low
 *  2D projection). Used by the live-capture flow to decide when to
 *  auto-lock the shoulder-rotation baseline. */
export function isShoulderRotationNeutral(
  keypoints: Keypoint[],
  side: "left" | "right",
): boolean {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  const w = keypoints[idx.wrist];
  for (const k of [s, e, w]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return false;
  }
  const upperArmLen = Math.hypot(e.x - s.x, e.y - s.y);
  if (upperArmLen < MIN_UPPER_ARM_PX) return false;
  const forearmLen = Math.hypot(w.x - e.x, w.y - e.y);
  return forearmLen / upperArmLen <= SHOULDER_NEUTRAL_FOREARM_RATIO_MAX;
}

/** Snapshot the patient's neutral-position upper-arm length as the
 *  rotation reference. Called once after isShoulderRotationNeutral
 *  has held true for the calibration-stable window. */
export function captureShoulderRotationBaseline(
  keypoints: Keypoint[],
  side: "left" | "right",
): ShoulderRotationCalibration | null {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  if (!s || !e) return null;
  if ((s.score ?? 0) < VIS_THRESHOLD || (e.score ?? 0) < VIS_THRESHOLD) return null;
  const len = Math.hypot(e.x - s.x, e.y - s.y);
  if (len < MIN_UPPER_ARM_PX) return null;
  return { R_upperArmLength: len, side };
}

/** Typical anatomical ratio of forearm length to upper-arm length in
 *  adults. The arcsin formula uses upper-arm pixel length as a proxy
 *  for forearm length (we can't measure forearm directly at neutral
 *  because the forearm is pointing at the camera and projects to
 *  near zero). Most adults' forearms are ~88% as long as their
 *  upper arms; using the upper-arm length as-is therefore makes
 *  full ROM read low (the projection ratio caps around 0.85-0.90,
 *  so asin returns 58°-65° even when the patient is actually at
 *  90° rotation). Scaling the denominator by this constant brings
 *  the output much closer to the patient's true peak rotation. */
const FOREARM_TO_UPPER_ARM_PROXY = 0.88;

/** Calibrated rotation magnitude in degrees, capped at 90°.
 *  Returns null when wrist or elbow visibility drops (typical at the
 *  extreme of rotation when the wrist crosses behind the body). The
 *  upstream peak-tracker holds the last good value in that case. */
export function computeShoulderRotationFromBaseline(
  keypoints: Keypoint[],
  side: "left" | "right",
  baseline: ShoulderRotationCalibration,
): number | null {
  if (baseline.side !== side) return null; // wrong-side baseline guard
  const idx = SIDE_INDICES[side];
  const e = keypoints[idx.elbow];
  const w = keypoints[idx.wrist];
  if (!e || !w) return null;
  if ((e.score ?? 0) < VIS_THRESHOLD || (w.score ?? 0) < VIS_THRESHOLD) return null;
  const forearmProjLen = Math.hypot(w.x - e.x, w.y - e.y);
  // arcsin saturates at 1; clamp the ratio. The
  // FOREARM_TO_UPPER_ARM_PROXY factor compensates for the upper-arm
  // proxy being anatomically longer than the actual forearm, so full
  // ROM reads close to 90° instead of 56°-65°.
  const effectiveRef = baseline.R_upperArmLength * FOREARM_TO_UPPER_ARM_PROXY;
  const ratio = Math.max(0, Math.min(1, forearmProjLen / effectiveRef));
  return (Math.asin(ratio) * 180) / Math.PI;
}

// ─── Direction detection for merged (dual-direction) movements ──
//
// The "rotation" and "abduction_adduction" merged tests record both
// directions in one session. Magnitude is what the existing formulas
// already return; direction is detected from joint positions relative
// to the body's vertical centreline (midpoint of the two shoulders).
//
// Deadband: near neutral, the lateral signal is tiny and direction
// flips on noise. We require a minimum offset (as a fraction of
// shoulder width) before committing to a direction — otherwise return
// null so the caller doesn't touch either peak.

export type ShoulderRotationDirection = "external" | "internal";
export type ShoulderAbAdDirection = "abduction" | "adduction";

/** Minimum lateral offset (as a fraction of shoulder width) before
 *  a direction is committed. Tighter = more responsive but flips on
 *  noise; looser = stable but laggy at small ROMs. 0.05 ≈ 5% of
 *  shoulder width — works well for both rotation and abduction. */
const DIRECTION_DEADBAND_FRAC = 0.05;

/** Body's vertical centreline x-coordinate in image space, derived
 *  from the midpoint of the two shoulders. Returns null when either
 *  shoulder isn't reliably visible or the shoulders are nearly on
 *  top of each other (patient turned away from camera). */
function bodyCentreX(keypoints: Keypoint[]): { centreX: number; shoulderWidth: number } | null {
  const ls = keypoints[LM.LEFT_SHOULDER];
  const rs = keypoints[LM.RIGHT_SHOULDER];
  if (!ls || !rs) return null;
  if ((ls.score ?? 0) < VIS_THRESHOLD || (rs.score ?? 0) < VIS_THRESHOLD) return null;
  const shoulderWidth = Math.abs(rs.x - ls.x);
  // Tiny epsilon — caller may pass either pixel-space or normalised
  // [0,1] coordinates. The downstream math is scale-invariant
  // (ratios), so all this check needs to guard against is a literal
  // div-by-zero / patient turned exactly 90° to the camera.
  if (shoulderWidth < 1e-4) return null;
  return { centreX: (ls.x + rs.x) / 2, shoulderWidth };
}

/** Detect external vs internal rotation from wrist position relative
 *  to the elbow's distance from the body centreline. External =
 *  wrist swings further OUT (more lateral) than the elbow. Internal
 *  = wrist crosses INWARD (less lateral, or past the centreline).
 *  Returns null if visibility is insufficient or the lateral signal
 *  is within the deadband (near neutral pose). */
export function detectShoulderRotationDirection(
  keypoints: Keypoint[],
  side: "left" | "right",
): ShoulderRotationDirection | null {
  const idx = SIDE_INDICES[side];
  const e = keypoints[idx.elbow];
  const w = keypoints[idx.wrist];
  if (!e || !w) return null;
  if ((e.score ?? 0) < VIS_THRESHOLD || (w.score ?? 0) < VIS_THRESHOLD) return null;
  const body = bodyCentreX(keypoints);
  if (!body) return null;
  const elbowLateral = Math.abs(e.x - body.centreX);
  const wristLateral = Math.abs(w.x - body.centreX);
  const ratio = (wristLateral - elbowLateral) / body.shoulderWidth;
  if (Math.abs(ratio) < DIRECTION_DEADBAND_FRAC) return null;
  return ratio > 0 ? "external" : "internal";
}

/** Detect abduction vs adduction from the elbow's position relative
 *  to the test-side shoulder along the lateral axis. Abduction =
 *  elbow drifts further OUTWARD (away from body centre on the test
 *  side). Adduction = elbow drifts INWARD (across body centre).
 *  Direction is normalised by the side so left and right tests behave
 *  the same. Returns null on insufficient visibility / within
 *  deadband. */
export function detectShoulderAbAdDirection(
  keypoints: Keypoint[],
  side: "left" | "right",
): ShoulderAbAdDirection | null {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  if (!s || !e) return null;
  if ((s.score ?? 0) < VIS_THRESHOLD || (e.score ?? 0) < VIS_THRESHOLD) return null;
  const body = bodyCentreX(keypoints);
  if (!body) return null;
  // Outward unit direction in image x — points from body centre toward
  // the test-side shoulder. Multiplying by this lets the same formula
  // work for left and right sides regardless of camera-mirroring.
  const outwardSign = Math.sign(s.x - body.centreX);
  if (outwardSign === 0) return null;
  const elbowOutward = (e.x - s.x) * outwardSign;
  const ratio = elbowOutward / body.shoulderWidth;
  if (Math.abs(ratio) < DIRECTION_DEADBAND_FRAC) return null;
  return ratio > 0 ? "abduction" : "adduction";
}

/** Magnitude + direction for the merged Rotation test. Magnitude
 *  reuses computeShoulderRotationFromBaseline; direction is detected
 *  geometrically. Returns null when either signal is unavailable. */
export function computeShoulderRotationWithDirection(
  keypoints: Keypoint[],
  side: "left" | "right",
  baseline: ShoulderRotationCalibration,
): { magnitude: number; direction: ShoulderRotationDirection } | null {
  const direction = detectShoulderRotationDirection(keypoints, side);
  if (!direction) return null;
  const magnitude = computeShoulderRotationFromBaseline(keypoints, side, baseline);
  if (magnitude === null) return null;
  return { magnitude, direction };
}

/** Magnitude + direction for the merged Abduction/Adduction test.
 *  Magnitude reuses computeShoulderAngle("abduction"); direction is
 *  detected geometrically. */
export function computeShoulderAbAdWithDirection(
  keypoints: Keypoint[],
  side: "left" | "right",
): { magnitude: number; direction: ShoulderAbAdDirection } | null {
  const direction = detectShoulderAbAdDirection(keypoints, side);
  if (!direction) return null;
  // computeShoulderAngle returns Math.abs(angleBetween(...)) for the
  // abduction/adduction branch — the same magnitude regardless of
  // which direction the arm went, which is exactly what we want here.
  const magnitude = computeShoulderAngle("abduction", keypoints, side);
  if (magnitude === null) return null;
  return { magnitude, direction };
}
