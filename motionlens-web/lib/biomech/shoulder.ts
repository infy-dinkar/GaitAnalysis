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
  id: ShoulderMovementId;
  label: string;
  description: string;
  target: [number, number];
}

export const SHOULDER_MOVEMENTS: ShoulderMovement[] = [
  { id: "flexion",            label: "Flexion",            description: "Lift the arm forward and overhead",          target: [150, 180] },
  { id: "extension",          label: "Extension",          description: "Move the arm backward behind the body",       target: [45, 60] },
  { id: "abduction",          label: "Abduction",          description: "Lift the arm sideways away from the body",    target: [150, 180] },
  { id: "adduction",          label: "Adduction",          description: "Bring the arm across the chest",              target: [30, 50] },
  // Rotation targets adjusted to clinical reality. The previous 80-90 /
  // 70-90 ranges were the theoretical anatomical maxima; in practice
  // most healthy adults peak at 70-85° (external) and 60-75° (internal)
  // with the elbow tucked at 90°. The calibrated arcsin formula now
  // returns true projected rotation, so the previous targets would
  // false-flag healthy patients as "below normal".
  { id: "external_rotation",  label: "External Rotation",  description: "Rotate the forearm outward (elbow at 90°)",  target: [70, 90] },
  { id: "internal_rotation",  label: "Internal Rotation",  description: "Rotate the forearm inward (elbow at 90°)",   target: [60, 80] },
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
  // Shoulder + elbow MUST be confidently visible. At the neutral pose
  // (forearm aimed at the camera) the upper arm hangs vertically and
  // both these joints are always cleanly in view, so demanding
  // confidence here is correct.
  for (const k of [s, e]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return false;
  }
  // The wrist deliberately does NOT get the visibility gate. The
  // neutral pose IS "forearm pointing straight at the lens", which
  // foreshortens the wrist to nearly a point — MoveNet routinely
  // scores it well below VIS_THRESHOLD in exactly this pose. Requiring
  // a confident wrist here is self-contradictory and was why baseline
  // auto-lock never fired (the detector rejected the very pose it was
  // waiting for). We still need a wrist *position* to measure the
  // forearm projection, so the keypoint must at least exist.
  if (!w) return false;
  const upperArmLen = Math.hypot(e.x - s.x, e.y - s.y);
  if (upperArmLen < MIN_UPPER_ARM_PX) return false;
  const forearmLen = Math.hypot(w.x - e.x, w.y - e.y);
  // Short forearm projection = forearm aimed at the camera = neutral.
  // This ratio check is itself the safeguard against a false lock:
  // if the patient is holding the forearm out to the side, the
  // projection is long and the ratio fails — so a non-neutral pose
  // still can't slip through even though the wrist confidence gate
  // was relaxed above.
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

/** Valid elbow flexion window (in degrees) for the rotation test pose.
 *  Anatomically the patient is supposed to hold the elbow tucked at
 *  the side, bent at ~90°. If the elbow is straighter than ~115° (arm
 *  hanging by the side) or sharper than ~65° (over-bent), the geometry
 *  the formula relies on is invalid: the forearm projection no longer
 *  represents pure shoulder-axis rotation. We reject such frames so a
 *  patient who simply lowers their arm doesn't get a phantom 90° peak
 *  from the formula reading "forearm = full vertical = sin⁻¹(1)". */
const ELBOW_FLEX_MIN_DEG = 65;
const ELBOW_FLEX_MAX_DEG = 115;

/** Calibrated rotation magnitude in degrees, capped at 90°.
 *
 *  Returns null when:
 *    • wrist / elbow / shoulder visibility drops, OR
 *    • the elbow is not bent ≈90° (test pose violated — patient let
 *      the arm hang straight, lifted the elbow, etc.), OR
 *    • the live upper-arm scale is degenerate (zero pixel length).
 *
 *  Scale invariance: we normalise the forearm projection by the LIVE
 *  shoulder→elbow length each frame, not the baseline value frozen at
 *  calibration. That way, when the patient walks closer to the camera
 *  (or further), both numerator and denominator grow together and the
 *  ratio stays anchored to the true rotation angle. The baseline's
 *  R_upperArmLength field is retained for diagnostics but no longer
 *  drives the arcsin denominator. */
export function computeShoulderRotationFromBaseline(
  keypoints: Keypoint[],
  side: "left" | "right",
  baseline: ShoulderRotationCalibration,
): number | null {
  if (baseline.side !== side) return null; // wrong-side baseline guard
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  const w = keypoints[idx.wrist];
  if (!s || !e || !w) return null;
  if (
    (s.score ?? 0) < VIS_THRESHOLD ||
    (e.score ?? 0) < VIS_THRESHOLD ||
    (w.score ?? 0) < VIS_THRESHOLD
  ) {
    return null;
  }

  // Live upper-arm length — used as the rotation reference each frame
  // so changes in patient-to-camera distance scale numerator and
  // denominator together.
  const liveUpperArm = Math.hypot(s.x - e.x, s.y - e.y);
  if (liveUpperArm < 1) return null;

  // Elbow flexion check (Fix A). Inner angle at the elbow between
  // shoulder→elbow and elbow→wrist vectors. ≈180° = arm straight,
  // ≈90° = test pose, ≈0° = fully bent. We require the patient to be
  // in roughly the ±25° window around 90°.
  const ux = s.x - e.x, uy = s.y - e.y;       // upper arm (elbow → shoulder)
  const fx = w.x - e.x, fy = w.y - e.y;       // forearm  (elbow → wrist)
  const forearmProjLen = Math.hypot(fx, fy);
  if (forearmProjLen < 1) return null;
  const cosElbow = (ux * fx + uy * fy) / (liveUpperArm * forearmProjLen);
  const elbowAngleDeg =
    (Math.acos(Math.max(-1, Math.min(1, cosElbow))) * 180) / Math.PI;
  if (elbowAngleDeg < ELBOW_FLEX_MIN_DEG || elbowAngleDeg > ELBOW_FLEX_MAX_DEG) {
    return null;
  }

  // Scale-invariant rotation: ratio = forearm-projection / live-upper-arm.
  // The upper-arm length is anatomically close to the forearm length
  // (proxy assumption), and crucially it tracks camera distance in
  // real time so the patient can move toward/away from the lens
  // without inflating the ratio past 1.
  const ratio = Math.max(0, Math.min(1, forearmProjLen / liveUpperArm));
  return (Math.asin(ratio) * 180) / Math.PI;
}
