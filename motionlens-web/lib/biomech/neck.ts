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

// BlazePose-tfjs scores are lower than MediaPipe's `visibility` field —
// typical visible joints score 0.2-0.4. 0.15 lets math run on most
// in-frame joints without false-passing fully occluded ones.
const VIS_THRESHOLD = 0.15;

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

  // rotation — nose offset relative to ear midline, ratio × 90°.
  // This is a fallback used only before the rotation baseline has been
  // captured. Once the live capture flow locks the patient's facing-
  // forward reference, it routes to computeNeckRotationFromBaseline()
  // which is more biomechanically accurate (head-sphere arcsin model
  // with per-patient anatomy correction).
  const earWidth = Math.abs(lEar.x - rEar.x);
  if (earWidth < 1e-3) return 0;
  const offsetRatio = (nose.x - earMidX) / (earWidth / 2);
  return Math.min(Math.abs(offsetRatio * 90), 120);
}

// ─── Calibrated rotation (baseline-locked) ──────────────────────
//
// Treats the head as a sphere of radius r ≈ ear-width / 2 with the
// nose on its surface. When the patient faces the camera squarely we
// snapshot R — the reference position — and use it as 0°. When they
// rotate by angle θ around the vertical neck axis, the nose's lateral
// projection in the image plane is (r · sin θ); inverting gives the
// rotation angle. Subtracting the baseline ratio also cancels out
// anatomical asymmetry (some people's nose isn't perfectly centred
// even when facing straight forward).

export interface NeckRotationCalibration {
  /** Ear-to-ear pixel width AT calibration (head-sphere diameter). */
  R_earWidth: number;
  /** (nose.x - earMidX) / (earWidth/2) AT calibration — anatomical
   *  baseline. Subtracted from current ratio so deviations are
   *  measured FROM the patient's own neutral, not from a generic
   *  "nose-is-perfectly-centred" assumption. */
  baselineRatio: number;
}

/** Maximum tolerated baseline ratio when auto-detecting "facing
 *  forward". 0.15 = nose sits within 15% of half-ear-width of the
 *  ear midline. Larger than that and we suspect the patient isn't
 *  actually facing the camera squarely. */
const FACING_FORWARD_RATIO = 0.15;

/** True when MoveNet sees all 5 required keypoints AND the patient's
 *  nose is roughly between their ears (i.e. they are reasonably
 *  facing the camera). Used by the capture flow's calibration phase
 *  to decide when to auto-lock the baseline. */
export function isStableFacingForward(keypoints: Keypoint[]): boolean {
  const nose = keypoints[LM.NOSE];
  const lEar = keypoints[LM.LEFT_EAR];
  const rEar = keypoints[LM.RIGHT_EAR];
  const lSh  = keypoints[LM.LEFT_SHOULDER];
  const rSh  = keypoints[LM.RIGHT_SHOULDER];
  for (const k of [nose, lEar, rEar, lSh, rSh]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return false;
  }
  const earWidth = Math.abs(lEar.x - rEar.x);
  if (earWidth < 1e-3) return false;
  const earMidX = (lEar.x + rEar.x) / 2;
  const ratio = Math.abs((nose.x - earMidX) / (earWidth / 2));
  return ratio <= FACING_FORWARD_RATIO;
}

/** Snapshot the patient's facing-forward reference. Returns null if
 *  any required keypoint is below visibility threshold. The caller
 *  (live capture flow) typically only invokes this after
 *  isStableFacingForward() has held true for ~1.5 s, so the snapshot
 *  isn't taken on a single jittery frame. */
export function captureNeckRotationBaseline(
  keypoints: Keypoint[],
): NeckRotationCalibration | null {
  const nose = keypoints[LM.NOSE];
  const lEar = keypoints[LM.LEFT_EAR];
  const rEar = keypoints[LM.RIGHT_EAR];
  for (const k of [nose, lEar, rEar]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }
  const earWidth = Math.abs(lEar.x - rEar.x);
  if (earWidth < 1e-3) return null;
  const earMidX = (lEar.x + rEar.x) / 2;
  const baselineRatio = (nose.x - earMidX) / (earWidth / 2);
  return { R_earWidth: earWidth, baselineRatio };
}

/** Calibrated rotation angle in degrees. Signed: positive = rotated
 *  toward the patient's right (camera left), negative = toward the
 *  patient's left (camera right). Maxes out near ±90° only when one
 *  ear is fully behind the head (anatomical limit).
 *
 *  Math — ear-width foreshortening:
 *
 *      currentEarWidth = R_earWidth × cos(θ)
 *      ⇒ θ = arccos(currentEarWidth / R_earWidth)
 *
 *  Why this beats the nose-displacement model: nose protrudes
 *  forward from the ear-axis by an amount that varies per patient
 *  (typically MORE than half-ear-width). A nose-displacement formula
 *  assumes nose-protrusion ≈ half-ear-width — which over-estimates
 *  rotation for anyone with a longer nose, saturating to 90° at
 *  even moderate rotations like 45°.
 *
 *  Ear-width foreshortening is independent of nose anatomy: as the
 *  head rotates, the ears' projected separation shrinks purely as a
 *  function of the rotation angle. This gives clinically realistic
 *  readings (peak ~70-85° for healthy adults) and only approaches
 *  90° when the head has truly rotated to where one ear is hidden.
 *
 *  Sign comes from nose direction (which side did the head turn
 *  toward) — magnitude from ear-width.
 */
export function computeNeckRotationFromBaseline(
  keypoints: Keypoint[],
  baseline: NeckRotationCalibration,
): number | null {
  const nose = keypoints[LM.NOSE];
  const lEar = keypoints[LM.LEFT_EAR];
  const rEar = keypoints[LM.RIGHT_EAR];
  // Both ears must be reasonably visible — if one is fully occluded
  // (rotation past ~60-70°) the foreshortening signal is gone and
  // we return null. The peak-tracker upstream then holds the last
  // good value, which captures the rotation right up to the
  // occlusion boundary.
  for (const k of [nose, lEar, rEar]) {
    if (!k || (k.score ?? 0) < VIS_THRESHOLD) return null;
  }

  // Magnitude from ear-width foreshortening.
  const currentEarWidth = Math.abs(lEar.x - rEar.x);
  // Clamp ratio to [0, 1] — it can mathematically exceed 1 if the
  // patient stepped CLOSER to the camera since calibration (apparent
  // ear-width grows). Clamping keeps acos defined; if the patient
  // actually moved, we under-report rather than crash.
  const ratio = Math.max(
    0,
    Math.min(1, currentEarWidth / baseline.R_earWidth),
  );
  const magnitudeDeg = (Math.acos(ratio) * 180) / Math.PI;

  // Sign from nose direction relative to the ear midline. Subtract
  // the baseline ratio so anatomical asymmetry (off-centre nose at
  // calibration) is cancelled out — only the DELTA from baseline
  // determines rotation direction.
  const earMidX = (lEar.x + rEar.x) / 2;
  const noseRatio =
    currentEarWidth > 1e-3
      ? (nose.x - earMidX) / (currentEarWidth / 2)
      : 0;
  const delta = noseRatio - baseline.baselineRatio;
  // Treat near-zero deltas as positive so the very first stable
  // frame doesn't return -0; doesn't affect peak tracking either way.
  const sign = delta < 0 ? -1 : 1;

  return sign * magnitudeDeg;
}
