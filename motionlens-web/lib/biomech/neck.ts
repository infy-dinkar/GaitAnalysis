// UI metadata + browser-side angle math for the neck biomech flow.
// Ports neck_engine.py exactly. The neck engine has no per-side
// parameter — it uses both ears + the shoulder midline regardless of
// which way the patient tilts/turns.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

export type NeckMovementId =
  | "flexion"
  | "extension"
  | "flexion_extension"
  | "lateral_flexion"
  | "rotation";

export interface NeckMovement {
  id: string;
  label: string;
  description: string;
  target: [number, number];
  /** True for merged tests that capture two directions in one
   *  recording (flexion_extension = flexion + extension). Live and
   *  upload modes detect direction per frame and track separate
   *  peaks. */
  merged?: boolean;
  primaryLabel?: string;
  secondaryLabel?: string;
  /** Normal range for the secondary direction (extension target
   *  when merged is flexion+extension). */
  secondaryTarget?: [number, number];
  /** Hidden from the movement chooser. Legacy single-direction
   *  entries stay in the metadata table so saved reports referring
   *  to "flexion" / "extension" alone resolve labels + targets,
   *  but they no longer appear when starting a new trial. */
  hidden?: boolean;
  /** Optional reference illustration. See MovementGrid's
   *  MovementOption.imageUrl for the path convention. */
  imageUrl?: string;
}

export const NECK_MOVEMENTS: NeckMovement[] = [
  // Combined Flexion + Extension. Patient in lateral view (camera
  // sees one side of the head). One recording captures forward
  // tilt (chin to chest) and backward tilt (head back). Direction
  // is detected per frame from the signed neck-vs-vertical angle
  // normalised by the patient's facing direction.
  {
    id: "flexion_extension",
    label: "Flexion + Extension",
    description:
      "Tilt the head forward (chin to chest, flexion) then backward (extension). One recording captures both peaks.",
    target: [45, 80],
    merged: true,
    primaryLabel: "Flexion",
    secondaryLabel: "Extension",
    secondaryTarget: [50, 70],
    imageUrl: "/images/biomech/neck/neck_flexion_extension.png",
  },
  // Lateral flexion is also a merged test — patient tilts head
  // toward each shoulder in turn, and the report captures the peak
  // on both sides plus a neutral upright thumbnail. Direction is
  // detected from the sign of the neck-vs-vertical angle (signed
  // value, not absolute).
  {
    id: "lateral_flexion",
    label: "Lateral Flexion",
    description:
      "Tilt the head sideways toward each shoulder in turn. One recording captures the peak tilt on both sides.",
    target: [20, 45],
    merged: true,
    primaryLabel: "Right Lateral Flexion",
    secondaryLabel: "Left Lateral Flexion",
    secondaryTarget: [20, 45],
    imageUrl: "/images/biomech/neck/neck_lateral_flexion.png",
  },
  {
    id: "rotation",
    label: "Rotation",
    description: "Turn the head to either side",
    target: [70, 90],
    merged: true,
    primaryLabel: "Left Rotation",
    secondaryLabel: "Right Rotation",
    secondaryTarget: [60, 80],
    imageUrl: "/images/biomech/neck/neck_rotation.png",
  },
  // Legacy single-direction entries — kept so saved reports
  // referring to them still resolve labels + targets, but hidden
  // from the chooser since the merged version above replaces them.
  { id: "flexion",   label: "Flexion",   description: "Tilt the head forward, chin to chest", target: [45, 80], hidden: true },
  { id: "extension", label: "Extension", description: "Tilt the head backward",                target: [50, 70], hidden: true },
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
    movement === "flexion_extension"
  ) {
    // Head-orientation angle from the ear→nose vector. The legacy
    // shoulder_mid → ear_mid formula only captured the CERVICAL
    // spine portion of flex/ext (~25° max anatomically), but
    // clinically full neck flexion (45-80°) and extension (50-70°)
    // include the atlanto-occipital joint + head-on-neck pitch as
    // well. Tracking the face direction via ear→nose captures the
    // combined motion.
    //
    // Sign convention:
    //   positive = nose pointing below ear (flexion / chin-down)
    //   negative = nose pointing above ear (extension / head-back)
    //
    // |faceVecX| in the atan2 denominator makes the sign depend only
    // on the vertical component — so the same formula works for
    // patients facing image-left OR image-right (the lateral
    // orientation cancels out).
    //
    // 10° baseline subtraction approximates the typical neutral
    // pose where the nose sits slightly below the ear-axis (the
    // ear-tragus to nose-tip line points roughly horizontal but
    // with a small downward bias). Patients with a flatter face
    // profile may read slightly higher, those with a longer nose
    // slightly lower — but the per-frame variation from this is
    // small compared to actual flex/ext motion.
    const faceVecX = nose.x - earMidX;
    const faceVecY = nose.y - earMidY;
    if (Math.hypot(faceVecX, faceVecY) < 1e-4) return null;
    const tiltDeg = (Math.atan2(faceVecY, Math.abs(faceVecX)) * 180) / Math.PI;
    const NEUTRAL_TILT_BASELINE = 10;
    return tiltDeg - NEUTRAL_TILT_BASELINE;
  }

  if (movement === "lateral_flexion") {
    // Lateral flexion stays on the cervical-spine formula — the
    // ear→nose approach only measures sagittal (forward/back) tilt.
    // Returns the SIGNED angle (not absolute) so the merged-test
    // direction routing can tell left tilt from right tilt by sign;
    // peak tracking elsewhere takes Math.abs when displaying.
    const neckVecX = earMidX - shldrMidX;
    const neckVecY = earMidY - shldrMidY;
    const verticalX = 0;
    const verticalY = -1;
    return angleBetween(verticalX, verticalY, neckVecX, neckVecY);
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

// ─── Direction detection for merged flexion + extension ─────────

export type NeckFlexExtDirection = "flexion" | "extension";

/** Minimum signed-angle magnitude before direction commits. Below
 *  this the head is too close to neutral for a reliable
 *  classification. Same role the corresponding constant plays in
 *  the shoulder flex/ext detector. */
const NECK_FLEXEXT_DEADBAND_DEG = 5;

/** Detect neck FLEXION (chin to chest) vs EXTENSION (head back) from
 *  the sign of the ear→nose tilt angle. Delegates to computeNeckAngle
 *  so the direction always matches the magnitude the rest of the
 *  pipeline displays. The angle's sign is already symmetric for both
 *  lateral orientations (the formula uses |faceVecX| in its atan2
 *  denominator), so no extra facing-direction normalisation is
 *  needed here. */
export function detectNeckFlexExtDirection(
  keypoints: Keypoint[],
): NeckFlexExtDirection | null {
  const angle = computeNeckAngle("flexion", keypoints);
  if (angle === null || Number.isNaN(angle)) return null;
  if (Math.abs(angle) < NECK_FLEXEXT_DEADBAND_DEG) return null;
  return angle > 0 ? "flexion" : "extension";
}

export type NeckLateralDirection = "right" | "left";

/** Detect which side the patient is tilting their head toward
 *  during a neck lateral flexion test. Uses the SIGN of
 *  computeNeckAngle("lateral_flexion"). Convention assumes the
 *  camera view is selfie-mirrored (live mode default): a positive
 *  signed angle means the ear-midpoint sits right of the
 *  shoulder-midpoint in the image, which corresponds to the
 *  patient's LEFT (their left side appears on image-right in a
 *  mirrored selfie). Reverses for non-mirrored uploaded videos —
 *  the labels swap but the dual-direction capture still works. */
export function detectNeckLateralDirection(
  keypoints: Keypoint[],
): NeckLateralDirection | null {
  const angle = computeNeckAngle("lateral_flexion", keypoints);
  if (angle === null || Number.isNaN(angle)) return null;
  if (Math.abs(angle) < NECK_FLEXEXT_DEADBAND_DEG) return null;
  // Negative signed → ear_mid left of shoulder_mid in image →
  // patient's right side (in mirrored selfie view).
  return angle < 0 ? "right" : "left";
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
