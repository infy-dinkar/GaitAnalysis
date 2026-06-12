// UI metadata + browser-side angle math for the shoulder biomech flow.
// The compute function ports shoulder_engine.py exactly so live and
// final measurements use the same formulas. Used by the live mode
// (client-side TF.js BlazePose + this math). Server-side upload mode
// still uses the Python engine of record for the authoritative report.

import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

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
  /** Optional reference illustration. See MovementGrid's
   *  MovementOption.imageUrl for the path convention. */
  imageUrl?: string;
}

export const SHOULDER_MOVEMENTS: ShoulderMovement[] = [
  // Combined Flexion + Extension. Patient is in lateral view (test
  // side toward camera). One recording captures the forward-overhead
  // arc (flexion) and the backward-behind-body arc (extension); the
  // live engine detects direction per frame from the signed
  // arm-vs-trunk angle, normalised by the patient's facing direction.
  {
    id: "flexion_extension",
    label: "Flexion + Extension",
    description:
      "Lift the arm forward and overhead (flexion), then move it backward behind the body (extension). One session captures both peaks.",
    target: [150, 180],
    merged: true,
    primaryLabel: "Flexion",
    secondaryLabel: "Extension",
    secondaryTarget: [45, 60],
    imageUrl: "/images/biomech/shoulder/shoulder_flexion_extension.png",
  },
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
    imageUrl: "/images/biomech/shoulder/shoulder_abduction_adduction.png",
  },
  // Combined External + Internal Rotation. Patient rotates outward,
  // returns to neutral, then rotates inward. Both peaks are captured
  // in the same trial.
  {
    id: "rotation",
    label: "Rotation (External + Internal)",
    description:
      "Stand side-on to the camera. Raise your arm to shoulder height (90° abduction) and bend your elbow to 90° so the forearm hangs vertical. Rotate the forearm UP toward your head (external) then DOWN toward the floor (internal). One session captures both peaks.",
    target: [70, 90],
    merged: true,
    primaryLabel: "External Rotation",
    secondaryLabel: "Internal Rotation",
    secondaryTarget: [60, 80],
    imageUrl: "/images/biomech/shoulder/shoulder_rotation.png",
  },
  // Legacy single-direction entries — kept in the lookup table so
  // saved reports that referenced these IDs still resolve their label
  // / target, but hidden from the chooser since the merged versions
  // above replace them in the UI flow.
  { id: "flexion",           label: "Flexion",           description: "Lift the arm forward and overhead",          target: [150, 180], hidden: true },
  { id: "extension",         label: "Extension",         description: "Move the arm backward behind the body",       target: [45, 60],   hidden: true },
  { id: "abduction",         label: "Abduction",         description: "Lift the arm sideways away from the body",   target: [150, 180], hidden: true },
  { id: "adduction",         label: "Adduction",         description: "Bring the arm across the chest",              target: [30, 50],   hidden: true },
  { id: "external_rotation", label: "External Rotation", description: "Rotate the forearm outward (elbow at 90°)",  target: [70, 90],   hidden: true },
  { id: "internal_rotation", label: "Internal Rotation", description: "Rotate the forearm inward (elbow at 90°)",   target: [60, 80],   hidden: true },
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

// ─── Lateral-view shoulder rotation ─────────────────────────────
//
// Setup: patient stands SIDE-ON to the camera. Arm abducted to 90°
// (upper arm horizontal, projecting toward / away from the camera
// in the lateral view). Elbow bent 90° so the forearm sweeps in a
// plane parallel to the image plane:
//
//   • Forearm hanging straight down  →  0°  reference
//   • Forearm rotating UP toward head →  External Rotation (target 60–90°)
//   • Forearm rotating DOWN toward floor → Internal Rotation (target 60–80°)
//
// Math (per frame, no baseline calibration required):
//
//     reference     = (0, 1)              // straight down, image y-down convention
//     forearmVector = (wrist − elbow)
//     magnitude     = |angleBetween(reference, forearmVector)|
//
// Direction: pure y comparison of wrist vs elbow.
//   wrist.y < elbow.y  → External (wrist above elbow on screen)
//   wrist.y > elbow.y  → Internal (wrist below elbow on screen)
// A small deadband on |wrist.y - elbow.y| / shoulderWidth suppresses
// flips at the exact horizontal crossover.
//
// The `ShoulderRotationCalibration` type + neutral / baseline helpers
// below are KEPT for API compatibility with LiveAssessment.tsx,
// LiveBiomechCamera.tsx and uploadAnalyze.ts — they don't supply real
// scale information any more; the new compute function ignores their
// payload aside from the wrong-side guard.

export interface ShoulderRotationCalibration {
  /** Retained for API compatibility — the new lateral-view formula
   *  does not use a length proxy. Populated from the still-cheap
   *  shoulder→elbow distance so saved sessions stay readable. */
  R_upperArmLength: number;
  /** Side this baseline was captured for — guards against accidental
   *  cross-side application if the operator switches mid-flow. */
  side: "left" | "right";
}

/** Minimum forearm vector length (px) below which the angle becomes
 *  unstable; the peak tracker holds the last good value. */
const ROTATION_MIN_VECTOR_PX = 4;

/** Direction deadband on (wrist.y − elbow.y) expressed as a fraction
 *  of shoulder width — scale-invariant. Suppresses direction flips
 *  when the wrist sits within a thin band of the elbow's image y. */
const ROTATION_DIRECTION_DEADBAND_FRAC = 0.03;

/** Returns true once the test-side shoulder, elbow and wrist are
 *  visible. The lateral-view formula needs no baseline lock-in, so
 *  the live flow can transition to "recording" the moment landmarks
 *  are visible. Function name retained for API compatibility with
 *  LiveAssessment.tsx. */
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
  return true;
}

/** Build a calibration handle. The new lateral-view formula doesn't
 *  use upper-arm length, but we keep the field populated so saved
 *  sessions and the existing call sites continue to work. The
 *  `side` field IS still meaningful — it powers the wrong-side
 *  guard inside computeShoulderRotationFromBaseline. */
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
  return { R_upperArmLength: len, side };
}

/** Lateral-view rotation magnitude in CLINICAL degrees (0–90°).
 *
 *  Step 1: compute the raw angle from the straight-down reference
 *          vector (0, 1) — that puts forearm-down at 0°, forearm-
 *          horizontal at 90°, forearm-up at 180°.
 *  Step 2: convert to the clinical reading, which is the angle FROM
 *          horizontal (anatomical neutral for an abducted-90° arm):
 *
 *              clinical = |raw − 90°|
 *
 *          Above elbow (ER direction): raw 90→180 maps to clinical
 *          0→90, matching the AAOS ER range of 0–90°.
 *          Below elbow (IR direction): raw 90→0 maps to clinical
 *          0→90 in the same way (anatomical IR max ~70–80°).
 *
 *  Returns null when wrist or elbow visibility drops; the upstream
 *  peak-tracker holds the last good value. The `baseline` parameter
 *  is retained only for its `side` field (wrong-side guard); its
 *  R_upperArmLength is ignored. */
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
  const ax = w.x - e.x;
  const ay = w.y - e.y;
  if (Math.hypot(ax, ay) < ROTATION_MIN_VECTOR_PX) return null;
  // angleBetween((0, 1), (ax, ay)) with atan2(cross, dot):
  //   dot = ay, cross = −ax.
  const rawAngle = Math.abs(angleBetween(0, 1, ax, ay));
  return Math.abs(rawAngle - 90);
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
export type ShoulderFlexExtDirection = "flexion" | "extension";

/** Minimum signed-angle magnitude (degrees) before a flexion vs
 *  extension classification is committed. The shoulderFlexionExtension
 *  formula returns ~0° when the arm hangs straight at the side, so a
 *  small magnitude is the natural neutral-pose deadband. */
const FLEXEXT_DEADBAND_DEG = 5;

/** Anatomical maximum for shoulder extension (degrees). Past this
 *  threshold a frame can only be on the flexion arc — extension
 *  doesn't physically reach 75°+. We use it as a sanity override
 *  on the sign-based classifier: at the overhead end of the flexion
 *  arc (elbow.x ≈ shoulder.x, signedAngle close to ±180°) tiny
 *  keypoint jitter can flip the angle's sign and route an overhead
 *  frame into the extension slot, producing the implausible "180°
 *  extension" peak users saw on uploaded videos. Magnitudes above
 *  this cutoff are always classified as flexion regardless of the
 *  signed value. */
const FLEXEXT_EXTENSION_ANATOMICAL_MAX_DEG = 75;

/** Detect shoulder FLEXION (arm forward) vs EXTENSION (arm backward)
 *  from the signed angle between the trunk and arm vectors, then
 *  normalised by the patient's facing direction (inferred from the
 *  nose position relative to the test-side shoulder). The test runs
 *  in lateral view, so nose.x is offset toward the patient's face
 *  ("front"). Using that offset's sign as the facing direction lets
 *  the same logic classify both lateral orientations (patient facing
 *  image-left OR image-right). */
export function detectShoulderFlexExtDirection(
  keypoints: Keypoint[],
  side: "left" | "right",
): ShoulderFlexExtDirection | null {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  const h = keypoints[idx.hip];
  const nose = keypoints[LM.NOSE];
  if (!s || !e || !h || !nose) return null;
  if (
    (s.score ?? 0) < VIS_THRESHOLD ||
    (e.score ?? 0) < VIS_THRESHOLD ||
    (h.score ?? 0) < VIS_THRESHOLD ||
    (nose.score ?? 0) < VIS_THRESHOLD
  ) {
    return null;
  }

  // Facing direction: nose horizontal offset from the test-side
  // shoulder. In lateral view the nose sits in front of the shoulder
  // axis, so the sign of (nose.x - shoulder.x) cleanly distinguishes
  // image-left-facing from image-right-facing setups. Scale by upper-
  // arm length (a reliable, mostly camera-invariant unit) — if the
  // offset is too small the patient is in a near-frontal view and
  // sagittal flexion/extension can't be measured reliably.
  const upperArmLen = Math.hypot(e.x - s.x, e.y - s.y);
  if (upperArmLen < 1e-4) return null;
  const facingDx = nose.x - s.x;
  if (Math.abs(facingDx) / upperArmLen < 0.10) return null;
  const facingSign = Math.sign(facingDx);

  // Signed arm-vs-trunk angle (same as shoulderFlexionExtension).
  // Positive sign on the unadjusted value corresponds to "arm forward"
  // for a patient facing image-right; multiplying by facingSign
  // normalises to "positive = flexion" regardless of which lateral
  // orientation the camera is on.
  const tx = h.x - s.x, ty = h.y - s.y;
  const ax = e.x - s.x, ay = e.y - s.y;
  const cross = tx * ay - ty * ax;
  const dot = tx * ax + ty * ay;
  const signedAngle = -(Math.atan2(cross, dot) * 180) / Math.PI;
  const adjusted = signedAngle * facingSign;

  if (Math.abs(adjusted) < FLEXEXT_DEADBAND_DEG) return null;
  // Anatomical override: any magnitude past ~75° must be on the
  // flexion arc (extension physically can't reach that range). At
  // the overhead end of flexion (elbow ≈ above shoulder) the signed
  // angle approaches ±180° and tiny keypoint jitter can flip its
  // sign, which in upload mode (no EMA smoothing) was routing
  // overhead frames into the extension slot and producing
  // implausible 170°+ "extension" peaks.
  if (Math.abs(adjusted) > FLEXEXT_EXTENSION_ANATOMICAL_MAX_DEG) return "flexion";
  return adjusted > 0 ? "flexion" : "extension";
}

/** Minimum lateral offset (as a fraction of shoulder width) before
 *  a direction is committed. Tighter = more responsive but flips on
 *  noise; looser = stable but laggy at small ROMs. 0.03 ≈ 3% of
 *  shoulder width — keeps natural-ROM adduction (which is anatomically
 *  only 30–50°, so the elbow's medial drift is modest) inside the
 *  detection zone while still excluding pure-noise frames. */
const DIRECTION_DEADBAND_FRAC = 0.03;

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

/** Lateral-view direction detection.
 *  Wrist ABOVE elbow on screen (image y smaller) → External Rotation.
 *  Wrist BELOW elbow on screen (image y larger)  → Internal Rotation.
 *  Within the y-deadband (small fraction of shoulder width) → null,
 *  neither direction committed. */
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
  const dy = (w.y - e.y) / body.shoulderWidth;
  if (Math.abs(dy) < ROTATION_DIRECTION_DEADBAND_FRAC) return null;
  // Image y grows DOWNWARD: dy < 0 means wrist above elbow on screen.
  return dy < 0 ? "external" : "internal";
}

/** Elbow-above-shoulder threshold (in shoulder-width units) above
 *  which the y-axis alone is enough to classify the motion as
 *  abduction, regardless of how noisy the x-axis signal is. Set
 *  permissively (0.20 ≈ elbow clearly above shoulder) so it only
 *  fires for the genuine overhead end of the abduction arc — small
 *  upward drifts during adduction (e.g. the elbow rising slightly
 *  as the arm sweeps across the chest) don't accidentally trigger
 *  the override and lose the adduction label. */
const ABDUCTION_OVERHEAD_FRAC = 0.20;

/** Detect abduction vs adduction from the elbow's position relative
 *  to the test-side SHOULDER, with a y-axis override for the
 *  overhead portion of the abduction arc.
 *
 *  Rules:
 *    • Elbow above shoulder (anywhere on the upper abduction arc):
 *      always classified as abduction. This is the critical override
 *      — at the overhead end of the abduction motion elbow.x ≈
 *      shoulder.x and pure x-axis classification flipped on keypoint
 *      noise, producing phantom 180° "adduction" peaks.
 *    • Otherwise: use the elbow's lateral offset from the shoulder.
 *      Outward (away from body) → abduction. Inward (medial, toward
 *      midline) → adduction. The anatomical adduction range is only
 *      ~30-50° so the elbow barely crosses past the shoulder; we
 *      cannot wait for it to cross the body centreline (it doesn't
 *      get that far in a normal adduction).
 *    • Near neutral (arm hanging straight at side): both dx and dy
 *      are small → deadband → null. Direction isn't classified and
 *      no peak slot is updated. */
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
  // the test-side shoulder. Multiplying by this normalises the math
  // for left and right sides regardless of camera mirroring.
  const outwardSign = Math.sign(s.x - body.centreX);
  if (outwardSign === 0) return null;

  // Image y axis grows DOWNWARD, so (s.y - e.y) > 0 means the elbow
  // is above the shoulder on screen.
  const dyRatio = (s.y - e.y) / body.shoulderWidth;
  if (dyRatio > ABDUCTION_OVERHEAD_FRAC) {
    // Y-axis override: elbow clearly above shoulder = somewhere on
    // the abduction arc (60°–180°). Skip the x check entirely so
    // noise at the overhead end can't flip the label to adduction.
    return "abduction";
  }

  // At-or-below shoulder height: use x position relative to the
  // test-side shoulder. Outward = away from body = abduction.
  // Medial = elbow has drifted toward the body midline = adduction.
  const dxRatio = ((e.x - s.x) * outwardSign) / body.shoulderWidth;
  if (Math.abs(dxRatio) < DIRECTION_DEADBAND_FRAC) return null;
  return dxRatio > 0 ? "abduction" : "adduction";
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

// ─── Compensatory movement detection (Shoulder Flexion+Extension) ─
//
// Mirror of shoulder.ts. See that file for full rationale; this twin
// runs against BlazePose-tfjs (LM_LIVE) keypoints in the live flow.

export type CompensationSeverity = "high" | "medium" | "low";

export interface Compensation {
  type: "trunk_lean" | "shoulder_elevation" | "elbow_bend";
  label: string;
  severity: CompensationSeverity;
  flagged: boolean;
  details?: string;
}

const TRUNK_LEAN_THRESHOLD_DEG = 15;
const SHOULDER_ELEVATION_THRESHOLD_PX = 15;
const ELBOW_BEND_THRESHOLD_DEG = 160;
const COMPENSATION_BASELINE_FRAME_COUNT = 10;
/** ±N frames around each peak in which the elbow-bend check is run.
 *  See shoulder.ts for the rationale. */
const ELBOW_PEAK_WINDOW_FRAMES = 3;

/** Angle between the NOSE → test-side HIP line and image vertical,
 *  in degrees. Upright posture ≈ 0°.
 *
 *  NOSE (not shoulder) because the shoulder acromion landmark moves
 *  substantially during shoulder ROM tests (scapular elevation +
 *  protraction), which contaminates the trunk-lean signal. The nose
 *  stays put while the trunk actually moves. See shoulder.ts for
 *  the long-form rationale. Null when visibility fails. */
export function computeTrunkLeanAngle(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const nose = keypoints[LM.NOSE];
  const h = keypoints[idx.hip];
  if (!nose || !h) return null;
  if ((nose.score ?? 0) < VIS_THRESHOLD || (h.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = nose.x - h.x;
  const dy = nose.y - h.y;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  return Math.abs(angleBetween(0, -1, dx, dy));
}

/** Pixel distance between the test-side shoulder and hip landmarks.
 *  Decreases when the patient shrugs. Null when visibility fails. */
export function computeShoulderHipDistance(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const h = keypoints[idx.hip];
  if (!s || !h) return null;
  if ((s.score ?? 0) < VIS_THRESHOLD || (h.score ?? 0) < VIS_THRESHOLD) return null;
  return Math.hypot(s.x - h.x, s.y - h.y);
}

/** Interior angle at the elbow (shoulder → elbow → wrist), in degrees.
 *  Fully extended arm ≈ 180°. Null when any landmark fails. */
export function computeElbowAngle(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const idx = SIDE_INDICES[side];
  const s = keypoints[idx.shoulder];
  const e = keypoints[idx.elbow];
  const w = keypoints[idx.wrist];
  if (!s || !e || !w) return null;
  if (
    (s.score ?? 0) < VIS_THRESHOLD ||
    (e.score ?? 0) < VIS_THRESHOLD ||
    (w.score ?? 0) < VIS_THRESHOLD
  ) return null;
  const v1x = s.x - e.x, v1y = s.y - e.y;
  const v2x = w.x - e.x, v2y = w.y - e.y;
  const dot = v1x * v2x + v1y * v2y;
  const cross = v1x * v2y - v1y * v2x;
  return (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
}

/** Stateful tracker — see shoulder.ts for full docs. */
export class ShoulderFlexExtCompensationTracker {
  private readonly side: "left" | "right";
  private trunkBaselineSamples: number[] = [];
  private hipDistBaselineSamples: number[] = [];
  private trunkBaseline: number | null = null;
  private hipDistBaseline: number | null = null;
  private trunkLeanPeak = 0;
  private shoulderRisePeak = 0;
  private elbowBendPeak = 180;
  private trunkFlagged = false;
  private elevFlagged = false;
  private bendFlagged = false;
  private currentTrunkActive = false;
  private currentElevActive = false;
  private currentBendActive = false;
  // Per-frame elbow-angle history. The peak-windowed check in
  // finish() reads slices of this around the marked peak indices.
  private elbowAnglesPerFrame: (number | null)[] = [];
  private frameCounter = 0;
  private primaryPeakFrame: number | null = null;
  private secondaryPeakFrame: number | null = null;

  constructor(side: "left" | "right") {
    this.side = side;
  }

  feed(keypoints: Keypoint[]): void {
    this.currentTrunkActive = false;
    this.currentElevActive = false;
    this.currentBendActive = false;

    const trunk = computeTrunkLeanAngle(keypoints, this.side);
    const hipDist = computeShoulderHipDistance(keypoints, this.side);
    const elbow = computeElbowAngle(keypoints, this.side);

    if (trunk !== null && this.trunkBaselineSamples.length < COMPENSATION_BASELINE_FRAME_COUNT) {
      this.trunkBaselineSamples.push(trunk);
      if (this.trunkBaselineSamples.length === COMPENSATION_BASELINE_FRAME_COUNT) {
        this.trunkBaseline = compMean(this.trunkBaselineSamples);
      }
    }
    if (hipDist !== null && this.hipDistBaselineSamples.length < COMPENSATION_BASELINE_FRAME_COUNT) {
      this.hipDistBaselineSamples.push(hipDist);
      if (this.hipDistBaselineSamples.length === COMPENSATION_BASELINE_FRAME_COUNT) {
        this.hipDistBaseline = compMean(this.hipDistBaselineSamples);
      }
    }

    if (trunk !== null && this.trunkBaseline !== null) {
      const deviation = Math.abs(trunk - this.trunkBaseline);
      if (deviation > this.trunkLeanPeak) this.trunkLeanPeak = deviation;
      if (deviation > TRUNK_LEAN_THRESHOLD_DEG) {
        this.trunkFlagged = true;
        this.currentTrunkActive = true;
      }
    }
    if (hipDist !== null && this.hipDistBaseline !== null) {
      // Shrug detection: shoulder rises away from the hip, so the
      // euclidean shoulder→hip distance INCREASES above baseline.
      const rise = hipDist - this.hipDistBaseline;
      if (rise > this.shoulderRisePeak) this.shoulderRisePeak = rise;
      if (rise > SHOULDER_ELEVATION_THRESHOLD_PX) {
        this.elevFlagged = true;
        this.currentElevActive = true;
      }
    }
    // Elbow: collect now, evaluate in finish() against the peak
    // windows. Banner still shows current-frame state.
    this.elbowAnglesPerFrame.push(elbow);
    if (elbow !== null && elbow < ELBOW_BEND_THRESHOLD_DEG) {
      this.currentBendActive = true;
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
    if (this.currentTrunkActive) {
      out.push({ type: "trunk_lean", label: "Trunk Lean", severity: "high", flagged: true });
    }
    if (this.currentElevActive) {
      out.push({ type: "shoulder_elevation", label: "Shoulder Elevation", severity: "medium", flagged: true });
    }
    if (this.currentBendActive) {
      out.push({ type: "elbow_bend", label: "Elbow Bend", severity: "medium", flagged: true });
    }
    return out;
  }

  finish(): Compensation[] {
    // Peak-windowed elbow scan. Mirror of shoulder.ts.
    for (const peakIdx of [this.primaryPeakFrame, this.secondaryPeakFrame]) {
      if (peakIdx === null) continue;
      const lo = Math.max(0, peakIdx - ELBOW_PEAK_WINDOW_FRAMES);
      const hi = Math.min(this.elbowAnglesPerFrame.length, peakIdx + ELBOW_PEAK_WINDOW_FRAMES + 1);
      for (let i = lo; i < hi; i++) {
        const ang = this.elbowAnglesPerFrame[i];
        if (ang === null || ang === undefined) continue;
        if (ang < this.elbowBendPeak) this.elbowBendPeak = ang;
        if (ang < ELBOW_BEND_THRESHOLD_DEG) this.bendFlagged = true;
      }
    }

    // Always populate `details` (diagnostic visibility).
    return [
      {
        type: "trunk_lean",
        label: "Trunk Lean",
        severity: "high",
        flagged: this.trunkFlagged,
        details: `Peak deviation ${this.trunkLeanPeak.toFixed(1)}° from baseline (threshold ${TRUNK_LEAN_THRESHOLD_DEG}°)`,
      },
      {
        type: "shoulder_elevation",
        label: "Shoulder Elevation",
        severity: "medium",
        flagged: this.elevFlagged,
        details: `Shoulder-hip distance rose ${this.shoulderRisePeak.toFixed(0)} px above baseline (threshold ${SHOULDER_ELEVATION_THRESHOLD_PX} px)`,
      },
      {
        type: "elbow_bend",
        label: "Elbow Bend",
        severity: "medium",
        flagged: this.bendFlagged,
        details: `Minimum elbow angle at peak: ${this.elbowBendPeak.toFixed(0)}° (threshold ${ELBOW_BEND_THRESHOLD_DEG}°)`,
      },
    ];
  }
}

function compMean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}
