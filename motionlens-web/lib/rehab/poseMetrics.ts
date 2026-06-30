// Pure pose-derived metrics for the rehab module.
//
// Mirrors the style of lib/biomech/*-live.ts helpers
// (computeShoulderLineAngleFromHorizontal etc.) but lives under
// lib/rehab/ so it never collides with — or risks modification of —
// existing biomech code. New rehab-specific pose metrics that don't
// have an equivalent in biomech go here.

import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";

const VIS_THRESHOLD = 0.15;

/** Signed pelvic tilt (degrees) of the line from LEFT_HIP to
 *  RIGHT_HIP relative to horizontal. Image coordinates are y-down,
 *  so:
 *    • 0°     = pelvis perfectly level
 *    • +5°    = right hip dropped relative to left
 *    • -5°    = left hip dropped relative to right
 *
 *  Returns null when either hip landmark fails visibility, or when
 *  the two hips project to coincident image points (degenerate).
 *  Pure — no state, no side effects.
 */
export function computePelvicTiltDeg(
  keypoints: Keypoint[],
): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!lHip || !rHip) return null;
  if ((lHip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rHip.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = rHip.x - lHip.x;
  const dy = rHip.y - lHip.y;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

/** Pixel-space midpoint x of LEFT_HIP and RIGHT_HIP — a cheap
 *  medio-lateral centre-of-mass proxy. Returns null when either
 *  hip fails visibility. Pure helper for the weight-shift game.
 *
 *  Note: no equivalent exists in lib/biomech/*-live.ts —
 *  computeHipShoulderMidX in hip-live.ts is the SHOULDER midpoint,
 *  not the hip midpoint. Added here rather than in biomech to keep
 *  the "rehab module never modifies biomech" invariant.
 */
export function computeHipMidX(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!lHip || !rHip) return null;
  if ((lHip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rHip.score ?? 0) < VIS_THRESHOLD) return null;
  return (lHip.x + rHip.x) / 2;
}

/** Shoulder pixel-width — |rSh.x - lSh.x|. Used as a body-scale
 *  reference (e.g. mapping medio-lateral hip shifts onto a
 *  normalised cursor). Duplicates the helper in
 *  lib/biomech/neck-live.ts intentionally so the rehab module
 *  carries no compile-time dependency on a biomech file that might
 *  evolve for assessment-specific reasons. */
export function computeShoulderWidth(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!lSh || !rSh) return null;
  if ((lSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rSh.score ?? 0) < VIS_THRESHOLD) return null;
  return Math.abs(rSh.x - lSh.x);
}

/** Trunk angle from HORIZONTAL in degrees — angle between the
 *  trunk segment (shoulder-mid → hip-mid) and the horizontal x-axis.
 *  Used by B4 Bird-Dog where the target is a HORIZONTAL trunk
 *  (quadruped position, neutral spine):
 *    • 0°    = trunk horizontal (good bird-dog form)
 *    • 30°   = trunk drifted off horizontal (lumbar sag / pike up)
 *    • 90°   = trunk vertical (standing — not relevant here)
 *
 *  Unsigned magnitude — sagging back and piked-up back BOTH read
 *  as positive deviations from horizontal. Pure helper, mirrors
 *  the style of the other trunk-angle functions in this file.
 *
 *  NOTE — distinct from computeTrunkExtensionAngleDeg which uses
 *  vertical-up as its reference (designed for standing back-
 *  extension where the trunk's neutral is vertical). Bird-dog
 *  needs the horizontal reference because the patient is in
 *  quadruped position.
 */
export function computeTrunkAngleFromHorizontal(
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
  const hipMidX = (lHip.x + rHip.x) / 2;
  const hipMidY = (lHip.y + rHip.y) / 2;
  const dx = hipMidX - shMidX;
  const dy = hipMidY - shMidY;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  // Unsigned angle from horizontal-x axis. atan2(|dy|, |dx|):
  // dy small (trunk horizontal) ⇒ angle ~0; dy large (trunk
  // vertical) ⇒ angle ~90°.
  return (Math.atan2(Math.abs(dy), Math.abs(dx)) * 180) / Math.PI;
}

/** LOW-CONFIDENCE proxy for spinal flexion / extension at the
 *  thoracolumbar region, derived from head position relative to
 *  shoulder line. Used by B6 Cat-Cow.
 *
 *  Why a proxy is the best we can do here — BlazePose has no
 *  mid-spine landmark, so direct curvature isn't measurable. In
 *  cat-cow specifically, the HEAD tucks down (chin to chest) at
 *  spinal flexion ("cat") and lifts up (look forward / up) at
 *  spinal extension ("cow"). Head position is a reliable PROXY
 *  for the spinal cycle even though it's not a direct curvature
 *  measurement. The B6 UI must surface a "trend only — gentle
 *  spinal mobility" caveat.
 *
 *  Signed:
 *    • POSITIVE  = head dropped below shoulder line (CAT, flexion)
 *    • ZERO      = head at shoulder height (neutral quadruped)
 *    • NEGATIVE  = head lifted above shoulder line (COW, extension)
 *
 *  Normalised by shoulder-width so the magnitude is roughly
 *  comparable across patient sizes and clinic-camera distances.
 */
export function computeSpineFlexionProxyDeg(
  keypoints: Keypoint[],
): number | null {
  const nose = keypoints[LM.NOSE];
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!nose || !lSh || !rSh) return null;
  if ((nose.score ?? 0) < VIS_THRESHOLD) return null;
  if ((lSh.score ?? 0) < VIS_THRESHOLD) return null;
  if ((rSh.score ?? 0) < VIS_THRESHOLD) return null;
  const shMidY = (lSh.y + rSh.y) / 2;
  const shoulderWidth = Math.abs(rSh.x - lSh.x);
  if (shoulderWidth < 1e-3) return null;
  // Vertical offset of nose from shoulder line, normalised by
  // shoulder width, scaled to a degrees-like range so the Trace
  // mechanic's accuracyTolerance feels familiar.
  const normalisedOffset = (nose.y - shMidY) / shoulderWidth;
  // Map ratio (~−2 to +2 in practice) to a degree-like scale by
  // multiplying by 30. Cat (nose-below-shoulder ratio ~+1.5)
  // produces +45-ish; cow (nose-above-shoulder ratio ~−1.0)
  // produces −30-ish. Tunable downstream.
  return normalisedOffset * 30;
}

/** Hip-hinge tilt angle in degrees — alias of
 *  computeTrunkExtensionAngleDeg, exported under a hinge-specific
 *  name so the B5 page reads cleanly. Math is identical: trunk-tilt
 *  magnitude from vertical-up in image space.
 *    • 0°    = upright neutral
 *    • 20°   = mid-range hinge (trunk leaning forward)
 *    • 45°+  = deep hinge (parallel-to-floor approached)
 *
 *  Unsigned: forward bend (the prescribed direction for hip hinge)
 *  and backward bend both read positive. B5 setup help instructs
 *  forward-only motion so this is clean for the intended drill.
 *
 *  The `side` parameter is accepted for future extension (signed
 *  lateral-aware direction) but is currently unused — the math
 *  uses the trunk midline only, which is symmetric.
 */
export function computeHipHingeAngleDeg(
  keypoints: Keypoint[],
  _side?: "left" | "right",
): number | null {
  return computeTrunkExtensionAngleDeg(keypoints);
}

/** Unsigned trunk-tilt angle in degrees — angle between the
 *  trunk segment (hip-mid → shoulder-mid) and vertical-up,
 *  measured in image space (lateral-view assumption).
 *
 *  Reference cases:
 *    • 0°    = upright neutral posture
 *    • 10°   = mild lean from vertical
 *    • 20-25° = clinically meaningful back-extension range
 *
 *  Returns MAGNITUDE only — both forward flexion and backward
 *  extension produce positive values. The B2 Back Extension
 *  exercise instructs the patient to extend BACKWARD only, so the
 *  signal cleanly tracks the extension arc; forward-bending during
 *  the drill would inflate the reading and is documented in setup
 *  guidance.
 *
 *  NOTE — no equivalent in biomech. The hip-live helper measures
 *  thigh-vs-trunk (sagittal hip flexion/extension); back extension
 *  is a SPINE motion that biomech doesn't surface. Added here so
 *  the rehab module can drive Rep-Count without modifying biomech.
 */
export function computeTrunkExtensionAngleDeg(
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
  const hipMidX = (lHip.x + rHip.x) / 2;
  const hipMidY = (lHip.y + rHip.y) / 2;
  // Trunk vector from hip-mid UP to shoulder-mid.
  // Image y is down-positive, so shoulder above hip means dy < 0.
  const dx = shMidX - hipMidX;
  const dy = shMidY - hipMidY;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  // Angle from vertical-up (= (0, -1) in image coords).
  // atan2(|dx|, -dy) — both args positive for any upright torso.
  return (Math.atan2(Math.abs(dx), -dy) * 180) / Math.PI;
}

/** Unsigned forward-head offset in degrees — angle between the
 *  (shoulder → ear) vector and vertical-up, in image space.
 *  Captures the head-jutted-forward postural fault commonly seen
 *  in screen workers / cervical pain patients.
 *
 *  Reference cases (typical clinic camera, lateral or near-lateral):
 *    • 0-5°    = good posture (ear roughly above shoulder)
 *    • 10-15°  = mild forward head
 *    • 20-30°  = pronounced forward-head posture
 *
 *  Unsigned — left/right side of the body is selectable via the
 *  `side` parameter. Forward vs backward is conflated (both produce
 *  positive); for the B1 Posture Hold exercise this is fine because
 *  the in-zone target is LOW angle (good posture), not directional.
 *
 *  NOTE — no equivalent in biomech. The neck-live helper measures
 *  cervical flexion/extension (head pitch); forward-head is the
 *  HORIZONTAL translation of the head over the shoulder which the
 *  neck-live model doesn't isolate.
 */
export function computeForwardHeadOffsetDeg(
  keypoints: Keypoint[],
  side: "left" | "right" = "right",
): number | null {
  const earIdx = side === "left" ? LM.LEFT_EAR : LM.RIGHT_EAR;
  const shoulderIdx =
    side === "left" ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER;
  const ear = keypoints[earIdx];
  const shoulder = keypoints[shoulderIdx];
  if (!ear || !shoulder) return null;
  if ((ear.score ?? 0) < VIS_THRESHOLD) return null;
  if ((shoulder.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = ear.x - shoulder.x;
  const dy = ear.y - shoulder.y;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  // Ear sits ABOVE shoulder (dy < 0). Angle of (shoulder → ear)
  // from vertical-up, unsigned.
  return (Math.atan2(Math.abs(dx), -dy) * 180) / Math.PI;
}

/** Signed lateral trunk flexion angle in degrees — angle of the
 *  trunk segment from vertical, measured in the frontal plane.
 *  Sign convention: POSITIVE when the trunk leans to the patient's
 *  RIGHT side (anatomical right). Negative when leaning to the
 *  patient's left.
 *
 *  Reference cases:
 *    • 0°       = upright neutral, no lateral lean
 *    • +15°     = mid-range bend to patient's right
 *    • +25-30°  = upper end of clinical active ROM (right bend)
 *    • Negative = mirrored for left bends
 *
 *  Derivation — at neutral, hip-mid sits directly below shoulder-mid
 *  (raw image: hipMid.y > shMid.y, hipMid.x ≈ shMid.x). When the
 *  patient bends to their RIGHT anatomically, BOTH shoulders' avg x
 *  shifts to their right side, which is LOWER x in the raw image
 *  (because BlazePose returns un-mirrored pixel coords; the patient's
 *  right is on the image's left). So (hipMid.x − shMid.x) increases
 *  ⇒ atan2(positive, positive) > 0 ⇒ positive angle for right bend. ✓
 *
 *  NOTE — no biomech equivalent. Hip-live + neck-live model sagittal
 *  motion; lateral trunk flexion is a coronal-plane spine motion.
 */
export function computeLateralTrunkFlexionDeg(
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
  const hipMidX = (lHip.x + rHip.x) / 2;
  const hipMidY = (lHip.y + rHip.y) / 2;
  // Vector pointing FROM shoulder-mid DOWN to hip-mid (the trunk
  // axis as seen "from above"). At neutral upright: dx ≈ 0, dy > 0.
  const dx = hipMidX - shMidX;
  const dy = hipMidY - shMidY;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  // Signed angle from vertical-down (= +y axis). atan2(dx, dy)
  // returns 0 at neutral, positive when shoulder-mid is to the
  // RIGHT of hip-mid in raw image (= patient bending RIGHT, since
  // BlazePose raw coords are un-mirrored).
  return (Math.atan2(dx, dy) * 180) / Math.PI;
}

/** COARSE proxy for scapular retraction, derived from the
 *  narrowing of shoulder-to-shoulder pixel width relative to a
 *  calibrated baseline. This is a coaching-cue signal, NOT a
 *  scapular position measurement.
 *
 *  Why a proxy is the best we can do here — the scapula has no
 *  BlazePose landmark. True retraction is mostly posterior
 *  translation (invisible in 2-D frontal view) plus a small
 *  medial component (the shoulder landmarks pulling slightly
 *  toward midline). Only the medial component is detectable here,
 *  and only as a small (~3-10 %) reduction in shoulder pixel
 *  width vs the relaxed baseline. The PRD explicitly flags S6 as
 *  a coarse cue; callers MUST surface this in the UI.
 *
 *  Returns a number in [0, ~15] under typical clinic geometry:
 *    • 0 at baseline (relaxed shoulders, blades neutral)
 *    • Positive as the patient retracts; magnitude ≈ percentage
 *      of width narrowing
 *
 *  Caller responsibility — track the baseline width separately
 *  (e.g. average shoulder width over the first ~10 valid frames
 *  while the patient stands neutral) and pass it in. The helper
 *  stays pure: no module-level state.
 */
export function computeScapularRetractionProxy(
  keypoints: Keypoint[],
  baselineShoulderWidth: number,
): number | null {
  if (baselineShoulderWidth <= 0) return null;
  const current = computeShoulderWidth(keypoints);
  if (current === null) return null;
  // Clamp to zero floor — width WIDER than baseline (patient
  // protracted further or moved closer to the camera) is not
  // retraction; treat as "neutral".
  return Math.max(0, (1 - current / baselineShoulderWidth) * 100);
}

/** LOW-CONFIDENCE proxy for shoulder external rotation in degrees,
 *  derived from the lateral excursion of the wrist relative to the
 *  elbow in a 2-D frontal image. This is a TREND-ONLY signal, NOT
 *  an absolute clinical ER measurement.
 *
 *  Why a proxy is the best we can do here — true gleno-humeral
 *  rotation is axial (humerus spinning about its long axis) and is
 *  invisible in a single 2-D frontal view. The PRD explicitly
 *  warns the S5 reading should be reported as a within-patient
 *  trend, not an absolute angle. Callers MUST surface this caveat
 *  in the UI.
 *
 *  Method — with the elbow tucked at 90° flexion (forearm forward,
 *  pointing toward camera), external rotation swings the forearm
 *  out laterally in the image plane:
 *    • Neutral, forearm-forward: wrist projects near the elbow's
 *      x → lateral component ≈ 0 → proxy ≈ 0°
 *    • Half rotated: lateral ≈ half forearm length → proxy ≈ 30°
 *    • Full ER, forearm-lateral: lateral ≈ full forearm length →
 *      proxy ≈ 90°
 *
 *  Assumes the elbow stays at ~90° flexion AND remains tucked to
 *  the side throughout the rep. Either elbow extension OR drift
 *  of the elbow away from the side will distort the reading —
 *  caller should detect and coach against drift separately.
 */
export function computeForearmRotationProxyDeg(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const elbowIdx = side === "left" ? LM.LEFT_ELBOW : LM.RIGHT_ELBOW;
  const wristIdx = side === "left" ? LM.LEFT_WRIST : LM.RIGHT_WRIST;
  const elbow = keypoints[elbowIdx];
  const wrist = keypoints[wristIdx];
  if (!elbow || !wrist) return null;
  if ((elbow.score ?? 0) < VIS_THRESHOLD) return null;
  if ((wrist.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = wrist.x - elbow.x;
  const dy = wrist.y - elbow.y;
  const forearmLen = Math.hypot(dx, dy);
  if (forearmLen < 1e-4) return null;
  // sin(rotation) = |lateral component| / |forearm|.
  // Unsigned magnitude — side picker handles left-vs-right.
  const ratio = Math.min(1, Math.max(0, Math.abs(dx) / forearmLen));
  return (Math.asin(ratio) * 180) / Math.PI;
}

/** Unsigned hip abduction angle in degrees — angle between the
 *  thigh vector (hip → knee) and vertical-down, measured in image
 *  space (frontal-view assumption).
 *
 *  Reference cases:
 *    • 0°   = leg hanging straight down (standing neutral)
 *    • 30°  = mid-range active abduction
 *    • 45°  = upper end of typical clinical active ROM
 *    • 90°  = leg held horizontal (rare; usually passive only)
 *
 *  NOTE — no equivalent in lib/biomech/hip-live.ts. That file
 *  models sagittal flexion / extension only; abduction is not
 *  surfaced anywhere in biomech. Added here so the rehab module
 *  can drive cursor + score without modifying biomech.
 *
 *  CAVEAT — this is a 2-D image-plane projection. If the patient
 *  also flexes the hip (lifts leg forward), the apparent
 *  abduction angle inflates because the knee moves upward in the
 *  frontal image. The exercise's setup guidance instructs the
 *  patient to keep the leg purely lateral.
 */
export function computeHipAbductionDeg(
  keypoints: Keypoint[],
  side: "left" | "right",
): number | null {
  const hipIdx = side === "left" ? LM.LEFT_HIP : LM.RIGHT_HIP;
  const kneeIdx = side === "left" ? LM.LEFT_KNEE : LM.RIGHT_KNEE;
  const hip = keypoints[hipIdx];
  const knee = keypoints[kneeIdx];
  if (!hip || !knee) return null;
  if ((hip.score ?? 0) < VIS_THRESHOLD) return null;
  if ((knee.score ?? 0) < VIS_THRESHOLD) return null;
  const dx = knee.x - hip.x;
  const dy = knee.y - hip.y;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  // Unsigned magnitude — atan2 of |horizontal| over |vertical|.
  // Side picker handles left-vs-right; we don't care which way the
  // thigh swings.
  return (Math.atan2(Math.abs(dx), Math.abs(dy)) * 180) / Math.PI;
}
