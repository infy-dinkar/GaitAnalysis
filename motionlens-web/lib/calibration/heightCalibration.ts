// Height-based scale calibration.
//
// How it works:
//   1. Doctor enters / confirms the patient's standing height in cm.
//   2. Patient stands straight, fully in frame.
//   3. The system samples the patient's body PIXEL height — from the
//      highest visible head landmark (nose / eye / ear) down to the
//      lowest visible foot landmark (heel / ankle / foot_index) — and
//      requires the reading to be stable for a short averaged window
//      before locking in.
//   4. pixels_per_cm = body_pixel_height / (height_cm × NOSE_TO_FLOOR_HEIGHT_FRACTION)
//      The anthropometric correction accounts for the fact that the
//      nose (our highest reliable head landmark) sits at ~87 % of
//      total stature for adults.
//
// This module is consumer-agnostic — Functional Reach uses it today,
// and D3 Single-Leg Hop / D4 CMJ will reuse it without duplication.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import {
  NOSE_TO_FLOOR_HEIGHT_FRACTION,
  type CalibrationResult,
} from "./types";

/** Score below which a head/foot keypoint is treated as occluded
 *  and excluded from the body-pixel-height measurement. */
export const HEAD_FOOT_VIS_THRESHOLD = 0.35;

/** Minimum height entry accepted. Anything below this is almost
 *  certainly a typo. */
export const MIN_HEIGHT_CM = 80;
export const MAX_HEIGHT_CM = 230;

/** How long the body-pixel-height reading must stay stable before
 *  the calibration locks in (sample frames at ~6 Hz detection). */
export const STABLE_FRAMES_REQUIRED = 6;

/** Two body-pixel-height readings agree when they differ by less
 *  than this fraction. 3 % is comfortably tighter than typical
 *  MediaPipe head/foot landmark jitter, so a stable reading really
 *  is stable. */
export const STABLE_TOLERANCE_FRACTION = 0.03;

const HEAD_LANDMARK_INDICES: number[] = [
  LM.NOSE,
  LM.LEFT_EYE_INNER, LM.LEFT_EYE, LM.LEFT_EYE_OUTER,
  LM.RIGHT_EYE_INNER, LM.RIGHT_EYE, LM.RIGHT_EYE_OUTER,
  LM.LEFT_EAR, LM.RIGHT_EAR,
];

const FOOT_LANDMARK_INDICES: number[] = [
  LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
  LM.LEFT_HEEL, LM.RIGHT_HEEL,
  LM.LEFT_FOOT_INDEX, LM.RIGHT_FOOT_INDEX,
];

const TORSO_REQUIRED: number[] = [
  LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  LM.LEFT_HIP, LM.RIGHT_HIP,
];

export interface BodyHeightReading {
  /** Body pixel height (max foot y − min head y) in source image px. */
  body_pixel_height_px: number;
  /** Highest visible head landmark y (top of body in image y-down). */
  top_y_px: number;
  /** Lowest visible foot landmark y (bottom of body in image y-down). */
  bottom_y_px: number;
}

/** Compute the patient's body pixel height from a single pose. Returns
 *  null when not enough head + foot + torso landmarks are visible. */
export function computeBodyPixelHeight(kp: Keypoint[]): BodyHeightReading | null {
  if (!kp || kp.length === 0) return null;

  // Torso anchor must be visible — otherwise we're probably looking at
  // only the upper or lower body and the height reading would be
  // meaningless.
  for (const i of TORSO_REQUIRED) {
    const p = kp[i];
    if (!p || (p.score ?? 0) < HEAD_FOOT_VIS_THRESHOLD) return null;
  }

  let topY: number | null = null;
  for (const i of HEAD_LANDMARK_INDICES) {
    const p = kp[i];
    if (!p || (p.score ?? 0) < HEAD_FOOT_VIS_THRESHOLD) continue;
    if (topY === null || p.y < topY) topY = p.y;
  }
  if (topY === null) return null;

  let bottomY: number | null = null;
  for (const i of FOOT_LANDMARK_INDICES) {
    const p = kp[i];
    if (!p || (p.score ?? 0) < HEAD_FOOT_VIS_THRESHOLD) continue;
    if (bottomY === null || p.y > bottomY) bottomY = p.y;
  }
  if (bottomY === null) return null;

  const bodyPx = bottomY - topY;
  if (bodyPx <= 0) return null;
  return {
    body_pixel_height_px: bodyPx,
    top_y_px: topY,
    bottom_y_px: bottomY,
  };
}

/** Is the patient fully in frame for calibration?  Required so the
 *  body-pixel-height reading isn't artificially truncated. Returns a
 *  human-readable "reason" hint when the answer is no, for the
 *  coaching layer. */
export function checkBodyInFrame(
  kp: Keypoint[],
  reading: BodyHeightReading | null,
  frameHeightPx: number,
): { ok: boolean; reason: string } {
  if (!reading) {
    // Discriminate between "no head" / "no feet" / "no torso" so the
    // operator gets actionable coaching instead of a generic message.
    const headSeen = HEAD_LANDMARK_INDICES.some(
      (i) => (kp[i]?.score ?? 0) >= HEAD_FOOT_VIS_THRESHOLD,
    );
    const feetSeen = FOOT_LANDMARK_INDICES.some(
      (i) => (kp[i]?.score ?? 0) >= HEAD_FOOT_VIS_THRESHOLD,
    );
    const torsoSeen = TORSO_REQUIRED.every(
      (i) => (kp[i]?.score ?? 0) >= HEAD_FOOT_VIS_THRESHOLD,
    );
    if (!torsoSeen) return { ok: false, reason: "torso_missing" };
    if (!headSeen) return { ok: false, reason: "head_missing" };
    if (!feetSeen) return { ok: false, reason: "feet_missing" };
    return { ok: false, reason: "body_partial" };
  }

  // Margin: avoid locking in when head or feet are right at the
  // frame edge (they'd be cropped and pixel-height under-reads).
  const edgeMarginPx = Math.max(8, frameHeightPx * 0.02);
  if (reading.top_y_px < edgeMarginPx) {
    return { ok: false, reason: "head_at_frame_edge" };
  }
  if (reading.bottom_y_px > frameHeightPx - edgeMarginPx) {
    return { ok: false, reason: "feet_at_frame_edge" };
  }
  return { ok: true, reason: "" };
}

/** Two body-height readings are stable iff they differ by less than
 *  STABLE_TOLERANCE_FRACTION. */
export function areReadingsStable(
  a: BodyHeightReading,
  b: BodyHeightReading,
  tolerance = STABLE_TOLERANCE_FRACTION,
): boolean {
  const ref = Math.min(a.body_pixel_height_px, b.body_pixel_height_px);
  if (ref <= 0) return false;
  return (
    Math.abs(a.body_pixel_height_px - b.body_pixel_height_px) / ref <= tolerance
  );
}

/** Pure conversion. Computes pixels_per_cm from a confirmed body
 *  pixel height (e.g. the median of the stable window) + the
 *  doctor-entered patient height in cm.
 *
 *  The anthropometric NOSE_TO_FLOOR fraction (~0.87) converts the
 *  *measured* body span (highest head landmark down to the floor)
 *  into the *true* head-top-to-floor span — i.e. the patient's
 *  standing height. */
export function pixelsPerCmFromHeight(
  bodyPixelHeightPx: number,
  patientHeightCm: number,
): number | null {
  if (
    !Number.isFinite(bodyPixelHeightPx) ||
    !Number.isFinite(patientHeightCm) ||
    bodyPixelHeightPx <= 0 ||
    patientHeightCm < MIN_HEIGHT_CM ||
    patientHeightCm > MAX_HEIGHT_CM
  ) {
    return null;
  }
  const effectiveHeightCm = patientHeightCm * NOSE_TO_FLOOR_HEIGHT_FRACTION;
  if (effectiveHeightCm <= 0) return null;
  return bodyPixelHeightPx / effectiveHeightCm;
}

/** Assemble a CalibrationResult from the locked reading + patient
 *  height. */
export function buildHeightCalibration(
  bodyPixelHeightPx: number,
  patientHeightCm: number,
  sourceFrame: { width: number; height: number },
): CalibrationResult | null {
  const ppc = pixelsPerCmFromHeight(bodyPixelHeightPx, patientHeightCm);
  if (ppc === null) return null;
  return {
    pixels_per_cm: ppc,
    detected_at_ms: Date.now(),
    source: "height_based",
    patient_height_cm: patientHeightCm,
    body_pixel_height_px: bodyPixelHeightPx,
    source_frame_px: { width: sourceFrame.width, height: sourceFrame.height },
  };
}
