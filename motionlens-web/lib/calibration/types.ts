// Shared calibration types.
//
// MotionLens distance-based tests (C6 Functional Reach, future D3
// Single-Leg Hop, D4 CMJ) convert pixel measurements to centimetres
// using a CalibrationResult acquired before the test. The provider
// is pluggable — currently only height-based calibration is
// implemented (lib/calibration/heightCalibration.ts). The `source`
// field records which provider produced the result so the report
// can label it accurately.

/** Anthropometric ratio: an upright adult's NOSE sits at ~0.87 of
 *  the total head-top-to-floor height. We measure body pixel height
 *  from the highest visible head landmark (often the nose or an
 *  ear) down to the lowest visible foot landmark; this constant
 *  converts that pixel span to the equivalent CM span using the
 *  doctor-entered total height. */
export const NOSE_TO_FLOOR_HEIGHT_FRACTION = 0.87;

export type CalibrationSource =
  | "height_based"
  | "manual"
  | "unknown";

export interface CalibrationResult {
  /** Multiply px by 1 / pixels_per_cm to get cm. Always > 0. */
  pixels_per_cm: number;
  /** Epoch ms when the calibration was accepted. */
  detected_at_ms: number;
  /** Where the calibration came from. */
  source: CalibrationSource;
  /** Patient height (cm) — populated by the height provider; null
   *  for other providers. */
  patient_height_cm: number | null;
  /** Measured body pixel height (highest head landmark to lowest
   *  foot landmark) used to derive pixels_per_cm; null for non-
   *  height providers. */
  body_pixel_height_px: number | null;
  /** Source video / camera frame dimensions, for downstream sanity. */
  source_frame_px: { width: number; height: number } | null;
}

export function pxToCm(
  px: number,
  calibration: CalibrationResult | null,
): number | null {
  if (
    !calibration ||
    !Number.isFinite(calibration.pixels_per_cm) ||
    calibration.pixels_per_cm <= 0
  ) {
    return null;
  }
  return px / calibration.pixels_per_cm;
}

export function formatDistance(
  px: number,
  calibration: CalibrationResult | null,
  digits = 1,
): { text: string; calibrated: boolean; value_cm: number | null; value_px: number } {
  const cm = pxToCm(px, calibration);
  if (cm === null) {
    return {
      text: `${px.toFixed(digits)} px (relative)`,
      calibrated: false,
      value_cm: null,
      value_px: px,
    };
  }
  return {
    text: `${cm.toFixed(digits)} cm`,
    calibrated: true,
    value_cm: cm,
    value_px: px,
  };
}
