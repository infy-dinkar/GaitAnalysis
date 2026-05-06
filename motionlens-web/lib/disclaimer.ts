// Single source of truth for the platform-wide report disclaimer.
// Used by every results screen (web) and by the gait PDF generator,
// so the wording stays identical everywhere.

export const REPORT_DISCLAIMER =
  "DISCLAIMER: This report is generated from 2D pose estimation and is " +
  "intended for movement tracking and screening purposes only. Measurements " +
  "are approximate and may be affected by camera angle, lighting, and patient " +
  "positioning. Rotation measurements (internal/external rotation) are " +
  "particularly sensitive to these factors. This is not a medical diagnosis " +
  "and should not replace clinical assessment by a qualified practitioner.";
