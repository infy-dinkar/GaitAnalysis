// Trendelenburg single-leg-stance test — math, conventions, classification.
//
// Spec: MotionLens Test Battery v1.0, Test A4 (pp. 6-7).
// Pose model: MoveNet 17-keypoint (LM indices in @/lib/pose/landmarks).
//
// Sign conventions (PDF Appendix B):
//   pelvic tilt:  angle of (hip-23 → hip-24) line vs horizontal,
//                 positive = LEFT side down.
//   trunk lean:   angle of (hip-mid → shoulder-mid) line vs vertical,
//                 frontal view, positive = lean to patient's RIGHT.
//
// Drop magnitude: in a Trendelenburg test the dropping side is the
// side OPPOSITE the stance leg. We re-orient pelvic-tilt sign so that
// "drop on the lifted side" reads positive regardless of which leg is
// the stance leg. See `dropForStance()` below.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

// MoveNet score for a confidently visible joint typically 0.4-0.99.
// 0.3 lets metric math run on most in-frame joints without false-
// passing fully-occluded ones.
const VIS_THRESHOLD = 0.3;

// Sample rate for the per-frame time-series. 10 Hz balances clinical
// resolution against the size of the JSON payload we persist.
export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// Hold + early-termination thresholds (PDF Test A4).
export const TARGET_HOLD_SECONDS = 30;
export const STABLE_PORTION_START_SEC = 2;   // metrics use frames after this
export const SHORT_HOLD_THRESHOLD_SEC = 10;  // PDF: <10s → additional concern
export const PELVIC_SPIKE_TERMINATION_DEG = 15;
export const COMPENSATORY_TRUNK_LEAN_DEG = 7;

// Drop-magnitude classification (PDF Test A4 cutoffs).
export const NEGATIVE_DROP_MAX_DEG = 2;       // <2° → negative
export const COMPENSATED_DROP_MAX_DEG = 5;    // 2-5° → compensated, >5° → positive

// Single-leg-stance auto-detection: lifted ankle must be at least this
// fraction of the body's pixel-height above the stance ankle to count.
// 6% catches a clearly-lifted leg without false-positives from
// alignment noise.
export const STANCE_LIFT_RATIO = 0.06;

export type Side = "left" | "right";
export type Classification = "negative" | "compensated" | "positive";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

function angleFromHorizontal(vx: number, vy: number): number {
  return (Math.atan2(vy, vx) * 180) / Math.PI;
}

function angleFromVertical(vx: number, vy: number): number {
  return (Math.atan2(vx, vy) * 180) / Math.PI;
}

// ─── Per-frame computations ──────────────────────────────────────
//
// Pelvic tilt: line of (right-hip → left-hip) vs horizontal. In
// image-y-down coords, left-side-down → lHip.y > rHip.y → vy > 0
// → atan2(positive, positive) → small positive angle. Matches PDF
// Appendix B (positive = left-side-down).
export function computePelvicTilt(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!visible(lHip) || !visible(rHip)) return null;
  return angleFromHorizontal(lHip.x - rHip.x, lHip.y - rHip.y);
}

// Trunk lean (frontal): line of (hip-mid → shoulder-mid) vs vertical.
// Patient's RIGHT = camera's LEFT in mirror frontal view, so vx < 0
// when the patient leans to their own right → return positive.
export function computeTrunkLean(keypoints: Keypoint[]): number | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  const lSh  = keypoints[LM.LEFT_SHOULDER];
  const rSh  = keypoints[LM.RIGHT_SHOULDER];
  if (![lHip, rHip, lSh, rSh].every(visible)) return null;
  const hipMidX = (lHip.x + rHip.x) / 2;
  const hipMidY = (lHip.y + rHip.y) / 2;
  const shMidX  = (lSh.x  + rSh.x ) / 2;
  const shMidY  = (lSh.y  + rSh.y ) / 2;
  const vx = shMidX - hipMidX;
  const vy = shMidY - hipMidY;
  const magnitude = Math.abs(angleFromVertical(vx, vy));
  return (vx < 0 ? 1 : -1) * magnitude;
}

// Body-pixel-height proxy for "is this ankle clearly lifted?" check.
// shoulder-midpoint to ankle-midpoint distance in pixels.
function bodyPxHeight(keypoints: Keypoint[]): number | null {
  const ls = keypoints[LM.LEFT_SHOULDER];
  const rs = keypoints[LM.RIGHT_SHOULDER];
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (![ls, rs, la, ra].every(visible)) return null;
  const shMidY = (ls.y + rs.y) / 2;
  const anMidY = (la.y + ra.y) / 2;
  return Math.abs(anMidY - shMidY);
}

// Auto-detect single-leg stance:
//   stance = the ankle with LARGER y (lower in image, still on ground)
//   lifted = the ankle with SMALLER y (higher in image)
// Only fires when the lift exceeds STANCE_LIFT_RATIO × body height,
// which prevents false positives during normal weight shifts.
export function detectStanceSide(keypoints: Keypoint[]): Side | null {
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (!visible(la) || !visible(ra)) return null;
  const bodyH = bodyPxHeight(keypoints);
  if (!bodyH) return null;
  const liftPx = Math.abs(la.y - ra.y);
  if (liftPx < bodyH * STANCE_LIFT_RATIO) return null;
  // Lower image-y means lifted; stance is the OTHER side.
  return la.y < ra.y ? "right" : "left";
}

// ─── Hold-timeline aggregation ───────────────────────────────────
//
// Re-orients pelvic-tilt sign so "drop on the lifted side" is
// positive regardless of which leg is the stance leg. The PDF
// classification thresholds (2°, 5°) apply to this reoriented value.
export function dropForStance(pelvicTilt: number, stance: Side): number {
  // Stance right → lifted leg is left → left-down spec sign is already
  // positive when the pelvis drops on the lifted (left) side.
  // Stance left  → lifted leg is right → flip sign so right-down reads
  // as positive drop.
  return stance === "right" ? pelvicTilt : -pelvicTilt;
}

// Compensatory trunk lean: lean toward stance side reads positive.
// Trunk lean spec sign is positive = lean to patient's RIGHT, so:
//   stance right → lean right is compensatory → keep sign
//   stance left  → lean left  is compensatory → flip sign
export function leanTowardStance(trunkLean: number, stance: Side): number {
  return stance === "right" ? trunkLean : -trunkLean;
}

export function classifyMaxDrop(maxDropDeg: number): Classification {
  const v = Math.abs(maxDropDeg);
  if (v < NEGATIVE_DROP_MAX_DEG)    return "negative";
  if (v <= COMPENSATED_DROP_MAX_DEG) return "compensated";
  return "positive";
}

// ─── Per-side aggregate result ───────────────────────────────────
export interface TrendelenburgFrameSample {
  /** Ms since hold-start. */
  t_ms: number;
  /** Pelvic tilt at this frame, raw spec convention (left-down = +). */
  pelvic_tilt_deg: number | null;
  /** Trunk lean (frontal, raw spec convention: lean to patient's right = +). */
  trunk_lean_deg: number | null;
}

export interface TrendelenburgSideResult {
  side_tested: Side;
  hold_seconds: number;
  /** Max pelvic drop on the LIFTED side, in degrees (always >= 0). */
  max_drop_deg: number;
  /** Mean pelvic drop over the stable portion (after first 2s). */
  mean_drop_deg: number;
  /** Maximum compensatory trunk lean toward the stance side, in degrees. */
  max_compensatory_lean_deg: number;
  classification: Classification;
  /** True if the patient could not maintain the stance for >= 10s. */
  short_hold: boolean;
  /** True if compensatory trunk lean exceeded the threshold. */
  trendelenburg_gait_pattern: boolean;
  /** Reason the hold ended: "completed" (full 30s) | "foot_touch" | "spike". */
  termination: "completed" | "foot_touch" | "spike";
  /** Time-series of pelvic tilt + trunk lean over the hold (10 Hz). */
  samples: TrendelenburgFrameSample[];
  /** Per-frame keypoints over the hold (PDF Section 2 (a) compliance). */
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  /** JPEG data-URL of the peak-drop frame (skeleton-overlaid). */
  peak_screenshot_data_url: string | null;
}

export interface TrendelenburgFullResult {
  left:  TrendelenburgSideResult | null;
  right: TrendelenburgSideResult | null;
}

// ─── Aggregator: turn a recorded sample stream into a side result ─
export function summarizeSide(
  side: Side,
  startedAtMs: number,
  endedAtMs: number,
  termination: TrendelenburgSideResult["termination"],
  samples: TrendelenburgFrameSample[],
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>,
  peakScreenshotDataUrl: string | null,
): TrendelenburgSideResult {
  const holdSec = Math.max(0, (endedAtMs - startedAtMs) / 1000);

  // Drops with "drop on lifted side = positive" reorientation.
  const drops: number[] = [];
  const stableDrops: number[] = [];
  let maxLean = 0;

  for (const s of samples) {
    if (s.pelvic_tilt_deg !== null) {
      const d = dropForStance(s.pelvic_tilt_deg, side);
      drops.push(d);
      if (s.t_ms / 1000 >= STABLE_PORTION_START_SEC) {
        stableDrops.push(d);
      }
    }
    if (s.trunk_lean_deg !== null) {
      const lean = leanTowardStance(s.trunk_lean_deg, side);
      if (lean > maxLean) maxLean = lean;
    }
  }

  const maxDrop = drops.length ? Math.max(0, ...drops) : 0;
  const meanDrop = stableDrops.length
    ? stableDrops.reduce((a, b) => a + b, 0) / stableDrops.length
    : 0;

  return {
    side_tested: side,
    hold_seconds: holdSec,
    max_drop_deg: maxDrop,
    mean_drop_deg: meanDrop,
    max_compensatory_lean_deg: maxLean,
    classification: classifyMaxDrop(maxDrop),
    short_hold: holdSec < SHORT_HOLD_THRESHOLD_SEC,
    trendelenburg_gait_pattern: maxLean > COMPENSATORY_TRUNK_LEAN_DEG,
    termination,
    samples,
    keypoints,
    peak_screenshot_data_url: peakScreenshotDataUrl,
  };
}

// ─── Plain-language interpretation paragraph ─────────────────────
export function buildInterpretation(result: TrendelenburgFullResult): string {
  const parts: string[] = [];

  for (const side of ["left", "right"] as const) {
    const r = side === "left" ? result.left : result.right;
    if (!r) continue;
    const sideLabel = side === "left"
      ? "Left-leg stance test"
      : "Right-leg stance test";

    if (r.classification === "negative") {
      parts.push(
        `${sideLabel}: negative — max pelvic drop ${r.max_drop_deg.toFixed(1)}° ` +
        `(< ${NEGATIVE_DROP_MAX_DEG}°) over a ${r.hold_seconds.toFixed(0)}s hold ` +
        `indicates well-functioning hip abductors on the stance side.`,
      );
    } else if (r.classification === "compensated") {
      parts.push(
        `${sideLabel}: compensated / mild abductor weakness — max pelvic drop ` +
        `${r.max_drop_deg.toFixed(1)}° (${NEGATIVE_DROP_MAX_DEG}–${COMPENSATED_DROP_MAX_DEG}°) ` +
        `suggests the gluteus medius is contributing but not fully stabilising the pelvis.`,
      );
    } else {
      parts.push(
        `${sideLabel}: positive Trendelenburg — max pelvic drop ` +
        `${r.max_drop_deg.toFixed(1)}° (> ${COMPENSATED_DROP_MAX_DEG}°) ` +
        `indicates significant gluteus medius weakness on the stance side.`,
      );
    }

    if (r.short_hold) {
      parts.push(
        `${sideLabel}: hold ended early at ${r.hold_seconds.toFixed(1)}s ` +
        `(< ${SHORT_HOLD_THRESHOLD_SEC}s) — additional balance / strength concern.`,
      );
    }
    if (r.trendelenburg_gait_pattern) {
      parts.push(
        `${sideLabel}: trunk leaned ${r.max_compensatory_lean_deg.toFixed(1)}° ` +
        `toward the stance side (> ${COMPENSATORY_TRUNK_LEAN_DEG}°) — ` +
        `compensatory Trendelenburg gait pattern observed.`,
      );
    }
  }

  if (parts.length === 0) {
    return "No completed stance recordings to interpret.";
  }
  return parts.join(" ");
}

export const CLASSIFICATION_LABEL: Record<Classification, string> = {
  negative:    "Negative",
  compensated: "Compensated",
  positive:    "Positive",
};

export const CLASSIFICATION_TONE: Record<Classification, string> = {
  negative:    "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  compensated: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  positive:    "bg-red-500/10 text-red-700 dark:text-red-400",
};
