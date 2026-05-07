// Single-Leg Stance test — math, stance detection, sway path,
// 95% confidence ellipse, termination triggers, classification,
// asymmetry, plain-language interpretation.
//
// Spec: MotionLens Test Battery v1.0, Test C5 (pp. 16-17).
// Pose model: MoveNet 17-keypoint (LM indices in @/lib/pose/landmarks).
//
// Bilateral test — L and R reported separately (PDF Appendix B:
// "never average across sides"). Each side is captured under one
// or two conditions: eyes-open (mandatory) and eyes-closed
// (optional). The full session can include up to four trials.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";
import { getSingleLegStanceNorm } from "@/lib/orthopedic/normsDatabase";

const VIS_THRESHOLD = 0.3;

export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// PDF Test C5 spec values.
export const MAX_EYES_OPEN_SEC = 60;
export const MAX_EYES_CLOSED_SEC = 30;

// L–R asymmetry flag (PDF: > 30% asymmetry = targeted intervention).
export const ASYMMETRY_FLAG_PCT = 30;

// ─── Termination-trigger thresholds ─────────────────────────────
// All ratios scale by body height (shoulder-mid → ankle-mid in
// pixels) so they adapt to camera distance.

// Lifted ankle is "back on the ground" when its Y is within this
// fraction of body-height of the stance ankle's Y.
const FOOT_TOUCHDOWN_RATIO = 0.05;

// Lift-onset detection — lifted ankle must rise this much above
// the stance ankle to count as a single-leg stance start.
const LIFT_ONSET_RATIO = 0.10;

// Arm-grab: shoulder→wrist vector that abducts > 45° from vertical
// → patient reached for support → terminate. Matches PDF guidance.
const ARM_GRAB_DEG = 45;

// Hop / stance-foot reposition: stance ankle Y moves > this fraction
// of body height within HOP_WINDOW_MS → terminate.
const HOP_DISPLACEMENT_RATIO = 0.06;
const HOP_WINDOW_MS = 500;

// Pre-onset grace — give the patient a few seconds to lift the
// leg after Start. If no lift detected within this window, end
// the trial as "no_lift_detected".
export const ONSET_TIMEOUT_SEC = 8;

export type Side = "left" | "right";
export type Condition = "eyes_open" | "eyes_closed";
export type Termination =
  | "max_time"          // trial ran the full max-time
  | "foot_touchdown"    // lifted foot returned to ground
  | "arm_grab"          // wrist abducted past 45° from vertical
  | "hop"               // stance foot repositioned
  | "no_lift_detected"  // patient never lifted within onset window
  | "stopped";          // operator clicked stop

export type Classification = "pass" | "fail";

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

// ─── Per-frame computations ─────────────────────────────────────

export function computeHipMidpoint(keypoints: Keypoint[]): { x: number; y: number } | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!visible(lHip) || !visible(rHip)) return null;
  return { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
}

export function computeShoulderMidpoint(keypoints: Keypoint[]): { x: number; y: number } | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (!visible(lSh) || !visible(rSh)) return null;
  return { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 };
}

export function computeBodyHeightPx(keypoints: Keypoint[]): number | null {
  const sh = computeShoulderMidpoint(keypoints);
  const lA = keypoints[LM.LEFT_ANKLE];
  const rA = keypoints[LM.RIGHT_ANKLE];
  if (!sh || (!visible(lA) && !visible(rA))) return null;
  const ankleY =
    visible(lA) && visible(rA) ? (lA.y + rA.y) / 2
    : visible(lA) ? lA.y
    : rA.y;
  return Math.abs(ankleY - sh.y);
}

// Trunk lean (frontal): hip-midpoint → shoulder-midpoint vs
// vertical, positive = lean to patient's RIGHT (mirror frontal).
export function computeTrunkLean(keypoints: Keypoint[]): number | null {
  const hipMid = computeHipMidpoint(keypoints);
  const shMid  = computeShoulderMidpoint(keypoints);
  if (!hipMid || !shMid) return null;
  const vx = shMid.x - hipMid.x;
  const vy = shMid.y - hipMid.y;
  const mag = Math.abs((Math.atan2(vx, vy) * 180) / Math.PI);
  return (vx < 0 ? 1 : -1) * mag;
}

// Auto-detect single-leg stance: lifted ankle is well above the
// stance ankle. Returns the STANCE side (the leg still on the
// ground). Null when no clear lift is seen.
export function detectStanceSide(keypoints: Keypoint[]): Side | null {
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (!visible(la) || !visible(ra)) return null;
  const bodyH = computeBodyHeightPx(keypoints);
  if (!bodyH) return null;
  const lift = Math.abs(la.y - ra.y);
  if (lift < bodyH * LIFT_ONSET_RATIO) return null;
  // Lower image-y = lifted; the OTHER side is the stance side.
  return la.y < ra.y ? "right" : "left";
}

// Foot-touchdown: lifted ankle has dropped close to the stance
// ankle's Y. Returns true only when stance is `expected` AND the
// lifted ankle is now within touchdown range.
export function isFootTouchdown(
  keypoints: Keypoint[],
  expected: Side,
): boolean {
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (!visible(la) || !visible(ra)) return false;
  const bodyH = computeBodyHeightPx(keypoints);
  if (!bodyH) return false;
  const stance = expected === "left" ? la : ra;
  const lifted = expected === "left" ? ra : la;
  return Math.abs(lifted.y - stance.y) < bodyH * FOOT_TOUCHDOWN_RATIO;
}

// Arm-grab: angle of shoulder→wrist vs vertical (image-y-down,
// vertical = (0, +1)). > ARM_GRAB_DEG on either side = grab event.
export function isArmGrab(keypoints: Keypoint[]): boolean {
  const sides: Array<[number, number]> = [
    [LM.LEFT_SHOULDER,  LM.LEFT_WRIST],
    [LM.RIGHT_SHOULDER, LM.RIGHT_WRIST],
  ];
  for (const [shIdx, wrIdx] of sides) {
    const sh = keypoints[shIdx];
    const wr = keypoints[wrIdx];
    if (!visible(sh) || !visible(wr)) continue;
    const vx = wr.x - sh.x;
    const vy = wr.y - sh.y;
    if (Math.hypot(vx, vy) === 0) continue;
    // Angle from straight-down (0, +1).
    const angle = Math.abs((Math.atan2(vx, vy) * 180) / Math.PI);
    if (angle > ARM_GRAB_DEG) return true;
  }
  return false;
}

// ─── Sway analytics ─────────────────────────────────────────────

export function swayPathLength(positions: ReadonlyArray<{ x: number; y: number }>): number {
  let total = 0;
  for (let i = 1; i < positions.length; i++) {
    total += Math.hypot(positions[i].x - positions[i - 1].x, positions[i].y - positions[i - 1].y);
  }
  return total;
}

// 95% confidence ellipse area of a 2D point cloud, computed via
// the eigenvalues of the covariance matrix per the PDF spec
// formula: π × λ1 × λ2 × 5.991. Reported in pixels² (relative —
// not calibrated to cm; suitable for trend tracking only).
export function swayEllipse95Area(positions: ReadonlyArray<{ x: number; y: number }>): number {
  if (positions.length < 3) return 0;
  const n = positions.length;
  let mx = 0, my = 0;
  for (const p of positions) { mx += p.x; my += p.y; }
  mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const p of positions) {
    const dx = p.x - mx;
    const dy = p.y - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  sxx /= n; syy /= n; sxy /= n;
  // Eigenvalues of 2×2 covariance matrix.
  const trace = sxx + syy;
  const det   = sxx * syy - sxy * sxy;
  const disc  = Math.sqrt(Math.max(0, trace * trace / 4 - det));
  const l1 = trace / 2 + disc;
  const l2 = Math.max(0, trace / 2 - disc);
  return Math.PI * l1 * l2 * 5.991;
}

// ─── Per-trial result + aggregator ──────────────────────────────

export interface FrameSample {
  t_ms: number;
  hip_x: number | null;
  hip_y: number | null;
  trunk_lean_deg: number | null;
}

export interface TrialResult {
  side: Side;
  condition: Condition;
  hold_seconds: number;
  hold_capped_at: number;            // max-time ceiling for this condition
  termination: Termination;
  norm_threshold_sec: number;
  norm_band_label: string;
  norm_comparable: boolean;
  classification: Classification;
  sway_path_px: number;
  sway_95_ellipse_px2: number;
  mean_trunk_lean_deg: number;
  max_trunk_lean_deg: number;
  /** All hip-midpoint positions during the hold — feeds the sway
   *  plot in the report. Position units are video pixels. */
  hip_path: Array<{ x: number; y: number }>;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  screenshot_data_url: string | null;
}

export function summarizeTrial(args: {
  side: Side;
  condition: Condition;
  startedAtMs: number;
  endedAtMs: number;
  termination: Termination;
  hipPath: Array<{ x: number; y: number }>;
  trunkLeans: number[];
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  screenshotDataUrl: string | null;
  patientAge: number | null;
}): TrialResult {
  const {
    side, condition, startedAtMs, endedAtMs, termination,
    hipPath, trunkLeans, samples, keypoints, screenshotDataUrl,
    patientAge,
  } = args;

  const holdSec = Math.max(0, (endedAtMs - startedAtMs) / 1000);
  const cap = condition === "eyes_open" ? MAX_EYES_OPEN_SEC : MAX_EYES_CLOSED_SEC;
  const norm = getSingleLegStanceNorm(patientAge, condition === "eyes_closed");
  const passed = holdSec >= norm.passThresholdSec;

  const meanLean =
    trunkLeans.length === 0 ? 0 : trunkLeans.reduce((a, b) => a + b, 0) / trunkLeans.length;
  const maxLean =
    trunkLeans.length === 0 ? 0 : Math.max(...trunkLeans.map((v) => Math.abs(v)));

  return {
    side,
    condition,
    hold_seconds: holdSec,
    hold_capped_at: cap,
    termination,
    norm_threshold_sec: norm.passThresholdSec,
    norm_band_label: norm.bandLabel,
    norm_comparable: norm.comparable,
    classification: passed ? "pass" : "fail",
    sway_path_px: swayPathLength(hipPath),
    sway_95_ellipse_px2: swayEllipse95Area(hipPath),
    mean_trunk_lean_deg: meanLean,
    max_trunk_lean_deg: maxLean,
    hip_path: hipPath,
    samples,
    keypoints,
    screenshot_data_url: screenshotDataUrl,
  };
}

// ─── Session-level aggregator ───────────────────────────────────

export interface SessionResult {
  /** Patient age snapshot for norm lookup at trial-end. */
  patient_age: number | null;
  trials: {
    left_open?:    TrialResult;
    right_open?:   TrialResult;
    left_closed?:  TrialResult;
    right_closed?: TrialResult;
  };
  /** Eyes-open L–R asymmetry as a percentage. Undefined when one
   *  side is missing. */
  eyes_open_asymmetry_pct: number | null;
  /** Eyes-closed L–R asymmetry as a percentage. */
  eyes_closed_asymmetry_pct: number | null;
  asymmetry_flag: boolean; // either condition exceeds threshold
}

export function asymmetryPct(left: TrialResult | undefined, right: TrialResult | undefined): number | null {
  if (!left || !right) return null;
  const max = Math.max(left.hold_seconds, right.hold_seconds);
  if (max === 0) return 0;
  return Math.abs(left.hold_seconds - right.hold_seconds) / max * 100;
}

export function buildSession(trials: SessionResult["trials"], patientAge: number | null): SessionResult {
  const eo = asymmetryPct(trials.left_open, trials.right_open);
  const ec = asymmetryPct(trials.left_closed, trials.right_closed);
  const asymFlag =
    (eo !== null && eo > ASYMMETRY_FLAG_PCT) ||
    (ec !== null && ec > ASYMMETRY_FLAG_PCT);
  return {
    patient_age: patientAge,
    trials,
    eyes_open_asymmetry_pct: eo,
    eyes_closed_asymmetry_pct: ec,
    asymmetry_flag: asymFlag,
  };
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(session: SessionResult): string {
  const t = session.trials;
  const lines: string[] = [];

  function describeTrial(r: TrialResult | undefined, label: string) {
    if (!r) return;
    const cond = r.condition === "eyes_open" ? "eyes-open" : "eyes-closed";
    if (r.classification === "pass") {
      lines.push(
        `${label} (${cond}): held ${r.hold_seconds.toFixed(1)} s — at or above the ` +
        `${r.norm_band_label} threshold of ${r.norm_threshold_sec} s.`,
      );
    } else {
      lines.push(
        `${label} (${cond}): held ${r.hold_seconds.toFixed(1)} s — below the ` +
        `${r.norm_band_label} threshold of ${r.norm_threshold_sec} s. Positive screen for balance impairment.`,
      );
    }
    if (r.termination !== "max_time" && r.termination !== "stopped") {
      const reason =
        r.termination === "foot_touchdown" ? "lifted foot returned to the ground"
        : r.termination === "arm_grab"     ? "patient reached for support (arm grab)"
        : r.termination === "hop"          ? "stance foot repositioned (hop)"
        :                                    "no clear leg lift was detected within the onset window";
      lines.push(`${label}: terminated when ${reason}.`);
    }
  }

  describeTrial(t.left_open,    "Left-leg stance");
  describeTrial(t.right_open,   "Right-leg stance");
  describeTrial(t.left_closed,  "Left-leg stance");
  describeTrial(t.right_closed, "Right-leg stance");

  if (session.eyes_open_asymmetry_pct !== null && session.eyes_open_asymmetry_pct > ASYMMETRY_FLAG_PCT) {
    const dom = (t.left_open?.hold_seconds ?? 0) > (t.right_open?.hold_seconds ?? 0) ? "right" : "left";
    lines.push(
      `Eyes-open L–R asymmetry ${session.eyes_open_asymmetry_pct.toFixed(0)}% (> ${ASYMMETRY_FLAG_PCT}%) — ` +
      `targeted intervention indicated on the ${dom} side.`,
    );
  }
  if (session.eyes_closed_asymmetry_pct !== null && session.eyes_closed_asymmetry_pct > ASYMMETRY_FLAG_PCT) {
    const dom = (t.left_closed?.hold_seconds ?? 0) > (t.right_closed?.hold_seconds ?? 0) ? "right" : "left";
    lines.push(
      `Eyes-closed L–R asymmetry ${session.eyes_closed_asymmetry_pct.toFixed(0)}% (> ${ASYMMETRY_FLAG_PCT}%) — ` +
      `consider targeted intervention on the ${dom} side.`,
    );
  }

  // Norm-comparable flag (any trial flagged it).
  const anyNonComparable = Object.values(t).some((r): r is TrialResult => !!r && !r.norm_comparable);
  if (anyNonComparable) {
    lines.push(
      "Norm comparison limited — patient age was missing, so the strictest threshold band was applied.",
    );
  }

  if (lines.length === 0) {
    return "No completed trials to interpret — re-run the test.";
  }
  return lines.join(" ");
}

// ─── Display helpers ────────────────────────────────────────────

export const CLASSIFICATION_LABEL: Record<Classification, string> = {
  pass: "Pass",
  fail: "Fail",
};

export const CLASSIFICATION_TONE: Record<Classification, string> = {
  pass: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  fail: "bg-red-500/10 text-red-700 dark:text-red-400",
};

// Live-loop helper to detect a hop using a rolling window of stance
// ankle Y values. Returns true when displacement exceeds threshold
// within the window. The caller maintains the window (an array of
// {t_ms, y} samples for the past HOP_WINDOW_MS). Pure function so
// it stays unit-testable.
export function isHopInWindow(
  samples: Array<{ t_ms: number; y: number }>,
  bodyH: number,
): boolean {
  if (samples.length < 2 || bodyH <= 0) return false;
  let minY = samples[0].y;
  let maxY = samples[0].y;
  for (const s of samples) {
    if (s.y < minY) minY = s.y;
    if (s.y > maxY) maxY = s.y;
  }
  return (maxY - minY) > bodyH * HOP_DISPLACEMENT_RATIO;
}

export const HOP_WINDOW_DURATION_MS = HOP_WINDOW_MS;
