// 4-Stage Balance Test (Test C4) — math, stage-position validators,
// termination triggers, sway analytics, classification, plain-language
// interpretation.
//
// Spec: MotionLens Test Battery v1.0, Test C4 (pp. 15-16).
// Pose model: MoveNet 17-keypoint (LM indices in @/lib/pose/landmarks).
//
// Sequential-progression test — patient holds 4 progressively harder
// static stances (side-by-side → semi-tandem → tandem → single-leg)
// for 10 s each. Test STOPS at first failure (PDF mandate: "progress
// only if previous stage held for full 10 s. No retry, no skip.").

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";
import { getFourStageBalanceNorm } from "@/lib/orthopedic/normsDatabase";

const VIS_THRESHOLD = 0.3;

export const SAMPLE_HZ = 10;
export const SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;

// PDF Test C4 spec value — every stage capped at 10 s.
export const STAGE_HOLD_SEC = 10;

// Pre-stage grace — give the patient up to this long to get into the
// required stance before we time out the stage as "position_lost".
export const POSITION_TIMEOUT_SEC = 12;

// Position must be valid for this long continuously before the 10 s
// hold timer starts. Avoids false-starts when the patient is still
// adjusting their feet.
export const POSITION_LOCK_MS = 600;

// Once the hold has started, position-validation tolerates this much
// drift before counting as "position lost". Avoids termination on
// brief MoveNet jitter / single-frame drops.
export const POSITION_DRIFT_GRACE_MS = 800;

// Minimum ankle X separation to trust the landmarks aren't merged.
// Stages 1–2 expect feet close together; if MoveNet collapses ankles
// (separation < 8 px), we pause the timer and coach the patient. Spec
// (PDF C4) suggests BlazePose foot-index fallback — out of scope for
// MoveNet, so we coach instead.
export const MIN_ANKLE_SEPARATION_PX = 8;

// ─── Stage-geometry validators (PDF spec values, raw px) ─────────
//
// All thresholds are in raw image pixels. They assume a typical
// frontal MotionLens setup (camera at hip height, ~2 m from patient,
// ≥480p video) where the patient occupies most of the frame. If
// future testing shows variance across devices, these can be made
// body-height-relative — for now, we keep the spec-literal values.

// Stage 1 — Side-by-Side. Both ankles roughly level, natural-stance
// horizontal separation.
const S1_Y_MAX = 20;
const S1_X_MIN = 20;
const S1_X_MAX = 80;

// Stage 2 — Semi-Tandem. One ankle slightly forward, narrow X.
const S2_Y_MIN = 15;
const S2_Y_MAX = 60;
const S2_X_MAX = 30;

// Stage 3 — Tandem. Heel-to-toe in line: very small X, large Y.
const S3_X_MAX = 15;
const S3_Y_MIN = 40;

// Stage 4 — Single-Leg. One ankle clearly above (smaller image-Y
// than) the other.
const S4_LIFT_MIN_PX = 100;

// Arm-grab termination — wrist abducts past 45° from vertical
// (matches C5 / Trendelenburg).
const ARM_GRAB_DEG = 45;

// ─── Public types ───────────────────────────────────────────────

export type StageIndex = 1 | 2 | 3 | 4;

export type StageOutcome = "pass" | "fail" | "not_attempted";

export type FailureMode =
  | "foot_touchdown"   // stage 4 only — lifted foot returned to ground
  | "arm_grab"         // patient reached for support
  | "position_lost"    // current stage's geometry sustained drift
  | "stopped";         // operator clicked stop early

export type SessionClassification =
  | "high_fall_risk"
  | "elevated_fall_risk"
  | "normal";

export interface FrameSample {
  t_ms: number;
  hip_x: number | null;
  hip_y: number | null;
  ankle_l_x: number | null;
  ankle_l_y: number | null;
  ankle_r_x: number | null;
  ankle_r_y: number | null;
}

export interface StageResult {
  stage: StageIndex;
  outcome: StageOutcome;
  hold_seconds: number;            // capped at STAGE_HOLD_SEC
  failure_mode: FailureMode | null;
  sway_path_px: number;
  sway_95_ellipse_px2: number;
  hip_path: Array<{ x: number; y: number }>;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  screenshot_data_url: string | null;
  /** When pass: 10. When fail: time held before termination. */
  duration_seconds: number;
}

export interface SessionResult {
  patient_age: number | null;
  stages: {
    1?: StageResult;
    2?: StageResult;
    3?: StageResult;
    4?: StageResult;
  };
  /** Highest stage the patient COMPLETED (held the full 10 s). 0 = none. */
  final_stage_completed: 0 | 1 | 2 | 3 | 4;
  /** First stage to FAIL, or null when all passed. */
  first_failed_stage: StageIndex | null;
  classification: SessionClassification;
  norm_band_label: string;
  norm_comparable: boolean;
}

// ─── Per-frame helpers (re-used pattern from C5) ────────────────

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

export function computeHipMidpoint(
  keypoints: Keypoint[],
): { x: number; y: number } | null {
  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (!visible(lHip) || !visible(rHip)) return null;
  return { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
}

export interface AnkleReading {
  lx: number; ly: number;
  rx: number; ry: number;
  dx_abs: number;   // |L.x - R.x|
  dy_abs: number;   // |L.y - R.y|
}

export function readAnkles(keypoints: Keypoint[]): AnkleReading | null {
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (!visible(la) || !visible(ra)) return null;
  return {
    lx: la.x, ly: la.y,
    rx: ra.x, ry: ra.y,
    dx_abs: Math.abs(la.x - ra.x),
    dy_abs: Math.abs(la.y - ra.y),
  };
}

// ─── Stage-position validators ──────────────────────────────────

export function isStage1Position(a: AnkleReading): boolean {
  return a.dy_abs < S1_Y_MAX && a.dx_abs > S1_X_MIN && a.dx_abs < S1_X_MAX;
}

export function isStage2Position(a: AnkleReading): boolean {
  return a.dy_abs > S2_Y_MIN && a.dy_abs < S2_Y_MAX && a.dx_abs < S2_X_MAX;
}

export function isStage3Position(a: AnkleReading): boolean {
  return a.dx_abs < S3_X_MAX && a.dy_abs > S3_Y_MIN;
}

// Stage 4 returns the stance side (the foot still on the ground)
// when one ankle is clearly lifted above the other. Null = neither
// foot is meaningfully lifted yet.
export function detectStage4Stance(a: AnkleReading): "left" | "right" | null {
  const lift = a.dy_abs;
  if (lift < S4_LIFT_MIN_PX) return null;
  // Lower image-y (smaller value) = lifted foot. Stance = the OTHER foot.
  return a.ly < a.ry ? "right" : "left";
}

export function isStagePosition(
  stage: StageIndex,
  a: AnkleReading,
): boolean {
  if (a.dx_abs < MIN_ANKLE_SEPARATION_PX && (stage === 1 || stage === 2)) {
    // Stages 1 and 2 expect feet close together — but if MoveNet has
    // collapsed both ankles on top of each other, we cannot trust the
    // reading. Treat as "position not valid yet" and let the caller
    // coach the patient to adjust.
    return false;
  }
  switch (stage) {
    case 1: return isStage1Position(a);
    case 2: return isStage2Position(a);
    case 3: return isStage3Position(a);
    case 4: return detectStage4Stance(a) !== null;
  }
}

export function ankleMergeWarning(a: AnkleReading): boolean {
  return a.dx_abs < MIN_ANKLE_SEPARATION_PX;
}

// ─── Termination triggers ───────────────────────────────────────

// Stage 4 only — lifted foot returns near the stance foot.
const FOOT_TOUCHDOWN_PX = 30;

export function isStage4FootTouchdown(
  keypoints: Keypoint[],
  stanceSide: "left" | "right",
): boolean {
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (!visible(la) || !visible(ra)) return false;
  const stance = stanceSide === "left" ? la : ra;
  const lifted = stanceSide === "left" ? ra : la;
  return Math.abs(lifted.y - stance.y) < FOOT_TOUCHDOWN_PX;
}

// Arm-grab — same logic as C5 / Trendelenburg. Wrist abducts past
// 45° from straight-down on either side → patient reached for
// support → terminate.
export function isArmGrab(keypoints: Keypoint[]): boolean {
  const sides: Array<[number, number]> = [
    [LM.LEFT_SHOULDER, LM.LEFT_WRIST],
    [LM.RIGHT_SHOULDER, LM.RIGHT_WRIST],
  ];
  for (const [shIdx, wrIdx] of sides) {
    const sh = keypoints[shIdx];
    const wr = keypoints[wrIdx];
    if (!visible(sh) || !visible(wr)) continue;
    const vx = wr.x - sh.x;
    const vy = wr.y - sh.y;
    if (Math.hypot(vx, vy) === 0) continue;
    const angle = Math.abs((Math.atan2(vx, vy) * 180) / Math.PI);
    if (angle > ARM_GRAB_DEG) return true;
  }
  return false;
}

// ─── Sway analytics (re-used from C5) ───────────────────────────

export function swayPathLength(
  positions: ReadonlyArray<{ x: number; y: number }>,
): number {
  let total = 0;
  for (let i = 1; i < positions.length; i++) {
    total += Math.hypot(
      positions[i].x - positions[i - 1].x,
      positions[i].y - positions[i - 1].y,
    );
  }
  return total;
}

// 95 % confidence ellipse area: π × λ1 × λ2 × 5.991. Pixels² —
// relative units, suitable for trend tracking, not calibrated.
export function swayEllipse95Area(
  positions: ReadonlyArray<{ x: number; y: number }>,
): number {
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
  const trace = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
  const l1 = trace / 2 + disc;
  const l2 = Math.max(0, trace / 2 - disc);
  return Math.PI * l1 * l2 * 5.991;
}

// ─── Per-stage summarizer ───────────────────────────────────────

export function summarizeStage(args: {
  stage: StageIndex;
  outcome: StageOutcome;
  failureMode: FailureMode | null;
  startedAtMs: number;
  endedAtMs: number;
  hipPath: Array<{ x: number; y: number }>;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  screenshotDataUrl: string | null;
}): StageResult {
  const {
    stage, outcome, failureMode, startedAtMs, endedAtMs,
    hipPath, samples, keypoints, screenshotDataUrl,
  } = args;
  const duration = Math.max(0, (endedAtMs - startedAtMs) / 1000);
  const holdSec = Math.min(duration, STAGE_HOLD_SEC);
  return {
    stage,
    outcome,
    hold_seconds: holdSec,
    duration_seconds: duration,
    failure_mode: outcome === "pass" ? null : failureMode,
    sway_path_px: swayPathLength(hipPath),
    sway_95_ellipse_px2: swayEllipse95Area(hipPath),
    hip_path: hipPath,
    samples,
    keypoints,
    screenshot_data_url: screenshotDataUrl,
  };
}

// ─── Session-level aggregator ───────────────────────────────────

export function buildSession(
  stages: SessionResult["stages"],
  patientAge: number | null,
): SessionResult {
  // Highest stage that PASSED (in sequence). Sequential rule means
  // we can just walk 1..4 and stop at the first non-pass.
  let lastPassed: 0 | 1 | 2 | 3 | 4 = 0;
  let firstFailed: StageIndex | null = null;
  for (const s of [1, 2, 3, 4] as const) {
    const r = stages[s];
    if (r && r.outcome === "pass") {
      lastPassed = s;
      continue;
    }
    if (r && r.outcome === "fail") {
      firstFailed = s;
    }
    break;
  }

  const stage4HoldSec = stages[4]?.hold_seconds ?? null;
  const norm = getFourStageBalanceNorm(patientAge);
  const classification = classifySession(lastPassed, stage4HoldSec, patientAge);

  return {
    patient_age: patientAge,
    stages,
    final_stage_completed: lastPassed,
    first_failed_stage: firstFailed,
    classification,
    norm_band_label: norm.bandLabel,
    norm_comparable: norm.comparable,
  };
}

// PDF C4 cutoffs:
//   - Failed before holding tandem (stage 3) for 10 s = high fall risk.
//   - Age > 60 AND single-leg (stage 4) < 5 s = high fall risk.
//   - Held tandem but failed stage 4 before 5 s (or any age) =
//     elevated fall risk.
//   - Held all four full 10 s = normal.
function classifySession(
  lastPassed: 0 | 1 | 2 | 3 | 4,
  stage4HoldSec: number | null,
  age: number | null,
): SessionClassification {
  if (lastPassed < 3) return "high_fall_risk";
  if (lastPassed === 3) {
    // Tandem held but stage 4 either not attempted or failed before
    // the age-based cutoff.
    if (age !== null && age > 60) {
      // PDF: < 5 s for age > 60 = high fall risk.
      if (stage4HoldSec !== null && stage4HoldSec < 5) return "high_fall_risk";
    }
    return "elevated_fall_risk";
  }
  // lastPassed === 4: completed every stage.
  return "normal";
}

// ─── Plain-language interpretation ──────────────────────────────

export function buildInterpretation(session: SessionResult): string {
  const lines: string[] = [];
  const final = session.final_stage_completed;

  // Per-stage status.
  for (const s of [1, 2, 3, 4] as const) {
    const r = session.stages[s];
    const label = STAGE_LABEL[s];
    if (!r) {
      lines.push(`${label}: not attempted (test stopped after stage ${final}).`);
      continue;
    }
    if (r.outcome === "pass") {
      lines.push(`${label}: held full ${STAGE_HOLD_SEC} s.`);
    } else {
      const reason = describeFailure(r.failure_mode);
      lines.push(
        `${label}: failed after ${r.duration_seconds.toFixed(1)} s — ${reason}.`,
      );
    }
  }

  // Classification commentary.
  if (session.classification === "normal") {
    lines.push(
      `Completed all 4 stages — within the normal range for this CDC fall-risk screen.`,
    );
  } else if (session.classification === "elevated_fall_risk") {
    lines.push(
      `Held tandem (stage 3) but did not complete single-leg stance — elevated fall risk per CDC criteria.`,
    );
  } else {
    if (final < 3) {
      lines.push(
        `Did not hold tandem stance (stage 3) for the full ${STAGE_HOLD_SEC} s — significantly elevated fall risk per CDC criteria.`,
      );
    } else if (final === 3 && session.patient_age !== null && session.patient_age > 60) {
      const sec = session.stages[4]?.hold_seconds ?? 0;
      lines.push(
        `Single-leg stance held only ${sec.toFixed(1)} s — below the 5 s threshold for age > 60. High fall risk.`,
      );
    }
  }

  if (!session.norm_comparable) {
    lines.push(
      "Norm comparison limited — patient age was missing, so the age-based stage-4 sub-criterion has been applied conservatively.",
    );
  }

  return lines.join(" ");
}

function describeFailure(mode: FailureMode | null): string {
  switch (mode) {
    case "foot_touchdown":
      return "lifted foot returned to the ground";
    case "arm_grab":
      return "patient reached for support (arm grab)";
    case "position_lost":
      return "stance position drifted out of tolerance";
    case "stopped":
      return "operator stopped the trial";
    default:
      return "trial ended";
  }
}

// ─── Display helpers ────────────────────────────────────────────

export const STAGE_LABEL: Record<StageIndex, string> = {
  1: "Stage 1 · Side-by-side",
  2: "Stage 2 · Semi-tandem",
  3: "Stage 3 · Tandem",
  4: "Stage 4 · Single-leg",
};

export const STAGE_INSTRUCTION: Record<StageIndex, string> = {
  1: "Stand with both feet next to each other, toes aligned.",
  2: "Place the heel of one foot beside the big toe of the other (semi-tandem).",
  3: "Place one foot directly in front of the other, heel-to-toe (tandem).",
  4: "Lift one leg so the knee is at roughly 90° hip flexion.",
};

export const CLASSIFICATION_LABEL: Record<SessionClassification, string> = {
  normal: "Normal",
  elevated_fall_risk: "Elevated fall risk",
  high_fall_risk: "High fall risk",
};

export const CLASSIFICATION_TONE: Record<SessionClassification, string> = {
  normal: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  elevated_fall_risk: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  high_fall_risk: "bg-red-500/10 text-red-700 dark:text-red-400",
};
