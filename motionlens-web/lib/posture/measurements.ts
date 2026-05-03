// Postural deviation measurements derived from MoveNet 17-keypoint output.
// Front-view checks bilateral symmetry (head/shoulder/hip tilt + knee
// alignment). Side-view checks sagittal-plane shifts (forward head,
// rounded shoulders, anterior pelvic shift, knee position).
//
// All values are reported in **degrees** (for tilts/angles) or
// **percentage of body height** (for offsets) so they're comparable
// across patients without per-image scale calibration.

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

const VIS = 0.2;

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS;
}

// Signed angle of a vector (vx, vy) from the horizontal axis (1, 0).
// Positive = clockwise tilt in image coords (image y axis points down).
function angleFromHorizontal(vx: number, vy: number): number {
  return (Math.atan2(vy, vx) * 180) / Math.PI;
}

function angleFromVertical(vx: number, vy: number): number {
  // Image y-axis points down, so vertical-down = (0, +1).
  return (Math.atan2(vx, vy) * 180) / Math.PI;
}

// Reference body-height proxy: shoulder-midpoint to ankle-midpoint
// distance in pixels. Used to normalize horizontal offsets so the
// "10% forward head" reads consistently across image sizes.
function bodyHeightPx(keypoints: Keypoint[]): number | null {
  const ls = keypoints[LM.LEFT_SHOULDER];
  const rs = keypoints[LM.RIGHT_SHOULDER];
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (![ls, rs, la, ra].every(visible)) return null;
  const shMidY = (ls.y + rs.y) / 2;
  const anMidY = (la.y + ra.y) / 2;
  return Math.abs(anMidY - shMidY);
}

// ─── Front-view measurements ─────────────────────────────────────
export interface FrontMeasurements {
  headTilt: number | null;        // degrees, positive = right-side down
  shoulderTilt: number | null;
  hipTilt: number | null;
  leftKneeAlignment: number | null;   // Q-angle approximation, degrees
  rightKneeAlignment: number | null;
}

export function computeFrontMeasurements(
  keypoints: Keypoint[],
): FrontMeasurements {
  const result: FrontMeasurements = {
    headTilt: null,
    shoulderTilt: null,
    hipTilt: null,
    leftKneeAlignment: null,
    rightKneeAlignment: null,
  };

  const lEar = keypoints[LM.LEFT_EAR];
  const rEar = keypoints[LM.RIGHT_EAR];
  if (visible(lEar) && visible(rEar)) {
    result.headTilt = angleFromHorizontal(rEar.x - lEar.x, rEar.y - lEar.y);
  }

  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (visible(lSh) && visible(rSh)) {
    result.shoulderTilt = angleFromHorizontal(rSh.x - lSh.x, rSh.y - lSh.y);
  }

  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (visible(lHip) && visible(rHip)) {
    result.hipTilt = angleFromHorizontal(rHip.x - lHip.x, rHip.y - lHip.y);
  }

  // Knee alignment (Q-angle proxy) — angle of hip-knee-ankle from a
  // straight vertical line. Large value = valgus/varus deviation.
  const lKnee = keypoints[LM.LEFT_KNEE];
  const lAnk = keypoints[LM.LEFT_ANKLE];
  if (visible(lHip) && visible(lKnee) && visible(lAnk)) {
    // Thigh vector (hip→knee) and shin vector (knee→ankle).
    const thighX = lKnee.x - lHip.x;
    const thighY = lKnee.y - lHip.y;
    const shinX = lAnk.x - lKnee.x;
    const shinY = lAnk.y - lKnee.y;
    // Interior angle at the knee — 180° = perfectly straight leg.
    const dot = thighX * shinX + thighY * shinY;
    const mag = Math.hypot(thighX, thighY) * Math.hypot(shinX, shinY);
    if (mag > 0) {
      const interior =
        (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
      result.leftKneeAlignment = interior;
    }
  }
  const rKnee = keypoints[LM.RIGHT_KNEE];
  const rAnk = keypoints[LM.RIGHT_ANKLE];
  if (visible(rHip) && visible(rKnee) && visible(rAnk)) {
    const thighX = rKnee.x - rHip.x;
    const thighY = rKnee.y - rHip.y;
    const shinX = rAnk.x - rKnee.x;
    const shinY = rAnk.y - rKnee.y;
    const dot = thighX * shinX + thighY * shinY;
    const mag = Math.hypot(thighX, thighY) * Math.hypot(shinX, shinY);
    if (mag > 0) {
      const interior =
        (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
      result.rightKneeAlignment = interior;
    }
  }

  return result;
}

// ─── Side-view measurements ──────────────────────────────────────
//
// Patient stands sideways to the camera. We use whichever side has
// the more confident keypoints; in practice MoveNet detects both, but
// the camera-facing side is more reliable. We expose the better-side
// values plus a "pickedSide" indicator so the report can label them.
export interface SideMeasurements {
  pickedSide: "left" | "right" | null;
  // Horizontal offsets, expressed as percentage of body height
  // (positive = forward of plumb line through ankle).
  forwardHeadPct: number | null;
  shoulderShiftPct: number | null;
  hipShiftPct: number | null;
  kneeShiftPct: number | null;
  // Trunk lean (degrees from vertical, positive = forward).
  trunkLeanDeg: number | null;
}

export function computeSideMeasurements(
  keypoints: Keypoint[],
): SideMeasurements {
  const result: SideMeasurements = {
    pickedSide: null,
    forwardHeadPct: null,
    shoulderShiftPct: null,
    hipShiftPct: null,
    kneeShiftPct: null,
    trunkLeanDeg: null,
  };

  const sides: Array<{ name: "left" | "right"; idx: { ear: number; shoulder: number; hip: number; knee: number; ankle: number } }> = [
    {
      name: "left",
      idx: {
        ear: LM.LEFT_EAR, shoulder: LM.LEFT_SHOULDER,
        hip: LM.LEFT_HIP, knee: LM.LEFT_KNEE, ankle: LM.LEFT_ANKLE,
      },
    },
    {
      name: "right",
      idx: {
        ear: LM.RIGHT_EAR, shoulder: LM.RIGHT_SHOULDER,
        hip: LM.RIGHT_HIP, knee: LM.RIGHT_KNEE, ankle: LM.RIGHT_ANKLE,
      },
    },
  ];

  // Pick the side with the highest min-score across the 5 needed points.
  let bestSide: typeof sides[number] | null = null;
  let bestMinScore = 0;
  for (const s of sides) {
    const pts = [
      keypoints[s.idx.ear],
      keypoints[s.idx.shoulder],
      keypoints[s.idx.hip],
      keypoints[s.idx.knee],
      keypoints[s.idx.ankle],
    ];
    if (!pts.every(visible)) continue;
    const minScore = Math.min(...pts.map((p) => p.score ?? 0));
    if (minScore > bestMinScore) {
      bestMinScore = minScore;
      bestSide = s;
    }
  }
  if (!bestSide) return result;

  result.pickedSide = bestSide.name;
  const ear   = keypoints[bestSide.idx.ear];
  const sh    = keypoints[bestSide.idx.shoulder];
  const hip   = keypoints[bestSide.idx.hip];
  const knee  = keypoints[bestSide.idx.knee];
  const ankle = keypoints[bestSide.idx.ankle];

  const bodyH = bodyHeightPx(keypoints);
  if (bodyH && bodyH > 0) {
    // Horizontal offsets relative to ankle (the "plumb line").
    result.forwardHeadPct =     ((ear.x - ankle.x) / bodyH) * 100;
    result.shoulderShiftPct =   ((sh.x  - ankle.x) / bodyH) * 100;
    result.hipShiftPct =        ((hip.x - ankle.x) / bodyH) * 100;
    result.kneeShiftPct =       ((knee.x - ankle.x) / bodyH) * 100;
  }

  // Trunk lean: shoulder-to-hip vector vs vertical.
  const trunkX = sh.x - hip.x;
  const trunkY = sh.y - hip.y;
  result.trunkLeanDeg = angleFromVertical(trunkX, -Math.abs(trunkY));

  return result;
}

// ─── Interpretation helpers ──────────────────────────────────────
export interface PostureFinding {
  label: string;
  value: string;          // already-formatted with unit
  severity: "ok" | "mild" | "notable";
  detail: string;
}

const TILT_OK = 1.5;       // degrees considered normal
const TILT_MILD = 3.5;
const SHIFT_OK = 3;        // % body height
const SHIFT_MILD = 7;

function gradeTilt(value: number, label: string, direction: [string, string]): PostureFinding {
  const abs = Math.abs(value);
  const dir = value >= 0 ? direction[1] : direction[0];
  let severity: PostureFinding["severity"];
  let detail: string;
  if (abs < TILT_OK) {
    severity = "ok";
    detail = `${label} is well aligned.`;
  } else if (abs < TILT_MILD) {
    severity = "mild";
    detail = `Mild ${label.toLowerCase()} tilt toward the ${dir}.`;
  } else {
    severity = "notable";
    detail = `Notable ${label.toLowerCase()} tilt toward the ${dir} — worth noting.`;
  }
  return { label, value: `${value.toFixed(1)}°`, severity, detail };
}

function gradeShift(value: number, label: string): PostureFinding {
  const abs = Math.abs(value);
  const dir = value >= 0 ? "forward" : "backward";
  let severity: PostureFinding["severity"];
  let detail: string;
  if (abs < SHIFT_OK) {
    severity = "ok";
    detail = `${label} is aligned with the plumb line.`;
  } else if (abs < SHIFT_MILD) {
    severity = "mild";
    detail = `Mild ${dir} shift of ${label.toLowerCase()}.`;
  } else {
    severity = "notable";
    detail = `Notable ${dir} shift of ${label.toLowerCase()}.`;
  }
  return { label, value: `${value.toFixed(1)}%`, severity, detail };
}

export function buildFrontFindings(m: FrontMeasurements): PostureFinding[] {
  const out: PostureFinding[] = [];
  if (m.headTilt !== null)
    out.push(gradeTilt(m.headTilt, "Head tilt", ["left", "right"]));
  if (m.shoulderTilt !== null)
    out.push(gradeTilt(m.shoulderTilt, "Shoulder tilt", ["left", "right"]));
  if (m.hipTilt !== null)
    out.push(gradeTilt(m.hipTilt, "Hip tilt", ["left", "right"]));
  if (m.leftKneeAlignment !== null) {
    const dev = 180 - m.leftKneeAlignment;
    out.push({
      label: "Left knee alignment",
      value: `${dev.toFixed(1)}°`,
      severity: Math.abs(dev) < 5 ? "ok" : Math.abs(dev) < 10 ? "mild" : "notable",
      detail:
        Math.abs(dev) < 5
          ? "Left knee is well aligned in the frontal plane."
          : `Left knee deviates ${dev.toFixed(1)}° from a straight hip-knee-ankle line.`,
    });
  }
  if (m.rightKneeAlignment !== null) {
    const dev = 180 - m.rightKneeAlignment;
    out.push({
      label: "Right knee alignment",
      value: `${dev.toFixed(1)}°`,
      severity: Math.abs(dev) < 5 ? "ok" : Math.abs(dev) < 10 ? "mild" : "notable",
      detail:
        Math.abs(dev) < 5
          ? "Right knee is well aligned in the frontal plane."
          : `Right knee deviates ${dev.toFixed(1)}° from a straight hip-knee-ankle line.`,
    });
  }
  return out;
}

export function buildSideFindings(m: SideMeasurements): PostureFinding[] {
  const out: PostureFinding[] = [];
  if (m.forwardHeadPct !== null)
    out.push(gradeShift(m.forwardHeadPct, "Head"));
  if (m.shoulderShiftPct !== null)
    out.push(gradeShift(m.shoulderShiftPct, "Shoulders"));
  if (m.hipShiftPct !== null)
    out.push(gradeShift(m.hipShiftPct, "Hips"));
  if (m.kneeShiftPct !== null)
    out.push(gradeShift(m.kneeShiftPct, "Knees"));
  if (m.trunkLeanDeg !== null) {
    const abs = Math.abs(m.trunkLeanDeg);
    out.push({
      label: "Trunk lean",
      value: `${m.trunkLeanDeg.toFixed(1)}°`,
      severity: abs < 2 ? "ok" : abs < 5 ? "mild" : "notable",
      detail:
        abs < 2
          ? "Trunk is upright."
          : `Trunk leans ${m.trunkLeanDeg > 0 ? "forward" : "backward"} ${abs.toFixed(1)}° from vertical.`,
    });
  }
  return out;
}
