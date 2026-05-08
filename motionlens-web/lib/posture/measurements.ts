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

// ─── Tilt-angle helpers ─────────────────────────────────────────
//
// These all return angles in (-90°, 90°] — the range humans expect
// for a "tilt" reading. atan2 alone returns (-180°, 180°], which
// flips to ±180° when the vector traverses the keypoints in the
// "wrong" direction; that's how a perfectly level body part used to
// read as -179°. By taking the absolute X (for horizontal-ish lines)
// or absolute Y (for vertical-ish lines) before atan2, we always
// land in the well-behaved tilt range.

/** Tilt angle (degrees) of the LINE through `a` and `b` from
 *  horizontal, signed in (-90°, 90°]. Positive = `b` is LOWER in the
 *  image than `a`. (For frontal tilts: pass (left, right) so positive
 *  means the right keypoint is lower.) */
function lineTiltFromHorizontal(a: { x: number; y: number }, b: { x: number; y: number }): number {
  const dx = Math.abs(b.x - a.x);  // always ≥ 0 — line direction agnostic
  const dy = b.y - a.y;            // signed (image y points down)
  if (dx < 1e-6) return dy > 0 ? 90 : dy < 0 ? -90 : 0;
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

/** Tilt angle (degrees) of the LINE through `top` and `bottom` from
 *  vertical, signed in (-90°, 90°]. Positive = `top` is to the RIGHT
 *  of `bottom` in the image. */
function lineTiltFromVertical(top: { x: number; y: number }, bottom: { x: number; y: number }): number {
  const dx = top.x - bottom.x;          // signed
  const dyAbs = Math.abs(top.y - bottom.y); // always ≥ 0
  if (dyAbs < 1e-6) return dx > 0 ? 90 : dx < 0 ? -90 : 0;
  return (Math.atan2(dx, dyAbs) * 180) / Math.PI;
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
  // Spec (Appendix B): trunk lean (frontal) — degrees from vertical
  // of the hip-midpoint → shoulder-midpoint line, positive = lean to
  // patient's RIGHT (camera's left in mirror frontal view).
  frontalTrunkLean: number | null;
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
    frontalTrunkLean: null,
  };

  const lEar = keypoints[LM.LEFT_EAR];
  const rEar = keypoints[LM.RIGHT_EAR];
  if (visible(lEar) && visible(rEar)) {
    // Sign convention: positive = patient's right ear is LOWER.
    // lineTiltFromHorizontal(a, b) returns positive when b is lower
    // than a, so we pass (left, right) to get "right-side-down" sign.
    result.headTilt = lineTiltFromHorizontal(lEar, rEar);
  }

  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  if (visible(lSh) && visible(rSh)) {
    // Same convention — positive = right shoulder is LOWER.
    result.shoulderTilt = lineTiltFromHorizontal(lSh, rSh);
  }

  const lHip = keypoints[LM.LEFT_HIP];
  const rHip = keypoints[LM.RIGHT_HIP];
  if (visible(lHip) && visible(rHip)) {
    // Spec convention (Appendix B): positive = left-side-down.
    // Pass (right, left) so the helper's "second arg lower → positive"
    // maps to "left side lower → positive".
    result.hipTilt = lineTiltFromHorizontal(rHip, lHip);
  }

  // Knee alignment (Q-angle proxy) — interior angle of the
  // hip-knee-ankle joint at the knee. 180° = perfectly straight
  // leg, smaller = bent or laterally deviated (valgus / varus).
  // The interior angle requires both vectors EMANATING from the
  // knee — earlier code used (knee→hip) and (knee→ankle) traversed
  // in the same direction (both downward), giving 0° for straight
  // legs. The fix is to flip the thigh vector so it points up
  // (knee→hip), then a straight leg gives 180° (antiparallel) and
  // dev = 180 - interior reads ~0° for healthy alignment.
  const lKnee = keypoints[LM.LEFT_KNEE];
  const lAnk = keypoints[LM.LEFT_ANKLE];
  if (visible(lHip) && visible(lKnee) && visible(lAnk)) {
    const upX = lHip.x - lKnee.x;       // knee → hip (going up)
    const upY = lHip.y - lKnee.y;
    const downX = lAnk.x - lKnee.x;     // knee → ankle (going down)
    const downY = lAnk.y - lKnee.y;
    const dot = upX * downX + upY * downY;
    const mag = Math.hypot(upX, upY) * Math.hypot(downX, downY);
    if (mag > 0) {
      const interior =
        (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
      result.leftKneeAlignment = interior;
    }
  }
  const rKnee = keypoints[LM.RIGHT_KNEE];
  const rAnk = keypoints[LM.RIGHT_ANKLE];
  if (visible(rHip) && visible(rKnee) && visible(rAnk)) {
    const upX = rHip.x - rKnee.x;
    const upY = rHip.y - rKnee.y;
    const downX = rAnk.x - rKnee.x;
    const downY = rAnk.y - rKnee.y;
    const dot = upX * downX + upY * downY;
    const mag = Math.hypot(upX, upY) * Math.hypot(downX, downY);
    if (mag > 0) {
      const interior =
        (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
      result.rightKneeAlignment = interior;
    }
  }

  // Spec convention (Appendix B): trunk lean (frontal) =
  // angle of (hip-midpoint to shoulder-midpoint) line vs vertical
  // in frontal view. Positive = lean to patient's RIGHT.
  // (Patient's right = camera's LEFT in mirror frontal view, so
  // shoulder-mid sitting LEFT of hip-mid in image = lean to right.)
  if (visible(lSh) && visible(rSh) && visible(lHip) && visible(rHip)) {
    const hipMid = {
      x: (lHip.x + rHip.x) / 2,
      y: (lHip.y + rHip.y) / 2,
    };
    const shoulderMid = {
      x: (lSh.x + rSh.x) / 2,
      y: (lSh.y + rSh.y) / 2,
    };
    // lineTiltFromVertical(top, bottom) returns positive when top is
    // to the RIGHT of bottom in image. Negate to flip to patient's
    // right (which is image left).
    result.frontalTrunkLean = -lineTiltFromVertical(shoulderMid, hipMid);
  }

  return result;
}

// ─── Side-view measurements ──────────────────────────────────────
//
// Patient stands sideways to the camera. We use whichever side has
// the more confident keypoints; in practice MoveNet detects both, but
// the camera-facing side is more reliable. We expose the better-side
// values plus a "pickedSide" indicator so the report can label them.
// Per-side metric block. Horizontal offsets are percentage of body
// height (positive = forward of plumb line through ankle). Trunk
// lean is in degrees from vertical with the sign convention from
// Fix 2 (positive = anatomical forward, anchored via pickedSide).
export interface SideMetrics {
  forwardHeadPct: number | null;
  shoulderShiftPct: number | null;
  hipShiftPct: number | null;
  kneeShiftPct: number | null;
  trunkLeanDeg: number | null;
}

// Spec convention (Appendix B): "All angles report L and R
// separately; never average across sides." Both sides are
// computed independently when landmarks permit. `pickedSide`
// is preserved for sign anchoring on the bilateral trunk-lean
// midpoint and for any UI that needs to highlight the more
// confidently detected side.
export interface SideMeasurements {
  pickedSide: "left" | "right" | null;
  left: SideMetrics | null;
  right: SideMetrics | null;
}

const SIDE_INDICES = {
  left: {
    ear: LM.LEFT_EAR, shoulder: LM.LEFT_SHOULDER,
    hip: LM.LEFT_HIP, knee: LM.LEFT_KNEE, ankle: LM.LEFT_ANKLE,
  },
  right: {
    ear: LM.RIGHT_EAR, shoulder: LM.RIGHT_SHOULDER,
    hip: LM.RIGHT_HIP, knee: LM.RIGHT_KNEE, ankle: LM.RIGHT_ANKLE,
  },
} as const;

interface SideKeypointIndices {
  ear: number;
  shoulder: number;
  hip: number;
  knee: number;
  ankle: number;
}

function computeOneSide(
  keypoints: Keypoint[],
  idx: SideKeypointIndices,
  bodyH: number | null,
  trunkLean: number | null,
): SideMetrics | null {
  const ear   = keypoints[idx.ear];
  const sh    = keypoints[idx.shoulder];
  const hip   = keypoints[idx.hip];
  const knee  = keypoints[idx.knee];
  const ankle = keypoints[idx.ankle];
  if (![ear, sh, hip, knee, ankle].every(visible)) return null;

  const out: SideMetrics = {
    forwardHeadPct: null,
    shoulderShiftPct: null,
    hipShiftPct: null,
    kneeShiftPct: null,
    trunkLeanDeg: trunkLean,
  };
  if (bodyH && bodyH > 0) {
    out.forwardHeadPct =   ((ear.x   - ankle.x) / bodyH) * 100;
    out.shoulderShiftPct = ((sh.x    - ankle.x) / bodyH) * 100;
    out.hipShiftPct =      ((hip.x   - ankle.x) / bodyH) * 100;
    out.kneeShiftPct =     ((knee.x  - ankle.x) / bodyH) * 100;
  }
  return out;
}

export function computeSideMeasurements(
  keypoints: Keypoint[],
): SideMeasurements {
  // pickedSide kept for backwards-compatibility / display: the side
  // with the highest min-score across its 5 keypoints.
  let pickedSide: "left" | "right" | null = null;
  let bestMinScore = 0;
  for (const name of ["left", "right"] as const) {
    const idx = SIDE_INDICES[name];
    const pts = [
      keypoints[idx.ear],
      keypoints[idx.shoulder],
      keypoints[idx.hip],
      keypoints[idx.knee],
      keypoints[idx.ankle],
    ];
    if (!pts.every(visible)) continue;
    const minScore = Math.min(...pts.map((p) => p.score ?? 0));
    if (minScore > bestMinScore) {
      bestMinScore = minScore;
      pickedSide = name;
    }
  }

  // Spec convention (Appendix B): trunk lean (sagittal) =
  // angle of (hip-midpoint to shoulder-midpoint) line vs vertical.
  // Positive = anatomical forward. Sign anchored via pickedSide
  // because "forward" vs "backward" depends on which side is facing
  // the camera (left-side facing → patient-forward = +x in image;
  // right-side facing → patient-forward = -x in image).
  let trunkLean: number | null = null;
  const lHipKp = keypoints[LM.LEFT_HIP];
  const rHipKp = keypoints[LM.RIGHT_HIP];
  const lShKp  = keypoints[LM.LEFT_SHOULDER];
  const rShKp  = keypoints[LM.RIGHT_SHOULDER];
  if (
    visible(lHipKp) && visible(rHipKp) &&
    visible(lShKp)  && visible(rShKp)
  ) {
    const hipMid = {
      x: (lHipKp.x + rHipKp.x) / 2,
      y: (lHipKp.y + rHipKp.y) / 2,
    };
    const shoulderMid = {
      x: (lShKp.x + rShKp.x) / 2,
      y: (lShKp.y + rShKp.y) / 2,
    };
    // lineTiltFromVertical returns positive when shoulder is to the
    // right of hip in image. Apply pickedSide-anchored sign so
    // positive consistently maps to "anatomical forward".
    const rawTilt = lineTiltFromVertical(shoulderMid, hipMid);
    if (pickedSide === "left") {
      // Left side of patient faces camera → patient's anterior is
      // toward image +x, so positive rawTilt = forward.
      trunkLean = rawTilt;
    } else if (pickedSide === "right") {
      // Right side of patient faces camera → anterior is toward
      // image -x, so flip the sign.
      trunkLean = -rawTilt;
    }
  }

  const bodyH = bodyHeightPx(keypoints);
  return {
    pickedSide,
    left:  computeOneSide(keypoints, SIDE_INDICES.left,  bodyH, trunkLean),
    right: computeOneSide(keypoints, SIDE_INDICES.right, bodyH, trunkLean),
  };
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
  // Spec (Appendix B): frontal trunk lean — positive = lean to
  // patient's right. Reuse existing TILT_OK / TILT_MILD thresholds
  // (1.5° / 3.5°) — spec is silent on frontal-trunk-lean cutoffs.
  if (m.frontalTrunkLean !== null) {
    const v = m.frontalTrunkLean;
    const abs = Math.abs(v);
    let severity: PostureFinding["severity"];
    let detail: string;
    const dir = v >= 0 ? "right" : "left";
    if (abs < TILT_OK) {
      severity = "ok";
      detail = "Trunk is upright in the frontal plane (positive = lean to patient's right).";
    } else if (abs < TILT_MILD) {
      severity = "mild";
      detail = `Mild trunk lean toward the patient's ${dir} (positive = lean to patient's right).`;
    } else {
      severity = "notable";
      detail = `Notable trunk lean toward the patient's ${dir} (positive = lean to patient's right).`;
    }
    out.push({
      label: "Frontal trunk lean",
      value: `${v.toFixed(1)}°`,
      severity,
      detail,
    });
  }
  return out;
}

// Spec convention (Appendix B): "All angles report L and R
// separately; never average across sides." This builder emits one
// findings row per metric per visible side, with the view labelled.
// Trunk lean is bilateral (same value on both sides) so it's
// emitted once, not duplicated per side.
export function buildSideFindings(m: SideMeasurements): PostureFinding[] {
  // Backwards compatibility: saved reports written before Fix 4
  // stored the flat shape `{ forwardHeadPct, shoulderShiftPct, ... }`.
  // If we see that, fall through to the legacy renderer rather than
  // dropping the data on the floor.
  const legacy = m as unknown as {
    forwardHeadPct?: number | null;
    shoulderShiftPct?: number | null;
    hipShiftPct?: number | null;
    kneeShiftPct?: number | null;
    trunkLeanDeg?: number | null;
  };
  const isLegacyShape =
    m.left === undefined && m.right === undefined &&
    (legacy.forwardHeadPct !== undefined ||
      legacy.shoulderShiftPct !== undefined ||
      legacy.hipShiftPct !== undefined ||
      legacy.kneeShiftPct !== undefined ||
      legacy.trunkLeanDeg !== undefined);
  if (isLegacyShape) {
    return buildSideFindingsLegacy(legacy);
  }

  const out: PostureFinding[] = [];
  for (const view of ["left", "right"] as const) {
    const block = view === "left" ? m.left : m.right;
    if (!block) continue;
    const tag = `(${view} side view)`;
    if (block.forwardHeadPct !== null)
      out.push(labelTag(gradeShift(block.forwardHeadPct, "Head"), tag));
    if (block.shoulderShiftPct !== null)
      out.push(labelTag(gradeShift(block.shoulderShiftPct, "Shoulders"), tag));
    if (block.hipShiftPct !== null)
      out.push(labelTag(gradeShift(block.hipShiftPct, "Hips"), tag));
    if (block.kneeShiftPct !== null)
      out.push(labelTag(gradeShift(block.kneeShiftPct, "Knees"), tag));
  }
  // Trunk lean is bilateral — emit once.
  const trunk =
    (m.left?.trunkLeanDeg ?? m.right?.trunkLeanDeg) ?? null;
  if (trunk !== null) {
    const abs = Math.abs(trunk);
    out.push({
      label: "Trunk lean",
      value: `${trunk.toFixed(1)}°`,
      severity: abs < 2 ? "ok" : abs < 5 ? "mild" : "notable",
      detail:
        abs < 2
          ? "Trunk is upright."
          : `Trunk leans ${trunk > 0 ? "forward" : "backward"} ${abs.toFixed(1)}° from vertical.`,
    });
  }
  return out;
}

function labelTag(f: PostureFinding, tag: string): PostureFinding {
  return { ...f, label: `${f.label} ${tag}` };
}

// Legacy renderer for posture reports saved before the L/R split.
// Reads the old flat shape and produces the original single-side
// findings.
function buildSideFindingsLegacy(m: {
  forwardHeadPct?: number | null;
  shoulderShiftPct?: number | null;
  hipShiftPct?: number | null;
  kneeShiftPct?: number | null;
  trunkLeanDeg?: number | null;
}): PostureFinding[] {
  const out: PostureFinding[] = [];
  if (m.forwardHeadPct != null)
    out.push(gradeShift(m.forwardHeadPct, "Head"));
  if (m.shoulderShiftPct != null)
    out.push(gradeShift(m.shoulderShiftPct, "Shoulders"));
  if (m.hipShiftPct != null)
    out.push(gradeShift(m.hipShiftPct, "Hips"));
  if (m.kneeShiftPct != null)
    out.push(gradeShift(m.kneeShiftPct, "Knees"));
  if (m.trunkLeanDeg != null) {
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
