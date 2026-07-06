// Rehab compensation checks — additive helpers for the
// compensation_flags payload field. Reuses the biomech Compensation
// shape by convention (never imports/edits biomech files beyond
// existing pure-math exports).
//
// Each helper follows the same pattern:
//   const tracker = createXxxTracker(opts);
//   tracker.update(kp);         // per frame
//   const flag = tracker.finalize();  // Compensation | null
//
// A page calls .update() inside handleFrame and .finalize() inside
// buildRehabPayload. Trackers are stateful (they count frames or
// track sustained conditions) and pure — no React, no DOM.

import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

const VIS = 0.35;
const OK_VIS = (p) => p && (p.score ?? p.visibility ?? 0) >= VIS;

/** Signed vertical distance (y-down canvas). Negative = a is above b. */
function verticalDelta(a, b) {
  return b.y - a.y;
}

// ─── Rounding (thoracic flexion proxy) ──────────────────────────
// Detected when the shoulder-mid drifts FORWARD of the hip-mid axis
// beyond a threshold — i.e. the head projects out over the hips in
// a hunched fashion (a proxy for thoracic flexion because BlazePose
// has no mid-spine landmarks).
//
// Uses the sagittal-plane projection (raw x delta in the source
// video). Callers pass keypoints already in norm space; this
// helper does not know about mirroring — the caller's cameraSide
// determines whether "forward" means +x or -x. Default assumes the
// patient faces the camera in profile with head-forward = +x.

export function createRoundingTracker({
  thresholdFraction = 0.15,   // shoulder-hip x delta > 15 % of trunk length
  sustainedFrames = 5,        // must persist for N frames to flag
  cameraSideFactor = 1,       // -1 if the camera faces the opposite side
} = {}) {
  let frames = 0;
  let peakDelta = 0;
  return {
    update(kp) {
      if (!kp) return;
      const lSh = kp[LM.LEFT_SHOULDER];
      const rSh = kp[LM.RIGHT_SHOULDER];
      const lHip = kp[LM.LEFT_HIP];
      const rHip = kp[LM.RIGHT_HIP];
      if (!OK_VIS(lSh) || !OK_VIS(rSh) || !OK_VIS(lHip) || !OK_VIS(rHip)) return;
      const shMidX = (lSh.x + rSh.x) / 2;
      const shMidY = (lSh.y + rSh.y) / 2;
      const hipMidX = (lHip.x + rHip.x) / 2;
      const hipMidY = (lHip.y + rHip.y) / 2;
      const trunkLen = Math.hypot(hipMidX - shMidX, hipMidY - shMidY);
      if (trunkLen < 1) return;
      const forwardDelta = cameraSideFactor * (shMidX - hipMidX);
      const fraction = forwardDelta / trunkLen;
      if (fraction > thresholdFraction) {
        frames += 1;
        if (fraction > peakDelta) peakDelta = fraction;
      } else {
        frames = 0;
      }
    },
    finalize() {
      if (frames < sustainedFrames) return null;
      const severity = peakDelta > thresholdFraction * 2 ? "high"
        : peakDelta > thresholdFraction * 1.4 ? "medium"
        : "low";
      return {
        type: "rounding",
        label: "Trunk rounding",
        severity,
        flagged: true,
        details: `Shoulders drifted forward ${(peakDelta * 100).toFixed(0)}% of trunk length.`,
      };
    },
  };
}

// ─── Hand-use (wrist near/below hip for support) ─────────────────
// Fires when EITHER wrist lands at or below the same-side hip's y
// position for enough sustained frames — the classic "hand on
// thigh / hand on chair" support pattern during single-leg squats,
// wall sits, or fatigue offloading in bridges.

export function createHandUseTracker({
  sustainedFrames = 8,       // ~0.5-0.7 s at typical detection rate
  yToleranceFraction = 0.0,  // extend: hip.y + yTolerance × trunkLen
} = {}) {
  let frames = 0;
  return {
    update(kp) {
      if (!kp) return;
      const lW = kp[LM.LEFT_WRIST];
      const rW = kp[LM.RIGHT_WRIST];
      const lHip = kp[LM.LEFT_HIP];
      const rHip = kp[LM.RIGHT_HIP];
      if (!OK_VIS(lHip) || !OK_VIS(rHip)) return;

      const lSh = kp[LM.LEFT_SHOULDER];
      const rSh = kp[LM.RIGHT_SHOULDER];
      let trunkLen = 100;
      if (OK_VIS(lSh) && OK_VIS(rSh)) {
        const shMidY = (lSh.y + rSh.y) / 2;
        const hipMidY = (lHip.y + rHip.y) / 2;
        trunkLen = Math.max(1, Math.abs(hipMidY - shMidY));
      }
      const tol = yToleranceFraction * trunkLen;
      const leftSupport = OK_VIS(lW) && verticalDelta(lHip, lW) < tol;
      const rightSupport = OK_VIS(rW) && verticalDelta(rHip, rW) < tol;

      if (leftSupport || rightSupport) {
        frames += 1;
      } else {
        frames = Math.max(0, frames - 1);
      }
    },
    finalize() {
      if (frames < sustainedFrames) return null;
      return {
        type: "hand_use",
        label: "Used hands for support",
        severity: frames > 30 ? "high" : "medium",
        flagged: true,
        details: `Wrist detected at/below hip level for ${frames} frames.`,
      };
    },
  };
}

// ─── Step-out (foot moved outside its baseline stance) ─────────
// Simple ankle-lift-or-shift detector. Baseline auto-captured
// during a "calibrating" phase (first CALIB_FRAMES seen), then any
// frame where either ankle's y rises above baseline − yRiseThresh
// or the ankle's x shifts by xShiftThresh × trunkLen counts a step.

export function createStepOutTracker({
  yRiseThreshFraction = 0.03,   // 3 % of frame height rise
  xShiftThreshFraction = 0.05,  // 5 % lateral shift
  calibrationFrames = 10,
} = {}) {
  let calibFrames = 0;
  let baseline = null; // { lAy, rAy, lAx, rAx }
  let stepFrames = 0;
  let stepDetected = false;
  return {
    update(kp, videoWidth, videoHeight) {
      if (!kp) return;
      const lA = kp[LM.LEFT_ANKLE];
      const rA = kp[LM.RIGHT_ANKLE];
      if (!OK_VIS(lA) || !OK_VIS(rA)) return;
      if (calibFrames < calibrationFrames) {
        if (!baseline) baseline = { lAy: lA.y, rAy: rA.y, lAx: lA.x, rAx: rA.x };
        else {
          baseline.lAy = (baseline.lAy + lA.y) / 2;
          baseline.rAy = (baseline.rAy + rA.y) / 2;
          baseline.lAx = (baseline.lAx + lA.x) / 2;
          baseline.rAx = (baseline.rAx + rA.x) / 2;
        }
        calibFrames += 1;
        return;
      }
      if (!baseline) return;
      const yRise = yRiseThreshFraction * (videoHeight || 720);
      const xShift = xShiftThreshFraction * (videoWidth || 1280);
      const leftMoved =
        (baseline.lAy - lA.y) > yRise ||
        Math.abs(lA.x - baseline.lAx) > xShift;
      const rightMoved =
        (baseline.rAy - rA.y) > yRise ||
        Math.abs(rA.x - baseline.rAx) > xShift;
      if (leftMoved || rightMoved) {
        stepFrames += 1;
        if (stepFrames >= 3) stepDetected = true;
      } else {
        stepFrames = Math.max(0, stepFrames - 1);
      }
    },
    isStepping() {
      return stepDetected;
    },
    finalize() {
      if (!stepDetected) return null;
      return {
        type: "step_out",
        label: "Step-out detected",
        severity: "medium",
        flagged: true,
        details: "Ankle displacement exceeded stance-baseline threshold.",
      };
    },
  };
}
