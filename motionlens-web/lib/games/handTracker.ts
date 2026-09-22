// Single-hand tracking state shared between the React pose loop and the
// Phaser scene.
//
// Deliberately a MUTABLE object rather than React state: the pose loop
// runs at camera rate and Phaser renders on its own rAF, so routing a
// position through setState would re-render the tree ~30x/s for nothing.
// React reads it only for low-frequency UI (the readiness ticks).

import { LM_LIVE } from "@/lib/pose/landmarks-live";

export type Hand = "left" | "right";

/**
 * Which BlazePose landmark drives the cursor, per hand choice.
 *
 * These indices are ANATOMICAL — landmark 16 is the patient's own right
 * wrist regardless of how the picture is drawn. The live shells mirror
 * the frame for display (`x = 1 - p.x / videoWidth`, see
 * RehabCameraShell.tsx:358), which is a selfie view: the patient's right
 * hand ends up on the RIGHT of the screen, exactly as it would in a
 * mirror. Mirroring changes only where the point is drawn, never which
 * index it came from — so "Right" maps to RIGHT_WRIST and nothing else
 * needs to flip.
 */
export const WRIST_INDEX: Record<Hand, number> = {
  left: LM_LIVE.LEFT_WRIST, // 15
  right: LM_LIVE.RIGHT_WRIST, // 16
};

/** Same-side elbow, used to project the wrist forward to the palm when
 *  the hand landmarks themselves are unavailable. */
export const ELBOW_INDEX: Record<Hand, number> = {
  left: LM_LIVE.LEFT_ELBOW, // 13
  right: LM_LIVE.RIGHT_ELBOW, // 14
};

/** Same-side hand points. BlazePose emits these as part of its 33-point
 *  pose model, so no extra detector is involved. Their mean with the
 *  wrist lands close to the centre of the palm. */
export const PINKY_INDEX: Record<Hand, number> = {
  left: LM_LIVE.LEFT_PINKY, // 17
  right: LM_LIVE.RIGHT_PINKY, // 18
};
export const FINGER_INDEX: Record<Hand, number> = {
  left: LM_LIVE.LEFT_INDEX, // 19
  right: LM_LIVE.RIGHT_INDEX, // 20
};

/** Where the drawn palm came from this frame. */
export type PalmSource = "hand" | "elbow" | "wrist";

/** Every landmark index for one side of the body. */
export interface SideMap {
  wrist: number;
  elbow: number;
  pinky: number;
  finger: number;
  shoulder: number;
}

const SIDE: Record<Hand, SideMap> = {
  left: {
    wrist: LM_LIVE.LEFT_WRIST,
    elbow: LM_LIVE.LEFT_ELBOW,
    pinky: LM_LIVE.LEFT_PINKY,
    finger: LM_LIVE.LEFT_INDEX,
    shoulder: LM_LIVE.LEFT_SHOULDER,
  },
  right: {
    wrist: LM_LIVE.RIGHT_WRIST,
    elbow: LM_LIVE.RIGHT_ELBOW,
    pinky: LM_LIVE.RIGHT_PINKY,
    finger: LM_LIVE.RIGHT_INDEX,
    shoulder: LM_LIVE.RIGHT_SHOULDER,
  },
};

/**
 * Landmark set for the hand the patient chose.
 *
 * `swapped` exists because BlazePose's left/right labels are only
 * anatomical if the detector is fed an UN-mirrored frame. Reading the
 * code, this one is: `pose.send({ image: video })` passes the raw
 * element (CSS transforms do not touch its pixels) and no `selfieMode`
 * is set, so the default of false applies — the mirror at
 * `nx = 1 - p.x / videoW` happens afterwards, for display only.
 *
 * CONFIRMED ON CAMERA: the labels ARE anatomical — choosing Right and
 * using the right hand tracks correctly. So `swapped` is false in
 * normal play and exists only as a debug override (?handswap=1) for
 * re-testing this on a different browser or camera. Flipping it here
 * flips every consumer at once — cursor, palm, calibration, spawn
 * anchor, headroom — because they all read through this one function.
 */
export function sideLandmarks(hand: Hand, swapped: boolean): SideMap {
  const use: Hand = swapped ? (hand === "left" ? "right" : "left") : hand;
  return SIDE[use];
}

/** One landmark, as the debug overlay reports it. */
export interface Probe {
  x: number;
  y: number;
  vis: number;
  inFrame: boolean;
}

/**
 * How far past the wrist the palm sits, as a fraction of the forearm.
 *
 * BlazePose gives a wrist, but the ✋ cursor reads as a palm, so drawing
 * it on the wrist puts it visibly short of the hand. Extending along
 * the forearm by this much lands it near the middle of the palm.
 *
 * Computed in CANVAS pixels, not normalised space: normalised space is
 * anisotropic whenever the video is not square (dispW != dispH), so
 * extrapolating a direction there would skew it.
 */
export const PALM_REACH = 0.3;

/**
 * Headroom required above the chosen shoulder, as a multiple of arm
 * length, before the setup check passes.
 *
 * A raised arm puts the wrist almost exactly one arm length above the
 * shoulder, so 1.0 is the bare minimum for the wrist to be on screen
 * at all. The extra 0.1 covers the wrist rising a little past vertical
 * and the patient swaying, so the landmark does not clip in and out
 * during a 5 s hold.
 */
export const HEADROOM_RATIO_MIN = 1.1;

/** Below this visibility a landmark is treated as not present. Matches
 *  the value the production camera shells use for their overlay. */
export const VIS_FLOOR = 0.35;

/** Half-frame slack allowed before a landmark counts as out of frame.
 *  MediaPipe extrapolates joints it cannot see and still reports a
 *  passing visibility, so `live` alone is not enough to trust a point. */
export const FRAME_SLACK = 0.05;

export interface HandState {
  /** Normalised MIRRORED video space, 0..1 across the frame. */
  nx: number;
  ny: number;
  /** WRIST in canvas-space pixels, after object-cover compensation.
   *  Calibration and the reach box are defined in wrist terms, so this
   *  keeps its original meaning. */
  x: number;
  y: number;
  /** PALM in canvas-space pixels — the wrist projected along the
   *  forearm. This is what the cursor is drawn at and what the hit test
   *  uses, so what the patient sees is what collects the fruit.
   *  Falls back to the wrist when the elbow is not usable. */
  palmX: number;
  palmY: number;
  /**
   * How the palm was derived this frame:
   *   "hand"  — mean of wrist + index and/or pinky. What we want.
   *   "elbow" — wrist projected along the forearm. Good enough.
   *   "wrist" — nothing better available; the cursor is short of the
   *             hand and the debug overlay flags it in red.
   */
  palmSource: PalmSource;
  /** Running tally per source, so a whole round can be judged rather
   *  than a glanced-at frame. */
  palmCounts: { hand: number; elbow: number; wrist: number };
  score: number;
  /** Visibility passed the floor. */
  live: boolean;
  /** Inside the frame (plus slack). */
  inFrame: boolean;
  /** live && inFrame — the only condition under which x/y may be used. */
  usable: boolean;

  /** Setup-check readiness flags. */
  shouldersOk: boolean;
  hipsOk: boolean;
  wristOk: boolean;
  ready: boolean;
  /** Head landmark — needed to frame the top of the body. */
  noseOk: boolean;
  /** Chosen side's elbow. */
  elbowOk: boolean;

  /** Shoulder -> elbow -> wrist, summed, in canvas px. With the arm
   *  down this is the whole arm, which is also how far above the
   *  shoulder the wrist will go when the arm is raised. */
  armLenPx: number;
  /** Distance from the chosen shoulder up to the top of the VISIBLE
   *  canvas. Measured after object-cover cropping, so it is the space
   *  actually on screen, not space in the raw camera frame that has
   *  been cropped away. */
  headroomPx: number;
  /** headroomPx / armLenPx. Below 1 the raised wrist cannot be seen at
   *  all, so abduction cannot be measured. */
  headroomRatio: number;
  headroomOk: boolean;
  /** Everything Fruit Harvest needs: head, both shoulders, the chosen
   *  elbow and wrist, and room above the head. Hips and legs are NOT
   *  required — this game never looks below the waist. */
  setupOk: boolean;

  /** Body midline in mirrored normalised x, from the shoulders. Used to
   *  split the reach box into same-side and across-midline halves. */
  midX: number;
  midXValid: boolean;

  /** CHOSEN SIDE's shoulder in canvas pixels — the anchor every spawn
   *  direction is measured from, so fruit placement follows the patient
   *  rather than the frame. */
  shoulderX: number;
  shoulderY: number;
  shoulderOk: boolean;
  /** The OTHER wrist and shoulder, canvas px. Calibration needs them
   *  to tell "you are not in the pose" from "you are using the wrong
   *  hand" — which look identical from the chosen wrist alone. */
  otherX: number;
  otherY: number;
  otherUsable: boolean;
  otherShoulderX: number;
  otherShoulderY: number;
  otherShoulderOk: boolean;
  /** Body midline in canvas px, from both shoulders. The across-body
   *  hold is judged against this. */
  midShoulderX: number;
  /** Forearm length (wrist to elbow) in canvas px. The palm sits
   *  PALM_REACH of this beyond the wrist, so a target placed at the
   *  calibrated WRIST reach is reachable without full extension —
   *  spawning adds this back on. 0 when the elbow is not usable. */
  forearmPx: number;

  /** Object-cover geometry used for the last mapping. */
  cover: { dispW: number; dispH: number; offX: number; offY: number };

  /** True once any pose has arrived. */
  hasPose: boolean;
  /** Timestamp of the last usable wrist sample, ms (performance.now). */
  lastUsableMs: number;
  /** Raw readouts for BOTH wrists, so which physical hand a label
   *  follows can be settled by looking rather than reasoning. Screen
   *  coordinates, i.e. after the display mirror. */
  probeL15: Probe;
  probeR16: Probe;
  /** Landmark index the cursor is actually reading this frame. */
  usingWrist: number;
  /** Whether the side mapping is currently swapped. */
  swapped: boolean;

  /** Detector callbacks per second, smoothed. Written by the pose loop.
   *  The cursor can never be fresher than this, so it sets the floor on
   *  how much lag any amount of filtering can remove. */
  poseHz: number;
}

export function createHandState(): HandState {
  return {
    nx: 0.5,
    ny: 0.5,
    x: 0,
    y: 0,
    palmX: 0,
    palmY: 0,
    palmSource: "wrist",
    palmCounts: { hand: 0, elbow: 0, wrist: 0 },
    score: 0,
    live: false,
    inFrame: false,
    usable: false,
    shouldersOk: false,
    hipsOk: false,
    wristOk: false,
    ready: false,
    noseOk: false,
    elbowOk: false,
    armLenPx: 0,
    headroomPx: 0,
    headroomRatio: 0,
    headroomOk: false,
    setupOk: false,
    midX: 0.5,
    midXValid: false,
    shoulderX: 0,
    shoulderY: 0,
    shoulderOk: false,
    otherX: 0,
    otherY: 0,
    otherUsable: false,
    otherShoulderX: 0,
    otherShoulderY: 0,
    otherShoulderOk: false,
    midShoulderX: 0,
    forearmPx: 0,
    cover: { dispW: 0, dispH: 0, offX: 0, offY: 0 },
    hasPose: false,
    lastUsableMs: 0,
    poseHz: 0,
    probeL15: { x: 0, y: 0, vis: 0, inFrame: false },
    probeR16: { x: 0, y: 0, vis: 0, inFrame: false },
    usingWrist: LM_LIVE.RIGHT_WRIST,
    swapped: false,
  };
}

/**
 * Reproduce the browser's `object-cover` geometry, exactly as
 * RehabCameraShell does (lines 215-234): scale by the LARGER ratio so
 * the video COVERS the box, then centre the overflow. When the
 * container and the video share an aspect ratio this collapses to
 * `nx * containerW, ny * containerH`.
 */
export function coverGeometry(
  containerW: number,
  containerH: number,
  videoW: number,
  videoH: number,
) {
  const scale =
    videoW > 0 && videoH > 0
      ? Math.max(containerW / videoW, containerH / videoH)
      : 0;
  const dispW = scale > 0 ? videoW * scale : containerW;
  const dispH = scale > 0 ? videoH * scale : containerH;
  return {
    dispW,
    dispH,
    offX: (containerW - dispW) / 2,
    offY: (containerH - dispH) / 2,
  };
}

function inFrameN(nx: number, ny: number): boolean {
  return (
    nx >= -FRAME_SLACK
    && nx <= 1 + FRAME_SLACK
    && ny >= -FRAME_SLACK
    && ny <= 1 + FRAME_SLACK
  );
}

/**
 * Fill a HandState from one detector result.
 *
 * `kp` is the raw keypoint array in VIDEO PIXEL space — exactly what
 * usePoseDetectionLive returns. Mirroring matches the production shells
 * (`1 - x / videoW`) so the patient sees their hand on the side they
 * expect. The OTHER wrist is never read.
 */
export function updateHandState(
  s: HandState,
  kp: { x: number; y: number; score: number }[] | null,
  hand: Hand,
  /** True when BlazePose's left/right labels are the opposite of the
   *  patient's anatomy. See sideLandmarks(). */
  swapped: boolean,
  videoW: number,
  videoH: number,
  containerW: number,
  containerH: number,
  nowMs: number,
): void {
  if (!kp || videoW <= 0 || videoH <= 0) {
    s.live = false;
    s.inFrame = false;
    s.usable = false;
    s.shouldersOk = false;
    s.hipsOk = false;
    s.wristOk = false;
    s.ready = false;
    s.noseOk = false;
    s.elbowOk = false;
    s.headroomOk = false;
    s.setupOk = false;
    return;
  }

  s.hasPose = true;
  const cover = coverGeometry(containerW, containerH, videoW, videoH);
  s.cover = cover;

  const read = (idx: number) => {
    const p = kp[idx];
    if (!p) return null;
    const nx = 1 - p.x / videoW;
    const ny = p.y / videoH;
    const score = p.score ?? 0;
    return {
      nx,
      ny,
      score,
      ok: score >= VIS_FLOOR && inFrameN(nx, ny),
    };
  };

  let elbowRead: ReturnType<typeof read> = null;
  const side = sideLandmarks(hand, swapped);
  s.swapped = swapped;
  s.usingWrist = side.wrist;

  // Raw probes for BOTH wrists, mapped to screen space, regardless of
  // which one the game is using. This is what settles the question.
  const probe = (idx: number, out: Probe) => {
    const p = kp[idx];
    if (!p) { out.vis = 0; out.inFrame = false; return; }
    const nx = 1 - p.x / videoW;
    const ny = p.y / videoH;
    out.x = Math.round(cover.offX + nx * cover.dispW);
    out.y = Math.round(cover.offY + ny * cover.dispH);
    out.vis = p.score ?? 0;
    out.inFrame = inFrameN(nx, ny);
  };
  probe(LM_LIVE.LEFT_WRIST, s.probeL15);
  probe(LM_LIVE.RIGHT_WRIST, s.probeR16);

  const wrist = read(side.wrist);
  if (wrist) {
    s.nx = wrist.nx;
    s.ny = wrist.ny;
    s.score = wrist.score;
    s.live = wrist.score >= VIS_FLOOR;
    s.inFrame = inFrameN(wrist.nx, wrist.ny);
    s.usable = wrist.ok;
    s.x = cover.offX + wrist.nx * cover.dispW;
    s.y = cover.offY + wrist.ny * cover.dispH;
    if (wrist.ok) s.lastUsableMs = nowMs;

    elbowRead = read(side.elbow);
    const elbow = elbowRead;
    s.elbowOk = !!elbow?.ok;
    if (elbow?.ok && wrist.ok) {
      const ex = cover.offX + elbow.nx * cover.dispW;
      const ey = cover.offY + elbow.ny * cover.dispH;
      s.forearmPx = Math.hypot(s.x - ex, s.y - ey);
    }

    // ── Palm, best source first. Everything in CANVAS pixels: the
    //    normalised space is anisotropic whenever the video is not
    //    square, so averaging or extrapolating there would skew.
    const pinky = read(side.pinky);
    const finger = read(side.finger);
    const haveFinger = !!(finger?.ok || pinky?.ok);

    if (wrist.ok && haveFinger) {
      // 1. Mean of the hand points. The wrist plus the index and pinky
      //    knuckles straddle the palm, so their mean sits in it.
      let sx = s.x;
      let sy = s.y;
      let n = 1;
      for (const p of [finger, pinky]) {
        if (!p?.ok) continue;
        sx += cover.offX + p.nx * cover.dispW;
        sy += cover.offY + p.ny * cover.dispH;
        n++;
      }
      s.palmX = sx / n;
      s.palmY = sy / n;
      s.palmSource = "hand";
    } else if (wrist.ok && elbow?.ok) {
      // 2. Project the wrist along the forearm.
      const ex = cover.offX + elbow.nx * cover.dispW;
      const ey = cover.offY + elbow.ny * cover.dispH;
      s.palmX = s.x + (s.x - ex) * PALM_REACH;
      s.palmY = s.y + (s.y - ey) * PALM_REACH;
      s.palmSource = "elbow";
    } else {
      // 3. Nothing better — sit on the wrist rather than guess a
      //    direction from a landmark MediaPipe extrapolated. The cursor
      //    is visibly short of the hand here; the overlay says so.
      s.palmX = s.x;
      s.palmY = s.y;
      s.palmSource = "wrist";
    }
    if (s.usable) s.palmCounts[s.palmSource] += 1;
  } else {
    s.live = false;
    s.inFrame = false;
    s.usable = false;
  }

  const lSh = read(LM_LIVE.LEFT_SHOULDER);
  const rSh = read(LM_LIVE.RIGHT_SHOULDER);
  const lHip = read(LM_LIVE.LEFT_HIP);
  const rHip = read(LM_LIVE.RIGHT_HIP);

  s.shouldersOk = !!(lSh?.ok && rSh?.ok);
  s.hipsOk = !!(lHip?.ok && rHip?.ok);
  s.wristOk = s.usable;
  s.ready = s.shouldersOk && s.hipsOk && s.wristOk;

  if (lSh?.ok && rSh?.ok) {
    s.midX = (lSh.nx + rSh.nx) / 2;
    s.midXValid = true;
  }

  // Chosen side's shoulder — the spawn anchor.
  // Chosen side's shoulder, through the same mapping as the hand.
  const ownSh = read(side.shoulder);
  if (ownSh?.ok) {
    s.shoulderX = cover.offX + ownSh.nx * cover.dispW;
    s.shoulderY = cover.offY + ownSh.ny * cover.dispH;
    s.shoulderOk = true;
  } else {
    s.shoulderOk = false;
  }

  s.noseOk = !!read(LM_LIVE.NOSE)?.ok;

  // The other side, for the wrong-hand check.
  const otherHand: Hand = hand === "left" ? "right" : "left";
  const other = sideLandmarks(otherHand, swapped);
  const oWrist = read(other.wrist);
  if (oWrist) {
    s.otherX = cover.offX + oWrist.nx * cover.dispW;
    s.otherY = cover.offY + oWrist.ny * cover.dispH;
    s.otherUsable = oWrist.ok;
  } else {
    s.otherUsable = false;
  }
  const oSh = read(other.shoulder);
  if (oSh?.ok) {
    s.otherShoulderX = cover.offX + oSh.nx * cover.dispW;
    s.otherShoulderY = cover.offY + oSh.ny * cover.dispH;
    s.otherShoulderOk = true;
  } else {
    s.otherShoulderOk = false;
  }
  if (s.shoulderOk && s.otherShoulderOk) {
    s.midShoulderX = (s.shoulderX + s.otherShoulderX) / 2;
  }

  // ── Headroom.
  //
  // A raised arm puts the wrist about one arm length ABOVE the
  // shoulder. If that much canvas does not exist above the shoulder,
  // the raised wrist is off screen, the in-frame guard rejects it, and
  // the calibration ring can never fill — which is exactly how an "arm
  // up" hold ends up recorded below shoulder height.
  if (s.shoulderOk && s.elbowOk && elbowRead) {
    const ex = cover.offX + elbowRead.nx * cover.dispW;
    const ey = cover.offY + elbowRead.ny * cover.dispH;
    const upper = Math.hypot(s.shoulderX - ex, s.shoulderY - ey);
    s.armLenPx = upper + s.forearmPx;
  }
  // Canvas y = 0 IS the top of what is visible; anything above it was
  // cropped away by object-cover and can never be shown.
  s.headroomPx = s.shoulderOk ? Math.max(0, s.shoulderY) : 0;
  s.headroomRatio = s.armLenPx > 1 ? s.headroomPx / s.armLenPx : 0;
  s.headroomOk = s.headroomRatio >= HEADROOM_RATIO_MIN;

  s.setupOk =
    s.noseOk && s.shouldersOk && s.elbowOk && s.wristOk && s.headroomOk;
}
