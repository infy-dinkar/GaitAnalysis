"use client";
// Shared spine overlay — the side-profile Hermite curve and the
// front-view 4-point lateral bend, lifted VERBATIM out of
// RehabCameraShell so other live shells can draw the same trunk
// without copying ~450 lines of geometry.
//
// Pure motion: this module is the rehab code as it stood, with the
// four per-frame refs passed in instead of closed over. No tuning, no
// behaviour change. The numbers below were measured on real clips and
// are documented where they are used.
//
// CONTRACT
//   • ctx is already translated into DISPLAY space by the caller
//     (the object-cover offX/offY shift), so every coordinate here is
//     in dispW/dispH units, matching what skeletonExtras expects.
//   • landmarks are normalised, mirrored Norm[] — the same array the
//     bone loop draws from.
//   • refs carry four frames of smoothing state. They are draw-only:
//     nothing here reaches onFrame or any mechanic engine.
//   • The neck centreline is drawn here too. It is not a separate
//     concern — it consumes neckAnchor, the spine's own top point, so
//     that the neck lands on the drawn spine rather than on the raw
//     shoulder-mid. Splitting them would mean returning neckAnchor to
//     the caller and trusting every caller to use it.

import {
  drawCenterline,
  drawSpineSegment,
} from "@/lib/pose/skeletonExtras";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

interface Norm {
  x: number;
  y: number;
  visibility: number;
}

// Matches the threshold the calling shells use for bones and dots.
// Kept local so this module has no dependency on any one shell.
const OVERLAY_VIS_THRESHOLD = 0.35;

/** Four frames of draw-only smoothing state, one set per camera. */
export interface SpineRefs {
  /** Last frame's side/front verdict, for the gate's hysteresis. */
  sideView: { current: boolean };
  /** EMA-smoothed unit vector ear -> shoulder-mid. */
  spineDir: { current: { x: number; y: number } | null };
  /** EMA-smoothed unit vector pointing POSTERIORLY (away from the nose). */
  spinePerp: { current: { x: number; y: number } | null };
  /** Front-view bend: smoothed lateral offset + the dead-zone latch. */
  frontLean: { current: { b: number; on: boolean } | null };
}

/**
 * Optional per-shell appearance. Every field defaults to the rehab
 * look, so a caller that passes nothing renders exactly as rehab does
 * today — the geometry is shared, the palette is not.
 *
 * NOT covered, because it lives in skeletonExtras and that file is
 * shared by other callers: the spine stroke's glow
 * (`rgba(249,115,22,0.55)`, blur 12) and the neck line's glow
 * (same colour, blur 8) are hardcoded there. A shell with a different
 * halo treatment keeps the orange glow on those two strokes until
 * those become options too.
 */
export interface SpineStyle {
  /** Spine polyline colour. Default: the helper's orange. */
  strokeStyle?: string;
  /** Spine polyline width. Default: the helper's own auto-scale. */
  lineWidth?: number;
  /** Front-bend knot dot fill. Default "#F97316". */
  dotColor?: string;
  /** Front-bend knot dot radius. Default max(4, dispW * 0.006). */
  dotRadius?: number;
  /** Knot dot glow colour. Default "rgba(249, 115, 22, 0.6)". */
  dotShadowColor?: string;
  /** Knot dot glow blur. Default 10. */
  dotShadowBlur?: number;
  /** Neck centreline colour. Default "rgba(249, 115, 22, 0.55)". */
  neckStrokeStyle?: string;
  /** Neck centreline width. Default max(2, dispW * 0.002). */
  neckLineWidth?: number;
}

/** Fresh state for one camera. Hold it in a single useRef. */
export function createSpineRefs(): SpineRefs {
  return {
    sideView: { current: false },
    spineDir: { current: null },
    spinePerp: { current: null },
    frontLean: { current: null },
  };
}

// Front-view spine: how much of their measured lateral share the two
// interior control points actually take. Counter-intuitively this goes
// DOWN to deepen the bend, because the visible bow is the departure
// from the straight S→H chord, and A and B sit INSIDE that chord — so
// pulling them further in bows the curve harder. Scaling them UP walks
// them onto the chord instead: A lands on it at gain 14/9 = 1.556 and
// the spine flattens into a diagonal stick.
//
// 0.45 gives ~2.22x the bow depth of the geometrically faithful 1.0
// while keeping both points comfortably inside the chord.
const BEND_GAIN = 0.45;

// Uniform Catmull-Rom through every supplied point, with the two
// endpoints duplicated so the curve starts exactly on the first control
// and ends exactly on the last. Catmull-Rom INTERPOLATES its controls,
// so the knots are on the curve, not merely near it — the front spine's
// bend points keep their computed positions and only the corners
// between them round off.
//
// Every sample is an affine combination of four controls (the weights
// sum to 1), so collinear controls return an exactly straight line —
// which is what a zero bend must draw.
function catmullRomPath(
  pts: { x: number; y: number }[],
  samples: number,
): { x: number; y: number }[] {
  const first = pts[0];
  const last = pts[pts.length - 1];
  if (!first || !last || pts.length < 2 || samples < 2) return pts;
  const ctrl = [first, ...pts, last];
  const segs = pts.length - 1;
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i < samples; i++) {
    const u = (i / (samples - 1)) * segs;
    const seg = Math.min(segs - 1, Math.floor(u));
    const t = u - seg;
    const p0 = ctrl[seg];
    const p1 = ctrl[seg + 1];
    const p2 = ctrl[seg + 2];
    const p3 = ctrl[seg + 3];
    if (!p0 || !p1 || !p2 || !p3) continue;
    const t2 = t * t;
    const t3 = t2 * t;
    out.push({
      x: 0.5 * (2 * p1.x
        + (-p0.x + p2.x) * t
        + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2
        + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
      y: 0.5 * (2 * p1.y
        + (-p0.y + p2.y) * t
        + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2
        + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
    });
  }
  return out;
}

// Torso side edges (shoulder→hip verticals) — skipped in the bone
// loop so the shoulders + hips no longer close into a rectangle.
// The horizontal shoulder-shoulder and hip-hip bars stay; the
// vertebra chain drawn by drawSpineSegment fills the middle.
export const TORSO_SIDE_EDGES: ReadonlySet<string> = new Set<string>([
  `${LM.LEFT_SHOULDER}-${LM.LEFT_HIP}`,
  `${LM.RIGHT_SHOULDER}-${LM.RIGHT_HIP}`,
]);

/**
 * Draw the trunk: neck centreline, then the spine as either a
 * side-profile Hermite curve or a front-view 4-point bend.
 *
 * ctx must already be translated into display space.
 */
export function drawSpineOverlay(
  ctx: CanvasRenderingContext2D,
  landmarks: Norm[],
  dispW: number,
  dispH: number,
  refs: SpineRefs,
  style?: SpineStyle,
): void {
  // Aliased so the geometry below is the rehab source unchanged,
  // down to the identifier names.
  const sideViewRef = refs.sideView;
  const spineDirRef = refs.spineDir;
  const spinePerpRef = refs.spinePerp;
  const frontLeanRef = refs.frontLean;

  // Spine as a STRAIGHT shoulder-mid → hip-mid line.
  //
  // drawSpineSegment's own geometry infers a bow from the
  // shoulder-mid → hip-mid x-offset alone, which reads a forward
  // lean, a torso rotation, or plain landmark jitter as spinal
  // curvature — the spine visibly bent on a straight back. Passing
  // explicit points bypasses that inference entirely. Same
  // dispW/dispH space as the other extras, so the enclosing
  // ctx.translate(offX, offY) applies unchanged.
  //
  // showDots:false — the interior "vertebra" dots were never
  // measured positions, just evenly-spaced marks on the segment.
  // On a straight line they add nothing but visual noise, so the
  // trunk renders as a single clean stroke.
  const lShP = landmarks[LM.LEFT_SHOULDER];
  const rShP = landmarks[LM.RIGHT_SHOULDER];
  const lHipP = landmarks[LM.LEFT_HIP];
  const rHipP = landmarks[LM.RIGHT_HIP];
  const spinePts =
    lShP && rShP && lHipP && rHipP
      ? [
          {
            x: ((lShP.x + rShP.x) / 2) * dispW,
            y: ((lShP.y + rShP.y) / 2) * dispH,
          },
          {
            x: ((lHipP.x + rHipP.x) / 2) * dispW,
            y: ((lHipP.y + rHipP.y) / 2) * dispH,
          },
        ]
      : undefined;
  // ── Side-profile curvature ────────────────────────────────
  // The spine only curves in a LATERAL view. Front-on there is no
  // depth information at all — any apparent bow would be torso
  // rotation or landmark jitter, which is exactly the artefact the
  // straight-line rewrite removed. So the curve is gated on view.
  //
  // Gate = shoulder width / trunk length, both in display pixels.
  // Facing the camera the shoulders span roughly half the trunk or
  // more (~0.5+); in true profile one shoulder hides behind the
  // other and the projected span collapses to ~0.15. 0.30 sits in
  // the empty middle. Hysteresis (enter < 0.30, leave > 0.38) stops
  // the spine snapping straight/curved as a patient turns through
  // the boundary.
  //
  // The direction comes from the EAR: as the thoracic spine rounds,
  // the head translates forward and the ear goes with it, so the
  // ear → shoulder-mid vector carries the actual curvature. It is
  // also the steadiest head landmark in profile — the nose swings
  // with head rotation and the eyes occlude.
  //
  // POSTERIOR OFFSET (side view only): shoulder-mid and hip-mid sit
  // at the body's mid-depth, so a line between them runs through the
  // middle of the trunk, not along the back the clinician is looking
  // at. Shifting both ends backward by a fraction of trunk length —
  // more at the chest (0.12) than the pelvis (0.07), because the
  // ribcage is the deeper of the two — puts the stroke on the
  // posterior surface. "Backward" is derived from the ear → nose
  // vector, which is the only reliable facing signal in profile.
  let spineDraw = spinePts;
  let neckAnchor: { x: number; y: number } | undefined;
  let spineTangentFrom: { x: number; y: number } | undefined;
  if (spinePts && lShP && rShP) {
    const S = spinePts[0];
    const H = spinePts[1];
    const shoulderW = Math.hypot(
      (lShP.x - rShP.x) * dispW,
      (lShP.y - rShP.y) * dispH,
    );
    const trunkLen = Math.hypot(H.x - S.x, H.y - S.y);

    // SIDE/FRONT GATE. The denominator is thigh length, not trunk
    // length: the femur keeps its projected length through a forward
    // bend, the trunk does not. Measured over the gait and
    // sit-to-stand clips, shoulderW/trunkLen pushed 1.5–8.3% of
    // genuine side frames past the old 0.38 line, and a deep bend
    // held FRONT for the whole pose. shoulderW/thighLen separates
    // with an empty gap — side median 0.20–0.26, front median
    // 0.82–1.07, forward-lean side frames 0.10–0.13 — so the
    // 0.60/0.75 hysteresis band sits in clear air.
    //
    // Both thighs are averaged when both pass, so one occluded knee
    // cannot halve the denominator. With no usable thigh, or a
    // shoulder below threshold, the previous verdict is HELD rather
    // than recomputed: a stale verdict is cheaper than a wrong flip.
    const lKneeP = landmarks[LM.LEFT_KNEE];
    const rKneeP = landmarks[LM.RIGHT_KNEE];
    const thighs: number[] = [];
    if (
      lHipP && lKneeP
      && lHipP.visibility >= OVERLAY_VIS_THRESHOLD
      && lKneeP.visibility >= OVERLAY_VIS_THRESHOLD
    ) {
      thighs.push(Math.hypot(
        (lHipP.x - lKneeP.x) * dispW,
        (lHipP.y - lKneeP.y) * dispH,
      ));
    }
    if (
      rHipP && rKneeP
      && rHipP.visibility >= OVERLAY_VIS_THRESHOLD
      && rKneeP.visibility >= OVERLAY_VIS_THRESHOLD
    ) {
      thighs.push(Math.hypot(
        (rHipP.x - rKneeP.x) * dispW,
        (rHipP.y - rKneeP.y) * dispH,
      ));
    }
    const shouldersVisible =
      lShP.visibility >= OVERLAY_VIS_THRESHOLD
      && rShP.visibility >= OVERLAY_VIS_THRESHOLD;
    // Hoisted out of the branch below only so the front-view bend's
    // rotation guard can read it; null on a held frame. The gate
    // logic and its 0.60/0.75 edges are unchanged.
    let gateRatio: number | null = null;
    if (thighs.length > 0 && shouldersVisible) {
      const thighLen = thighs.reduce((a, b) => a + b, 0) / thighs.length;
      gateRatio = shoulderW / Math.max(thighLen, 1);
      sideViewRef.current = sideViewRef.current
        ? gateRatio <= 0.75
        : gateRatio < 0.60;
    }
    const isSide = sideViewRef.current;

    const lEar = landmarks[LM.LEFT_EAR];
    const rEar = landmarks[LM.RIGHT_EAR];
    const nearEar =
      (lEar?.visibility ?? 0) >= (rEar?.visibility ?? 0) ? lEar : rEar;
    const ear =
      nearEar && nearEar.visibility >= OVERLAY_VIS_THRESHOLD
        ? nearEar
        : undefined;
    const noseP = landmarks[LM.NOSE];
    const nose =
      noseP && noseP.visibility >= OVERLAY_VIS_THRESHOLD ? noseP : undefined;

    if (!isSide || !ear || trunkLen < 1) {
      // Front view, or the ear dropped out: forget both smoothed
      // vectors so re-entry seeds fresh instead of easing out of a
      // stale one, and leave tangentFrom/spineDraw untouched — the
      // helper then draws the same straight [S, H] line as before
      // this change.
      spineDirRef.current = null;
      spinePerpRef.current = null;
    } else {
      // Nose lost: no facing signal, so the offset is skipped for
      // this frame and both refs re-seed on the next good one.
      if (!nose) {
        spineDirRef.current = null;
        spinePerpRef.current = null;
      }
      const rawX = S.x - ear.x * dispW;
      const rawY = S.y - ear.y * dispH;
      const rawLen = Math.hypot(rawX, rawY);
      if (rawLen >= 1) {
        let dx = rawX / rawLen;
        let dy = rawY / rawLen;
        const prev = spineDirRef.current;
        if (prev) {
          dx = prev.x + 0.3 * (dx - prev.x);
          dy = prev.y + 0.3 * (dy - prev.y);
          const l = Math.hypot(dx, dy);
          if (l >= 1e-6) {
            dx /= l;
            dy /= l;
          } else {
            dx = rawX / rawLen;
            dy = rawY / rawLen;
          }
        }
        spineDirRef.current = { x: dx, y: dy };

        // ── Posterior offset ──────────────────────────────
        // fwd = ear → nose (which way the patient faces);
        // p    = the trunk-axis perpendicular pointing AWAY from
        //        the nose, i.e. toward the back.
        if (nose) {
          const fx = (nose.x - ear.x) * dispW;
          const fy = (nose.y - ear.y) * dispH;
          const fLen = Math.hypot(fx, fy);
          if (fLen >= 1) {
            const ax = (H.x - S.x) / trunkLen;
            const ay = (H.y - S.y) / trunkLen;
            let px = ay;
            let py = -ax;
            if (px * (fx / fLen) + py * (fy / fLen) > 0) {
              px = -px;
              py = -py;
            }
            const prevP = spinePerpRef.current;
            if (prevP) {
              px = prevP.x + 0.3 * (px - prevP.x);
              py = prevP.y + 0.3 * (py - prevP.y);
              const pl = Math.hypot(px, py);
              if (pl >= 1e-6) {
                px /= pl;
                py /= pl;
              }
            }
            spinePerpRef.current = { x: px, y: py };
            const sOff = 0.12 * trunkLen;
            const hOff = 0.07 * trunkLen;
            spineDraw = [
              { x: S.x + px * sOff, y: S.y + py * sOff },
              { x: H.x + px * hOff, y: H.y + py * hOff },
            ];
            // The neck must land on the spine's new top, not the
            // raw shoulder-mid, or it stops short and leaves a gap.
            neckAnchor = spineDraw[0];
          }
        }

        // Only the DIRECTION matters downstream (drawSpineSegment
        // renormalises), so project a synthetic anchor one trunk
        // length back along the smoothed vector. Anchored at the
        // DRAWN top point so the offset does not tilt θ.
        const top = neckAnchor ?? S;
        spineTangentFrom = {
          x: top.x - dx * trunkLen,
          y: top.y - dy * trunkLen,
        };
      }
    }

    // ── Front-view lateral bend ─────────────────────────────────
    // Front-on there is no depth, so the side-profile Hermite above
    // is meaningless here — but a LATERAL lean is genuinely visible
    // and is what a clinician reads from this view.
    //
    // The bend is driven by the LATERAL OFFSET of shoulder-mid from
    // hip-mid, measured in the PELVIS frame. Shoulder-line tilt was
    // tried first and cannot see this pose at all: in pure lateral
    // flexion the shoulder line rotates WITH the trunk, so its angle
    // relative to the trunk axis barely changes while the trunk is
    // plainly bent, and no bow was drawn.
    //
    //   u     = unit(rHip − lHip)  pelvis lateral direction
    //   n     = ⟂ u, oriented H → S
    //   v     = S − H
    //   a     = v · n              trunk height in the pelvis frame
    //   b     = v · u              lateral offset — the bend signal
    //   lean  = |atan2(b, a)|
    //
    // b needs no scaling: it already is the displacement being drawn.
    //
    // S and H are never moved — they stay exactly on the shoulder-mid
    // and hip-mid landmarks, so only A (at 1/3) and B (at 2/3) carry
    // the bow, at 3/7 and 1/7 of b.
    //
    // Straight segments between the four points: BlazePose has no
    // mid-torso landmark, so A and B are inferred bend points, not
    // measured vertebrae, and the two corners say so honestly.
    if (!isSide && trunkLen >= 1 && lHipP && rHipP) {
      // PELVIS FRAME. In a lateral flexion the pelvis is the base the
      // trunk bends away from, so the axes come from the HIP LINE:
      // u runs along the hips, n is square to it and oriented up the
      // body (H → S).
      //
      // Bend source = the LATERAL OFFSET of shoulder-mid from
      // hip-mid, v·u. Shoulder-line tilt cannot see this pose: in
      // pure lateral flexion the shoulder line rotates WITH the
      // trunk, so its angle relative to the trunk axis barely moves
      // while the trunk is visibly bent. The offset moves with the
      // full excursion, and needs no scaling — it already IS the
      // displacement being drawn.
      const hipVX = (rHipP.x - lHipP.x) * dispW;
      const hipVY = (rHipP.y - lHipP.y) * dispH;
      const hipLen = Math.hypot(hipVX, hipVY);
      const vX = S.x - H.x;
      const vY = S.y - H.y;

      let uX: number;
      let uY: number;
      let nX: number;
      let nY: number;
      if (hipLen >= 1) {
        uX = hipVX / hipLen;
        uY = hipVY / hipLen;
        // Square to u, then flipped if it points down the body, so n
        // is always H → S whichever way the hip line is ordered.
        nX = -uY;
        nY = uX;
        if (nX * vX + nY * vY < 0) {
          nX = -nX;
          nY = -nY;
        }
      } else {
        // Degenerate hip line (hips coincident): fall back to the
        // trunk chord so the straight line still draws correctly.
        nX = vX / trunkLen;
        nY = vY / trunkLen;
        uX = -nY;
        uY = nX;
      }

      // a = trunk height in the pelvis frame, b = lateral excursion.
      const a = vX * nX + vY * nY;
      const bRaw = vX * uX + vY * uY;

      // Visibility guard: a bend inferred from a landmark the model
      // is guessing at is worse than no bend, so fall back to the
      // straight line — still four points, so the dots stay put.
      const trunkVisible =
        lShP.visibility >= OVERLAY_VIS_THRESHOLD
        && rShP.visibility >= OVERLAY_VIS_THRESHOLD
        && lHipP.visibility >= OVERLAY_VIS_THRESHOLD
        && rHipP.visibility >= OVERLAY_VIS_THRESHOLD;

      let b = 0;
      if (!trunkVisible || hipLen < 1) {
        frontLeanRef.current = null;
      } else {
        // EMA on the offset at 0.15, plus a hard slew limit expressed
        // as an ANGLE — 2.5° per frame, converted to px at this trunk
        // height so the cap means the same thing on a tall adult and
        // a child. Lateral flexion is a slow movement, so trading
        // responsiveness for steadiness is the right way round here:
        // a single mis-tracked frame can only walk the spine 2.5°
        // sideways, never snap it.
        const prev = frontLeanRef.current;
        let smoothed = bRaw;
        if (prev !== null) {
          const eased = prev.b + 0.15 * (bRaw - prev.b);
          const maxStep = Math.abs(a) * Math.tan((2.5 * Math.PI) / 180);
          smoothed = prev.b
            + Math.max(-maxStep, Math.min(maxStep, eased - prev.b));
        }

        // Dead zone with hysteresis: the bend switches ON above 7°
        // and OFF below 5°, and holds whatever it already was in
        // between, so a lean hovering on the boundary cannot flicker
        // the curve on and off frame to frame. Seeded OFF, so a fresh
        // entry has to clear 7° before anything is drawn.
        //
        // The ref is NOT cleared here — the offset keeps smoothing
        // through neutral so crossing the threshold is continuous
        // rather than a jump from a fresh seed.
        const leanDeg = Math.abs((Math.atan2(smoothed, a) * 180) / Math.PI);
        const wasOn = prev?.on ?? false;
        const on = leanDeg > 7 ? true : leanDeg < 5 ? false : wasOn;
        frontLeanRef.current = { b: smoothed, on };

        if (on) {
          // ROTATION GUARD, now reading gateRatio (shoulder span /
          // thigh length) rather than shoulder/trunk. Measured, the
          // old input never suppressed a genuine lateral flexion —
          // it sits at confidence 1.000 through 35° of bend — but it
          // has a hole: trunk length foreshortens on a forward bend,
          // so a patient turned 60° away AND bent forward reads
          // ratio 0.610 and gets full confidence in a bend that is
          // pure foreshortening. gateRatio cannot foreshorten, and
          // its 0.60→0.75 band is exactly the gate's own hysteresis,
          // so the bend fades to nothing precisely as the gate
          // commits to SIDE. smoothstep, so neither edge flicks.
          //
          // No gateRatio (knee or shoulder dropped out, gate holding
          // its previous verdict): trust the four trunk landmarks,
          // which passed their own guard above, rather than blanking
          // a bend the clinician can plainly see.
          const g = gateRatio ?? 1;
          const t = Math.min(1, Math.max(0, (g - 0.60) / (0.75 - 0.60)));
          const confidence = t * t * (3 - 2 * t);

          b = smoothed * confidence;
        }
      }

      // Lateral shares of b, read top-down: S carries the full offset
      // (it IS the shoulder landmark), A three sevenths and B one
      // seventh, both scaled by BEND_GAIN, H none.
      //
      // At gain 1.0 the gaps are 4/7, 2/7, 1/7 — the 4:2:1 split. No
      // gain can preserve that: the three gaps telescope from H to S
      // so they always sum to b, which leaves 4:2:1 exactly one
      // solution and no freedom to deepen. The closed form is
      // (7/gain − 3) : 2 : 1, so 0.45 gives 12.556 : 2 : 1 — most of
      // the excursion spent up near the shoulders, which is where a
      // laterally-flexed spine actually does most of its travelling.
      //
      // b = 0 collapses every lateral term whatever the gain, so the
      // dead zone, the rotation guard and the visibility guard all
      // still produce exactly the straight S→H line, as four points.
      // S and H are never moved off their landmarks.
      spineDraw = [
        S,
        {
          x: H.x + nX * ((2 * a) / 3) + uX * ((3 / 7) * b * BEND_GAIN),
          y: H.y + nY * ((2 * a) / 3) + uY * ((3 / 7) * b * BEND_GAIN),
        },
        {
          x: H.x + nX * (a / 3) + uX * ((1 / 7) * b * BEND_GAIN),
          y: H.y + nY * (a / 3) + uY * ((1 / 7) * b * BEND_GAIN),
        },
        H,
      ];
    } else {
      frontLeanRef.current = null;
    }
  }
  // Neck (nose -> shoulder-mid) deliberately thinner and more
  // transparent than the spine below it. At the spine's own weight
  // the two read as one continuous orange rod running through the
  // head; dropping to ~2/3 the width and 0.55 alpha separates them.
  // Drawn here (after the spine geometry, before the spine stroke)
  // only so it can reuse neckAnchor; z-order is unchanged.
  drawCenterline(ctx, landmarks, dispW, dispH, {
    visibilityThreshold: OVERLAY_VIS_THRESHOLD,
    strokeStyle: style?.neckStrokeStyle ?? "rgba(249, 115, 22, 0.55)",
    lineWidth: style?.neckLineWidth ?? Math.max(2, dispW * 0.002),
    endPoint: neckAnchor,
  });
  // The four control points describe the bend correctly, but three
  // straight lineTo segments render it as a bent stick with two hard
  // corners. Resampling them through a Catmull-Rom spline rounds the
  // corners while leaving A and B exactly where the maths put them.
  // Side view is untouched: the Hermite path is already a smooth
  // 13-sample curve and the 2-point straight line has nothing to
  // round.
  const frontBend = spineDraw?.length === 4;
  const strokePts =
    frontBend && spineDraw ? catmullRomPath(spineDraw, 16) : spineDraw;

  // showDots is false on every branch now. The helper dots every
  // interior point, which on a 16-sample curve would be 14 marks
  // strung along the spine; the meaningful positions are the four
  // KNOTS, so they are drawn below instead. Nothing was added to
  // skeletonExtras for this.
  drawSpineSegment(ctx, landmarks, dispW, dispH, {
    visibilityThreshold: OVERLAY_VIS_THRESHOLD,
    points: strokePts,
    showDots: false,
    // Undefined passes straight through to drawSpineSegment's own
    // defaults, so rehab is unchanged.
    strokeStyle: style?.strokeStyle,
    lineWidth: style?.lineWidth,
    tangentFrom: spineTangentFrom,
  });

  // Knot markers, front bend only: S, A, B and H. Fill, shadow and
  // radius match what the helper would have drawn, and its own
  // save/restore has already unwound by this point, so this sets up
  // and tears down its own canvas state. Radius sits slightly under
  // the helper's default (max(5, w*0.008)) so the bend markers stay
  // subordinate to the body joint dots.
  if (frontBend && spineDraw) {
    ctx.save();
    ctx.fillStyle = style?.dotColor ?? "#F97316";
    ctx.shadowColor = style?.dotShadowColor ?? "rgba(249, 115, 22, 0.6)";
    ctx.shadowBlur = style?.dotShadowBlur ?? 10;
    const r = style?.dotRadius ?? Math.max(4, dispW * 0.006);
    for (const p of spineDraw) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}
