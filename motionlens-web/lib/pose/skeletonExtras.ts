// Additive skeleton-overlay extras — centerline (body axis), spine
// segment, and joint angle arcs. Pure Canvas 2D functions, no React,
// no pose-detection knowledge beyond the shared LM_LIVE indices.
//
// Ownership boundary:
//   • This file is called by RehabCameraShell AFTER its existing
//     skeleton bone + dot loops complete. The extras render as
//     annotation ON TOP of the underlying skeleton.
//   • These helpers ONLY draw. They do not modify landmarks, do not
//     compute exercise angles, do not know about the mechanic
//     engines. All measurement math still happens elsewhere.
//   • Additive by design: absence of any call yields the classic
//     skeleton overlay unchanged.
//
// Norm shape here is defined locally to match RehabCameraShell's
// internal Norm without a cross-module import that would create a
// circular dependency (the shell owns the concrete Norm; this file
// only reads it structurally).

import { LM_LIVE } from "@/lib/pose/landmarks-live";

// Structural equivalent of the Norm interface used by
// RehabCameraShell + LiveBiomechCamera. Kept structural (duck-typed)
// so callers with a slightly wider type still work.
export interface Norm {
  x: number;
  y: number;
  visibility: number;
}

const DEFAULT_VIS_THRESHOLD = 0.35;

// ─── Small pixel helpers ────────────────────────────────────────
function toPx(n: Norm, w: number, h: number): { x: number; y: number } {
  return { x: n.x * w, y: n.y * h };
}

function midpointPx(
  a: Norm,
  b: Norm,
  w: number,
  h: number,
): { x: number; y: number } {
  return { x: ((a.x + b.x) / 2) * w, y: ((a.y + b.y) / 2) * h };
}

function visible(kp: Norm | undefined, threshold: number): boolean {
  return !!kp && kp.visibility >= threshold;
}

// ─── 1. Centerline (body axis) ──────────────────────────────────
// Polyline NOSE → shoulder-mid → hip-mid. Rendered as a dashed,
// semi-transparent guide so it never competes visually with the
// solid opaque bones drawn earlier.
//
// Each segment is independently visibility-gated: if the nose has
// dropped out (common in lateral views) the shoulder→hip segment
// still renders. Skip degenerate cases (zero-length).

export interface CenterlineOptions {
  /** Per-landmark visibility gate. Default 0.35 (matches
   *  OVERLAY_VIS_THRESHOLD in the camera shells). */
  visibilityThreshold?: number;
  /** Stroke colour — use rgba so the dashed line reads as guide,
   *  not bone. */
  strokeStyle?: string;
  /** Base width; scaled by canvas width at call time. */
  lineWidth?: number;
  /** Dash pattern. */
  dash?: [number, number];
}

export function drawCenterline(
  ctx: CanvasRenderingContext2D,
  landmarks: Norm[],
  w: number,
  h: number,
  opts?: CenterlineOptions,
): void {
  if (!landmarks || landmarks.length === 0) return;
  const threshold = opts?.visibilityThreshold ?? DEFAULT_VIS_THRESHOLD;
  const nose = landmarks[LM_LIVE.NOSE];
  const lSh = landmarks[LM_LIVE.LEFT_SHOULDER];
  const rSh = landmarks[LM_LIVE.RIGHT_SHOULDER];

  // Neck line only — nose → shoulder-mid. The shoulder-mid → hip-mid
  // trunk is now rendered as a vertebra chain by drawSpineSegment,
  // so we no longer duplicate a straight segment through the torso.
  if (!visible(nose, threshold)) return;
  if (!visible(lSh, threshold) || !visible(rSh, threshold)) return;
  const shMid = midpointPx(lSh!, rSh!, w, h);
  const nosePx = toPx(nose!, w, h);
  if (Math.hypot(nosePx.x - shMid.x, nosePx.y - shMid.y) < 1) return;

  ctx.save();
  ctx.strokeStyle = opts?.strokeStyle ?? "rgba(249, 115, 22, 0.9)"; // orange
  ctx.lineWidth = opts?.lineWidth ?? Math.max(2, w * 0.003);
  ctx.lineCap = "round";
  ctx.setLineDash([]);
  ctx.shadowColor = "rgba(249, 115, 22, 0.55)";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(nosePx.x, nosePx.y);
  ctx.lineTo(shMid.x, shMid.y);
  ctx.stroke();
  ctx.restore();
}

// ─── 2. Spine — flexible vertebra chain ─────────────────────────
// Renders the trunk axis as N short segments with a small dot at
// each interior vertebra. Because both endpoints (shoulder-mid,
// hip-mid) are recomputed each frame from live landmarks, the
// whole chain automatically tilts + translates with the trunk
// — no interpolation of pose-invisible "spine" landmarks needed.
//
// Replaces the previous single-segment straight line so the trunk
// reads as a Kemtai-style flexible spine rather than a rigid rod.

export interface SpineSegmentOptions {
  visibilityThreshold?: number;
  strokeStyle?: string;
  lineWidth?: number;
  /** Number of segments in the chain. 4 → 3 interior vertebra dots
   *  + 2 shoulder/hip-mid endpoints (which are already skeleton
   *  joint dots elsewhere — so no double-dot). Default 4. */
  segments?: number;
  /** Dot fill colour for the interior vertebrae. */
  dotColor?: string;
  /** Interior vertebra dot radius. Auto-scales to canvas by default. */
  dotRadius?: number;
}

export function drawSpineSegment(
  ctx: CanvasRenderingContext2D,
  landmarks: Norm[],
  w: number,
  h: number,
  opts?: SpineSegmentOptions,
): void {
  if (!landmarks || landmarks.length === 0) return;
  const threshold = opts?.visibilityThreshold ?? DEFAULT_VIS_THRESHOLD;
  const lSh = landmarks[LM_LIVE.LEFT_SHOULDER];
  const rSh = landmarks[LM_LIVE.RIGHT_SHOULDER];
  const lHip = landmarks[LM_LIVE.LEFT_HIP];
  const rHip = landmarks[LM_LIVE.RIGHT_HIP];
  if (
    !visible(lSh, threshold)
    || !visible(rSh, threshold)
    || !visible(lHip, threshold)
    || !visible(rHip, threshold)
  ) return;

  const shMid = midpointPx(lSh!, rSh!, w, h);
  const hipMid = midpointPx(lHip!, rHip!, w, h);
  const axisX = hipMid.x - shMid.x;
  const axisY = hipMid.y - shMid.y;
  const axisLen = Math.hypot(axisX, axisY);
  if (axisLen < 1) return;

  // ── Bow the spine to a curve driven by the body's lean ────────
  //
  // Linear interpolation between shMid and hipMid would put every
  // "vertebra" on the straight line between them — the chain would
  // rotate as a rigid rod but never actually curve. To match the
  // body's real bend we push the middle of the chain sideways,
  // perpendicular to the trunk axis, by an amount proportional to
  // how far the shoulder-mid has drifted off the vertical stack
  // over the hip-mid.
  //
  // Signal: bowSignal = hipMid.x - shMid.x
  //   • upright body → shMid.x ≈ hipMid.x → bowSignal ≈ 0 → straight
  //   • lateral lean → shoulders shift sideways → bowSignal grows
  //   • forward hinge (profile view) → shoulders shift forward → same
  //
  // Direction: perp to the trunk axis, chosen so a positive
  // bowSignal pushes points toward the lean side.
  // Scale: 0.6 × the raw offset produces a visible but not
  // exaggerated bow; clamp to 40 % of trunk length as a safety.
  //
  // Sign note: an earlier version used `shMid.x - hipMid.x` here,
  // which combined with the perpendicular (axisY, -axisX) and the
  // canvas y-down coordinate system produced a bow OPPOSITE to
  // the actual body lean (the spine curved away from where the
  // patient was actually bending). Flipping the subtraction order
  // to `hipMid.x - shMid.x` inverts the sign so the curve now
  // bows TOWARD the lean, verified for both left and right leans.

  const perpX =  axisY / axisLen;
  const perpY = -axisX / axisLen;
  const rawBow = (hipMid.x - shMid.x) * 0.6;
  const maxBow = axisLen * 0.4;
  const bow = Math.max(-maxBow, Math.min(maxBow, rawBow));

  // Straight midpoint (linear lerp) that we bow away from.
  const straightMidX = (shMid.x + hipMid.x) / 2;
  const straightMidY = (shMid.y + hipMid.y) / 2;

  // Quadratic Bezier control point placed so the curve's peak
  // lands at (straightMid + perp × bow). For a quadratic
  // curve at t=0.5 the point is P0/4 + C/2 + P2/4 — so to hit
  // (straightMid + offset) we set C = straightMid + 2 × offset.
  const bowOffsetX = perpX * bow;
  const bowOffsetY = perpY * bow;
  const controlX = straightMidX + 2 * bowOffsetX;
  const controlY = straightMidY + 2 * bowOffsetY;

  // Sample vertebra dots along the ACTUAL bowed curve so they lie
  // on the visible spine, not on the underlying straight axis.
  const nSegments = Math.max(2, opts?.segments ?? 4);
  const bezierAt = (t: number) => {
    const u = 1 - t;
    return {
      x: u * u * shMid.x + 2 * u * t * controlX + t * t * hipMid.x,
      y: u * u * shMid.y + 2 * u * t * controlY + t * t * hipMid.y,
    };
  };

  const baseLineWidth = Math.max(3, w * 0.005);
  ctx.save();
  ctx.strokeStyle = opts?.strokeStyle ?? "rgba(249, 115, 22, 0.9)"; // orange
  ctx.lineWidth = opts?.lineWidth ?? baseLineWidth * 1.2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash([]);
  ctx.shadowColor = "rgba(249, 115, 22, 0.55)";
  ctx.shadowBlur = 12;

  // Smooth quadratic curve from shoulder-mid to hip-mid, bulging
  // toward the lean direction. This is the "spine" the eye reads.
  ctx.beginPath();
  ctx.moveTo(shMid.x, shMid.y);
  ctx.quadraticCurveTo(controlX, controlY, hipMid.x, hipMid.y);
  ctx.stroke();

  // Interior vertebra dots — orange fills with orange glow so
  // they match the centre column of the side-coded skeleton.
  // Sampled directly ON the bowed curve so they visually snap
  // onto the visible spine.
  ctx.fillStyle = opts?.dotColor ?? "#F97316";
  ctx.shadowColor = "rgba(249, 115, 22, 0.6)";
  ctx.shadowBlur = 10;
  const r = opts?.dotRadius ?? Math.max(5, w * 0.008);
  for (let i = 1; i < nSegments; i++) {
    const t = i / nSegments;
    const p = bezierAt(t);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// ─── 3. Angle arc at a joint ────────────────────────────────────
// Partial arc between two "arm" directions from a vertex joint.
// Color-banded when a band is supplied:
//   • Inside band (with 15 % inset)              → green
//   • Near a band edge (inside or just outside)  → amber
//   • Clearly outside band                       → red
// If no band is supplied, arc renders in a neutral cyan tone.
// Optional degree label rendered near the arc midpoint.

export interface AngleArcConfig {
  /** Vertex landmark index (LM_LIVE.*). */
  vertex: number;
  /** First arm landmark index — one of the two limbs meeting at
   *  the vertex. */
  armA: number;
  /** Second arm landmark index. */
  armB: number;
  /** Current measured angle in degrees (already computed by the
   *  page's per-frame handler). Used only for the label + colour
   *  band; the arc's geometry uses the actual landmark positions. */
  currentDeg: number;
  /** Optional target band. Colours the arc green when currentDeg
   *  is comfortably inside, amber at the edges, red outside. */
  band?: { min: number; max: number };
  /** Optional label. If omitted, defaults to `${round(currentDeg)}°`. */
  label?: string;
  /** Per-landmark visibility gate. */
  visibilityThreshold?: number;
}

export function drawAngleArc(
  ctx: CanvasRenderingContext2D,
  landmarks: Norm[],
  w: number,
  h: number,
  cfg: AngleArcConfig,
): void {
  if (!landmarks || landmarks.length === 0) return;
  const threshold = cfg.visibilityThreshold ?? DEFAULT_VIS_THRESHOLD;
  const V = landmarks[cfg.vertex];
  const A = landmarks[cfg.armA];
  const B = landmarks[cfg.armB];
  if (
    !visible(V, threshold)
    || !visible(A, threshold)
    || !visible(B, threshold)
  ) return;

  const Vp = toPx(V!, w, h);
  const Ap = toPx(A!, w, h);
  const Bp = toPx(B!, w, h);

  const lenA = Math.hypot(Ap.x - Vp.x, Ap.y - Vp.y);
  const lenB = Math.hypot(Bp.x - Vp.x, Bp.y - Vp.y);
  // Only guard against a true zero-length arm (would leave atan2
  // undefined). Don't reject small arms — that silently killed the
  // arc on small camera tiles / distant subjects.
  if (lenA < 1 || lenB < 1) return;

  // Radius floor keeps the arc visually distinguishable from a bone
  // dot on any tile size. Baseline scales with the canvas so the arc
  // reads at similar visual weight on both small and large tiles.
  const canvasScale = Math.min(w, h);
  const radiusFloor = Math.max(14, canvasScale * 0.032);
  const radiusCeil = Math.max(radiusFloor + 4, canvasScale * 0.075);
  const armBased = Math.min(lenA, lenB) * 0.22;
  const radius = Math.max(radiusFloor, Math.min(radiusCeil, armBased));

  // Angles of the two arm directions (image-space, y-down).
  const a1 = Math.atan2(Ap.y - Vp.y, Ap.x - Vp.x);
  const a2 = Math.atan2(Bp.y - Vp.y, Bp.x - Vp.x);
  // Signed shorter delta in (-π, π]. Canvas arc's anticlockwise flag
  // then picks the visually-correct direction so we always draw the
  // interior (short) arc, never the long way around.
  let delta = a2 - a1;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta <= -Math.PI) delta += 2 * Math.PI;

  const color = colourForBand(cfg.currentDeg, cfg.band);
  const mainLineWidth = Math.max(4, canvasScale * 0.008);

  ctx.save();
  ctx.setLineDash([]);
  ctx.lineCap = "round";
  // Halo underlayer — dark translucent wide stroke so the coloured
  // arc always contrasts against the bright bones / video below.
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.lineWidth = mainLineWidth + 4;
  ctx.beginPath();
  ctx.arc(Vp.x, Vp.y, radius, a1, a1 + delta, delta < 0);
  ctx.stroke();
  // Main coloured arc.
  ctx.strokeStyle = color;
  ctx.lineWidth = mainLineWidth;
  ctx.beginPath();
  ctx.arc(Vp.x, Vp.y, radius, a1, a1 + delta, delta < 0);
  ctx.stroke();
  // Vertex marker — small filled circle so the eye latches onto the
  // joint centre even before it notices the arc itself.
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(Vp.x, Vp.y, Math.max(3, mainLineWidth * 0.6), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Label — small chip near the arc midpoint. Offsets outward
  // from the vertex along the arc's mid-angle direction.
  const midAngle = a1 + delta / 2;
  const labelDist = radius + 14;
  const lx = Vp.x + Math.cos(midAngle) * labelDist;
  const ly = Vp.y + Math.sin(midAngle) * labelDist;
  const text = cfg.label ?? `${Math.round(cfg.currentDeg)}°`;

  ctx.save();
  ctx.font = "600 12px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const padX = 6;
  const padY = 3;
  const metrics = ctx.measureText(text);
  const boxW = metrics.width + padX * 2;
  const boxH = 12 + padY * 2;
  // Chip background — dark translucent so text is legible over any
  // camera image.
  ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
  ctx.beginPath();
  const bx = lx - boxW / 2;
  const by = ly - boxH / 2;
  roundedRect(ctx, bx, by, boxW, boxH, 4);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#FFFFFF";
  ctx.fillText(text, lx, ly);
  ctx.restore();
}

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
}

function colourForBand(
  currentDeg: number,
  band?: { min: number; max: number },
): string {
  const NEUTRAL = "rgba(56, 189, 248, 0.9)"; // sky
  const GREEN = "rgba(52, 211, 153, 0.95)"; // emerald
  const AMBER = "rgba(251, 191, 36, 0.95)"; // amber
  const RED = "rgba(248, 113, 113, 0.95)"; // rose
  if (!band) return NEUTRAL;
  const range = Math.max(1e-3, band.max - band.min);
  const edge = range * 0.15;
  // Comfortably inside — well within the band, not near either edge
  if (
    currentDeg >= band.min + edge
    && currentDeg <= band.max - edge
  ) return GREEN;
  // Inside but near an edge → amber
  if (currentDeg >= band.min && currentDeg <= band.max) return AMBER;
  // Just outside (within one edge-margin of the band) → amber
  if (
    currentDeg >= band.min - edge
    && currentDeg <= band.max + edge
  ) return AMBER;
  // Clearly outside
  return RED;
}
