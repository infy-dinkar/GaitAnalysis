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
  const lHip = landmarks[LM_LIVE.LEFT_HIP];
  const rHip = landmarks[LM_LIVE.RIGHT_HIP];

  const hasShoulders = visible(lSh, threshold) && visible(rSh, threshold);
  const hasHips = visible(lHip, threshold) && visible(rHip, threshold);
  if (!hasShoulders || !hasHips) return; // spine mid required for any centerline

  const shMid = midpointPx(lSh!, rSh!, w, h);
  const hipMid = midpointPx(lHip!, rHip!, w, h);
  if (Math.hypot(shMid.x - hipMid.x, shMid.y - hipMid.y) < 1) return;

  ctx.save();
  ctx.strokeStyle = opts?.strokeStyle ?? "rgba(148, 233, 255, 0.65)"; // soft cyan
  ctx.lineWidth = opts?.lineWidth ?? Math.max(1.5, w * 0.002);
  ctx.setLineDash(opts?.dash ?? [6, 4]);
  ctx.shadowBlur = 0;

  // Segment 1: NOSE → shoulder-mid (only when nose is visible)
  if (visible(nose, threshold)) {
    const nosePx = toPx(nose!, w, h);
    ctx.beginPath();
    ctx.moveTo(nosePx.x, nosePx.y);
    ctx.lineTo(shMid.x, shMid.y);
    ctx.stroke();
  }
  // Segment 2: shoulder-mid → hip-mid  (this is the trunk axis)
  ctx.beginPath();
  ctx.moveTo(shMid.x, shMid.y);
  ctx.lineTo(hipMid.x, hipMid.y);
  ctx.stroke();

  ctx.restore();
}

// ─── 2. Spine segment (back joint) ──────────────────────────────
// Solid line shoulder-mid → hip-mid, drawn slightly thicker than
// a regular bone and tinted teal so it reads as "trunk axis" not
// just another bone. Independent of centerline so a caller can pick
// one, both, or neither.

export interface SpineSegmentOptions {
  visibilityThreshold?: number;
  strokeStyle?: string;
  lineWidth?: number;
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
  if (Math.hypot(shMid.x - hipMid.x, shMid.y - hipMid.y) < 1) return;

  // Baseline bone thickness in RehabCameraShell = max(2, w * 0.0035).
  // Spine sits 1.5× that so it clearly reads as trunk axis.
  const baseLineWidth = Math.max(2, w * 0.0035);
  ctx.save();
  ctx.strokeStyle = opts?.strokeStyle ?? "rgba(45, 212, 191, 0.9)"; // teal
  ctx.lineWidth = opts?.lineWidth ?? baseLineWidth * 1.5;
  ctx.lineCap = "round";
  ctx.setLineDash([]);
  ctx.shadowColor = "rgba(45, 212, 191, 0.55)";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.moveTo(shMid.x, shMid.y);
  ctx.lineTo(hipMid.x, hipMid.y);
  ctx.stroke();
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
