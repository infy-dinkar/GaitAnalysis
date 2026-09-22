// Three-hold reach calibration.
//
// The patient holds the chosen hand still in three poses; each hold
// fills a 5 s ring, and the ring RESETS the moment the wrist drifts.
// The recorded point is the median of the last second of samples, so a
// single bad frame cannot move it.
//
// Samples that fail the visibility floor or fall outside the frame are
// dropped at source. An earlier prototype skipped that and recorded
// MediaPipe's extrapolated guesses for a hand that was off-camera,
// which produced a "reach" extending well past the frame edge.

import type { Hand } from "@/lib/games/handTracker";

export const HOLD_MS = 5000;
/** Wrist drift (normalised units) that resets the ring. ~4% of frame. */
export const STILL_TOLERANCE = 0.04;
/** Window whose median becomes the recorded point. */
export const MEDIAN_WINDOW_MS = 1000;

export type HoldId = "up" | "side" | "across";

export interface HoldDef {
  id: HoldId;
  title: string;
  instruction: string;
}

export const HOLDS: HoldDef[] = [
  {
    id: "up",
    title: "Arm straight up",
    instruction: "Reach as high as you comfortably can and hold still.",
  },
  {
    id: "side",
    title: "Arm out to the side",
    instruction: "Reach out sideways, same side as your playing hand.",
  },
  {
    id: "across",
    title: "Arm across your body",
    instruction: "Reach across toward your other shoulder and hold still.",
  },
];

export interface Point {
  nx: number;
  ny: number;
}

interface Sample extends Point {
  t: number;
}

/**
 * Drives one hold. Feed it every frame; it reports ring progress and
 * hands back a point once the hold completes.
 */
export class HoldTracker {
  private samples: Sample[] = [];
  private anchor: Point | null = null;
  private startedAt = 0;

  /** 0..1 ring fill. */
  progress = 0;
  /** True while the wrist is inside tolerance and the ring is filling. */
  holding = false;

  reset(): void {
    this.samples = [];
    this.anchor = null;
    this.startedAt = 0;
    this.progress = 0;
    this.holding = false;
  }

  /**
   * @returns the recorded point when the hold completes, else null.
   */
  feed(nx: number, ny: number, usable: boolean, nowMs: number): Point | null {
    // Rejected sample — visibility floor or out of frame. Treat exactly
    // like a break in the hold rather than guessing a position.
    if (!usable) {
      this.anchor = null;
      this.samples = [];
      this.startedAt = 0;
      this.progress = 0;
      this.holding = false;
      return null;
    }

    if (!this.anchor) {
      this.anchor = { nx, ny };
      this.startedAt = nowMs;
      this.samples = [{ nx, ny, t: nowMs }];
      this.progress = 0;
      this.holding = true;
      return null;
    }

    const drift = Math.hypot(nx - this.anchor.nx, ny - this.anchor.ny);
    if (drift > STILL_TOLERANCE) {
      // Moved too far — restart the ring from the new position.
      this.anchor = { nx, ny };
      this.startedAt = nowMs;
      this.samples = [{ nx, ny, t: nowMs }];
      this.progress = 0;
      this.holding = true;
      return null;
    }

    this.samples.push({ nx, ny, t: nowMs });
    this.holding = true;
    const elapsed = nowMs - this.startedAt;
    this.progress = Math.min(1, elapsed / HOLD_MS);

    if (elapsed < HOLD_MS) return null;

    // Median of the last MEDIAN_WINDOW_MS of samples.
    const cutoff = nowMs - MEDIAN_WINDOW_MS;
    const tail = this.samples.filter((s) => s.t >= cutoff);
    const use = tail.length > 0 ? tail : this.samples;
    const point = { nx: median(use.map((s) => s.nx)), ny: median(use.map((s) => s.ny)) };
    this.reset();
    return point;
  }
}

function median(values: number[]): number {
  const a = [...values].sort((p, q) => p - q);
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

export interface ReachBox {
  xLo: number;
  xHi: number;
  yLo: number;
  yHi: number;
  /** Body midline in mirrored normalised x. */
  midX: number;
  hand: Hand;
  /** The three held points themselves, kept so spawning can work in
   *  polar terms around the shoulder. The bounding box alone throws
   *  away the directions, which is what let fruit pile up near the
   *  chest instead of at the limit of each direction. */
  holds: { up: Point; side: Point; across: Point };
}

/** Keep fruit off the very edge of what the patient can reach. */
const EDGE_INSET = 0.06;
/** How far below the lowest hold the box extends, as a fraction of its
 *  own height — the three holds all sit at or above shoulder height, so
 *  without this every fruit would spawn in the top half of the frame. */
const BOTTOM_EXTEND = 0.3;
/** A box narrower than this in either axis is unusable. */
const MIN_SPAN = 0.1;

/**
 * Build the spawn box from the three recorded holds.
 *
 * The across-midline hold is what pulls `xLo`/`xHi` past the body
 * centre, and fruit placed there is what exercises adduction — so the
 * box is deliberately NOT clamped to the playing side.
 */
export function buildReachBox(
  up: Point,
  side: Point,
  across: Point,
  midX: number,
  hand: Hand,
): ReachBox {
  const xs = [up.nx, side.nx, across.nx];
  const ys = [up.ny, side.ny, across.ny];

  let xLo = Math.min(...xs);
  let xHi = Math.max(...xs);
  let yLo = Math.min(...ys);
  let yHi = Math.max(...ys);

  yHi = yHi + (yHi - yLo) * BOTTOM_EXTEND;

  const insetX = (xHi - xLo) * EDGE_INSET;
  const insetY = (yHi - yLo) * EDGE_INSET;
  xLo += insetX;
  xHi -= insetX;
  yLo += insetY;
  yHi -= insetY;

  // Guard degenerate boxes (a patient who barely moved) by widening
  // around the centre rather than letting spawns collapse to a point.
  if (xHi - xLo < MIN_SPAN) {
    const c = (xHi + xLo) / 2;
    xLo = c - MIN_SPAN / 2;
    xHi = c + MIN_SPAN / 2;
  }
  if (yHi - yLo < MIN_SPAN) {
    const c = (yHi + yLo) / 2;
    yLo = c - MIN_SPAN / 2;
    yHi = c + MIN_SPAN / 2;
  }

  return {
    xLo: clamp01(xLo),
    xHi: clamp01(xHi),
    yLo: clamp01(yLo),
    yHi: clamp01(yHi),
    midX: clamp01(midX),
    hand,
    holds: { up, side, across },
  };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Pick a spawn point inside the box, on the requested side of the
 * midline.
 *
 * Mirrored space: the patient's right hand is drawn on the RIGHT of the
 * screen (higher nx), so for a right-handed session "same side" is the
 * high-x half and "across" is the low-x half. Reversed for the left.
 */
export function spawnPoint(
  box: ReachBox,
  region: "same" | "across",
  rand: () => number,
): Point {
  const sameIsHighX = box.hand === "right";
  const wantHighX = region === "same" ? sameIsHighX : !sameIsHighX;

  let lo: number;
  let hi: number;
  if (wantHighX) {
    lo = Math.max(box.xLo, box.midX);
    hi = box.xHi;
  } else {
    lo = box.xLo;
    hi = Math.min(box.xHi, box.midX);
  }
  // The midline can fall outside the measured box (a patient who never
  // crossed it). Fall back to the whole box rather than an empty range.
  if (hi - lo < 0.02) {
    lo = box.xLo;
    hi = box.xHi;
  }

  return {
    nx: lo + rand() * (hi - lo),
    ny: box.yLo + rand() * (box.yHi - box.yLo),
  };
}

// ── Edge-of-reach spawning ────────────────────────────────────────
//
// The bounding box was the wrong shape for the job. Uniform sampling
// inside it puts most fruit near the middle of the box — which is the
// middle of the body, at chest height, right where the hand already
// rests. A round could be won with almost no arm movement.
//
// Everything below works in POLAR terms around the chosen shoulder: a
// direction on the arc from across-the-body, through straight out, to
// overhead, and a distance that is a high fraction of the measured
// reach in that direction.

export interface Cover {
  dispW: number;
  dispH: number;
  offX: number;
  offY: number;
}

export interface ReachGeometry {
  /** Anchor, canvas px. */
  sx: number;
  sy: number;
  /** Radii in canvas px, already extended for the palm offset. */
  rAcross: number;
  rSide: number;
  rUp: number;
  /** Direction of each measured hold from the shoulder, radians. */
  aAcross: number;
  aSide: number;
  aUp: number;
}

/**
 * Build the polar model from the calibration holds.
 *
 * Radii are measured from the LIVE shoulder rather than a stored one,
 * so if the patient drifts a step the whole reach fan follows them.
 *
 * `palmOffsetPx` is added to every radius. Calibration records the
 * WRIST, but the cursor is the palm, which sits PALM_REACH of a
 * forearm further out — so a target at the raw wrist radius is
 * touchable without extending. Adding the offset back makes 100%
 * reach mean 100% reach.
 */
export function reachGeometry(
  box: ReachBox,
  shoulderX: number,
  shoulderY: number,
  cover: Cover,
  palmOffsetPx: number,
): ReachGeometry | null {
  if (cover.dispW <= 0 || cover.dispH <= 0) return null;
  const toPx = (p: Point) => ({
    x: cover.offX + p.nx * cover.dispW,
    y: cover.offY + p.ny * cover.dispH,
  });
  const pick = (p: Point) => {
    const q = toPx(p);
    const dx = q.x - shoulderX;
    const dy = q.y - shoulderY;
    return { r: Math.hypot(dx, dy) + palmOffsetPx, a: Math.atan2(dy, dx) };
  };
  const up = pick(box.holds.up);
  const side = pick(box.holds.side);
  const across = pick(box.holds.across);
  if (
    !Number.isFinite(up.r)
    || !Number.isFinite(side.r)
    || !Number.isFinite(across.r)
  ) {
    return null;
  }
  return {
    sx: shoulderX,
    sy: shoulderY,
    rUp: up.r,
    rSide: side.r,
    rAcross: across.r,
    aUp: up.a,
    aSide: side.a,
    aAcross: across.a,
  };
}

/** Shortest-arc interpolation, so the fan never takes the long way
 *  round when two holds straddle the -PI/+PI wrap. */
function lerpAngle(a: number, b: number, t: number): number {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

/**
 * Direction and radius at fan position `u`:
 *   u = 0   -> across the body   (adduction)
 *   u = 0.5 -> straight out      (mid abduction)
 *   u = 1   -> overhead          (full abduction)
 */
function fanAt(g: ReachGeometry, u: number): { a: number; r: number } {
  if (u <= 0.5) {
    const t = u / 0.5;
    return {
      a: lerpAngle(g.aAcross, g.aSide, t),
      r: g.rAcross + (g.rSide - g.rAcross) * t,
    };
  }
  const t = (u - 0.5) / 0.5;
  return {
    a: lerpAngle(g.aSide, g.aUp, t),
    r: g.rSide + (g.rUp - g.rSide) * t,
  };
}

/** Largest distance along `a` from the shoulder that stays inside the
 *  margin box. Used to pull a point IN along its own direction rather
 *  than dropping a fruit whose direction runs off-canvas. */
function maxAlong(
  sx: number,
  sy: number,
  a: number,
  w: number,
  h: number,
  m: number,
): number {
  const dx = Math.cos(a);
  const dy = Math.sin(a);
  let t = Infinity;
  if (dx > 1e-6) t = Math.min(t, (w - m - sx) / dx);
  else if (dx < -1e-6) t = Math.min(t, (m - sx) / dx);
  if (dy > 1e-6) t = Math.min(t, (h - m - sy) / dy);
  else if (dy < -1e-6) t = Math.min(t, (m - sy) / dy);
  return t === Infinity ? 0 : Math.max(0, t);
}

export type Zone = "abduction" | "adduction";

/** Fan positions each zone draws from. The gap in the middle keeps the
 *  two zones visibly distinct instead of blurring into one arc. */
const ZONE_U: Record<Zone, [number, number]> = {
  adduction: [0.0, 0.42],
  abduction: [0.58, 1.0],
};

/** Fruit sits at this fraction of the reach in its direction. */
export const REACH_MIN = 0.7;
export const REACH_MAX = 1.0;
/** New fruit must be at least this fraction of the radius away from
 *  both the cursor and the previous fruit. */
export const MIN_SEPARATION = 0.35;
/** In the abduction zone nothing may sit below the shoulder by more
 *  than this fraction of the radius — reaching down is not the point. */
const BELOW_SHOULDER_ALLOW = 0.12;

export interface SpawnOpts {
  canvasW: number;
  canvasH: number;
  /** Keep fruit this many px clear of the canvas edge. */
  margin: number;
  cursor: { x: number; y: number } | null;
  prev: { x: number; y: number } | null;
  rand: () => number;
}

export interface SpawnResult {
  x: number;
  y: number;
  /** Direction from the shoulder in degrees, relative to straight out
   *  on the playing side: 0 = out, positive = toward overhead,
   *  negative = across the body. */
  angleDeg: number;
  /** Distance as a percentage of the reach in that direction. */
  reachPct: number;
  /** How many candidates were rejected before this one. */
  tries: number;
  clamped: boolean;
}

/**
 * Pick a point at the edge of reach, in the requested zone, honouring
 * the separation, canvas and below-shoulder constraints.
 *
 * Constraints are a preference, not a guarantee: after MAX_TRIES the
 * best candidate so far is returned. A round that cannot place a
 * perfect fruit should still place one.
 */
export function spawnAtReachEdge(
  g: ReachGeometry,
  zone: Zone,
  o: SpawnOpts,
): SpawnResult {
  const MAX_TRIES = 12;
  const [uLo, uHi] = ZONE_U[zone];
  let best: SpawnResult | null = null;
  let bestScore = -Infinity;

  for (let i = 0; i < MAX_TRIES; i++) {
    const u = uLo + o.rand() * (uHi - uLo);
    const { a, r } = fanAt(g, u);
    const frac = REACH_MIN + o.rand() * (REACH_MAX - REACH_MIN);
    let dist = r * frac;

    // Pull in along the same direction if it leaves the canvas.
    const limit = maxAlong(g.sx, g.sy, a, o.canvasW, o.canvasH, o.margin);
    const clamped = dist > limit;
    if (clamped) dist = limit;
    if (dist <= 0) continue;

    const x = g.sx + Math.cos(a) * dist;
    const y = g.sy + Math.sin(a) * dist;

    // Abduction stays at or above shoulder height — but never demands
    // higher than the patient actually demonstrated. If the "up" hold
    // came in below the shoulder (a short raise, or the top of frame
    // cropping the arm), the strict rule rejects the ENTIRE zone and
    // every spawn falls through to the fallback. Relaxing to the
    // measured up-hold height keeps the zone usable on a poor
    // calibration instead of silently collapsing it.
    if (zone === "abduction") {
      const upY = g.sy + Math.sin(g.aUp) * g.rUp;
      const yLimit = Math.max(g.sy + r * BELOW_SHOULDER_ALLOW, upY);
      if (y > yLimit) continue;
    }

    const sep = r * MIN_SEPARATION;
    const dCursor = o.cursor
      ? Math.hypot(x - o.cursor.x, y - o.cursor.y)
      : Infinity;
    const dPrev = o.prev ? Math.hypot(x - o.prev.x, y - o.prev.y) : Infinity;

    const cand: SpawnResult = {
      x,
      y,
      angleDeg: reportAngle(a, g),
      reachPct: r > 0 ? Math.round((dist / r) * 100) : 0,
      tries: i,
      clamped,
    };
    if (dCursor >= sep && dPrev >= sep) return cand;

    // Keep the roomiest near-miss, biased toward the outer edge.
    const score = Math.min(dCursor, dPrev) - sep + frac * 20;
    if (score > bestScore) {
      bestScore = score;
      best = cand;
    }
  }

  if (best) return best;

  // Everything was rejected (a degenerate fan). Fall back to the MIDDLE
  // OF THE REQUESTED ZONE, not straight out — a zone-blind fallback
  // turns every abduction spawn into an adduction one.
  const { a, r } = fanAt(g, (uLo + uHi) / 2);
  const limit = maxAlong(g.sx, g.sy, a, o.canvasW, o.canvasH, o.margin);
  const dist = Math.min(r * REACH_MIN, limit);
  return {
    x: g.sx + Math.cos(a) * dist,
    y: g.sy + Math.sin(a) * dist,
    angleDeg: reportAngle(a, g),
    reachPct: r > 0 ? Math.round((dist / r) * 100) : 0,
    tries: MAX_TRIES,
    clamped: true,
  };
}

/** Angle relative to "straight out on the playing side", so the number
 *  reads the same for a left- and a right-handed session. Canvas y
 *  grows downward, so the sign is flipped to make overhead positive. */
function reportAngle(a: number, g: ReachGeometry): number {
  let d = a - g.aSide;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return Math.round((-d * 180) / Math.PI);
}
