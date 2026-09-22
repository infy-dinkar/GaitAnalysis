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
export type HoldStatus = "idle" | "holding" | "drifted" | "blocked";

/** Fallback text per status. A "blocked" hold carries its own reason —
 *  the pose is wrong, the wrong hand is up, the hand is out of view,
 *  the patient has stepped too close — so the caller supplies that. */
export const HOLD_MESSAGE: Record<HoldStatus, string> = {
  idle: "Get into position to start the timer",
  holding: "Hold still…",
  drifted: "Hold still",
  blocked: "",
};

export class HoldTracker {
  private samples: Sample[] = [];
  private anchor: Point | null = null;
  /** Time actually spent holding. Accumulated rather than taken from a
   *  wall-clock start, so a pause can stop the clock without losing
   *  the progress already earned. */
  private heldMs = 0;
  private lastFeedMs = 0;

  /** 0..1 ring fill. */
  progress = 0;
  /** True while the wrist is inside tolerance and the ring is filling. */
  holding = false;
  /** Why the ring is doing what it is doing. The ring used to reset
   *  silently when the wrist left the frame, which is indistinguishable
   *  to the patient from "hold stiller" — so they lower the arm until
   *  it fills, and the recorded reach is wrong. */
  status: HoldStatus = "idle";

  reset(): void {
    this.samples = [];
    this.anchor = null;
    this.heldMs = 0;
    this.lastFeedMs = 0;
    this.progress = 0;
    this.holding = false;
    this.status = "idle";
    this.blockReason = "";
  }

  /**
   * @returns the recorded point when the hold completes, else null.
   */
  /** Why the ring is currently held, when status is "blocked". */
  blockReason = "";

  /**
   * @param block non-null to hold the ring, carrying the reason to
   *              show the patient. Null to let it fill.
   */
  feed(nx: number, ny: number, block: string | null, nowMs: number): Point | null {
    const prev = this.lastFeedMs;
    this.lastFeedMs = nowMs;
    // Clamp so a backgrounded tab or a long stall cannot award seconds
    // of hold time in a single frame.
    const dt = prev > 0 ? Math.max(0, Math.min(200, nowMs - prev)) : 0;

    // Held for any reason — out of view, wrong pose, wrong hand, too
    // close. PAUSE rather than reset: the wrist leaving the top of the
    // frame is exactly what happens when the arm goes properly up, and
    // throwing the progress away there is what taught patients to
    // lower the arm until the ring filled.
    if (block !== null) {
      this.status = "blocked";
      this.blockReason = block;
      this.holding = false;
      return null;
    }
    this.blockReason = "";

    if (!this.anchor) {
      this.anchor = { nx, ny };
      this.heldMs = 0;
      this.samples = [{ nx, ny, t: nowMs }];
      this.progress = 0;
      this.holding = true;
      this.status = "holding";
      return null;
    }

    const drift = Math.hypot(nx - this.anchor.nx, ny - this.anchor.ny);
    if (drift > STILL_TOLERANCE) {
      // Moved too far — restart the ring from the new position.
      this.anchor = { nx, ny };
      this.heldMs = 0;
      this.samples = [{ nx, ny, t: nowMs }];
      this.progress = 0;
      this.holding = true;
      this.status = "drifted";
      return null;
    }

    this.samples.push({ nx, ny, t: nowMs });
    this.holding = true;
    this.status = "holding";
    this.heldMs += dt;
    this.progress = Math.min(1, this.heldMs / HOLD_MS);

    if (this.heldMs < HOLD_MS) return null;

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

/**
 * Sanity check on the "arm up" hold.
 *
 * It is the same arm in every direction, so the overhead reach should
 * be close to the sideways reach. A much shorter "up" radius means the
 * arm never actually went up — almost always because the raised wrist
 * left the top of the frame, the ring refused to fill, and the patient
 * lowered the arm until it did.
 */
export const UP_REACH_TOLERANCE = 0.25;

export function upReachLooksShort(
  up: Point,
  side: Point,
  shoulderX: number,
  shoulderY: number,
  cover: Cover,
): { short: boolean; rUp: number; rSide: number; ratio: number } {
  const px = (p: Point) => ({
    x: cover.offX + p.nx * cover.dispW,
    y: cover.offY + p.ny * cover.dispH,
  });
  const u = px(up);
  const s = px(side);
  const rUp = Math.hypot(u.x - shoulderX, u.y - shoulderY);
  const rSide = Math.hypot(s.x - shoulderX, s.y - shoulderY);
  const ratio = rSide > 1 ? rUp / rSide : 0;
  return { short: ratio < 1 - UP_REACH_TOLERANCE, rUp, rSide, ratio };
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

// ── Pose validation per hold ──────────────────────────────────────
//
// Stillness alone is not a hold. A hand hanging at the patient's side
// is perfectly still, so all three holds could be — and were —
// recorded with the arm down, giving a reach box across the bottom of
// the frame and fruit that spawned by the basket.
//
// Each hold now has to be IN THE POSE before the ring fills. The
// thresholds are fractions of the arm measured at setup, so they hold
// for any body size and any distance from the camera.

/** Wrist must clear the shoulder by this much of an arm length. */
const POSE_REACH = 0.6;
/** How far off shoulder height the SIDE hold may sit. */
const SIDE_HEIGHT_TOL = 0.35;

export interface PoseGeom {
  /** Wrist being judged, canvas px. */
  wx: number;
  wy: number;
  /** That wrist's own shoulder, canvas px. */
  sx: number;
  sy: number;
  /** Body midline, canvas px. */
  midX: number;
  /** Arm length, canvas px. */
  arm: number;
  /**
   * Which side of the body this arm is on, in SCREEN terms. The view
   * is mirrored, so the patient's right arm is on screen-right: +1 for
   * a right arm, -1 for a left one.
   */
  dir: 1 | -1;
}

export interface PoseVerdict {
  ok: boolean;
  /** What to tell the patient. Empty when ok. */
  message: string;
  /** Signed reach along the axis the hold cares about, in arm lengths.
   *  Reported on the debug overlay so a near-miss is visible. */
  reach: number;
}

/**
 * Is this wrist in the pose for `hold`?
 *
 * Canvas y grows downward, so "above the shoulder" is a NEGATIVE dy.
 */
export function checkHoldPose(hold: HoldId, g: PoseGeom): PoseVerdict {
  if (g.arm < 1) return { ok: false, message: "Line up with the camera", reach: 0 };

  if (hold === "up") {
    const above = (g.sy - g.wy) / g.arm;
    return above >= POSE_REACH
      ? { ok: true, message: "", reach: above }
      : { ok: false, message: "Raise your arm higher", reach: above };
  }

  if (hold === "side") {
    const out = ((g.wx - g.sx) * g.dir) / g.arm;
    const drop = Math.abs(g.wy - g.sy) / g.arm;
    if (out < POSE_REACH) {
      return { ok: false, message: "Stretch your arm out to the side", reach: out };
    }
    if (drop > SIDE_HEIGHT_TOL) {
      return { ok: false, message: "Keep your arm at shoulder level", reach: out };
    }
    return { ok: true, message: "", reach: out };
  }

  // across — the wrist must have crossed the midline toward the far
  // side. Distance past it is deliberately small: crossing at all is
  // the adduction we are measuring.
  const past = ((g.midX - g.wx) * g.dir) / g.arm;
  return past > 0
    ? { ok: true, message: "", reach: past }
    : { ok: false, message: "Bring your arm across your body", reach: past };
}

/** Screen-space side of the body for an arm, under the display mirror:
 *  the patient's right arm appears on screen-right. */
export function screenDir(hand: Hand): 1 | -1 {
  return hand === "right" ? 1 : -1;
}

/**
 * After the fact: does a RECORDED point actually satisfy its hold?
 *
 * The same rules, applied to the stored normalised point once the hold
 * has completed, so a hold that slipped through cannot poison the
 * reach box. Returns null when the point is acceptable.
 */
export function recordedHoldFails(
  hold: HoldId,
  point: Point,
  hand: Hand,
  shoulderX: number,
  shoulderY: number,
  midX: number,
  arm: number,
  cover: Cover,
): string | null {
  if (cover.dispW <= 0 || arm < 1) return null;
  const v = checkHoldPose(hold, {
    wx: cover.offX + point.nx * cover.dispW,
    wy: cover.offY + point.ny * cover.dispH,
    sx: shoulderX,
    sy: shoulderY,
    midX,
    arm,
    dir: screenDir(hand),
  });
  return v.ok ? null : v.message;
}
