// Where Cloudburst's big strike lands.
//
// Pure geometry, no Phaser: the scene hands in the patient's reach
// range and the palm position and gets back a band. Keeping it here
// rather than in the scene means the three rules below can be checked
// directly, which matters because two of them are safety properties —
// a strike the patient cannot escape is not a test of anything, and a
// strike that lands where the last one did teaches them to stand still.
//
// All lengths are in NX: the cover-normalised x space the reach box and
// the falling items already use, so nothing here depends on the canvas
// size or on which way round the patient is standing.
//
// THE THREE RULES
//
//  1. Inside reach. The whole band sits within [loNx, hiNx].
//  2. Dodgeable. At least one side of the band has a full band-width of
//     free reach for the palm to move into. If the reach is too narrow
//     for that, the band is SHRUNK rather than made inescapable.
//  3. Different. The centre is at least one band-width from the
//     previous strike's centre.
//
// Rule 2 is the reason a centred band is no longer used: with the band
// at 40% of the reach, dead centre leaves only 30% of the reach on each
// side — less than the 40% the rule requires. The feasible centres are
// therefore everything EXCEPT a gap around the middle.

/** Share of strikes aimed at the palm rather than placed at random. */
export const STRIKE_AIMED_FRACTION = 0.5;

/** A closed interval of nx. */
export interface Span {
  lo: number;
  hi: number;
}

export interface BandChoice {
  centreNx: number;
  halfNx: number;
  /** True when this band was placed on the palm rather than at random. */
  aimed: boolean;
  /** Distance from the previous strike's centre, in nx, or null for the
   *  first strike of a round. */
  shiftNx: number | null;
  /** True when the band had to be narrowed to keep rule 2. */
  shrunk: boolean;
}

function len(s: Span): number {
  return Math.max(0, s.hi - s.lo);
}

/** Remove the open interval (c - r, c + r) from a set of spans. */
export function subtractBall(spans: Span[], c: number, r: number): Span[] {
  const out: Span[] = [];
  for (const s of spans) {
    const lo = c - r;
    const hi = c + r;
    if (hi <= s.lo || lo >= s.hi) {
      out.push(s);
      continue;
    }
    if (lo > s.lo) out.push({ lo: s.lo, hi: Math.min(s.hi, lo) });
    if (hi < s.hi) out.push({ lo: Math.max(s.lo, hi), hi: s.hi });
  }
  return out.filter((s) => len(s) > 1e-9);
}

/** Nearest point of the union of `spans` to `x`. */
export function clampToSpans(x: number, spans: Span[]): number {
  let best = x;
  let bestD = Infinity;
  for (const s of spans) {
    const p = Math.min(s.hi, Math.max(s.lo, x));
    const d = Math.abs(p - x);
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  return best;
}

/** Uniform sample over the union, weighted by span length — so a long
 *  span is proportionally more likely than a short one. */
export function sampleSpans(spans: Span[], rand: () => number): number {
  const total = spans.reduce((a, s) => a + len(s), 0);
  if (total <= 0) return spans.length > 0 ? spans[0].lo : 0;
  let t = rand() * total;
  for (const s of spans) {
    const l = len(s);
    if (t <= l) return s.lo + t;
    t -= l;
  }
  const last = spans[spans.length - 1];
  return last.hi;
}

/** The endpoint of `spans` furthest from `x`. Used when every rule
 *  cannot be satisfied at once and the best available compromise is
 *  simply "as far from last time as possible". */
function furthestFrom(x: number, spans: Span[]): number {
  let best = x;
  let bestD = -1;
  for (const s of spans) {
    for (const p of [s.lo, s.hi]) {
      const d = Math.abs(p - x);
      if (d > bestD) {
        bestD = d;
        best = p;
      }
    }
  }
  return best;
}

/**
 * Centres at which a band of half-width `half` is both inside reach and
 * dodgeable — at least `bandW` of free reach on one side.
 *
 * Returns one span when the reach is roomy enough that every position
 * works, and two when a gap around the middle has to be excluded.
 */
export function dodgeableCentres(
  loNx: number,
  hiNx: number,
  half: number,
): Span[] {
  const bandW = half * 2;
  // Inside reach at all.
  const minC = loNx + half;
  const maxC = hiNx - half;
  if (maxC <= minC) return [{ lo: (loNx + hiNx) / 2, hi: (loNx + hiNx) / 2 }];

  // Free space on the left is (centre - half) - loNx; requiring a full
  // band-width there puts the centre at or above `fromLeft`.
  const fromLeft = loNx + half + bandW;
  // Mirror image on the right.
  const toRight = hiNx - half - bandW;

  const left: Span = { lo: minC, hi: Math.min(toRight, maxC) };
  const right: Span = { lo: Math.max(fromLeft, minC), hi: maxC };
  // When the reach is wide relative to the band the two overlap and the
  // whole range is usable.
  if (left.hi >= right.lo) return [{ lo: minC, hi: maxC }];

  const out: Span[] = [];
  if (len(left) > 1e-9) out.push(left);
  if (len(right) > 1e-9) out.push(right);
  // Neither side can take a full band-width: the caller has already
  // shrunk the band as far as it will go, so fall back to the whole
  // inside-reach range rather than returning nothing.
  return out.length > 0 ? out : [{ lo: minC, hi: maxC }];
}

export function chooseStrikeBand(args: {
  /** The calibrated reach, in nx. */
  loNx: number;
  hiNx: number;
  /** Nominal band width as a share of the reach. */
  fraction: number;
  /** Palm position in nx, or null when the hand cannot be read. */
  palmNx: number | null;
  /** Previous strike's centre, or null for the first of a round. */
  prevCentreNx: number | null;
  aimedChance: number;
  rand: () => number;
}): BandChoice {
  const { loNx, hiNx, fraction, palmNx, prevCentreNx, aimedChance, rand } = args;
  const reach = Math.max(1e-6, hiNx - loNx);

  // Rule 2, the shrink guard. For one side to hold a full band-width the
  // band can be at most half the reach. With the shipped 0.4 fraction
  // this never binds; it is here so the invariant is enforced by the
  // code rather than by a comment about the current constants.
  const nominal = reach * fraction;
  const maxDodgeable = reach * 0.5;
  const bandW = Math.min(nominal, maxDodgeable * 0.98);
  const shrunk = bandW < nominal - 1e-9;
  const half = bandW / 2;

  const feasible = dodgeableCentres(loNx, hiNx, half);
  // Rule 3: never within one band-width of last time.
  const allowed = prevCentreNx === null
    ? feasible
    : subtractBall(feasible, prevCentreNx, bandW);

  const aimed = palmNx !== null && rand() < aimedChance;
  let centreNx: number;

  if (allowed.length === 0) {
    // Rules 2 and 3 cannot both hold — a very narrow reach. Dodgeable
    // wins, because it is the safety property; take the feasible point
    // furthest from last time and accept the repeat.
    centreNx = prevCentreNx === null
      ? sampleSpans(feasible, rand)
      : furthestFrom(prevCentreNx, feasible);
  } else if (aimed && palmNx !== null) {
    // "Aimed" means as close to the hand as the rules allow — clamping
    // into `allowed` can push the band off the palm, and that is
    // correct: a band centred on a hand with no room beside it would be
    // undodgeable, and one that repeats the last strike would teach the
    // patient to stand still.
    centreNx = clampToSpans(palmNx, allowed);
  } else {
    centreNx = sampleSpans(allowed, rand);
  }

  return {
    centreNx,
    halfNx: half,
    aimed: aimed && palmNx !== null,
    shiftNx: prevCentreNx === null ? null : Math.abs(centreNx - prevCentreNx),
    shrunk,
  };
}
