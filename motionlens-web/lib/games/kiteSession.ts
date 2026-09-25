// The scoring half of a Kite Flying round.
//
// Pulled out of the scene so it can be driven by a scripted path
// without a camera or a canvas — which is the only way to actually
// SHOW that the Normal/Large display setting changes nothing about the
// game, rather than asserting it. Nothing here takes a display scale,
// and that absence is the guarantee.
//
// TWO POINTS, TWO PURPOSES, ON PURPOSE:
//
//   • `kite` is the DRAWN anchor. It drives the feedback — the glow,
//     the wobble, the fall — because what the patient sees has to be
//     what the game responds to.
//   • `palm` is the RAW anchor. It drives every saved number, because
//     the drawn one is smoothed and the smoothing would flatter them.
//
// They differ by a few pixels of filter lag, and that is correct.

import {
  deviationRatio,
  factorAt,
  halfNyAt,
  isNarrowAt,
  type Corridor,
} from "@/lib/games/kiteCorridor";

export interface KitePoint {
  worldX: number;
  ny: number;
}

export interface KiteFrame {
  dtMs: number;
  half: 1 | 2;
  /** The drawn anchor, or null while the kite is away (falling). */
  kite: KitePoint | null;
  /** The raw anchor, or null when the hand cannot be measured. */
  palm: KitePoint | null;
}

export interface KiteStep {
  kiteInside: boolean;
  /** True on the frame the kite ran out of grace and fell. */
  fellNow: boolean;
  /** Local geometry at the kite, for the overlay. */
  localHalfNy: number;
  factor: number;
  narrow: boolean;
  /** Deviation of the RAW palm, as a multiple of the local half-width. */
  devRatio: number;
}

/** Everything the round accumulates. Times are in ms. */
export interface KiteTotals {
  insideMs: number;
  measuredMs: number;
  devSum: number;
  devCount: number;
  falls: number;
  firstHalf: { inside: number; total: number };
  secondHalf: { inside: number; total: number };
  narrowInsideMs: number;
  narrowMs: number;
  wideInsideMs: number;
  wideMs: number;
}

export function blankTotals(): KiteTotals {
  return {
    insideMs: 0,
    measuredMs: 0,
    devSum: 0,
    devCount: 0,
    falls: 0,
    firstHalf: { inside: 0, total: 0 },
    secondHalf: { inside: 0, total: 0 },
    narrowInsideMs: 0,
    narrowMs: 0,
    wideInsideMs: 0,
    wideMs: 0,
  };
}

export class KiteSession {
  readonly totals: KiteTotals = blankTotals();
  private corridor: Corridor;
  private fallAfterMs: number;
  /** Continuous time the kite has been outside. */
  private outsideMs = 0;
  private away = false;

  constructor(corridor: Corridor, fallAfterMs: number) {
    this.corridor = corridor;
    this.fallAfterMs = fallAfterMs;
  }

  /** The kite has tumbled and is not steerable; stop the grace clock. */
  setAway(away: boolean): void {
    this.away = away;
    if (away) this.outsideMs = 0;
  }

  get outsideForMs(): number {
    return this.outsideMs;
  }

  step(f: KiteFrame): KiteStep {
    const c = this.corridor;
    const out: KiteStep = {
      kiteInside: false,
      fellNow: false,
      localHalfNy: 0,
      factor: 0,
      narrow: false,
      devRatio: 0,
    };

    // ── Feedback, from the drawn anchor.
    if (f.kite && !this.away) {
      out.localHalfNy = halfNyAt(c, f.kite.worldX);
      out.factor = factorAt(c, f.kite.worldX);
      out.narrow = isNarrowAt(c, f.kite.worldX);
      out.kiteInside = deviationRatio(c, f.kite.worldX, f.kite.ny) <= 1;
      if (out.kiteInside) {
        this.outsideMs = 0;
      } else {
        this.outsideMs += f.dtMs;
        if (this.outsideMs >= this.fallAfterMs) {
          out.fellNow = true;
          this.totals.falls += 1;
          this.outsideMs = 0;
        }
      }
    }

    // ── Scoring, from the raw anchor.
    if (f.palm) {
      const dev = deviationRatio(c, f.palm.worldX, f.palm.ny);
      const inside = dev <= 1;
      out.devRatio = dev;

      const t = this.totals;
      t.measuredMs += f.dtMs;
      t.devSum += dev;
      t.devCount += 1;
      if (inside) t.insideMs += f.dtMs;

      const h = f.half === 1 ? t.firstHalf : t.secondHalf;
      h.total += f.dtMs;
      if (inside) h.inside += f.dtMs;

      // Narrow vs wide is judged where the PALM is, not where the kite
      // is, so it lines up with the deviation it is explaining.
      if (isNarrowAt(c, f.palm.worldX)) {
        t.narrowMs += f.dtMs;
        if (inside) t.narrowInsideMs += f.dtMs;
      } else {
        t.wideMs += f.dtMs;
        if (inside) t.wideInsideMs += f.dtMs;
      }
    }

    return out;
  }
}
