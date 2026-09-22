// One-euro filter — adaptive low-pass used to smooth the hand cursor.
//
// THE DERIVATIVE MUST COME FROM THE PREVIOUS **RAW** INPUT.
//
// An earlier prototype kept a single `xPrev` and set it to the filter's
// own output. The velocity estimate then became
//   dx = (raw - filtered) / dt
// which measures the filter's own lag rather than the hand's speed. At
// rest that inflates the cutoff and quietly defeats the smoothing the
// caller asked for. The canonical filter keeps two separate histories,
// which is what `xRawPrev` / `xHatPrev` are for below. Do not merge them.

export interface OneEuroOptions {
  /** Cutoff in Hz at zero speed. Lower = smoother but laggier. */
  minCutoff?: number;
  /** How much the cutoff opens up with speed. Higher = less lag. */
  beta?: number;
  /** Cutoff for the velocity estimate itself. */
  dCutoff?: number;
}

export class OneEuro {
  private readonly minCutoff: number;
  private readonly beta: number;
  private readonly dCutoff: number;

  /** Previous RAW input — the derivative is taken against this. */
  private xRawPrev: number | null = null;
  /** Previous FILTERED output — the low-pass carries this forward. */
  private xHatPrev = 0;
  private dxPrev = 0;

  /** Cutoff actually applied on the last frame, in Hz. Diagnostic only. */
  lastCutoff = 0;

  /** Low-passed velocity estimate in units/second, as used to open the
   *  cutoff. Exposed so callers can extrapolate with the SAME velocity
   *  the filter is already computing rather than differencing the
   *  output again (which would re-introduce the noise this removes). */
  get velocity(): number {
    return this.dxPrev;
  }

  constructor(opts: OneEuroOptions = {}) {
    this.minCutoff = opts.minCutoff ?? 1.1;
    this.beta = opts.beta ?? 0.02;
    this.dCutoff = opts.dCutoff ?? 1;
  }

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  /** Drop all history — call when the tracked point teleports. */
  reset(): void {
    this.xRawPrev = null;
    this.xHatPrev = 0;
    this.dxPrev = 0;
    this.lastCutoff = 0;
  }

  filter(x: number, dt: number): number {
    if (this.xRawPrev === null || dt <= 0) {
      this.xRawPrev = x;
      this.xHatPrev = x;
      this.lastCutoff = this.minCutoff;
      return x;
    }
    const dx = (x - this.xRawPrev) / dt;
    const ad = this.alpha(this.dCutoff, dt);
    const edx = ad * dx + (1 - ad) * this.dxPrev;
    this.dxPrev = edx;

    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    this.lastCutoff = cutoff;
    const a = this.alpha(cutoff, dt);
    const out = a * x + (1 - a) * this.xHatPrev;

    this.xRawPrev = x;
    this.xHatPrev = out;
    return out;
  }
}

/** Convenience pair for filtering a 2-D point with one call. */
export class OneEuro2D {
  private readonly fx: OneEuro;
  private readonly fy: OneEuro;

  constructor(opts: OneEuroOptions = {}) {
    this.fx = new OneEuro(opts);
    this.fy = new OneEuro(opts);
  }

  reset(): void {
    this.fx.reset();
    this.fy.reset();
  }

  filter(x: number, y: number, dt: number): { x: number; y: number } {
    return { x: this.fx.filter(x, dt), y: this.fy.filter(y, dt) };
  }

  /** Effective cutoff actually applied, in Hz — the larger of the two
   *  axes, which is the one doing the least smoothing. */
  get lastCutoff(): number {
    return Math.max(this.fx.lastCutoff, this.fy.lastCutoff);
  }

  /** Low-passed velocity in px/second on each axis. */
  get velocity(): { vx: number; vy: number } {
    return { vx: this.fx.velocity, vy: this.fy.velocity };
  }
}
