// How smooth the patient's hand was.
//
// This is the part of Kite Flying that has to be right, so it is a
// plain class with no Phaser in it and can be driven by a synthetic
// trace without a camera. It was built wrong twice before this version
// and both faults were only visible because it can be: see the notes
// on rate and on filtering below.
//
// ── IT MEASURES THE RAW PALM, NOT THE DRAWN CURSOR ──
//
// The cursor the patient sees is deliberately smoothed: a one-euro
// filter plus a short velocity extrapolation (lib/games/gameSceneBase.ts).
// That is right for the game — an unfiltered cursor is unpleasant to
// aim — and completely wrong for measurement, because the filter's job
// is to remove exactly the jerkiness this file exists to count. Feeding
// the drawn cursor in here would report every patient as smooth.
//
// ── IT RESAMPLES DOWN, NEVER UP ──
//
// The pose detector delivers roughly 20 Hz, irregularly, while the
// render loop runs at 60. Sampling once per rendered frame repeats each
// pose two or three times and manufactures stretches of exactly-zero
// speed. Resampling to a rate ABOVE the source is no better: linear
// interpolation between 20 Hz observations onto a 30 Hz grid beats the
// two rates against each other and produces a sawtooth in the speed
// profile that is pure artefact. A first attempt did exactly that and
// counted more peaks for a smooth sine than for a deliberately jerky
// staircase.
//
// So: new observations are recorded only when the value actually
// changes, output is emitted on a fixed clock at or below the source
// rate, and a sample is only ever emitted for a time that lies BETWEEN
// two real observations — never extrapolated past the newest one.
//
// ── IT LOW-PASSES THE SPEED, NOT THE POSITION ──
//
// Counting turning points in an unfiltered speed profile counts tracker
// noise: a hand held still with a couple of pixels of jitter produced
// more "peaks" than either real movement in testing. Smoothing the
// SPEED before counting is what the movement-units literature does, and
// it is not the same as smoothing the position — the position stays
// raw, so a real correction is still a real correction.
//
// ── UNITS ──
//
// Positions are divided by the live arm length, so speed is in ARM
// LENGTHS PER SECOND. A tall patient and a short one standing at
// different distances from the camera then produce comparable numbers,
// which raw pixels never would.

/** Fixed output rate of the resampler, in Hz. At the pose rate, not
 *  above it — see the note on resampling. */
export const RESAMPLE_HZ = 20;
const RESAMPLE_MS = 1000 / RESAMPLE_HZ;

/**
 * Moving average applied to the SPEED profile before peak counting.
 *
 * Three taps at 20 Hz is a 150 ms window, first null at 6.7 Hz. Five
 * taps (250 ms) was tried first and was too wide: voluntary
 * corrections in a tracking task come 2-3 per second, and a 250 ms
 * window flattened them below the prominence threshold, scoring a
 * deliberately jerky trace as smooth as a sine.
 */
export const SPEED_TAPS = 3;

/**
 * Speed below which the hand counts as still, in arm lengths per
 * second. Peaks under this are noise, not movement, and the time is
 * excluded from the "per second of movement" denominator.
 */
export const PEAK_MIN_SPEED = 0.2;

/**
 * How far the smoothed speed must fall after a peak (and rise after a
 * trough) before the next turning point counts, in arm lengths per
 * second.
 *
 * This prominence test is what makes the count mean something: a smooth
 * reach ridden by a little jitter has one peak, not thirty.
 */
export const PEAK_PROMINENCE = 0.12;

/** Peaks per second of movement that map to a smoothness of 100 and
 *  of 0. See smoothnessScore(). */
export const SMOOTH_BEST_PPS = 1.0;
export const SMOOTH_WORST_PPS = 6.0;

/**
 * Below this much actual movement the measure has no basis and is
 * reported as null rather than as a number.
 *
 * A hand held still for a whole round produces a handful of noise
 * peaks over a fraction of a second of "movement", which divides out
 * to a huge and entirely meaningless peaks-per-second. Refusing to
 * answer is the honest result; a smoothness of 0 would read as
 * "severely impaired" when it means "did not play".
 */
export const MIN_MOVING_SEC = 3;

/**
 * 0–100, higher is smoother.
 *
 *   score = 100 * (WORST - peaksPerSec) / (WORST - BEST), clamped
 *         = 100 * (6.0 - peaksPerSec) / 5.0
 *
 * A straight linear map from velocity peaks per second of movement:
 * 1.0 peaks/s or fewer scores 100, 6.0 or more scores 0, 3.5 scores 50.
 *
 * Deliberately simple and deliberately linear, because the underlying
 * measure — the number of peaks in the speed profile, sometimes called
 * movement units — is the thing with meaning, and any curve applied on
 * top would be decoration presented as precision.
 *
 * This is an ORDINAL score for comparing a patient with themselves
 * across sessions. It is not a validated clinical instrument, and both
 * endpoints were chosen by judgement rather than from a normative
 * sample.
 */
export function smoothnessScore(peaksPerSec: number): number {
  const t = (SMOOTH_WORST_PPS - peaksPerSec) / (SMOOTH_WORST_PPS - SMOOTH_BEST_PPS);
  return Math.round(Math.min(1, Math.max(0, t)) * 100);
}

interface Knot {
  t: number;
  x: number;
  y: number;
}

/**
 * Feed it raw palm positions every frame; ask it for the answer at the
 * end.
 *
 * `push` is safe to call at any rate and with repeated values — only a
 * changed position starts a new knot, and output is emitted on the
 * fixed clock regardless.
 */
export class MovementAnalyser {
  /** The two most recent real observations, for interpolation. Their
   *  timestamps are never edited: a moving endpoint would slide the
   *  interpolation and put a wobble into the speed profile. */
  private prevKnot: Knot | null = null;
  private lastKnot: Knot | null = null;
  /** Next scheduled output time, on the caller's clock. */
  private nextOutAt = -1;
  /** Previous emitted sample, for the speed difference. */
  private outPrev: { x: number; y: number } | null = null;

  /** Ring buffer for the speed low-pass. */
  private taps: number[] = [];
  private tapSum = 0;

  // ── Peak detector. A standard turning-point counter with a
  //    prominence test: a peak is banked only once the smoothed speed
  //    has fallen PEAK_PROMINENCE below it.
  private rising = true;
  private ref = 0;
  private started = false;
  private peaks = 0;
  private movingMs = 0;
  private totalMs = 0;
  private speedSum = 0;
  private speedCount = 0;

  /** Running count, for the debug overlay. */
  get peakCount(): number {
    return this.peaks;
  }

  get movingSec(): number {
    return this.movingMs / 1000;
  }

  /**
   * Peaks per second OF MOVEMENT, or null when the patient barely
   * moved — time the hand was still is not in the denominator, or
   * someone who rested would score as smooth.
   */
  get peaksPerSec(): number | null {
    const sec = this.movingMs / 1000;
    if (sec < MIN_MOVING_SEC) return null;
    return this.peaks / sec;
  }

  get meanSpeed(): number {
    return this.speedCount > 0 ? this.speedSum / this.speedCount : 0;
  }

  get sampledSec(): number {
    return this.totalMs / 1000;
  }

  /**
   * One raw palm observation.
   *
   * @param t   update()'s own clock, ms
   * @param x   raw palm x, canvas px (NOT the drawn cursor)
   * @param y   raw palm y, canvas px
   * @param arm live arm length in px, for the scale-free conversion
   */
  push(t: number, x: number, y: number, arm: number): void {
    if (!(arm > 1) || !Number.isFinite(x) || !Number.isFinite(y)) return;

    if (this.lastKnot === null) {
      this.lastKnot = { t, x, y };
      this.nextOutAt = t;
      return;
    }
    // A repeated position means the detector has not reported since the
    // last frame — not new data, and the existing knot's timestamp is
    // left exactly where it was.
    if (x !== this.lastKnot.x || y !== this.lastKnot.y) {
      this.prevKnot = this.lastKnot;
      this.lastKnot = { t, x, y };
    }

    // Emit only for times that fall BETWEEN two real observations.
    // Never past the newest knot, so nothing is ever extrapolated.
    const until = this.lastKnot.t;
    let guard = 0;
    while (this.nextOutAt <= until && guard++ < 64) {
      this.emit(this.nextOutAt, arm);
      this.nextOutAt += RESAMPLE_MS;
    }
  }

  /** Linear interpolation of the raw trace at `t`, then one step of the
   *  speed profile. */
  private emit(t: number, arm: number): void {
    const a = this.prevKnot;
    const b = this.lastKnot;
    if (!b) return;
    let x = b.x;
    let y = b.y;
    if (a && b.t > a.t) {
      const u = Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t)));
      x = a.x + (b.x - a.x) * u;
      y = a.y + (b.y - a.y) * u;
    }

    const prev = this.outPrev;
    this.outPrev = { x, y };
    if (!prev) return;

    // Arm lengths per second.
    const raw = (Math.hypot(x - prev.x, y - prev.y) / arm) * RESAMPLE_HZ;

    // Low-pass the SPEED. The position above stays raw.
    this.taps.push(raw);
    this.tapSum += raw;
    if (this.taps.length > SPEED_TAPS) this.tapSum -= this.taps.shift() as number;
    // Wait for a full window so the first few samples are not counted
    // against a half-filled average.
    if (this.taps.length < SPEED_TAPS) return;
    const speed = this.tapSum / SPEED_TAPS;

    this.totalMs += RESAMPLE_MS;
    this.speedSum += speed;
    this.speedCount += 1;
    if (speed >= PEAK_MIN_SPEED) this.movingMs += RESAMPLE_MS;

    if (!this.started) {
      this.started = true;
      this.ref = speed;
      return;
    }

    // Turning-point count with a prominence test.
    if (this.rising) {
      if (speed > this.ref) {
        this.ref = speed;
      } else if (this.ref - speed >= PEAK_PROMINENCE) {
        if (this.ref >= PEAK_MIN_SPEED) this.peaks += 1;
        this.rising = false;
        this.ref = speed;
      }
    } else {
      if (speed < this.ref) {
        this.ref = speed;
      } else if (speed - this.ref >= PEAK_PROMINENCE) {
        this.rising = true;
        this.ref = speed;
      }
    }
  }

  /** The hand went away, or the kite is falling and the patient has
   *  nothing to steer. Break the trace so the gap is not read as one
   *  enormous movement. */
  breakTrace(): void {
    this.prevKnot = null;
    this.lastKnot = null;
    this.outPrev = null;
    this.nextOutAt = -1;
    this.taps = [];
    this.tapSum = 0;
    this.started = false;
  }
}
