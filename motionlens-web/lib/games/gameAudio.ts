// Small WebAudio cue player for the game.
//
// Everything is SYNTHESISED — no audio files are downloaded or shipped.
// Two cues only: a bright blip when fruit is harvested, a dull thud
// when one is missed. Audio is a nicety, so every call is wrapped: a
// failure here must never interrupt the game loop.

type Ctor = typeof AudioContext;

export class GameAudio {
  private ctx: AudioContext | null = null;
  private out: GainNode | null = null;

  /** Must be called from a user gesture (a click) or the context stays
   *  suspended under the browser's autoplay policy. */
  prime(): void {
    try {
      if (!this.ctx) {
        const w = window as unknown as { webkitAudioContext?: Ctor };
        const Impl: Ctor | undefined = window.AudioContext ?? w.webkitAudioContext;
        if (!Impl) return;
        this.ctx = new Impl();
        this.out = this.ctx.createGain();
        this.out.gain.value = 0.85;
        this.out.connect(this.ctx.destination);
      }
      if (this.ctx.state === "suspended") void this.ctx.resume();
    } catch {
      this.ctx = null;
      this.out = null;
    }
  }

  /** Rising two-tone blip — fruit landed in the basket. */
  harvest(): void {
    this.blip(660, 990, 0.16, "triangle");
  }

  /** Low falling tone — fruit dropped away. */
  miss(): void {
    this.blip(240, 150, 0.22, "sine");
  }

  private blip(
    from: number,
    to: number,
    dur: number,
    type: OscillatorType,
  ): void {
    const ctx = this.ctx;
    const out = this.out;
    if (!ctx || !out) return;
    try {
      const t = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(from, t);
      osc.frequency.exponentialRampToValueAtTime(Math.max(40, to), t + dur);
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.5, t + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      osc.connect(gain);
      gain.connect(out);
      osc.start(t);
      osc.stop(t + dur + 0.02);
    } catch {
      // Ignore — a dropped cue is never worth breaking a round over.
    }
  }

  close(): void {
    try {
      void this.ctx?.close();
    } catch {
      // ignore
    }
    this.ctx = null;
    this.out = null;
  }
}
