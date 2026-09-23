// Orchard wildlife — decoration only.
//
// A monkey that turns up on a branch now and then, and birds drifting
// across the sky. They exist to make the orchard feel alive between
// reaches; they are NOT part of the game.
//
// Three rules keep them decoration:
//
//   1. NO HIT DETECTION. Nothing here is ever added to the scene's
//      fruit list, so the hit test cannot see it. A hand passing over
//      the monkey does nothing at all.
//   2. BEHIND THE FRUIT, always. Depths sit below the fruit (10) and
//      its sprig (9), above the backdrop (-10).
//   3. CALM. Slow tweens, long durations, nothing sudden, and the
//      monkey only ever appears in the upper canopy well away from
//      where fruit spawn. These patients are elderly; a shape darting
//      past the hand would be startling, and worse, it would pull the
//      eye away from the fruit at exactly the wrong moment.
//
// They are also deliberately smaller and dimmer than fruit, so at 2 m
// the bright saturated fruit still wins every time.

import Phaser from "phaser";

/** Depth band: above the backdrop, below fruit and sprigs. */
const DEPTH_BIRD = 2;
const DEPTH_MONKEY = 3;

/** Dimmed so the eye is not drawn to them. */
const ANIMAL_ALPHA = 0.55;
const ANIMAL_TINT = 0xbfc9bf;

/** Size as a fraction of the canvas unit — smaller than fruit (0.13). */
const MONKEY_FRACTION = 0.085;
const BIRD_FRACTION = 0.045;

/** Gaps between appearances, in ms. Deliberately long: this is a
 *  background event, not a feature. */
const MONKEY_FIRST_MIN = 6000;
const MONKEY_FIRST_MAX = 14000;
const MONKEY_GAP_MIN = 16000;
const MONKEY_GAP_MAX = 28000;
const BIRD_FIRST_MIN = 3000;
const BIRD_FIRST_MAX = 9000;
const BIRD_GAP_MIN = 9000;
const BIRD_GAP_MAX = 20000;

/** How long the monkey sits before climbing away. */
const MONKEY_SIT_MS = 4200;
/** A bird's crossing time. Slow — it should drift, not dart. */
const BIRD_CROSS_MIN_MS = 9000;
const BIRD_CROSS_MAX_MS = 14000;

export interface AnimalTextures {
  /** Null when the glyph failed to rasterise — that animal is skipped
   *  rather than drawn as a blank square. */
  monkey: string | null;
  bird: string | null;
}

/**
 * Owns the orchard's wildlife for one round.
 *
 * `update(timeMs)` is cheap: two timestamp comparisons. Everything
 * else is tween-driven, so there is no per-frame work per animal.
 */
export class BackgroundLife {
  private scene: Phaser.Scene;
  private tex: AnimalTextures;
  private nextMonkeyAt: number;
  private nextBirdAt: number;
  private monkeyAlive = false;
  private alive: Phaser.GameObjects.Image[] = [];
  private started = false;

  constructor(scene: Phaser.Scene, tex: AnimalTextures) {
    this.scene = scene;
    this.tex = tex;
    this.nextMonkeyAt = -1;
    this.nextBirdAt = -1;
  }

  private rand(lo: number, hi: number): number {
    return lo + Math.random() * (hi - lo);
  }

  private get unit(): number {
    return Math.min(this.scene.scale.width, this.scene.scale.height);
  }

  update(time: number): void {
    // Seed the first appearances from the first frame's clock, so the
    // schedule is relative to the round rather than to page load.
    if (!this.started) {
      this.started = true;
      this.nextMonkeyAt = time + this.rand(MONKEY_FIRST_MIN, MONKEY_FIRST_MAX);
      this.nextBirdAt = time + this.rand(BIRD_FIRST_MIN, BIRD_FIRST_MAX);
      return;
    }
    if (this.tex.monkey && !this.monkeyAlive && time >= this.nextMonkeyAt) {
      this.nextMonkeyAt = time + this.rand(MONKEY_GAP_MIN, MONKEY_GAP_MAX);
      this.spawnMonkey();
    }
    if (this.tex.bird && time >= this.nextBirdAt) {
      this.nextBirdAt = time + this.rand(BIRD_GAP_MIN, BIRD_GAP_MAX);
      this.spawnBird();
    }
  }

  /** One monkey at a time: arrives, settles, looks about, climbs off. */
  private spawnMonkey(): void {
    const key = this.tex.monkey;
    if (!key) return;
    const u = this.unit;
    const w = this.scene.scale.width;
    const h = this.scene.scale.height;
    const size = u * MONKEY_FRACTION;

    // Upper canopy, and off to one side — never the middle, where the
    // fruit and the patient's attention are.
    const onLeft = Math.random() < 0.5;
    const x = onLeft ? this.rand(w * 0.08, w * 0.24) : this.rand(w * 0.76, w * 0.92);
    const sitY = this.rand(h * 0.18, h * 0.34);

    const m = this.scene.add
      .image(x, sitY - u * 0.12, key)
      .setDisplaySize(size, size)
      .setAlpha(0)
      .setTint(ANIMAL_TINT)
      .setDepth(DEPTH_MONKEY);
    if (!onLeft) m.setFlipX(true);
    this.monkeyAlive = true;
    this.alive.push(m);

    const done = () => {
      this.monkeyAlive = false;
      this.alive = this.alive.filter((o) => o !== m);
      m.destroy();
    };

    // Climb down into view.
    this.scene.tweens.add({
      targets: m,
      y: sitY,
      alpha: ANIMAL_ALPHA,
      duration: 1600,
      ease: "Sine.easeOut",
      onComplete: () => {
        // Sit and look around — a slow head-tilt, nothing more.
        this.scene.tweens.add({
          targets: m,
          angle: onLeft ? 8 : -8,
          duration: 900,
          yoyo: true,
          repeat: 1,
          ease: "Sine.easeInOut",
        });
        // Then climb away.
        this.scene.tweens.add({
          targets: m,
          y: sitY - u * 0.16,
          alpha: 0,
          duration: 1800,
          delay: MONKEY_SIT_MS,
          ease: "Sine.easeIn",
          onComplete: done,
        });
      },
    });
  }

  /** A bird drifting across the sky, well above the fruit. */
  private spawnBird(): void {
    const key = this.tex.bird;
    if (!key) return;
    const u = this.unit;
    const w = this.scene.scale.width;
    const h = this.scene.scale.height;
    const size = u * BIRD_FRACTION;

    const leftToRight = Math.random() < 0.5;
    const y = this.rand(h * 0.06, h * 0.2);
    const from = leftToRight ? -size * 2 : w + size * 2;
    const to = leftToRight ? w + size * 2 : -size * 2;

    const b = this.scene.add
      .image(from, y, key)
      .setDisplaySize(size, size)
      .setAlpha(ANIMAL_ALPHA * 0.8)
      .setTint(ANIMAL_TINT)
      .setDepth(DEPTH_BIRD);
    if (!leftToRight) b.setFlipX(true);
    this.alive.push(b);

    const dur = this.rand(BIRD_CROSS_MIN_MS, BIRD_CROSS_MAX_MS);
    this.scene.tweens.add({
      targets: b,
      x: to,
      duration: dur,
      ease: "Linear",
      onComplete: () => {
        this.alive = this.alive.filter((o) => o !== b);
        b.destroy();
      },
    });
    // A gentle rise and fall along the way — a glide, not a flap.
    this.scene.tweens.add({
      targets: b,
      y: y + u * 0.03,
      duration: dur / 3,
      yoyo: true,
      repeat: 1,
      ease: "Sine.easeInOut",
    });
  }

  /** Called when the round ends, so nothing outlives the scene. */
  destroy(): void {
    for (const o of this.alive) {
      this.scene.tweens.killTweensOf(o);
      o.destroy();
    }
    this.alive = [];
    this.monkeyAlive = false;
  }
}
