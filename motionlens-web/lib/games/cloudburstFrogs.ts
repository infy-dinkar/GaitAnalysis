// Frogs on Cloudburst's ground strip.
//
// Decoration only. They have no hit test, they are never given to the
// item list, and they live strictly BELOW the catch area — the miss
// line is clamped to the ground line (SKY_GROUND) and these sit below
// that, so a frog can never be mistaken for something to reach for.
//
// Same arrangement as backgroundLife.ts: the scene rasterises the glyph
// and passes the key (or null if the platform has no frog, in which
// case there simply are no frogs), and update() is two timestamp
// comparisons. Everything else is tween-driven, so there is no
// per-frame cost per frog.

import Phaser from "phaser";
import { SKY_GROUND } from "@/lib/games/skyScene";

/** Below the items (depth 9/10) and above the sky and clouds. */
const DEPTH_FROG = 4;

/** Muted, so they read as part of the scenery rather than as targets.
 *  The items are saturated and haloed; these deliberately are not. */
const FROG_ALPHA = 0.72;
const FROG_TINT = 0xa8c4a0;

/** Size as a fraction of the canvas unit. Well under an item's 0.12. */
const FROG_FRACTION = 0.055;

/** Vertical band they sit in, as a fraction of canvas height. Starts
 *  below the ground line so nothing overlaps the play area. */
const FROG_TOP = SKY_GROUND + 0.015;
const FROG_BOTTOM = 0.97;

/** Gap between one frog doing something and the next. */
const ACT_GAP_MIN = 2200;
const ACT_GAP_MAX = 5200;
/** First action, so they are not all still for the opening seconds. */
const FIRST_MIN = 1200;
const FIRST_MAX = 3600;

const HOP_MS = 420;
const CROAK_MS = 260;
/** Hop distance, as a fraction of canvas width. Short — a frog that
 *  crosses the screen would pull the eye down off the play area. */
const HOP_MIN = 0.02;
const HOP_MAX = 0.06;

interface Frog {
  img: Phaser.GameObjects.Image;
  /** Fraction of canvas width, so a resize keeps them spread out. */
  nx: number;
  ny: number;
  facingRight: boolean;
  busy: boolean;
}

export class FrogPond {
  private scene: Phaser.Scene;
  private frogs: Frog[] = [];
  private nextActAt = -1;
  private started = false;

  /** @param key texture key, or null to have no frogs at all. */
  constructor(scene: Phaser.Scene, key: string | null, count: number) {
    this.scene = scene;
    if (!key) return;
    const w = scene.scale.width;
    const h = scene.scale.height;
    const size = this.unit * FROG_FRACTION;
    for (let i = 0; i < count; i++) {
      // Spread across the width in even slots with a little jitter, so
      // they never land on top of each other.
      const slot = (i + 0.5) / count;
      const nx = Math.min(0.94, Math.max(0.06, slot + (Math.random() - 0.5) * 0.14));
      const ny = FROG_TOP + Math.random() * (FROG_BOTTOM - FROG_TOP);
      const facingRight = Math.random() < 0.5;
      const img = scene.add
        .image(nx * w, ny * h, key)
        .setDisplaySize(size, size)
        .setAlpha(FROG_ALPHA)
        .setTint(FROG_TINT)
        .setDepth(DEPTH_FROG);
      img.setFlipX(!facingRight);
      this.frogs.push({ img, nx, ny, facingRight, busy: false });
    }
  }

  private get unit(): number {
    return Math.min(this.scene.scale.width, this.scene.scale.height);
  }

  private rand(lo: number, hi: number): number {
    return lo + Math.random() * (hi - lo);
  }

  /** Re-place and re-size after a canvas resize. Positions are held in
   *  fractions, so nothing drifts off screen when fullscreen changes. */
  layout(): void {
    const w = this.scene.scale.width;
    const h = this.scene.scale.height;
    const size = this.unit * FROG_FRACTION;
    for (const f of this.frogs) {
      // A frog mid-hop is being tweened; moving it now would fight the
      // tween, and it will be within a few pixels of right anyway.
      if (f.busy) continue;
      f.img.setPosition(f.nx * w, f.ny * h).setDisplaySize(size, size);
    }
  }

  /** One frog acts at a time, on the scene's own clock. */
  update(time: number): void {
    if (this.frogs.length === 0) return;
    if (!this.started) {
      this.started = true;
      this.nextActAt = time + this.rand(FIRST_MIN, FIRST_MAX);
      return;
    }
    if (time < this.nextActAt) return;
    this.nextActAt = time + this.rand(ACT_GAP_MIN, ACT_GAP_MAX);

    const idle = this.frogs.filter((f) => !f.busy);
    if (idle.length === 0) return;
    const frog = idle[Math.floor(Math.random() * idle.length)];
    if (Math.random() < 0.55) this.hop(frog);
    else this.croak(frog);
  }

  /** A short arc to one side, turning to face the way it went. */
  private hop(f: Frog): void {
    const w = this.scene.scale.width;
    const h = this.scene.scale.height;
    const dist = this.rand(HOP_MIN, HOP_MAX);
    // Turn round rather than hop off the edge.
    let dir = f.facingRight ? 1 : -1;
    if (f.nx + dir * dist > 0.94 || f.nx + dir * dist < 0.06) dir = -dir;
    f.nx = Math.min(0.94, Math.max(0.06, f.nx + dir * dist));
    f.facingRight = dir > 0;
    f.img.setFlipX(!f.facingRight);
    f.busy = true;

    const lift = this.unit * 0.035;
    this.scene.tweens.chain({
      targets: f.img,
      tweens: [
        { y: f.ny * h - lift, duration: HOP_MS * 0.45, ease: "Sine.easeOut" },
        { y: f.ny * h, duration: HOP_MS * 0.55, ease: "Sine.easeIn" },
      ],
      onComplete: () => {
        f.busy = false;
      },
    });
    this.scene.tweens.add({
      targets: f.img,
      x: f.nx * w,
      duration: HOP_MS,
      ease: "Sine.easeInOut",
    });
  }

  /** A puff: the frog swells briefly and settles. */
  private croak(f: Frog): void {
    const base = f.img.scaleX;
    f.busy = true;
    this.scene.tweens.chain({
      targets: f.img,
      tweens: [
        {
          scaleX: base * 1.16,
          scaleY: base * 1.1,
          duration: CROAK_MS * 0.5,
          ease: "Sine.easeOut",
        },
        {
          scaleX: base,
          scaleY: base,
          duration: CROAK_MS * 0.5,
          ease: "Sine.easeIn",
        },
      ],
      onComplete: () => {
        f.busy = false;
      },
    });
  }

  /** Called when the round ends, so nothing outlives the scene. */
  destroy(): void {
    for (const f of this.frogs) {
      this.scene.tweens.killTweensOf(f.img);
      f.img.destroy();
    }
    this.frogs = [];
  }
}
