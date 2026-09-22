// Fruit Harvest — the play phase only.
//
// Every phase before and after this one (hand pick, setup check,
// countdowns, calibration, result) is plain DOM in FruitHarvestGame.tsx.
// Phaser is mounted for the 60 s round and torn down straight after, so
// no other route pays for it.
//
// SIZING RULE: every dimension below is a fraction of the canvas, never
// a fixed pixel count. A 2 m viewing distance and an unknown display
// size make absolute pixels meaningless.

import Phaser from "phaser";
import type { HandState } from "@/lib/games/handTracker";
import type { ReachBox } from "@/lib/games/calibration";
import { spawnPoint } from "@/lib/games/calibration";
import { OneEuro2D } from "@/lib/games/oneEuro";
import type { GameAudio } from "@/lib/games/gameAudio";

export const ROUND_MS = 60_000;
const MAX_FRUIT = 3;
/** How long a fruit waits to be picked before it falls away. */
const FRUIT_TTL_MS = 5200;
/** Gap between spawns so three do not appear on the same frame. */
const SPAWN_GAP_MS = 600;

const FRUITS = ["🍎", "🍊", "🍐", "🍋", "🍓", "🍇", "🍑", "🥝"];
const BASKET = "🧺";

/** Glyph texture size. Generous so a large sprite stays crisp. */
const GLYPH_TEX = 256;

export interface FruitHarvestControl {
  /** Live hand state, mutated by the React pose loop. */
  state: HandState;
  box: ReachBox;
  /** 1 = normal, >1 when the patient asked for bigger visuals. */
  visualScale: number;
  audio: GameAudio;
  /** Counters the React layer polls for the HUD-free result screen. */
  harvested: number;
  missed: number;
  remainingMs: number;
  finished: boolean;
  onFinish: (r: { harvested: number; missed: number }) => void;
}

interface Fruit {
  img: Phaser.GameObjects.Image;
  nx: number;
  ny: number;
  bornAt: number;
  dying: boolean;
}

/**
 * Render an emoji into a canvas texture.
 * @returns false when the glyph did not actually rasterise, so the
 *          caller can fall back rather than show an empty square.
 */
function makeGlyphTexture(
  scene: Phaser.Scene,
  key: string,
  glyph: string,
): boolean {
  if (scene.textures.exists(key)) return true;
  const tex = scene.textures.createCanvas(key, GLYPH_TEX, GLYPH_TEX);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  ctx.clearRect(0, 0, GLYPH_TEX, GLYPH_TEX);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `${Math.round(GLYPH_TEX * 0.78)}px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif`;
  ctx.fillText(glyph, GLYPH_TEX / 2, GLYPH_TEX * 0.54);

  // Confirm something was actually drawn — emoji coverage varies by
  // platform and a missing glyph renders as blank or as a box.
  let painted = false;
  try {
    const data = ctx.getImageData(0, 0, GLYPH_TEX, GLYPH_TEX).data;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] !== 0) {
        painted = true;
        break;
      }
    }
  } catch {
    // Canvas read blocked — assume it drew rather than losing the art.
    painted = true;
  }
  if (!painted) {
    scene.textures.remove(key);
    return false;
  }
  tex.refresh();
  return true;
}

/** Fallback art when an emoji glyph is unavailable: a lit disc. */
function makeDiscTexture(scene: Phaser.Scene, key: string, colour: number): void {
  if (scene.textures.exists(key)) return;
  const tex = scene.textures.createCanvas(key, GLYPH_TEX, GLYPH_TEX);
  if (!tex) return;
  const ctx = tex.getContext();
  if (!ctx) return;
  const r = GLYPH_TEX * 0.42;
  const cx = GLYPH_TEX / 2;
  const g = ctx.createRadialGradient(cx - r * 0.3, cx - r * 0.3, r * 0.1, cx, cx, r);
  const hex = colour.toString(16).padStart(6, "0");
  g.addColorStop(0, "#ffffff");
  g.addColorStop(0.45, `#${hex}`);
  g.addColorStop(1, "#00000055");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(cx, cx, r, 0, Math.PI * 2);
  ctx.fill();
  tex.refresh();
}

export class FruitHarvestScene extends Phaser.Scene {
  private control!: FruitHarvestControl;
  private cursor!: Phaser.GameObjects.Image;
  private basket!: Phaser.GameObjects.Image;
  private timerText!: Phaser.GameObjects.Text;
  private scoreText!: Phaser.GameObjects.Text;
  private fruits: Fruit[] = [];
  private keys: string[] = [];
  private filter = new OneEuro2D({ minCutoff: 1.1, beta: 0.02 });
  private startedAt = 0;
  private lastSpawnAt = 0;
  private lastFrameAt = 0;
  private nextRegion: "same" | "across" = "same";
  private cursorSeeded = false;

  constructor() {
    super("fruit-harvest");
  }

  init(data: { control: FruitHarvestControl }) {
    this.control = data.control;
  }

  /** Shortest canvas edge — the unit every size is expressed in. */
  private get unit(): number {
    return Math.min(this.scale.width, this.scale.height);
  }

  create() {
    const s = this.control.visualScale;

    this.keys = FRUITS.map((glyph, i) => {
      const key = `fruit-${i}`;
      if (!makeGlyphTexture(this, key, glyph)) {
        makeDiscTexture(this, key, 0xef4444);
      }
      return key;
    });
    if (!makeGlyphTexture(this, "basket", BASKET)) {
      makeDiscTexture(this, "basket", 0xb45309);
    }
    if (!makeGlyphTexture(this, "cursor", "✋")) {
      makeDiscTexture(this, "cursor", 0x22d3ee);
    }

    this.basket = this.add
      .image(this.scale.width / 2, this.scale.height * 0.93, "basket")
      .setDisplaySize(this.unit * 0.17 * s, this.unit * 0.17 * s)
      .setDepth(5);

    this.cursor = this.add
      .image(this.scale.width / 2, this.scale.height / 2, "cursor")
      .setDisplaySize(this.unit * 0.12 * s, this.unit * 0.12 * s)
      .setDepth(20)
      .setAlpha(0);

    const hud = Math.round(this.unit * 0.095 * s);
    this.timerText = this.add
      .text(this.scale.width * 0.04, this.scale.height * 0.03, "60", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${hud}px`,
        color: "#ffffff",
        stroke: "#000000",
        strokeThickness: Math.max(2, hud * 0.1),
      })
      .setDepth(30);

    this.scoreText = this.add
      .text(this.scale.width * 0.96, this.scale.height * 0.03, "0", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${hud}px`,
        color: "#fde047",
        stroke: "#000000",
        strokeThickness: Math.max(2, hud * 0.1),
      })
      .setOrigin(1, 0)
      .setDepth(30);

    this.startedAt = this.time.now;
    this.lastFrameAt = this.time.now;
    this.lastSpawnAt = 0;
  }

  update(time: number) {
    const c = this.control;
    if (c.finished) return;

    const dt = Math.max(0.001, (time - this.lastFrameAt) / 1000);
    this.lastFrameAt = time;

    // ── Clock
    const elapsed = time - this.startedAt;
    c.remainingMs = Math.max(0, ROUND_MS - elapsed);
    this.timerText.setText(String(Math.ceil(c.remainingMs / 1000)));
    if (c.remainingMs <= 0) {
      c.finished = true;
      c.onFinish({ harvested: c.harvested, missed: c.missed });
      return;
    }

    // ── Cursor. Map FIRST, then smooth: filtering in canvas space keeps
    //    the cutoff in the same units as the on-screen motion the
    //    patient sees, and survives a resize without a jump.
    if (c.state.usable) {
      const target = { x: c.state.x, y: c.state.y };
      if (!this.cursorSeeded) {
        // Snap on the first good frame instead of sliding in from the
        // middle of the screen.
        this.filter.reset();
        this.cursor.setPosition(target.x, target.y);
        this.cursor.setAlpha(1);
        this.cursorSeeded = true;
        this.filter.filter(target.x, target.y, dt);
      } else {
        const p = this.filter.filter(target.x, target.y, dt);
        this.cursor.setPosition(p.x, p.y);
        this.cursor.setAlpha(1);
      }
    } else {
      this.cursor.setAlpha(0.25);
    }

    // ── Spawn
    if (
      this.fruits.length < MAX_FRUIT
      && time - this.lastSpawnAt > SPAWN_GAP_MS
    ) {
      this.spawn(time);
      this.lastSpawnAt = time;
    }

    // ── Fruit: reposition (so a resize keeps them in reach), test the
    //    cursor, and expire.
    const cover = c.state.cover;
    const hitR = this.unit * 0.075 * c.visualScale;
    for (let i = this.fruits.length - 1; i >= 0; i--) {
      const f = this.fruits[i];
      if (f.dying) continue;

      if (cover.dispW > 0) {
        f.img.setPosition(
          cover.offX + f.nx * cover.dispW,
          cover.offY + f.ny * cover.dispH,
        );
      }

      if (c.state.usable && this.cursorSeeded) {
        const d = Math.hypot(this.cursor.x - f.img.x, this.cursor.y - f.img.y);
        if (d < hitR) {
          this.harvest(f, i);
          continue;
        }
      }

      if (time - f.bornAt > FRUIT_TTL_MS) this.dropAway(f, i);
    }
  }

  private spawn(time: number) {
    const c = this.control;
    const p = spawnPoint(c.box, this.nextRegion, Math.random);
    // Alternate the halves so the across-midline reach — the part that
    // exercises adduction — is never crowded out by chance.
    this.nextRegion = this.nextRegion === "same" ? "across" : "same";

    const cover = c.state.cover;
    const x = cover.dispW > 0 ? cover.offX + p.nx * cover.dispW : this.scale.width / 2;
    const y = cover.dispH > 0 ? cover.offY + p.ny * cover.dispH : this.scale.height / 2;

    const key = this.keys[Math.floor(Math.random() * this.keys.length)];
    const size = this.unit * 0.13 * c.visualScale;
    const img = this.add.image(x, y, key).setDisplaySize(size, size).setDepth(10);
    img.setScale(img.scaleX * 0.4);
    this.tweens.add({
      targets: img,
      scaleX: img.scaleX / 0.4,
      scaleY: img.scaleY / 0.4,
      duration: 220,
      ease: "Back.easeOut",
    });

    this.fruits.push({ img, nx: p.nx, ny: p.ny, bornAt: time, dying: false });
  }

  private harvest(f: Fruit, index: number) {
    f.dying = true;
    this.fruits.splice(index, 1);
    this.control.harvested += 1;
    this.scoreText.setText(String(this.control.harvested));
    this.control.audio.harvest();
    this.tweens.add({
      targets: f.img,
      x: this.basket.x,
      y: this.basket.y,
      scaleX: f.img.scaleX * 0.3,
      scaleY: f.img.scaleY * 0.3,
      duration: 320,
      ease: "Cubic.easeIn",
      onComplete: () => f.img.destroy(),
    });
    this.tweens.add({
      targets: this.basket,
      scaleX: this.basket.scaleX * 1.15,
      scaleY: this.basket.scaleY * 0.88,
      duration: 110,
      yoyo: true,
    });
  }

  private dropAway(f: Fruit, index: number) {
    f.dying = true;
    this.fruits.splice(index, 1);
    this.control.missed += 1;
    this.control.audio.miss();
    this.tweens.add({
      targets: f.img,
      y: this.scale.height + this.unit * 0.2,
      alpha: 0,
      angle: 140,
      duration: 620,
      ease: "Quad.easeIn",
      onComplete: () => f.img.destroy(),
    });
  }
}
