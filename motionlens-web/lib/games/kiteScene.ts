// Kite Flying — the play phase only.
//
// Everything before and after the round is GameShell's, and the frame
// clock, the hand-lost pause, the palm cursor and the round timer are
// GameSceneBase's. What is here is the mechanic: a ribbon of wind
// scrolling past, and a kite on the end of the patient's arm that has
// to stay in it.
//
// WHY A CORRIDOR. Fruit Harvest asks for reach and Cloudburst asks for
// timing and inhibition. This one asks for SUSTAINED CONTROL — a
// continuous tracking task, where the measure is not how many things
// were hit but how steadily the hand held a moving line. That is why
// its headline number is a percentage of time rather than a count.
//
// TWO SOURCES OF TRUTH, DELIBERATELY. The kite follows the DRAWN
// cursor, because feedback has to match what the patient sees. The
// metrics follow the RAW palm, because the drawn cursor is smoothed and
// the smoothing would erase the thing being measured. See
// lib/games/kiteMovement.ts.
//
// SIZING RULE: every dimension is a fraction of the canvas unit.

import Phaser from "phaser";
import { GameSceneBase, type FrameInfo } from "@/lib/games/gameSceneBase";
import type { KiteControl } from "@/lib/games/kiteControl";
import {
  blankKiteResult,
  pct,
  type KiteResult,
} from "@/lib/games/kiteMetrics";
import { MovementAnalyser } from "@/lib/games/kiteMovement";
import {
  centreNy,
  deviationRatio,
  makeCorridor,
  type Corridor,
} from "@/lib/games/kiteCorridor";
import {
  FALL_AFTER_MS,
  KITE_FRACTION,
  RESPAWN_MS,
  ROUND_MS,
  TUMBLE_MS,
} from "@/lib/games/kiteLevels";
import { makeMeadowTexture, MEADOW_GROUND } from "@/lib/games/meadowScene";
import { makeCloudTexture } from "@/lib/games/skyScene";
import { makeSoftDotTexture } from "@/lib/games/fruitEffects";
import { BackgroundLife } from "@/lib/games/backgroundLife";

const KITE = "🪁";
const BIRD = "🐦";

/** Glyph texture size. Generous so a large sprite stays crisp. */
const GLYPH_TEX = 256;

/** The ribbon, drawn as three stacked fills so its edge is soft rather
 *  than a hard band. Light blue: the one thing on screen that is not
 *  sky, hill or grass. */
const RIBBON_TINT = 0x7dd3fc;
const RIBBON_LAYERS: { widthMul: number; alpha: number }[] = [
  { widthMul: 1.9, alpha: 0.1 },
  { widthMul: 1.35, alpha: 0.16 },
  { widthMul: 1.0, alpha: 0.3 },
];
/** Horizontal samples across the ribbon. Enough that the curve reads as
 *  smooth, few enough that redrawing it every frame is free. */
const RIBBON_COLS = 48;

/** Kite glow: green inside the wind, amber outside. At 2 m this, not
 *  the glyph, is what tells the patient how they are doing. */
const GLOW_IN = 0x4ade80;
const GLOW_OUT = 0xfbbf24;
const GLOW_SCALE = 2.0;
const GLOW_ALPHA = 0.55;

/** Gentle sway while flying, and a faster wobble while outside. */
const SWAY_DEG = 7;
const SWAY_MS = 1700;
const WOBBLE_DEG = 18;
const WOBBLE_MS = 260;

/** Drifting background clouds. */
const CLOUD_COUNT = 4;
const CLOUD_ALPHA = 0.55;
const CLOUD_CROSS_MIN_MS = 40_000;
const CLOUD_CROSS_MAX_MS = 70_000;

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

/** Fallback art when the kite glyph is unavailable: a lit diamond. The
 *  glow already carries the meaning, so a round without emoji support
 *  is still playable. */
function makeKiteFallback(scene: Phaser.Scene, key: string): void {
  if (scene.textures.exists(key)) return;
  const tex = scene.textures.createCanvas(key, GLYPH_TEX, GLYPH_TEX);
  if (!tex) return;
  const ctx = tex.getContext();
  if (!ctx) return;
  const c = GLYPH_TEX / 2;
  const r = GLYPH_TEX * 0.4;
  ctx.beginPath();
  ctx.moveTo(c, c - r);
  ctx.lineTo(c + r * 0.72, c);
  ctx.lineTo(c, c + r);
  ctx.lineTo(c - r * 0.72, c);
  ctx.closePath();
  const g = ctx.createLinearGradient(c, c - r, c, c + r);
  g.addColorStop(0, "#fef3c7");
  g.addColorStop(1, "#f59e0b");
  ctx.fillStyle = g;
  ctx.fill();
  ctx.strokeStyle = "#78350f";
  ctx.lineWidth = GLYPH_TEX * 0.02;
  ctx.stroke();
  tex.refresh();
}

export class KiteScene extends GameSceneBase {
  protected declare control: KiteControl;
  private backdrop: Phaser.GameObjects.Image | null = null;
  private ribbon!: Phaser.GameObjects.Graphics;
  private string!: Phaser.GameObjects.Graphics;
  private kite!: Phaser.GameObjects.Image;
  private glow: Phaser.GameObjects.Image | null = null;
  private pctText!: Phaser.GameObjects.Text;
  private fallText: Phaser.GameObjects.Text | null = null;
  private levelText: Phaser.GameObjects.Text | null = null;
  private life: BackgroundLife | null = null;

  private corridor!: Corridor;
  private analyser = new MovementAnalyser();
  private res: KiteResult = blankKiteResult(1, "right");

  /** Distance scrolled, in canvas widths. The ribbon's whole shape is
   *  a function of this. */
  private scrollU = 0;
  /** Where the string is tied. */
  private anchorX = 0;
  private anchorY = 0;

  /** Continuous time the kite has been outside the ribbon. */
  private outsideMs = 0;
  /** While falling the patient has nothing to steer, so neither the
   *  corridor clock nor the movement trace runs. */
  private falling = false;
  private fallEndsAt = -1;
  private wasInside = true;
  /** Live deviation, for the debug overlay. */
  private lastDev = 0;

  constructor() {
    super("kite-flying");
  }

  create() {
    const s = this.control.visualScale;
    this.res = blankKiteResult(this.control.level.id, this.control.hand);
    this.corridor = makeCorridor(this.control.box, this.control.level);

    const w = Math.round(this.scale.width);
    const h = Math.round(this.scale.height);
    if (makeMeadowTexture(this, "kf-sky", w, h, Math.floor(Math.random() * 1e6))) {
      this.backdrop = this.add
        .image(0, 0, "kf-sky")
        .setOrigin(0, 0)
        .setDisplaySize(w, h)
        .setDepth(-10);
    } else {
      // Texture creation failed — a flat wash still hides the camera.
      this.add.rectangle(0, 0, w, h, 0x8ec4e2).setOrigin(0, 0).setDepth(-10);
    }
    this.canvasSize = { w, h };

    makeSoftDotTexture(this, "fx-dot");
    if (!makeGlyphTexture(this, "kf-kite", KITE)) makeKiteFallback(this, "kf-kite");

    if (makeCloudTexture(this, "kf-cloud", 31)) {
      for (let i = 0; i < CLOUD_COUNT; i++) this.spawnCloud(true);
    }
    // Birds only — BackgroundLife skips any animal whose glyph is null.
    const birdKey = makeGlyphTexture(this, "fx-bird", BIRD) ? "fx-bird" : null;
    this.life = new BackgroundLife(this, { monkey: null, bird: birdKey });

    // The ribbon sits under the kite but over the clouds.
    this.ribbon = this.add.graphics().setDepth(4);
    this.string = this.add.graphics().setDepth(9);

    this.glow = this.textures.exists("fx-dot")
      ? this.add
        .image(w / 2, h / 2, "fx-dot")
        .setTint(GLOW_IN)
        .setAlpha(0)
        .setDepth(10)
      : null;
    this.kite = this.add
      .image(w / 2, h / 2, "kf-kite")
      .setDisplaySize(this.unit * KITE_FRACTION * s, this.unit * KITE_FRACTION * s)
      .setDepth(11);

    // The base drives `cursor`: position and alpha, every frame. Here
    // that is an INVISIBLE stand-in, because the kite has to be free to
    // tumble away from the hand during a fall — if the kite were the
    // cursor, the base would drag it straight back.
    this.cursor = this.add
      .image(w / 2, h / 2, "kf-kite")
      .setDisplaySize(2, 2)
      .setVisible(false)
      .setDepth(0);

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

    this.levelText = this.add
      .text(0, 0, this.control.level.label, {
        fontFamily: "system-ui, sans-serif",
        fontSize: "16px",
        color: "#ffffff",
        stroke: "#000000",
        strokeThickness: 3,
      })
      .setDepth(30);

    this.pctText = this.add
      .text(this.scale.width * 0.96, this.scale.height * 0.03, "0%", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${hud}px`,
        color: "#bbf7d0",
        stroke: "#000000",
        strokeThickness: Math.max(2, hud * 0.1),
      })
      .setOrigin(1, 0)
      .setDepth(30);

    this.fallText = this.add
      .text(this.scale.width * 0.96, 0, "", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${Math.round(this.unit * 0.04 * s)}px`,
        color: "#fbbf24",
        stroke: "#000000",
        strokeThickness: 3,
      })
      .setOrigin(1, 0)
      .setDepth(30);

    // In-canvas FPS readout. Read straight from the query string so the
    // scene needs nothing from the React layer to show it.
    let debugOn = false;
    try {
      debugOn = new URLSearchParams(window.location.search).get("gamedebug") === "1";
    } catch {
      debugOn = false;
    }
    if (debugOn) {
      const fs = Math.round(this.unit * 0.032);
      this.fpsText = this.add
        .text(this.scale.width * 0.02, this.scale.height * 0.82, "fps —", {
          fontFamily: "ui-monospace, monospace",
          fontSize: `${fs}px`,
          color: "#a3e635",
          backgroundColor: "#000000b0",
          padding: { x: 6, y: 3 },
        })
        .setDepth(31);
    }

    // ── "Hand lost" banner. Hidden until needed.
    const lost = Math.round(this.unit * 0.062 * s);
    this.lostBand = this.add
      .rectangle(w / 2, h / 2, w, this.unit * 0.3, 0x000000, 0.72)
      .setDepth(40)
      .setVisible(false);
    this.lostText = this.add
      .text(w / 2, h / 2, "We can't see your hand\nstep back into view", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${lost}px`,
        color: "#ffffff",
        align: "center",
        stroke: "#000000",
        strokeThickness: Math.max(2, lost * 0.08),
      })
      .setOrigin(0.5)
      .setDepth(41)
      .setVisible(false);

    // NOTE: startedAt / lastFrameAt are deliberately left at -1 here and
    // seeded on the first update frame — see GameSceneBase for why this
    // must not use this.time.now.
    this.layout();
    this.palmBase = { ...this.control.state.palmCounts };

    const d = this.control.debug;
    d.sceneState = "created";
    d.texturesOk = this.textures.exists("kf-kite");
    d.boxN = {
      x0: this.control.box.xLo,
      x1: this.control.box.xHi,
      y0: this.control.box.yLo,
      y1: this.control.box.yHi,
    };
    // Created here, in the order the debug panel should print them.
    d.extra["corridor width"] = "—";
    d.extra["deviation"] = "—";
    d.extra["inside"] = "—";
    d.extra["raw vs drawn"] = "—";
    d.extra["velocity peaks"] = "0";
  }

  protected finishRound(): void {
    const c = this.control;
    this.life?.destroy();
    this.life = null;
    this.res.peaksPerSec = this.analyser.peaksPerSec;
    this.res.peakCount = this.analyser.peakCount;
    this.res.movingSec = Math.round(this.analyser.movingSec * 10) / 10;
    c.onFinish({ ...this.res, level: c.level.id, hand: c.hand });
  }

  protected layout(): void {
    const w = this.scale.width;
    const h = this.scale.height;
    const u = Math.min(w, h);
    const s = this.control.visualScale;
    this.canvasSize = { w, h };

    this.backdrop?.setDisplaySize(w, h);
    this.kite?.setDisplaySize(u * KITE_FRACTION * s, u * KITE_FRACTION * s);

    // The string is tied to the ground, off to the side the patient is
    // NOT playing with, so it never cuts across the working area.
    this.anchorX = this.control.hand === "right" ? w * 0.18 : w * 0.82;
    this.anchorY = h * MEADOW_GROUND;

    const hud = Math.round(u * 0.095 * s);
    this.timerText
      ?.setPosition(w * 0.04, h * 0.03)
      .setFontSize(hud)
      .setStroke("#000000", Math.max(2, hud * 0.1));
    this.pctText
      ?.setPosition(w * 0.96, h * 0.03)
      .setFontSize(hud)
      .setStroke("#000000", Math.max(2, hud * 0.1));
    this.fallText
      ?.setPosition(w * 0.96, h * 0.03 + hud * 1.05)
      .setFontSize(Math.round(u * 0.04 * s));
    const lvl = Math.round(u * 0.035 * s);
    this.levelText
      ?.setPosition(w * 0.04, h * 0.03 + hud * 1.05)
      .setFontSize(lvl)
      .setStroke("#000000", Math.max(2, lvl * 0.1));

    this.lostBand?.setPosition(w / 2, h / 2).setSize(w, u * 0.3);
    const lost = Math.round(u * 0.062 * s);
    this.lostText
      ?.setPosition(w / 2, h / 2)
      .setFontSize(lost)
      .setStroke("#000000", Math.max(2, lost * 0.08));
    this.fpsText?.setPosition(w * 0.02, h * 0.82).setFontSize(Math.round(u * 0.032));
  }

  update(time: number) {
    const c = this.control;

    // Clocks and the hand-lost pause. Null means the round is over.
    const frame = this.beginFrame(time);
    if (!frame) return;
    const { dtMs, dt, lost } = frame;

    if (!lost) this.life?.update(time);
    this.syncCanvasSize();
    if (this.advanceRoundClock(frame)) return;
    this.updateCursor(frame);

    // The wind keeps blowing even while the round is held — but it does
    // not while the hand is lost, or the corridor would scroll past
    // unseen and the patient would come back to a different shape.
    if (!lost) this.scrollU += c.level.scrollPerSec * dt;
    this.drawRibbon();

    if (lost) {
      // Nothing to measure and nothing to steer. Break the movement
      // trace so the gap is not read as one enormous slow movement.
      this.analyser.breakTrace();
      if (this.fallEndsAt >= 0) this.fallEndsAt += dtMs;
      this.writeDebug(frame, false);
      return;
    }

    // ── The kite. It follows the DRAWN cursor, because what the
    //    patient sees has to be what the game responds to.
    if (this.falling) {
      if (time >= this.fallEndsAt) this.respawn();
    } else {
      this.kite.setPosition(this.cursor.x, this.cursor.y);
    }
    this.glow?.setPosition(this.kite.x, this.kite.y);
    this.drawString();

    const cover = c.state.cover;
    const W = this.scale.width || 1;

    // Kite against the corridor — this drives the feedback.
    let kiteInside = false;
    if (cover.dispH > 0 && !this.falling) {
      const ny = (this.kite.y - cover.offY) / cover.dispH;
      const worldX = this.scrollU + this.kite.x / W;
      const dev = deviationRatio(this.corridor, worldX, ny);
      this.lastDev = dev;
      kiteInside = dev <= 1;
      this.applyKiteMood(kiteInside);

      if (kiteInside) {
        this.outsideMs = 0;
      } else {
        this.outsideMs += dtMs;
        if (this.outsideMs >= FALL_AFTER_MS) this.startFall(time);
      }
      if (kiteInside !== this.wasInside) {
        this.wasInside = kiteInside;
        // A short cue on the crossing rather than a tone held while
        // inside: the audio here is blips, and one repeating twice a
        // second would be worse than nothing.
        if (kiteInside) c.audio.harvest();
        else c.audio.miss();
      }
    }

    // ── Metrics, from the RAW palm. Only while the hand is genuinely
    //    visible and the patient has a kite to steer.
    const measurable = c.state.live && c.state.inFrame && !this.falling;
    if (measurable && cover.dispH > 0) {
      const rawNy = (c.state.palmY - cover.offY) / cover.dispH;
      const rawWorldX = this.scrollU + c.state.palmX / W;
      const rawDev = deviationRatio(this.corridor, rawWorldX, rawNy);
      const palmInside = rawDev <= 1;

      this.res.measuredMs += dtMs;
      this.res.devSum += rawDev;
      this.res.devCount += 1;
      if (palmInside) this.res.insideMs += dtMs;

      const half = this.half() === 1 ? this.res.firstHalf : this.res.secondHalf;
      half.total += dtMs;
      if (palmInside) half.inside += dtMs;

      this.analyser.push(time, c.state.palmX, c.state.palmY, c.state.armLenPx);
    } else {
      this.analyser.breakTrace();
    }

    c.insidePct = pct(this.res.insideMs, this.res.measuredMs);
    this.pctText.setText(`${c.insidePct}%`);

    this.writeDebug(frame, kiteInside);
  }

  /** 1 for the first half of the round, 2 for the second. */
  private half(): 1 | 2 {
    return this.control.remainingMs > ROUND_MS / 2 ? 1 : 2;
  }

  /** Redraw the ribbon for the current scroll. Three stacked fills, so
   *  the edge is soft rather than a ruled band. */
  private drawRibbon(): void {
    const cover = this.control.state.cover;
    const g = this.ribbon;
    g.clear();
    if (cover.dispH <= 0) return;
    const W = this.scale.width || 1;
    const halfPx = this.corridor.halfNy * cover.dispH;
    if (halfPx <= 0) return;

    for (const L of RIBBON_LAYERS) {
      const hp = halfPx * L.widthMul;
      g.fillStyle(RIBBON_TINT, L.alpha);
      g.beginPath();
      for (let i = 0; i <= RIBBON_COLS; i++) {
        const x = (i / RIBBON_COLS) * W;
        const y = cover.offY
          + centreNy(this.corridor, this.scrollU + x / W) * cover.dispH - hp;
        if (i === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      }
      for (let i = RIBBON_COLS; i >= 0; i--) {
        const x = (i / RIBBON_COLS) * W;
        const y = cover.offY
          + centreNy(this.corridor, this.scrollU + x / W) * cover.dispH + hp;
        g.lineTo(x, y);
      }
      g.closePath();
      g.fillPath();
    }
  }

  /** A thin line from the anchor on the ground up to the kite. */
  private drawString(): void {
    const g = this.string;
    g.clear();
    if (!this.kite.visible) return;
    g.lineStyle(Math.max(1, this.unit * 0.0035), 0xf8fafc, 0.55);
    g.beginPath();
    g.moveTo(this.anchorX, this.anchorY);
    // A slight sag, so it reads as string rather than as a laser.
    const mx = (this.anchorX + this.kite.x) / 2;
    const my = (this.anchorY + this.kite.y) / 2 + this.unit * 0.04;
    g.lineTo(mx, my);
    g.lineTo(this.kite.x, this.kite.y);
    g.strokePath();
  }

  /** Green and swaying, or amber and wobbling. */
  private applyKiteMood(inside: boolean): void {
    if (!this.glow) {
      this.kite.setAngle(inside ? this.kite.angle : this.kite.angle);
      return;
    }
    const size = this.kite.displayWidth * GLOW_SCALE;
    this.glow
      .setDisplaySize(size, size)
      .setTint(inside ? GLOW_IN : GLOW_OUT)
      .setAlpha(GLOW_ALPHA);

    // Sway and wobble are driven off the round clock rather than tweens
    // so they cannot fight the per-frame position updates.
    const t = this.lastFrameAt;
    this.kite.setAngle(
      inside
        ? Math.sin((t / SWAY_MS) * Math.PI * 2) * SWAY_DEG
        : Math.sin((t / WOBBLE_MS) * Math.PI * 2) * WOBBLE_DEG,
    );
  }

  private startFall(time: number): void {
    if (this.falling) return;
    this.falling = true;
    this.outsideMs = 0;
    this.res.falls += 1;
    this.control.falls = this.res.falls;
    this.fallText?.setText(`Falls: ${this.res.falls}`);
    this.control.audio.error();
    // The trace ends here: the patient has nothing to steer until the
    // kite comes back, so that time is not theirs to be judged on.
    this.analyser.breakTrace();
    this.fallEndsAt = time + TUMBLE_MS + RESPAWN_MS;

    this.tweens.killTweensOf(this.kite);
    this.tweens.add({
      targets: this.kite,
      y: this.scale.height * MEADOW_GROUND,
      angle: this.kite.angle + 540,
      alpha: 0,
      duration: TUMBLE_MS,
      ease: "Quad.easeIn",
      onComplete: () => this.kite.setVisible(false),
    });
    if (this.glow) {
      this.tweens.killTweensOf(this.glow);
      this.tweens.add({
        targets: this.glow,
        alpha: 0,
        duration: TUMBLE_MS * 0.6,
      });
    }
  }

  private respawn(): void {
    this.falling = false;
    this.fallEndsAt = -1;
    this.outsideMs = 0;
    this.wasInside = true;
    this.tweens.killTweensOf(this.kite);
    this.kite
      .setVisible(true)
      .setPosition(this.cursor.x, this.cursor.y)
      .setAngle(0)
      .setAlpha(0);
    this.tweens.add({
      targets: this.kite,
      alpha: 1,
      duration: 220,
      ease: "Quad.easeOut",
    });
  }

  /** One drifting cloud, tweened across and replaced when it leaves. */
  private spawnCloud(initial: boolean): void {
    const w = this.scale.width;
    const h = this.scale.height;
    const u = this.unit;
    const size = u * (0.3 + Math.random() * 0.24);
    const y = h * (0.04 + Math.random() * 0.22);
    const startX = initial ? Math.random() * w : -size;
    const img = this.add
      .image(startX, y, "kf-cloud")
      .setDisplaySize(size, size * 0.5)
      .setAlpha(CLOUD_ALPHA)
      .setDepth(1);
    const travel = w + size * 2;
    const remaining = (travel - (startX + size)) / travel;
    const dur = (CLOUD_CROSS_MIN_MS
      + Math.random() * (CLOUD_CROSS_MAX_MS - CLOUD_CROSS_MIN_MS))
      * Math.max(0.15, remaining);
    this.tweens.add({
      targets: img,
      x: w + size,
      duration: dur,
      ease: "Linear",
      onComplete: () => {
        img.destroy();
        if (!this.control.finished) this.spawnCloud(false);
      },
    });
  }

  private writeDebug(frame: FrameInfo, inside: boolean): void {
    this.writeCoreDebug(frame);
    const c = this.control;
    const d = c.debug;
    const cover = c.state.cover;
    const halfPx = cover.dispH > 0 ? this.corridor.halfNy * cover.dispH : 0;
    d.extra["corridor width"] =
      `${Math.round(halfPx * 2)} px (${(this.corridor.halfNy * 2).toFixed(3)} ny)`;
    d.extra["deviation"] = `${(this.lastDev * 100).toFixed(0)}% of half-width`;
    d.extra["inside"] = this.falling
      ? "falling"
      : inside ? "yes" : `no (${Math.round(this.outsideMs)} ms)`;
    d.extra["raw vs drawn"] = `${Math.round(
      Math.hypot(this.cursor.x - c.state.palmX, this.cursor.y - c.state.palmY),
    )} px`;
    const pps = this.analyser.peaksPerSec;
    d.extra["velocity peaks"] =
      `${this.analyser.peakCount} over ${this.analyser.movingSec.toFixed(1)}s`
      + ` = ${pps === null ? "n/a" : pps.toFixed(2)}/s`;

    if (this.fpsText) {
      this.fpsText.setText(
        `fps ${d.fps} min ${d.fpsMin}  pose ${d.poseHz}Hz  `
        + `cutoff ${d.cutoffHz}Hz  lag ${d.lagPx}px  `
        + `palm ${d.palmSource}\n`
        + `arm ${d.armLenPx}px  headroom ${d.headroomPx}px  `
        + `ratio ${d.headroomRatio}\n`
        + `in corridor ${c.insidePct}%  dev ${(this.lastDev * 100).toFixed(0)}%  `
        + `falls ${this.res.falls}  peaks ${this.analyser.peakCount}`,
      );
    }
  }
}
