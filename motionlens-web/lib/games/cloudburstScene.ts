// Cloudburst — the play phase only.
//
// Everything before and after the round (hand pick, setup, calibration,
// countdowns, result, save) is GameShell's, and the frame clock, the
// hand-lost pause, the palm cursor and the round timer are
// GameSceneBase's. What is here is the mechanic: items falling down
// lanes that the patient can actually reach, half of them worth
// catching and half worth leaving alone.
//
// WHY LANES. Fruit Harvest places targets at the EDGE of reach, which
// makes the patient extend. Cloudburst instead drops items down a
// vertical line whose x sits inside the calibrated box, so the demand
// is timing and inhibition rather than range. The two games ask for
// different things on purpose.
//
// WHY A BAD ITEM. Catching is a commission task; not catching is an
// inhibition task, and a patient can be intact at one and impaired at
// the other. The lightning exists so the round measures both, and the
// two scores are never averaged together.
//
// SIZING RULE: every dimension is a fraction of the canvas unit, never
// a fixed pixel count.

import Phaser from "phaser";
import { spawnPoint } from "@/lib/games/calibration";
import { GameSceneBase, PREDICT_DEADBAND } from "@/lib/games/gameSceneBase";
import type { CloudburstControl } from "@/lib/games/cloudburstControl";
import {
  blankCloudburstResult,
  type CloudburstResult,
  type CloudburstZone,
} from "@/lib/games/cloudburstMetrics";
import {
  BASE_HIT_FRACTION,
  BASE_ITEM_FRACTION,
  MAX_ITEMS,
  ROUND_MS,
  SPAWN_LEAD,
  SPEED_RAMP_TO,
} from "@/lib/games/cloudburstLevels";
import { makeCloudTexture, makeSkyTexture, SKY_GROUND } from "@/lib/games/skyScene";
import { makeRingTexture, makeSoftDotTexture } from "@/lib/games/fruitEffects";
import { BackgroundLife } from "@/lib/games/backgroundLife";

const DROP = "💧";
const BOLT = "⚡";
const BIRD = "🐦";

/** Halo colours. These, not the glyphs, are what the patient reads at
 *  2 m — an emoji is a few dozen pixels at that distance, a halo is a
 *  couple of hundred. Blue = catch, amber = leave alone. Chosen to stay
 *  distinguishable in greyscale too: the blue is markedly darker. */
const DROP_HALO = 0x38bdf8;
const BOLT_HALO = 0xf59e0b;
/** The bolt's halo is stronger than the drop's. Missing a drop costs a
 *  point; grabbing a bolt is an error, so it gets the louder signal. */
const DROP_HALO_ALPHA = 0.55;
const BOLT_HALO_ALPHA = 0.72;
const HALO_SCALE = 2.1;

/** Fade-out when an item leaves without being touched. */
const FADE_MS = 260;
/** Contact flash on a caught drop / a touched bolt. */
const SPLASH_MS = 320;
const ZAP_MS = 260;

/** Glyph texture size. Generous so a large sprite stays crisp. */
const GLYPH_TEX = 256;

/** Nearer drifting clouds, over the baked sky. Kept in the top third
 *  and very faint, so nothing competes with a falling item. */
const CLOUD_COUNT = 4;
const CLOUD_ALPHA = 0.3;
const CLOUD_CROSS_MIN_MS = 34_000;
const CLOUD_CROSS_MAX_MS = 58_000;

type ItemKind = "drop" | "bolt";

interface Item {
  img: Phaser.GameObjects.Image;
  halo: Phaser.GameObjects.Image | null;
  kind: ItemKind;
  /** Lane, in the mirrored normalised video space. Stored normalised so
   *  a resize keeps the lane where the patient could reach it. */
  nx: number;
  /** Current height, in the same cover-normalised space. */
  ny: number;
  bornAt: number;
  /** First frame the cursor moved toward this item, or null. Only
   *  meaningful for drops. */
  reactedAt: number | null;
  dying: boolean;
  zone: CloudburstZone;
  baseScale: number;
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

/** Fallback art when an emoji glyph is unavailable: a lit disc in the
 *  item's own colour. The halo already carries the meaning, so a round
 *  played without emoji support is still playable. */
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

export class CloudburstScene extends GameSceneBase {
  protected declare control: CloudburstControl;
  private scoreText!: Phaser.GameObjects.Text;
  private levelText: Phaser.GameObjects.Text | null = null;
  private errorText: Phaser.GameObjects.Text | null = null;
  private backdrop: Phaser.GameObjects.Image | null = null;
  private flash: Phaser.GameObjects.Rectangle | null = null;
  private life: BackgroundLife | null = null;

  private items: Item[] = [];
  private lastSpawnAt = -1;
  private nextRegion: "same" | "across" = "same";
  /** Running totals; the result object is assembled from these at the
   *  end rather than kept in sync twice. */
  private res: CloudburstResult = blankCloudburstResult(1);
  private spawnedDrops = 0;
  private spawnedBolts = 0;
  private speedMult = 1;
  private lastReactionMs: number | null = null;

  constructor() {
    super("cloudburst");
  }

  create() {
    const s = this.control.visualScale;
    this.res = blankCloudburstResult(this.control.level.id);

    // ── Art. Every texture is generated; nothing is downloaded.
    const w = Math.round(this.scale.width);
    const h = Math.round(this.scale.height);
    if (makeSkyTexture(this, "cb-sky", w, h, Math.floor(Math.random() * 1e6))) {
      this.backdrop = this.add
        .image(0, 0, "cb-sky")
        .setOrigin(0, 0)
        .setDisplaySize(w, h)
        .setDepth(-10);
    } else {
      // Texture creation failed — a flat wash still hides the camera.
      this.add.rectangle(0, 0, w, h, 0x44607a).setOrigin(0, 0).setDepth(-10);
    }
    this.canvasSize = { w, h };

    makeSoftDotTexture(this, "fx-dot");
    makeRingTexture(this, "fx-ring");
    if (!makeGlyphTexture(this, "cb-drop", DROP)) {
      makeDiscTexture(this, "cb-drop", DROP_HALO);
    }
    if (!makeGlyphTexture(this, "cb-bolt", BOLT)) {
      makeDiscTexture(this, "cb-bolt", BOLT_HALO);
    }
    if (!makeGlyphTexture(this, "cursor", "✋")) {
      makeDiscTexture(this, "cursor", 0x22d3ee);
    }

    // ── Drifting cloud layer, depth 1: above the sky, far below the
    //    items. Tween-driven, so there is no per-frame cost.
    if (makeCloudTexture(this, "cb-cloud", 7)) {
      for (let i = 0; i < CLOUD_COUNT; i++) this.spawnCloud(true);
    }

    // Birds only — BackgroundLife skips any animal whose glyph is null,
    // so the orchard's monkey is simply not asked for.
    const birdKey = makeGlyphTexture(this, "fx-bird", BIRD) ? "fx-bird" : null;
    this.life = new BackgroundLife(this, { monkey: null, bird: birdKey });

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

    this.levelText = this.add
      .text(this.scale.width * 0.04, this.scale.height * 0.03, "", {
        fontFamily: "system-ui, sans-serif",
        fontSize: "16px",
        color: "#ffffff",
        stroke: "#000000",
        strokeThickness: 3,
      })
      .setDepth(30);
    this.levelText.setText(this.control.level.label);

    this.scoreText = this.add
      .text(this.scale.width * 0.96, this.scale.height * 0.03, "0", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${hud}px`,
        color: "#7dd3fc",
        stroke: "#000000",
        strokeThickness: Math.max(2, hud * 0.1),
      })
      .setOrigin(1, 0)
      .setDepth(30);

    // Error tally, under the catch count and in the bolt's own colour,
    // so the two numbers are never confused with each other.
    this.errorText = this.add
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

    // Full-screen tint for the error flash. Invisible until a bolt is
    // touched; one object rather than one per event.
    this.flash = this.add
      .rectangle(0, 0, this.scale.width, this.scale.height, 0xff3b30, 1)
      .setOrigin(0, 0)
      .setDepth(35)
      .setAlpha(0);

    // ── "Hand lost" banner. Hidden until needed. Built here so the
    //    message never has to be created mid-round.
    const lost = Math.round(this.unit * 0.062 * s);
    this.lostBand = this.add
      .rectangle(
        this.scale.width / 2,
        this.scale.height / 2,
        this.scale.width,
        this.unit * 0.3,
        0x000000,
        0.72,
      )
      .setDepth(40)
      .setVisible(false);
    this.lostText = this.add
      .text(
        this.scale.width / 2,
        this.scale.height / 2,
        "We can't see your hand\nstep back into view",
        {
          fontFamily: "system-ui, sans-serif",
          fontSize: `${lost}px`,
          color: "#ffffff",
          align: "center",
          stroke: "#000000",
          strokeThickness: Math.max(2, lost * 0.08),
        },
      )
      .setOrigin(0.5)
      .setDepth(41)
      .setVisible(false);

    // NOTE: startedAt / lastFrameAt / lastSpawnAt are deliberately left
    // at -1 here and seeded on the first update frame instead — see
    // GameSceneBase for why this must not use this.time.now.
    this.layout();
    this.palmBase = { ...this.control.state.palmCounts };

    const d = this.control.debug;
    d.sceneState = "created";
    d.texturesOk = this.textures.exists("cb-drop") && this.textures.exists("cb-bolt");
    d.boxN = {
      x0: this.control.box.xLo,
      x1: this.control.box.xHi,
      y0: this.control.box.yLo,
      y1: this.control.box.yHi,
    };
    // Created here, in the order the debug panel should print them.
    d.extra["items spawned"] = "drops 0 / bolts 0";
    d.extra["speed"] = "1.00x";
    d.extra["last reaction"] = "none yet";
    d.extra["lightning touched"] = "0";
  }

  /** The first update frame anchors the spawn timer to the same clock,
   *  one full gap behind so the first item appears immediately. */
  protected onClockSeeded(time: number): void {
    this.lastSpawnAt = time - this.control.level.spawnGapMs - 1;
  }

  protected finishRound(): void {
    const c = this.control;
    this.life?.destroy();
    this.life = null;
    c.onFinish({ ...this.res, level: c.level.id });
  }

  /**
   * Position and size everything anchored to the canvas.
   *
   * Items are NOT handled here: they carry cover-normalised
   * coordinates and are re-mapped from the live object-cover geometry
   * every frame, so a resize keeps them in the patient's reach on their
   * own — the same contract Fruit Harvest's fruit use.
   */
  protected layout(): void {
    const w = this.scale.width;
    const h = this.scale.height;
    const u = Math.min(w, h);
    const s = this.control.visualScale;
    this.canvasSize = { w, h };

    this.backdrop?.setDisplaySize(w, h);
    this.flash?.setPosition(0, 0).setSize(w, h);
    this.cursor?.setDisplaySize(u * 0.12 * s, u * 0.12 * s);

    const hud = Math.round(u * 0.095 * s);
    this.timerText
      ?.setPosition(w * 0.04, h * 0.03)
      .setFontSize(hud)
      .setStroke("#000000", Math.max(2, hud * 0.1));
    this.scoreText
      ?.setPosition(w * 0.96, h * 0.03)
      .setFontSize(hud)
      .setStroke("#000000", Math.max(2, hud * 0.1));
    const err = Math.round(u * 0.04 * s);
    this.errorText
      ?.setPosition(w * 0.96, h * 0.03 + hud * 1.05)
      .setFontSize(err);

    this.lostBand?.setPosition(w / 2, h / 2).setSize(w, u * 0.3);
    const lost = Math.round(u * 0.062 * s);
    this.lostText
      ?.setPosition(w / 2, h / 2)
      .setFontSize(lost)
      .setStroke("#000000", Math.max(2, lost * 0.08));

    this.fpsText?.setPosition(w * 0.02, h * 0.82).setFontSize(Math.round(u * 0.032));
    const lvl = Math.round(u * 0.035 * s);
    this.levelText
      ?.setPosition(w * 0.04, h * 0.03 + hud * 1.05)
      .setFontSize(lvl)
      .setStroke("#000000", Math.max(2, lvl * 0.1));
  }

  update(time: number) {
    const c = this.control;

    // Clocks and the hand-lost pause. Null means the round is over.
    // Named `frame` rather than `f`, which below means an Item.
    const frame = this.beginFrame(time);
    if (!frame) return;
    const { dtMs, lost } = frame;

    if (!lost) this.life?.update(time);

    this.syncCanvasSize();

    if (this.advanceRoundClock(frame)) return;

    this.updateCursor(frame);

    // ── Speed ramp. Linear on PLAYED time, so a round that was held
    //    while the patient stepped back still ends at the same speed it
    //    would have reached otherwise.
    const t = Math.min(1, Math.max(0, frame.elapsedMs / ROUND_MS));
    this.speedMult = 1 + (SPEED_RAMP_TO - 1) * t;

    // ── Spawn. Held while the hand is lost: a new item would only fall
    //    past unseen and count against a patient who could not see to
    //    move. The gap is nudged forward so nothing bursts out at once
    //    the moment they come back into view.
    if (lost) {
      this.lastSpawnAt += dtMs;
      for (const it of this.items) it.bornAt += dtMs;
    } else if (
      this.items.length < MAX_ITEMS
      && time - this.lastSpawnAt > c.level.spawnGapMs
    ) {
      this.spawn(time);
      this.lastSpawnAt = time;
    }

    if (!lost) this.stepItems(frame);

    // ── Diagnostics (?gamedebug=1). Everything generic is the base's;
    //    these are Cloudburst's own rows.
    this.writeCoreDebug(frame);
    const d = c.debug;
    d.extra["items spawned"] =
      `drops ${this.spawnedDrops} / bolts ${this.spawnedBolts}`;
    d.extra["speed"] = `${this.speedMult.toFixed(2)}x`;
    d.extra["last reaction"] = this.lastReactionMs === null
      ? "none yet"
      : `${Math.round(this.lastReactionMs)} ms`;
    d.extra["lightning touched"] = String(this.res.lightningTouched);
    if (this.fpsText) {
      this.fpsText.setText(
        `fps ${d.fps} min ${d.fpsMin}  pose ${d.poseHz}Hz  `
        + `cutoff ${d.cutoffHz}Hz  lag ${d.lagPx}px  `
        + `palm ${d.palmSource} `
        + `(h${d.palmCounts.hand}/e${d.palmCounts.elbow}/w${d.palmCounts.wrist})\n`
        + `arm ${d.armLenPx}px  headroom ${d.headroomPx}px  `
        + `ratio ${d.headroomRatio}\n`
        + `speed ${this.speedMult.toFixed(2)}x  on screen ${this.items.length}  `
        + `caught ${this.res.caught}  zapped ${this.res.lightningTouched}`,
      );
    }
  }

  /** 1 for the first half of the round, 2 for the second. */
  private half(): 1 | 2 {
    return this.control.remainingMs > ROUND_MS / 2 ? 1 : 2;
  }

  /** Top and bottom of the fall, in cover-normalised y. */
  private get fallTop(): number {
    return this.control.box.yLo - SPAWN_LEAD;
  }

  private get fallBottom(): number {
    return this.control.box.yHi;
  }

  private spawn(time: number) {
    const c = this.control;
    const region = this.nextRegion;
    const zone: CloudburstZone = region === "same" ? "same_side" : "across";
    // Alternate the halves so the across-midline lane is never crowded
    // out by chance — the same rule Fruit Harvest uses for its zones.
    this.nextRegion = region === "same" ? "across" : "same";

    // Only the x is used. spawnPoint already splits the calibrated box
    // into a same-side and an across-the-midline half, with its own
    // fallback for a patient whose box never crossed the midline.
    const p = spawnPoint(c.box, region, Math.random);
    if (!Number.isFinite(p.nx)) {
      c.debug.extra["items spawned"] = "REJECTED: non-finite lane";
      return;
    }

    const kind: ItemKind = Math.random() < c.level.lightningFraction ? "bolt" : "drop";
    if (kind === "drop") this.spawnedDrops += 1;
    else this.spawnedBolts += 1;

    const cover = c.state.cover;
    const size = this.unit * BASE_ITEM_FRACTION * c.level.itemScale * c.visualScale;
    const ny = this.fallTop;
    const x = cover.dispW > 0 ? cover.offX + p.nx * cover.dispW : this.scale.width / 2;
    const y = cover.dispH > 0 ? cover.offY + ny * cover.dispH : 0;

    // Halo first, so it sits behind the glyph. It is the thing the
    // patient actually reads at 2 m.
    let halo: Phaser.GameObjects.Image | null = null;
    if (this.textures.exists("fx-dot")) {
      halo = this.add
        .image(x, y, "fx-dot")
        .setDisplaySize(size * HALO_SCALE, size * HALO_SCALE)
        .setTint(kind === "drop" ? DROP_HALO : BOLT_HALO)
        .setAlpha(kind === "drop" ? DROP_HALO_ALPHA : BOLT_HALO_ALPHA)
        .setDepth(9);
    }
    const img = this.add
      .image(x, y, kind === "drop" ? "cb-drop" : "cb-bolt")
      .setDisplaySize(size, size)
      .setDepth(10);

    this.items.push({
      img,
      halo,
      kind,
      nx: p.nx,
      ny,
      bornAt: time,
      reactedAt: null,
      dying: false,
      zone,
      baseScale: img.scaleX,
    });
  }

  /**
   * Advance every item, test the cursor, and retire the ones that leave.
   *
   * Speed is derived from the level's fall TIME and the patient's own
   * reach height, so "3000 ms to cross" means the same on any screen
   * and for any body — rather than a pixel velocity that would be a
   * different game on a tall monitor.
   */
  private stepItems(frame: { time: number; dt: number }) {
    const c = this.control;
    const cover = c.state.cover;
    const span = this.fallBottom - this.fallTop;
    const perSec = span > 0
      ? (span / (c.level.fallMs / 1000)) * this.speedMult
      : 0;
    const hitR = this.unit * BASE_HIT_FRACTION * c.level.itemScale * c.visualScale;

    // The miss line is the bottom of the reach box, but never below the
    // drawn ground: on a patient whose calibrated box reaches low, the
    // untrimmed line would let an item sink into the ground and sit
    // there for a moment before being counted, which reads as a bug.
    const boxBottomY = cover.dispH > 0
      ? cover.offY + this.fallBottom * cover.dispH
      : this.scale.height;
    const missY = Math.min(boxBottomY, this.scale.height * SKY_GROUND);

    // Cursor velocity, for the reaction stamp below. This is the
    // filter's own low-passed velocity — the same one the base uses to
    // extrapolate — so the reaction is judged on the motion the patient
    // actually made, not on re-differenced noise.
    const { vx, vy } = this.filter.velocity;
    const deadband = PREDICT_DEADBAND * this.unit;

    for (let i = this.items.length - 1; i >= 0; i--) {
      const it = this.items[i];
      if (it.dying) continue;

      it.ny += perSec * frame.dt;
      if (cover.dispW > 0) {
        it.img.setPosition(
          cover.offX + it.nx * cover.dispW,
          cover.offY + it.ny * cover.dispH,
        );
        it.halo?.setPosition(it.img.x, it.img.y);
      }

      // ── Reaction stamp, drops only.
      //
      // The rule: the component of the cursor's velocity POINTING AT
      // this item must exceed the same deadband the cursor prediction
      // uses. Projecting rather than just checking speed means a hand
      // sweeping the other way does not count, and reusing the existing
      // deadband means "moving" has one definition in this codebase.
      //
      // KNOWN LIMIT: each item is judged independently, so one sweep
      // toward the nearest drop can also stamp a second drop that
      // happens to lie in roughly the same direction. With up to three
      // items on screen that will sometimes flatter the average.
      if (
        it.kind === "drop"
        && it.reactedAt === null
        && c.state.usable
        && this.cursorSeeded
      ) {
        const dx = it.img.x - this.cursor.x;
        const dy = it.img.y - this.cursor.y;
        const dist = Math.hypot(dx, dy) || 1;
        const toward = (vx * dx + vy * dy) / dist;
        if (toward > deadband) it.reactedAt = frame.time;
      }

      // ── Hit test, against the DRAWN cursor — whatever the patient
      //    sees on screen is exactly what touches the item.
      if (c.state.usable && this.cursorSeeded) {
        const d = Math.hypot(this.cursor.x - it.img.x, this.cursor.y - it.img.y);
        if (d < hitR) {
          if (it.kind === "drop") this.catchDrop(it, i);
          else this.zap(it, i);
          continue;
        }
      }

      // ── Left the reach area.
      if (it.img.y > missY) {
        if (it.kind === "drop") {
          this.res.dropsMissed += 1;
          this.bumpHalf(false);
          c.audio.miss();
        } else {
          // Not touching a bolt IS the correct answer — it is the whole
          // point of the item, so it is counted, not ignored.
          this.res.lightningAvoided += 1;
        }
        this.retire(it, i);
      }
    }
  }

  private bumpHalf(hit: boolean) {
    const h = this.half() === 1 ? this.res.firstHalf : this.res.secondHalf;
    h.total += 1;
    if (hit) h.hit += 1;
  }

  private catchDrop(it: Item, index: number) {
    it.dying = true;
    this.items.splice(index, 1);
    const c = this.control;
    this.res.caught += 1;
    this.res.zoneHits[it.zone] += 1;
    this.bumpHalf(true);
    if (it.reactedAt !== null) {
      const ms = it.reactedAt - it.bornAt;
      this.lastReactionMs = ms;
      this.res.reactionSecs.push(Math.round(ms) / 1000);
    }
    c.caught = this.res.caught;
    this.scoreText.setText(String(this.res.caught));
    c.audio.harvest();

    this.tweens.killTweensOf(it.img);
    if (it.halo) this.tweens.killTweensOf(it.halo);

    // Splash: the halo expands and fades while the glyph shrinks into
    // it. Cheap — two tweens, no emitter.
    if (it.halo) {
      this.tweens.add({
        targets: it.halo,
        displayWidth: it.halo.displayWidth * 1.9,
        displayHeight: it.halo.displayHeight * 1.9,
        alpha: 0,
        duration: SPLASH_MS,
        ease: "Cubic.easeOut",
        onComplete: () => it.halo?.destroy(),
      });
    }
    if (this.textures.exists("fx-ring")) {
      const ring = this.add
        .image(it.img.x, it.img.y, "fx-ring")
        .setDisplaySize(it.img.displayWidth, it.img.displayHeight)
        .setTint(DROP_HALO)
        .setDepth(11);
      this.tweens.add({
        targets: ring,
        displayWidth: it.img.displayWidth * 2.4,
        displayHeight: it.img.displayHeight * 2.4,
        alpha: 0,
        duration: SPLASH_MS,
        ease: "Cubic.easeOut",
        onComplete: () => ring.destroy(),
      });
    }
    this.tweens.add({
      targets: it.img,
      scaleX: it.baseScale * 0.2,
      scaleY: it.baseScale * 0.2,
      alpha: 0,
      duration: SPLASH_MS,
      ease: "Cubic.easeIn",
      onComplete: () => it.img.destroy(),
    });
  }

  private zap(it: Item, index: number) {
    it.dying = true;
    this.items.splice(index, 1);
    const c = this.control;
    this.res.lightningTouched += 1;
    c.lightningTouched = this.res.lightningTouched;
    this.errorText?.setText(`⚡ ${this.res.lightningTouched}`);
    c.audio.error();

    // Brief full-screen flash. Short and low-alpha on purpose: this is
    // feedback, not a penalty screen, and the round keeps running.
    if (this.flash) {
      this.tweens.killTweensOf(this.flash);
      this.flash.setAlpha(0.28);
      this.tweens.add({
        targets: this.flash,
        alpha: 0,
        duration: ZAP_MS,
        ease: "Quad.easeOut",
      });
    }

    this.tweens.killTweensOf(it.img);
    if (it.halo) this.tweens.killTweensOf(it.halo);
    if (it.halo) {
      this.tweens.add({
        targets: it.halo,
        alpha: 0,
        duration: ZAP_MS,
        onComplete: () => it.halo?.destroy(),
      });
    }
    // The bolt vanishes rather than falling on: it has been answered.
    this.tweens.add({
      targets: it.img,
      scaleX: it.baseScale * 1.5,
      scaleY: it.baseScale * 1.5,
      alpha: 0,
      duration: ZAP_MS,
      ease: "Quad.easeOut",
      onComplete: () => it.img.destroy(),
    });
  }

  /** Left the play area untouched: fade out without ceremony. */
  private retire(it: Item, index: number) {
    it.dying = true;
    this.items.splice(index, 1);
    this.tweens.killTweensOf(it.img);
    if (it.halo) this.tweens.killTweensOf(it.halo);
    this.tweens.add({
      targets: it.img,
      alpha: 0,
      duration: FADE_MS,
      onComplete: () => it.img.destroy(),
    });
    if (it.halo) {
      this.tweens.add({
        targets: it.halo,
        alpha: 0,
        duration: FADE_MS,
        onComplete: () => it.halo?.destroy(),
      });
    }
  }

  /** One drifting cloud, tweened across and replaced by a fresh one
   *  when it leaves, so the count stays constant. `initial` spreads the
   *  first set across the sky instead of queueing them all at the left
   *  edge. */
  private spawnCloud(initial: boolean) {
    const w = this.scale.width;
    const h = this.scale.height;
    const u = this.unit;
    const size = u * (0.34 + Math.random() * 0.26);
    const y = h * (0.05 + Math.random() * 0.26);
    const startX = initial ? Math.random() * w : -size;
    const img = this.add
      .image(startX, y, "cb-cloud")
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
}
