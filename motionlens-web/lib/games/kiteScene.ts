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
import { pct, type KiteResult } from "@/lib/games/kiteMetrics";
import { KiteSession } from "@/lib/games/kiteSession";
import { MovementAnalyser } from "@/lib/games/kiteMovement";
import {
  amplitudeEnvelope,
  centreNy,
  halfNyAt,
  makeCorridor,
  requiredHandSpeed,
  type Corridor,
} from "@/lib/games/kiteCorridor";
import {
  FALL_AFTER_MS,
  RESPAWN_MS,
  ROUND_MS,
  TUMBLE_MS,
} from "@/lib/games/kiteLevels";
import { makeMeadowTexture, MEADOW_GROUND } from "@/lib/games/meadowScene";
import { makeCloudTexture } from "@/lib/games/skyScene";
import { makeSoftDotTexture } from "@/lib/games/fruitEffects";
import { BackgroundLife } from "@/lib/games/backgroundLife";

const BIRD = "🐦";

/** Glyph texture size. Generous so a large sprite stays crisp. */
const GLYPH_TEX = 256;

// -- Palette.
//
// The first version put a light-blue ribbon and a blue kite on a blue
// sky, and on camera neither could be found from 2 m. NOTHING in the
// play area is blue any more: the sky and the clouds own that end of
// the spectrum, so the corridor is gold and the kite is red.
//
// The corridor also has to separate from WHITE cloud as well as from
// blue sky, which a fill alone cannot do at any alpha -- so it carries
// solid orange rails, and a thin dark line outside those, which is what
// gives it an edge against anything behind it.

/** Semi-opaque warm gold. Alpha kept below half so a cloud passing
 *  behind still reads as a cloud rather than as part of the ribbon. */
const RIBBON_FILL = 0xfbbf24;
const RIBBON_FILL_ALPHA = 0.45;
/** Solid orange rails down both sides. */
const RIBBON_EDGE = 0xf97316;
const RIBBON_EDGE_W = 0.009;
/** A thin dark line just outside the orange, so the band still has an
 *  edge when it crosses a white cloud. */
const RIBBON_OUTLINE = 0x5c3a00;
const RIBBON_OUTLINE_ALPHA = 0.55;
const RIBBON_OUTLINE_W = 0.014;

/** Horizontal samples across the ribbon. Enough that the curve reads as
 *  smooth, few enough that redrawing it every frame is free. */
const RIBBON_COLS = 48;

// -- Wind streaks: faint white dashes flowing right to left inside the
//    band, so it reads as moving air rather than as a painted stripe.
const STREAK_COUNT = 22;
const STREAK_TINT = 0xffffff;
const STREAK_ALPHA = 0.5;
/** Canvas widths per second. Faster than the corridor itself scrolls,
 *  which is what sells the flow. */
const STREAK_SPEED = 0.55;
const STREAK_LEN_MIN = 0.05;
const STREAK_LEN_MAX = 0.12;

/** Kite glow: green inside the wind, amber outside. It is a separate
 *  sprite BEHIND the kite and is never tinted onto the body, so the
 *  kite stays unmistakably red whatever the state. */
const GLOW_IN = 0x4ade80;
const GLOW_OUT = 0xfbbf24;
const GLOW_SCALE = 1.7;
const GLOW_ALPHA = 0.7;
/** Greyed while the hand is lost, so the kite stays visible behind the
 *  message without looking live. */
const KITE_LOST_TINT = 0x8b8b8b;

/** The instruction shown over the opening seconds of the round. */
const INTRO_TEXT = "Keep the kite inside the wind";
const INTRO_MS = 2600;
const INTRO_FADE_MS = 600;

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

/**
 * The kite, drawn rather than rasterised from an emoji.
 *
 * The kite glyph is BLUE on most platforms, which is the exact problem
 * this replaces, and a sprite tint cannot fix one colour without
 * staining the whole thing. A diamond costs a few lines and looks the
 * same on every device: a red body with a white outline and white
 * spars, which holds up against blue sky, white cloud and green grass
 * alike.
 */
function makeKiteTexture(scene: Phaser.Scene, key: string): boolean {
  if (scene.textures.exists(key)) return true;
  const tex = scene.textures.createCanvas(key, GLYPH_TEX, GLYPH_TEX);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  ctx.clearRect(0, 0, GLYPH_TEX, GLYPH_TEX);
  const c = GLYPH_TEX / 2;
  // Short of the texture edge, so the outline has room.
  const r = GLYPH_TEX * 0.4;
  const wide = r * 0.74;

  const path = () => {
    ctx.beginPath();
    ctx.moveTo(c, c - r);
    ctx.lineTo(c + wide, c - r * 0.12);
    ctx.lineTo(c, c + r);
    ctx.lineTo(c - wide, c - r * 0.12);
    ctx.closePath();
  };

  // Body: red at the top shading deeper below, so it has some form
  // rather than reading as a flat cut-out.
  const g = ctx.createLinearGradient(c, c - r, c, c + r);
  g.addColorStop(0, "#ff5a4d");
  g.addColorStop(0.45, "#ef4444");
  g.addColorStop(1, "#b91c1c");
  path();
  ctx.fillStyle = g;
  ctx.fill();

  // White outline — the part that does the work against a dark or a
  // busy background.
  path();
  ctx.strokeStyle = "#ffffff";
  ctx.lineJoin = "round";
  ctx.lineWidth = GLYPH_TEX * 0.045;
  ctx.stroke();

  // Spars, so the shape reads as a kite and not a lozenge.
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = GLYPH_TEX * 0.022;
  ctx.beginPath();
  ctx.moveTo(c, c - r * 0.86);
  ctx.lineTo(c, c + r * 0.86);
  ctx.moveTo(c - wide * 0.86, c - r * 0.12);
  ctx.lineTo(c + wide * 0.86, c - r * 0.12);
  ctx.stroke();

  tex.refresh();
  return true;
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
  /** Owns every accumulated number and the inside/fall rules. */
  private session!: KiteSession;
  /** Kite height in canvas px, from the level and the reach — never
   *  from the display scale. */
  private kitePx = 0;
  /** Live readings for the overlay. */
  private lastFactor = 0;
  private lastHalfPx = 0;
  private lastNeedSpeed = 0;

  /** Distance scrolled, in canvas widths. The ribbon's whole shape is
   *  a function of this. */
  private scrollU = 0;
  /** Where the string is tied. */
  private anchorX = 0;
  private anchorY = 0;

  /** While falling the patient has nothing to steer, so neither the
   *  corridor clock nor the movement trace runs. */
  private falling = false;
  private fallEndsAt = -1;
  private wasInside = true;
  /** Live deviation, for the debug overlay. */
  private lastDev = 0;
  /** Wind streaks inside the band. `u` is in canvas widths, `v` is an
   *  offset from the centreline as a fraction of the half-width. */
  private streaks: { u: number; v: number; len: number }[] = [];
  private introText: Phaser.GameObjects.Text | null = null;
  private introDone = false;
  /** The point the game actually judges: the palm, drawn so the patient
   *  can see what is being asked of them. Because the corridor is a
   *  fixed multiple of the kite, "this dot is inside" and "the kite
   *  looks inside" agree. */
  private anchorDot: Phaser.GameObjects.Image | null = null;

  constructor() {
    super("kite-flying");
  }

  create() {
    // `s` scales TEXT AND HUD ONLY. The kite and the corridor are sized
    // from the level and the patient's reach — see kiteLevels.ts.
    const s = this.control.visualScale;

    // Arm lengths per unit of ny, measured now, so the level's hand
    // speed ceiling can be enforced in units that mean the same for
    // every patient at every distance from the camera.
    const st = this.control.state;
    const nyToArm = st.armLenPx > 1 && st.cover.dispH > 0
      ? st.cover.dispH / st.armLenPx
      : 0;
    this.corridor = makeCorridor(this.control.box, this.control.level, nyToArm);
    this.session = new KiteSession(this.corridor, FALL_AFTER_MS);

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
    makeKiteTexture(this, "kf-kite");

    if (makeCloudTexture(this, "kf-cloud", 31)) {
      for (let i = 0; i < CLOUD_COUNT; i++) this.spawnCloud(true);
    }
    // Birds only — BackgroundLife skips any animal whose glyph is null.
    const birdKey = makeGlyphTexture(this, "fx-bird", BIRD) ? "fx-bird" : null;
    this.life = new BackgroundLife(this, { monkey: null, bird: birdKey });

    // The ribbon sits under the kite but OVER the clouds (depth 1) and
    // the birds (depth 2), so nothing is ever drawn across the corridor
    // or the kite.
    this.ribbon = this.add.graphics().setDepth(4);
    this.string = this.add.graphics().setDepth(9);
    this.streaks = Array.from({ length: STREAK_COUNT }, () => ({
      u: Math.random() * 1.4,
      v: (Math.random() * 2 - 1) * 0.8,
      len: STREAK_LEN_MIN + Math.random() * (STREAK_LEN_MAX - STREAK_LEN_MIN),
    }));

    this.glow = this.textures.exists("fx-dot")
      ? this.add
        .image(w / 2, h / 2, "fx-dot")
        .setTint(GLOW_IN)
        .setAlpha(0)
        .setDepth(10)
      : null;
    this.kitePx = this.kiteSizePx();
    this.kite = this.add
      .image(w / 2, h / 2, "kf-kite")
      .setDisplaySize(this.kitePx, this.kitePx)
      .setDepth(11);

    // The anchor: the point the corridor test actually uses.
    this.anchorDot = this.textures.exists("fx-dot")
      ? this.add
        .image(w / 2, h / 2, "fx-dot")
        .setDisplaySize(this.anchorPx(), this.anchorPx())
        .setTint(0xffffff)
        .setAlpha(0.95)
        .setDepth(12)
      : null;

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
      // Lighter than the other two games' banner: here the corridor
      // and the kite are meant to stay readable behind the message.
      .rectangle(w / 2, h / 2, w, this.unit * 0.3, 0x000000, 0.45)
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

    // Opening instruction. The corridor and the kite are already drawn
    // behind it, so the patient reads what to do while looking at the
    // thing they have to do it to.
    //
    // NOTE ON PLACEMENT: the brief asked for this during the shell's
    // 3-2-1, but the Phaser canvas is not mounted until the play phase
    // — showing it there would mean changing GameShell, which is shared
    // with the other two games and out of scope here. It runs over the
    // opening seconds of the round instead. The clock is NOT paused: the
    // corridor is live and playable from the first frame, so this is an
    // overlay, not a delay.
    const intro = Math.round(this.unit * 0.062 * s);
    this.introText = this.add
      .text(w / 2, h * 0.2, INTRO_TEXT, {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${intro}px`,
        color: "#ffffff",
        align: "center",
        stroke: "#000000",
        strokeThickness: Math.max(3, intro * 0.14),
      })
      .setOrigin(0.5)
      .setDepth(32);

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
    d.extra["lead-in"] = "—";
    d.extra["amplitude now"] = "—";
    d.extra["kite height"] = "—";
    d.extra["corridor width"] = "—";
    d.extra["anchor inside"] = "—";
    d.extra["deviation"] = "—";
    d.extra["needs hand speed"] = "—";
    d.extra["amplitude"] = "—";
    d.extra["raw vs drawn"] = "—";
    d.extra["velocity peaks"] = "0";
  }

  protected finishRound(): void {
    const c = this.control;
    this.life?.destroy();
    this.life = null;
    const result: KiteResult = {
      ...this.session.totals,
      level: c.level.id,
      hand: c.hand,
      scoredMs: Math.max(0, c.roundMs - this.corridor.scoredFromMs),
      peaksPerSec: this.analyser.peaksPerSec,
      peakCount: this.analyser.peakCount,
      movingSec: Math.round(this.analyser.movingSec * 10) / 10,
    };
    c.onFinish(result);
  }

  protected layout(): void {
    const w = this.scale.width;
    const h = this.scale.height;
    const u = Math.min(w, h);
    const s = this.control.visualScale;
    this.canvasSize = { w, h };

    this.backdrop?.setDisplaySize(w, h);
    this.kitePx = this.kiteSizePx();
    this.kite?.setDisplaySize(this.kitePx, this.kitePx);
    this.anchorDot?.setDisplaySize(this.anchorPx(), this.anchorPx());

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
    this.introText
      ?.setPosition(w / 2, h * 0.2)
      .setFontSize(Math.round(u * 0.062 * s));
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
    // The lane's shape this frame. Set BEFORE anything reads the
    // corridor, so the ribbon that is drawn and the lane that is scored
    // are the same lane.
    this.corridor.ampScale = amplitudeEnvelope(this.corridor, frame.elapsedMs);

    if (!lost) {
      this.scrollU += c.level.scrollPerSec * dt;
      this.stepStreaks(dt);
    }
    this.drawRibbon();

    if (!this.introDone && frame.elapsedMs > INTRO_MS) {
      this.introDone = true;
      const t = this.introText;
      if (t) {
        this.tweens.add({
          targets: t,
          alpha: 0,
          duration: INTRO_FADE_MS,
          onComplete: () => t.setVisible(false),
        });
      }
    }

    if (lost) {
      // Nothing to measure and nothing to steer. Break the movement
      // trace so the gap is not read as one enormous slow movement.
      this.analyser.breakTrace();
      if (this.fallEndsAt >= 0) this.fallEndsAt += dtMs;
      // The corridor keeps being drawn above and the kite stays where
      // it was, greyed — so the patient can see what they are coming
      // back to rather than a black band over an empty sky.
      this.kite.setTint(KITE_LOST_TINT);
      this.glow?.setTint(KITE_LOST_TINT).setAlpha(GLOW_ALPHA * 0.4);
      this.writeDebug(frame, false);
      return;
    }
    this.kite.clearTint();

    // ── The kite. It follows the DRAWN cursor, because what the
    //    patient sees has to be what the game responds to.
    if (this.falling) {
      if (time >= this.fallEndsAt) this.respawn();
    } else {
      this.kite.setPosition(this.cursor.x, this.cursor.y);
    }
    this.glow?.setPosition(this.kite.x, this.kite.y);
    this.anchorDot
      ?.setPosition(this.cursor.x, this.cursor.y)
      .setVisible(!this.falling);
    this.drawString();

    const cover = c.state.cover;
    const W = this.scale.width || 1;

    // ── One frame through the session, which owns every rule and
    //    every total. The kite point is the DRAWN anchor and the palm
    //    point is the RAW one; see lib/games/kiteSession.ts.
    const toPoint = (x: number, y: number) => ({
      worldX: this.scrollU + x / W,
      ny: (y - cover.offY) / cover.dispH,
    });
    // The straight lead-in and the fade-in are not a tracking task, so
    // nothing in them is scored. The kite still flies and can still
    // fall — that is feedback, and the patient is meant to be flying.
    const scored = frame.elapsedMs >= this.corridor.scoredFromMs;
    const measurable = c.state.live && c.state.inFrame && !this.falling && scored;
    const step = cover.dispH > 0
      ? this.session.step({
        dtMs,
        half: this.half(),
        kite: this.falling ? null : toPoint(this.kite.x, this.kite.y),
        palm: measurable ? toPoint(c.state.palmX, c.state.palmY) : null,
      })
      : null;

    const kiteInside = step?.kiteInside ?? false;
    if (step && !this.falling) {
      this.lastDev = step.devRatio;
      this.lastFactor = step.factor;
      this.lastHalfPx = step.localHalfNy * 2 * cover.dispH;
      this.lastNeedSpeed = requiredHandSpeed(
        this.corridor,
        this.scrollU + this.kite.x / W,
        c.level.scrollPerSec,
      );
      this.applyKiteMood(kiteInside);
      if (step.fellNow) this.startFall(time);

      if (kiteInside !== this.wasInside) {
        this.wasInside = kiteInside;
        // A short cue on the crossing rather than a tone held while
        // inside: the audio here is blips, and one repeating twice a
        // second would be worse than nothing.
        if (kiteInside) c.audio.harvest();
        else c.audio.miss();
      }
    }

    // The movement analyser takes the raw palm on exactly the frames
    // the session scored, so the two measures cover the same time.
    if (measurable && cover.dispH > 0) {
      this.analyser.push(time, c.state.palmX, c.state.palmY, c.state.armLenPx);
    } else {
      this.analyser.breakTrace();
    }

    const t = this.session.totals;
    c.insidePct = pct(t.insideMs, t.measuredMs);
    c.falls = t.falls;
    this.pctText.setText(`${c.insidePct}%`);
    this.fallText?.setText(t.falls > 0 ? `Falls: ${t.falls}` : "");

    this.writeDebug(frame, kiteInside);
  }

  /**
   * Kite height in canvas px.
   *
   * From the level and the calibrated reach, projected through the live
   * object-cover geometry — and NOT from `visualScale`. Before a pose
   * has arrived there is no reach to measure against, so it falls back
   * to a fraction of the canvas unit just so something is drawn.
   */
  private kiteSizePx(): number {
    const cover = this.control.state.cover;
    if (cover.dispH > 0) return this.corridor.kiteNy * cover.dispH;
    return this.unit * 0.12;
  }

  /**
   * The anchor dot's size.
   *
   * A share of the kite, but never below a fraction of the canvas: the
   * anti-cheat cap can drive the kite down to about 30 px on a 1080
   * screen, and three tenths of that is a speck. The dot marks the
   * point the game actually judges, so it has to stay findable at 2 m
   * whatever the kite ends up as.
   */
  private anchorPx(): number {
    return Math.max(this.kitePx * 0.3, this.unit * 0.018);
  }

  /** 1 for the first half of the round, 2 for the second. */
  private half(): 1 | 2 {
    return this.control.remainingMs > ROUND_MS / 2 ? 1 : 2;
  }

  /**
   * Redraw the ribbon for the current scroll.
   *
   * The half-width varies along the path now, so every pass samples it
   * per column rather than using one number — that is what makes the
   * lane visibly breathe between the level's two factors.
   *
   * Four passes, outermost first: a dark outline, the gold fill, the
   * orange rails, then the wind streaks. The rails and the outline are
   * what make it readable over a white cloud — a translucent fill alone
   * vanishes against one at any alpha that still lets the sky through.
   */
  private drawRibbon(): void {
    const cover = this.control.state.cover;
    const g = this.ribbon;
    g.clear();
    if (cover.dispH <= 0) return;
    const W = this.scale.width || 1;
    const u = this.unit;

    // Sample the centreline and the local half-width once per column
    // and reuse both for every pass.
    const xs: number[] = [];
    const cy: number[] = [];
    const hp: number[] = [];
    for (let i = 0; i <= RIBBON_COLS; i++) {
      const x = (i / RIBBON_COLS) * W;
      const worldX = this.scrollU + x / W;
      xs.push(x);
      cy.push(cover.offY + centreNy(this.corridor, worldX) * cover.dispH);
      hp.push(halfNyAt(this.corridor, worldX) * cover.dispH);
    }

    const rail = (sign: number, colour: number, alpha: number, width: number) => {
      g.lineStyle(width, colour, alpha);
      g.beginPath();
      for (let i = 0; i <= RIBBON_COLS; i++) {
        const y = cy[i] + sign * hp[i];
        if (i === 0) g.moveTo(xs[i], y);
        else g.lineTo(xs[i], y);
      }
      g.strokePath();
    };

    // Dark outline, just outside the rails.
    const ow = u * RIBBON_OUTLINE_W;
    rail(-1, RIBBON_OUTLINE, RIBBON_OUTLINE_ALPHA, ow);
    rail(1, RIBBON_OUTLINE, RIBBON_OUTLINE_ALPHA, ow);

    // Gold fill between the rails.
    g.fillStyle(RIBBON_FILL, RIBBON_FILL_ALPHA);
    g.beginPath();
    for (let i = 0; i <= RIBBON_COLS; i++) {
      const y = cy[i] - hp[i];
      if (i === 0) g.moveTo(xs[i], y);
      else g.lineTo(xs[i], y);
    }
    for (let i = RIBBON_COLS; i >= 0; i--) g.lineTo(xs[i], cy[i] + hp[i]);
    g.closePath();
    g.fillPath();

    // Solid orange rails on top of the fill's own edge.
    const ew = u * RIBBON_EDGE_W;
    rail(-1, RIBBON_EDGE, 1, ew);
    rail(1, RIBBON_EDGE, 1, ew);

    // Wind streaks, inside the band only. Their offset is a fraction of
    // the LOCAL half-width, so they narrow with the lane instead of
    // spilling over the rails where it pinches.
    const at = (x: number) => {
      const worldX = this.scrollU + x / W;
      return {
        c: cover.offY + centreNy(this.corridor, worldX) * cover.dispH,
        h: halfNyAt(this.corridor, worldX) * cover.dispH,
      };
    };
    g.lineStyle(Math.max(1, u * 0.004), STREAK_TINT, STREAK_ALPHA);
    for (const st of this.streaks) {
      const x0 = st.u * W;
      const x1 = (st.u + st.len) * W;
      if (x1 < 0 || x0 > W) continue;
      const a = Math.max(0, x0);
      const b = Math.min(W, x1);
      if (b - a < 1) continue;
      const pa = at(a);
      const pb = at(b);
      g.beginPath();
      g.moveTo(a, pa.c + st.v * pa.h);
      g.lineTo(b, pb.c + st.v * pb.h);
      g.strokePath();
    }
  }

  /** Advance the streaks with the wind and recycle the ones that leave
   *  on the left. Positions are in canvas widths, like the scroll. */
  private stepStreaks(dt: number): void {
    for (const st of this.streaks) {
      st.u -= STREAK_SPEED * dt;
      if (st.u + st.len < 0) {
        st.u = 1 + Math.random() * 0.4;
        st.v = (Math.random() * 2 - 1) * 0.8;
        st.len = STREAK_LEN_MIN + Math.random() * (STREAK_LEN_MAX - STREAK_LEN_MIN);
      }
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

    // A short tail below the kite, hanging the opposite way to the
    // string so the two never lie on top of each other. Drawn in the
    // kite's own red with white bows, and it drifts with the wobble,
    // which is a second cue that something has gone wrong.
    const len = this.kite.displayHeight * 0.75;
    const lean = Math.sin(this.lastFrameAt / 600) * len * 0.22;
    const tx = this.kite.x + lean;
    const ty = this.kite.y + this.kite.displayHeight * 0.5 + len;
    g.lineStyle(Math.max(1, this.unit * 0.005), 0xef4444, 0.9);
    g.beginPath();
    g.moveTo(this.kite.x, this.kite.y + this.kite.displayHeight * 0.45);
    g.lineTo(this.kite.x + lean * 0.5, this.kite.y + this.kite.displayHeight * 0.45 + len * 0.55);
    g.lineTo(tx, ty);
    g.strokePath();
    g.fillStyle(0xffffff, 0.95);
    for (const f of [0.45, 0.85]) {
      const bx = this.kite.x + lean * f;
      const by = this.kite.y + this.kite.displayHeight * 0.45 + len * f;
      g.fillCircle(bx, by, Math.max(1.5, this.unit * 0.008));
    }
  }

  /** Green and swaying, or amber and wobbling. */
  private applyKiteMood(inside: boolean): void {
    // The glow is a separate sprite BEHIND the kite, so the body keeps
    // its red whichever state it is in — the colour change is the halo
    // around it, never the kite itself.
    if (this.glow) {
      const size = this.kite.displayWidth * GLOW_SCALE;
      this.glow
        .setDisplaySize(size, size)
        .setTint(inside ? GLOW_IN : GLOW_OUT)
        .setAlpha(GLOW_ALPHA);
    }

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
    // The session counted the fall on the frame it decided; here we
    // only stop its grace clock and play the tumble.
    this.session.setAway(true);
    this.control.falls = this.session.totals.falls;
    this.fallText?.setText(`Falls: ${this.session.totals.falls}`);
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
    this.session.setAway(false);
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
    const t = this.session.totals;

    const leadLeft =
      Math.max(0, this.corridor.scoredFromMs - frame.elapsedMs) / 1000;
    d.extra["lead-in"] = frame.elapsedMs < this.corridor.leadInMs
      ? `straight, ${((this.corridor.leadInMs - frame.elapsedMs) / 1000).toFixed(1)}s left`
      : leadLeft > 0
        ? `fading in, ${leadLeft.toFixed(1)}s to scoring`
        : "over — scoring";
    d.extra["amplitude now"] =
      `${Math.round(this.corridor.ampScale * 100)}% of full`;
    d.extra["kite height"] = `${Math.round(this.kitePx)} px`
      + ` (${(this.corridor.kiteNy * 100).toFixed(1)}% of reach)`;
    d.extra["corridor width"] = `${Math.round(this.lastHalfPx)} px`
      + ` = ${this.lastFactor.toFixed(2)}x kite`
      + ` (${this.corridor.factorMin}-${this.corridor.factorMax})`;
    d.extra["anchor inside"] = this.falling
      ? "falling"
      : inside ? "yes" : `no (${Math.round(this.session.outsideForMs)} ms)`;
    d.extra["deviation"] = `${(this.lastDev * 100).toFixed(0)}% of local half-width`;
    d.extra["needs hand speed"] = `${this.lastNeedSpeed.toFixed(2)} arm/s`
      + ` (peak ${this.corridor.peakHandSpeedArmPerSec.toFixed(2)},`
      + ` cap ${c.level.maxHandSpeedArmPerSec})`;
    d.extra["amplitude"] =
      `${(this.corridor.ampNy * 100).toFixed(1)}% of reach`
      + ` (limited by ${this.corridor.ampLimitedBy})`;
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
        + `palm ${d.palmSource}
`
        + `kite ${Math.round(this.kitePx)}px  lane ${Math.round(this.lastHalfPx)}px `
        + `= ${this.lastFactor.toFixed(2)}x  needs ${this.lastNeedSpeed.toFixed(2)} arm/s
`
        + `in corridor ${c.insidePct}%  dev ${(this.lastDev * 100).toFixed(0)}%  `
        + `falls ${t.falls}  peaks ${this.analyser.peakCount}`,
      );
    }
  }
}
