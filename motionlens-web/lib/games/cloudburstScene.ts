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
  AIMED_LIGHTNING_FRACTION,
  BASE_HIT_FRACTION,
  BASE_ITEM_FRACTION,
  MAX_ITEMS,
  MIN_LANE_SEPARATION,
  ROUND_MS,
  SPAWN_LEAD,
  SPEED_RAMP_TO,
  STRIKE_BAND_FRACTION,
  STRIKE_FLASH_MS,
  STRIKE_WARN_MS,
} from "@/lib/games/cloudburstLevels";
import { makeCloudTexture, makeSkyTexture, SKY_GROUND } from "@/lib/games/skyScene";
import { StormStrike } from "@/lib/games/stormStrike";
import {
  chooseStrikeBand,
  STRIKE_AIMED_FRACTION,
} from "@/lib/games/cloudburstBand";
import { makeRingTexture, makeSoftDotTexture } from "@/lib/games/fruitEffects";
import { BackgroundLife } from "@/lib/games/backgroundLife";
import { FrogPond } from "@/lib/games/cloudburstFrogs";

const DROP = "💧";
const BOLT = "⚡";
const BIRD = "🐦";
const FROG = "🐸";

/** How many frogs sit on the ground. Two or three: enough to feel
 *  alive, few enough that the eye is pulled down off the play area.
 *  Rolled per round in create(), not once per module load. */
const FROG_MIN = 2;
const FROG_MAX = 3;

// ── Big centre strike. Everything the event DRAWS lives in
//    lib/games/stormStrike.ts; what is left here is the timing, the
//    band, and who was hit.
/** How long after the flash the thunder arrives. Short enough to read
 *  as the same event, long enough to read as sound chasing light. */
const THUNDER_DELAY_MS = 190;

/**
 * Minimum gap between two error tints, in ms.
 *
 * PHOTOSENSITIVITY GUARD. A patient flailing through a cluster of bolts
 * could otherwise retrigger the red tint several times a second. The
 * error is still counted every time and the buzz still plays — only
 * the light is rate-limited, to at most 2.5 per second.
 */
const SAFE_FLASH_GAP_MS = 400;

type StrikePhase = "idle" | "warning" | "striking";

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
  private frogs: FrogPond | null = null;

  private items: Item[] = [];
  private lastSpawnAt = -1;
  private nextRegion: "same" | "across" = "same";
  /** Running totals; the result object is assembled from these at the
   *  end rather than kept in sync twice. */
  private res: CloudburstResult = blankCloudburstResult(1);
  private spawnedDrops = 0;
  private spawnedBolts = 0;
  private aimedBolts = 0;
  private speedMult = 1;
  private lastReactionMs: number | null = null;
  /** Last time the error tint was shown, for the flash rate limit. */
  private lastFlashAt = -1e9;

  // ── Big centre strike.
  private strikePhase: StrikePhase = "idle";
  /** Scene-clock time the next phase change is due. */
  private strikeDueAt = -1;
  /** When the current warning began, for the dodge time. */
  private strikeWarnedAt = -1;
  /** Was the palm inside the band when the warning appeared? Only then
   *  is there a dodge to time. */
  private strikeWasInside = false;
  /** First moment the palm left the band during this warning. */
  private strikeLeftAt: number | null = null;
  /** Band edges in canvas px, fixed for the life of one strike so the
   *  target cannot slide out from under the patient mid-warning. */
  private bandLoX = 0;
  private bandHiX = 0;
  /** The chosen band in nx. Held separately from the pixel edges so a
   *  resize re-projects the same band instead of choosing a new one. */
  private bandCentreNx = 0.5;
  private bandHalfNx = 0.2;
  /** Previous strike's centre, for the "never twice in the same place"
   *  rule. Null until the first strike of the round. */
  private prevBandCentreNx: number | null = null;
  private bandAimed = false;
  private bandShiftNx: number | null = null;
  /** Every pixel of the strike's art. The band itself is no longer
   *  drawn as a shape — the storm IS the marker. */
  private storm: StormStrike | null = null;
  /** Vertical extent of the band, kept alongside bandLoX/bandHiX. */
  private bandTopY = 0;
  private bandBottomY = 0;
  /** update()-clock moment the thunder should follow the flash, or
   *  null when none is pending. */
  private thunderAt: number | null = null;
  /** Set by the debug-only S key or Strike now button; consumed on the
   *  next idle frame. Never set in normal play. */
  private strikeRequested = false;
  private strikeButton: Phaser.GameObjects.Text | null = null;
  private strikeText: Phaser.GameObjects.Text | null = null;
  private strikeCue: Phaser.GameObjects.Text | null = null;

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

    // Frogs on the ground strip. Decoration: no hit test, and they sit
    // below the ground line, which the miss line is clamped to — so a
    // frog is never inside the catch area.
    const frogKey = makeGlyphTexture(this, "cb-frog", FROG) ? "cb-frog" : null;
    this.frogs = new FrogPond(
      this,
      frogKey,
      FROG_MIN + Math.floor(Math.random() * (FROG_MAX - FROG_MIN + 1)),
    );

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

      // ── Review shortcut, debug builds only. Waiting 12-15 s to look
      //    at one strike makes it impossible to iterate on how it
      //    looks, so S — or this button — brings the next one forward.
      //    Neither exists without ?gamedebug=1.
      this.input.keyboard?.on("keydown-S", () => {
        this.strikeRequested = true;
      });
      const btn = this.add
        .text(this.scale.width * 0.02, this.scale.height * 0.76, " ⚡ Strike now ", {
          fontFamily: "ui-monospace, monospace",
          fontSize: `${Math.round(this.unit * 0.03)}px`,
          color: "#0b1220",
          backgroundColor: "#a3e635",
          padding: { x: 8, y: 5 },
        })
        .setDepth(31)
        .setInteractive({ useHandCursor: true });
      btn.on("pointerdown", () => {
        this.strikeRequested = true;
      });
      this.strikeButton = btn;
    }

    // Everything the strike DRAWS lives in lib/games/stormStrike.ts.
    // It reuses the soft white radial the effects module already
    // generates, so no new texture is created per event.
    this.storm = new StormStrike(this, "fx-dot");
    this.strikeText = this.add
      .text(this.scale.width / 2, this.scale.height * 0.16, "Move to the side!", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${Math.round(this.unit * 0.075 * s)}px`,
        color: "#fecaca",
        align: "center",
        stroke: "#000000",
        strokeThickness: Math.max(3, this.unit * 0.008),
      })
      .setOrigin(0.5)
      .setDepth(33)
      .setVisible(false);
    this.strikeCue = this.add
      .text(this.scale.width / 2, this.scale.height * 0.28, "", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${Math.round(this.unit * 0.06 * s)}px`,
        color: "#ffffff",
        align: "center",
        stroke: "#000000",
        strokeThickness: Math.max(3, this.unit * 0.007),
      })
      .setOrigin(0.5)
      .setDepth(34)
      .setAlpha(0);

    // Partial red tint for the error feedback. NOT white and NOT full
    // brightness; rate-limited in zap(). One object, not one per event.
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
    d.extra["lightning share"] =
      `${Math.round(this.control.level.lightningFraction * 100)}%`;
    d.extra["aimed bolts"] = "0";
    d.extra["speed"] = "1.00x";
    d.extra["last reaction"] = "none yet";
    d.extra["lightning touched"] = "0";
    d.extra["next strike"] = "—";
    d.extra["strike band"] = "—";
    d.extra["strike aim"] = "—";
    d.extra["strike shift"] = "first of round";
    d.extra["palm in band"] = "—";
  }

  /** The first update frame anchors every timer to the same clock: the
   *  spawn gap one full gap behind so the first item appears at once,
   *  and the first strike a full interval away so the round does not
   *  open with one. */
  protected onClockSeeded(time: number): void {
    const lv = this.control.level;
    this.lastSpawnAt = time - lv.spawnGapMs - 1;
    this.strikeDueAt = time + this.nextStrikeGap();
  }

  private nextStrikeGap(): number {
    const lv = this.control.level;
    return lv.strikeGapMinMs
      + Math.random() * (lv.strikeGapMaxMs - lv.strikeGapMinMs);
  }

  protected finishRound(): void {
    const c = this.control;
    this.life?.destroy();
    this.life = null;
    this.frogs?.destroy();
    this.frogs = null;
    this.storm?.destroy();
    this.storm = null;
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
    this.strikeButton
      ?.setPosition(w * 0.02, h * 0.76)
      .setFontSize(Math.round(u * 0.03));
    const lvl = Math.round(u * 0.035 * s);
    this.levelText
      ?.setPosition(w * 0.04, h * 0.03 + hud * 1.05)
      .setFontSize(lvl)
      .setStroke("#000000", Math.max(2, lvl * 0.1));

    this.strikeText
      ?.setPosition(w / 2, h * 0.16)
      .setFontSize(Math.round(u * 0.075 * s));
    this.strikeCue
      ?.setPosition(w / 2, h * 0.28)
      .setFontSize(Math.round(u * 0.06 * s));
    this.frogs?.layout();
    // A strike in flight was sized against the old canvas; re-measure
    // the band so the danger zone still matches the patient's reach.
    if (this.strikePhase !== "idle") this.sizeStrikeBand();
  }

  update(time: number) {
    const c = this.control;

    // Clocks and the hand-lost pause. Null means the round is over.
    // Named `frame` rather than `f`, which below means an Item.
    const frame = this.beginFrame(time);
    if (!frame) return;
    const { dtMs, lost } = frame;

    if (!lost) {
      this.life?.update(time);
      this.frogs?.update(time);
    }

    this.syncCanvasSize();

    if (this.advanceRoundClock(frame)) return;

    this.updateCursor(frame);

    // ── Speed ramp. Linear on PLAYED time, so a round that was held
    //    while the patient stepped back still ends at the same speed it
    //    would have reached otherwise.
    const t = Math.min(1, Math.max(0, frame.elapsedMs / ROUND_MS));
    this.speedMult = 1 + (SPEED_RAMP_TO - 1) * t;

    // ── Big centre strike. Runs before spawning, because the warning
    //    and the strike hold normal spawning: the patient is being
    //    asked to move their whole body out of a band, and new items
    //    arriving mid-dodge would only punish them for doing it.
    if (lost) {
      this.strikeDueAt += dtMs;
      // Both ends of the dodge measurement move together, or a pause
      // between the warning and the strike would shorten — and with a
      // dodge already stamped, invert — the recorded time.
      if (this.strikeWarnedAt >= 0) this.strikeWarnedAt += dtMs;
      if (this.strikeLeftAt !== null) this.strikeLeftAt += dtMs;
      if (this.thunderAt !== null) this.thunderAt += dtMs;
    } else {
      this.stepStrike(time);
    }

    // ── Spawn. Held while the hand is lost: a new item would only fall
    //    past unseen and count against a patient who could not see to
    //    move. The gap is nudged forward so nothing bursts out at once
    //    the moment they come back into view.
    if (lost) {
      this.lastSpawnAt += dtMs;
      for (const it of this.items) it.bornAt += dtMs;
    } else if (this.strikePhase !== "idle") {
      // Same nudge during a strike event, for the same reason.
      this.lastSpawnAt += dtMs;
    } else if (
      this.items.length < MAX_ITEMS
      && time - this.lastSpawnAt > c.level.spawnGapMs
    ) {
      this.spawn(time);
      this.lastSpawnAt = time;
    }

    // Items already falling keep falling through a strike — freezing
    // them mid-air would look broken, and they are still catchable.
    if (!lost) this.stepItems(frame);

    // ── Diagnostics (?gamedebug=1). Everything generic is the base's;
    //    these are Cloudburst's own rows.
    this.writeCoreDebug(frame);
    const d = c.debug;
    d.extra["items spawned"] =
      `drops ${this.spawnedDrops} / bolts ${this.spawnedBolts}`;
    d.extra["lightning share"] =
      `${Math.round(c.level.lightningFraction * 100)}%`;
    d.extra["aimed bolts"] =
      `${this.aimedBolts} of ${this.spawnedBolts}`;
    d.extra["speed"] = `${this.speedMult.toFixed(2)}x`;
    d.extra["last reaction"] = this.lastReactionMs === null
      ? "none yet"
      : `${Math.round(this.lastReactionMs)} ms`;
    d.extra["lightning touched"] = String(this.res.lightningTouched);
    d.extra["next strike"] = this.strikePhase === "idle"
      ? `${Math.max(0, (this.strikeDueAt - time) / 1000).toFixed(1)}s`
      : `${this.strikePhase.toUpperCase()} `
        + `(${Math.max(0, (this.strikeDueAt - time) / 1000).toFixed(2)}s)`;
    d.extra["strike band"] = this.strikePhase === "idle"
      ? "—"
      : `x ${Math.round(this.bandLoX)}..${Math.round(this.bandHiX)}`;
    d.extra["strike aim"] = this.strikePhase === "idle"
      ? "—"
      : this.bandAimed ? "AIMED at palm" : "random";
    d.extra["strike shift"] = this.bandShiftNx === null
      ? "first of round"
      : `${(this.bandShiftNx / Math.max(1e-6, this.bandHalfNx * 2)).toFixed(2)}`
        + ` band-widths (${this.bandShiftNx.toFixed(3)} nx)`;
    d.extra["palm in band"] = this.strikePhase === "idle"
      ? "—"
      : this.palmInBand() ? "YES — will be hit" : "no";
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

  /** Where the palm sits in the lane space, or null when it cannot be
   *  read this frame. */
  private palmNx(): number | null {
    const c = this.control;
    const cover = c.state.cover;
    if (!c.state.usable || !this.cursorSeeded || cover.dispW <= 0) return null;
    const nx = (this.cursor.x - cover.offX) / cover.dispW;
    return Number.isFinite(nx) ? nx : null;
  }

  /**
   * Is this lane clear of any DROP still near the spawn height?
   *
   * Without this an aimed bolt can land exactly on top of a drop that
   * has only just appeared, and the two read as one object — the
   * patient cannot tell whether to reach or pull away, which is not a
   * test of anything.
   */
  private laneIsClear(nx: number, sepNx: number, sepNy: number): boolean {
    for (const it of this.items) {
      if (it.dying || it.kind !== "drop") continue;
      // Only drops still near the spawn height can be overlapped; one
      // that has fallen away is no longer in the way.
      if (Math.abs(it.ny - this.fallTop) > sepNy) continue;
      if (Math.abs(it.nx - nx) < sepNx) return false;
    }
    return true;
  }

  private spawn(time: number) {
    const c = this.control;
    const cover = c.state.cover;
    const size = this.unit * BASE_ITEM_FRACTION * c.level.itemScale * c.visualScale;
    // nx is a fraction of dispW and ny a fraction of dispH, so the
    // same pixel separation is a different number on each axis.
    const sepNx = cover.dispW > 0
      ? (size * MIN_LANE_SEPARATION) / cover.dispW
      : 0.08;
    const sepNy = cover.dispH > 0
      ? (size * MIN_LANE_SEPARATION) / cover.dispH
      : 0.08;

    // Decide WHAT before WHERE: an aimed bolt takes its lane from the
    // hand, and only an unaimed item consumes the same/across
    // alternation — so the drops keep alternating exactly even as
    // bolts are steered.
    const kind: ItemKind = Math.random() < c.level.lightningFraction ? "bolt" : "drop";
    const palmNx = this.palmNx();
    const aimed = kind === "bolt"
      && palmNx !== null
      && Math.random() < AIMED_LIGHTNING_FRACTION;

    let nx: number;
    let zone: CloudburstZone;
    if (aimed && palmNx !== null) {
      // The patient's own lane, clamped into the reach box — "or the
      // nearest lane" when the hand is outside the calibrated width.
      nx = Math.min(c.box.xHi, Math.max(c.box.xLo, palmNx));
      const sameIsHighX = c.box.hand === "right";
      zone = (nx >= c.box.midX) === sameIsHighX ? "same_side" : "across";
    } else {
      const region = this.nextRegion;
      zone = region === "same" ? "same_side" : "across";
      // Alternate the halves so the across-midline lane is never
      // crowded out by chance — the rule Fruit Harvest uses for zones.
      this.nextRegion = region === "same" ? "across" : "same";

      // Only the x is used. spawnPoint already splits the calibrated
      // box into a same-side and an across-the-midline half, with its
      // own fallback for a patient whose box never crossed the midline.
      const p = spawnPoint(c.box, region, Math.random);
      if (!Number.isFinite(p.nx)) {
        c.debug.extra["items spawned"] = "REJECTED: non-finite lane";
        return;
      }
      nx = p.nx;
    }

    // A bolt must not be laid on a drop. Try nudging aside first —
    // an aimed bolt stays aimed to within one item width — and skip
    // the spawn entirely rather than overlap. The caller has already
    // reset the gap, so it simply tries again next time.
    if (kind === "bolt" && !this.laneIsClear(nx, sepNx, sepNy)) {
      const nudged = [nx + sepNx * 1.2, nx - sepNx * 1.2]
        .map((v) => Math.min(c.box.xHi, Math.max(c.box.xLo, v)))
        .find((v) => this.laneIsClear(v, sepNx, sepNy));
      if (nudged === undefined) return;
      nx = nudged;
    }

    if (kind === "drop") this.spawnedDrops += 1;
    else {
      this.spawnedBolts += 1;
      if (aimed) this.aimedBolts += 1;
    }

    const ny = this.fallTop;
    const x = cover.dispW > 0 ? cover.offX + nx * cover.dispW : this.scale.width / 2;
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
      nx,
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
          else this.zap(it, i, frame.time);
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

  private zap(it: Item, index: number, time: number) {
    it.dying = true;
    this.items.splice(index, 1);
    const c = this.control;
    this.res.lightningTouched += 1;
    c.lightningTouched = this.res.lightningTouched;
    this.errorText?.setText(`⚡ ${this.res.lightningTouched}`);
    c.audio.error();

    // Brief red tint. Short, partial and never white: this is feedback,
    // not a penalty screen, and the round keeps running.
    //
    // RATE-LIMITED. A patient flailing through a cluster of bolts could
    // otherwise retrigger it several times a second. The error is still
    // counted and the buzz still plays; only the light is capped, at
    // 2.5 per second, well under the 3 Hz photosensitivity ceiling.
    // `time` is update()'s own clock, the only one this scene uses.
    if (this.flash && time - this.lastFlashAt >= SAFE_FLASH_GAP_MS) {
      this.lastFlashAt = time;
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

  // ── Big centre strike ───────────────────────────────────────────

  /**
   * Pick where this strike lands. Once per strike, never on a resize.
   *
   * The band is no longer pinned to the middle of the reach: it can be
   * anywhere the rules in lib/games/cloudburstBand.ts allow — inside
   * reach, with a full band-width of escape space on at least one side,
   * and at least a band-width from where the last one fell. About half
   * are placed on the patient's hand, so standing still is not a
   * strategy.
   */
  private chooseBand(): void {
    const box = this.control.box;
    const choice = chooseStrikeBand({
      loNx: box.xLo,
      hiNx: box.xHi,
      fraction: STRIKE_BAND_FRACTION,
      palmNx: this.palmNx(),
      prevCentreNx: this.prevBandCentreNx,
      aimedChance: STRIKE_AIMED_FRACTION,
      rand: Math.random,
    });
    this.bandCentreNx = choice.centreNx;
    this.bandHalfNx = choice.halfNx;
    this.bandAimed = choice.aimed;
    this.bandShiftNx = choice.shiftNx;
    this.prevBandCentreNx = choice.centreNx;
    this.sizeStrikeBand();
  }

  /** Project the chosen band onto the CURRENT canvas. Called when a
   *  strike starts, and again on a resize so the danger zone keeps
   *  matching the patient's reach. */
  private sizeStrikeBand(): void {
    const c = this.control;
    const cover = c.state.cover;
    const midNx = this.bandCentreNx;
    const halfNx = this.bandHalfNx;

    if (cover.dispW > 0) {
      this.bandLoX = cover.offX + (midNx - halfNx) * cover.dispW;
      this.bandHiX = cover.offX + (midNx + halfNx) * cover.dispW;
    } else {
      // No pose geometry yet: fall back to the middle of the canvas so
      // the event is still coherent rather than zero-width.
      const half = (this.scale.width * STRIKE_BAND_FRACTION) / 2;
      this.bandLoX = this.scale.width / 2 - half;
      this.bandHiX = this.scale.width / 2 + half;
    }

    // Top of the play area down to the miss line — the band covers
    // exactly the height the patient's hand works in.
    const top = cover.dispH > 0
      ? cover.offY + this.fallTop * cover.dispH
      : 0;
    const bottom = Math.min(
      cover.dispH > 0 ? cover.offY + this.fallBottom * cover.dispH : this.scale.height,
      this.scale.height * SKY_GROUND,
    );
    // The band's WIDTH is bandLoX..bandHiX; only its vertical extent
    // has to be derived, and it is what the storm is drawn into.
    this.bandTopY = Math.max(0, top);
    this.bandBottomY = this.bandTopY + Math.max(2, bottom - this.bandTopY);
  }

  /** Is the drawn palm cursor inside the struck band right now? An
   *  unreadable hand counts as inside: the patient has not been seen to
   *  move out, and crediting a dodge nobody saw would be generous in
   *  the wrong direction. */
  private palmInBand(): boolean {
    if (!this.control.state.usable || !this.cursorSeeded) return true;
    return this.cursor.x >= this.bandLoX && this.cursor.x <= this.bandHiX;
  }

  /** The whole event, driven from update(). */
  private stepStrike(time: number): void {
    if (this.strikeDueAt < 0) return;

    // Thunder trails the flash, and outlives the 150 ms strike phase.
    if (this.thunderAt !== null && time >= this.thunderAt) {
      this.thunderAt = null;
      this.control.audio.rumble();
    }

    if (this.strikePhase === "idle") {
      // ?gamedebug=1 only: S, or the on-canvas button, brings the next
      // strike forward so it can be reviewed without waiting.
      if (this.strikeRequested) {
        this.strikeRequested = false;
        this.beginWarning(time);
        return;
      }
      if (time < this.strikeDueAt) return;
      this.beginWarning(time);
      return;
    }

    if (this.strikePhase === "warning") {
      // The band does NOT chase the hand during the warning: it is
      // chosen once in chooseBand() and only re-projected on a resize.
      // A band that followed the palm could never be dodged.
      if (this.strikeWasInside && this.strikeLeftAt === null && !this.palmInBand()) {
        this.strikeLeftAt = time;
      }
      if (time >= this.strikeDueAt) this.fireStrike(time);
      return;
    }

    // striking
    if (time >= this.strikeDueAt) {
      this.strikePhase = "idle";
      this.strikeDueAt = time + this.nextStrikeGap();
      this.strikeWarnedAt = -1;
      // Clouds drift apart over ~1.5 s rather than blinking out.
      this.storm?.dissipate();
    }
  }

  private beginWarning(time: number): void {
    this.strikePhase = "warning";
    this.strikeWarnedAt = time;
    this.strikeDueAt = time + STRIKE_WARN_MS;
    this.strikeLeftAt = null;
    // Choose BEFORE reading the palm: the storm, the ground glow and
    // the heavier rain are all built from this band, so they follow
    // wherever it lands.
    this.chooseBand();
    this.strikeWasInside = this.palmInBand();

    // The storm IS the warning: clouds rolling in and piling over the
    // band, the sky darkening under them as a soft gradient, heavier
    // rain inside the band and a glow on the ground where the bolt will
    // land. No filled rectangle is drawn anywhere.
    this.storm?.beginWarning(
      {
        loX: this.bandLoX,
        hiX: this.bandHiX,
        top: this.bandTopY,
        bottom: this.bandBottomY,
      },
      STRIKE_WARN_MS,
    );
    this.strikeText?.setVisible(true).setAlpha(1);
    this.control.audio.rumble();
  }

  private fireStrike(time: number): void {
    const c = this.control;
    this.strikePhase = "striking";
    this.strikeDueAt = time + STRIKE_FLASH_MS;

    // Judgement happens HERE, on the frame the bolt lands — not over
    // the warning. A patient who steps out and back in is hit, which
    // is the honest answer.
    const hit = this.palmInBand();
    this.res.bigStrikes += 1;
    if (hit) {
      this.res.bigStrikesHit += 1;
      c.audio.error();
      // A shake, not a flash: it carries the impact without adding
      // another light event.
      this.cameras.main.shake(180, 0.006);
      this.showCue("Hit!", "#fca5a5");
    } else {
      this.res.bigStrikesDodged += 1;
      if (this.strikeWasInside && this.strikeLeftAt !== null) {
        this.res.dodgeSecs.push(
          Math.round(this.strikeLeftAt - this.strikeWarnedAt) / 1000,
        );
      }
      c.audio.harvest();
      this.showCue("Dodged!", "#86efac");
    }

    // Warning text down, bolt up. The bolt, the ground impact, the
    // sparks and the clouds lighting from inside all land on this
    // frame and fade together: ONE bright event, no strobe.
    this.strikeText?.setVisible(false);
    this.storm?.fire();

    // Thunder arrives after the light, the way it does outdoors. Kept
    // on update()'s clock rather than a scene timer so it pauses with
    // the round if the hand is lost.
    this.thunderAt = time + THUNDER_DELAY_MS;
  }

  /** Short centred word after a strike resolves. */
  private showCue(text: string, colour: string): void {
    const cue = this.strikeCue;
    if (!cue) return;
    this.tweens.killTweensOf(cue);
    cue.setText(text).setColor(colour).setAlpha(1);
    this.tweens.add({
      targets: cue,
      alpha: 0,
      delay: 500,
      duration: 400,
      ease: "Quad.easeIn",
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
