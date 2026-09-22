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
import type { HandState, PalmSource } from "@/lib/games/handTracker";
import type { ReachBox } from "@/lib/games/calibration";
import {
  reachGeometry,
  spawnAtReachEdge,
  spawnPoint,
  type Zone,
} from "@/lib/games/calibration";
import { PALM_REACH } from "@/lib/games/handTracker";
import { OneEuro2D } from "@/lib/games/oneEuro";
import type { GameAudio } from "@/lib/games/gameAudio";
import {
  collectBurst,
  floatScore,
  glyphAverageColour,
  makeRingTexture,
  makeSoftDotTexture,
} from "@/lib/games/fruitEffects";
import { makeOrchardTexture, makeSprigTexture } from "@/lib/games/orchardScene";

export const ROUND_MS = 60_000;
const MAX_FRUIT = 3;
/** How long a fruit waits to be picked before it falls away. */
const FRUIT_TTL_MS = 5200;
/** Gap between spawns so three do not appear on the same frame. */
const SPAWN_GAP_MS = 600;

const FRUITS = ["🍎", "🍊", "🍐", "🍋", "🍓", "🍇", "🍑", "🥝"];
const BASKET = "🧺";
const LEAF = "🍃";

// ── Animation timings. None of these change the round clock, the TTL,
//    the hit test or the score — they only shape what is drawn.
/** Squash-and-stretch on contact, before the fruit sets off. */
const SNAP_MS = 140;
/**
 * Wobble warning before a fruit falls away.
 *
 * It starts at TTL − this, NOT after the TTL, so the moment a fruit is
 * counted missed is exactly where it was before. The fruit is still
 * collectable while it wobbles — that is the point of the warning.
 */
const MISS_WARN_MS = 300;

// ── Cursor responsiveness.
//
// Measured against a simulated 1500 px/s sweep at pose 22 Hz / render
// 60 fps. The headline finding: the filter is NOT the main source of
// trailing. Half a pose interval alone is ~34 px of lag at that speed,
// and opening the filter right up (beta 0.02 -> 0.25) only moved the
// total from 45 px to 37 px. Short prediction is what actually helps —
// it halves the lag to ~19 px.
/** Cutoff at zero speed. As low as it can go without visible jitter. */
const CURSOR_MIN_CUTOFF = 0.9;
/** Opens the cutoff with speed. 6x the old value; costs ~0.2 px of
 *  jitter at rest and removes the filter as the limiting factor. */
const CURSOR_BETA = 0.12;
/** Extrapolate forward by this fraction of a pose interval. */
const PREDICT_HORIZON = 0.5;
/** Hard cap on the extrapolation, as a fraction of the canvas unit.
 *  Sized so a hard stop overshoots ~25 px — under half the hit radius
 *  (0.075 unit). Without a cap a fast stop overshoots much further. */
const PREDICT_CAP = 0.035;
/** Below this speed (canvas units/second) prediction is switched off
 *  entirely, so a still hand cannot drift. */
const PREDICT_DEADBAND = 0.25;
/** Assumed pose rate before a real measurement arrives. */
const FALLBACK_POSE_HZ = 20;

/** How long the hand may be missing before the round pauses. The
 *  patient can no longer see themselves, so they need telling. */
const HAND_LOST_MS = 1000;
/** Below this headroom ratio during PLAY the patient is warned, but the
 *  round is not held — they are mid-game and a hard stop would be worse
 *  than a slightly clipped overhead reach. Calibration uses the
 *  stricter HEADROOM_RATIO_MIN and does pause. */
const PLAY_HEADROOM_MIN = 1.0;
/** Ground line, as a fraction of height — matches the orchard's. */
const GROUND_Y = 0.9;
/** Fall time into the basket when a fruit is collected. */
const FALL_MS = 520;

/** Glyph texture size. Generous so a large sprite stays crisp. */
const GLYPH_TEX = 256;

/** Diagnostics surfaced by the ?gamedebug=1 overlay. Written by the
 *  scene, polled by React. Keep it cheap — it is updated every frame. */
export interface GameDebug {
  phaserCreated: boolean;
  sceneState: string;
  canvasW: number;
  canvasH: number;
  canvasZ: string;
  /** Reach box projected into canvas pixels, as the scene sees it. */
  boxPx: { x0: number; x1: number; y0: number; y1: number } | null;
  boxN: { x0: number; x1: number; y0: number; y1: number } | null;
  spawnedTotal: number;
  onScreen: number;
  lastSpawn: string;
  handLive: boolean;
  handInFrame: boolean;
  cursorX: number;
  cursorY: number;
  elapsedMs: number;
  remainingMs: number;
  texturesOk: boolean;
  error: string | null;
  /** Smoothed framerate from Phaser's own loop. */
  fps: number;
  /** Worst framerate seen since the round began — where a costly
   *  effect would show up. Ignores the first second of warm-up. */
  fpsMin: number;
  /** Live tween count; the animation cost in one number. */
  tweens: number;
  /** Display-list size, so a leak would be visible as steady growth. */
  objects: number;
  /** Detector callbacks per second — the freshness ceiling. */
  poseHz: number;
  /** One-euro cutoff actually applied this frame, in Hz. */
  cutoffHz: number;
  /** Distance from the raw mapped palm to the drawn cursor, px. */
  lagPx: number;
  /** Where the palm came from this frame. */
  palmSource: PalmSource;
  /** Frames per source SINCE THE ROUND STARTED, so setup and
   *  calibration do not pollute the tally. */
  palmCounts: { hand: number; elbow: number; wrist: number };
  /** Spawn anchor — the chosen side's shoulder, canvas px. */
  shoulderPx: { x: number; y: number } | null;
  /** Reach radius in each calibrated direction, canvas px. */
  reachR: { across: number; side: number; up: number } | null;
  /** Last spawn's direction, degrees from straight-out on the playing
   *  side: positive = toward overhead, negative = across the body. */
  lastSpawnDeg: number;
  /** Last spawn's distance as a percentage of the reach that way. */
  lastSpawnPct: number;
  /** Shoulder -> elbow -> wrist, canvas px. */
  armLenPx: number;
  /** Space above the chosen shoulder on the VISIBLE canvas, px. */
  headroomPx: number;
  /** headroomPx / armLenPx. Below ~1.1 a raised arm cannot be seen. */
  headroomRatio: number;
  /** Round is held because the hand has not been seen for a second. */
  handLost: boolean;
}

export function createGameDebug(): GameDebug {
  return {
    phaserCreated: false,
    sceneState: "not started",
    canvasW: 0,
    canvasH: 0,
    canvasZ: "—",
    boxPx: null,
    boxN: null,
    spawnedTotal: 0,
    onScreen: 0,
    lastSpawn: "none yet",
    handLive: false,
    handInFrame: false,
    cursorX: 0,
    cursorY: 0,
    elapsedMs: 0,
    remainingMs: ROUND_MS,
    texturesOk: false,
    error: null,
    fps: 0,
    fpsMin: 0,
    tweens: 0,
    objects: 0,
    poseHz: 0,
    cutoffHz: 0,
    lagPx: 0,
    palmSource: "wrist",
    palmCounts: { hand: 0, elbow: 0, wrist: 0 },
    shoulderPx: null,
    reachR: null,
    lastSpawnDeg: 0,
    lastSpawnPct: 0,
    armLenPx: 0,
    headroomPx: 0,
    headroomRatio: 0,
    handLost: false,
  };
}

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
  debug: GameDebug;
  onFinish: (r: { harvested: number; missed: number }) => void;
}

interface Fruit {
  img: Phaser.GameObjects.Image;
  nx: number;
  ny: number;
  bornAt: number;
  dying: boolean;
  /** Base scale from setDisplaySize, so effects can return to it. */
  baseScale: number;
  /** The pre-fall wobble has been started. */
  warned: boolean;
  /** Stem and leaf drawn behind the fruit, so it reads as attached. */
  sprig: Phaser.GameObjects.Image | null;
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
  private filter = new OneEuro2D({
    minCutoff: CURSOR_MIN_CUTOFF,
    beta: CURSOR_BETA,
  });
  /**
   * -1 until the first update frame seeds it.
   *
   * It MUST come from update()'s own `time` argument and nothing else.
   * Phaser has two unrelated clocks: `time` here is the raw rAF
   * timestamp (ms since the PAGE loaded), while `this.time.now` is
   * `game.loop.time`, which starts at 0 when THIS game boots. Seeding
   * this from `this.time.now` and comparing it against `time` measured
   * the age of the page, so the round ended on frame one with nothing
   * spawned. Seed from the same clock you compare against.
   */
  private startedAt = -1;
  private lastSpawnAt = -1;
  private lastFrameAt = -1;
  private nextRegion: "same" | "across" = "same";
  private cursorSeeded = false;
  private fruitColour: Record<string, number> = {};
  private leafKey: string | null = null;
  private fpsText: Phaser.GameObjects.Text | null = null;
  private fpsMin = Infinity;
  /** Where the previous fruit went, for the separation rule. */
  private lastSpawnPoint: { x: number; y: number } | null = null;
  /** Palm-source tallies as they stood when the round began. */
  private palmBase = { hand: 0, elbow: 0, wrist: 0 };
  private backdrop: Phaser.GameObjects.Image | null = null;
  private backdropSize = { w: 0, h: 0 };
  private lostText: Phaser.GameObjects.Text | null = null;
  private closeText: Phaser.GameObjects.Text | null = null;
  private lostBand: Phaser.GameObjects.Rectangle | null = null;
  /** Milliseconds the hand has been missing, and the total time the
   *  round clock has been held for. */
  private handLostMs = 0;
  private pausedMs = 0;

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
      // Sampled once, here — never in the frame loop.
      this.fruitColour[key] = glyphAverageColour(this, key);
      return key;
    });

    // Effect art, also generated rather than loaded.
    makeSoftDotTexture(this, "fx-dot");
    makeRingTexture(this, "fx-ring");
    makeSprigTexture(this, "fx-sprig");
    this.leafKey = makeGlyphTexture(this, "fx-leaf", LEAF) ? "fx-leaf" : null;

    // ── Orchard backdrop. Generated once here, then one sprite per
    //    frame. The camera is hidden during play, so this is what the
    //    patient actually sees behind the fruit.
    const bw = Math.round(this.scale.width);
    const bh = Math.round(this.scale.height);
    if (makeOrchardTexture(this, "orchard", bw, bh, Math.floor(Math.random() * 1e6))) {
      this.backdrop = this.add
        .image(0, 0, "orchard")
        .setOrigin(0, 0)
        .setDisplaySize(bw, bh)
        .setDepth(-10);
      this.backdropSize = { w: bw, h: bh };
    } else {
      // Texture creation failed — a flat wash still hides the camera.
      this.add
        .rectangle(0, 0, bw, bh, 0x24512f)
        .setOrigin(0, 0)
        .setDepth(-10);
    }
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

    // NOTE: startedAt / lastFrameAt / lastSpawnAt are deliberately left
    // at -1 here and seeded on the first update frame instead — see the
    // field declaration for why this must not use this.time.now.
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

    // Non-blocking "you have moved closer" warning.
    this.closeText = this.add
      .text(0, 0, "Step back — top of your reach is off screen", {
        fontFamily: "system-ui, sans-serif",
        fontSize: `${Math.round(this.unit * 0.035 * s)}px`,
        color: "#fbbf24",
        backgroundColor: "#00000099",
        padding: { x: 10, y: 5 },
      })
      .setOrigin(0.5, 0)
      .setDepth(32)
      .setVisible(false);

    this.layout();
    this.palmBase = { ...this.control.state.palmCounts };

    const d = this.control.debug;
    d.sceneState = "created";
    d.texturesOk = this.textures.exists("basket") && this.textures.exists(this.keys[0]);
    d.boxN = {
      x0: this.control.box.xLo,
      x1: this.control.box.xHi,
      y0: this.control.box.yLo,
      y1: this.control.box.yHi,
    };
  }

  /**
   * Position and size everything that is anchored to the canvas.
   *
   * Called once at the end of create() and again whenever the canvas
   * changes size — which is every fullscreen enter and exit. Fruit are
   * NOT handled here: they carry normalised coordinates and are
   * re-mapped from the live object-cover geometry every frame, so they
   * stay reachable across a resize on their own.
   */
  private layout(): void {
    const w = this.scale.width;
    const h = this.scale.height;
    const u = Math.min(w, h);
    const s = this.control.visualScale;
    this.backdropSize = { w, h };

    this.backdrop?.setDisplaySize(w, h);

    this.basket
      ?.setPosition(w / 2, h * 0.93)
      .setDisplaySize(u * 0.17 * s, u * 0.17 * s);
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

    this.lostBand?.setPosition(w / 2, h / 2).setSize(w, u * 0.3);
    const lost = Math.round(u * 0.062 * s);
    this.lostText
      ?.setPosition(w / 2, h / 2)
      .setFontSize(lost)
      .setStroke("#000000", Math.max(2, lost * 0.08));

    this.fpsText?.setPosition(w * 0.02, h * 0.82).setFontSize(Math.round(u * 0.032));
    this.closeText
      ?.setPosition(w / 2, h * 0.14)
      .setFontSize(Math.round(u * 0.035 * s));
  }

  update(time: number) {
    const c = this.control;
    if (c.finished) return;

    // First frame: anchor every clock to update()'s own time base.
    if (this.startedAt < 0) {
      this.startedAt = time;
      this.lastFrameAt = time;
      // One full gap behind, so the first fruit appears immediately.
      this.lastSpawnAt = time - SPAWN_GAP_MS - 1;
      c.debug.sceneState = "running";
    }

    const dtMs = Math.max(0, Math.min(200, time - this.lastFrameAt));
    const dt = Math.max(0.001, (time - this.lastFrameAt) / 1000);
    this.lastFrameAt = time;

    // ── Hand lost. With the camera hidden the patient cannot tell why
    //    nothing is happening, so say so, and hold the round rather
    //    than burning their time.
    if (c.state.usable) {
      this.handLostMs = 0;
    } else {
      this.handLostMs += dtMs;
    }
    const lost = this.handLostMs > HAND_LOST_MS;
    if (lost) this.pausedMs += dtMs;
    this.lostBand?.setVisible(lost);
    this.lostText?.setVisible(lost);
    c.debug.handLost = lost;

    // Headroom during play: warn, never pause. Hidden while the
    // hand-lost banner is up so the two cannot stack.
    this.closeText?.setVisible(
      !lost
      && c.state.armLenPx > 1
      && c.state.headroomRatio < PLAY_HEADROOM_MIN,
    );

    // Entering or leaving fullscreen changes the canvas under us.
    // Everything positioned in create() has to be laid out again, or
    // the HUD and basket stay where they were on the old canvas.
    if (
      this.backdropSize.w !== this.scale.width
      || this.backdropSize.h !== this.scale.height
    ) {
      this.layout();
    }

    // ── Clock. Paused time is subtracted, so a round always gives the
    //    patient the full ROUND_MS of play.
    const elapsed = time - this.startedAt - this.pausedMs;
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
    //    The target is the PALM, not the wrist — the ✋ glyph reads as a
    //    palm and the hit test below uses the same point, so what the
    //    patient sees is what collects the fruit.
    if (c.state.usable) {
      const rawX = c.state.palmX;
      const rawY = c.state.palmY;
      if (!this.cursorSeeded) {
        // Snap on the first good frame instead of sliding in from the
        // middle of the screen.
        this.filter.reset();
        this.cursor.setPosition(rawX, rawY);
        this.cursor.setAlpha(1);
        this.cursorSeeded = true;
        this.filter.filter(rawX, rawY, dt);
      } else {
        const p = this.filter.filter(rawX, rawY, dt);

        // Light prediction. Uses the filter's OWN low-passed velocity
        // rather than differencing the output again, which would put
        // back the noise the filter just removed.
        const hz = c.state.poseHz > 1 ? c.state.poseHz : FALLBACK_POSE_HZ;
        const ahead = (PREDICT_HORIZON / hz);
        const { vx, vy } = this.filter.velocity;
        const speed = Math.hypot(vx, vy);
        let ex = 0;
        let ey = 0;
        if (speed > PREDICT_DEADBAND * this.unit) {
          const cap = PREDICT_CAP * this.unit;
          const dx = vx * ahead;
          const dy = vy * ahead;
          const mag = Math.hypot(dx, dy);
          // Scale the vector as a whole so capping cannot bend its
          // direction the way clamping each axis would.
          const k = mag > cap ? cap / mag : 1;
          ex = dx * k;
          ey = dy * k;
        }
        this.cursor.setPosition(p.x + ex, p.y + ey);
        this.cursor.setAlpha(1);
      }
      // Lag: raw mapped palm vs where the cursor is actually drawn.
      c.debug.lagPx = Math.round(
        Math.hypot(this.cursor.x - rawX, this.cursor.y - rawY),
      );
    } else {
      this.cursor.setAlpha(0.25);
    }

    // ── Spawn. Held while the hand is lost: new fruit would only time
    //    out unseen and count as misses the patient never had a chance
    //    at. The gap is nudged forward so nothing bursts out at once
    //    the moment they come back into view.
    if (lost) {
      this.lastSpawnAt += dtMs;
      // Fruit already on the branch must not time out either — they
      // would be counted missed for a hand the patient could not see
      // to move. Shifting bornAt holds their TTL exactly as the round
      // clock is held.
      for (const f of this.fruits) f.bornAt += dtMs;
    } else if (
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
        f.sprig?.setPosition(f.img.x, f.img.y);
      }

      // Hit test reads the DRAWN cursor, so it is the palm point after
      // smoothing and prediction — whatever the patient sees on screen
      // is exactly what collects the fruit.
      if (c.state.usable && this.cursorSeeded) {
        const d = Math.hypot(this.cursor.x - f.img.x, this.cursor.y - f.img.y);
        if (d < hitR) {
          this.harvest(f, i);
          continue;
        }
      }

      const age = time - f.bornAt;
      if (!f.warned && age > FRUIT_TTL_MS - MISS_WARN_MS) {
        f.warned = true;
        this.startWobble(f);
      }
      if (age > FRUIT_TTL_MS) this.dropAway(f, i);
    }

    // ── Diagnostics (?gamedebug=1)
    const d = c.debug;
    d.elapsedMs = elapsed;
    d.remainingMs = c.remainingMs;
    d.onScreen = this.fruits.length;
    d.handLive = c.state.live;
    d.handInFrame = c.state.inFrame;
    d.cursorX = Math.round(this.cursor.x);
    d.cursorY = Math.round(this.cursor.y);
    d.canvasW = Math.round(this.scale.width);
    d.canvasH = Math.round(this.scale.height);

    const fps = this.game.loop.actualFps;
    d.fps = Math.round(fps);
    // Skip the first second: the very first frames are always slow
    // (shader compile, texture upload) and would mask a real dip.
    if (elapsed > 1000 && fps > 0) this.fpsMin = Math.min(this.fpsMin, fps);
    d.fpsMin = Number.isFinite(this.fpsMin) ? Math.round(this.fpsMin) : 0;
    d.tweens = this.tweens.getTweens().length;
    d.objects = this.children.list.length;
    d.poseHz = Math.round(c.state.poseHz * 10) / 10;
    d.cutoffHz = Math.round(this.filter.lastCutoff * 100) / 100;
    d.palmSource = c.state.palmSource;
    // Report deltas against the baseline taken in create(), so the
    // tally covers this round only.
    d.palmCounts = {
      hand: c.state.palmCounts.hand - this.palmBase.hand,
      elbow: c.state.palmCounts.elbow - this.palmBase.elbow,
      wrist: c.state.palmCounts.wrist - this.palmBase.wrist,
    };
    d.armLenPx = Math.round(c.state.armLenPx);
    d.headroomPx = Math.round(c.state.headroomPx);
    d.headroomRatio = Math.round(c.state.headroomRatio * 100) / 100;
    if (this.fpsText) {
      const sh = d.shoulderPx ? `${d.shoulderPx.x},${d.shoulderPx.y}` : "—";
      const rr = d.reachR
        ? `across ${d.reachR.across} side ${d.reachR.side} up ${d.reachR.up}`
        : "—";
      this.fpsText.setText(
        `fps ${d.fps} min ${d.fpsMin}  pose ${d.poseHz}Hz  `
        + `cutoff ${d.cutoffHz}Hz  lag ${d.lagPx}px  `
        + `palm ${d.palmSource} `
        + `(h${d.palmCounts.hand}/e${d.palmCounts.elbow}/w${d.palmCounts.wrist})\n`
        + `shoulder ${sh}  reach px: ${rr}\n`
        + `arm ${d.armLenPx}px  headroom ${d.headroomPx}px  `
        + `ratio ${d.headroomRatio}\n`
        + `last spawn ${d.lastSpawnDeg >= 0 ? "+" : ""}${d.lastSpawnDeg}deg `
        + `@ ${d.lastSpawnPct}% of reach`,
      );
    }
    if (cover.dispW > 0) {
      d.boxPx = {
        x0: Math.round(cover.offX + c.box.xLo * cover.dispW),
        x1: Math.round(cover.offX + c.box.xHi * cover.dispW),
        y0: Math.round(cover.offY + c.box.yLo * cover.dispH),
        y1: Math.round(cover.offY + c.box.yHi * cover.dispH),
      };
    }
  }

  private spawn(time: number) {
    const c = this.control;
    const region = this.nextRegion;
    const zone: Zone = region === "same" ? "abduction" : "adduction";
    const st = c.state;
    const cover = st.cover;

    // Polar placement around the LIVE shoulder. The palm offset is
    // added to every radius inside reachGeometry, because calibration
    // recorded the wrist while the cursor is the palm.
    const geo = st.shoulderOk
      ? reachGeometry(
        c.box,
        st.shoulderX,
        st.shoulderY,
        cover,
        PALM_REACH * st.forearmPx,
      )
      : null;

    let x: number;
    let y: number;
    let note: string;

    if (geo) {
      const r = spawnAtReachEdge(geo, zone, {
        canvasW: this.scale.width,
        canvasH: this.scale.height,
        margin: this.unit * 0.08 * c.visualScale,
        cursor: this.cursorSeeded
          ? { x: this.cursor.x, y: this.cursor.y }
          : null,
        prev: this.lastSpawnPoint,
        rand: Math.random,
      });
      x = r.x;
      y = r.y;
      c.debug.lastSpawnDeg = r.angleDeg;
      c.debug.lastSpawnPct = r.reachPct;
      c.debug.reachR = {
        across: Math.round(geo.rAcross),
        side: Math.round(geo.rSide),
        up: Math.round(geo.rUp),
      };
      c.debug.shoulderPx = { x: Math.round(geo.sx), y: Math.round(geo.sy) };
      note =
        `${zone} ${r.angleDeg}deg ${r.reachPct}% of reach`
        + `${r.clamped ? " (pulled in)" : ""}`
        + `${r.tries > 0 ? ` after ${r.tries} rejected` : ""}`;
    } else {
      // No usable shoulder this frame. Fall back to the old uniform box
      // rather than stalling the round, and say so on the overlay.
      const p = spawnPoint(c.box, region, Math.random);
      if (!Number.isFinite(p.nx) || !Number.isFinite(p.ny)) {
        c.debug.lastSpawn = "REJECTED: non-finite point from reach box";
        return;
      }
      x = cover.dispW > 0
        ? cover.offX + p.nx * cover.dispW
        : this.scale.width / 2;
      y = cover.dispH > 0
        ? cover.offY + p.ny * cover.dispH
        : this.scale.height / 2;
      c.debug.lastSpawnDeg = 0;
      c.debug.lastSpawnPct = 0;
      note = `${zone} FALLBACK uniform box (no live shoulder)`;
    }

    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      c.debug.lastSpawn = "REJECTED: non-finite spawn point";
      return;
    }
    this.lastSpawnPoint = { x, y };
    // Store normalised so a resize keeps the fruit where the patient
    // reached for it — same contract the update loop expects.
    const p = {
      nx: cover.dispW > 0 ? (x - cover.offX) / cover.dispW : 0.5,
      ny: cover.dispH > 0 ? (y - cover.offY) / cover.dispH : 0.5,
    };
    // Alternate the halves so the across-midline reach — the part that
    // exercises adduction — is never crowded out by chance.
    this.nextRegion = this.nextRegion === "same" ? "across" : "same";

    const key = this.keys[Math.floor(Math.random() * this.keys.length)];
    const size = this.unit * 0.13 * c.visualScale;
    // Stem and leaf, behind the fruit and slightly above it.
    let sprig: Phaser.GameObjects.Image | null = null;
    if (this.textures.exists("fx-sprig")) {
      sprig = this.add
        .image(x, y, "fx-sprig")
        .setDisplaySize(size * 1.15, size * 1.15)
        .setDepth(9);
    }

    const img = this.add.image(x, y, key).setDisplaySize(size, size).setDepth(10);
    const base = img.scaleX;
    const sprigBase = sprig ? (size * 1.15) / this.textures.get("fx-sprig").getSourceImage().width : 0;

    // GROWS on the branch rather than popping out of nowhere: a slow
    // swell with only a whisper of overshoot, so it reads as ripening
    // rather than being thrown at the screen.
    img.setScale(0);
    if (sprig) sprig.setScale(0);
    this.tweens.chain({
      targets: img,
      tweens: [
        {
          scaleX: base * 1.06,
          scaleY: base * 1.06,
          duration: 340,
          ease: "Sine.easeOut",
        },
        {
          scaleX: base,
          scaleY: base,
          duration: 130,
          ease: "Sine.easeInOut",
        },
      ],
    });
    if (sprig) {
      this.tweens.add({
        targets: sprig,
        scaleX: sprigBase,
        scaleY: sprigBase,
        duration: 300,
        ease: "Sine.easeOut",
      });
    }

    this.fruits.push({
      img,
      nx: p.nx,
      ny: p.ny,
      bornAt: time,
      dying: false,
      baseScale: base,
      warned: false,
      sprig,
    });
    c.debug.spawnedTotal += 1;
    c.debug.lastSpawn = `${note} px(${Math.round(x)}, ${Math.round(y)})`;
  }

  private harvest(f: Fruit, index: number) {
    f.dying = true;
    this.fruits.splice(index, 1);
    this.control.harvested += 1;
    this.scoreText.setText(String(this.control.harvested));
    this.control.audio.harvest();

    // The wobble warning may be mid-flight; drop it so it cannot fight
    // the collect animation.
    this.tweens.killTweensOf(f.img);

    const u = this.unit;
    const s = this.control.visualScale;
    const cx = f.img.x;
    const cy = f.img.y;
    const colour = this.fruitColour[f.img.texture.key] ?? 0xffd166;

    // 1 — snap. Squash-and-stretch on contact, then straight back.
    f.img.setScale(f.baseScale);
    this.tweens.add({
      targets: f.img,
      scaleX: f.baseScale * 1.3,
      scaleY: f.baseScale * 1.18,
      duration: SNAP_MS / 2,
      yoyo: true,
      ease: "Quad.easeOut",
    });

    // 2 — flash ring, particle burst, leaves. Tinted to this fruit.
    collectBurst(this, {
      x: cx,
      y: cy,
      unit: u,
      colour,
      visualScale: s,
      leafKey: this.leafKey,
      dotKey: "fx-dot",
      ringKey: "fx-ring",
    });

    // 5 — floating "+1" from the contact point.
    floatScore(this, cx, cy, u, s);

    // The stem stays on the branch — only the fruit detaches.
    if (f.sprig) {
      const sp = f.sprig;
      this.tweens.add({
        targets: sp,
        alpha: 0,
        duration: 420,
        delay: 180,
        onComplete: () => sp.destroy(),
      });
      f.sprig = null;
    }

    // 3 — DETACH AND FALL. Gravity down into the basket rather than a
    //     lofted arc: the fruit was hanging on a branch, so it should
    //     drop off it. Quad.easeIn on y is constant acceleration; x
    //     drifts linearly toward the basket so it still lands in it.
    this.tweens.add({
      targets: f.img,
      y: this.basket.y - u * 0.02,
      duration: FALL_MS,
      delay: SNAP_MS,
      ease: "Quad.easeIn",
      onComplete: () => {
        f.img.destroy();
        this.bounceBasket();
      },
    });
    this.tweens.add({
      targets: f.img,
      x: this.basket.x,
      duration: FALL_MS,
      delay: SNAP_MS,
      ease: "Sine.easeInOut",
    });
    // A slow tumble and a little shrink as it drops away from the eye.
    this.tweens.add({
      targets: f.img,
      angle: 110,
      scaleX: f.baseScale * 0.6,
      scaleY: f.baseScale * 0.6,
      duration: FALL_MS,
      delay: SNAP_MS,
      ease: "Sine.easeIn",
    });
  }

  /** 4 — the basket takes the weight. */
  private bounceBasket() {
    const sx = this.basket.scaleX;
    const sy = this.basket.scaleY;
    this.tweens.killTweensOf(this.basket);
    this.basket.setScale(sx, sy);
    this.tweens.chain({
      targets: this.basket,
      tweens: [
        { scaleX: sx * 1.18, scaleY: sy * 0.82, duration: 90, ease: "Quad.easeOut" },
        { scaleX: sx * 0.94, scaleY: sy * 1.08, duration: 90, ease: "Quad.easeInOut" },
        { scaleX: sx, scaleY: sy, duration: 110, ease: "Back.easeOut" },
      ],
    });
  }

  /** 6 — the warning wobble, started MISS_WARN_MS before the TTL. The
   *  fruit is still collectable throughout; only the look changes. */
  private startWobble(f: Fruit) {
    this.tweens.add({
      targets: f.img,
      angle: 9,
      duration: 70,
      yoyo: true,
      repeat: Math.ceil(MISS_WARN_MS / 140),
      ease: "Sine.easeInOut",
    });
    this.tweens.add({
      targets: f.img,
      scaleX: f.baseScale * 0.92,
      scaleY: f.baseScale * 0.92,
      duration: MISS_WARN_MS,
      ease: "Quad.easeIn",
    });
  }

  /** 7 — falls out of frame under gravity, turning and fading. */
  private dropAway(f: Fruit, index: number) {
    f.dying = true;
    this.fruits.splice(index, 1);
    this.control.missed += 1;
    this.control.audio.miss();
    this.tweens.killTweensOf(f.img);

    const u = this.unit;
    const drift = (Math.random() * 2 - 1) * u * 0.07;
    // The stem is left behind on the branch and withers.
    if (f.sprig) {
      const sp = f.sprig;
      this.tweens.add({
        targets: sp,
        alpha: 0,
        duration: 500,
        onComplete: () => sp.destroy(),
      });
      f.sprig = null;
    }

    // Lands ON THE GROUND beside the basket rather than falling out of
    // frame — there is a ground plane now, so fruit should reach it.
    // Quad.easeIn on y is constant acceleration, so it reads as gravity.
    const groundY = this.scale.height * GROUND_Y + u * 0.02;
    const fall = 720;
    this.tweens.add({
      targets: f.img,
      y: groundY,
      duration: fall,
      ease: "Quad.easeIn",
      onComplete: () => {
        // Settle where it landed, then fade into the grass.
        this.tweens.add({
          targets: f.img,
          scaleY: f.baseScale * 0.78,
          scaleX: f.baseScale * 1.05,
          duration: 90,
          yoyo: true,
          ease: "Quad.easeOut",
        });
        this.tweens.add({
          targets: f.img,
          alpha: 0,
          delay: 260,
          duration: 420,
          onComplete: () => f.img.destroy(),
        });
      },
    });
    this.tweens.add({
      targets: f.img,
      x: f.img.x + drift,
      angle: drift > 0 ? 190 : -190,
      duration: fall,
      ease: "Sine.easeIn",
    });
  }
}
