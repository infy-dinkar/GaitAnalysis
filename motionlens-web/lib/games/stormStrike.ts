// The visuals of Cloudburst's big centre strike.
//
// Purely presentational. The event's timing, the band it targets, and
// whether the patient was hit or dodged all stay in the scene's state
// machine — this file is handed a rectangle and a moment and draws a
// storm into it. Nothing here is read back.
//
// TECHNIQUE: procedural, not a sprite.
//
// A bolt is drawn as a midpoint-displaced polyline stroked FOUR times
// at decreasing width and increasing brightness, with additive
// blending, so the passes sum into a soft halo around a hard core.
// That is how 2-D lightning is done, and it beats a bitmap here for
// three reasons: the band's width and height depend on the patient's
// calibrated reach, so a sprite would have to be stretched by a
// different amount for every person; a new shape is generated per
// strike, which a sprite sheet can only fake with many frames; and the
// whole games module already generates its art at runtime and ships no
// binary assets.
//
// PHOTOSENSITIVITY. One bright moment per strike: the bolt, the
// ground impact and the clouds lighting from inside all happen on the
// same frame and fade together. The warning's cloud flickers are
// small, dim (alpha <= 0.28) and spaced at least 420 ms apart. Nothing
// is full-screen and nothing is white except a thin core line.

import Phaser from "phaser";

/** Deterministic RNG, so one strike's shape is stable across the
 *  frames it is drawn on. */
function mulberry(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Depths. The storm sits BELOW the falling items (9/10) so a drop
//    crossing the danger zone is never hidden — clinically the item
//    has to stay readable, even at the cost of the clouds looking
//    slightly less like they are in front. The bolt alone goes above,
//    because at that moment it is the only thing that matters.
const D_DARKEN = 5;
const D_RAIN = 6;
const D_GROUND_GLOW = 7;
const D_CLOUD = 8;
const D_AFTERIMAGE = 11;
const D_BOLT = 12;

// ── Bolt layers, outermost first. Widths are fractions of the canvas
//    unit; with ADD blending the four passes sum to a glow.
const BOLT_LAYERS: { w: number; colour: number; alpha: number }[] = [
  { w: 0.055, colour: 0x2a5ea8, alpha: 0.18 },
  { w: 0.026, colour: 0x5b9bff, alpha: 0.3 },
  { w: 0.011, colour: 0xaed4ff, alpha: 0.55 },
  { w: 0.0035, colour: 0xffffff, alpha: 0.95 },
];
/** Branches are thinner, dimmer, and never get the white core. */
const BRANCH_SCALE = 0.55;
const BRANCH_ALPHA = 0.6;

/** How long the bolt holds at full before it starts to go, and how
 *  long the fade takes. The scene's strike PHASE is shorter; these
 *  tweens simply outlive it, which changes no timing or logic. */
const BOLT_HOLD_MS = 170;
const BOLT_FADE_MS = 250;
const AFTERIMAGE_FADE_MS = 420;

const CLOUD_TEX_W = 320;
const CLOUD_TEX_H = 200;

interface Pt {
  x: number;
  y: number;
}

/**
 * One storm cloud, soft-edged and shaded: lighter along the top,
 * markedly darker along the base, so it reads as lit from above rather
 * than as a grey blob.
 */
export function makeStormCloudTexture(
  scene: Phaser.Scene,
  key: string,
  seed: number,
): boolean {
  if (scene.textures.exists(key)) return true;
  const tex = scene.textures.createCanvas(key, CLOUD_TEX_W, CLOUD_TEX_H);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  const rnd = mulberry(seed);
  ctx.clearRect(0, 0, CLOUD_TEX_W, CLOUD_TEX_H);

  // Lobes, each a soft radial. The lobe's own colour is picked from
  // its height in the cloud, which is what gives the internal shading
  // — a flat fill with a gradient laid over the top reads as a sticker.
  const lobes = 12 + Math.floor(rnd() * 6);
  for (let i = 0; i < lobes; i++) {
    const t = rnd();
    // Bias lobes toward the upper half so the base stays flatter, the
    // way a storm cloud's underside does.
    const cy = CLOUD_TEX_H * (0.3 + t * 0.42);
    const cx = CLOUD_TEX_W * (0.1 + rnd() * 0.8);
    const r = CLOUD_TEX_W * (0.1 + rnd() * 0.16);
    // 0 at the top of the cloud, 1 at the base.
    const depth = Math.min(1, Math.max(0, (cy / CLOUD_TEX_H - 0.25) / 0.6));
    const lum = Math.round(150 - depth * 108);
    const g = ctx.createRadialGradient(cx, cy - r * 0.25, r * 0.1, cx, cy, r);
    g.addColorStop(0, `rgba(${lum + 22},${lum + 28},${lum + 40},0.92)`);
    g.addColorStop(0.55, `rgba(${lum},${lum + 6},${lum + 18},0.6)`);
    g.addColorStop(1, `rgba(${lum},${lum + 6},${lum + 18},0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(cx, cy, r, r * 0.8, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // Deepen the base only where something was already drawn, so the
  // cloud keeps its soft silhouette instead of gaining a straight edge.
  ctx.globalCompositeOperation = "source-atop";
  const base = ctx.createLinearGradient(0, CLOUD_TEX_H * 0.45, 0, CLOUD_TEX_H);
  base.addColorStop(0, "rgba(10,14,24,0)");
  base.addColorStop(1, "rgba(6,9,16,0.75)");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, CLOUD_TEX_W, CLOUD_TEX_H);
  ctx.globalCompositeOperation = "source-over";

  tex.refresh();
  return true;
}

/** A single rain streak: a soft vertical line, brightest in the
 *  middle, transparent at both ends. */
export function makeRainStreakTexture(scene: Phaser.Scene, key: string): boolean {
  if (scene.textures.exists(key)) return true;
  const W = 6;
  const H = 64;
  const tex = scene.textures.createCanvas(key, W, H);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  ctx.clearRect(0, 0, W, H);
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, "rgba(255,255,255,0)");
  g.addColorStop(0.4, "rgba(255,255,255,0.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(W * 0.3, 0, W * 0.4, H);
  tex.refresh();
  return true;
}

/**
 * Midpoint displacement between two points.
 *
 * Each generation inserts a midpoint pushed perpendicular to its
 * segment, with the push halving each time — so the path has both a
 * big overall wander and fine kinks, which is what makes it read as
 * lightning rather than as a zigzag.
 */
function displacePath(
  a: Pt,
  b: Pt,
  generations: number,
  offset: number,
  rnd: () => number,
): Pt[] {
  let pts: Pt[] = [a, b];
  let push = offset;
  for (let g = 0; g < generations; g++) {
    const next: Pt[] = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const p = pts[i];
      const q = pts[i + 1];
      const mx = (p.x + q.x) / 2;
      const my = (p.y + q.y) / 2;
      const dx = q.x - p.x;
      const dy = q.y - p.y;
      const len = Math.hypot(dx, dy) || 1;
      // Perpendicular to this segment, either side.
      const nx = -dy / len;
      const ny = dx / len;
      const d = (rnd() * 2 - 1) * push;
      next.push(p, { x: mx + nx * d, y: my + ny * d });
    }
    next.push(pts[pts.length - 1]);
    pts = next;
    push *= 0.5;
  }
  return pts;
}

function strokePolyline(
  g: Phaser.GameObjects.Graphics,
  pts: Pt[],
  width: number,
  colour: number,
  alpha: number,
): void {
  if (pts.length < 2 || width <= 0) return;
  g.lineStyle(width, colour, alpha);
  g.beginPath();
  g.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) g.lineTo(pts[i].x, pts[i].y);
  g.strokePath();
}

/**
 * Stroke a path in chunks of slightly varying width.
 *
 * A bolt of exactly constant thickness looks like a drawn line. Five
 * chunks is enough to break that up without the cost of restyling
 * every segment.
 */
function strokeVarying(
  g: Phaser.GameObjects.Graphics,
  pts: Pt[],
  width: number,
  colour: number,
  alpha: number,
  rnd: () => number,
): void {
  const CHUNKS = 5;
  const per = Math.max(2, Math.ceil(pts.length / CHUNKS));
  for (let i = 0; i < pts.length - 1; i += per - 1) {
    const slice = pts.slice(i, Math.min(pts.length, i + per));
    strokePolyline(g, slice, width * (0.72 + rnd() * 0.62), colour, alpha);
  }
}

export interface StormBand {
  loX: number;
  hiX: number;
  top: number;
  bottom: number;
}

/**
 * Owns every object the strike event draws.
 *
 * The scene calls beginWarning -> fire -> dissipate in that order and
 * never inspects what comes back.
 */
export class StormStrike {
  private scene: Phaser.Scene;
  private cloudKeys: string[] = [];
  private rainKey: string | null = null;
  /** The soft white radial the rest of the game already generates. */
  private dotKey: string;

  private clouds: Phaser.GameObjects.Image[] = [];
  private rain: Phaser.GameObjects.Image[] = [];
  private darken: Phaser.GameObjects.Image | null = null;
  private groundGlow: Phaser.GameObjects.Image | null = null;
  private transient: Phaser.GameObjects.GameObject[] = [];
  private band: StormBand = { loX: 0, hiX: 0, top: 0, bottom: 0 };
  private seed = 1;

  constructor(scene: Phaser.Scene, dotKey: string) {
    this.scene = scene;
    this.dotKey = dotKey;
    for (let i = 0; i < 3; i++) {
      const key = `cb-storm-${i}`;
      if (makeStormCloudTexture(scene, key, 1000 + i * 977)) this.cloudKeys.push(key);
    }
    this.rainKey = makeRainStreakTexture(scene, "cb-rain") ? "cb-rain" : null;
  }

  private get unit(): number {
    return Math.min(this.scene.scale.width, this.scene.scale.height);
  }

  private rnd = Math.random;

  /**
   * The ~1.5 s build-up.
   *
   * Clouds slide in from both sides and thicken, the sky under them
   * darkens as a soft radial (never a box), heavier rain falls inside
   * the band, and the ground where the bolt will land starts to glow.
   * Together those are what mark the danger area — there is no filled
   * rectangle anywhere in this file.
   */
  beginWarning(band: StormBand, durationMs: number): void {
    this.clear();
    this.band = band;
    this.seed = Math.floor(Math.random() * 1e6);
    const s = this.scene;
    const u = this.unit;
    const cx = (band.loX + band.hiX) / 2;
    const bw = Math.max(u * 0.12, band.hiX - band.loX);
    const playH = Math.max(u * 0.2, band.bottom - band.top);

    // ── Sky darkening. A wide soft radial centred over the band: at
    //    its strongest above the band and falling away in every
    //    direction, so the danger reads as weather rather than as UI.
    if (s.textures.exists(this.dotKey)) {
      this.darken = s.add
        .image(cx, band.top + playH * 0.15, this.dotKey)
        .setDisplaySize(bw * 4.2, playH * 1.9)
        .setTint(0x070d18)
        .setAlpha(0)
        .setDepth(D_DARKEN);
      s.tweens.add({
        targets: this.darken,
        alpha: 0.55,
        duration: durationMs * 0.8,
        ease: "Sine.easeIn",
      });

      // ── Ground glow where the bolt will land.
      this.groundGlow = s.add
        .image(cx, band.bottom, this.dotKey)
        .setDisplaySize(bw * 1.4, u * 0.1)
        .setTint(0x8fb8ff)
        .setAlpha(0)
        .setBlendMode(Phaser.BlendModes.ADD)
        .setDepth(D_GROUND_GLOW);
      s.tweens.add({
        targets: this.groundGlow,
        alpha: 0.35,
        duration: durationMs * 0.9,
        ease: "Sine.easeIn",
      });
    }

    // ── Clouds roll in from BOTH sides and pile up over the band.
    //    Each slides and grows at once, so they gather rather than pop.
    if (this.cloudKeys.length > 0) {
      const count = 6;
      for (let i = 0; i < count; i++) {
        const key = this.cloudKeys[i % this.cloudKeys.length];
        const fromLeft = i % 2 === 0;
        const w = bw * (1.0 + Math.random() * 0.9);
        const h = w * (CLOUD_TEX_H / CLOUD_TEX_W) * (0.85 + Math.random() * 0.3);
        // Final resting place: spread across the band's width and a
        // little past it, stacked in a shallow bank.
        const toX = cx + (Math.random() - 0.5) * bw * 1.7;
        const toY = band.top + u * 0.02 + (Math.random() - 0.5) * u * 0.05;
        const img = s.add
          .image(fromLeft ? -w : s.scale.width + w, toY, key)
          .setDisplaySize(w * 0.6, h * 0.6)
          .setAlpha(0)
          .setDepth(D_CLOUD);
        this.clouds.push(img);
        s.tweens.add({
          targets: img,
          x: toX,
          displayWidth: w,
          displayHeight: h,
          alpha: 0.82 + Math.random() * 0.15,
          duration: durationMs * (0.68 + Math.random() * 0.3),
          delay: i * (durationMs * 0.05),
          ease: "Cubic.easeOut",
        });
      }
    }

    // ── Heavier rain, inside the band only.
    if (this.rainKey) {
      const streaks = 16;
      for (let i = 0; i < streaks; i++) {
        const len = u * (0.07 + Math.random() * 0.06);
        const img = s.add
          .image(
            band.loX + Math.random() * bw,
            band.top + Math.random() * playH,
            this.rainKey,
          )
          .setDisplaySize(Math.max(1.5, u * 0.004), len)
          .setTint(0xcfe0f5)
          .setAlpha(0.32 + Math.random() * 0.2)
          .setDepth(D_RAIN);
        this.rain.push(img);
        s.tweens.add({
          targets: img,
          y: band.bottom,
          duration: 420 + Math.random() * 320,
          repeat: -1,
          ease: "Linear",
          onRepeat: () => {
            img.setPosition(band.loX + Math.random() * bw, band.top - len);
          },
        });
      }
    }

    // ── Dim flickers inside the clouds. Two or three, small, never
    //    outside a cloud, and spaced so they cannot read as a strobe.
    const flickers = 2 + Math.round(Math.random());
    for (let i = 0; i < flickers; i++) {
      s.time.delayedCall(durationMs * 0.3 + i * 430, () => this.flicker());
    }
  }

  /** One small, dim glow inside a cloud. */
  private flicker(): void {
    if (this.clouds.length === 0 || !this.scene.textures.exists(this.dotKey)) return;
    const cloud = this.clouds[Math.floor(Math.random() * this.clouds.length)];
    if (!cloud.active) return;
    const r = this.unit * (0.07 + Math.random() * 0.05);
    const dot = this.scene.add
      .image(
        cloud.x + (Math.random() - 0.5) * cloud.displayWidth * 0.5,
        cloud.y + (Math.random() - 0.5) * cloud.displayHeight * 0.35,
        this.dotKey,
      )
      .setDisplaySize(r * 2, r * 1.4)
      .setTint(0x9ec5ff)
      .setAlpha(0)
      .setBlendMode(Phaser.BlendModes.ADD)
      .setDepth(D_CLOUD + 0.1);
    this.transient.push(dot);
    this.scene.tweens.add({
      targets: dot,
      alpha: 0.28,
      duration: 70,
      yoyo: true,
      ease: "Quad.easeOut",
      onComplete: () => dot.destroy(),
    });
  }

  /**
   * The strike. One bright moment: bolt, ground impact, sparks and the
   * clouds lighting from inside, all on this frame and fading together.
   */
  fire(): void {
    const s = this.scene;
    const u = this.unit;
    const band = this.band;
    const rnd = mulberry(this.seed);
    const bw = Math.max(u * 0.12, band.hiX - band.loX);

    // Start under the cloud bank, end on the ground, both inside the
    // band — the bolt must land where the patient was told it would.
    const startX = band.loX + bw * (0.3 + rnd() * 0.4);
    const startY = band.top + u * 0.05;
    const endX = band.loX + bw * (0.25 + rnd() * 0.5);
    const endY = band.bottom;
    const drop = Math.max(u * 0.1, endY - startY);

    const main = displacePath(
      { x: startX, y: startY },
      { x: endX, y: endY },
      6,
      Math.min(bw * 0.42, drop * 0.16),
      rnd,
    );

    // 2-4 thinner branches, each leaving the main path partway down.
    const branches: Pt[][] = [];
    const nBranches = 2 + Math.floor(rnd() * 3);
    for (let i = 0; i < nBranches; i++) {
      const at = Math.floor(main.length * (0.2 + rnd() * 0.55));
      const from = main[Math.min(main.length - 2, at)];
      const remaining = endY - from.y;
      if (remaining < u * 0.05) continue;
      const side = rnd() < 0.5 ? -1 : 1;
      const to = {
        x: from.x + side * bw * (0.2 + rnd() * 0.45),
        y: from.y + remaining * (0.25 + rnd() * 0.45),
      };
      branches.push(displacePath(from, to, 4, bw * 0.12, rnd));
    }

    // Self-destroying objects from earlier strikes are still listed
    // here; drop them before adding more, so the list stays bounded
    // over a round rather than growing by ~15 every strike.
    this.transient = this.transient.filter((o) => o.active);

    const g = s.add.graphics().setDepth(D_BOLT);
    g.setBlendMode(Phaser.BlendModes.ADD);
    // The outer two layers again on their own object: this is what is
    // left hanging in the air after the core has gone.
    const after = s.add.graphics().setDepth(D_AFTERIMAGE);
    after.setBlendMode(Phaser.BlendModes.ADD);

    for (let li = 0; li < BOLT_LAYERS.length; li++) {
      const L = BOLT_LAYERS[li];
      const w = u * L.w;
      // Vary the thickness of the two inner layers only; the outer
      // glow should stay even or the halo looks lumpy.
      if (li >= 2) strokeVarying(g, main, w, L.colour, L.alpha, rnd);
      else strokePolyline(g, main, w, L.colour, L.alpha);

      for (const b of branches) {
        // Branches never get the pure-white core — a branch as bright
        // as the trunk reads as two bolts.
        const colour = li === BOLT_LAYERS.length - 1 ? 0xdcecff : L.colour;
        strokePolyline(g, b, w * BRANCH_SCALE, colour, L.alpha * BRANCH_ALPHA);
      }
      if (li < 2) {
        strokePolyline(after, main, w * 1.15, L.colour, L.alpha * 0.9);
        for (const b of branches) {
          strokePolyline(after, b, w * BRANCH_SCALE, L.colour, L.alpha * 0.5);
        }
      }
    }

    // Phaser 4's per-object Glow filter, if this build and renderer
    // support it. The layered strokes above already carry the look, so
    // this is an enhancement and a failure here must not cost the
    // strike — hence the try/catch and no reference kept.
    try {
      const withFilters = g as unknown as {
        enableFilters?: () => void;
        filters?: {
          internal?: {
            addGlow?: (...a: unknown[]) => { setPaddingOverride?: (v: null) => void };
          };
        };
      };
      withFilters.enableFilters?.();
      // Modest strength and quality: this runs on one object for under
      // half a second, at most once every 9-15 s, and the additive
      // passes are already carrying the look — it is a top-up, not the
      // effect. setPaddingOverride(null) lets Phaser grow the render
      // target so the glow is not clipped at the object's edges.
      const glow = withFilters.filters?.internal?.addGlow?.(
        0x9ec5ff, 3, 0, 1, false, 6, 8,
      );
      glow?.setPaddingOverride?.(null);
    } catch {
      // No filter support: the additive passes stand on their own.
    }

    after.setAlpha(0.45);
    this.transient.push(g, after);
    s.tweens.add({
      targets: g,
      alpha: 0,
      delay: BOLT_HOLD_MS,
      duration: BOLT_FADE_MS,
      ease: "Quad.easeIn",
      onComplete: () => g.destroy(),
    });
    s.tweens.add({
      targets: after,
      alpha: 0,
      delay: BOLT_HOLD_MS,
      duration: AFTERIMAGE_FADE_MS,
      ease: "Quad.easeIn",
      onComplete: () => after.destroy(),
    });

    if (!s.textures.exists(this.dotKey)) return;

    // ── Clouds lit from inside. Same frame as the bolt, so the whole
    //    strike is ONE bright event rather than two.
    const lit = s.add
      .image((band.loX + band.hiX) / 2, band.top + u * 0.02, this.dotKey)
      .setDisplaySize(bw * 2.6, u * 0.26)
      .setTint(0xbcd8ff)
      .setAlpha(0)
      .setBlendMode(Phaser.BlendModes.ADD)
      .setDepth(D_CLOUD + 0.2);
    this.transient.push(lit);
    s.tweens.add({
      targets: lit,
      alpha: 0.5,
      duration: 60,
      yoyo: true,
      hold: 60,
      ease: "Quad.easeOut",
      onComplete: () => lit.destroy(),
    });

    // ── Impact on the ground. Wide and flat, not a disc, so it reads
    //    as light spreading along the ground.
    const impact = s.add
      .image(endX, endY, this.dotKey)
      .setDisplaySize(bw * 0.4, u * 0.05)
      .setTint(0xdbeaff)
      .setAlpha(0.85)
      .setBlendMode(Phaser.BlendModes.ADD)
      .setDepth(D_BOLT);
    this.transient.push(impact);
    s.tweens.add({
      targets: impact,
      displayWidth: bw * 1.6,
      displayHeight: u * 0.11,
      alpha: 0,
      duration: 260,
      ease: "Cubic.easeOut",
      onComplete: () => impact.destroy(),
    });

    // ── Sparks. A handful, thrown up and out, falling back.
    const sparks = 6 + Math.floor(rnd() * 4);
    for (let i = 0; i < sparks; i++) {
      const size = u * (0.006 + rnd() * 0.008);
      const sp = s.add
        .image(endX, endY, this.dotKey)
        .setDisplaySize(size * 2, size * 2)
        .setTint(0xcfe4ff)
        .setAlpha(0.9)
        .setBlendMode(Phaser.BlendModes.ADD)
        .setDepth(D_BOLT);
      this.transient.push(sp);
      const dir = rnd() < 0.5 ? -1 : 1;
      s.tweens.add({
        targets: sp,
        x: endX + dir * bw * (0.1 + rnd() * 0.4),
        y: endY - u * (0.02 + rnd() * 0.06),
        alpha: 0,
        duration: 300 + rnd() * 260,
        ease: "Quad.easeOut",
        onComplete: () => sp.destroy(),
      });
    }
  }

  /** The storm moves off: clouds drift apart and fade over ~1.5 s. */
  dissipate(): void {
    const s = this.scene;
    const cx = (this.band.loX + this.band.hiX) / 2;
    for (const c of this.clouds) {
      const away = c.x < cx ? -1 : 1;
      s.tweens.killTweensOf(c);
      s.tweens.add({
        targets: c,
        x: c.x + away * s.scale.width * 0.35,
        alpha: 0,
        duration: 1500,
        ease: "Sine.easeIn",
        onComplete: () => c.destroy(),
      });
    }
    this.clouds = [];
    for (const r of this.rain) {
      s.tweens.killTweensOf(r);
      s.tweens.add({
        targets: r,
        alpha: 0,
        duration: 600,
        onComplete: () => r.destroy(),
      });
    }
    this.rain = [];
    if (this.darken) {
      const d = this.darken;
      this.darken = null;
      s.tweens.killTweensOf(d);
      s.tweens.add({
        targets: d,
        alpha: 0,
        duration: 1200,
        onComplete: () => d.destroy(),
      });
    }
    if (this.groundGlow) {
      const gg = this.groundGlow;
      this.groundGlow = null;
      s.tweens.killTweensOf(gg);
      s.tweens.add({
        targets: gg,
        alpha: 0,
        duration: 500,
        onComplete: () => gg.destroy(),
      });
    }
  }

  /** Hard stop: used when a new warning starts before the last storm
   *  has finished drifting away, and on teardown. */
  clear(): void {
    const s = this.scene;
    const all = [
      ...this.clouds,
      ...this.rain,
      ...this.transient,
      this.darken,
      this.groundGlow,
    ];
    for (const o of all) {
      if (!o) continue;
      s.tweens.killTweensOf(o);
      o.destroy();
    }
    this.clouds = [];
    this.rain = [];
    this.transient = [];
    this.darken = null;
    this.groundGlow = null;
  }

  destroy(): void {
    this.clear();
  }
}
