// Visual effects for Fruit Harvest — collect burst, floating score.
//
// Everything is generated at runtime from Phaser primitives; no asset
// is downloaded or shipped. Each helper is self-contained and destroys
// what it creates, so a round leaves nothing behind.
//
// SIZING RULE, as everywhere in this module: `unit` is the shortest
// canvas edge and every dimension is a fraction of it. Nothing here
// may use a fixed pixel count.

import Phaser from "phaser";

const FX_TEX = 128;

/** Soft round dot, white so it can be tinted to any fruit colour. */
export function makeSoftDotTexture(scene: Phaser.Scene, key: string): boolean {
  if (scene.textures.exists(key)) return true;
  const tex = scene.textures.createCanvas(key, FX_TEX, FX_TEX);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  const c = FX_TEX / 2;
  const g = ctx.createRadialGradient(c, c, 0, c, c, c);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.45, "rgba(255,255,255,0.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, FX_TEX, FX_TEX);
  tex.refresh();
  return true;
}

/** Hollow ring with a soft edge — the contact flash. */
export function makeRingTexture(scene: Phaser.Scene, key: string): boolean {
  if (scene.textures.exists(key)) return true;
  const tex = scene.textures.createCanvas(key, FX_TEX, FX_TEX);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  const c = FX_TEX / 2;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = FX_TEX * 0.1;
  ctx.shadowColor = "rgba(255,255,255,0.9)";
  ctx.shadowBlur = FX_TEX * 0.12;
  ctx.beginPath();
  ctx.arc(c, c, c * 0.72, 0, Math.PI * 2);
  ctx.stroke();
  tex.refresh();
  return true;
}

/**
 * Mean colour of an already-rasterised glyph texture, weighted by
 * alpha so transparent margins do not wash it grey. Used to tint the
 * burst so a strawberry throws red and a kiwi throws green.
 *
 * Called once per fruit at create() — never in the frame loop.
 */
export function glyphAverageColour(
  scene: Phaser.Scene,
  key: string,
  fallback = 0xffd166,
): number {
  try {
    const tex = scene.textures.get(key);
    const src = tex?.getSourceImage() as HTMLCanvasElement | undefined;
    if (!src || typeof src.getContext !== "function") return fallback;
    const ctx = src.getContext("2d");
    if (!ctx) return fallback;
    const { width: w, height: h } = src;
    const data = ctx.getImageData(0, 0, w, h).data;
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    // Step 4 pixels: a 256x256 glyph is far denser than this needs.
    for (let i = 0; i < data.length; i += 16) {
      const a = data[i + 3];
      if (a < 120) continue;
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      n++;
    }
    if (n === 0) return fallback;
    return ((r / n) << 16) | ((g / n) << 8) | (b / n);
  } catch {
    // Canvas reads can be blocked; a fallback tint is not worth failing.
    return fallback;
  }
}

export interface BurstOpts {
  x: number;
  y: number;
  /** Shortest canvas edge. */
  unit: number;
  colour: number;
  visualScale: number;
  /** Texture key for the leaf glyph, or null to skip the leaves. */
  leafKey: string | null;
  dotKey: string;
  ringKey: string;
}

/**
 * Contact flash + particle burst + fluttering leaves.
 *
 * Object budget per call: 1 ring image, 1 particle emitter (12
 * particles, self-destroying), 3 leaf images, 6 tweens. All are torn
 * down within ~1.1 s.
 */
export function collectBurst(scene: Phaser.Scene, o: BurstOpts): void {
  const u = o.unit;
  const s = o.visualScale;

  // ── Flash ring, behind the fruit.
  if (scene.textures.exists(o.ringKey)) {
    const ring = scene.add
      .image(o.x, o.y, o.ringKey)
      .setDisplaySize(u * 0.09 * s, u * 0.09 * s)
      .setTint(o.colour)
      .setAlpha(0.95)
      .setDepth(8);
    scene.tweens.add({
      targets: ring,
      scaleX: ring.scaleX * 3.2,
      scaleY: ring.scaleY * 3.2,
      alpha: 0,
      duration: 340,
      ease: "Quad.easeOut",
      onComplete: () => ring.destroy(),
    });
  }

  // ── Particle burst, tinted to the fruit.
  if (scene.textures.exists(o.dotKey)) {
    const em = scene.add.particles(o.x, o.y, o.dotKey, {
      speed: { min: u * 0.15, max: u * 0.55 },
      angle: { min: 0, max: 360 },
      lifespan: { min: 320, max: 520 },
      scale: { start: (u * 0.055 * s) / FX_TEX, end: 0 },
      alpha: { start: 0.95, end: 0 },
      gravityY: u * 0.9,
      tint: o.colour,
      blendMode: "ADD",
      emitting: false,
    });
    em.setDepth(12);
    em.explode(12);
    // The emitter cannot free itself; one timer per burst.
    scene.time.delayedCall(700, () => em.destroy());
  }

  // ── Leaves: three, fluttering down.
  if (o.leafKey && scene.textures.exists(o.leafKey)) {
    for (let i = 0; i < 3; i++) {
      const dir = i - 1 || 0.6;
      const leaf = scene.add
        .image(o.x, o.y, o.leafKey)
        .setDisplaySize(u * 0.042 * s, u * 0.042 * s)
        .setDepth(11);
      const fall = 620 + i * 130;
      scene.tweens.add({
        targets: leaf,
        y: o.y + u * 0.2,
        angle: 180 * dir,
        alpha: 0,
        duration: fall,
        ease: "Sine.easeIn",
        onComplete: () => leaf.destroy(),
      });
      // Sway, independent of the fall — that pairing is the flutter.
      scene.tweens.add({
        targets: leaf,
        x: o.x + dir * u * 0.055,
        duration: fall / 3,
        ease: "Sine.easeInOut",
        yoyo: true,
        repeat: 1,
      });
    }
  }
}

/** Floating "+1", sized to read across a room. */
export function floatScore(
  scene: Phaser.Scene,
  x: number,
  y: number,
  unit: number,
  visualScale: number,
  label = "+1",
): void {
  const size = Math.round(unit * 0.075 * visualScale);
  const t = scene.add
    .text(x, y, label, {
      fontFamily: "system-ui, sans-serif",
      fontSize: `${size}px`,
      color: "#fde047",
      stroke: "#000000",
      strokeThickness: Math.max(2, size * 0.14),
    })
    .setOrigin(0.5)
    .setDepth(28);
  scene.tweens.add({
    targets: t,
    y: y - unit * 0.17,
    alpha: 0,
    duration: 780,
    ease: "Quad.easeOut",
    onComplete: () => t.destroy(),
  });
}
