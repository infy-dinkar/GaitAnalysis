// Procedural rain-sky backdrop for Cloudburst.
//
// Same arrangement as orchardScene.ts: drawn ONCE into a canvas texture
// at create() and blitted as a single sprite. Nothing here runs in the
// frame loop.
//
// CONTRAST IS THE WHOLE JOB. The items falling through this are a blue
// drop and an amber bolt, each on its own bright halo, and the patient
// is 2 m away. So the sky is deliberately desaturated and mid-dark, the
// clouds are low-contrast, and the rain is barely there. If a change
// makes the background more interesting, it is probably wrong: check it
// in greyscale, and if the items stop being the brightest things on
// screen, revert it.
//
// The drifting clouds are NOT in this texture — they are separate
// sprites tweened by the scene, because a drifting cloud cannot be
// baked into a still image.

import Phaser from "phaser";

/** Deterministic RNG so one round's sky is stable. */
function mulberry(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Ground line, as a fraction of height. Low, so it never intrudes on
 *  the play area — the patient reaches in the upper two thirds. */
export const SKY_GROUND = 0.88;

/** One soft cloud shape, as overlapping lobes. */
function cloudBlob(
  ctx: CanvasRenderingContext2D,
  rnd: () => number,
  x: number,
  y: number,
  r: number,
): void {
  const lobes = 5 + Math.floor(rnd() * 3);
  for (let i = 0; i < lobes; i++) {
    const lx = x + (i - lobes / 2) * r * 0.55 + (rnd() - 0.5) * r * 0.3;
    const ly = y + (rnd() - 0.5) * r * 0.35;
    const lr = r * (0.55 + rnd() * 0.5);
    ctx.beginPath();
    ctx.ellipse(lx, ly, lr, lr * 0.62, 0, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * The still part of the sky: gradient, far clouds, rain, ground.
 * @returns false when the texture could not be created, so the caller
 *          can fall back to a flat wash rather than show nothing.
 */
export function makeSkyTexture(
  scene: Phaser.Scene,
  key: string,
  w: number,
  h: number,
  seed: number,
): boolean {
  if (scene.textures.exists(key)) scene.textures.remove(key);
  const tex = scene.textures.createCanvas(key, Math.max(2, w), Math.max(2, h));
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  const rnd = mulberry(seed);
  const u = Math.min(w, h);

  // ── Sky. Darker overhead than at the horizon, which is both what a
  //    rain sky looks like and what puts the most contrast behind the
  //    items — they spend most of their fall in the upper half.
  const sky = ctx.createLinearGradient(0, 0, 0, h * SKY_GROUND);
  sky.addColorStop(0, "#2c3f57");
  sky.addColorStop(0.5, "#4a6379");
  sky.addColorStop(1, "#8ea3b3");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // ── Far clouds, baked in. These are the distant layer; the scene
  //    tweens a few nearer ones over the top.
  ctx.fillStyle = "rgba(226,234,242,0.16)";
  for (let i = 0; i < 7; i++) {
    cloudBlob(
      ctx,
      rnd,
      rnd() * w,
      h * (0.06 + rnd() * 0.38),
      u * (0.07 + rnd() * 0.09),
    );
  }

  // ── Rain. Thin, near-vertical, low alpha and thin enough to read as
  //    atmosphere rather than as something to reach for. Leaning the
  //    same way as the clouds drift keeps the scene coherent.
  ctx.lineWidth = Math.max(1, u * 0.0016);
  ctx.strokeStyle = "rgba(214,230,244,0.13)";
  const len = u * 0.045;
  for (let i = 0; i < 220; i++) {
    const x = rnd() * w;
    const y = rnd() * h * SKY_GROUND;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - len * 0.18, y + len);
    ctx.stroke();
  }

  // ── Ground. A wet band with a paler wash at the waterline, so the
  //    bottom of the play area has an edge the patient can see.
  const ground = ctx.createLinearGradient(0, h * SKY_GROUND, 0, h);
  ground.addColorStop(0, "#5d6f63");
  ground.addColorStop(1, "#3b4a44");
  ctx.fillStyle = ground;
  ctx.fillRect(0, h * SKY_GROUND, w, h * (1 - SKY_GROUND));
  ctx.fillStyle = "rgba(220,236,240,0.20)";
  ctx.fillRect(0, h * SKY_GROUND, w, Math.max(2, h * 0.006));

  // Puddles: flat ellipses, a shade lighter, low and sparse.
  ctx.fillStyle = "rgba(196,220,226,0.13)";
  for (let i = 0; i < 6; i++) {
    const px = rnd() * w;
    const py = h * (SKY_GROUND + 0.02 + rnd() * 0.07);
    const pr = u * (0.03 + rnd() * 0.06);
    ctx.beginPath();
    ctx.ellipse(px, py, pr, pr * 0.16, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  tex.refresh();
  return true;
}

/** Soft white cloud sprite, tinted and tweened by the scene for the
 *  nearer drifting layer. Transparent background so it can sit over the
 *  baked sky. */
export function makeCloudTexture(
  scene: Phaser.Scene,
  key: string,
  seed: number,
): boolean {
  if (scene.textures.exists(key)) return true;
  const W = 256;
  const H = 128;
  const tex = scene.textures.createCanvas(key, W, H);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  const rnd = mulberry(seed);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  cloudBlob(ctx, rnd, W / 2, H * 0.55, H * 0.33);
  tex.refresh();
  return true;
}
