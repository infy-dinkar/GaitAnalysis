// Procedural orchard backdrop for Fruit Harvest.
//
// Drawn ONCE into a single canvas texture at scene create() and then
// blitted as one sprite. Nothing here runs in the frame loop — the
// per-frame cost of the whole backdrop is one full-screen image.
//
// The canopy fills the entire upper play area edge to edge rather than
// being fitted to the patient's reach box, so a fruit placed anywhere
// the calibration allows still reads as hanging on a tree.
//
// SIZING: every dimension is a fraction of the canvas. The one
// exception is the texture itself, which is created at the canvas's
// own pixel size.
//
// CONTRAST: the canopy is deliberately dark and low-saturation. Fruit
// are bright saturated emoji with their own dark halo, so they stay
// separable from the leaves at 2 m. Any change to the palette below
// should be checked in greyscale — if the fruit stop being the
// brightest things on screen, the change is wrong.

import Phaser from "phaser";

/** Deterministic RNG so one round's orchard is stable. */
function mulberry(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Horizon line, as a fraction of height. */
const HORIZON = 0.74;
/** Canopy reaches down to here; below it is trunk, hills and ground. */
const CANOPY_BOTTOM = 0.70;

/**
 * One leaf cluster: a lobed blob rather than a circle, so the canopy
 * edge reads as foliage instead of bubbles.
 */
function blob(
  ctx: CanvasRenderingContext2D,
  rnd: () => number,
  x: number,
  y: number,
  r: number,
): void {
  ctx.beginPath();
  const lobes = 6 + Math.floor(rnd() * 4);
  for (let i = 0; i <= lobes; i++) {
    const a = (i / lobes) * Math.PI * 2;
    const rr = r * (0.72 + rnd() * 0.45);
    const px = x + Math.cos(a) * rr;
    const py = y + Math.sin(a) * rr * 0.8;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();
}

function drawTrunk(
  ctx: CanvasRenderingContext2D,
  rnd: () => number,
  u: number,
  rootX: number,
  rootY: number,
  topX: number,
  topY: number,
  scale: number,
): void {
  const wRoot = u * 0.055 * scale;
  const wTop = u * 0.018 * scale;
  ctx.fillStyle = "#4d3422";
  ctx.beginPath();
  ctx.moveTo(rootX - wRoot, rootY);
  ctx.quadraticCurveTo(rootX - wRoot * 0.5, (rootY + topY) / 2, topX - wTop, topY);
  ctx.lineTo(topX + wTop, topY);
  ctx.quadraticCurveTo(rootX + wRoot * 0.5, (rootY + topY) / 2, rootX + wRoot, rootY);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "rgba(0,0,0,0.25)";
  ctx.lineWidth = Math.max(1, u * 0.004 * scale);
  ctx.lineCap = "round";
  for (let i = 0; i < 5; i++) {
    const t = (i + 1) / 6;
    const sx = rootX + (t - 0.5) * wRoot * 1.4;
    ctx.beginPath();
    ctx.moveTo(sx, rootY - u * 0.01);
    ctx.quadraticCurveTo(
      sx + (topX - rootX) * 0.35,
      (rootY + topY) / 2,
      topX + (t - 0.5) * wTop * 1.4,
      topY,
    );
    ctx.stroke();
  }

  // Branches lifting into the canopy, so the leaves have structure
  // beneath them instead of floating.
  ctx.strokeStyle = "#4d3422";
  ctx.lineWidth = u * 0.012 * scale;
  for (let i = -2; i <= 2; i++) {
    if (i === 0) continue;
    ctx.beginPath();
    ctx.moveTo(topX, topY + u * 0.02 * scale);
    ctx.quadraticCurveTo(
      topX + i * u * 0.05 * scale,
      topY - u * 0.03 * scale,
      topX + i * u * 0.11 * scale,
      topY - u * 0.09 * scale * (1 + rnd() * 0.5),
    );
    ctx.stroke();
  }

  ctx.fillStyle = "#3f2b1c";
  ctx.beginPath();
  ctx.ellipse(rootX, rootY, wRoot * 1.9, u * 0.016 * scale, 0, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Build the orchard texture.
 * @returns false if the canvas texture could not be created, so the
 *          caller can carry on with a plain background.
 */
export function makeOrchardTexture(
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

  // ── Sky
  const sky = ctx.createLinearGradient(0, 0, 0, h * HORIZON);
  sky.addColorStop(0, "#255f8c");
  sky.addColorStop(0.55, "#6fb3d2");
  sky.addColorStop(1, "#cfe6d8");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // Sun glow, kept low and to one side so it never sits behind fruit.
  const g = ctx.createRadialGradient(w * 0.12, h * 0.2, 0, w * 0.12, h * 0.2, u * 0.45);
  g.addColorStop(0, "rgba(255,246,214,0.5)");
  g.addColorStop(1, "rgba(255,246,214,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  // ── Distant hills. Each layer paler than the one in front: cheap
  //    atmospheric perspective, depth without detail.
  const hills = [
    { y: 0.64, amp: 0.05, col: "#7fae95", n: 3 },
    { y: 0.69, amp: 0.04, col: "#5e9678", col2: 0, n: 4 },
    { y: 0.73, amp: 0.03, col: "#417a5c", n: 5 },
  ];
  for (const hl of hills) {
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 8) {
      let y = h * hl.y;
      for (let k = 1; k <= hl.n; k++) {
        y -= Math.sin((x / w) * Math.PI * k * 1.3 + k * 2.1) * h * hl.amp / k;
      }
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = hl.col;
    ctx.fill();
  }

  // ── Ground
  const gr = ctx.createLinearGradient(0, h * HORIZON, 0, h);
  gr.addColorStop(0, "#37703f");
  gr.addColorStop(1, "#24512f");
  ctx.fillStyle = gr;
  ctx.fillRect(0, h * HORIZON, w, h * (1 - HORIZON));
  ctx.strokeStyle = "rgba(255,255,255,0.09)";
  ctx.lineWidth = Math.max(1, u * 0.003);
  for (let i = 0; i < 160; i++) {
    const x = rnd() * w;
    const y = h * (HORIZON + 0.01) + rnd() * h * (1 - HORIZON - 0.01);
    const l = u * 0.012 * (0.5 + rnd());
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + (rnd() - 0.5) * l * 0.6, y - l);
    ctx.stroke();
  }

  // ── Trunks, spread across the full width at varying scale. Back row
  //    first so the front row overlaps it.
  const rows = [
    { n: 5, scale: 0.62, y: 0.78, jitter: 0.02 },
    { n: 4, scale: 1.0, y: 0.9, jitter: 0.03 },
  ];
  for (const row of rows) {
    for (let i = 0; i < row.n; i++) {
      const fx = (i + 0.5) / row.n + (rnd() - 0.5) * row.jitter;
      const rootX = fx * w;
      const rootY = h * row.y;
      const topX = rootX + (rnd() - 0.5) * u * 0.08;
      const topY = h * (CANOPY_BOTTOM - 0.06 - rnd() * 0.08);
      drawTrunk(ctx, rnd, u, rootX, rootY, topX, topY, row.scale);
    }
  }

  // ── Canopy. A continuous band of foliage from above the top edge
  //    down to CANOPY_BOTTOM, across the FULL width — so wherever the
  //    calibrated reach puts a fruit, there are leaves behind it.
  //
  //    Three tone tiers, all dark and desaturated. Density is highest
  //    at the top and thins toward the bottom edge so the canopy has a
  //    ragged underside rather than a hard line.
  const tiers = [
    { col: "#17301d", n: 150, r: 0.105 },
    { col: "#1f4526", n: 120, r: 0.088 },
    { col: "#2a5a30", n: 85, r: 0.068 },
  ];
  const top = -h * 0.06;
  const bottom = h * CANOPY_BOTTOM;
  for (const t of tiers) {
    ctx.fillStyle = t.col;
    for (let i = 0; i < t.n; i++) {
      const x = -u * 0.1 + rnd() * (w + u * 0.2);
      // Bias upward: sqrt pushes samples toward the top of the band.
      const v = Math.sqrt(rnd());
      const y = top + v * (bottom - top);
      // Thin out near the bottom edge for a ragged underside.
      if (v > 0.8 && rnd() > 0.45) continue;
      blob(ctx, rnd, x, y, u * t.r * (0.6 + rnd() * 0.7));
    }
  }

  // Lit rim on the sun side only. Kept off the middle of the screen so
  // it never competes with the fruit for attention.
  ctx.fillStyle = "rgba(140,195,110,0.28)";
  for (let i = 0; i < 40; i++) {
    const x = rnd() * w * 0.45;
    const y = top + Math.sqrt(rnd()) * (bottom - top) * 0.7;
    blob(ctx, rnd, x, y, u * 0.045 * (0.5 + rnd() * 0.6));
  }

  // Gentle vignette: pulls the eye to the middle where the fruit are.
  const vg = ctx.createRadialGradient(
    w / 2, h * 0.45, u * 0.25,
    w / 2, h * 0.45, u * 0.85,
  );
  vg.addColorStop(0, "rgba(0,0,0,0)");
  vg.addColorStop(1, "rgba(0,0,0,0.34)");
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, w, h);

  tex.refresh();
  return true;
}

/**
 * Stem plus one leaf, drawn behind each fruit so it reads as attached
 * to a branch rather than floating. Anchored so the fruit's centre sits
 * at the texture's centre.
 */
export function makeSprigTexture(
  scene: Phaser.Scene,
  key: string,
  size = 128,
): boolean {
  if (scene.textures.exists(key)) return true;
  const tex = scene.textures.createCanvas(key, size, size);
  if (!tex) return false;
  const ctx = tex.getContext();
  if (!ctx) {
    scene.textures.remove(key);
    return false;
  }
  const c = size / 2;
  ctx.clearRect(0, 0, size, size);

  ctx.strokeStyle = "#4a3120";
  ctx.lineWidth = size * 0.055;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(c, c * 0.62);
  ctx.quadraticCurveTo(c + size * 0.03, c * 0.3, c + size * 0.02, size * 0.1);
  ctx.stroke();

  ctx.fillStyle = "#3f8442";
  ctx.beginPath();
  ctx.ellipse(c + size * 0.13, size * 0.2, size * 0.13, size * 0.062, 0.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.ellipse(c + size * 0.13, size * 0.19, size * 0.07, size * 0.026, 0.6, 0, Math.PI * 2);
  ctx.fill();

  tex.refresh();
  return true;
}
