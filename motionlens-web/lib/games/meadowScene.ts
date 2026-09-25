// Procedural open-sky backdrop for Kite Flying.
//
// Same arrangement as orchardScene.ts and skyScene.ts: drawn ONCE into
// a canvas texture at create() and blitted as a single sprite. Nothing
// here runs in the frame loop.
//
// CONTRAST. The things that must read at 2 m are a light-blue ribbon
// and a kite with a green or amber glow. So the sky is a clean pale
// gradient with no busy detail, the hills are muted and low, and the
// grass is dark enough that a kite never disappears into it. If a
// change makes the background more interesting it is probably wrong.

import Phaser from "phaser";

/** Deterministic RNG so one round's meadow is stable. */
function mulberry(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Where the grass starts, as a fraction of height. Low, so the sky —
 *  which is where the whole game happens — keeps the screen. */
export const MEADOW_GROUND = 0.86;

export function makeMeadowTexture(
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

  // ── Sky. Deeper at the top, pale at the horizon. Kept light overall:
  //    the kite carries a bright glow and needs somewhere to stand out
  //    against without the whole screen going dark.
  const sky = ctx.createLinearGradient(0, 0, 0, h * MEADOW_GROUND);
  sky.addColorStop(0, "#4a8fc4");
  sky.addColorStop(0.55, "#8ec4e2");
  sky.addColorStop(1, "#d6e9f2");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // Sun glow, high and to one side, well away from the middle band
  // where the corridor lives.
  const g = ctx.createRadialGradient(
    w * 0.85, h * 0.12, 0, w * 0.85, h * 0.12, u * 0.4,
  );
  g.addColorStop(0, "rgba(255,250,224,0.45)");
  g.addColorStop(1, "rgba(255,250,224,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  // ── Distant hills. Each layer paler than the one in front: cheap
  //    atmospheric perspective, depth without detail.
  const hills = [
    { y: 0.76, amp: 0.045, col: "#9db9a8", n: 3 },
    { y: 0.81, amp: 0.035, col: "#7aa089", n: 4 },
  ];
  for (const hl of hills) {
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 8) {
      let y = h * hl.y;
      for (let k = 1; k <= hl.n; k++) {
        y += Math.sin((x / w) * Math.PI * 2 * k + rnd() * 0.0001 + k)
          * h * hl.amp / k;
      }
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = hl.col;
    ctx.fill();
  }

  // ── Grass.
  const grass = ctx.createLinearGradient(0, h * MEADOW_GROUND, 0, h);
  grass.addColorStop(0, "#5f8a51");
  grass.addColorStop(1, "#3d5c36");
  ctx.fillStyle = grass;
  ctx.fillRect(0, h * MEADOW_GROUND, w, h * (1 - MEADOW_GROUND));

  // A few blade tufts along the skyline, so the grass edge is not a
  // ruled line.
  ctx.strokeStyle = "rgba(120,160,100,0.55)";
  ctx.lineWidth = Math.max(1, u * 0.0022);
  for (let i = 0; i < 160; i++) {
    const x = rnd() * w;
    const base = h * MEADOW_GROUND + rnd() * h * 0.02;
    const len = u * (0.012 + rnd() * 0.02);
    ctx.beginPath();
    ctx.moveTo(x, base);
    ctx.lineTo(x + (rnd() - 0.5) * len * 0.6, base - len);
    ctx.stroke();
  }

  tex.refresh();
  return true;
}
