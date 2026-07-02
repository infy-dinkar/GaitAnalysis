// Static skeleton redraw from saved landmark coordinates.
//
// This helper is the report-side counterpart to the LIVE
// draw path in RehabCameraShell.drawSkeleton — it reproduces a
// bones-and-dots skeleton on a stand-alone canvas from a saved
// keypoint array, with auto-fit + centering so the framing is
// controllable regardless of how the patient was positioned in
// the original camera view.
//
// Design boundary:
//   • Pure Canvas 2D. No React, no MediaPipe, no camera.
//   • Reads only `SKELETON_EDGES_LIVE` from lib/pose/landmarks-live
//     for the bone topology — same edges the live overlay uses, so
//     the redrawn skeleton is anatomically identical.
//   • Never mutates its inputs; two calls with the same landmarks
//     produce byte-identical output.
//
// The saved landmark array is expected to carry raw source-frame
// pixel coordinates (x, y) + a visibility score. This helper
// normalises to [0, 1], mirrors x so the redrawn skeleton reads
// selfie-oriented (matches the live view), computes the visible-
// bounding-box, then auto-fits into the target canvas with a
// configurable padding.

import {
  SKELETON_EDGES_LIVE as SKELETON_EDGES,
} from "@/lib/pose/landmarks-live";

export interface SavedLandmark {
  x: number;
  y: number;
  score?: number;
  name?: string;
}

export interface SourceFrame {
  width: number;
  height: number;
}

export interface DrawSkeletonOpts {
  /** Score below which a landmark is skipped for the bounding-box
   *  computation AND for edge/dot drawing. Default 0.35 — matches
   *  the OVERLAY_VIS_THRESHOLD the live shells use. */
  visibilityThreshold?: number;
  /** Fraction of the SHORTER canvas side reserved as padding on
   *  each edge. Default 0.10 (10 % all round). */
  paddingFraction?: number;
  /** Bone stroke colour. Default "#0F172A" (slate-900). */
  boneColor?: string;
  /** Bone stroke width in canvas pixels. Default 3. */
  boneWidth?: number;
  /** Dot fill colour. Default "#0EA5E9" (sky-500). */
  dotColor?: string;
  /** Dot radius in canvas pixels. Default 5. */
  dotRadius?: number;
  /** Background fill. Pass `null` (default) for a transparent
   *  canvas so the report card's own surface shows through. */
  background?: string | null;
}

/**
 * Redraw a clean skeleton on the given Canvas 2D context.
 *
 * @param ctx           target canvas 2d context
 * @param landmarks     saved landmark array (raw source-pixel coords)
 * @param sourceFrame   dimensions of the original video frame the
 *                      landmarks came from — needed for normalisation
 * @param targetW       width of the target canvas (in canvas pixels)
 * @param targetH       height of the target canvas (in canvas pixels)
 * @param opts          styling knobs
 */
export function drawSkeletonFromLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: SavedLandmark[],
  sourceFrame: SourceFrame,
  targetW: number,
  targetH: number,
  opts?: DrawSkeletonOpts,
): void {
  if (!landmarks || landmarks.length === 0) return;
  if (!sourceFrame || sourceFrame.width <= 0 || sourceFrame.height <= 0) return;

  const threshold = opts?.visibilityThreshold ?? 0.35;
  const paddingFraction = opts?.paddingFraction ?? 0.10;
  const boneColor = opts?.boneColor ?? "#0F172A";
  const boneWidth = opts?.boneWidth ?? 3;
  const dotColor = opts?.dotColor ?? "#0EA5E9";
  const dotRadius = opts?.dotRadius ?? 5;
  const background = opts?.background ?? null;

  // Background fill (or transparent).
  if (background !== null) {
    ctx.save();
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, targetW, targetH);
    ctx.restore();
  }

  // Normalise to [0, 1] AND mirror x so the redraw reads
  // selfie-oriented (patient's right → screen right). Matches the
  // convention `RehabCameraShell` uses: `1 - p.x / sw`.
  const norm = landmarks.map((k) => ({
    x: 1 - k.x / sourceFrame.width,
    y: k.y / sourceFrame.height,
    visibility: k.score ?? 0,
  }));

  // Bounding box over VISIBLE-only landmarks. Skipping occluded
  // points prevents a stray low-score keypoint from dragging the
  // fit off-centre.
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let visibleCount = 0;
  for (const p of norm) {
    if (p.visibility < threshold) continue;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
    visibleCount++;
  }
  if (visibleCount === 0) return;

  const bboxW = Math.max(1e-6, maxX - minX);
  const bboxH = Math.max(1e-6, maxY - minY);

  // Auto-fit — scale so the visible bounding box occupies as much
  // usable space as possible without exceeding the padded frame.
  const pad = Math.min(targetW, targetH) * paddingFraction;
  const usableW = Math.max(1, targetW - 2 * pad);
  const usableH = Math.max(1, targetH - 2 * pad);
  const scale = Math.min(usableW / bboxW, usableH / bboxH);

  // Centre the bounding box on the canvas centre.
  const cxCanvas = targetW / 2;
  const cyCanvas = targetH / 2;
  const cxBbox = (minX + maxX) / 2;
  const cyBbox = (minY + maxY) / 2;
  const project = (nx: number, ny: number): { x: number; y: number } => ({
    x: cxCanvas + (nx - cxBbox) * scale,
    y: cyCanvas + (ny - cyBbox) * scale,
  });

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Bones — one line per edge, both endpoints visible.
  ctx.strokeStyle = boneColor;
  ctx.lineWidth = boneWidth;
  for (const [a, b] of SKELETON_EDGES) {
    const p = norm[a];
    const q = norm[b];
    if (!p || !q) continue;
    if (p.visibility < threshold || q.visibility < threshold) continue;
    const A = project(p.x, p.y);
    const B = project(q.x, q.y);
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.stroke();
  }

  // Dots — one filled circle per visible landmark.
  ctx.fillStyle = dotColor;
  for (const p of norm) {
    if (p.visibility < threshold) continue;
    const P = project(p.x, p.y);
    ctx.beginPath();
    ctx.arc(P.x, P.y, dotRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}
