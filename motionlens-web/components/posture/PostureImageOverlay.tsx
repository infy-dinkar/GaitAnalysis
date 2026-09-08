"use client";
// Renders an analyzed posture image with annotation overlay drawn via
// a layered canvas. Keypoints, reference horizontal/vertical lines,
// and measurement badges are drawn so the final composition looks
// like a clinical screening shot.

import { useEffect, useRef } from "react";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";
import type {
  FrontMeasurements,
  SideMeasurements,
} from "@/lib/posture/measurements";
import type { PostureSilhouette } from "@/lib/posture/analyzer";

interface Props {
  view: "front" | "side";
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  keypoints: Keypoint[];
  front?: FrontMeasurements;
  side?: SideMeasurements;
  /** Names this view in the ?posturedebug=1 log. The `view` prop above
   *  only selects WHICH renderer runs — back reuses "front" and the
   *  explicit sides reuse "side" — so without this the log could not
   *  tell the five views apart. Defaults to `view`. */
  viewLabel?: string;
  /** Body-relative reference geometry from the segmentation mask.
   *  Absent on reports saved before it existed and whenever the mask
   *  was unusable — every consumer below falls back to the previous
   *  landmark-anchored geometry, so the overlay is unchanged in that
   *  case. Nothing here affects a measurement. */
  silhouette?: PostureSilhouette;
}

/** How far a body-relative horizontal overshoots the silhouette at
 *  each end, as a fraction of that row's width. Enough that the line
 *  visibly clears the body edge instead of dying exactly on it. */
const EXTENT_OVERSHOOT = 0.06;

/** Matches posture_engine._POSTURE_VIS_THRESHOLD and the dot loops
 *  below. */
const VIS_THRESHOLD = 0.2;

/** A landmark is usable only above the visibility floor.
 *
 *  ⚠️ Object truthiness is NOT enough. The keypoint array always has
 *  17 entries; an undetected landmark is a PLACEHOLDER at (0, 0) with
 *  score 0, not a hole (posture_engine._empty_kp). `if (a && b)`
 *  therefore passes for an occluded ankle and averages in x = 0 —
 *  which is exactly how the side plumb ended up at half the body's x
 *  in profile views, where the far ankle drops out. */
function usable(p: Keypoint | undefined): p is Keypoint {
  return !!p && (p.score ?? 0) >= VIS_THRESHOLD;
}

/** Mean x of whichever of the two landmarks are actually visible;
 *  null when neither is, so the caller draws nothing rather than a
 *  confident line through a placeholder. */
function meanVisibleX(
  a: Keypoint | undefined, b: Keypoint | undefined,
): number | null {
  const av = usable(a);
  const bv = usable(b);
  if (av && bv) return (a.x + b.x) / 2;
  if (av) return a.x;
  if (bv) return b.x;
  return null;
}

/** One-shot per-view diagnostic, opt-in via ?posturedebug=1. Says
 *  whether the drawn vertical came from the mask or from the landmark
 *  fallback — the question that is otherwise unanswerable from a
 *  screenshot. */
function logVerticalSource(
  view: string,
  source: "silhouette" | "fallback",
  /** What the BACKEND referenced its shifts to — "silhouette" or
   *  "ankle". It should agree with `source`; a disagreement means the
   *  line and the numbers measure from different x again. */
  shiftReference?: string,
): void {
  if (typeof window === "undefined") return;
  try {
    if (new URLSearchParams(window.location.search).get("posturedebug") !== "1") {
      return;
    }
  } catch {
    return;
  }
  console.log(
    `[POSTURE] ${view} vertical source = ${source}`
    + (shiftReference ? ` | metric shiftReference = ${shiftReference}` : "")
    + (shiftReference
      && ((source === "silhouette") !== (shiftReference === "silhouette"))
      ? "  ⚠ MISMATCH — line and numbers use different references"
      : ""),
  );
}

const FRONT_DOTS = [
  LM.LEFT_EAR, LM.RIGHT_EAR,
  LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  LM.LEFT_HIP, LM.RIGHT_HIP,
  LM.LEFT_KNEE, LM.RIGHT_KNEE,
  LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
];

const SIDE_DOTS = [
  LM.LEFT_EAR, LM.RIGHT_EAR,
  LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  LM.LEFT_HIP, LM.RIGHT_HIP,
  LM.LEFT_KNEE, LM.RIGHT_KNEE,
  LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
];

export function PostureImageOverlay({
  view,
  imageUrl,
  imageWidth,
  imageHeight,
  keypoints,
  front,
  side,
  silhouette,
  viewLabel,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = imageWidth;
    canvas.height = imageHeight;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    img.onload = () => {
      ctx.clearRect(0, 0, imageWidth, imageHeight);
      ctx.drawImage(img, 0, 0, imageWidth, imageHeight);
      drawOverlay(
        ctx, view, keypoints, imageWidth, imageHeight, front, side, silhouette,
        viewLabel ?? view,
      );
    };
  }, [
    imageUrl, imageWidth, imageHeight, keypoints, view, front, side,
    silhouette, viewLabel,
  ]);

  return (
    <div className="overflow-hidden rounded-card border border-border bg-black">
      <canvas
        ref={canvasRef}
        className="block h-auto w-full"
        style={{ aspectRatio: `${imageWidth} / ${imageHeight}` }}
      />
    </div>
  );
}

// ── Drawing helpers ────────────────────────────────────────────────
type ExtentKey = "ear" | "shoulder" | "hip" | "knee" | "ankle";

/** Horizontal span for one reference line: the body's own extent at
 *  that height plus a small overshoot, or the full frame width when
 *  the silhouette is absent (old reports, unusable mask). */
function extentSpan(
  silhouette: PostureSilhouette | undefined,
  key: ExtentKey,
  w: number,
): [number, number] {
  const ext = silhouette?.extents?.[key];
  if (!ext || ext.length < 2) return [0, w];
  const [xL, xR] = ext;
  const pad = Math.abs(xR - xL) * EXTENT_OVERSHOOT;
  return [xL - pad, xR + pad];
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  view: "front" | "side",
  kp: Keypoint[],
  w: number,
  h: number,
  front?: FrontMeasurements,
  side?: SideMeasurements,
  silhouette?: PostureSilhouette,
  label = "front",
) {
  const dotR = Math.max(4, Math.min(w, h) * 0.006);
  const lineW = Math.max(2, Math.min(w, h) * 0.003);

  if (view === "front") {
    drawFrontReferenceLines(ctx, kp, w, h, lineW, silhouette, label);
  } else {
    const pickedBlock = side?.pickedSide === "left"
      ? side.left
      : side?.pickedSide === "right"
        ? side.right
        : null;
    drawSideReferenceLines(
      ctx, kp, w, h, lineW, silhouette, label, side?.pickedSide ?? null,
      (pickedBlock as { shiftReference?: string } | null | undefined)
        ?.shiftReference,
    );
  }

  // Keypoint dots
  ctx.fillStyle = "#FACC15";
  ctx.strokeStyle = "#0F172A";
  ctx.lineWidth = 1.5;
  const dots = view === "front" ? FRONT_DOTS : SIDE_DOTS;
  for (const i of dots) {
    const p = kp[i];
    if (!p || (p.score ?? 0) < 0.2) continue;
    ctx.beginPath();
    ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Measurement badges
  if (view === "front" && front) {
    drawFrontBadges(ctx, kp, front, w, h);
  } else if (view === "side" && side) {
    drawSideBadges(ctx, kp, side, w, h);
  }
}

function drawFrontReferenceLines(
  ctx: CanvasRenderingContext2D,
  kp: Keypoint[],
  w: number,
  h: number,
  lineW: number,
  silhouette?: PostureSilhouette,
  label = "",
) {
  ctx.strokeStyle = "rgba(34, 197, 94, 0.85)";
  ctx.lineWidth = lineW;
  ctx.setLineDash([8, 6]);

  // Vertical plumb line. Preferred anchor is the silhouette's median
  // body centre; the shoulder midpoint is the fallback for reports
  // saved before the mask existed. The old anchor was never a
  // measurement axis — no front metric is taken against it — so this
  // moves a purely visual reference and changes no number.
  const ls = kp[LM.LEFT_SHOULDER];
  const rs = kp[LM.RIGHT_SHOULDER];
  const midlineX = silhouette?.midline_x ?? meanVisibleX(ls, rs);
  logVerticalSource(
    label,
    silhouette?.midline_x != null ? "silhouette" : "fallback",
  );
  if (midlineX !== null) {
    ctx.beginPath();
    ctx.moveTo(midlineX, 0);
    ctx.lineTo(midlineX, h);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // The measured body centre per row, drawn faintly against the plumb
  // above: where the two separate is the lateral deviation, visible
  // directly instead of inferred from the numbers.
  const centerline = silhouette?.centerline;
  if (centerline && centerline.length > 1) {
    ctx.strokeStyle = "rgba(34, 197, 94, 0.5)";
    ctx.lineWidth = Math.max(1, lineW * 0.5);
    ctx.beginPath();
    ctx.moveTo(centerline[0][0], centerline[0][1]);
    for (let i = 1; i < centerline.length; i++) {
      ctx.lineTo(centerline[i][0], centerline[i][1]);
    }
    ctx.stroke();
  }

  // Horizontal reference lines at ear / shoulder / hip / knee / ankle
  // midpoints. Spanned to the BODY at that height (plus a small
  // overshoot) rather than edge to edge, so each line reads as a
  // measurement of this patient rather than a grid over the photo.
  // Falls back to the full frame width when no extent is available.
  ctx.strokeStyle = "rgba(34, 197, 94, 0.6)";
  ctx.lineWidth = lineW * 0.7;
  const pairs: Array<[number, number, ExtentKey]> = [
    [LM.LEFT_EAR, LM.RIGHT_EAR, "ear"],
    [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, "shoulder"],
    [LM.LEFT_HIP, LM.RIGHT_HIP, "hip"],
    [LM.LEFT_KNEE, LM.RIGHT_KNEE, "knee"],
    [LM.LEFT_ANKLE, LM.RIGHT_ANKLE, "ankle"],
  ];
  for (const [a, b, key] of pairs) {
    const pa = kp[a];
    const pb = kp[b];
    // Both ends must clear the visibility floor: a placeholder at
    // (0, 0) would otherwise drag the line's height to half the real
    // joint height, the same failure as the side plumb.
    if (!usable(pa) || !usable(pb)) continue;
    const my = (pa.y + pb.y) / 2;
    const [x0, x1] = extentSpan(silhouette, key, w);
    ctx.beginPath();
    ctx.moveTo(x0, my);
    ctx.lineTo(x1, my);
    ctx.stroke();
  }

  // Actual joint-pair connecting lines (red) — untouched by the
  // silhouette work; these mark the real landmarks.
  ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
  ctx.lineWidth = lineW;
  for (const [a, b] of pairs) {
    const pa = kp[a];
    const pb = kp[b];
    if (!usable(pa) || !usable(pb)) continue;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
}

function drawSideReferenceLines(
  ctx: CanvasRenderingContext2D,
  kp: Keypoint[],
  w: number,
  h: number,
  lineW: number,
  silhouette?: PostureSilhouette,
  label = "",
  pickedSide: "left" | "right" | null = null,
  shiftReference?: string,
) {
  // Plumb line. Preferred anchor is the centre of the FOOT silhouette
  // at ankle height: in profile the ankle landmark sits inside the leg
  // while the foot extends forward, so a line on the landmark hangs
  // off the front of the foot.
  //
  // The BACKEND METRICS NOW USE THIS SAME RULE — silhouette.plumb_x
  // when the mask gave one, else the PICKED SIDE's own ankle landmark
  // (posture_engine._compute_one_side). The fallback here used to be
  // the MEAN of both ankles, so on a maskless photo the drawn line and
  // the numbers referenced different x values. Line and number must
  // share one reference.
  //
  // meanVisibleX stays as the last resort for a view with no
  // pickedSide (legacy saved data); it never averages in an occluded
  // placeholder.
  const pickedAnkle = pickedSide === "left"
    ? kp[LM.LEFT_ANKLE]
    : pickedSide === "right"
      ? kp[LM.RIGHT_ANKLE]
      : undefined;
  let plumbX = usable(pickedAnkle)
    ? pickedAnkle.x
    : meanVisibleX(kp[LM.LEFT_ANKLE], kp[LM.RIGHT_ANKLE]);
  const fromMask = silhouette?.plumb_x != null;
  if (fromMask) plumbX = silhouette!.plumb_x as number;
  logVerticalSource(
    label, fromMask ? "silhouette" : "fallback", shiftReference,
  );

  if (plumbX !== null) {
    ctx.strokeStyle = "rgba(34, 197, 94, 0.85)";
    ctx.lineWidth = lineW;
    ctx.setLineDash([10, 6]);
    ctx.beginPath();
    ctx.moveTo(plumbX, 0);
    ctx.lineTo(plumbX, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Body line ear → shoulder → hip → knee → ankle (red)
  const sequence = [
    [LM.LEFT_EAR, LM.LEFT_SHOULDER],
    [LM.LEFT_SHOULDER, LM.LEFT_HIP],
    [LM.LEFT_HIP, LM.LEFT_KNEE],
    [LM.LEFT_KNEE, LM.LEFT_ANKLE],
    [LM.RIGHT_EAR, LM.RIGHT_SHOULDER],
    [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
    [LM.RIGHT_HIP, LM.RIGHT_KNEE],
    [LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
  ] as const;

  ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
  ctx.lineWidth = lineW;
  for (const [a, b] of sequence) {
    const pa = kp[a];
    const pb = kp[b];
    if (!pa || !pb || (pa.score ?? 0) < 0.2 || (pb.score ?? 0) < 0.2) continue;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
}

function drawBadge(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  w: number,
) {
  const fontPx = Math.max(14, Math.min(w * 0.022, 22));
  ctx.font = `600 ${fontPx}px sans-serif`;
  const padX = fontPx * 0.5;
  const padY = fontPx * 0.3;
  const metrics = ctx.measureText(text);
  const bw = metrics.width + padX * 2;
  const bh = fontPx + padY * 2;
  ctx.fillStyle = "#EF4444";
  ctx.fillRect(x, y - bh, bw, bh);
  ctx.fillStyle = "#FFFFFF";
  ctx.fillText(text, x + padX, y - padY - 2);
}

function drawFrontBadges(
  ctx: CanvasRenderingContext2D,
  kp: Keypoint[],
  m: FrontMeasurements,
  w: number,
  _h: number,
) {
  if (m.shoulderTilt !== null) {
    const rs = kp[LM.RIGHT_SHOULDER];
    if (rs) drawBadge(ctx, `${Math.abs(m.shoulderTilt).toFixed(2)}°`, rs.x + 12, rs.y, w);
  }
  if (m.hipTilt !== null) {
    const rh = kp[LM.RIGHT_HIP];
    if (rh) drawBadge(ctx, `${Math.abs(m.hipTilt).toFixed(2)}°`, rh.x + 12, rh.y, w);
  }
  if (m.headTilt !== null) {
    const rE = kp[LM.RIGHT_EAR];
    if (rE) drawBadge(ctx, `${Math.abs(m.headTilt).toFixed(2)}°`, rE.x + 12, rE.y, w);
  }
  if (m.leftKneeAlignment !== null) {
    const lk = kp[LM.LEFT_KNEE];
    if (lk) {
      const dev = Math.abs(180 - m.leftKneeAlignment);
      drawBadge(ctx, `${dev.toFixed(2)}°`, lk.x - w * 0.18, lk.y, w);
    }
  }
  if (m.rightKneeAlignment !== null) {
    const rk = kp[LM.RIGHT_KNEE];
    if (rk) {
      const dev = Math.abs(180 - m.rightKneeAlignment);
      drawBadge(ctx, `${dev.toFixed(2)}°`, rk.x + 12, rk.y, w);
    }
  }
}

function drawSideBadges(
  ctx: CanvasRenderingContext2D,
  kp: Keypoint[],
  m: SideMeasurements,
  w: number,
  _h: number,
) {
  // After Fix 4 the metrics live under m.left / m.right. Use whichever
  // side has data, preferring `pickedSide` for the on-image overlay
  // (only one set of badges fits on a single photo).
  const block =
    (m.pickedSide === "left" ? m.left : null) ??
    (m.pickedSide === "right" ? m.right : null) ??
    m.left ?? m.right;
  if (!block) return;
  const overlaySide = m.pickedSide ?? (m.left ? "left" : "right");
  const idx = overlaySide === "left"
    ? { ear: LM.LEFT_EAR, sh: LM.LEFT_SHOULDER, hip: LM.LEFT_HIP, knee: LM.LEFT_KNEE }
    : { ear: LM.RIGHT_EAR, sh: LM.RIGHT_SHOULDER, hip: LM.RIGHT_HIP, knee: LM.RIGHT_KNEE };

  const drawAt = (label: string, kpIdx: number) => {
    const p = kp[kpIdx];
    if (!p) return;
    drawBadge(ctx, label, p.x + 12, p.y, w);
  };

  if (block.forwardHeadPct !== null)
    drawAt(`${block.forwardHeadPct.toFixed(2)}%`, idx.ear);
  if (block.shoulderShiftPct !== null)
    drawAt(`${block.shoulderShiftPct.toFixed(2)}%`, idx.sh);
  if (block.hipShiftPct !== null)
    drawAt(`${block.hipShiftPct.toFixed(2)}%`, idx.hip);
  if (block.kneeShiftPct !== null)
    drawAt(`${block.kneeShiftPct.toFixed(2)}%`, idx.knee);
}
