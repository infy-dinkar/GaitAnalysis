"use client";
// Posture-analyzer adapter — posts the two captured photos to the
// FastAPI /api/analyze-posture endpoint and unwraps the response.
//
// Before this PR, the analyzer ran the singleton MoveNet detector
// in-browser on each photo. That gave only 17 keypoints and produced
// device-dependent readings (different GPUs, different TF.js builds,
// different results). The backend pipeline runs BlazePose Full
// IMAGE-mode + applies EXIF rotation correction (mobile portraits)
// + returns keypoints in the same MoveNet-indexed 17-element layout
// the saved-report overlay viewer reads. Phase A sign conventions
// (left-down hip tilt, separate L+R side blocks, frontal trunk lean
// with patient's-right-positive sign) and Phase B persistence
// (full keypoints + relative-units caveat) are preserved verbatim
// on the backend.
//
// MoveNet has been COMPLETELY REMOVED from the posture module —
// `getDetector` is no longer imported here.

import { authedFetch } from "@/lib/auth";
import type {
  FrontMeasurements,
  PostureFinding,
  SideMeasurements,
} from "@/lib/posture/measurements";

// MoveNet-compatible keypoint shape — same {x, y, score, name}
// fields the browser detector produced, so PostureImageOverlay and
// SavedPostureReport keep reading keypoints by index unchanged.
// Backend always emits all 17 entries (MoveNet contract); low-
// visibility / undetected landmarks come back with score=0 and
// placeholder coordinates, so downstream visibility checks (score
// >= 0.2) skip them naturally without needing nullish guards.
export interface PostureKeypoint {
  x: number;
  y: number;
  score: number;
  name: string;
}

export interface PostureAnalysisResult {
  view: "front" | "side";
  imageUrl: string;       // ObjectURL — caller is responsible for revoking
  imageWidth: number;
  imageHeight: number;
  keypoints: PostureKeypoint[];
  front?: FrontMeasurements;
  side?: SideMeasurements;
  /** Server-computed findings (matches buildFrontFindings /
   *  buildSideFindings in measurements.ts). Optional so older
   *  callers that re-grade locally still work. */
  findings?: PostureFinding[];
}

// ─── Additive multi-view types (4-view expansion) ─────────────
// All fields OPTIONAL so old saved reports typecheck as-is.

/** Back-view measurements — the honest subset a single flat 2D
 *  back-view can compute. Everything else lives on the sibling
 *  `not_assessed` array with a reason. */
export interface BackMeasurements {
  shoulderTilt: number | null;         // deg
  hipTilt: number | null;              // deg
  lateralTrunkShiftPct: number | null; // % of body height
  leftKneeAlignment: number | null;    // interior angle
  rightKneeAlignment: number | null;
}

/** One item on the back view's not_assessed list. */
export interface PostureNotAssessed {
  label: string;
  reason: string;
}

/** Back-view analysis result. `view` is "back". */
export interface PostureBackResult {
  view: "back";
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  keypoints: PostureKeypoint[];
  back?: BackMeasurements;
  not_assessed?: PostureNotAssessed[];
  findings?: PostureFinding[];
  /** True when the L/R keypoint swap was applied. Honest debug
   *  affordance — see engines/posture_engine_multi_view.py. */
  lr_swap_applied?: boolean;
}

/** Explicit-side (left_side / right_side) analysis result. Uses the
 *  same SideMeasurements shape as the auto-picked `side` view, but
 *  `pickedSide` is FORCED to the declared side. */
export interface PostureExplicitSideResult {
  view: "left_side" | "right_side";
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  keypoints: PostureKeypoint[];
  side?: SideMeasurements;
  findings?: PostureFinding[];
  explicit_side?: "left" | "right";
}

/** Union of every posture view result the multi-view API returns. */
export type PostureViewResult =
  | PostureAnalysisResult
  | PostureBackResult
  | PostureExplicitSideResult;

/** Optional-input shape for the multi-view analyzer. `front` +
 *  `side` remain REQUIRED to preserve the existing endpoint
 *  contract; the three new views are OPTIONAL. */
export interface PostureMultiViewInput {
  frontFile: File;
  sideFile: File;
  backFile?: File;
  leftSideFile?: File;
  rightSideFile?: File;
}

/** Multi-view analyzer result. Existing `front` + `side` keys are
 *  the SAME shape as `analyzePostureCombined` returns today. New
 *  view keys are populated only when their file was provided AND
 *  the analysis succeeded. Per-view failures land as
 *    { view: "<key>", error: "<code>" }
 *  so one bad view doesn't break the rest. */
export interface PostureMultiViewResult {
  front: PostureAnalysisResult;
  side: PostureAnalysisResult;
  back?: PostureBackResult | PostureViewError;
  left_side?: PostureExplicitSideResult | PostureViewError;
  right_side?: PostureExplicitSideResult | PostureViewError;
  relativeUnits: boolean;
}

/** Per-view failure shape. Present under a view key when that
 *  view's analysis failed but other views succeeded. */
export interface PostureViewError {
  view: string;
  error: string;
}

/** Type guard — narrow a view union to "error". */
export function isPostureViewError(
  v: unknown,
): v is PostureViewError {
  return (
    !!v && typeof v === "object"
    && typeof (v as { error?: unknown }).error === "string"
  );
}

/** Combined two-photo posture analysis. Single HTTP POST to the
 *  backend with both photos in one multipart body. Returns the
 *  TWO PostureAnalysisResult objects (front + side) so the
 *  PostureCapture component can plug them straight into the
 *  existing PostureReport without any shape remapping. */
export async function analyzePostureCombined(
  frontFile: File,
  sideFile: File,
): Promise<{
  front: PostureAnalysisResult;
  side: PostureAnalysisResult;
  /** Phase B: % shifts are relative to body height, not cm.
   *  Surfaces as the RelativeUnitsCaveat banner in the report. */
  relativeUnits: boolean;
}> {
  const form = new FormData();
  form.append("front_image", frontFile, frontFile.name || "posture_front.jpg");
  form.append("side_image",  sideFile,  sideFile.name  || "posture_side.jpg");

  const res = await authedFetch("/api/analyze-posture", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(formatPostureError(body.detail, res.status));
  }
  const wrapper = (await res.json()) as {
    success: boolean;
    data: {
      front: {
        view: "front";
        imageWidth: number;
        imageHeight: number;
        keypoints: PostureKeypoint[];
        front?: FrontMeasurements;
        findings: PostureFinding[];
      };
      side: {
        view: "side";
        imageWidth: number;
        imageHeight: number;
        keypoints: PostureKeypoint[];
        side?: SideMeasurements;
        findings: PostureFinding[];
      };
      relative_units: boolean;
    } | null;
    error: string | null;
  };
  if (!wrapper.success || !wrapper.data) {
    throw new Error(
      wrapper.error ?? "Posture analysis failed. Please try again.",
    );
  }
  const data = wrapper.data;

  // ObjectURLs are created here so the report's annotated overlay
  // has a stable image source even after the original File is out
  // of scope. Caller is responsible for revoking them on reset.
  const frontUrl = URL.createObjectURL(frontFile);
  const sideUrl  = URL.createObjectURL(sideFile);

  return {
    front: {
      view: "front",
      imageUrl: frontUrl,
      imageWidth:  data.front.imageWidth,
      imageHeight: data.front.imageHeight,
      keypoints:   data.front.keypoints,
      front:       data.front.front,
      findings:    data.front.findings,
    },
    side: {
      view: "side",
      imageUrl: sideUrl,
      imageWidth:  data.side.imageWidth,
      imageHeight: data.side.imageHeight,
      keypoints:   data.side.keypoints,
      side:        data.side.side,
      findings:    data.side.findings,
    },
    relativeUnits: !!data.relative_units,
  };
}

// Map the backend's structured error tokens to user-facing strings.
// Anything not in the token list falls back to the raw detail string
// so debugging never loses information.
function formatPostureError(detail: unknown, status: number): string {
  const raw = typeof detail === "string" ? detail : Array.isArray(detail)
    ? detail.map((d) => (typeof d === "string" ? d : JSON.stringify(d))).join("; ")
    : JSON.stringify(detail ?? {});

  if (raw.startsWith("file_too_large")) {
    return "Photo too large. Maximum 10 MB per photo.";
  }
  if (raw.startsWith("invalid_image")) {
    return "Invalid image. Please retake the photo.";
  }
  if (raw.startsWith("poor_visibility")) {
    return (
      "Person not clearly visible. Please retake with the full body in " +
      "frame and good lighting."
    );
  }
  if (status >= 500) {
    return "Analysis failed. Please check your connection and try again.";
  }
  return raw || `Analysis failed (HTTP ${status}).`;
}

// ─── Additive: multi-view analyzer (4 views) ───────────────────
// Existing `analyzePostureCombined` above stays byte-identical for
// back-compat. This new function is the way callers request the new
// back / left_side / right_side views. It POSTs to the SAME endpoint
// with the additional optional file fields the api.py handler now
// accepts.
export async function analyzePostureMultiView(
  input: PostureMultiViewInput,
): Promise<PostureMultiViewResult> {
  const {
    frontFile, sideFile, backFile, leftSideFile, rightSideFile,
  } = input;

  const form = new FormData();
  form.append(
    "front_image", frontFile, frontFile.name || "posture_front.jpg",
  );
  form.append(
    "side_image", sideFile, sideFile.name || "posture_side.jpg",
  );
  if (backFile) {
    form.append(
      "back_image", backFile, backFile.name || "posture_back.jpg",
    );
  }
  if (leftSideFile) {
    form.append(
      "left_side_image", leftSideFile,
      leftSideFile.name || "posture_left_side.jpg",
    );
  }
  if (rightSideFile) {
    form.append(
      "right_side_image", rightSideFile,
      rightSideFile.name || "posture_right_side.jpg",
    );
  }

  const res = await authedFetch("/api/analyze-posture", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(formatPostureError(body.detail, res.status));
  }
  const wrapper = (await res.json()) as {
    success: boolean;
    data: {
      front: {
        view: "front";
        imageWidth: number;
        imageHeight: number;
        keypoints: PostureKeypoint[];
        front?: FrontMeasurements;
        findings: PostureFinding[];
      };
      side: {
        view: "side";
        imageWidth: number;
        imageHeight: number;
        keypoints: PostureKeypoint[];
        side?: SideMeasurements;
        findings: PostureFinding[];
      };
      back?: {
        view: "back";
        imageWidth: number;
        imageHeight: number;
        keypoints: PostureKeypoint[];
        back?: BackMeasurements;
        not_assessed?: PostureNotAssessed[];
        findings: PostureFinding[];
        lr_swap_applied?: boolean;
      } | PostureViewError;
      left_side?: {
        view: "left_side";
        imageWidth: number;
        imageHeight: number;
        keypoints: PostureKeypoint[];
        side?: SideMeasurements;
        findings: PostureFinding[];
        explicit_side?: "left";
      } | PostureViewError;
      right_side?: {
        view: "right_side";
        imageWidth: number;
        imageHeight: number;
        keypoints: PostureKeypoint[];
        side?: SideMeasurements;
        findings: PostureFinding[];
        explicit_side?: "right";
      } | PostureViewError;
      relative_units: boolean;
    } | null;
    error: string | null;
  };
  if (!wrapper.success || !wrapper.data) {
    throw new Error(
      wrapper.error ?? "Posture analysis failed. Please try again.",
    );
  }
  const data = wrapper.data;

  const frontUrl = URL.createObjectURL(frontFile);
  const sideUrl  = URL.createObjectURL(sideFile);
  const backUrl = backFile ? URL.createObjectURL(backFile) : null;
  const leftUrl = leftSideFile ? URL.createObjectURL(leftSideFile) : null;
  const rightUrl = rightSideFile ? URL.createObjectURL(rightSideFile) : null;

  const out: PostureMultiViewResult = {
    front: {
      view: "front",
      imageUrl: frontUrl,
      imageWidth: data.front.imageWidth,
      imageHeight: data.front.imageHeight,
      keypoints: data.front.keypoints,
      front: data.front.front,
      findings: data.front.findings,
    },
    side: {
      view: "side",
      imageUrl: sideUrl,
      imageWidth: data.side.imageWidth,
      imageHeight: data.side.imageHeight,
      keypoints: data.side.keypoints,
      side: data.side.side,
      findings: data.side.findings,
    },
    relativeUnits: !!data.relative_units,
  };

  if (data.back && backUrl) {
    if (isPostureViewError(data.back)) {
      out.back = data.back;
    } else {
      out.back = {
        view: "back",
        imageUrl: backUrl,
        imageWidth: data.back.imageWidth,
        imageHeight: data.back.imageHeight,
        keypoints: data.back.keypoints,
        back: data.back.back,
        not_assessed: data.back.not_assessed,
        findings: data.back.findings,
        lr_swap_applied: data.back.lr_swap_applied,
      };
    }
  }
  if (data.left_side && leftUrl) {
    if (isPostureViewError(data.left_side)) {
      out.left_side = data.left_side;
    } else {
      out.left_side = {
        view: "left_side",
        imageUrl: leftUrl,
        imageWidth: data.left_side.imageWidth,
        imageHeight: data.left_side.imageHeight,
        keypoints: data.left_side.keypoints,
        side: data.left_side.side,
        findings: data.left_side.findings,
        explicit_side: data.left_side.explicit_side,
      };
    }
  }
  if (data.right_side && rightUrl) {
    if (isPostureViewError(data.right_side)) {
      out.right_side = data.right_side;
    } else {
      out.right_side = {
        view: "right_side",
        imageUrl: rightUrl,
        imageWidth: data.right_side.imageWidth,
        imageHeight: data.right_side.imageHeight,
        keypoints: data.right_side.keypoints,
        side: data.right_side.side,
        findings: data.right_side.findings,
        explicit_side: data.right_side.explicit_side,
      };
    }
  }
  return out;
}
