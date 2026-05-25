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
// Backend emits null for landmarks below the visibility floor so
// downstream code can skip them with a single nullish check.
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
  keypoints: (PostureKeypoint | null)[];
  front?: FrontMeasurements;
  side?: SideMeasurements;
  /** Server-computed findings (matches buildFrontFindings /
   *  buildSideFindings in measurements.ts). Optional so older
   *  callers that re-grade locally still work. */
  findings?: PostureFinding[];
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
        keypoints: (PostureKeypoint | null)[];
        front?: FrontMeasurements;
        findings: PostureFinding[];
      };
      side: {
        view: "side";
        imageWidth: number;
        imageHeight: number;
        keypoints: (PostureKeypoint | null)[];
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
