"use client";
// Single-image posture analyzer. Loads the image into an off-screen
// HTMLImageElement, runs the singleton MoveNet detector once, then
// hands the keypoints to the measurement layer. Same model used by the
// live + biomech pipelines, so behaviour is consistent.

import { getDetector } from "@/lib/pose/detector";
import {
  computeFrontMeasurements,
  computeSideMeasurements,
  type FrontMeasurements,
  type SideMeasurements,
} from "@/lib/posture/measurements";
import type { Keypoint } from "@tensorflow-models/pose-detection";

export interface PostureAnalysisResult {
  view: "front" | "side";
  imageUrl: string;     // ObjectURL — caller is responsible for revoking
  imageWidth: number;
  imageHeight: number;
  keypoints: Keypoint[];
  front?: FrontMeasurements;
  side?: SideMeasurements;
}

export async function analyzePostureImage(
  file: File,
  view: "front" | "side",
): Promise<PostureAnalysisResult> {
  const detector = await getDetector();
  const imageUrl = URL.createObjectURL(file);

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = imageUrl;
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () =>
      reject(new Error("Could not load image. Check the file format."));
  });

  const poses = await detector.estimatePoses(img, { flipHorizontal: false });
  const pose = poses[0];
  if (!pose) {
    URL.revokeObjectURL(imageUrl);
    throw new Error(
      "No person detected in the image. Make sure the full body is visible.",
    );
  }

  const result: PostureAnalysisResult = {
    view,
    imageUrl,
    imageWidth: img.naturalWidth,
    imageHeight: img.naturalHeight,
    keypoints: pose.keypoints,
  };

  if (view === "front") {
    result.front = computeFrontMeasurements(pose.keypoints);
  } else {
    result.side = computeSideMeasurements(pose.keypoints);
  }

  return result;
}
