"use client";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";

// Singleton MoveNet detector. We previously tried BlazePose-tfjs but it
// returned NaN for x/y in the WebGL backend on some machines (high
// score, valid name, NaN coords) — a known instability. MoveNet is the
// canonical tfjs pose model: rock-solid output, 17 keypoints (still
// covers all joints biomech needs), and Lightning variant easily hits
// 50+ FPS. Model load is expensive (~1 s first time) so we reuse across
// page lifetime.

let detectorPromise: Promise<poseDetection.PoseDetector> | null = null;
let backendUsed: string | null = null;

export function getBackend(): string | null {
  return backendUsed;
}

export function getDetector() {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      // Try WebGL first (fast on most machines), fall back to CPU.
      try {
        await tf.setBackend("webgl");
        backendUsed = "webgl";
      } catch {
        try {
          await tf.setBackend("cpu");
          backendUsed = "cpu";
        } catch {
          backendUsed = "unknown";
        }
      }
      await tf.ready();
      const det = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        },
      );
      return det;
    })();
  }
  return detectorPromise;
}
