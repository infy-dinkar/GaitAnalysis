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
//
// CDN failure handling: the default pose-detection package URL points
// at tfhub.dev which redirects through Kaggle. Both have intermittent
// outages and some networks/ad-blockers block them. We try multiple
// sources and ONLY cache the singleton AFTER one succeeds — so a
// transient fetch failure does not permanently brick every
// browser-side test in the session.

let detectorPromise: Promise<poseDetection.PoseDetector> | null = null;
let backendUsed: string | null = null;

export function getBackend(): string | null {
  return backendUsed;
}

// Ordered list of model sources. First is the package default
// (tfhub.dev → Kaggle); second is the direct Google Cloud Storage
// bucket, which is the underlying host with no redirect chain and
// no Kaggle dependency. If the first fails (e.g. ad-blocker or
// Kaggle outage) the GCS URL almost always succeeds.
const MODEL_URL_FALLBACKS: Array<string | undefined> = [
  undefined,   // package default
  "https://storage.googleapis.com/tfjs-models/savedmodel/movenet/singlepose/lightning/4/model.json",
];

async function createMoveNet(): Promise<poseDetection.PoseDetector> {
  // Backend: WebGL first (fast), fall back to CPU.
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

  let lastError: unknown = null;
  for (const modelUrl of MODEL_URL_FALLBACKS) {
    try {
      return await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
          ...(modelUrl ? { modelUrl } : {}),
        },
      );
    } catch (e) {
      lastError = e;
      // Fall through to the next URL.
    }
  }
  // Surface the last failure with a clearer message — the user gets
  // "Pose model failed: <our text>" rather than the bare "Failed to
  // fetch" that hides which dependency died.
  throw new Error(
    `Could not load MoveNet from any source. Last error: ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }. Check your network / ad-blocker — the model is fetched from Google CDN at runtime.`,
  );
}

export function getDetector() {
  if (!detectorPromise) {
    const p = createMoveNet();
    // Only cache the promise once it resolves. If it rejects, clear
    // the cache so the NEXT call to getDetector() re-attempts the
    // load instead of returning the same dead promise forever.
    detectorPromise = p;
    p.catch(() => {
      if (detectorPromise === p) detectorPromise = null;
    });
  }
  return detectorPromise;
}
