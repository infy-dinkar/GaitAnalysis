"use client";
// Singleton MediaPipe BlazePose Full detector for the LIVE camera
// stack. Completely independent from the upload-mode detector
// (lib/pose/detector.ts) which still runs MoveNet via TF.js.
//
// Why a separate file: lib/pose/detector.ts is shared by the upload
// pipeline (lib/biomech/uploadAnalyze.ts calls getDetector() at 4
// sites). Touching it would change MoveNet behaviour for upload mode.
// This file is read ONLY by the *-live.ts math files +
// hooks/usePoseDetectionLive.ts.
//
// Runtime loading note: @mediapipe/pose ships a UMD bundle
// (`pose.js`) that registers `window.Pose` instead of providing ESM
// exports. Turbopack / webpack therefore cannot resolve
// `import { Pose } from "@mediapipe/pose"` as a runtime value (the
// build error is "The module has no exports at all"). The package
// ships TypeScript types in `index.d.ts` though, so we keep a
// type-only import for compile-time and load the actual class via
// a `<script>` tag from CDN at runtime. This is the standard pattern
// every Next.js / webpack project that uses MediaPipe Pose follows.

import type { Pose, PoseConfig, ResultsListener, Options } from "@mediapipe/pose";

// Pin to the version we declared in package.json so the WASM, model
// files, and JS wrapper all come from a matching release.
const POSE_VERSION = "0.5.1675469404";
const POSE_SCRIPT_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/pose.js`;

// Minimal interface for the global ctor we read off of `window` after
// the CDN script loads. Mirrors the relevant subset of
// `@mediapipe/pose`'s exported class shape so we get type-safe calls
// to setOptions / onResults / send / initialize.
interface PoseLikeCtor {
  new (config?: PoseConfig): {
    setOptions(options: Options): void;
    onResults(listener: ResultsListener): void;
    initialize(): Promise<void>;
    send(inputs: { image: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement }, at?: number): Promise<void>;
    close(): Promise<void>;
    reset(): void;
  };
}

let poseInstance: Pose | null = null;
let readyPromise: Promise<Pose> | null = null;
let scriptLoadPromise: Promise<void> | null = null;

function loadPoseScript(): Promise<void> {
  if (scriptLoadPromise) return scriptLoadPromise;
  scriptLoadPromise = new Promise<void>((resolve, reject) => {
    if (typeof window === "undefined") {
      reject(new Error("Pose detector cannot load on the server"));
      return;
    }
    // Already loaded? (HMR, multiple consumers, etc.)
    if ((window as unknown as { Pose?: PoseLikeCtor }).Pose) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = POSE_SCRIPT_URL;
    script.async = true;
    script.crossOrigin = "anonymous";
    script.onload = () => resolve();
    script.onerror = () => {
      // Clear so a later getDetectorLive() retry can attempt the load
      // again instead of returning the same dead promise forever.
      scriptLoadPromise = null;
      reject(new Error(
        `Could not load @mediapipe/pose script from ${POSE_SCRIPT_URL}. ` +
        `Check network / ad-blocker — the MediaPipe runtime is fetched ` +
        `from a public CDN at runtime.`,
      ));
    };
    document.head.appendChild(script);
  });
  return scriptLoadPromise;
}

export async function getDetectorLive(): Promise<Pose> {
  if (readyPromise) return readyPromise;
  readyPromise = (async () => {
    await loadPoseScript();
    const PoseCtor = (window as unknown as { Pose?: PoseLikeCtor }).Pose;
    if (!PoseCtor) {
      throw new Error("@mediapipe/pose loaded but window.Pose is missing");
    }
    const pose = new PoseCtor({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/${file}`,
    }) as unknown as Pose;
    pose.setOptions({
      modelComplexity: 1,           // "Full" — matches backend model
      // Built-in temporal smoother ON: keeps the skeleton steady
      // (no shake). It adds a small trailing lag, which is the
      // accepted trade for a stable overlay.
      smoothLandmarks: true,
      enableSegmentation: false,    // expensive; clinical tests don't use it
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    await pose.initialize();
    poseInstance = pose;
    return pose;
  })().catch((e) => {
    // Clear so the NEXT call retries the load instead of returning
    // the same dead promise forever.
    readyPromise = null;
    throw e;
  });
  return readyPromise;
}

export function getPoseInstance(): Pose | null {
  return poseInstance;
}
