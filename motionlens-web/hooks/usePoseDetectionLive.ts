"use client";
// React hook around the singleton MediaPipe BlazePose detector
// (lib/pose/detector-live.ts). The return shape is intentionally
// kept compatible with the legacy `usePoseDetection` hook used by
// LiveBiomechCamera — `{ ready, error, detect }` — so the consumer
// component swaps its imports via an `as usePoseDetection` alias
// without ANY other code change.
//
// DESIGN DEVIATION vs the brief's reference snippet: the brief
// described a Camera-driven hook returning `{ startCamera, stopCamera,
// getKeypoints }`. That shape is incompatible with the way
// LiveBiomechCamera drives the camera itself (via the existing
// `useCamera` hook + a manual rAF loop calling `await detect(video)`
// per frame). Implementing the brief's literal shape would have
// required rewriting LiveBiomechCamera, which the brief explicitly
// forbids ("Zero changes needed inside component body"). The
// underlying detector (lib/pose/detector-live.ts) is implemented
// exactly as the brief specifies; only this hook deviates.

import { useCallback, useEffect, useRef, useState } from "react";
import type { Results } from "@mediapipe/pose";
import { getDetectorLive, getPoseInstance } from "@/lib/pose/detector-live";

export type LiveKeypoint = {
  x: number;
  y: number;
  score: number;
};

export interface LivePose {
  keypoints: LiveKeypoint[];
}

// Module-level bridge state — there is exactly one MediaPipe Pose
// singleton, so onResults() is registered once globally. Each
// detect() call pins its resolver into this slot and waits for the
// next onResults to fire. A second detect() while one is in flight
// returns null (drops the frame), which the existing LiveBiomechCamera
// rAF loop already handles gracefully.
let pendingResolve: ((r: LivePose | null) => void) | null = null;
let lastInputDims: { w: number; h: number } = { w: 0, h: 0 };
let onResultsRegistered = false;

function registerOnResultsOnce(pose: Awaited<ReturnType<typeof getDetectorLive>>) {
  if (onResultsRegistered) return;
  onResultsRegistered = true;
  pose.onResults((results: Results) => {
    const resolve = pendingResolve;
    pendingResolve = null;
    if (!resolve) return;
    const lm = results.poseLandmarks;
    if (!lm || lm.length === 0) {
      resolve(null);
      return;
    }
    const { w, h } = lastInputDims;
    resolve({
      keypoints: lm.map((l) => ({
        x: l.x * w,
        y: l.y * h,
        score: l.visibility ?? 0,
      })),
    });
  });
}

export function usePoseDetectionLive() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Track per-hook lifecycle so unmount-during-load doesn't flip state
  // on a stale instance.
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;
    setReady(false);
    setError(null);
    getDetectorLive()
      .then((pose) => {
        if (cancelledRef.current) return;
        registerOnResultsOnce(pose);
        setReady(true);
      })
      .catch((e: unknown) => {
        if (cancelledRef.current) return;
        setError(
          e instanceof Error ? e.message : "Failed to load pose detector",
        );
      });
    return () => {
      cancelledRef.current = true;
    };
  }, []);

  const detect = useCallback(
    async (
      input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    ): Promise<LivePose | null> => {
      const pose = getPoseInstance();
      if (!pose) return null;
      // MediaPipe Pose processes one frame at a time. Drop frames
      // while a previous send is in flight — matches the existing
      // rAF loop's null-skipping behaviour.
      if (pendingResolve) return null;

      const w =
        (input as HTMLVideoElement).videoWidth ||
        (input as HTMLImageElement).naturalWidth ||
        (input as HTMLCanvasElement).width;
      const h =
        (input as HTMLVideoElement).videoHeight ||
        (input as HTMLImageElement).naturalHeight ||
        (input as HTMLCanvasElement).height;
      if (!w || !h) return null;
      lastInputDims = { w, h };

      return new Promise<LivePose | null>((resolve) => {
        pendingResolve = resolve;
        void pose.send({ image: input }).catch(() => {
          const r = pendingResolve;
          pendingResolve = null;
          if (r) r(null);
        });
      });
    },
    [],
  );

  return { ready, error, detect };
}
