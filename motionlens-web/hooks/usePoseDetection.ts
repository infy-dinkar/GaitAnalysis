"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type * as poseDetection from "@tensorflow-models/pose-detection";
import { getDetector } from "@/lib/pose/detector";

/**
 * Hook for browser-side pose detection (BlazePose Lite, TF.js + WebGL).
 * Runs locally — no HTTP round-trip per frame, so easily hits 30+ FPS
 * on modern hardware.
 */
export function usePoseDetection() {
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setReady(false);
    setError(null);
    getDetector()
      .then((d) => {
        if (!cancelled) {
          detectorRef.current = d;
          setReady(true);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e?.message ?? "Failed to load pose detector");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const detect = useCallback(
    async (input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement) => {
      const det = detectorRef.current;
      if (!det) return null;
      try {
        const poses = await det.estimatePoses(input, {
          flipHorizontal: false,
        });
        return poses[0] ?? null;
      } catch {
        return null;
      }
    },
    [],
  );

  return { ready, error, detect };
}
