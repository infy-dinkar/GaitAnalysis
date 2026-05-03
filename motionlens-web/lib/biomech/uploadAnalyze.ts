"use client";
// Client-side biomech video analysis — runs the same MoveNet detector
// the live mode uses, but iterates frames from an uploaded file instead
// of a webcam stream. Produces the same BiomechDataDTO shape the server
// used to return, so the existing AssessmentReport renders unchanged.
//
// Strategy:
//   • Load the file into an off-screen <video> element via ObjectURL.
//   • Use requestVideoFrameCallback (Chrome/Edge/Safari 15+) to process
//     every rendered frame at high playback rate (4×), falling back to
//     manual seek-by-time on browsers without rVFC.
//   • Feed each frame to the singleton MoveNet detector.
//   • Compute the per-frame angle using the same shoulder/neck math as
//     the live tier, so the two pipelines are byte-consistent.

import { getDetector } from "@/lib/pose/detector";
import {
  computeShoulderAngle,
  SHOULDER_MOVEMENTS,
  type ShoulderMovementId,
} from "@/lib/biomech/shoulder";
import {
  computeNeckAngle,
  NECK_MOVEMENTS,
  type NeckMovementId,
} from "@/lib/biomech/neck";
import {
  computeKneeAngle,
  KNEE_MOVEMENTS,
  type KneeMovementId,
} from "@/lib/biomech/knee";
import {
  computeHipAngle,
  HIP_MOVEMENTS,
  type HipMovementId,
} from "@/lib/biomech/hip";
import {
  computeAnkleAngle,
  ANKLE_MOVEMENTS,
  type AnkleMovementId,
} from "@/lib/biomech/ankle";
import type { BiomechDataDTO } from "@/lib/api";

type BodyPart = "shoulder" | "neck" | "knee" | "hip" | "ankle";

interface AnalyzeOpts {
  file: File;
  bodyPart: BodyPart;
  movement: string;
  side?: "left" | "right";
  /** Receives playback progress 0..1. */
  onProgress?: (fraction: number) => void;
}

const SAMPLE_FPS_FALLBACK = 15; // seek-mode sample rate (10-sec video → 150 samples)

export async function analyzeBiomechVideo(
  opts: AnalyzeOpts,
): Promise<BiomechDataDTO> {
  const { file, bodyPart, movement, side, onProgress } = opts;
  const detector = await getDetector();

  // ── Set up off-screen video element ───────────────────────────────
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = "anonymous";

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () =>
      reject(new Error("Could not load video. Format may not be supported."));
  });

  // Force first frame to be ready for detection.
  video.currentTime = 0;
  await new Promise<void>((resolve) => {
    video.onseeked = () => resolve();
  });

  const duration = video.duration;
  if (!isFinite(duration) || duration <= 0) {
    URL.revokeObjectURL(url);
    throw new Error("Video duration is invalid.");
  }

  const angles: number[] = [];
  let validFrames = 0;
  let totalFrames = 0;
  let measuredFps = 30;

  // Helper to run one detection on the currently-displayed frame.
  const sideOrRight = side ?? "right";
  const detectCurrentFrame = async (): Promise<number | null> => {
    try {
      const poses = await detector.estimatePoses(video, {
        flipHorizontal: false,
      });
      const pose = poses[0];
      if (!pose) return null;
      switch (bodyPart) {
        case "shoulder":
          return computeShoulderAngle(
            movement as ShoulderMovementId,
            pose.keypoints,
            sideOrRight,
          );
        case "neck":
          return computeNeckAngle(
            movement as NeckMovementId,
            pose.keypoints,
          );
        case "knee":
          return computeKneeAngle(
            movement as KneeMovementId,
            pose.keypoints,
            sideOrRight,
          );
        case "hip":
          return computeHipAngle(
            movement as HipMovementId,
            pose.keypoints,
            sideOrRight,
          );
        case "ankle":
          return computeAnkleAngle(
            movement as AnkleMovementId,
            pose.keypoints,
            sideOrRight,
          );
      }
    } catch {
      return null;
    }
  };

  const supportsRVFC = typeof video.requestVideoFrameCallback === "function";

  if (supportsRVFC) {
    // ── Fast path: play at 4× and grab every rendered frame via rVFC ──
    video.playbackRate = 4;
    let firstMediaTime: number | null = null;
    let lastMediaTime = 0;

    await new Promise<void>((resolve, reject) => {
      const onFrame: VideoFrameRequestCallback = async (_now, metadata) => {
        if (firstMediaTime === null) firstMediaTime = metadata.mediaTime;
        lastMediaTime = metadata.mediaTime;

        const angle = await detectCurrentFrame();
        if (angle !== null && !isNaN(angle)) {
          angles.push(angle);
          validFrames += 1;
        } else {
          angles.push(NaN);
        }
        totalFrames += 1;
        onProgress?.(Math.min(1, metadata.mediaTime / duration));

        if (video.ended || metadata.mediaTime >= duration - 0.05) {
          // Estimate effective fps from frames captured during playback.
          const span = lastMediaTime - (firstMediaTime ?? 0);
          if (span > 0) measuredFps = totalFrames / span;
          resolve();
          return;
        }
        // Schedule next frame
        video.requestVideoFrameCallback(onFrame);
      };

      video.requestVideoFrameCallback(onFrame);
      video.play().catch(reject);
    });
    video.pause();
  } else {
    // ── Fallback: seek-by-time at fixed sample rate ───────────────────
    measuredFps = SAMPLE_FPS_FALLBACK;
    totalFrames = Math.max(1, Math.floor(duration * SAMPLE_FPS_FALLBACK));
    for (let i = 0; i < totalFrames; i++) {
      const t = i / SAMPLE_FPS_FALLBACK;
      video.currentTime = t;
      await new Promise<void>((resolve) => {
        video.onseeked = () => resolve();
      });

      const angle = await detectCurrentFrame();
      if (angle !== null && !isNaN(angle)) {
        angles.push(angle);
        validFrames += 1;
      } else {
        angles.push(NaN);
      }
      onProgress?.((i + 1) / totalFrames);

      // Yield to the event loop every few frames so the UI stays responsive.
      if (i % 4 === 0) {
        await new Promise<void>((r) => setTimeout(r, 0));
      }
    }
  }

  URL.revokeObjectURL(url);

  // ── Find peak (max absolute value, signed for fidelity) ────────────
  let peakAngle = 0;
  let peakMag = 0;
  for (const a of angles) {
    if (!isNaN(a) && Math.abs(a) > peakMag) {
      peakMag = Math.abs(a);
      peakAngle = a;
    }
  }

  // ── Reference range from movement metadata ─────────────────────────
  const refRange = getTargetRange(bodyPart, movement);
  const target = refRange[1];
  const percentage = target > 0 ? (peakMag / target) * 100 : 0;

  // ── Status classification (mirrors AssessmentReport.classify) ──────
  let status: "good" | "fair" | "poor";
  if (percentage >= 90) status = "good";
  else if (percentage >= 75) status = "fair";
  else status = "poor";

  return {
    body_part: bodyPart,
    movement,
    side: side ?? null,
    peak_angle: validFrames > 0 ? peakAngle : null,
    peak_magnitude: peakMag,
    reference_range: refRange,
    target,
    percentage,
    status,
    valid_frames: validFrames,
    total_frames: totalFrames,
    fps: measuredFps,
    interpretation: buildInterpretation(
      peakMag,
      refRange,
      status,
      bodyPart,
      movement,
      side,
    ),
  };
}

function getTargetRange(
  bodyPart: BodyPart,
  movement: string,
): [number, number] {
  let list: ReadonlyArray<{ id: string; target: [number, number] }> = [];
  switch (bodyPart) {
    case "shoulder": list = SHOULDER_MOVEMENTS; break;
    case "neck":     list = NECK_MOVEMENTS; break;
    case "knee":     list = KNEE_MOVEMENTS; break;
    case "hip":      list = HIP_MOVEMENTS; break;
    case "ankle":    list = ANKLE_MOVEMENTS; break;
  }
  const m = list.find((x) => x.id === movement);
  return m?.target ?? [0, 180];
}

function buildInterpretation(
  peakMag: number,
  range: [number, number],
  status: "good" | "fair" | "poor",
  bodyPart: BodyPart,
  movement: string,
  side?: "left" | "right",
): string {
  const sideText = side ? ` ${side}` : "";
  const movementText = movement.replace(/_/g, " ");
  const peakText = `${peakMag.toFixed(1)}°`;
  const rangeText = `${range[0]}°–${range[1]}°`;
  if (status === "good") {
    return `Peak${sideText} ${bodyPart} ${movementText} of ${peakText} is within the normal reference range of ${rangeText}.`;
  }
  if (status === "fair") {
    return `Peak${sideText} ${bodyPart} ${movementText} of ${peakText} is approaching the normal reference range of ${rangeText} but not within it.`;
  }
  return `Peak${sideText} ${bodyPart} ${movementText} of ${peakText} is below the expected range of ${rangeText}, which may indicate restricted mobility.`;
}
