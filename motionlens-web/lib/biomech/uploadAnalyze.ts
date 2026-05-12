"use client";
// Client-side biomech video analysis — runs the same MoveNet detector
// the live mode uses, but iterates frames from an uploaded file instead
// of a webcam stream. Produces the same BiomechDataDTO shape the server
// used to return, so the existing AssessmentReport renders unchanged.
//
// EXCEPTION — ankle:
//   Ankle dorsi/plantarflexion needs foot_index landmarks (MediaPipe
//   kp 31/32) that MoveNet's 17-keypoint set does not provide. The
//   browser-side pipeline therefore can't compute the ankle joint
//   angle accurately — for seated ankle ROM it just reports the
//   shin-from-vertical (~90° regardless of foot motion). Ankle
//   uploads are routed to a backend MediaPipe endpoint instead.
//
// Strategy (non-ankle):
//   • Load the file into an off-screen <video> element via ObjectURL.
//   • Use requestVideoFrameCallback (Chrome/Edge/Safari 15+) to process
//     every rendered frame at high playback rate (4×), falling back to
//     manual seek-by-time on browsers without rVFC.
//   • Feed each frame to the singleton MoveNet detector.
//   • Compute the per-frame angle using the same shoulder/neck math as
//     the live tier, so the two pipelines are byte-consistent.

import { authedFetch } from "@/lib/auth";
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
// Ankle types/math are NOT imported here — ankle uploads are dispatched
// to the backend MediaPipe endpoint via analyzeAnkleBackend() at the
// top of analyzeBiomechVideo. ANKLE_MOVEMENTS is still referenced in
// getTargetRange below for defensive parity with the other body parts,
// imported directly there.
import { ANKLE_MOVEMENTS } from "@/lib/biomech/ankle";
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

  // ── Ankle → backend MediaPipe (BlazePose Full, 33 kp with feet) ──
  // MoveNet's 17-keypoint set doesn't reach the foot, so the
  // browser-side path can't measure the actual ankle joint angle —
  // for seated ankle ROM it just returns shin-from-vertical (~90°
  // regardless of foot motion). The backend pipeline reuses gait's
  // MediaPipe setup + new shin/foot-vector math in ankle_engine.py.
  if (bodyPart === "ankle") {
    return analyzeAnkleBackend(file, movement, side ?? "right", onProgress);
  }

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
        // NOTE: "ankle" is intentionally absent here — the early
        // return at the top of analyzeBiomechVideo dispatches ankle
        // requests to the backend MediaPipe pipeline. TypeScript
        // therefore narrows `bodyPart` to non-ankle inside this
        // switch.
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
      let resolved = false;
      let stallTimer: ReturnType<typeof setTimeout> | null = null;
      const STALL_MS = 1500; // if no new rVFC for 1.5 s, treat as ended

      const finish = () => {
        if (resolved) return;
        resolved = true;
        if (stallTimer) clearTimeout(stallTimer);
        const span = lastMediaTime - (firstMediaTime ?? 0);
        if (span > 0) measuredFps = totalFrames / span;
        resolve();
      };

      const armStallTimer = () => {
        if (stallTimer) clearTimeout(stallTimer);
        stallTimer = setTimeout(() => {
          // rVFC hasn't fired in STALL_MS — assume end of stream.
          // Common on iOS Safari + some Android codec combos where
          // the final frame's mediaTime never reaches duration-0.05.
          finish();
        }, STALL_MS);
      };

      const onFrame: VideoFrameRequestCallback = async (_now, metadata) => {
        if (resolved) return;
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

        // Three end-conditions, any of which finalises:
        //   1. <video> ended event already fired
        //   2. mediaTime is within 2% of duration (loose tolerance —
        //      the original duration-0.05 was too tight; many mobile
        //      WebM/MP4 files stop at 99% of duration and stall)
        //   3. stall timer fires (no new frame for STALL_MS)
        const nearEnd =
          metadata.mediaTime >= Math.max(duration - 0.1, duration * 0.98);
        if (video.ended || nearEnd) {
          finish();
          return;
        }

        // Re-arm stall timer for the next frame.
        armStallTimer();
        video.requestVideoFrameCallback(onFrame);
      };

      // Also listen for the native `ended` event as a belt-and-braces
      // — some browsers fire it without producing a final rVFC.
      video.addEventListener("ended", finish, { once: true });

      armStallTimer();
      video.requestVideoFrameCallback(onFrame);
      video.play().catch((e) => {
        if (!resolved) reject(e);
      });
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

// ─── Ankle backend dispatch ─────────────────────────────────────
// Uploads the video to /api/analyze-ankle (MediaPipe BlazePose Full)
// and returns the same BiomechDataDTO shape the in-browser path
// produces, so the existing AssessmentReport renders unchanged.
//
// onProgress is faked with 3 coarse milestones (10% upload start,
// 50% during analysis, 100% on result) because the backend is a
// single request — no per-frame events stream back. The user
// already sees an "analysing on backend" status; the spinner is
// enough.
async function analyzeAnkleBackend(
  file: File,
  movement: string,
  side: "left" | "right",
  onProgress?: (fraction: number) => void,
): Promise<BiomechDataDTO> {
  return analyzeAnkleBlob(file, movement, side, null, onProgress);
}

// Blob-accepting variant used by the live-record path. MediaRecorder
// produces a Blob, not a File, and its WebM container often has a
// missing duration header — `recordingDurationMs` lets the backend
// repair the file via tug_engine._ensure_decodable_video.
export async function analyzeAnkleBlob(
  blob: Blob,
  movement: string,
  side: "left" | "right",
  recordingDurationMs: number | null,
  onProgress?: (fraction: number) => void,
): Promise<BiomechDataDTO> {
  onProgress?.(0.1);
  const form = new FormData();
  const filename = blob instanceof File ? blob.name : "ankle.webm";
  form.append("video", blob, filename);
  form.append("movement_type", movement);
  form.append("side", side);
  if (recordingDurationMs !== null && recordingDurationMs > 0) {
    form.append("recording_duration_ms", String(Math.round(recordingDurationMs)));
  }

  onProgress?.(0.3);
  const res = await authedFetch("/api/analyze-ankle", {
    method: "POST",
    body: form,
  });
  onProgress?.(0.85);

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    const detail = formatAnkleError(body.detail, res.status);
    throw new Error(detail);
  }
  const wrapper = (await res.json()) as {
    success: boolean;
    data: BiomechDataDTO | null;
    error: string | null;
  };
  onProgress?.(1.0);
  if (!wrapper.success || !wrapper.data) {
    throw new Error(wrapper.error ?? "Ankle analysis failed.");
  }
  return wrapper.data;
}

function formatAnkleError(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.length > 0) return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        if (!d || typeof d !== "object") return String(d);
        const obj = d as Record<string, unknown>;
        const loc = Array.isArray(obj.loc) ? obj.loc.join(".") : "";
        const msg = typeof obj.msg === "string" ? obj.msg : JSON.stringify(obj);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  return `Ankle analysis failed (${status})`;
}
