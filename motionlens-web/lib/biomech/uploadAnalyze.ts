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
  captureShoulderRotationBaseline,
  computeShoulderAngle,
  computeShoulderRotationFromBaseline,
  detectShoulderAbAdDirection,
  detectShoulderFlexExtDirection,
  detectShoulderRotationDirection,
  isShoulderRotationNeutral,
  SHOULDER_MOVEMENTS,
  type ShoulderMovementId,
  type ShoulderRotationCalibration,
} from "@/lib/biomech/shoulder";
import {
  computeNeckAngle,
  detectNeckFlexExtDirection,
  detectNeckLateralDirection,
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
import { LM } from "@/lib/pose/landmarks";
import type { Keypoint } from "@tensorflow-models/pose-detection";
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

/** Score-sustenance decay per frame. The smoother below keeps the
 *  higher of (current raw score) and (previous smoothed score *
 *  this factor). Closer to 1 = visibility holds longer through
 *  occlusion blips; closer to 0 = score follows raw more tightly.
 *  0.9 sustains a brief 1-2 frame visibility drop without making
 *  long-term occlusions look indefinitely visible. */
const SCORE_DECAY = 0.9;

/** Build a stateful keypoint helper for one video analysis pass.
 *  The output preserves RAW positions (no position-smoothing lag)
 *  but applies an EMA-max on confidence scores so brief
 *  one-or-two-frame visibility drops don't reject peak-ROM frames.
 *
 *  Why not EMA-smooth positions like LiveBiomechCamera does?
 *  Smoothing positions inevitably lags brief peaks — a knee
 *  flexion that's only at maximum bend for a few frames gets
 *  damped to a lower reading, and the upload report shows e.g. 61°
 *  for a video where the patient genuinely reaches 90°. Live mode
 *  compensates for this lag with a held-candidate confirmation
 *  algorithm that survives brief noise frames; the upload pipeline
 *  uses simple min/max tracking, so it needs the true per-frame
 *  positions to find the real peak. The score side of the smoother
 *  still helps — without it, single-frame ankle / wrist occlusions
 *  rejected legitimate peak frames as low-visibility. */
function createKeypointSmoother(): (raw: Keypoint[]) => Keypoint[] {
  let scoreBuffer: number[] | null = null;
  return (raw: Keypoint[]): Keypoint[] => {
    if (!scoreBuffer || scoreBuffer.length !== raw.length) {
      // First frame, or detector returned a different keypoint
      // count — seed scores and pass positions through.
      scoreBuffer = raw.map((k) => k.score ?? 0);
      return raw;
    }
    const out: Keypoint[] = new Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const r = raw[i];
      const decayed = scoreBuffer[i] * SCORE_DECAY;
      const rawScore = r.score ?? 0;
      const newScore = Math.max(rawScore, decayed);
      scoreBuffer[i] = newScore;
      out[i] = {
        x: r.x,
        y: r.y,
        score: newScore,
        name: r.name,
      };
    }
    return out;
  };
}

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

  // ── Shoulder flexion + extension → backend MediaPipe BlazePose Full ──
  // The browser MoveNet path (analyzeMergedShoulderVideo) is
  // GPU/CPU-dependent and gave inconsistent results across devices
  // for fast arm movements. Flexion + extension are the highest-
  // priority pair to be device-consistent, so they route to the
  // backend /api/analyze-shoulder pipeline (same MediaPipe pipeline
  // used by gait / ankle / TUG). The endpoint returns the extended
  // BiomechDataDTO shape with secondary_peak_* fields populated.
  if (bodyPart === "shoulder" && movement === "flexion_extension") {
    return analyzeShoulderBackend(file, "flexion_extension", side ?? "right", onProgress);
  }

  // ── Shoulder abduction + adduction → backend MediaPipe BlazePose Full ──
  // Same rationale as flex/ext: backend pipeline for device-
  // consistent fast-movement capture. The backend's merged ab/ad
  // branch mirrors the browser analyser's direction-detection rules
  // (elbow position vs test-side shoulder, with the overhead y-axis
  // override) so live and upload modes agree frame-by-frame.
  if (bodyPart === "shoulder" && movement === "abduction_adduction") {
    return analyzeShoulderBackend(file, "abduction_adduction", side ?? "right", onProgress);
  }

  // ── Shoulder rotation (internal + external) → backend MediaPipe ──
  // The browser MoveNet path ran a streaming baseline-calibration +
  // arcsin pipeline; the backend now implements the same formula
  // against BlazePose Full, so live and upload modes give the same
  // peak ±2°. The endpoint maps the calibration-failed error to
  // HTTP 400 with a user-actionable "Neutral pose not detected"
  // message which formatShoulderError surfaces verbatim.
  if (bodyPart === "shoulder" && movement === "rotation") {
    return analyzeShoulderBackend(file, "rotation", side ?? "right", onProgress);
  }

  // ── Merged knee flexion + extension → min/max analyser ──
  // Knee "flexion_extension" captures the full knee ROM in one
  // recording. The angle metric (180° - interior knee angle, range
  // 0-140°) is bidirectional: max-of-trial = peak flexion (most
  // bent), min-of-trial = peak extension (most straight). No
  // direction routing — both peaks update on every gated frame.
  if (bodyPart === "knee" && movement === "flexion_extension") {
    return analyzeMergedKneeVideo({
      file,
      side: side ?? "right",
      onProgress,
    });
  }

  // ── Merged neck tests → direction-aware analyser ──
  // Neck "flexion_extension" captures forward (chin to chest) and
  // backward (head back) tilt in one recording. Neck
  // "lateral_flexion" captures right + left ear-to-shoulder tilt
  // in one recording. Both share the same direction-routing flow
  // and the analyser parameterises which formula + detector to
  // use internally.
  if (
    bodyPart === "neck" &&
    (movement === "flexion_extension" || movement === "lateral_flexion")
  ) {
    return analyzeMergedNeckVideo({
      file,
      movement,
      onProgress,
    });
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

  // Key-frame screenshot tracking. Even for non-merged
  // single-direction tests we want the saved report to include
  // neutral + peak thumbnails (same UX as merged tests). Buffer
  // canvas locks the analysed frame so the saved screenshot
  // matches the keypoints that scored it.
  const compCanvas = document.createElement("canvas");
  const frameBuffer = document.createElement("canvas");
  const frameBufferCtx = frameBuffer.getContext("2d");
  const smoother = createKeypointSmoother();
  let neutralUrl: string | null = null;
  let neutralMag = Infinity;
  let peakUrl: string | null = null;
  let peakMagSoFar = -Infinity;

  // Helper to run one detection on the currently-displayed frame.
  const sideOrRight = side ?? "right";
  const detectCurrentFrame = async (): Promise<number | null> => {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || !frameBufferCtx) return null;
    // Snapshot to buffer BEFORE the async detector call so the
    // pose and the screenshot both come from the same locked
    // frame (the video advances during the await otherwise).
    if (frameBuffer.width !== vw || frameBuffer.height !== vh) {
      frameBuffer.width = vw;
      frameBuffer.height = vh;
    }
    frameBufferCtx.drawImage(video, 0, 0, vw, vh);

    let poses;
    try {
      poses = await detector.estimatePoses(frameBuffer, {
        flipHorizontal: false,
      });
    } catch {
      return null;
    }
    const pose = poses[0];
    if (!pose) return null;
    const kps = smoother(pose.keypoints);

    let angle: number | null = null;
    switch (bodyPart) {
      case "shoulder":
        angle = computeShoulderAngle(
          movement as ShoulderMovementId,
          kps,
          sideOrRight,
        );
        break;
      case "neck":
        angle = computeNeckAngle(movement as NeckMovementId, kps);
        break;
      case "knee":
        angle = computeKneeAngle(
          movement as KneeMovementId,
          kps,
          sideOrRight,
        );
        break;
      case "hip":
        angle = computeHipAngle(
          movement as HipMovementId,
          kps,
          sideOrRight,
        );
        break;
      // NOTE: "ankle" is intentionally absent here — the early
      // return at the top of analyzeBiomechVideo dispatches ankle
      // requests to the backend MediaPipe pipeline. TypeScript
      // therefore narrows `bodyPart` to non-ankle inside this
      // switch.
    }
    if (angle === null || Number.isNaN(angle)) return null;

    // Capture neutral (lowest |angle| seen so far) and peak
    // (highest |angle|). For tests where the magnitude formula is
    // already direction-symmetric this gives the natural "rest
    // pose" + "max ROM" thumbnails the report's Key Frames
    // section renders.
    const absA = Math.abs(angle);
    if (absA < neutralMag) {
      neutralMag = absA;
      neutralUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
    }
    if (absA > peakMagSoFar) {
      peakMagSoFar = absA;
      peakUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
    }
    return angle;
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

  // ── Status classification (asymmetric, matches AssessmentReport.classify) ──
  // Exceeding the normal range upper bound is not impairment for
  // ROM screening, so above-range gets graceful treatment; below-
  // range (restricted ROM) keeps the strict thresholds.
  let status: "good" | "fair" | "poor";
  const rangeWidth = Math.max(1, refRange[1] - refRange[0]);
  if (peakMag >= refRange[0] && peakMag <= refRange[1]) {
    status = "good";
  } else if (peakMag < refRange[0]) {
    const distFrac = (refRange[0] - peakMag) / rangeWidth;
    status = distFrac <= 0.30 ? "fair" : "poor";
  } else {
    const distFrac = (peakMag - refRange[1]) / rangeWidth;
    if (distFrac <= 0.30) status = "good";
    else if (distFrac <= 1.0) status = "fair";
    else status = "poor";
  }

  // Assemble neutral + peak key frames (only if we have them).
  const keyFrames: Array<{
    label: string;
    frame_index: number;
    image_data_url: string;
  }> = [];
  if (neutralUrl) {
    keyFrames.push({
      label: "Neutral — start",
      frame_index: 0,
      image_data_url: neutralUrl,
    });
  }
  if (peakUrl && peakMagSoFar > 0) {
    keyFrames.push({
      label: `Peak ${movement.replace(/_/g, " ")} (${peakMag.toFixed(1)}°)`,
      frame_index: 1,
      image_data_url: peakUrl,
    });
  }

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
    key_frames: keyFrames,
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

// ─── Merged shoulder video analyser ─────────────────────────────
// Handles the "rotation" and "abduction_adduction" merged tests for
// uploaded videos. Direction-aware peak tracking with two parallel
// slots, plus per-direction key-frame screenshots (neutral, primary
// peak, secondary peak) composited from the source video + a
// skeleton overlay.
//
// For rotation, the formula needs a calibration baseline (the
// patient's upper-arm pixel length while the forearm is pointed at
// the camera). We auto-detect it from early frames using the same
// isShoulderRotationNeutral check the live mode uses — once enough
// consecutive frames satisfy "neutral pose", we snapshot the
// baseline and switch to the calibrated arcsin formula. For
// abduction_adduction, no calibration is required (the magnitude
// formula is direction-symmetric).

interface MergedShoulderOpts {
  file: File;
  movement: "rotation" | "abduction_adduction" | "flexion_extension";
  side: "left" | "right";
  onProgress?: (fraction: number) => void;
}

const MERGED_BASELINE_STABLE_FRAMES = 5;

async function analyzeMergedShoulderVideo(
  opts: MergedShoulderOpts,
): Promise<BiomechDataDTO> {
  const { file, movement, side, onProgress } = opts;
  const isRotation = movement === "rotation";
  const isFlexExt = movement === "flexion_extension";

  // Look up labels + target ranges from the metadata table so the
  // response matches the chooser configuration.
  const meta = SHOULDER_MOVEMENTS.find((m) => m.id === movement);
  const fallbackPrimary: [number, number] = isRotation
    ? [70, 90]
    : isFlexExt
      ? [150, 180]
      : [150, 180];
  const fallbackSecondary: [number, number] = isRotation
    ? [60, 80]
    : isFlexExt
      ? [45, 60]
      : [30, 50];
  const fallbackPrimaryLabel = isRotation
    ? "External Rotation"
    : isFlexExt
      ? "Flexion"
      : "Abduction";
  const fallbackSecondaryLabel = isRotation
    ? "Internal Rotation"
    : isFlexExt
      ? "Extension"
      : "Adduction";
  const primaryTarget: [number, number] = meta?.target ?? fallbackPrimary;
  const secondaryTarget: [number, number] = meta?.secondaryTarget ?? fallbackSecondary;
  const primaryLabel = meta?.primaryLabel ?? fallbackPrimaryLabel;
  const secondaryLabel = meta?.secondaryLabel ?? fallbackSecondaryLabel;

  const detector = await getDetector();

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

  video.currentTime = 0;
  await new Promise<void>((resolve) => {
    video.onseeked = () => resolve();
  });

  const duration = video.duration;
  if (!isFinite(duration) || duration <= 0) {
    URL.revokeObjectURL(url);
    throw new Error("Video duration is invalid.");
  }

  // Per-frame state
  let baseline: ShoulderRotationCalibration | null = null;
  let baselineStable = 0;
  let peakPrimarySigned: number | null = null;
  let peakSecondarySigned: number | null = null;
  let neutralUrl: string | null = null;
  // Magnitude associated with the currently-saved neutralUrl. We
  // keep updating neutralUrl whenever a lower-magnitude (i.e. more
  // genuinely "at rest") frame appears, so the final neutral
  // screenshot reflects the patient's true resting pose rather than
  // whatever happened to be the first valid frame (which could be
  // mid-motion if the video starts with the patient already moving).
  let neutralMag = Infinity;
  let peakPrimaryUrl: string | null = null;
  let peakSecondaryUrl: string | null = null;
  let totalFrames = 0;
  let validFrames = 0;
  let measuredFps = 30;

  // Composite canvas reused for each captured screenshot.
  const compCanvas = document.createElement("canvas");

  // Frame-snapshot canvas. At the start of every onFrame fire we
  // synchronously draw the current video frame into this buffer and
  // run the rest of the analysis (pose detection + key-frame capture)
  // against the buffer instead of the live <video>. The live video
  // keeps playing (at 4× speed in the rVFC path) during the async
  // detector.estimatePoses call, so without snapshotting, the
  // keypoints and the saved screenshot would be drawn from different
  // frames — the captured peak photo would show a frame several
  // moments after the analysed pose. The buffer locks the frame.
  const frameBuffer = document.createElement("canvas");
  const frameBufferCtx = frameBuffer.getContext("2d");

  // Per-trial keypoint smoother — matches LiveBiomechCamera's EMA
  // smoothing pass so the upload pipeline isn't more noise-vulnerable
  // than live mode.
  const smoother = createKeypointSmoother();

  // Run one detection on the currently-displayed video frame and
  // apply the merged-test bookkeeping (baseline detection, direction
  // routing, dual peak update, key-frame capture). Returns `null` on
  // any failure so the iteration loop can keep going.
  const processCurrentFrame = async (): Promise<void> => {
    totalFrames += 1;
    // Snapshot the current video frame into the buffer canvas
    // BEFORE any awaits. The video element may advance during the
    // pose-detection call (esp. at 4× playback) and we want the
    // detected keypoints + the saved screenshot to come from the
    // same locked frame.
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || !frameBufferCtx) return;
    if (frameBuffer.width !== vw || frameBuffer.height !== vh) {
      frameBuffer.width = vw;
      frameBuffer.height = vh;
    }
    frameBufferCtx.drawImage(video, 0, 0, vw, vh);

    let poses;
    try {
      poses = await detector.estimatePoses(frameBuffer, { flipHorizontal: false });
    } catch {
      return;
    }
    const pose = poses[0];
    if (!pose) return;
    const kps = smoother(pose.keypoints);

    // ── Baseline calibration (rotation only) ─────────────────
    if (isRotation && !baseline) {
      if (isShoulderRotationNeutral(kps, side)) {
        baselineStable += 1;
        if (baselineStable >= MERGED_BASELINE_STABLE_FRAMES) {
          baseline = captureShoulderRotationBaseline(kps, side);
        }
      } else {
        baselineStable = 0;
      }
      // Don't process angle/peak until baseline locked.
      return;
    }

    // ── Per-frame angle ──────────────────────────────────────
    let angle: number | null;
    if (isRotation && baseline) {
      angle = computeShoulderRotationFromBaseline(kps, side, baseline);
    } else if (isFlexExt) {
      // Flex/Ext: reuse legacy "flexion" formula (signed angle
      // between trunk-down and arm; magnitude is what we display,
      // direction is determined separately below).
      angle = computeShoulderAngle(
        "flexion" as ShoulderMovementId,
        kps,
        side,
      );
    } else if (!isRotation) {
      // ab/ad: magnitude formula is direction-symmetric. Reuse the
      // legacy "abduction" branch (trunk-down vs arm angle).
      angle = computeShoulderAngle(
        "abduction" as ShoulderMovementId,
        kps,
        side,
      );
    } else {
      angle = null;
    }
    if (angle === null || isNaN(angle)) return;
    validFrames += 1;

    // Neutral screenshot — track the LOWEST-magnitude frame seen so
    // far rather than just the first valid frame. This way if the
    // video opens with the patient already mid-motion, the saved
    // neutral image will still be the genuine at-rest pose (when
    // the magnitude returns near zero between repetitions).
    const absForNeutral = Math.abs(angle);
    if (absForNeutral < neutralMag) {
      neutralMag = absForNeutral;
      neutralUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
    }

    // ── Direction routing ────────────────────────────────────
    let direction: "primary" | "secondary" | null = null;
    if (isRotation) {
      const r = detectShoulderRotationDirection(kps, side);
      if (r === "external") direction = "primary";
      else if (r === "internal") direction = "secondary";
    } else if (isFlexExt) {
      const fe = detectShoulderFlexExtDirection(kps, side);
      if (fe === "flexion") direction = "primary";
      else if (fe === "extension") direction = "secondary";
    } else {
      const a = detectShoulderAbAdDirection(kps, side);
      if (a === "abduction") direction = "primary";
      else if (a === "adduction") direction = "secondary";
    }
    if (!direction) return; // deadband — don't touch either peak

    const absA = Math.abs(angle);
    if (direction === "primary") {
      if (peakPrimarySigned === null || absA > Math.abs(peakPrimarySigned)) {
        peakPrimarySigned = angle;
        peakPrimaryUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
      }
    } else {
      if (peakSecondarySigned === null || absA > Math.abs(peakSecondarySigned)) {
        peakSecondarySigned = angle;
        peakSecondaryUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
      }
    }
  };

  const supportsRVFC = typeof video.requestVideoFrameCallback === "function";

  if (supportsRVFC) {
    video.playbackRate = 4;
    let firstMediaTime: number | null = null;
    let lastMediaTime = 0;

    await new Promise<void>((resolve, reject) => {
      let resolved = false;
      let stallTimer: ReturnType<typeof setTimeout> | null = null;
      const STALL_MS = 1500;

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
        stallTimer = setTimeout(finish, STALL_MS);
      };

      const onFrame: VideoFrameRequestCallback = async (_now, metadata) => {
        if (resolved) return;
        if (firstMediaTime === null) firstMediaTime = metadata.mediaTime;
        lastMediaTime = metadata.mediaTime;

        await processCurrentFrame();
        onProgress?.(Math.min(1, metadata.mediaTime / duration));

        const nearEnd =
          metadata.mediaTime >= Math.max(duration - 0.1, duration * 0.98);
        if (video.ended || nearEnd) {
          finish();
          return;
        }

        armStallTimer();
        video.requestVideoFrameCallback(onFrame);
      };

      video.addEventListener("ended", finish, { once: true });
      armStallTimer();
      video.requestVideoFrameCallback(onFrame);
      video.play().catch((e) => {
        if (!resolved) reject(e);
      });
    });
    video.pause();
  } else {
    measuredFps = SAMPLE_FPS_FALLBACK;
    const sampleCount = Math.max(1, Math.floor(duration * SAMPLE_FPS_FALLBACK));
    for (let i = 0; i < sampleCount; i++) {
      const t = i / SAMPLE_FPS_FALLBACK;
      video.currentTime = t;
      await new Promise<void>((resolve) => {
        video.onseeked = () => resolve();
      });
      await processCurrentFrame();
      onProgress?.((i + 1) / sampleCount);
      if (i % 4 === 0) {
        await new Promise<void>((r) => setTimeout(r, 0));
      }
    }
  }

  URL.revokeObjectURL(url);

  // ── Build result DTO ─────────────────────────────────────
  const peakAMag =
    peakPrimarySigned !== null ? Math.abs(peakPrimarySigned) : 0;
  const peakBMag =
    peakSecondarySigned !== null ? Math.abs(peakSecondarySigned) : 0;
  const targetUpper = primaryTarget[1];
  const percentage = targetUpper > 0 ? (peakAMag / targetUpper) * 100 : 0;
  const status: "good" | "fair" | "poor" =
    percentage >= 90 ? "good" : percentage >= 75 ? "fair" : "poor";

  const keyFrames: { label: string; frame_index: number; image_data_url: string }[] = [];
  if (neutralUrl) {
    keyFrames.push({
      label: "Neutral — start",
      frame_index: 0,
      image_data_url: neutralUrl,
    });
  }
  if (peakPrimaryUrl && peakAMag > 0) {
    keyFrames.push({
      label: `${primaryLabel} (${peakAMag.toFixed(1)}°)`,
      frame_index: 1,
      image_data_url: peakPrimaryUrl,
    });
  }
  if (peakSecondaryUrl && peakBMag > 0) {
    keyFrames.push({
      label: `${secondaryLabel} (${peakBMag.toFixed(1)}°)`,
      frame_index: 2,
      image_data_url: peakSecondaryUrl,
    });
  }

  const interpretation =
    `${primaryLabel} (${side}) measured ${peakAMag.toFixed(1)}°` +
    ` against the ${primaryTarget[0]}°–${primaryTarget[1]}° normal range. ` +
    (peakBMag > 0
      ? `${secondaryLabel} measured ${peakBMag.toFixed(1)}°` +
        ` against the ${secondaryTarget[0]}°–${secondaryTarget[1]}° normal range.`
      : `${secondaryLabel} direction was not detected in this recording.`);

  return {
    body_part: "shoulder",
    movement,
    side,
    peak_angle: peakPrimarySigned,
    peak_magnitude: peakAMag,
    reference_range: primaryTarget,
    target: targetUpper,
    percentage,
    status,
    valid_frames: validFrames,
    total_frames: totalFrames,
    fps: measuredFps,
    interpretation,
    key_frames: keyFrames,
    secondary_peak_angle: peakSecondarySigned,
    secondary_peak_magnitude: peakBMag,
    secondary_reference_range: secondaryTarget,
    primary_label: primaryLabel,
    secondary_label: secondaryLabel,
  };
}

// ─── Merged knee video analyser ─────────────────────────────────
// Knee "flexion_extension" — captures both ends of the knee ROM in
// one recording. The angle metric is the existing
// computeKneeAngle output (180° - interior_knee_angle); we track
// its running MAX (flexion peak, e.g. 130°) and MIN (extension
// peak / residual flexion at the patient's straightest position,
// e.g. 2°) over the trial. No baseline calibration needed — the
// formula returns a directly-interpretable value.
//
// Three composite key-frame thumbnails are produced: neutral
// (first usable frame), peak flexion, and peak extension.

interface MergedKneeOpts {
  file: File;
  side: "left" | "right";
  onProgress?: (fraction: number) => void;
}

async function analyzeMergedKneeVideo(
  opts: MergedKneeOpts,
): Promise<BiomechDataDTO> {
  const { file, side, onProgress } = opts;
  const movement = "flexion_extension";

  const meta = KNEE_MOVEMENTS.find((m) => m.id === movement);
  const primaryTarget: [number, number] = meta?.target ?? [125, 145];
  const secondaryTarget: [number, number] = meta?.secondaryTarget ?? [0, 10];
  const primaryLabel = meta?.primaryLabel ?? "Flexion";
  const secondaryLabel = meta?.secondaryLabel ?? "Extension";

  const detector = await getDetector();

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

  video.currentTime = 0;
  await new Promise<void>((resolve) => {
    video.onseeked = () => resolve();
  });

  const duration = video.duration;
  if (!isFinite(duration) || duration <= 0) {
    URL.revokeObjectURL(url);
    throw new Error("Video duration is invalid.");
  }

  // Per-trial min/max state.
  let peakFlex: number | null = null;   // max angle seen
  let peakExt: number | null = null;    // min angle seen
  let neutralUrl: string | null = null;
  // Magnitude associated with the currently-saved neutralUrl. We
  // keep updating it whenever a frame closer to true "knee straight"
  // (lowest flexion-from-straight reading) shows up, so the saved
  // neutral screenshot is the patient's genuine resting pose and not
  // whatever happened to be the first valid frame (which may be
  // mid-motion if the video starts with the patient already moving).
  let neutralMag = Infinity;
  let peakFlexUrl: string | null = null;
  let peakExtUrl: string | null = null;
  let totalFrames = 0;
  let validFrames = 0;
  let measuredFps = 30;

  const compCanvas = document.createElement("canvas");

  // Frame-snapshot canvas. Locks the frame at the start of each
  // analysis cycle so the pose detection + the saved key-frame
  // screenshot both reflect the SAME moment — the video element
  // would otherwise advance several frames during the async pose-
  // detection await (4× playback in the rVFC path) and the saved
  // peak photo would not match the keypoints it was scored against.
  const frameBuffer = document.createElement("canvas");
  const frameBufferCtx = frameBuffer.getContext("2d");

  // Per-trial keypoint smoother — same EMA behaviour LiveBiomechCamera
  // uses in live mode. Without this, single-frame visibility drops on
  // the test-side ankle / hip (common in profile-view knee flexion
  // where the lifted leg can briefly occlude its own keypoints)
  // returned a null angle from computeKneeAngle and the actual peak
  // never got recorded against the candidate slot.
  const smoother = createKeypointSmoother();

  const processCurrentFrame = async (): Promise<void> => {
    totalFrames += 1;
    // Snapshot first — before any awaits — so the buffer holds the
    // exact frame we score and screenshot.
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || !frameBufferCtx) return;
    if (frameBuffer.width !== vw || frameBuffer.height !== vh) {
      frameBuffer.width = vw;
      frameBuffer.height = vh;
    }
    frameBufferCtx.drawImage(video, 0, 0, vw, vh);

    let poses;
    try {
      poses = await detector.estimatePoses(frameBuffer, { flipHorizontal: false });
    } catch {
      return;
    }
    const pose = poses[0];
    if (!pose) return;
    const kps = smoother(pose.keypoints);

    const angle = computeKneeAngle(
      "flexion" as KneeMovementId,
      kps,
      side,
    );
    if (angle === null || isNaN(angle)) return;
    validFrames += 1;

    // Neutral screenshot = lowest flexion-from-straight reading
    // seen so far. For knee tests this is the patient's "most
    // straight" pose — also functions as the extension reference.
    if (angle < neutralMag) {
      neutralMag = angle;
      neutralUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
    }

    // MAX-tracking for flexion peak.
    if (peakFlex === null || angle > peakFlex) {
      peakFlex = angle;
      peakFlexUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
    }
    // MIN-tracking for extension peak.
    if (peakExt === null || angle < peakExt) {
      peakExt = angle;
      peakExtUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
    }
  };

  const supportsRVFC = typeof video.requestVideoFrameCallback === "function";

  if (supportsRVFC) {
    video.playbackRate = 4;
    let firstMediaTime: number | null = null;
    let lastMediaTime = 0;
    await new Promise<void>((resolve, reject) => {
      let resolved = false;
      let stallTimer: ReturnType<typeof setTimeout> | null = null;
      const STALL_MS = 1500;
      const finish = () => {
        if (resolved) return;
        resolved = true;
        if (stallTimer) clearTimeout(stallTimer);
        const span = lastMediaTime - (firstMediaTime ?? 0);
        if (span > 0) measuredFps = totalFrames / span;
        resolve();
      };
      const armStall = () => {
        if (stallTimer) clearTimeout(stallTimer);
        stallTimer = setTimeout(finish, STALL_MS);
      };
      const onFrame: VideoFrameRequestCallback = async (_now, metadata) => {
        if (resolved) return;
        if (firstMediaTime === null) firstMediaTime = metadata.mediaTime;
        lastMediaTime = metadata.mediaTime;
        await processCurrentFrame();
        onProgress?.(Math.min(1, metadata.mediaTime / duration));
        const nearEnd =
          metadata.mediaTime >= Math.max(duration - 0.1, duration * 0.98);
        if (video.ended || nearEnd) {
          finish();
          return;
        }
        armStall();
        video.requestVideoFrameCallback(onFrame);
      };
      video.addEventListener("ended", finish, { once: true });
      armStall();
      video.requestVideoFrameCallback(onFrame);
      video.play().catch((e) => {
        if (!resolved) reject(e);
      });
    });
    video.pause();
  } else {
    measuredFps = SAMPLE_FPS_FALLBACK;
    const sampleCount = Math.max(1, Math.floor(duration * SAMPLE_FPS_FALLBACK));
    for (let i = 0; i < sampleCount; i++) {
      const t = i / SAMPLE_FPS_FALLBACK;
      video.currentTime = t;
      await new Promise<void>((resolve) => {
        video.onseeked = () => resolve();
      });
      await processCurrentFrame();
      onProgress?.((i + 1) / sampleCount);
      if (i % 4 === 0) await new Promise<void>((r) => setTimeout(r, 0));
    }
  }

  URL.revokeObjectURL(url);

  const flexMag = peakFlex ?? 0;
  const extMag = peakExt ?? 0;
  const targetUpper = primaryTarget[1];
  const percentage = targetUpper > 0 ? (flexMag / targetUpper) * 100 : 0;
  // Asymmetric range-aware classification — same logic as
  // AssessmentReport.classify. Above-range = mostly good (exceeding
  // normal ROM isn't impairment); below-range keeps the strict
  // fair / poor thresholds.
  const classifyAsym = (v: number, r: [number, number]): "good" | "fair" | "poor" => {
    if (v >= r[0] && v <= r[1]) return "good";
    const w = Math.max(1, r[1] - r[0]);
    if (v < r[0]) return (r[0] - v) / w <= 0.30 ? "fair" : "poor";
    const above = (v - r[1]) / w;
    if (above <= 0.30) return "good";
    if (above <= 1.0) return "fair";
    return "poor";
  };
  const status: "good" | "fair" | "poor" = classifyAsym(flexMag, primaryTarget);

  const keyFrames: { label: string; frame_index: number; image_data_url: string }[] = [];
  if (neutralUrl) {
    keyFrames.push({
      label: "Neutral — start",
      frame_index: 0,
      image_data_url: neutralUrl,
    });
  }
  if (peakFlexUrl && peakFlex !== null) {
    keyFrames.push({
      label: `${primaryLabel} (${flexMag.toFixed(1)}°)`,
      frame_index: 1,
      image_data_url: peakFlexUrl,
    });
  }
  if (peakExtUrl && peakExt !== null) {
    keyFrames.push({
      label: `${secondaryLabel} (${extMag.toFixed(1)}°)`,
      frame_index: 2,
      image_data_url: peakExtUrl,
    });
  }

  const interpretation =
    `${primaryLabel} (${side}) measured ${flexMag.toFixed(1)}°` +
    ` against the ${primaryTarget[0]}°–${primaryTarget[1]}° normal range. ` +
    (peakExt !== null
      ? `${secondaryLabel} measured ${extMag.toFixed(1)}°` +
        ` against the ${secondaryTarget[0]}°–${secondaryTarget[1]}° normal range.`
      : `${secondaryLabel} direction was not detected in this recording.`);

  return {
    body_part: "knee",
    movement,
    side,
    peak_angle: peakFlex,
    peak_magnitude: flexMag,
    reference_range: primaryTarget,
    target: targetUpper,
    percentage,
    status,
    valid_frames: validFrames,
    total_frames: totalFrames,
    fps: measuredFps,
    interpretation,
    key_frames: keyFrames,
    secondary_peak_angle: peakExt,
    secondary_peak_magnitude: extMag,
    secondary_reference_range: secondaryTarget,
    primary_label: primaryLabel,
    secondary_label: secondaryLabel,
  };
}

// ─── Merged neck video analyser ─────────────────────────────────
// Neck "flexion_extension" — captures forward tilt (flexion) and
// backward tilt (extension) in one lateral-view recording.
// Magnitude comes from computeNeckAngle (signed angle between
// vertical and the shoulder-to-ear vector); direction is detected
// per frame via detectNeckFlexExtDirection.

interface MergedNeckOpts {
  file: File;
  movement: "flexion_extension" | "lateral_flexion";
  onProgress?: (fraction: number) => void;
}

async function analyzeMergedNeckVideo(
  opts: MergedNeckOpts,
): Promise<BiomechDataDTO> {
  const { file, movement, onProgress } = opts;
  const isLateral = movement === "lateral_flexion";

  const meta = NECK_MOVEMENTS.find((m) => m.id === movement);
  const fallbackPrimaryTarget: [number, number] = isLateral ? [20, 45] : [45, 80];
  const fallbackSecondaryTarget: [number, number] = isLateral ? [20, 45] : [50, 70];
  const fallbackPrimaryLabel = isLateral ? "Right Lateral Flexion" : "Flexion";
  const fallbackSecondaryLabel = isLateral ? "Left Lateral Flexion" : "Extension";
  const primaryTarget: [number, number] = meta?.target ?? fallbackPrimaryTarget;
  const secondaryTarget: [number, number] = meta?.secondaryTarget ?? fallbackSecondaryTarget;
  const primaryLabel = meta?.primaryLabel ?? fallbackPrimaryLabel;
  const secondaryLabel = meta?.secondaryLabel ?? fallbackSecondaryLabel;

  const detector = await getDetector();

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

  video.currentTime = 0;
  await new Promise<void>((resolve) => {
    video.onseeked = () => resolve();
  });

  const duration = video.duration;
  if (!isFinite(duration) || duration <= 0) {
    URL.revokeObjectURL(url);
    throw new Error("Video duration is invalid.");
  }

  // Per-trial state.
  let peakPrimarySigned: number | null = null;
  let peakSecondarySigned: number | null = null;
  let neutralUrl: string | null = null;
  let neutralMag = Infinity;
  let peakPrimaryUrl: string | null = null;
  let peakSecondaryUrl: string | null = null;
  let totalFrames = 0;
  let validFrames = 0;
  let measuredFps = 30;

  const compCanvas = document.createElement("canvas");
  const frameBuffer = document.createElement("canvas");
  const frameBufferCtx = frameBuffer.getContext("2d");
  const smoother = createKeypointSmoother();

  const processCurrentFrame = async (): Promise<void> => {
    totalFrames += 1;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || !frameBufferCtx) return;
    if (frameBuffer.width !== vw || frameBuffer.height !== vh) {
      frameBuffer.width = vw;
      frameBuffer.height = vh;
    }
    frameBufferCtx.drawImage(video, 0, 0, vw, vh);

    let poses;
    try {
      poses = await detector.estimatePoses(frameBuffer, { flipHorizontal: false });
    } catch {
      return;
    }
    const pose = poses[0];
    if (!pose) return;
    const kps = smoother(pose.keypoints);

    const angle = computeNeckAngle(
      (isLateral ? "lateral_flexion" : "flexion") as NeckMovementId,
      kps,
    );
    if (angle === null || isNaN(angle)) return;
    validFrames += 1;

    const absA = Math.abs(angle);
    if (absA < neutralMag) {
      neutralMag = absA;
      neutralUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
    }

    // Direction routing — flex/ext uses sagittal detector,
    // lateral uses coronal-plane detector. Both return "primary"
    // (positive direction) or "secondary" (negative); either may
    // also return null when the magnitude is inside the deadband.
    let primaryHit: boolean;
    let secondaryHit: boolean;
    if (isLateral) {
      const lat = detectNeckLateralDirection(kps);
      if (!lat) return;
      primaryHit = lat === "right";
      secondaryHit = lat === "left";
    } else {
      const fe = detectNeckFlexExtDirection(kps);
      if (!fe) return;
      primaryHit = fe === "flexion";
      secondaryHit = fe === "extension";
    }

    if (primaryHit) {
      if (peakPrimarySigned === null || absA > Math.abs(peakPrimarySigned)) {
        peakPrimarySigned = angle;
        peakPrimaryUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
      }
    } else if (secondaryHit) {
      if (peakSecondarySigned === null || absA > Math.abs(peakSecondarySigned)) {
        peakSecondarySigned = angle;
        peakSecondaryUrl = captureCompositeFrame(compCanvas, frameBuffer, kps);
      }
    }
  };

  const supportsRVFC = typeof video.requestVideoFrameCallback === "function";
  if (supportsRVFC) {
    video.playbackRate = 4;
    let firstMediaTime: number | null = null;
    let lastMediaTime = 0;
    await new Promise<void>((resolve, reject) => {
      let resolved = false;
      let stallTimer: ReturnType<typeof setTimeout> | null = null;
      const STALL_MS = 1500;
      const finish = () => {
        if (resolved) return;
        resolved = true;
        if (stallTimer) clearTimeout(stallTimer);
        const span = lastMediaTime - (firstMediaTime ?? 0);
        if (span > 0) measuredFps = totalFrames / span;
        resolve();
      };
      const armStall = () => {
        if (stallTimer) clearTimeout(stallTimer);
        stallTimer = setTimeout(finish, STALL_MS);
      };
      const onFrame: VideoFrameRequestCallback = async (_now, metadata) => {
        if (resolved) return;
        if (firstMediaTime === null) firstMediaTime = metadata.mediaTime;
        lastMediaTime = metadata.mediaTime;
        await processCurrentFrame();
        onProgress?.(Math.min(1, metadata.mediaTime / duration));
        const nearEnd =
          metadata.mediaTime >= Math.max(duration - 0.1, duration * 0.98);
        if (video.ended || nearEnd) {
          finish();
          return;
        }
        armStall();
        video.requestVideoFrameCallback(onFrame);
      };
      video.addEventListener("ended", finish, { once: true });
      armStall();
      video.requestVideoFrameCallback(onFrame);
      video.play().catch((e) => {
        if (!resolved) reject(e);
      });
    });
    video.pause();
  } else {
    measuredFps = SAMPLE_FPS_FALLBACK;
    const sampleCount = Math.max(1, Math.floor(duration * SAMPLE_FPS_FALLBACK));
    for (let i = 0; i < sampleCount; i++) {
      const t = i / SAMPLE_FPS_FALLBACK;
      video.currentTime = t;
      await new Promise<void>((resolve) => {
        video.onseeked = () => resolve();
      });
      await processCurrentFrame();
      onProgress?.((i + 1) / sampleCount);
      if (i % 4 === 0) await new Promise<void>((r) => setTimeout(r, 0));
    }
  }

  URL.revokeObjectURL(url);

  const peakAMag = peakPrimarySigned !== null ? Math.abs(peakPrimarySigned) : 0;
  const peakBMag = peakSecondarySigned !== null ? Math.abs(peakSecondarySigned) : 0;
  const targetUpper = primaryTarget[1];
  const percentage = targetUpper > 0 ? (peakAMag / targetUpper) * 100 : 0;
  // Asymmetric classification — see AssessmentReport.classify.
  const classifyAsym = (v: number, r: [number, number]): "good" | "fair" | "poor" => {
    if (v >= r[0] && v <= r[1]) return "good";
    const w = Math.max(1, r[1] - r[0]);
    if (v < r[0]) return (r[0] - v) / w <= 0.30 ? "fair" : "poor";
    const above = (v - r[1]) / w;
    if (above <= 0.30) return "good";
    if (above <= 1.0) return "fair";
    return "poor";
  };
  const status: "good" | "fair" | "poor" = classifyAsym(peakAMag, primaryTarget);

  const keyFrames: { label: string; frame_index: number; image_data_url: string }[] = [];
  if (neutralUrl) {
    keyFrames.push({
      label: "Neutral — start",
      frame_index: 0,
      image_data_url: neutralUrl,
    });
  }
  if (peakPrimaryUrl && peakAMag > 0) {
    keyFrames.push({
      label: `${primaryLabel} (${peakAMag.toFixed(1)}°)`,
      frame_index: 1,
      image_data_url: peakPrimaryUrl,
    });
  }
  if (peakSecondaryUrl && peakBMag > 0) {
    keyFrames.push({
      label: `${secondaryLabel} (${peakBMag.toFixed(1)}°)`,
      frame_index: 2,
      image_data_url: peakSecondaryUrl,
    });
  }

  const interpretation =
    `${primaryLabel} measured ${peakAMag.toFixed(1)}°` +
    ` against the ${primaryTarget[0]}°–${primaryTarget[1]}° normal range. ` +
    (peakBMag > 0
      ? `${secondaryLabel} measured ${peakBMag.toFixed(1)}°` +
        ` against the ${secondaryTarget[0]}°–${secondaryTarget[1]}° normal range.`
      : `${secondaryLabel} direction was not detected in this recording.`);

  return {
    body_part: "neck",
    movement,
    side: null,
    peak_angle: peakPrimarySigned,
    peak_magnitude: peakAMag,
    reference_range: primaryTarget,
    target: targetUpper,
    percentage,
    status,
    valid_frames: validFrames,
    total_frames: totalFrames,
    fps: measuredFps,
    interpretation,
    key_frames: keyFrames,
    secondary_peak_angle: peakSecondarySigned,
    secondary_peak_magnitude: peakBMag,
    secondary_reference_range: secondaryTarget,
    primary_label: primaryLabel,
    secondary_label: secondaryLabel,
  };
}

// Composite the current video frame + a skeleton overlay onto a
// reusable canvas and return a JPEG data URL. Used for the merged
// test's three key-frame thumbnails (neutral + per-direction peaks).
// The video is drawn as-is (uploaded videos aren't mirrored like the
// live selfie preview), so MoveNet's pixel-space keypoints can be
// scaled directly to canvas coords without any flip.
function captureCompositeFrame(
  canvas: HTMLCanvasElement,
  source: HTMLCanvasElement,
  keypoints: Keypoint[],
): string | null {
  const vw = source.width;
  const vh = source.height;
  if (!vw || !vh) return null;
  const targetW = Math.min(480, vw);
  const scale = targetW / vw;
  const cw = Math.round(vw * scale);
  const ch = Math.round(vh * scale);
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, cw, ch);

  const VIS = 0.35;
  const px = (i: number): { x: number; y: number; score: number } | null => {
    const k = keypoints[i];
    if (!k) return null;
    const score = k.score ?? 0;
    if (score < VIS) return null;
    return { x: (k.x / vw) * cw, y: (k.y / vh) * ch, score };
  };

  // Full-body skeleton edges + dots — same set the live capture uses.
  const edges: [number, number][] = [
    [LM.LEFT_SHOULDER, LM.LEFT_ELBOW],
    [LM.LEFT_ELBOW, LM.LEFT_WRIST],
    [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW],
    [LM.RIGHT_ELBOW, LM.RIGHT_WRIST],
    [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER],
    [LM.LEFT_SHOULDER, LM.LEFT_HIP],
    [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
    [LM.LEFT_HIP, LM.RIGHT_HIP],
    [LM.LEFT_HIP, LM.LEFT_KNEE],
    [LM.LEFT_KNEE, LM.LEFT_ANKLE],
    [LM.RIGHT_HIP, LM.RIGHT_KNEE],
    [LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
  ];
  const dots = [
    LM.NOSE,
    LM.LEFT_EAR, LM.RIGHT_EAR,
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_WRIST, LM.RIGHT_WRIST,
    LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_KNEE, LM.RIGHT_KNEE,
    LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
  ];

  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = Math.max(1.5, cw * 0.0035);
  ctx.shadowColor = "rgba(0,0,0,0.55)";
  ctx.shadowBlur = 2;
  for (const [a, b] of edges) {
    const A = px(a);
    const B = px(b);
    if (!A || !B) continue;
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.stroke();
  }
  ctx.fillStyle = "#EF4444";
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = Math.max(1, cw * 0.0025);
  for (const i of dots) {
    const P = px(i);
    if (!P) continue;
    const r = Math.max(3, cw * 0.005);
    ctx.beginPath();
    ctx.arc(P.x, P.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
  return canvas.toDataURL("image/jpeg", 0.78);
}

// ─── Shoulder backend dispatch ─────────────────────────────────
// Uploads the video to /api/analyze-shoulder (MediaPipe BlazePose
// Full) and returns the same BiomechDataDTO shape the in-browser
// merged analyser produced. The endpoint handles per-frame
// processing at native FPS server-side, which avoids the GPU/CPU
// device variability and frame-drop issues the browser MoveNet path
// had for fast arm movements.
//
// For shoulder "flexion_extension" the response carries both peaks
// (primary = flexion, secondary = extension) plus three key-frame
// thumbnails (neutral + flexion peak + extension peak) — the
// existing AssessmentReport dual-row rendering picks these up
// unchanged.
async function analyzeShoulderBackend(
  file: File,
  movement: string,
  side: "left" | "right",
  onProgress?: (fraction: number) => void,
): Promise<BiomechDataDTO> {
  onProgress?.(0.1);
  const form = new FormData();
  form.append("video", file, file.name || "shoulder.mp4");
  form.append("movement_type", movement);
  form.append("side", side);

  onProgress?.(0.3);
  const res = await authedFetch("/api/analyze-shoulder", {
    method: "POST",
    body: form,
  });
  onProgress?.(0.85);

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(formatShoulderError(body.detail, res.status));
  }
  const wrapper = (await res.json()) as {
    success: boolean;
    data: BiomechDataDTO | null;
    error: string | null;
  };
  onProgress?.(1.0);
  if (!wrapper.success || !wrapper.data) {
    throw new Error(wrapper.error ?? "Analysis failed. Please check connection and try again.");
  }
  return wrapper.data;
}

// Map the backend's structured error tokens (raised via
// HTTPException.detail in api.analyze_shoulder) to the user-facing
// strings the spec calls for. Anything not in the token list falls
// back to the raw detail string so debugging never loses information.
function formatShoulderError(detail: unknown, status: number): string {
  const raw = typeof detail === "string" ? detail : Array.isArray(detail)
    ? detail.map((d) => (typeof d === "string" ? d : JSON.stringify(d))).join("; ")
    : JSON.stringify(detail ?? {});

  if (raw.startsWith("fps_too_low")) {
    return "Video quality too low. Please record at 30 FPS or higher.";
  }
  if (raw.startsWith("video_too_short")) {
    return "Video too short. Please record at least 3 seconds of movement.";
  }
  if (raw.startsWith("duration_too_long")) {
    return "Video too long. Maximum 60 seconds.";
  }
  if (raw.startsWith("file_too_large")) {
    return "File too large. Maximum size is 100 MB.";
  }
  if (raw.startsWith("poor_visibility")) {
    return (
      "Arm not clearly visible in video. Please ensure the full arm is in " +
      "frame with good lighting."
    );
  }
  if (raw.startsWith("Neutral pose")) {
    // Rotation test specifically — the backend couldn't lock a
    // calibration baseline because the recording never started
    // with the patient at neutral.
    return raw;
  }
  if (raw.startsWith("Camera angle")) {
    return raw;
  }
  if (raw.startsWith("Requested side")) {
    return raw;
  }
  if (status >= 500) {
    return "Analysis failed. Please check connection and try again.";
  }
  return raw || `Analysis failed (HTTP ${status}).`;
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
