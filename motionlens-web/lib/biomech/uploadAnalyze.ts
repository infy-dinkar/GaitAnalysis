"use client";
// Biomech video-upload dispatcher.
//
// Every (bodyPart, movement) pair is uploaded to its backend
// /api/analyze-* endpoint, which runs MediaPipe BlazePose against the
// upload variant (currently pose_landmarker_heavy.task, 33 keypoints).
// This module now only routes, posts multipart form data, and maps
// backend error payloads to user-facing strings — it performs no pose
// detection of its own.
//
// It previously carried a browser-side fallback built on MoveNet
// SinglePose Lightning (17 keypoints). That fallback was reachable
// only by deprecated single-direction movement IDs, and it silently
// produced readings from a different model that could not be compared
// with anything else the system records, so it was removed. Anything
// that reaches the end of the router now throws instead.

import { authedFetch } from "@/lib/auth";
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
  // The former in-browser MoveNet analyser was GPU/CPU-dependent
  // and gave inconsistent results across devices
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

  // ── Shoulder LEGACY single-direction IDs → backend ──────────────
  // flexion / extension / abduction / adduction / external_rotation /
  // internal_rotation are hidden from the chooser (the merged tests
  // replaced them) but still resolve from a saved report or a
  // hand-typed ?movement= URL. They used to fall through to the
  // in-browser MoveNet analyser — a DIFFERENT model with 17 keypoints
  // instead of BlazePose's 33, silently producing numbers that can't
  // be compared with anything else in the system. That analyser has
  // since been removed.
  //
  // No mapping or reinterpretation is needed: /api/analyze-shoulder
  // accepts each of these verbatim (it validates against
  // SHOULDER_NORMAL_RANGES in addition to the merged IDs), and the
  // engine still carries the single-direction code path. This is a
  // routing fix only — the movement semantics are unchanged.
  if (
    bodyPart === "shoulder" && (
      movement === "flexion" ||
      movement === "extension" ||
      movement === "abduction" ||
      movement === "adduction" ||
      movement === "external_rotation" ||
      movement === "internal_rotation"
    )
  ) {
    return analyzeShoulderBackend(file, movement, side ?? "right", onProgress);
  }

  // ── Knee flexion + extension → backend MediaPipe BlazePose Full ──
  // Same rationale as the shoulder migrations: device-consistent
  // BlazePose pipeline beats the GPU-dependent browser MoveNet
  // path. Knee math is the simplest of the merged tests — just
  // min/max tracking of (180° − interior_knee_angle), no
  // direction detection / no calibration. The backend reuses the
  // same shared gait pose pipeline + _pose_rotation correction
  // already in place from the shoulder PRs, so portrait videos
  // and platform-specific cv2 behaviour are handled automatically.
  if (bodyPart === "knee" && movement === "flexion_extension") {
    return analyzeKneeBackend(file, "flexion_extension", side ?? "right", onProgress);
  }

  // ── Hip flexion + extension + rotation → backend MediaPipe ──
  // All hip tests route to /api/analyze-hip. The backend's
  // analyze_hip dispatches on the movement string:
  //   • flexion_extension is the MERGED standing side-on test —
  //     the thigh's signed angle from the image VERTICAL routes
  //     each frame into the flexion or extension peak slot. The
  //     vertical reference makes it immune to trunk lean, which
  //     the legacy trunk-referenced formula folded into the
  //     reading (and the 30° extension clamp then hid).
  //   • flexion / extension are the legacy single-direction tests
  //     (`180° − interior(trunk, thigh)` + max-tracker), kept for
  //     saved-report re-runs.
  //   • rotation is a merged test (internal + external captured
  //     in one trial) with a calibration baseline locked at the
  //     supine neutral pose — patient must hold "knee at 90°,
  //     lower leg pointing at camera" for ~5 frames at the
  //     start. Backend uses ankle-displacement foreshortening
  //     (the calibrated-arcsin approach used by shoulder rotation).
  // The legacy single-direction internal_rotation /
  // external_rotation IDs still exist in HIP_MOVEMENTS for
  // saved-report compatibility (hidden from the chooser); only
  // the merged "rotation" ID is reachable from the new UI.
  if (bodyPart === "hip" && (
        movement === "flexion_extension" ||
        movement === "flexion" ||
        movement === "extension" ||
        movement === "rotation"
      )) {
    return analyzeHipBackend(file, movement, side ?? "right", onProgress);
  }

  // ── Merged neck tests → direction-aware analyser ──
  // Neck "flexion_extension" captures forward (chin to chest) and
  // backward (head back) tilt in one recording. Neck
  // "lateral_flexion" captures right + left ear-to-shoulder tilt
  // in one recording. Both share the same direction-routing flow
  // and the analyser parameterises which formula + detector to
  // use internally.
  // ── All three merged neck tests → backend MediaPipe ──
  // All neck merged tests now route to /api/analyze-neck. The
  // backend's analyze_neck dispatches on the movement string:
  //   • flexion_extension uses the ear→nose tilt formula (LATERAL
  //     camera profile required — patient seen from the side).
  //   • lateral_flexion uses the shoulder→ear vs vertical formula
  //     (FRONTAL camera profile required — patient facing camera).
  //   • rotation uses the ear-width foreshortening formula with a
  //     baseline locked from the patient's facing-forward neutral
  //     pose at the start of the recording (FRONTAL view required).
  // The view + neutral-pose requirements are enforced by the
  // backend pre-flight checks, which surface actionable HTTP 400
  // messages. Live + upload report the same metric for every
  // movement because the backend formulas are verbatim ports of
  // the browser implementations.
  if (bodyPart === "neck" && (
        movement === "flexion_extension" ||
        movement === "lateral_flexion" ||
        movement === "rotation"
      )) {
    return analyzeNeckBackend(file, movement, onProgress);
  }

  // Every reachable (bodyPart, movement) pair is handled by a
  // backend branch above. This used to fall through to a browser
  // MoveNet analyser — a different model with 17 keypoints instead
  // of BlazePose's 33 — which silently produced numbers that could
  // not be compared with anything else the system records. The only
  // IDs that ever reached it are the deprecated single-direction
  // movements, and the upload pages now redirect those to their
  // merged equivalents before analysis starts. Fail loudly instead
  // of quietly switching models.
  throw new Error(
    `No backend analysis for ${bodyPart}/${movement}. `
    + `This movement is deprecated — use the merged test instead.`,
  );
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

// ─── Shoulder backend dispatch ─────────────────────────────────
// Uploads the video to /api/analyze-shoulder, which runs MediaPipe
// BlazePose against the upload variant server-side at native FPS.
// Doing it server-side avoids the GPU/CPU device variability and
// frame drops the removed in-browser analyser suffered on fast arm
// movements.
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

// ─── Knee backend client ─────────────────────────────────────────
// Posts the uploaded video to /api/analyze-knee and unwraps the
// BiomechResponse envelope. Mirrors analyzeShoulderBackend exactly;
// kept as a separate function so the body-part-specific error
// message map (formatKneeError) stays local.
async function analyzeKneeBackend(
  file: File,
  movement: string,
  side: "left" | "right",
  onProgress?: (fraction: number) => void,
): Promise<BiomechDataDTO> {
  onProgress?.(0.1);
  const form = new FormData();
  form.append("video", file, file.name || "knee.mp4");
  form.append("movement_type", movement);
  form.append("side", side);

  onProgress?.(0.3);
  const res = await authedFetch("/api/analyze-knee", {
    method: "POST",
    body: form,
  });
  onProgress?.(0.85);

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(formatKneeError(body.detail, res.status));
  }
  const wrapper = (await res.json()) as {
    success: boolean;
    data: BiomechDataDTO | null;
    error: string | null;
  };
  onProgress?.(1.0);
  if (!wrapper.success || !wrapper.data) {
    throw new Error(
      wrapper.error ?? "Analysis failed. Please check connection and try again.",
    );
  }
  return wrapper.data;
}

// Map the backend's structured error tokens (raised via
// HTTPException.detail in api.analyze_knee) to the user-facing
// strings the spec calls for. Anything not in the token list
// falls back to the raw detail string so debugging never loses
// information.
function formatKneeError(detail: unknown, status: number): string {
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
      "Knee not clearly visible in video. Please ensure the full leg is " +
      "visible with good lighting."
    );
  }
  if (raw.startsWith("Requested side")) {
    // Pre-flight wrong-side check — surface verbatim so the user
    // sees the exact actionable message (which side to switch to).
    return raw;
  }
  if (status >= 500) {
    return "Analysis failed. Please check connection and try again.";
  }
  return raw || `Analysis failed (HTTP ${status}).`;
}

// ─── Neck backend client ─────────────────────────────────────────
// Posts the uploaded video to /api/analyze-neck and unwraps the
// BiomechResponse envelope. Mirrors analyzeKneeBackend exactly
// (no `side` form field — neck flex/ext doesn't carry a side).
async function analyzeNeckBackend(
  file: File,
  movement: string,
  onProgress?: (fraction: number) => void,
): Promise<BiomechDataDTO> {
  onProgress?.(0.1);
  const form = new FormData();
  form.append("video", file, file.name || "neck.mp4");
  form.append("movement_type", movement);

  onProgress?.(0.3);
  const res = await authedFetch("/api/analyze-neck", {
    method: "POST",
    body: form,
  });
  onProgress?.(0.85);

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(formatNeckError(body.detail, res.status));
  }
  const wrapper = (await res.json()) as {
    success: boolean;
    data: BiomechDataDTO | null;
    error: string | null;
  };
  onProgress?.(1.0);
  if (!wrapper.success || !wrapper.data) {
    throw new Error(
      wrapper.error ?? "Analysis failed. Please check connection and try again.",
    );
  }
  return wrapper.data;
}

// Map the backend's structured error tokens (raised via
// HTTPException.detail in api.analyze_neck) to the user-facing
// strings the spec calls for. Anything not in the token list
// falls back to the raw detail string so debugging never loses
// information.
function formatNeckError(detail: unknown, status: number): string {
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
      "Face not clearly visible in video. Please ensure the head and " +
      "shoulders are in frame with good lighting throughout the recording."
    );
  }
  if (raw.startsWith("Camera angle")) {
    // Frontal/lateral view rejection — surface verbatim so the
    // user sees the exact view-required guidance for whichever
    // neck movement they uploaded.
    return raw;
  }
  if (raw.startsWith("Neutral pose")) {
    // Rotation test couldn't lock a calibration baseline at the
    // start of the recording — surface verbatim so the user sees
    // the "start facing forward" guidance.
    return raw;
  }
  if (status >= 500) {
    return "Analysis failed. Please check connection and try again.";
  }
  return raw || `Analysis failed (HTTP ${status}).`;
}

// ─── Hip backend client ─────────────────────────────────────────
// Posts the uploaded video to /api/analyze-hip and unwraps the
// BiomechResponse envelope. Mirrors analyzeKneeBackend exactly;
// kept as a separate function so the body-part-specific error
// message map (formatHipError) stays local.
async function analyzeHipBackend(
  file: File,
  movement: string,
  side: "left" | "right",
  onProgress?: (fraction: number) => void,
): Promise<BiomechDataDTO> {
  onProgress?.(0.1);
  const form = new FormData();
  form.append("video", file, file.name || "hip.mp4");
  form.append("movement_type", movement);
  form.append("side", side);

  onProgress?.(0.3);
  const res = await authedFetch("/api/analyze-hip", {
    method: "POST",
    body: form,
  });
  onProgress?.(0.85);

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(formatHipError(body.detail, res.status));
  }
  const wrapper = (await res.json()) as {
    success: boolean;
    data: BiomechDataDTO | null;
    error: string | null;
  };
  onProgress?.(1.0);
  if (!wrapper.success || !wrapper.data) {
    throw new Error(
      wrapper.error ?? "Analysis failed. Please check connection and try again.",
    );
  }
  return wrapper.data;
}

// Map the backend's structured error tokens (raised via
// HTTPException.detail in api.analyze_hip) to the user-facing
// strings the spec calls for. Anything not in the token list
// falls back to the raw detail string so debugging never loses
// information.
function formatHipError(detail: unknown, status: number): string {
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
      "Hip and leg not clearly visible. Please ensure the full leg is in " +
      "frame with good lighting throughout the recording."
    );
  }
  if (raw.startsWith("Requested side")) {
    // Pre-flight wrong-side check — surface verbatim so the user
    // sees the exact actionable message (which side to switch to).
    return raw;
  }
  if (raw.startsWith("Neutral pose")) {
    // Merged hip rotation couldn't lock a calibration baseline —
    // surface verbatim so the user sees the "supine + knee at 90°
    // + lower leg pointing at camera" guidance.
    return raw;
  }
  if (status >= 500) {
    return "Analysis failed. Please check connection and try again.";
  }
  return raw || `Analysis failed (HTTP ${status}).`;
}
