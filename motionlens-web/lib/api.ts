// Thin client for the MotionLens FastAPI backend.

// Empty default => fetch() calls use path-relative URLs (e.g. /api/auth/login)
// which the Next.js rewrite in next.config.ts proxies to the AWS EC2
// backend. Override via NEXT_PUBLIC_API_BASE_URL in .env.local when
// pointing a dev frontend at a remote backend directly (e.g. HF Space
// or a teammate's tunnelled local backend).
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "";

// ─── DTOs (match api_models.py) ──────────────────────────────────────
export interface PatientInfoDTO {
  name: string | null;
  height_cm: number;
}

export interface VideoInfoDTO {
  duration_sec: number;
  fps: number;
  total_frames: number;
  calibration_mm_per_px: number | null;
  valid_passes: number;
  frames_used: number;
  ankle_baseline_left: number;
  ankle_baseline_right: number;
  ankle_baseline_method: string;
  ankle_baseline_n_frames: number;
}

export interface MetricsBlockDTO {
  step_count: number;
  cadence: number | null;
  symmetry: number | null;
  knee_peak: number | null;
  stride_cv: number | null;
  step_length: number | null;
  step_length_unit: string;
  torso_lean: number | null;
  step_time: number | null;
  window_seconds: number;
  window_description: string;
  validated_passes: number | null;
  video_coverage_pct: number | null;
}

export interface JointDetailDTO {
  peak: number | null;
  min: number | null;
  rom: number | null;
  mean: number | null;
  time_series: (number | null)[];
}

export interface JointAnglesBlockDTO {
  left_knee: JointDetailDTO;
  right_knee: JointDetailDTO;
  left_hip: JointDetailDTO;
  right_hip: JointDetailDTO;
  left_ankle: JointDetailDTO;
  right_ankle: JointDetailDTO;
}

export interface GaitCycleCurveDTO {
  mean_curve: (number | null)[];
  std_curve: (number | null)[];
  K: number;
}

export interface CycleSideBlockDTO {
  cycles_accepted: number;
  cycles_rejected_amplitude: number;
  cycles_kept: number;
  cycles_rejected_too_long: number;
  cycles_rejected_too_short: number;
  hip: GaitCycleCurveDTO;
  knee: GaitCycleCurveDTO;
  ankle: GaitCycleCurveDTO;
}

export interface NormalRefCurveDTO {
  mean_curve: number[];
  lower_band: number[];
  upper_band: number[];
}

export interface GaitCycleDataDTO {
  left: CycleSideBlockDTO;
  right: CycleSideBlockDTO;
  normal_reference: { hip: NormalRefCurveDTO; knee: NormalRefCurveDTO; ankle: NormalRefCurveDTO };
  stance_phase_pct: number;
}

export interface NormalizedSeriesPairDTO {
  left: (number | null)[];
  right: (number | null)[];
}

export interface NormalizedOverviewDTO {
  time_axis: number[];
  leg_angle: NormalizedSeriesPairDTO;
  knee_flexion: NormalizedSeriesPairDTO;
  hip_flexion: NormalizedSeriesPairDTO;
  ankle_deflection: NormalizedSeriesPairDTO;
}

export interface PassSegmentDTO {
  start_sec: number;
  end_sec: number;
  core_start_sec: number;
  core_end_sec: number;
  direction: number;
}

export interface TabsDataDTO {
  heel_position: {
    time_axis: number[];
    left_x: (number | null)[];
    right_x: (number | null)[];
    left_strikes_t: number[];
    right_strikes_t: number[];
    left_strikes_x: number[];
    right_strikes_x: number[];
    left_count: number;
    right_count: number;
  };
  step_length: {
    left_values: number[];
    right_values: number[];
    left_mean: number;
    right_mean: number;
    unit: string;
  };
  timing: {
    left_intervals: number[];
    right_intervals: number[];
    left_mean: number;
    right_mean: number;
  };
  torso_lean: {
    time_axis: number[];
    angles: (number | null)[];
    mean: number;
    std: number;
  };
  ankle_trajectory: {
    time_axis: number[];
    left_x: (number | null)[];
    right_x: (number | null)[];
    left_y: (number | null)[];
    right_y: (number | null)[];
  };
  pass_segments: PassSegmentDTO[];
}

export interface ObservationsDTO {
  hip: string[];
  knee: string[];
  ankle: string[];
  overall: string[];
  suggestions: string[];
}

export interface GaitDataDTO {
  patient_info: PatientInfoDTO;
  video_info: VideoInfoDTO;
  walking_direction: string;
  metrics_total: MetricsBlockDTO;
  metrics_clean: MetricsBlockDTO;
  joint_angles: JointAnglesBlockDTO;
  gait_cycle_data: GaitCycleDataDTO | null;
  normalized_overview: NormalizedOverviewDTO;
  tabs_data: TabsDataDTO;
  observations: ObservationsDTO;
}

export interface BiomechKeyFrameDTO {
  label: string;
  frame_index: number;
  /** JPEG data URL ready for an <img src="…"> tag. */
  image_data_url: string;
}

export interface BiomechDataDTO {
  body_part: "shoulder" | "neck" | "knee" | "hip" | "ankle";
  movement: string;
  side: "left" | "right" | null;
  peak_angle: number | null;
  peak_magnitude: number;
  reference_range: [number, number];
  target: number;
  percentage: number;
  status: "good" | "fair" | "poor";
  valid_frames: number;
  total_frames: number;
  fps: number;
  interpretation: string;
  /** Annotated screenshots of test key moments (neutral, peak ROM).
   *  Backend-processed body parts (ankle today) return 2-3 entries;
   *  live-frontend body parts return an empty array. */
  key_frames?: BiomechKeyFrameDTO[];
  // ── Merged-test secondary direction (optional) ──────────────────
  // Populated only for merged movements that capture two directions
  // in one trial (shoulder "rotation" = external+internal,
  // "abduction_adduction" = abduction+adduction). When set, the
  // report renders a second results row, a second chart bar, and a
  // second interpretation sentence. peak_* fields above refer to the
  // primary direction.
  secondary_peak_angle?: number | null;
  secondary_peak_magnitude?: number;
  secondary_reference_range?: [number, number];
  primary_label?: string;
  secondary_label?: string;
  // ── Compensatory-movement detection (optional) ──────────────────
  // Populated only for movements that run compensation tracking
  // (shoulder flexion+extension today). Empty/undefined for every
  // other (bodyPart, movement) pair, so other surfaces silently
  // render no compensation section. Mirrors the dict shape returned
  // by _track_flexion_extension_compensations() in shoulder_engine.py
  // and the Compensation type in lib/biomech/shoulder.ts.
  compensations?: BiomechCompensationDTO[];
}

/** One compensation tracked during a recording. The report renderer
 *  shows a colored card when `flagged: true`, a green "no issues"
 *  line when ALL flags in the array are false. */
export interface BiomechCompensationDTO {
  type:
    | "trunk_lean"
    | "shoulder_elevation"
    | "elbow_bend"
    | "elbow_drop"
    | "elbow_drift_from_side";
  label: string;
  severity: "high" | "medium" | "low";
  flagged: boolean;
  /** Human-readable peak measurement when `flagged: true`; omitted
   *  when not flagged. e.g. "Peak deviation 22.4° from baseline
   *  (threshold 15°)". */
  details?: string | null;
}

// ─── Live biomech (per-frame) ────────────────────────────────────────
export interface LandmarkPointDTO {
  x: number;          // normalized [0, 1]
  y: number;          // normalized [0, 1]
  visibility: number; // 0..1
}

export interface LiveBiomechFrameDataDTO {
  status: "good" | "low_visibility" | "no_landmarks";
  landmarks: LandmarkPointDTO[];
  current_angle: number | null;
  current_magnitude: number;
}

interface ApiEnvelope<T> {
  success: boolean;
  data: T | null;
  error: string | null;
}

// ─── core POST helpers ──────────────────────────────────────────────
async function postMultipart<T>(
  endpoint: string,
  form: FormData,
  onProgress?: (uploadedBytes: number, totalBytes: number) => void,
): Promise<ApiEnvelope<T>> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}${endpoint}`);
    xhr.responseType = "json";

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(e.loaded, e.total);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response) {
        resolve(xhr.response as ApiEnvelope<T>);
      } else {
        const detail =
          xhr.response?.detail || xhr.response?.error || `HTTP ${xhr.status}`;
        resolve({ success: false, data: null, error: String(detail) });
      }
    };
    xhr.onerror = () =>
      reject(new Error(`Network error reaching ${API_BASE_URL}${endpoint}`));
    xhr.send(form);
  });
}

// ─── public api ─────────────────────────────────────────────────────
export async function analyzeGait(
  args: {
    video: File;
    heightCm: number;
    patientName?: string | null;
    /** When set, signals to the backend that this upload came from
     *  the in-browser record-then-upload flow. Wall-clock duration in
     *  ms between MediaRecorder.start() and stop(). Lets the backend
     *  repair MediaRecorder WebMs that ship with broken duration
     *  headers before its FPS/duration gates run. Normal file uploads
     *  omit this field; the backend then skips the repair (no-op). */
    recordingDurationMs?: number | null;
  },
  onProgress?: (uploaded: number, total: number) => void,
) {
  const fd = new FormData();
  fd.append("video", args.video);
  fd.append("height_cm", String(args.heightCm));
  if (args.patientName) fd.append("patient_name", args.patientName);
  if (args.recordingDurationMs && args.recordingDurationMs > 0) {
    fd.append("recording_duration_ms", String(args.recordingDurationMs));
  }
  return postMultipart<GaitDataDTO>("/api/analyze-gait", fd, onProgress);
}

export async function analyzeShoulder(
  args: {
    video: File;
    movement: string;
    side: "left" | "right";
    patientName?: string | null;
  },
  onProgress?: (uploaded: number, total: number) => void,
) {
  const fd = new FormData();
  fd.append("video", args.video);
  fd.append("movement_type", args.movement);
  fd.append("side", args.side);
  if (args.patientName) fd.append("patient_name", args.patientName);
  return postMultipart<BiomechDataDTO>("/api/analyze-shoulder", fd, onProgress);
}

export async function analyzeNeck(
  args: { video: File; movement: string; patientName?: string | null },
  onProgress?: (uploaded: number, total: number) => void,
) {
  const fd = new FormData();
  fd.append("video", args.video);
  fd.append("movement_type", args.movement);
  if (args.patientName) fd.append("patient_name", args.patientName);
  return postMultipart<BiomechDataDTO>("/api/analyze-neck", fd, onProgress);
}

export async function analyzeLiveFrame(args: {
  frame: Blob;
  bodyPart: "shoulder" | "neck";
  movement: string;
  side?: "left" | "right";
}): Promise<ApiEnvelope<LiveBiomechFrameDataDTO>> {
  const fd = new FormData();
  fd.append("frame", args.frame, "frame.jpg");
  fd.append("body_part", args.bodyPart);
  fd.append("movement_type", args.movement);
  if (args.side) fd.append("side", args.side);
  try {
    const res = await fetch(`${API_BASE_URL}/api/live/biomech-frame`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      let detail: string = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        detail = j?.detail || j?.error || detail;
      } catch {
        // ignore
      }
      return { success: false, data: null, error: detail };
    }
    return (await res.json()) as ApiEnvelope<LiveBiomechFrameDataDTO>;
  } catch (e) {
    return {
      success: false,
      data: null,
      error: e instanceof Error ? e.message : "Network error",
    };
  }
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) return { ok: false, detail: `HTTP ${res.status}` };
    const json = await res.json();
    return { ok: true, detail: `${json.service} v${json.version}` };
  } catch (e) {
    return {
      ok: false,
      detail:
        e instanceof Error
          ? e.message
          : "API unreachable — is the backend running?",
    };
  }
}
