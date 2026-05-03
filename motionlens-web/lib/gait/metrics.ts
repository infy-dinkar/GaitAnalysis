import type { HeelStrike, DetectionDiagnostics } from "@/lib/gait/cycleDetection";

export interface GaitMetrics {
  cadence: number | null;       // steps / minute
  strideTime: number | null;    // seconds
  stepCount: number;
  cycleCount: number;
  symmetry: number | null;      // 0..1
  walkingSpeed: number | null;  // m/s
}

export interface ComputeMetricsInput {
  heelStrikes: HeelStrike[];
  totalSeconds: number;
  pxPerMeter?: number | null;
  hipDisplacementPx?: number;
}

export function computeGaitMetrics(input: ComputeMetricsInput): GaitMetrics {
  const { heelStrikes, totalSeconds, pxPerMeter, hipDisplacementPx } = input;

  if (heelStrikes.length < 2 || totalSeconds <= 0) {
    return {
      cadence: null,
      strideTime: null,
      stepCount: heelStrikes.length,
      cycleCount: 0,
      symmetry: null,
      walkingSpeed: null,
    };
  }

  const intervals: number[] = [];
  for (let i = 1; i < heelStrikes.length; i++) {
    intervals.push(heelStrikes[i].time - heelStrikes[i - 1].time);
  }
  const meanInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  const cadence = meanInterval > 0 ? 60 / meanInterval : null;
  const strideTime = meanInterval * 2;

  const odd = intervals.filter((_, i) => i % 2 === 0);
  const even = intervals.filter((_, i) => i % 2 === 1);
  let symmetry: number | null = null;
  if (odd.length && even.length) {
    const meanOdd = odd.reduce((a, b) => a + b, 0) / odd.length;
    const meanEven = even.reduce((a, b) => a + b, 0) / even.length;
    symmetry = Math.min(meanOdd, meanEven) / Math.max(meanOdd, meanEven);
  }

  let walkingSpeed: number | null = null;
  if (pxPerMeter && hipDisplacementPx && totalSeconds > 0) {
    walkingSpeed = hipDisplacementPx / pxPerMeter / totalSeconds;
  }

  return {
    cadence,
    strideTime,
    stepCount: heelStrikes.length,
    cycleCount: Math.floor(heelStrikes.length / 2),
    symmetry,
    walkingSpeed,
  };
}

// Per-joint angle summary
export interface JointSummary {
  peak: number | null; // max angle observed (most extended)
  min: number | null;  // min angle observed (most flexed)
  rom: number | null;  // peak - min
  mean: number | null;
  validFrames: number;
}

export function summarizeJoint(values: number[]): JointSummary {
  const valid = values.filter((v) => Number.isFinite(v));
  if (valid.length === 0) {
    return { peak: null, min: null, rom: null, mean: null, validFrames: 0 };
  }
  const peak = Math.max(...valid);
  const min = Math.min(...valid);
  const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
  return { peak, min, rom: peak - min, mean, validFrames: valid.length };
}

export const NORMAL_RANGES = {
  cadence: [100, 130] as [number, number],
  strideTime: [1.0, 1.2] as [number, number],
  walkingSpeed: [1.2, 1.5] as [number, number],
  symmetry: [0.95, 1.0] as [number, number],
  // joint flexion peak (180° - min angle) — adult walking at comfortable pace
  kneeFlexion: [55, 70] as [number, number],   // peak knee flexion in swing
  hipFlexion: [25, 40] as [number, number],    // peak hip flexion
  kneeRom: [55, 75] as [number, number],
  hipRom: [40, 55] as [number, number],
};

// Observations / clinical hints
export type Tone = "good" | "warn" | "bad" | "info";
export interface Observation {
  tone: Tone;
  text: string;
}

function inRange(v: number | null, r: [number, number]): boolean {
  return v !== null && v >= r[0] && v <= r[1];
}

export function generateObservations(args: {
  metrics: GaitMetrics;
  leftKnee: JointSummary;
  rightKnee: JointSummary;
  leftHip: JointSummary;
  rightHip: JointSummary;
  diag: DetectionDiagnostics;
  hasHeight: boolean;
}): Observation[] {
  const obs: Observation[] = [];
  const { metrics: m, leftKnee, rightKnee, leftHip, rightHip, diag, hasHeight } = args;

  // diagnostic if pose tracking quality was low
  if (diag.validFraction < 0.5) {
    obs.push({
      tone: "warn",
      text: `Only ${Math.round(diag.validFraction * 100)}% of frames had a confident ankle keypoint — try better lighting, plain background, or a longer side-on shot.`,
    });
  }
  if (m.stepCount === 0) {
    obs.push({
      tone: "bad",
      text: "No heel strikes detected. The patient may not be walking parallel to the camera, or the ankle isn't visible end-to-end.",
    });
  }

  if (m.cadence !== null) {
    if (inRange(m.cadence, NORMAL_RANGES.cadence)) {
      obs.push({ tone: "good", text: `Cadence (${m.cadence.toFixed(0)} steps/min) is within the normal adult range.` });
    } else if (m.cadence < NORMAL_RANGES.cadence[0]) {
      obs.push({ tone: "warn", text: `Cadence (${m.cadence.toFixed(0)} steps/min) is below the typical 100–130 range — slow gait.` });
    } else {
      obs.push({ tone: "warn", text: `Cadence (${m.cadence.toFixed(0)} steps/min) is above the typical range — short, hurried steps.` });
    }
  }

  if (m.symmetry !== null) {
    if (m.symmetry >= 0.95) {
      obs.push({ tone: "good", text: `Step symmetry is excellent (${Math.round(m.symmetry * 100)}%).` });
    } else if (m.symmetry >= 0.85) {
      obs.push({ tone: "warn", text: `Step symmetry is fair (${Math.round(m.symmetry * 100)}%) — minor side-to-side asymmetry.` });
    } else {
      obs.push({ tone: "bad", text: `Step symmetry is poor (${Math.round(m.symmetry * 100)}%) — clear asymmetric gait pattern.` });
    }
  }

  // knee flexion (180 - min knee angle = max bend)
  function kneeFlexAt(s: JointSummary) {
    return s.min !== null ? 180 - s.min : null;
  }
  const lkFlex = kneeFlexAt(leftKnee);
  const rkFlex = kneeFlexAt(rightKnee);
  if (lkFlex !== null) {
    if (inRange(lkFlex, NORMAL_RANGES.kneeFlexion)) {
      obs.push({ tone: "good", text: `Left knee peak flexion ${lkFlex.toFixed(0)}° is within normal range.` });
    } else if (lkFlex < NORMAL_RANGES.kneeFlexion[0]) {
      obs.push({ tone: "warn", text: `Left knee peak flexion ${lkFlex.toFixed(0)}° is below normal — limited swing-phase flexion.` });
    }
  }
  if (rkFlex !== null) {
    if (inRange(rkFlex, NORMAL_RANGES.kneeFlexion)) {
      obs.push({ tone: "good", text: `Right knee peak flexion ${rkFlex.toFixed(0)}° is within normal range.` });
    } else if (rkFlex < NORMAL_RANGES.kneeFlexion[0]) {
      obs.push({ tone: "warn", text: `Right knee peak flexion ${rkFlex.toFixed(0)}° is below normal — limited swing-phase flexion.` });
    }
  }

  // hip ROM
  if (leftHip.rom !== null && leftHip.rom < NORMAL_RANGES.hipRom[0]) {
    obs.push({ tone: "warn", text: `Left hip ROM ${leftHip.rom.toFixed(0)}° is reduced — typical of stiff or compensated gait.` });
  }
  if (rightHip.rom !== null && rightHip.rom < NORMAL_RANGES.hipRom[0]) {
    obs.push({ tone: "warn", text: `Right hip ROM ${rightHip.rom.toFixed(0)}° is reduced — typical of stiff or compensated gait.` });
  }

  if (!hasHeight) {
    obs.push({ tone: "info", text: "Walking speed not calculated — patient height was not provided on the setup page." });
  }

  if (obs.length === 0) {
    obs.push({ tone: "info", text: "Insufficient data for detailed observations." });
  }

  return obs;
}
