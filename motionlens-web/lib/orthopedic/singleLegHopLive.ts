// D3 Single-Leg Hop — IN-BROWSER live analyzer (per leg).
//
// Mirrors engines/orthopedic/single_leg_hop_engine.py so a fully live
// (biomech-style) session builds the SAME SingleLegHopResult the report
// already renders — no video is sent to the backend. Runs per-frame on
// BlazePose-tfjs live keypoints (pixel space). One analyzer per leg;
// the LSI is computed client-side from the two legs' best distances.
//
// Per-leg detection (identical thresholds to the backend, all in
// singleLegHop.ts):
//   • Baseline lock — test-side ankle.y stable while grounded.
//   • Takeoff = ankle.y rises > AIRBORNE_LIFT_FRAC_OF_LEG above baseline.
//   • Landing = ankle.y returns within LANDED_BAND_FRAC_OF_LEG AND has
//     been airborne >= MIN_AIRBORNE_SEC.
//   • Hop distance = |heelX_landing − heelX_takeoff| (foot_index / ankle
//     fallback when the heel is occluded).
//   • Validity — contralateral foot must NOT enter the grounded band
//     during flight, and the distance must exceed the minimum.
//
// NOTE: the live pose model (BlazePose-tfjs) differs from the backend
// (MediaPipe), so live numbers can differ slightly from an uploaded
// clip — the same trade-off every biomech live mode makes.

import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import {
  AIRBORNE_LIFT_FRAC_OF_LEG,
  CONTRALATERAL_TOUCH_GRACE_SAMPLES,
  LANDED_BAND_FRAC_OF_LEG,
  MAX_TRIALS,
  MIN_AIRBORNE_SEC,
  MIN_TRIAL_GAP_SEC,
  VIS_THRESHOLD,
  type CalibrationResult,
  type Side,
  type SingleLegHopResult,
  type Trial,
} from "@/lib/orthopedic/singleLegHop";

const STANDING_HOLD_SAMPLES = 15; // ~0.5 s at 30 fps
const STANDING_TOL_FRAC = 0.04;

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const SIDE_LM = {
  left: {
    ankle: LM.LEFT_ANKLE, heel: LM.LEFT_HEEL, foot: LM.LEFT_FOOT_INDEX,
    hip: LM.LEFT_HIP, contraAnkle: LM.RIGHT_ANKLE,
  },
  right: {
    ankle: LM.RIGHT_ANKLE, heel: LM.RIGHT_HEEL, foot: LM.RIGHT_FOOT_INDEX,
    hip: LM.RIGHT_HIP, contraAnkle: LM.LEFT_ANKLE,
  },
} as const;

interface Row {
  tMs: number;
  frameIndex: number;
  ankleY: number;
  refX: number;            // heel.x → foot_index.x → ankle.x fallback
  hipY: number | null;
  contraAnkleY: number | null;
  contraGroundedRef: number | null; // contralateral baseline for touch test
}

interface RawTrial {
  takeoffFrame: number;
  landingFrame: number;
  takeoffTMs: number;
  landingTMs: number;
  hopDistancePx: number;
  contraTouched: boolean;
}

/** Live HUD snapshot from every pushFrame(). */
export interface SLHLiveFrame {
  tracking: boolean;
  baselineLocked: boolean;
  baselineProgress: number; // 0..1 while acquiring
  airborne: boolean;
  validTrials: number;
  totalTrials: number;
  maxTrials: number;
  lastHopCm: number | null;   // null when uncalibrated
  lastHopPx: number | null;
  lastValid: boolean | null;
  bestValidCm: number | null;
  bestValidPx: number | null;
}

export class SingleLegHopLiveAnalyzer {
  private readonly side: Side;
  private lm: (typeof SIDE_LM)[Side];

  private frameIndex = 0;
  private legLenProvisional = 0;
  private buf: Row[] = [];
  private calibration: CalibrationResult | null = null;
  private ppc: number | null = null;

  private locked = false;
  private baseAnkleY = 0;
  private legLen = 0;
  private contraBaselineY: number | null = null;

  private state: "grounded" | "airborne" = "grounded";
  private takeoffFrame = 0;
  private takeoffTMs = 0;
  private takeoffRefX = 0;
  private airborneStartMs = 0;
  private contraTouchRun = 0;
  private contraTouchedThisHop = false;
  private lastLandingMs = -1e9;
  private trials: RawTrial[] = [];

  constructor(side: Side) {
    this.side = side;
    this.lm = SIDE_LM[side];
  }

  reset(calibration: CalibrationResult | null): void {
    this.frameIndex = 0;
    this.legLenProvisional = 0;
    this.buf = [];
    this.calibration = calibration;
    this.ppc =
      calibration && calibration.pixels_per_cm > 0
        ? calibration.pixels_per_cm
        : null;
    this.locked = false;
    this.state = "grounded";
    this.contraTouchRun = 0;
    this.contraTouchedThisHop = false;
    this.lastLandingMs = -1e9;
    this.trials = [];
  }

  private vis(kp: LiveKeypoint[], idx: number): boolean {
    const p = kp[idx];
    return !!p && (p.score ?? 0) >= VIS_THRESHOLD;
  }

  private extractRow(kp: LiveKeypoint[], tMs: number): Row | null {
    if (!this.vis(kp, this.lm.ankle)) return null;
    const ankle = kp[this.lm.ankle];
    // Horizontal reference for hop distance: heel → foot_index → ankle.
    const refX = this.vis(kp, this.lm.heel)
      ? kp[this.lm.heel].x
      : this.vis(kp, this.lm.foot)
        ? kp[this.lm.foot].x
        : ankle.x;
    const hipY = this.vis(kp, this.lm.hip) ? kp[this.lm.hip].y : null;
    const contraAnkleY = this.vis(kp, this.lm.contraAnkle)
      ? kp[this.lm.contraAnkle].y
      : null;
    return {
      tMs,
      frameIndex: this.frameIndex,
      ankleY: ankle.y,
      refX,
      hipY,
      contraAnkleY,
      contraGroundedRef: contraAnkleY,
    };
  }

  private tryLockBaseline(): void {
    if (this.buf.length < STANDING_HOLD_SAMPLES) return;
    const window = this.buf.slice(-STANDING_HOLD_SAMPLES);
    const ys = window.map((r) => r.ankleY);
    const med = median(ys);
    const maxDev = Math.max(...ys.map((y) => Math.abs(y - med)));
    const tol = STANDING_TOL_FRAC * Math.max(1, this.legLenProvisional);
    if (maxDev > tol) return;

    this.baseAnkleY = med;
    const hips = window.map((r) => r.hipY).filter((v): v is number => v !== null);
    this.legLen = hips.length
      ? Math.max(1, Math.abs(med - median(hips)))
      : Math.max(1, this.legLenProvisional || 300);
    const contra = window
      .map((r) => r.contraAnkleY)
      .filter((v): v is number => v !== null);
    this.contraBaselineY = contra.length ? median(contra) : null;
    this.locked = true;
  }

  // Hop-distance minimum validation removed on request — a trial is
  // now invalid ONLY when the contralateral foot touches down (a real
  // single-leg violation). The distance is still recorded + shown.
  private trialValid(t: RawTrial): boolean {
    return !t.contraTouched;
  }

  pushFrame(kp: LiveKeypoint[], tMs: number): SLHLiveFrame {
    this.frameIndex += 1;
    const row = this.extractRow(kp, tMs);
    const hud = (over: Partial<SLHLiveFrame>): SLHLiveFrame => {
      const valid = this.trials.filter((t) => this.trialValid(t));
      const bestPx = valid.length
        ? Math.max(...valid.map((t) => t.hopDistancePx))
        : null;
      const last = this.trials.length
        ? this.trials[this.trials.length - 1]
        : null;
      return {
        tracking: row !== null,
        baselineLocked: this.locked,
        baselineProgress: Math.min(1, this.buf.length / STANDING_HOLD_SAMPLES),
        airborne: this.state === "airborne",
        validTrials: valid.length,
        totalTrials: this.trials.length,
        maxTrials: MAX_TRIALS,
        lastHopPx: last ? last.hopDistancePx : null,
        lastHopCm:
          last && this.ppc !== null ? last.hopDistancePx / this.ppc : null,
        lastValid: last ? this.trialValid(last) : null,
        bestValidPx: bestPx,
        bestValidCm: bestPx !== null && this.ppc !== null ? bestPx / this.ppc : null,
        ...over,
      };
    };

    if (!row) return hud({});

    const gap =
      row.hipY !== null ? Math.abs(row.ankleY - row.hipY) : 0;
    if (gap > this.legLenProvisional) this.legLenProvisional = gap;

    if (!this.locked) {
      this.buf.push(row);
      if (this.buf.length > STANDING_HOLD_SAMPLES * 3) this.buf.shift();
      this.tryLockBaseline();
      return hud({});
    }

    // ── Locked: takeoff / landing state machine ──────────────────
    const lift = this.baseAnkleY - row.ankleY; // +ve = foot up (y-down)
    const airborneThresh = this.legLen * AIRBORNE_LIFT_FRAC_OF_LEG;
    const landedBand = this.legLen * LANDED_BAND_FRAC_OF_LEG;

    if (this.state === "grounded") {
      if (
        lift > airborneThresh
        && row.tMs - this.lastLandingMs >= MIN_TRIAL_GAP_SEC * 1000
        && this.trials.length < MAX_TRIALS * 2
      ) {
        this.state = "airborne";
        this.takeoffFrame = row.frameIndex;
        this.takeoffTMs = row.tMs;
        this.takeoffRefX = row.refX;
        this.airborneStartMs = row.tMs;
        this.contraTouchRun = 0;
        this.contraTouchedThisHop = false;
      }
    } else {
      // Contralateral-touch validity: the OTHER foot must not land.
      if (
        this.contraBaselineY !== null
        && row.contraAnkleY !== null
        && Math.abs(row.contraAnkleY - this.contraBaselineY) < landedBand
      ) {
        this.contraTouchRun += 1;
        if (this.contraTouchRun > CONTRALATERAL_TOUCH_GRACE_SAMPLES) {
          this.contraTouchedThisHop = true;
        }
      } else {
        this.contraTouchRun = 0;
      }

      const landed = Math.abs(row.ankleY - this.baseAnkleY) < landedBand;
      const airborneMs = row.tMs - this.airborneStartMs;
      if (landed && airborneMs >= MIN_AIRBORNE_SEC * 1000) {
        this.trials.push({
          takeoffFrame: this.takeoffFrame,
          landingFrame: row.frameIndex,
          takeoffTMs: this.takeoffTMs,
          landingTMs: row.tMs,
          hopDistancePx: Math.abs(row.refX - this.takeoffRefX),
          contraTouched: this.contraTouchedThisHop,
        });
        this.state = "grounded";
        this.lastLandingMs = row.tMs;
      }
    }

    return hud({});
  }

  hasBaseline(): boolean {
    return this.locked;
  }
  validTrialCount(): number {
    return this.trials.filter((t) => this.trialValid(t)).length;
  }

  /** Build the per-leg SingleLegHopResult (or null if no baseline). */
  finalize(
    patientHeightCm: number | null,
    durationSeconds: number,
  ): SingleLegHopResult | null {
    if (!this.locked) return null;

    const trials: Trial[] = this.trials.map((t, i) => {
      const contra = t.contraTouched;
      const valid = !contra;
      const reason: string | null = contra
        ? "Contralateral foot touched down during the hop (not a single-leg hop)."
        : null;
      return {
        trial_index: i + 1,
        takeoff_frame_index: t.takeoffFrame,
        landing_frame_index: t.landingFrame,
        takeoff_t_ms: t.takeoffTMs,
        landing_t_ms: t.landingTMs,
        hop_distance_px: t.hopDistancePx,
        hop_distance_cm: this.ppc !== null ? t.hopDistancePx / this.ppc : null,
        valid,
        invalidation_reason: reason,
      };
    });

    const valid = trials.filter((t) => t.valid);
    let bestIdx: number | null = null;
    let bestPx: number | null = null;
    let bestCm: number | null = null;
    if (valid.length) {
      const best = valid.reduce((a, b) =>
        b.hop_distance_px > a.hop_distance_px ? b : a,
      );
      bestIdx = best.trial_index;
      bestPx = best.hop_distance_px;
      bestCm = best.hop_distance_cm;
    }

    const interpretation =
      valid.length > 0 && bestCm !== null
        ? `${this.side} leg: best valid hop ${bestCm.toFixed(1)} cm across ${valid.length} of ${trials.length} valid trial(s).`
        : valid.length > 0
          ? `${this.side} leg: ${valid.length} valid hop(s) (relative units — no calibration).`
          : trials.length > 0
            ? `${this.side} leg: ${trials.length} hop(s) detected but none passed the validity gate.`
            : `${this.side} leg: no hops detected.`;

    return {
      side_tested: this.side,
      patient_height_cm: patientHeightCm,
      calibration: this.calibration,
      baseline_ankle_y_px: this.baseAnkleY,
      leg_length_px: this.legLen,
      trials,
      best_valid_trial_index: bestIdx,
      best_valid_hop_px: bestPx,
      best_valid_hop_cm: bestCm,
      peak_screenshot_data_url: null,
      duration_seconds: durationSeconds,
      termination: "completed",
      fps: null,
      total_frames: this.frameIndex,
      valid_frames: this.frameIndex,
      interpretation,
    };
  }
}
