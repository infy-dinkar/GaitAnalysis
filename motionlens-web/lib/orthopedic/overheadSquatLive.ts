// B2 Overhead Squat — IN-BROWSER live analyzer.
//
// Mirrors engines/orthopedic/overhead_squat_engine.py so a fully live
// (biomech-style) session produces the SAME OverheadSquatResult shape
// the existing OverheadSquatReport already renders — no video is sent
// to the backend. Runs per-frame on BlazePose-tfjs live keypoints
// (pixel space). Only the 5 measurable frontal-plane items are scored
// live; torso-lean + heel-rise stay not_assessed (as in the backend).
//
// NOTE: the live pose model (BlazePose-tfjs) differs from the backend
// (MediaPipe), so live numbers can differ slightly from an uploaded
// clip. This is the same trade-off every biomech live mode makes.

import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import {
  ARM_OVERHEAD_MIN_FRAC_OF_TRUNK,
  DEPTH_TARGET_FRAC_OF_LEG,
  DESCENT_MIN_FRAC_OF_LEG,
  PELVIC_TILT_FAIL_FRAC_OF_HIPSPAN,
  VALGUS_KFPPA_FAIL_DEG,
  type CalibrationResult,
  type OverheadSquatChecklistItem,
  type OverheadSquatClassification,
  type OverheadSquatRep,
  type OverheadSquatResult,
} from "@/lib/orthopedic/overheadSquat";

// ─── Constants (mirror the backend engine) ─────────────────────
const VIS = 0.15;
const ARM_VIS = 0.1;
const STANDING_HOLD_SAMPLES = 9; // ~0.3 s at 30 fps
const STANDING_TOL_FRAC = 0.04;
const STANDING_RETURN_BAND_FRAC = 0.02;
const MIN_REP_DURATION_MS = 600;
const FOOT_MIN_RATIO = 0.7;
const FOOT_MAX_RATIO = 1.6;
// Arms must be overhead (worst wrist ≥ this fraction of trunk length
// ABOVE the shoulders) at the top when the descent begins — otherwise
// it's a plain squat, not an OVERHEAD squat, and the rep is ignored.
// Lower than the item-4 compensation threshold (0.30) so an arms-up
// rep that then DROPS still counts and gets flagged; only a clearly
// not-overhead setup (arms at/below shoulder) is rejected. A null arm
// reading (wrists clipping the frame) never blocks a rep.
const ARMS_UP_GATE = 0.1;

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Knee frontal-plane projection angle magnitude (deg). Higher = more
 *  valgus. Verbatim port of tuck_jump_engine._kfppa_deg. */
function kfppaDeg(
  hx: number, hy: number,
  kx: number, ky: number,
  ax: number, ay: number,
): number {
  const thighX = kx - hx, thighY = ky - hy;
  const shankX = ax - kx, shankY = ay - ky;
  const dot = thighX * shankX + thighY * shankY;
  const mag = Math.hypot(thighX, thighY) * Math.hypot(shankX, shankY);
  if (mag < 1e-6) return 0;
  const cosA = Math.max(-1, Math.min(1, dot / mag));
  const interior = (Math.acos(cosA) * 180) / Math.PI;
  return Math.max(0, Math.min(90, 180 - interior));
}

interface Row {
  tMs: number;
  frameIndex: number;
  hipY: number;
  lhipX: number; lhipY: number; rhipX: number; rhipY: number;
  lknX: number; lknY: number; rknX: number; rknY: number;
  lankX: number; lankY: number; rankX: number; rankY: number;
  lshY: number; rshY: number; shoulderY: number;
  lwrY: number | null; rwrY: number | null;
}

/** Live HUD snapshot returned by every pushFrame(). */
export interface OHSLiveFrame {
  tracking: boolean;
  baselineLocked: boolean;
  baselineProgress: number; // 0..1 while acquiring
  repCount: number;
  descending: boolean;
  depthFrac: number | null;      // current hip descent / leg length
  depthTargetFrac: number;       // pass threshold
  valgusWorse: number | null;    // current worse-side KFPPA (deg)
  pelvicFrac: number | null;     // current L/R hip-Y offset / hip span
  armWorstFrac: number | null;   // current worse-side wrist-above-shoulder / trunk
  armsDown: boolean;             // arms below the overhead gate right now
}

export class OverheadSquatLiveAnalyzer {
  private frameIndex = 0;
  private legLenProvisional = 0;
  private buf: Row[] = [];

  // Locked baseline
  private locked = false;
  private baseHipY = 0;
  private baseShoulderY = 0;
  private baseWristY: number | null = null;
  private baseLankX = 0;
  private baseRankX = 0;
  private baseHipSpan = 0;
  private legLen = 0;
  private trunkLen = 0;

  // Rep state machine
  private state: "standing" | "descending" = "standing";
  private descentStartMs: number | null = null;
  private descentStartFrame = 0;
  private bottomHipY: number | null = null;
  private bottomRow: Row | null = null;
  private reps: OverheadSquatRep[] = [];

  reset(): void {
    this.frameIndex = 0;
    this.legLenProvisional = 0;
    this.buf = [];
    this.locked = false;
    this.state = "standing";
    this.descentStartMs = null;
    this.bottomHipY = null;
    this.bottomRow = null;
    this.reps = [];
  }

  private extractRow(kp: LiveKeypoint[], tMs: number): Row | null {
    const need = [
      LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE,
      LM.LEFT_ANKLE, LM.RIGHT_ANKLE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    ];
    for (const idx of need) {
      const p = kp[idx];
      if (!p || (p.score ?? 0) < VIS) return null;
    }
    const g = (idx: number) => kp[idx];
    const lh = g(LM.LEFT_HIP), rh = g(LM.RIGHT_HIP);
    const lk = g(LM.LEFT_KNEE), rk = g(LM.RIGHT_KNEE);
    const la = g(LM.LEFT_ANKLE), ra = g(LM.RIGHT_ANKLE);
    const ls = g(LM.LEFT_SHOULDER), rs = g(LM.RIGHT_SHOULDER);
    const lw = g(LM.LEFT_WRIST), rw = g(LM.RIGHT_WRIST);
    return {
      tMs,
      frameIndex: this.frameIndex,
      hipY: (lh.y + rh.y) / 2,
      lhipX: lh.x, lhipY: lh.y, rhipX: rh.x, rhipY: rh.y,
      lknX: lk.x, lknY: lk.y, rknX: rk.x, rknY: rk.y,
      lankX: la.x, lankY: la.y, rankX: ra.x, rankY: ra.y,
      lshY: ls.y, rshY: rs.y, shoulderY: (ls.y + rs.y) / 2,
      lwrY: lw && (lw.score ?? 0) >= ARM_VIS ? lw.y : null,
      rwrY: rw && (rw.score ?? 0) >= ARM_VIS ? rw.y : null,
    };
  }

  private tryLockBaseline(): void {
    if (this.buf.length < STANDING_HOLD_SAMPLES) return;
    const window = this.buf.slice(-STANDING_HOLD_SAMPLES);
    const hys = window.map((r) => r.hipY);
    const med = median(hys);
    const maxDev = Math.max(...hys.map((y) => Math.abs(y - med)));
    const tol = STANDING_TOL_FRAC * Math.max(1, this.legLenProvisional);
    if (maxDev > tol) return;

    this.baseHipY = med;
    this.baseLankX = median(window.map((r) => r.lankX));
    this.baseRankX = median(window.map((r) => r.rankX));
    this.baseHipSpan = Math.max(
      1,
      Math.abs(median(window.map((r) => r.rhipX)) - median(window.map((r) => r.lhipX))),
    );
    this.baseShoulderY = median(window.map((r) => r.shoulderY));
    const ankY = median(window.map((r) => (r.lankY + r.rankY) / 2));
    this.legLen = Math.max(1, ankY - this.baseHipY);
    this.trunkLen = Math.max(1, this.baseHipY - this.baseShoulderY);
    const wrists: number[] = [];
    for (const r of window) {
      if (r.lwrY !== null) wrists.push(r.lwrY);
      if (r.rwrY !== null) wrists.push(r.rwrY);
    }
    this.baseWristY = wrists.length ? median(wrists) : null;
    this.locked = true;
  }

  /** Worse-side wrist-above-shoulder fraction of trunk length for a
   *  row (positive = wrist above shoulder). null when neither wrist is
   *  reliably visible. */
  private worstArmFracRow(row: Row): number | null {
    const shY = (row.lshY + row.rshY) / 2;
    const l = row.lwrY !== null ? (shY - row.lwrY) / this.trunkLen : null;
    const r = row.rwrY !== null ? (shY - row.rwrY) / this.trunkLen : null;
    const arms = [l, r].filter((x): x is number => x !== null);
    return arms.length ? Math.min(...arms) : null;
  }

  private closeRep(returnRow: Row): void {
    const b = this.bottomRow;
    if (!b || this.descentStartMs === null) return;
    const depthPx = Math.max(0, (this.bottomHipY ?? this.baseHipY) - this.baseHipY);
    const lK = kfppaDeg(b.lhipX, b.lhipY, b.lknX, b.lknY, b.lankX, b.lankY);
    const rK = kfppaDeg(b.rhipX, b.rhipY, b.rknX, b.rknY, b.rankX, b.rankY);
    const pelvicPx = Math.abs(b.lhipY - b.rhipY);
    const pelvicFrac = pelvicPx / this.baseHipSpan;
    const ankSpreadPx = Math.abs(b.rankX - b.lankX);
    const ankRatio = ankSpreadPx / this.baseHipSpan;
    const armFrac = (wr: number | null): number | null => {
      if (wr === null) return null;
      const shY = (b.lshY + b.rshY) / 2;
      return (shY - wr) / this.trunkLen; // positive = wrist above shoulder
    };
    const lArm = armFrac(b.lwrY);
    const rArm = armFrac(b.rwrY);
    const arms = [lArm, rArm].filter((x): x is number => x !== null);
    const worstArm = arms.length ? Math.min(...arms) : null;

    this.reps.push({
      rep_index: this.reps.length,
      descent_start_frame_index: this.descentStartFrame,
      bottom_frame_index: b.frameIndex,
      return_frame_index: returnRow.frameIndex,
      descent_start_t_ms: this.descentStartMs,
      bottom_t_ms: b.tMs,
      return_t_ms: returnRow.tMs,
      depth_px: depthPx,
      depth_cm: null, // filled at finalize when calibration is known
      depth_frac_of_leg: depthPx / this.legLen,
      bottom_kfppa_left_deg: lK,
      bottom_kfppa_right_deg: rK,
      bottom_kfppa_worse_deg: Math.max(lK, rK),
      bottom_pelvic_tilt_px: pelvicPx,
      bottom_pelvic_tilt_frac: pelvicFrac,
      bottom_ank_spread_px: ankSpreadPx,
      bottom_ank_spread_ratio: ankRatio,
      bottom_l_arm_overhead_frac: lArm,
      bottom_r_arm_overhead_frac: rArm,
      bottom_worst_arm_overhead_frac: worstArm,
    });
  }

  pushFrame(kp: LiveKeypoint[], tMs: number): OHSLiveFrame {
    this.frameIndex += 1;
    const row = this.extractRow(kp, tMs);
    if (!row) {
      return {
        tracking: false,
        baselineLocked: this.locked,
        baselineProgress: Math.min(1, this.buf.length / STANDING_HOLD_SAMPLES),
        repCount: this.reps.length,
        descending: this.state === "descending",
        depthFrac: null,
        depthTargetFrac: DEPTH_TARGET_FRAC_OF_LEG,
        valgusWorse: null,
        pelvicFrac: null,
        armWorstFrac: null,
        armsDown: false,
      };
    }

    // Running provisional leg length (for the baseline stability tol).
    const gap = Math.max(row.lankY, row.rankY) - row.hipY;
    if (gap > this.legLenProvisional) this.legLenProvisional = gap;

    if (!this.locked) {
      this.buf.push(row);
      if (this.buf.length > STANDING_HOLD_SAMPLES * 3) this.buf.shift();
      this.tryLockBaseline();
      return {
        tracking: true,
        baselineLocked: this.locked,
        baselineProgress: Math.min(1, this.buf.length / STANDING_HOLD_SAMPLES),
        repCount: 0,
        descending: false,
        depthFrac: null,
        depthTargetFrac: DEPTH_TARGET_FRAC_OF_LEG,
        valgusWorse: null,
        pelvicFrac: null,
        armWorstFrac: null,
        armsDown: false,
      };
    }

    // ── Locked: rep state machine ────────────────────────────────
    const drop = row.hipY - this.baseHipY; // +ve = descended
    const descentMinPx = this.legLen * DESCENT_MIN_FRAC_OF_LEG;
    const returnBandPx = this.legLen * STANDING_RETURN_BAND_FRAC;
    const worstArm = this.worstArmFracRow(row);
    const armsDown = worstArm !== null && worstArm < ARMS_UP_GATE;

    if (this.state === "standing") {
      // Gate: only start a rep when the arms are overhead at the top.
      // Arms clearly down → a plain squat, not an overhead squat →
      // ignore the descent. Null arm reading never blocks.
      if (drop >= descentMinPx && !armsDown) {
        this.state = "descending";
        this.descentStartMs = row.tMs;
        this.descentStartFrame = row.frameIndex;
        this.bottomHipY = row.hipY;
        this.bottomRow = row;
      }
    } else {
      // Track deepest point.
      if (this.bottomHipY === null || row.hipY > this.bottomHipY) {
        this.bottomHipY = row.hipY;
        this.bottomRow = row;
      }
      if (Math.abs(drop) <= returnBandPx) {
        const durMs =
          this.descentStartMs !== null ? row.tMs - this.descentStartMs : 0;
        if (durMs >= MIN_REP_DURATION_MS) this.closeRep(row);
        this.state = "standing";
        this.descentStartMs = null;
        this.bottomHipY = null;
        this.bottomRow = null;
      }
    }

    // ── Live per-frame metrics for the HUD ───────────────────────
    const curDepthFrac = Math.max(0, drop) / this.legLen;
    const curValgus = Math.max(
      kfppaDeg(row.lhipX, row.lhipY, row.lknX, row.lknY, row.lankX, row.lankY),
      kfppaDeg(row.rhipX, row.rhipY, row.rknX, row.rknY, row.rankX, row.rankY),
    );
    const curPelvic = Math.abs(row.lhipY - row.rhipY) / this.baseHipSpan;

    return {
      tracking: true,
      baselineLocked: true,
      baselineProgress: 1,
      repCount: this.reps.length,
      descending: this.state === "descending",
      depthFrac: curDepthFrac,
      depthTargetFrac: DEPTH_TARGET_FRAC_OF_LEG,
      valgusWorse: curValgus,
      pelvicFrac: curPelvic,
      armWorstFrac: worstArm,
      armsDown,
    };
  }

  hasBaseline(): boolean {
    return this.locked;
  }
  repCount(): number {
    return this.reps.length;
  }

  /** Build the final OverheadSquatResult (or null if no valid session).
   *  peakScreenshot is optional — live mode may not capture one. */
  finalize(
    calibration: CalibrationResult | null,
    patientHeightCm: number | null,
    durationSeconds: number,
    peakScreenshot: string | null = null,
  ): OverheadSquatResult | null {
    if (!this.locked || this.reps.length === 0) return null;

    const ppc =
      calibration && calibration.pixels_per_cm > 0
        ? calibration.pixels_per_cm
        : null;
    const pxToCm = (px: number): number | null => (ppc ? px / ppc : null);

    const reps = this.reps.map((r) => ({
      ...r,
      depth_cm: pxToCm(r.depth_px),
    }));

    const valgusWorst = reps.map((r) => r.bottom_kfppa_worse_deg);
    const pelvicFracs = reps.map((r) => r.bottom_pelvic_tilt_frac);
    const spreadRatios = reps.map((r) => r.bottom_ank_spread_ratio);
    const depthFracs = reps.map((r) => r.depth_frac_of_leg);
    const armWorst = reps
      .map((r) => r.bottom_worst_arm_overhead_frac)
      .filter((x): x is number => x !== null);

    const maxValgus = valgusWorst.length ? Math.max(...valgusWorst) : 0;
    const item1Fail = maxValgus > VALGUS_KFPPA_FAIL_DEG;

    const maxPelvic = pelvicFracs.length ? Math.max(...pelvicFracs) : 0;
    const item2Fail = maxPelvic > PELVIC_TILT_FAIL_FRAC_OF_HIPSPAN;

    const meanSpread = mean(spreadRatios);
    const item3Fail = meanSpread < FOOT_MIN_RATIO || meanSpread > FOOT_MAX_RATIO;

    let item4Fail = false;
    let minArm: number | null = null;
    let armDetail: string;
    if (armWorst.length) {
      minArm = Math.min(...armWorst);
      item4Fail = minArm < ARM_OVERHEAD_MIN_FRAC_OF_TRUNK;
      armDetail =
        `Worst wrist-above-shoulder gap ${(minArm * 100).toFixed(0)}% ` +
        `of trunk length (need ≥ ${Math.round(ARM_OVERHEAD_MIN_FRAC_OF_TRUNK * 100)}%)`;
    } else {
      armDetail =
        "Wrists not reliably visible for this session — arm position not " +
        "scored (raise camera or step back so wrists clear the top of frame).";
    }

    const maxDepthFrac = depthFracs.length ? Math.max(...depthFracs) : 0;
    const item5Fail = maxDepthFrac < DEPTH_TARGET_FRAC_OF_LEG;

    const measurableFails = [item1Fail, item2Fail, item3Fail, item4Fail, item5Fail]
      .filter(Boolean).length;
    const classification: OverheadSquatClassification =
      measurableFails <= 1 ? "good" : measurableFails <= 3 ? "moderate" : "poor";

    const checklist: OverheadSquatChecklistItem[] = [
      {
        index: 1,
        label: "Knees cave in at bottom (valgus)",
        status: item1Fail ? "fail" : "pass",
        detail: `Worst KFPPA ${maxValgus.toFixed(1)}° (threshold ${VALGUS_KFPPA_FAIL_DEG.toFixed(0)}°)`,
      },
      {
        index: 2,
        label: "Hip / pelvic drop asymmetry at bottom",
        status: item2Fail ? "fail" : "pass",
        detail: `Max L/R hip Y offset ${(maxPelvic * 100).toFixed(0)}% of hip span (threshold ${Math.round(PELVIC_TILT_FAIL_FRAC_OF_HIPSPAN * 100)}%)`,
      },
      {
        index: 3,
        label: "Foot placement not shoulder-width",
        status: item3Fail ? "fail" : "pass",
        detail: `Mean ankle spread ${meanSpread.toFixed(2)}× hip span (window ${FOOT_MIN_RATIO.toFixed(1)}–${FOOT_MAX_RATIO.toFixed(1)})`,
      },
      {
        index: 4,
        label: "Arms fall from overhead (wrist drops toward shoulder)",
        status: !armWorst.length ? "not_assessed" : item4Fail ? "fail" : "pass",
        detail: armDetail,
      },
      {
        index: 5,
        label: "Squat depth insufficient (proxy — frontal-view only)",
        status: item5Fail ? "fail" : "pass",
        detail: `Deepest rep hip descent ${(maxDepthFrac * 100).toFixed(0)}% of leg length (target ≥ ${Math.round(DEPTH_TARGET_FRAC_OF_LEG * 100)}%)`,
      },
      {
        index: 6,
        label: "Excessive torso forward lean",
        status: "not_assessed",
        detail: "Sagittal view required — a single frontal camera can't resolve trunk pitch.",
      },
      {
        index: 7,
        label: "Heels rise off the ground",
        status: "not_assessed",
        detail: "Feet close-up / sagittal view required — foot dorsiflexion signal at this scale is unreliable.",
      },
    ];

    const maxDepthPx = Math.max(...reps.map((r) => r.depth_px));

    return {
      patient_height_cm: patientHeightCm,
      calibration,
      baseline_hip_y_px: this.baseHipY,
      baseline_shoulder_y_px: this.baseShoulderY,
      baseline_wrist_y_px: this.baseWristY,
      baseline_ank_spread_px: Math.abs(this.baseRankX - this.baseLankX),
      baseline_hip_span_px: this.baseHipSpan,
      leg_length_px: this.legLen,
      trunk_length_px: this.trunkLen,
      reps,
      rep_count: reps.length,
      mean_depth_frac: mean(depthFracs),
      max_depth_frac: maxDepthFrac,
      max_depth_cm: pxToCm(maxDepthPx),
      mean_valgus_worse_deg: mean(valgusWorst),
      max_valgus_worse_deg: maxValgus,
      max_pelvic_tilt_frac: maxPelvic,
      mean_ank_spread_ratio: meanSpread,
      min_arm_overhead_frac: minArm,
      duration_seconds: durationSeconds,
      checklist,
      measurable_fails: measurableFails,
      classification,
      peak_screenshot_data_url: peakScreenshot,
      fps: null,
      total_frames: this.frameIndex,
      valid_frames: this.frameIndex,
      interpretation: null,
    };
  }
}
