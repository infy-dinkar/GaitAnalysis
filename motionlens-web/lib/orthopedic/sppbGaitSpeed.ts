// SPPB Component 2 — 4-meter walk gait-speed detection.
//
// Per the SPPB spec: patient walks 4 meters at usual pace, twice;
// the better (faster) of the two trials is used for scoring.
//
// Detection logic per the MotionLens TUG / SPPB spec:
//   - Track the average ankle X position (both MoveNet ankles).
//   - Walk start: ankle X displacement from a 1-second baseline
//     exceeds ~30 px for 5 consecutive frames.
//   - Walk end: motion stops — ankle X velocity below threshold
//     for >0.5 s — AND patient has traveled a meaningful distance.
//   - Manual fallback: operator can press "End walk" any time.
//
// No backend involved — analysis is per-frame in the browser using
// MoveNet keypoints (consistent with the rest of the SPPB battery).

import type { Keypoint } from "@tensorflow-models/pose-detection";
import { LM } from "@/lib/pose/landmarks";

const VIS_THRESHOLD = 0.3;

// ─── Spec values ──────────────────────────────────────────────
export const PATH_LENGTH_M = 4.0;
export const BASELINE_WINDOW_SEC = 1.0;
export const START_CONFIRM_FRAMES = 5;
export const STOP_HOLD_SEC = 0.5;
/** After detected start, require at least this much elapsed time
 *  before auto-end can fire. A 0.4 m/s walker (severe-limitation
 *  cutoff) takes 10 s to walk 4 m, so 2.5 s is plenty short to
 *  catch genuine stops without blowing up on hesitation. The
 *  previous value (1.2 s) regularly mis-fired on slow walkers — the
 *  SPPB target population. */
export const MIN_WALK_DURATION_SEC = 2.5;

// ─── Body-height-relative thresholds ──────────────────────────
// Replaced the old raw-px values (START_DISPLACEMENT_PX = 30,
// STOP_VELOCITY_PX_PER_SEC = 30). Pixel thresholds don't scale
// across camera distances: same patient closer to the camera =
// much larger pixel motion for identical real-world motion. Scale
// by body height (shoulder→ankle px) so the same logic works at
// any framing. C4 already uses this pattern for single-leg lift.

/** Walk-start trigger: ankle must displace from baseline by this
 *  fraction of body-height (shoulder→ankle px) for
 *  START_CONFIRM_FRAMES consecutive frames. 0.15 ≈ "ankle has
 *  moved ~one foot-length sideways", which is unambiguously a
 *  step. */
export const START_DISPLACEMENT_RATIO = 0.15;
/** Walk-stop trigger: ankle velocity below this fraction of body-
 *  height per second for STOP_HOLD_SEC counts as "motion stopped". */
export const STOP_VELOCITY_RATIO_PER_SEC = 0.15;
/** Fallback body-height in pixels when shoulder + ankle are not
 *  both visible at baseline time (e.g. mid-frame camera). Picked
 *  to roughly match a typical 2 m framing. */
export const BODY_HEIGHT_FALLBACK_PX = 300;

// ─── Per-frame helpers ───────────────────────────────────────

function visible(kp: Keypoint | undefined): boolean {
  return !!kp && (kp.score ?? 0) >= VIS_THRESHOLD;
}

/** Mean of L+R ankle X (pixels). Falls back to whichever ankle is
 *  visible if only one passes the threshold; returns null when
 *  neither is visible. */
export function computeAnkleMidX(keypoints: Keypoint[]): number | null {
  const la = keypoints[LM.LEFT_ANKLE];
  const ra = keypoints[LM.RIGHT_ANKLE];
  if (visible(la) && visible(ra)) return (la.x + ra.x) / 2;
  if (visible(la)) return la.x;
  if (visible(ra)) return ra.x;
  return null;
}

/** Body height proxy in pixels — shoulder-midpoint to ankle-midpoint
 *  vertical distance. Used to scale the displacement / velocity
 *  thresholds so they're camera-distance-independent. Falls back to
 *  one-sided when only one shoulder + ankle pair is visible. */
export function computeBodyHeightPx(keypoints: Keypoint[]): number | null {
  const lSh = keypoints[LM.LEFT_SHOULDER];
  const rSh = keypoints[LM.RIGHT_SHOULDER];
  const lA  = keypoints[LM.LEFT_ANKLE];
  const rA  = keypoints[LM.RIGHT_ANKLE];
  const shoulderY =
    visible(lSh) && visible(rSh) ? (lSh.y + rSh.y) / 2
    : visible(lSh) ? lSh.y
    : visible(rSh) ? rSh.y
    : null;
  const ankleY =
    visible(lA) && visible(rA) ? (lA.y + rA.y) / 2
    : visible(lA) ? lA.y
    : visible(rA) ? rA.y
    : null;
  if (shoulderY === null || ankleY === null) return null;
  const h = Math.abs(ankleY - shoulderY);
  return h > 1 ? h : null;
}

// ─── Trial state machine ──────────────────────────────────────
//
// Lifecycle:
//
//   "waiting_baseline" → keypoints accumulating, ankle X being
//                        averaged into the baseline window. Once
//                        BASELINE_WINDOW_SEC of data is collected,
//                        the baseline is locked.
//   "baselined"        → baseline locked; watching for movement.
//                        When ankle X has displaced by
//                        START_DISPLACEMENT_PX for
//                        START_CONFIRM_FRAMES consecutive frames,
//                        start time is recorded and we enter
//                        "walking".
//   "walking"          → tracking the walk. Detects "motion stopped"
//                        (velocity below STOP_VELOCITY for STOP_HOLD
//                        seconds, after MIN_WALK_DURATION_SEC has
//                        elapsed since start). When detected, end
//                        time is recorded and we enter "done".
//   "done"             → trial complete. Caller reads `duration_sec`.
//
// The operator can call `endManually()` at any time during "walking"
// to terminate the trial — the recorded duration is wall-clock from
// detected start to manual end.

export type TrialPhase =
  | "waiting_baseline"
  | "baselined"
  | "walking"
  | "done";

export interface GaitSpeedTrialState {
  phase: TrialPhase;
  /** Wall-clock ms when phase became "walking" (auto-detected start). */
  start_ms: number | null;
  /** Wall-clock ms when phase became "done" (end of trial). */
  end_ms: number | null;
  /** Locked baseline ankle-mid X position in pixels. */
  baseline_x: number | null;
  /** Locked baseline body-height in pixels (shoulder→ankle). Used
   *  to scale the displacement / velocity thresholds so the same
   *  detection logic works regardless of how close the camera is.
   *  Null until baseline locks. */
  baseline_body_h_px: number | null;
  /** Last ankle-mid X measurement (for displaying live distance). */
  last_x: number | null;
  /** Whether the trial completed via auto-detect (true) or operator
   *  manual end (false). False also when the trial timed out. */
  auto_completed: boolean;
}

export function newGaitTrialState(): GaitSpeedTrialState {
  return {
    phase: "waiting_baseline",
    start_ms: null,
    end_ms: null,
    baseline_x: null,
    baseline_body_h_px: null,
    last_x: null,
    auto_completed: false,
  };
}

interface TrialInternals {
  /** Per-frame ankle X readings used to compute the baseline. */
  baseline_samples: number[];
  /** Per-frame body-height readings during baseline (used to lock
   *  a stable scale; we take the median to reject outliers). */
  baseline_body_h_samples: number[];
  baseline_first_ms: number | null;
  /** Frames where ankle X is far enough from baseline to count as
   *  "started" — incremented when condition is met, reset to 0
   *  otherwise. */
  start_confirm_count: number;
  /** Last frame's ankle X — used to compute frame-to-frame velocity. */
  prev_x: number | null;
  prev_t_ms: number | null;
  /** Wall-clock ms when motion-stop window started accumulating. */
  stop_window_start_ms: number | null;
}

// We attach internals via a WeakMap so the public state stays clean
// and serialisable. Caller doesn't need to know about internals.
const _internals = new WeakMap<GaitSpeedTrialState, TrialInternals>();

function getInternals(state: GaitSpeedTrialState): TrialInternals {
  let i = _internals.get(state);
  if (!i) {
    i = {
      baseline_samples: [],
      baseline_body_h_samples: [],
      baseline_first_ms: null,
      start_confirm_count: 0,
      prev_x: null,
      prev_t_ms: null,
      stop_window_start_ms: null,
    };
    _internals.set(state, i);
  }
  return i;
}

/** Median helper — robust to a few low-visibility outliers in the
 *  baseline-collection window. */
function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return 0;
  return n % 2 === 1
    ? sorted[(n - 1) / 2]
    : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

/** Feed one MoveNet frame into the trial state. Mutates `state` in
 *  place and returns the (possibly-new) phase so the caller can
 *  react to transitions. Designed to be called from a rAF loop or
 *  per-frame onFrame callback. */
export function stepGaitTrial(
  state: GaitSpeedTrialState,
  keypoints: Keypoint[],
  now_ms: number,
): TrialPhase {
  if (state.phase === "done") return "done";

  const ankle_x = computeAnkleMidX(keypoints);
  if (ankle_x === null) return state.phase;
  state.last_x = ankle_x;

  const ints = getInternals(state);

  // ── waiting_baseline → baselined ──────────────────────────
  if (state.phase === "waiting_baseline") {
    if (ints.baseline_first_ms === null) ints.baseline_first_ms = now_ms;
    ints.baseline_samples.push(ankle_x);
    const body_h = computeBodyHeightPx(keypoints);
    if (body_h !== null) ints.baseline_body_h_samples.push(body_h);

    if (now_ms - ints.baseline_first_ms >= BASELINE_WINDOW_SEC * 1000) {
      // Lock the ankle baseline as the mean of the window.
      const sum = ints.baseline_samples.reduce((a, b) => a + b, 0);
      state.baseline_x = sum / ints.baseline_samples.length;
      // Lock body-height as the MEDIAN of the window (robust to
      // single-frame visibility drops where one shoulder or ankle
      // briefly went under threshold). Fall back to a sensible
      // default if shoulders/ankles never both became visible.
      state.baseline_body_h_px =
        ints.baseline_body_h_samples.length >= 3
          ? median(ints.baseline_body_h_samples)
          : BODY_HEIGHT_FALLBACK_PX;
      state.phase = "baselined";
    }
    return state.phase;
  }

  // From here on, the displacement and velocity thresholds are
  // body-height-relative. baseline_body_h_px is guaranteed non-null
  // once phase has advanced past waiting_baseline.
  const bodyH = state.baseline_body_h_px ?? BODY_HEIGHT_FALLBACK_PX;
  const startDisplacementPx = bodyH * START_DISPLACEMENT_RATIO;
  const stopVelocityPxPerSec = bodyH * STOP_VELOCITY_RATIO_PER_SEC;

  // ── baselined → walking ──────────────────────────────────
  if (state.phase === "baselined" && state.baseline_x !== null) {
    const displacement = Math.abs(ankle_x - state.baseline_x);
    if (displacement >= startDisplacementPx) {
      ints.start_confirm_count += 1;
      if (ints.start_confirm_count >= START_CONFIRM_FRAMES) {
        // Reset velocity tracker for the walking phase.
        ints.prev_x = ankle_x;
        ints.prev_t_ms = now_ms;
        ints.stop_window_start_ms = null;
        state.start_ms = now_ms;
        state.phase = "walking";
      }
    } else {
      ints.start_confirm_count = 0;
    }
    return state.phase;
  }

  // ── walking → done ───────────────────────────────────────
  if (state.phase === "walking" && state.start_ms !== null) {
    const elapsed_sec = (now_ms - state.start_ms) / 1000;
    if (ints.prev_x !== null && ints.prev_t_ms !== null) {
      const dt_ms = now_ms - ints.prev_t_ms;
      if (dt_ms > 0) {
        const vel = (Math.abs(ankle_x - ints.prev_x) / dt_ms) * 1000; // px/sec
        if (
          vel < stopVelocityPxPerSec &&
          elapsed_sec > MIN_WALK_DURATION_SEC
        ) {
          if (ints.stop_window_start_ms === null) {
            ints.stop_window_start_ms = now_ms;
          } else if (now_ms - ints.stop_window_start_ms >= STOP_HOLD_SEC * 1000) {
            state.end_ms = now_ms;
            state.phase = "done";
            state.auto_completed = true;
            return state.phase;
          }
        } else {
          ints.stop_window_start_ms = null;
        }
      }
    }
    ints.prev_x = ankle_x;
    ints.prev_t_ms = now_ms;
  }

  return state.phase;
}

/** Operator manual end — used when auto-detection is uncertain. The
 *  recorded duration is start_ms → now. Safe to call from any phase;
 *  if start_ms is null (patient never started moving), the trial
 *  ends as uncompleted. */
export function endGaitTrialManually(
  state: GaitSpeedTrialState,
  now_ms: number,
): void {
  if (state.phase === "done") return;
  state.end_ms = now_ms;
  state.auto_completed = false;
  state.phase = "done";
}

/** Trial duration in seconds (null if the patient never started). */
export function trialDurationSec(state: GaitSpeedTrialState): number | null {
  if (state.start_ms === null || state.end_ms === null) return null;
  return (state.end_ms - state.start_ms) / 1000;
}
