"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  Camera,
  CheckCircle2,
  Eye,
  Lock,
  RotateCcw,
} from "lucide-react";
import { LiveBiomechCamera } from "@/components/biomech/LiveBiomechCamera";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { AutoSaveToast } from "@/components/dashboard/AutoSaveToast";
import { LiveModeLayout } from "@/components/live/LiveModeLayout";
import {
  AutoFlowCountdownCard,
  AutoFlowCountdownOverlay,
} from "@/components/rehab/mechanics/AutoFlowChrome";
import { useRehabAutoFlow } from "@/lib/rehab/useAutoFlow";
import { Button } from "@/components/ui/Button";
import { usePatientContext } from "@/hooks/usePatientContext";
import { fmt } from "@/lib/utils";
import { getInstructions } from "@/lib/biomech/instructions";
import { resolveMovement } from "@/lib/biomech/movements";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import {
  captureNeckRotationBaseline,
  detectNeckFlexExtDirection,
  detectNeckLateralDirection,
  isStableFacingForward,
  type NeckRotationCalibration,
} from "@/lib/biomech/neck-live";
import {
  captureShoulderRotationBaseline,
  detectShoulderAbAdDirection,
  detectShoulderFlexExtDirection,
  detectShoulderRotationDirection,
  isShoulderRotationNeutral,
  ShoulderAbAdCompensationTracker,
  ShoulderFlexExtCompensationTracker,
  ShoulderRotationCompensationTracker,
  type Compensation,
  type ShoulderRotationCalibration,
} from "@/lib/biomech/shoulder-live";
import {
  NeckFlexExtCompensationTracker,
  NeckLateralFlexionCompensationTracker,
  NeckRotationCompensationTracker,
} from "@/lib/biomech/neck-live";
import type { BiomechCompensationDTO } from "@/lib/api";
import {
  detectHipRotationDirection,
  HipFlexionCompensationTracker,
  HipExtensionCompensationTracker,
  HipRotationCompensationTracker,
} from "@/lib/biomech/hip-live";
import { KneeFlexExtCompensationTracker } from "@/lib/biomech/knee-live";
import {
  AnkleFlexionCompensationTracker,
  AnkleExtensionCompensationTracker,
} from "@/lib/biomech/ankle-live";
import type { LiveKeypoint as Keypoint } from "@/hooks/usePoseDetectionLive";
import type { LiveBiomechFrameDataDTO } from "@/lib/api";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";

type PostureStatus = "idle" | "good" | "low_visibility" | "no_landmarks";

// Auto-calibration timing for neck rotation. The patient has to be
// stably facing the camera for this long continuously before the
// baseline is auto-locked. Operator can also click "Lock baseline"
// at any time to skip the auto-detect.
const CALIBRATION_STABLE_MS = 1500;

// ── Peak-screenshot stability guards ──────────────────────────
// Without these, the report's "peak ROM" thumbnail can end up showing
// a noisy frame near recording end rather than the real peak. Three
// issues conspire against the naive "capture on every new max" rule:
//   1. Angle formulas (esp. shoulder/neck rotation projected vectors)
//      saturate or jump near 90° when the limb crosses the camera
//      plane — so a phantom peak fires as the patient lowers their
//      arm at the end.
//   2. MoveNet keypoint scores collapse when the patient disengages
//      (steps away, drops the limb out of frame). Garbage-angle frames
//      from these moments can exceed the legitimate peak.
//   3. A single-frame spike was enough to overwrite the screenshot.
// Guards:
//   • PEAK_VIS_GATE — every primary joint must be at or above this
//     visibility score for the frame to be considered.
//   • PEAK_MAX_JUMP_DEG — single-frame angle jumps larger than this
//     are physically implausible (limbs can't rotate that fast at
//     30-60 FPS) and almost always indicate a formula glitch or a
//     keypoint switch — those frames are dropped from peak tracking.
//   • PEAK_HOLD_FRAMES + PEAK_HOLD_BAND_DEG — a new candidate peak
//     must be held (within band) for N frames before it commits to
//     peakSigned + peakUrl. Pure single-frame spikes can't reach the
//     hold count and are filtered out cleanly.
const PEAK_VIS_GATE = 0.55;
const PEAK_MAX_JUMP_DEG = 25.0;
const PEAK_HOLD_FRAMES = 5;
const PEAK_HOLD_BAND_DEG = 3.0;

// ── Rep-cycle auto-complete ────────────────────────────────────
// The session ends itself ("Show Analysis" auto-fires) once the
// patient has performed TARGET_REP_COUNT full movement cycles —
// neutral → raised past the rep threshold → back near neutral. For
// merged direction-routed tests both directions need the count.
//
// Thresholds are BASELINE-RELATIVE, not absolute: several movement
// formulas don't read 0° at rest (ankle neutral sits well above 0,
// signed formulas idle at an offset), so an absolute "return below
// 8°" would never trigger. Instead the per-slot session minimum
// |angle| is tracked as the neutral baseline and:
//   rise    = max(12°, 40% of (target lower bound − baseline))
//   raiseAt = baseline + rise
//   returnAt= baseline + 35% of rise  (must come ~2/3 back down)
// Small sways never span `rise`, so idle standing can't count.
const TARGET_REP_COUNT = 5;
const REP_RISE_FRAC = 0.4;
const REP_RISE_MIN_DEG = 12;
const REP_RETURN_FRAC = 0.35;

// Joints whose visibility actually drives the angle formula for the
// given (body part, movement) pair. We gate peak capture on the MIN
// visibility across these — if even one is below threshold, the angle
// reading isn't trustworthy enough to overwrite the peak screenshot.
//
// IMPORTANT: only include joints the formula *actually uses*. Earlier
// versions of this gate over-asked (e.g. demanded wrist visibility for
// shoulder extension, where the formula reads shoulder + elbow + hip
// only). At extreme ROM the unused joint (wrist behind body for arm
// extension, ankle out of frame for hip extension) drops below the
// gate threshold even though the formula is happily producing valid
// readings — and the peak update gets refused. Match the gate to the
// formula's actual inputs and the problem goes away.
function relevantJointIndices(
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle",
  movementId: string,
  side: "left" | "right" | undefined,
): number[] {
  const s = side === "left" ? "LEFT" : "RIGHT";
  const L = LM as unknown as Record<string, number>;
  switch (bodyPart) {
    case "shoulder": {
      // Rotation tests use wrist (forearm direction = wrist - elbow).
      // Everything else (flex/ext/abd/add) uses shoulder + elbow + hip
      // — wrist is irrelevant and shouldn't gate the peak.
      const isRotation =
        movementId === "external_rotation" || movementId === "internal_rotation";
      if (isRotation) {
        return [L[`${s}_SHOULDER`], L[`${s}_ELBOW`], L[`${s}_WRIST`]];
      }
      return [L[`${s}_SHOULDER`], L[`${s}_ELBOW`], L[`${s}_HIP`]];
    }
    case "neck":
      // Only the reliably-visible joints — nose + both shoulders —
      // gate the peak update. Ears are USED by the neck formulas
      // (lateral flexion, flex/ext, rotation) but their per-frame
      // visibility drops at extreme tilts (the tilt-side ear can
      // dip below the peak-gate threshold even though it's still
      // detected well above the formula's own VIS_THRESHOLD=0.15).
      // Including them in the peak gate stalled peak updates at
      // the very moment the patient hit their maximum tilt. The
      // formula's internal visibility check already ensures ears
      // are detected at the basic confidence level the math needs.
      return [LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER];
    case "knee":
      // Knee angle = hip-knee-ankle on the test side, all 3 needed.
      return [L[`${s}_HIP`], L[`${s}_KNEE`], L[`${s}_ANKLE`]];
    case "hip": {
      // Flex/ext use trunk (shoulder-hip) vs thigh (hip-knee); ankle
      // not used. Rotation tests use shin (knee-ankle) so they need
      // ankle but not shoulder.
      const isRotation =
        movementId === "external_rotation" || movementId === "internal_rotation";
      if (isRotation) {
        return [L[`${s}_HIP`], L[`${s}_KNEE`], L[`${s}_ANKLE`]];
      }
      return [L[`${s}_SHOULDER`], L[`${s}_HIP`], L[`${s}_KNEE`]];
    }
    case "ankle":
      return [L[`${s}_KNEE`], L[`${s}_ANKLE`]];
  }
}

// "Framing core" — the PROXIMAL joints that are reliably visible
// whenever the patient is properly positioned (standing ~2 m back,
// full limb in frame). Deliberately excludes joints that legitimately
// drop to low confidence during the movement itself (wrist when the
// forearm points at the camera for rotation, ankle when the leg
// swings behind the body for hip extension). If even the framing
// core isn't visible, it's a genuine setup problem — too close, out
// of frame, or bad lighting — not a mid-movement occlusion. This
// drives the actionable on-screen guidance.
function framingCoreIndices(
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle",
  side: "left" | "right" | undefined,
): number[] {
  const s = side === "left" ? "LEFT" : "RIGHT";
  const L = LM as unknown as Record<string, number>;
  switch (bodyPart) {
    case "shoulder":
      return [L[`${s}_SHOULDER`], L[`${s}_ELBOW`]];
    case "neck":
      return [LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER];
    case "knee":
      return [L[`${s}_HIP`], L[`${s}_KNEE`]];
    case "hip":
      return [L[`${s}_HIP`], L[`${s}_KNEE`]];
    case "ankle":
      return [L[`${s}_KNEE`], L[`${s}_ANKLE`]];
  }
}

// Human-readable limb description for the "part of X out of frame"
// guidance — keeps the message specific to the test being run.
const LIMB_FRAMING_TEXT: Record<
  "shoulder" | "neck" | "knee" | "hip" | "ankle",
  string
> = {
  shoulder: "your whole arm (shoulder to wrist)",
  neck: "your head and both shoulders",
  knee: "your hip, knee and ankle",
  hip: "your whole leg (hip to ankle)",
  ankle: "your knee and ankle",
};

// A joint is considered "in frame at all" above this score. MoveNet
// hallucinates out-of-frame joints with very low scores (~0.1), so
// 0.25 cleanly separates "actually visible" from "not in shot".
const FRAMING_VIS_FLOOR = 0.25;

interface LiveAssessmentProps {
  bodyPart: "shoulder" | "neck" | "knee" | "hip" | "ankle";
  movementId: string;
  /** Full title shown on the live screen, e.g. "Neck · Lateral Flexion". */
  movementLabel: string;
  /** Bare movement name for the report, e.g. "Lateral Flexion". */
  movementName?: string;
  description: string;
  target: [number, number];
  side?: "left" | "right";
  /** Merged movements bundle two directions in one recording session
   *  (rotation = external + internal, abduction_adduction = both). When
   *  set, the live engine detects direction per frame and tracks a
   *  separate peak for each. */
  merged?: boolean;
  /** Display label for the primary (target) direction in merged mode.
   *  Example: "External Rotation". */
  primaryLabel?: string;
  /** Display label for the secondary direction in merged mode.
   *  Example: "Internal Rotation". */
  secondaryLabel?: string;
  /** Normal range for the secondary direction. Required when merged. */
  secondaryTarget?: [number, number];
  /** Auto Mode: skip the "Start Assessment" intro and enter the
   *  fullscreen capture shell immediately on mount (the sequence
   *  runner already provided the user gesture + get-ready count). */
  autoEnter?: boolean;
  /** Auto Mode: called once when the session ends and the report
   *  view mounts (auto rep-complete, manual Show Analysis, or a
   *  forced completion) — lets the sequence runner schedule the
   *  next step. */
  onCompleted?: () => void;
  /** Auto Mode safety timeout: bump this number to force the
   *  session to complete NOW with whatever peaks were captured.
   *  No-op when no peak has committed yet (nothing to report) or
   *  when the report is already showing. */
  forceCompleteSignal?: number;
  /** Auto Mode: when set, the report view's "Try again" button
   *  delegates to the sequence runner (which discards this
   *  attempt's saved report + remounts the step fresh) instead of
   *  the in-place resetPeak. */
  onRetest?: () => void;
  /** Auto Mode: forwarded to the report view's AutoSaveToast —
   *  notified with the saved report id so a retest can delete the
   *  superseded attempt. */
  onSaved?: (reportId: string) => void;
}

/**
 * Continuous-capture biomech live assessment driven by the FastAPI
 * Python backend (same MediaPipe pose detector + same engine math the
 * Streamlit app uses). Per-frame state lives in a ref + 10 Hz tick to
 * keep the readouts smooth without React state churn.
 */
export function LiveAssessment({
  bodyPart,
  movementId,
  movementLabel,
  movementName,
  description,
  target,
  side,
  merged,
  primaryLabel,
  secondaryLabel,
  secondaryTarget,
  autoEnter = false,
  onCompleted,
  forceCompleteSignal,
  onRetest,
  onSaved,
}: LiveAssessmentProps) {
  const reportName = movementName ?? movementLabel.split(" · ").pop() ?? movementLabel;
  // Calibration-phase movements: rotation tests where 2D pose
  // estimation can't measure 3D rotation directly. We capture a
  // per-patient zero-degree reference at neutral, then only start
  // logging the peak angle once that baseline is locked. This avoids
  // the legacy formulas' two failure modes (anatomical asymmetry
  // baked in as fake rotation, and projected-vector saturation at
  // large angles).
  const isNeckRotation = bodyPart === "neck" && movementId === "rotation";
  // Shoulder rotation tests — covers the legacy single-direction IDs
  // AND the merged "rotation" ID (which is essentially both directions
  // in one trial; the same baseline-locked formula applies).
  const isShoulderRotation =
    bodyPart === "shoulder" &&
    (movementId === "external_rotation" ||
      movementId === "internal_rotation" ||
      movementId === "rotation");
  const isCalibratedRotation = isNeckRotation || isShoulderRotation;

  // Merged movements bundle two directions in one recording session.
  // The state machine grows a secondary peak slot, the UI renders a
  // dual readout, and the report receives two measured values.
  //
  // Shoulder merged tests route per-frame by spatial direction (see
  // detectShoulderRotationDirection / detectShoulderAbAdDirection).
  // Knee merged (flexion + extension) is different: the angle metric
  // itself is bidirectional — max(angle) = peak flexion, min(angle)
  // = peak extension (residual flexion at the straightest position).
  // No direction routing needed; primary slot tracks the MAX, the
  // secondary slot tracks the MIN over the trial.
  const isMergedShoulderRotation =
    !!merged && bodyPart === "shoulder" && movementId === "rotation";
  const isMergedShoulderAbAd =
    !!merged && bodyPart === "shoulder" && movementId === "abduction_adduction";
  const isMergedShoulderFlexExt =
    !!merged && bodyPart === "shoulder" && movementId === "flexion_extension";
  const isMergedKneeFE =
    !!merged && bodyPart === "knee" && movementId === "flexion_extension";
  const isMergedNeckFE =
    !!merged && bodyPart === "neck" && movementId === "flexion_extension";
  const isMergedNeckLateral =
    !!merged && bodyPart === "neck" && movementId === "lateral_flexion";
  const isMergedHipRotation =
    !!merged && bodyPart === "hip" && movementId === "rotation";
  const isMergedMovement =
    isMergedShoulderRotation ||
    isMergedShoulderAbAd ||
    isMergedShoulderFlexExt ||
    isMergedKneeFE ||
    isMergedNeckFE ||
    isMergedNeckLateral ||
    isMergedHipRotation;

  // Single-direction movements that ALSO run compensation tracking
  // (hip flex/ext, ankle flex/ext). These don't share the merged
  // direction-routing path but they still need their tracker to
  // receive every valid frame and emit compensations.
  const isHipFlexion = bodyPart === "hip" && movementId === "flexion";
  const isHipExtension = bodyPart === "hip" && movementId === "extension";
  const isAnkleFlexion = bodyPart === "ankle" && movementId === "flexion";
  const isAnkleExtension = bodyPart === "ankle" && movementId === "extension";

  const stateRef = useRef({
    current: null as number | null,
    peakSigned: null as number | null,
    validFrames: 0,
    totalFrames: 0,
    status: "idle" as PostureStatus,
    apiError: null as string | null,
    // Calibration substate — only meaningful when isNeckRotation. When
    // baseline is null we are still in the "calibrating" phase; once
    // it's set the loop transitions to "measuring" and the math layer
    // routes to the baseline-aware formula.
    calibStableSinceMs: null as number | null,
    calibFacingForward: false as boolean,
    // Held-peak tracking. A candidate is the largest |angle| seen so
    // far that's eligible for peak (passed the visibility + delta
    // gates). It only commits to `peakSigned` + `keyFramesRef.peakUrl`
    // once it has been held within ±PEAK_HOLD_BAND_DEG for at least
    // PEAK_HOLD_FRAMES frames — this filters out single-frame angle
    // spikes from formula saturation or keypoint dropouts.
    peakCandidateSigned: null as number | null,
    peakCandidateHeld: 0,
    peakCandidateUrl: null as string | null,
    prevAngleForDelta: null as number | null,
    // Actionable framing guidance shown under the status line. Null
    // when the subject is correctly framed; otherwise a specific
    // sentence telling the operator exactly what to fix.
    postureHint: null as string | null,
    // ── Secondary direction slot (used only when isMergedMovement) ─
    // Mirrors the primary peak fields above. The primary slot tracks
    // the test's "A" direction (external rotation, abduction); this
    // slot tracks "B" (internal rotation, adduction). Both peaks are
    // independently held-and-confirmed by the same candidate state
    // machine, just on different fields. For non-merged movements
    // these remain at their initial values (peakSignedB stays null).
    peakSignedB: null as number | null,
    peakCandidateSignedB: null as number | null,
    peakCandidateHeldB: 0,
    peakCandidateUrlB: null as string | null,
    prevAngleForDeltaB: null as number | null,
    // Last detected direction (for the live "Current" readout tag in
    // merged mode). null = inside the deadband / undetermined.
    currentDirection: null as "primary" | "secondary" | null,
    // Compensations currently active on the latest frame. Populated
    // by whichever per-test tracker is active (shoulder flex+ext,
    // ab+ad, rotation; neck flex+ext, lateral_flexion, rotation —
    // any movement that runs a tracker). Read by the live banner.
    // Typed as the widest cross-joint DTO so both shoulder's
    // narrower Compensation interface and neck's narrower one are
    // assignment-compatible here.
    currentCompensations: [] as BiomechCompensationDTO[],
    // ── Rep-cycle counter (auto-complete) ──────────────────────
    // One rep = neutral → past the raise threshold → back below the
    // return threshold. Per-slot so merged direction-routed tests
    // count each direction independently. Session auto-fires
    // showAnalysis at TARGET_REP_COUNT (see the per-frame handler).
    repsPrimary: 0,
    repsSecondary: 0,
    repPhasePrimary: "neutral" as "neutral" | "raised",
    repPhaseSecondary: "neutral" as "neutral" | "raised",
    // Session-minimum |angle| per slot — the movement formula's
    // actual neutral reading (ankle etc. don't rest at 0°). Rep
    // thresholds are computed relative to this.
    repBaselineMag: null as number | null,
    repBaselineMagB: null as number | null,
  });

  // Compensatory-movement tracker — scoped to shoulder flexion+
  // extension. Lazy-initialised on first frame inside onResult so the
  // side prop is locked at construction. Null for all other (bodyPart,
  // movementId) pairs.
  const compTrackerRef = useRef<ShoulderFlexExtCompensationTracker | null>(null);
  // Companion trackers for the other two merged shoulder tests. Each
  // is lazy-initialised the first time its movement's routing branch
  // sees a valid frame. Only one of the three trackers is ever non-
  // null for a given trial because only one merged movement is active.
  const compAbAdTrackerRef = useRef<ShoulderAbAdCompensationTracker | null>(null);
  const compRotationTrackerRef = useRef<ShoulderRotationCompensationTracker | null>(null);
  // Neck compensation trackers — one per merged neck test. Same
  // lazy-init pattern; only one is ever populated per trial.
  const compNeckFlexExtTrackerRef = useRef<NeckFlexExtCompensationTracker | null>(null);
  const compNeckLateralTrackerRef = useRef<NeckLateralFlexionCompensationTracker | null>(null);
  const compNeckRotationTrackerRef = useRef<NeckRotationCompensationTracker | null>(null);
  // Knee + hip + ankle compensation trackers — lazy-init per the
  // same pattern. Only one is non-null per trial (only one exercise
  // active at a time across knee/hip/ankle).
  const compKneeFETrackerRef = useRef<KneeFlexExtCompensationTracker | null>(null);
  const compHipFlexionTrackerRef = useRef<HipFlexionCompensationTracker | null>(null);
  const compHipExtensionTrackerRef = useRef<HipExtensionCompensationTracker | null>(null);
  const compHipRotationTrackerRef = useRef<HipRotationCompensationTracker | null>(null);
  const compAnkleFlexionTrackerRef = useRef<AnkleFlexionCompensationTracker | null>(null);
  const compAnkleExtensionTrackerRef = useRef<AnkleExtensionCompensationTracker | null>(null);

  // Annotated-frame screenshots for the report. Captured on the
  // composite canvas exposed by LiveBiomechCamera (via
  // window.__biomechCapture).
  //   - neutralUrl: first usable frame after movement starts → the
  //     patient's neutral / starting position.
  //   - peakUrl:    re-captured every time a new peak ROM is reached.
  // Both reset on "Reset Peak" so a re-attempt starts fresh.
  const keyFramesRef = useRef<{
    neutralUrl: string | null;
    /** Primary-direction peak screenshot (also the only peak for
     *  single-direction tests). */
    peakUrl: string | null;
    /** Secondary-direction peak screenshot (merged movements only). */
    peakUrlB: string | null;
  }>({
    neutralUrl: null,
    peakUrl: null,
    peakUrlB: null,
  });

  const grabBiomechFrame = useCallback((): string | null => {
    if (typeof window === "undefined") return null;
    const g = (window as unknown as {
      __biomechCapture?: () => string | null;
    }).__biomechCapture;
    return g ? g() ?? null : null;
  }, []);

  const [baseline, setBaseline] = useState<NeckRotationCalibration | null>(null);
  const [shoulderBaseline, setShoulderBaseline] =
    useState<ShoulderRotationCalibration | null>(null);
  const [, setVersion] = useState(0);
  const [showResult, setShowResult] = useState(false);

  // ── Auto-flow (fullscreen less-click live mode) ────────────────
  // One click ("Start Assessment") opens the fullscreen shell; the
  // camera auto-starts; once frames are flowing a 3-2-1 countdown
  // runs and capture arms by itself. For rotation tests the countdown
  // leads into the existing calibration step (baseline lock first,
  // then measurement) — the sequence is unchanged, only the entry is
  // automatic. "Show Analysis" still ends the session in the report
  // view, which auto-saves in the doctor flow.
  const [liveFullscreen, setLiveFullscreen] = useState<boolean>(false);
  const [camActive, setCamActive] = useState<boolean>(false);
  // Capture is armed only after the countdown completes. Gates peak
  // tracking, compensation-tracker feeds, screenshots, and the
  // rotation-calibration auto-lock so countdown-phase frames never
  // leak into the trial. A ref (not state) — read inside the
  // per-frame callbacks without re-binding them.
  const armedRef = useRef(false);
  // Rep-count auto-complete fires showAnalysis exactly once per
  // session; reset wherever a fresh attempt starts.
  const autoCompleteFiredRef = useRef(false);

  // Single source of truth for "calibration done" — used by gating
  // logic so the same code path handles both neck and shoulder.
  const calibrationLocked = isNeckRotation
    ? !!baseline
    : isShoulderRotation
      ? !!shoulderBaseline
      : true;

  // Doctor-flow context (no-op when accessed publicly). Live sessions
  // auto-save via the AutoSaveToast inside the report view.
  const { isDoctorFlow, patient } = usePatientContext();

  // 10 Hz UI sync — pulls latest values from the ref.
  useEffect(() => {
    const id = setInterval(() => setVersion((v) => v + 1), 100);
    return () => clearInterval(id);
  }, []);

  const onResult = useCallback((data: LiveBiomechFrameDataDTO | null) => {
    const s = stateRef.current;
    s.totalFrames += 1;
    if (!data) {
      s.status = "no_landmarks";
      s.current = null;
      s.prevAngleForDelta = null;
      s.postureHint =
        "No subject detected. Stand ~2 m (6 ft) from the camera, full upper body in frame, with good lighting.";
      return;
    }
    s.status = data.status as PostureStatus;
    s.apiError = null;

    // ── Actionable framing guidance ─────────────────────────
    // Runs every frame regardless of phase so the operator always
    // knows exactly why capture isn't working (the generic
    // "landmarks below threshold" status is too vague to act on).
    {
      const core = framingCoreIndices(bodyPart, side);
      let visibleCore = 0;
      for (const i of core) {
        if ((data.landmarks[i]?.visibility ?? 0) >= FRAMING_VIS_FLOOR) {
          visibleCore += 1;
        }
      }
      if (visibleCore === 0) {
        s.postureHint =
          "Subject not in frame. Stand ~2 m (6 ft) back, facing the camera, full upper body visible.";
      } else if (visibleCore < core.length) {
        s.postureHint = `Step back to ~2 m so ${LIMB_FRAMING_TEXT[bodyPart]} stays fully in frame — you're too close or partly out of shot.`;
      } else if (data.status !== "good") {
        s.postureHint =
          "Subject framed but the reading is unstable — improve lighting and face the camera squarely.";
      } else {
        s.postureHint = null;
      }
    }
    if (data.status !== "good" || data.current_angle === null) {
      // Lost the subject — clear delta tracking so when frames resume
      // we don't fire a spurious jump-rejection on the first one.
      s.current = null;
      s.prevAngleForDelta = null;
      s.prevAngleForDeltaB = null;
      s.currentDirection = null;
      return;
    }

    // ── Auto-flow arming gate ───────────────────────────────
    // Until the fullscreen countdown finishes, frames drive the
    // framing/status UI (and the live Current readout) only — no
    // peak tracking, no compensation-tracker feeds, no screenshots.
    // The measurement trail starts clean at the countdown→live
    // transition (see the useRehabAutoFlow onLive callback).
    if (!armedRef.current) {
      s.current = data.current_angle;
      s.prevAngleForDelta = null;
      s.prevAngleForDeltaB = null;
      s.currentDirection = null;
      return;
    }

    // Suppress angle / peak tracking during the calibration phase
    // for rotation tests. Until baseline is locked, the camera is
    // running the legacy fallback formula whose readings are
    // meaningless (and can saturate well past the anatomical max)
    // — surfacing them as Current / Peak pollutes the trial's
    // actual peak measurement.
    const inCalibration = isCalibratedRotation && !calibrationLocked;
    if (inCalibration) {
      s.current = null;
      s.prevAngleForDelta = null;
      s.prevAngleForDeltaB = null;
      s.currentDirection = null;
      return;
    }

    const angle = data.current_angle;
    s.current = angle;

    // ── Rep-cycle helpers (auto-complete) ───────────────────
    // Baseline-relative cycle detection — see the constants block
    // for the threshold rationale. `stepRep` advances one slot's
    // neutral↔raised machine; `maybeAutoComplete` fires
    // showAnalysis once the rep target(s) are met AND the peaks
    // the report needs have committed. Runs BEFORE the peak
    // visibility/delta gates on purpose: counting a rep needs a
    // "good" status frame, not peak-grade landmark confidence.
    const stepRep = (sl: "primary" | "secondary") => {
      const mag = Math.abs(angle);
      const baseKey = sl === "primary" ? "repBaselineMag" : "repBaselineMagB";
      const phaseKey =
        sl === "primary" ? "repPhasePrimary" : "repPhaseSecondary";
      const repsKey = sl === "primary" ? "repsPrimary" : "repsSecondary";
      const base =
        s[baseKey] === null ? mag : Math.min(s[baseKey] as number, mag);
      s[baseKey] = base;
      const lo =
        sl === "primary" ? target[0] : (secondaryTarget?.[0] ?? target[0]);
      const rise = Math.max(
        REP_RISE_MIN_DEG,
        Math.max(0, lo - base) * REP_RISE_FRAC,
      );
      const raiseAt = base + rise;
      const returnAt = base + rise * REP_RETURN_FRAC;
      if (s[phaseKey] === "neutral" && mag >= raiseAt) {
        s[phaseKey] = "raised";
      } else if (s[phaseKey] === "raised" && mag <= returnAt) {
        s[phaseKey] = "neutral";
        s[repsKey] += 1;
      }
    };
    const maybeAutoComplete = () => {
      const needBoth = isMergedMovement && !isMergedKneeFE;
      const repsDone =
        s.repsPrimary >= TARGET_REP_COUNT
        && (!needBoth || s.repsSecondary >= TARGET_REP_COUNT);
      const peaksReady =
        s.peakSigned !== null && (!needBoth || s.peakSignedB !== null);
      if (repsDone && peaksReady && !autoCompleteFiredRef.current) {
        autoCompleteFiredRef.current = true;
        showAnalysis();
      }
    };

    // ── Direction routing (merged movements only) ──────────
    // Shoulder merged tests route per-frame by spatial direction. Knee
    // merged (flexion + extension) doesn't route — its primary slot
    // tracks the running MAX (peak flexion) and the secondary slot
    // tracks the running MIN (peak extension / residual flexion),
    // both updated every valid frame. Direction is decided spatially
    // for shoulder, temporally (angle rising vs falling) for knee.
    let slot: "primary" | "secondary" = "primary";
    if (
      isMergedShoulderRotation ||
      isMergedShoulderAbAd ||
      isMergedShoulderFlexExt ||
      isMergedNeckFE ||
      isMergedNeckLateral ||
      isMergedHipRotation
    ) {
      const kpsForDir: Keypoint[] = data.landmarks.map((l) => ({
        x: l.x,
        y: l.y,
        score: l.visibility,
      }));
      const sideOrRight = side ?? "right";
      let dir: "primary" | "secondary" | null = null;
      if (isMergedShoulderRotation) {
        const r = detectShoulderRotationDirection(kpsForDir, sideOrRight);
        if (r === "external") dir = "primary";
        else if (r === "internal") dir = "secondary";
        // Rotation compensation tracker — same lazy-init pattern as
        // the flex+ext tracker. Feeds independently of the direction
        // state machine above.
        if (!compRotationTrackerRef.current) {
          compRotationTrackerRef.current = new ShoulderRotationCompensationTracker(sideOrRight);
        }
        compRotationTrackerRef.current.feed(kpsForDir);
        s.currentCompensations = compRotationTrackerRef.current.currentFlags();
      } else if (isMergedShoulderAbAd) {
        const a = detectShoulderAbAdDirection(kpsForDir, sideOrRight);
        if (a === "abduction") dir = "primary";
        else if (a === "adduction") dir = "secondary";
        // Ab/Ad compensation tracker — same lazy-init pattern.
        if (!compAbAdTrackerRef.current) {
          compAbAdTrackerRef.current = new ShoulderAbAdCompensationTracker(sideOrRight);
        }
        compAbAdTrackerRef.current.feed(kpsForDir);
        s.currentCompensations = compAbAdTrackerRef.current.currentFlags();
      } else if (isMergedShoulderFlexExt) {
        const fe = detectShoulderFlexExtDirection(kpsForDir, sideOrRight);
        if (fe === "flexion") dir = "primary";
        else if (fe === "extension") dir = "secondary";
        // Compensation tracker — same per-frame landmarks. Tracker
        // builds its own baseline from the first 10 valid frames and
        // flags any subsequent frame whose trunk-lean / shoulder-hip-
        // distance / elbow-bend signals cross threshold. Independent
        // of the peak/direction state machine above. Lazy-init keeps
        // the side prop in scope at construction.
        if (!compTrackerRef.current) {
          compTrackerRef.current = new ShoulderFlexExtCompensationTracker(sideOrRight);
        }
        compTrackerRef.current.feed(kpsForDir);
        s.currentCompensations = compTrackerRef.current.currentFlags();
      } else if (isMergedNeckFE) {
        const fe = detectNeckFlexExtDirection(kpsForDir);
        if (fe === "flexion") dir = "primary";
        else if (fe === "extension") dir = "secondary";
        // Neck flex/ext compensation tracker — lazy-init mirroring
        // the shoulder pattern. Feeds independently of the direction
        // state machine. Constructor takes no side (neck tests are
        // symmetric — both ears + both shoulders used).
        if (!compNeckFlexExtTrackerRef.current) {
          compNeckFlexExtTrackerRef.current = new NeckFlexExtCompensationTracker();
        }
        compNeckFlexExtTrackerRef.current.feed(kpsForDir);
        s.currentCompensations = compNeckFlexExtTrackerRef.current.currentFlags();
      } else if (isMergedNeckLateral) {
        const lat = detectNeckLateralDirection(kpsForDir);
        if (lat === "right") dir = "primary";
        else if (lat === "left") dir = "secondary";
        // Neck lateral_flexion compensation tracker.
        if (!compNeckLateralTrackerRef.current) {
          compNeckLateralTrackerRef.current = new NeckLateralFlexionCompensationTracker();
        }
        compNeckLateralTrackerRef.current.feed(kpsForDir);
        s.currentCompensations = compNeckLateralTrackerRef.current.currentFlags();
      } else if (isMergedHipRotation) {
        // hip-live.ts: primaryLabel="Internal Rotation",
        // secondaryLabel="External Rotation". Match that order here so
        // the report's primary-row stays "Internal" and the secondary
        // row stays "External" — same as the upload-mode backend
        // (_analyze_hip_rotation in hip_engine.py).
        const r = detectHipRotationDirection(kpsForDir, sideOrRight);
        if (r === "internal") dir = "primary";
        else if (r === "external") dir = "secondary";
        // Hip rotation compensation tracker — lazy-init.
        if (!compHipRotationTrackerRef.current) {
          compHipRotationTrackerRef.current = new HipRotationCompensationTracker(sideOrRight);
        }
        compHipRotationTrackerRef.current.feed(kpsForDir);
        s.currentCompensations = compHipRotationTrackerRef.current.currentFlags();
      }
      s.currentDirection = dir;
      if (!dir) {
        // ── Rep-cycle RETURN handling on deadband frames ──────
        // For routed merged tests the "back at neutral" half of a
        // rep lands exactly in this deadband — these frames return
        // early and never reach the main rep step below, so both
        // slots' machines (and the completion check) run here. A
        // swing straight from one direction to the other still
        // passes through the deadband, closing the first
        // direction's rep on the way.
        stepRep("primary");
        stepRep("secondary");
        maybeAutoComplete();
        return; // deadband — show Current but don't update peaks
      }
      slot = dir;
    } else if (isMergedKneeFE) {
      // Temporal direction tag only — doesn't gate peak updates,
      // both slots will update unconditionally below.
      const prevA = s.prevAngleForDelta;
      if (prevA !== null) {
        const trend = angle - prevA;
        if (trend > 0.5) s.currentDirection = "primary";       // flexing
        else if (trend < -0.5) s.currentDirection = "secondary"; // extending
        // else keep previous direction so the tag doesn't flicker
      }
      // Knee FE compensation tracker — feed every valid frame so its
      // baseline forms from the first 10 frames and subsequent
      // deviations get flagged.
      const kpsForComp: Keypoint[] = data.landmarks.map((l) => ({
        x: l.x, y: l.y, score: l.visibility,
      }));
      const sideOrRight = side ?? "right";
      if (!compKneeFETrackerRef.current) {
        compKneeFETrackerRef.current = new KneeFlexExtCompensationTracker(sideOrRight);
      }
      compKneeFETrackerRef.current.feed(kpsForComp);
      s.currentCompensations = compKneeFETrackerRef.current.currentFlags();
    } else {
      s.currentDirection = null;
    }

    // ── Rep-cycle counter step (active slot) ────────────────
    // Runs BEFORE the peak visibility/delta gates below: counting a
    // rep only needs a "good"-status frame — requiring peak-grade
    // landmark confidence made the return half of many cycles
    // (where visibility often dips) uncountable. Routed-merged
    // deadband frames were already stepped inside the routing block
    // above (they return early and never reach here).
    stepRep(slot);
    maybeAutoComplete();

    // Single-direction compensation feeds (hip flex/ext, ankle
    // flex/ext). These movements don't go through the merged
    // direction routing above, but their compensation trackers
    // still need every valid frame.
    if (isHipFlexion || isHipExtension || isAnkleFlexion || isAnkleExtension) {
      const kpsForComp: Keypoint[] = data.landmarks.map((l) => ({
        x: l.x, y: l.y, score: l.visibility,
      }));
      const sideOrRight = side ?? "right";
      if (isHipFlexion) {
        if (!compHipFlexionTrackerRef.current) {
          compHipFlexionTrackerRef.current = new HipFlexionCompensationTracker();
        }
        compHipFlexionTrackerRef.current.feed(kpsForComp);
        s.currentCompensations = compHipFlexionTrackerRef.current.currentFlags();
      } else if (isHipExtension) {
        if (!compHipExtensionTrackerRef.current) {
          compHipExtensionTrackerRef.current = new HipExtensionCompensationTracker();
        }
        compHipExtensionTrackerRef.current.feed(kpsForComp);
        s.currentCompensations = compHipExtensionTrackerRef.current.currentFlags();
      } else if (isAnkleFlexion) {
        if (!compAnkleFlexionTrackerRef.current) {
          compAnkleFlexionTrackerRef.current = new AnkleFlexionCompensationTracker(sideOrRight);
        }
        compAnkleFlexionTrackerRef.current.feed(kpsForComp);
        s.currentCompensations = compAnkleFlexionTrackerRef.current.currentFlags();
      } else if (isAnkleExtension) {
        if (!compAnkleExtensionTrackerRef.current) {
          compAnkleExtensionTrackerRef.current = new AnkleExtensionCompensationTracker(sideOrRight);
        }
        compAnkleExtensionTrackerRef.current.feed(kpsForComp);
        s.currentCompensations = compAnkleExtensionTrackerRef.current.currentFlags();
      }
    }

    // Neck rotation compensation tracker — runs OUTSIDE the merged-
    // direction routing block above because neck rotation uses the
    // calibrated-rotation path (its own angle math doesn't fit the
    // direction-routing paradigm). The tracker only needs per-frame
    // landmarks. Feeds on every valid frame regardless of the
    // calibration phase so baseline = first 10 valid frames during
    // the neutral hold + subsequent frames flagged against it.
    if (isNeckRotation) {
      const kpsForNR: Keypoint[] = data.landmarks.map((l) => ({
        x: l.x,
        y: l.y,
        score: l.visibility,
      }));
      if (!compNeckRotationTrackerRef.current) {
        compNeckRotationTrackerRef.current = new NeckRotationCompensationTracker();
      }
      compNeckRotationTrackerRef.current.feed(kpsForNR);
      s.currentCompensations = compNeckRotationTrackerRef.current.currentFlags();
    }

    s.validFrames += 1;

    // First usable frame after the start of measurement →
    // capture the patient's neutral / starting position once.
    if (keyFramesRef.current.neutralUrl === null) {
      keyFramesRef.current.neutralUrl = grabBiomechFrame();
    }

    // ── Peak-screenshot gates (visibility + delta) ──────────
    // Gate 1: visibility — only frames whose math-driving joints are
    // all confidently visible can update a peak. When the patient
    // disengages at end-of-trial keypoint scores collapse and the
    // angle formula can produce phantom highs; gating here stops
    // those frames from rewriting the peak screenshot.
    const relevant = relevantJointIndices(bodyPart, movementId, side);
    let minVis = 1;
    for (const i of relevant) {
      const lm = data.landmarks[i];
      const v = lm?.visibility ?? 0;
      if (v < minVis) minVis = v;
    }
    const visOk = minVis >= PEAK_VIS_GATE;

    // Gate 2: per-frame delta ceiling. Per-slot prev tracking so
    // merged movements don't fire spurious jump-rejections when the
    // patient transitions from one direction's peak back through
    // neutral to the other direction.
    const prevForDelta =
      slot === "primary" ? s.prevAngleForDelta : s.prevAngleForDeltaB;
    const deltaOk =
      prevForDelta === null ||
      Math.abs(angle - prevForDelta) <= PEAK_MAX_JUMP_DEG;
    if (slot === "primary") s.prevAngleForDelta = angle;
    else s.prevAngleForDeltaB = angle;

    if (!visOk || !deltaOk) return;

    // ── Knee merged: parallel max + min tracking ────────────
    // Both peaks update on every gated frame. Primary slot tracks
    // the running MAX angle (peak flexion); secondary slot tracks
    // the running MIN angle (peak extension / residual flexion at
    // the patient's straightest position). The min-tracking branch
    // mirrors the held-candidate algorithm but with inverted
    // comparisons — a "new lower" candidate replaces a higher one.
    if (isMergedKneeFE) {
      // Primary slot — track MAX (peak flexion).
      {
        let cs = s.peakCandidateSigned;
        let ch = s.peakCandidateHeld;
        let cu = s.peakCandidateUrl;
        let cp = s.peakSigned;
        let cpu = keyFramesRef.current.peakUrl;
        if (cs === null) {
          cs = angle; ch = 1; cu = grabBiomechFrame();
        } else if (angle > cs) {
          cs = angle; ch += 1; cu = grabBiomechFrame();
        } else if (angle >= cs - PEAK_HOLD_BAND_DEG) {
          ch += 1;
        } else {
          if (ch >= PEAK_HOLD_FRAMES && (cp === null || cs > cp)) {
            cp = cs; cpu = cu;
          }
          cs = angle; ch = 1; cu = grabBiomechFrame();
        }
        if (ch >= PEAK_HOLD_FRAMES && cs !== null && (cp === null || cs > cp)) {
          cp = cs; cpu = cu;
        }
        s.peakCandidateSigned = cs;
        s.peakCandidateHeld = ch;
        s.peakCandidateUrl = cu;
        s.peakSigned = cp;
        keyFramesRef.current.peakUrl = cpu;
      }
      // Secondary slot — track MIN (peak extension).
      {
        let cs = s.peakCandidateSignedB;
        let ch = s.peakCandidateHeldB;
        let cu = s.peakCandidateUrlB;
        let cp = s.peakSignedB;
        let cpu = keyFramesRef.current.peakUrlB;
        if (cs === null) {
          cs = angle; ch = 1; cu = grabBiomechFrame();
        } else if (angle < cs) {
          cs = angle; ch += 1; cu = grabBiomechFrame();
        } else if (angle <= cs + PEAK_HOLD_BAND_DEG) {
          ch += 1;
        } else {
          if (ch >= PEAK_HOLD_FRAMES && (cp === null || cs < cp)) {
            cp = cs; cpu = cu;
          }
          cs = angle; ch = 1; cu = grabBiomechFrame();
        }
        if (ch >= PEAK_HOLD_FRAMES && cs !== null && (cp === null || cs < cp)) {
          cp = cs; cpu = cu;
        }
        s.peakCandidateSignedB = cs;
        s.peakCandidateHeldB = ch;
        s.peakCandidateUrlB = cu;
        s.peakSignedB = cp;
        keyFramesRef.current.peakUrlB = cpu;
      }
      return;
    }

    // ── Held-candidate confirmation ─────────────────────────
    // Same algorithm as before, just operating on whichever slot
    // this frame's direction routes to. For non-merged movements
    // slot is always "primary" and the secondary fields stay null.
    const absA = Math.abs(angle);

    // Read current slot fields into locals, then write back when we
    // mutate. Keeps the candidate-logic block readable while still
    // letting one block service both slots.
    let candSigned =
      slot === "primary" ? s.peakCandidateSigned : s.peakCandidateSignedB;
    let candHeld =
      slot === "primary" ? s.peakCandidateHeld : s.peakCandidateHeldB;
    let candUrl =
      slot === "primary" ? s.peakCandidateUrl : s.peakCandidateUrlB;
    let confirmedPeak =
      slot === "primary" ? s.peakSigned : s.peakSignedB;
    let confirmedUrl =
      slot === "primary"
        ? keyFramesRef.current.peakUrl
        : keyFramesRef.current.peakUrlB;

    if (candSigned === null) {
      candSigned = angle;
      candHeld = 1;
      candUrl = grabBiomechFrame();
    } else if (absA > Math.abs(candSigned)) {
      // New higher candidate. The previous "reset to 1" behaviour
      // never let smooth continuous rotation (where every frame is a
      // new higher angle) accumulate enough hold to commit the peak —
      // the counter perpetually reset, in-place confirm never fired,
      // and the user saw a peak readout stuck at "—". The visibility
      // and delta gates above already filter out single-frame spikes,
      // so any frame that reaches this branch is part of a real,
      // gated trajectory. Increment the counter instead of resetting,
      // and re-capture the screenshot at the new high.
      candSigned = angle;
      candHeld += 1;
      candUrl = grabBiomechFrame();
    } else if (absA >= Math.abs(candSigned) - PEAK_HOLD_BAND_DEG) {
      // Within band — patient is holding near the candidate peak.
      candHeld += 1;
    } else {
      // Dropped clearly below candidate. If it was held long enough,
      // commit retroactively before resetting to the lower reading.
      if (
        candHeld >= PEAK_HOLD_FRAMES &&
        (confirmedPeak === null || Math.abs(candSigned) > Math.abs(confirmedPeak))
      ) {
        confirmedPeak = candSigned;
        confirmedUrl = candUrl;
      }
      candSigned = angle;
      candHeld = 1;
      candUrl = grabBiomechFrame();
    }

    // In-place confirm once held long enough — handles the patient-
    // holds-at-peak case while the operator decides to click Show
    // Analysis.
    if (
      candHeld >= PEAK_HOLD_FRAMES &&
      candSigned !== null &&
      (confirmedPeak === null || Math.abs(candSigned) > Math.abs(confirmedPeak))
    ) {
      confirmedPeak = candSigned;
      confirmedUrl = candUrl;
    }

    // Write back to the slot. For shoulder flexion+extension, also
    // mark the compensation tracker so its peak-windowed elbow
    // check focuses on the holding window around this peak — and
    // ignores transit-motion bends elsewhere in the recording.
    if (slot === "primary") {
      if (confirmedPeak !== null && confirmedPeak !== s.peakSigned) {
        if (isMergedShoulderFlexExt) compTrackerRef.current?.markPrimaryPeak();
        else if (isMergedShoulderAbAd) compAbAdTrackerRef.current?.markPrimaryPeak();
        else if (isMergedShoulderRotation) compRotationTrackerRef.current?.markPrimaryPeak();
        else if (isMergedNeckFE) compNeckFlexExtTrackerRef.current?.markPrimaryPeak();
        else if (isMergedNeckLateral) compNeckLateralTrackerRef.current?.markPrimaryPeak();
        else if (isMergedKneeFE) compKneeFETrackerRef.current?.markPrimaryPeak();
        else if (isMergedHipRotation) compHipRotationTrackerRef.current?.markPrimaryPeak();
      }
      s.peakCandidateSigned = candSigned;
      s.peakCandidateHeld = candHeld;
      s.peakCandidateUrl = candUrl;
      s.peakSigned = confirmedPeak;
      keyFramesRef.current.peakUrl = confirmedUrl;
    } else {
      if (confirmedPeak !== null && confirmedPeak !== s.peakSignedB) {
        if (isMergedShoulderFlexExt) compTrackerRef.current?.markSecondaryPeak();
        else if (isMergedShoulderAbAd) compAbAdTrackerRef.current?.markSecondaryPeak();
        else if (isMergedShoulderRotation) compRotationTrackerRef.current?.markSecondaryPeak();
        else if (isMergedNeckFE) compNeckFlexExtTrackerRef.current?.markSecondaryPeak();
        else if (isMergedNeckLateral) compNeckLateralTrackerRef.current?.markSecondaryPeak();
        else if (isMergedKneeFE) compKneeFETrackerRef.current?.markSecondaryPeak();
        else if (isMergedHipRotation) compHipRotationTrackerRef.current?.markSecondaryPeak();
      }
      s.peakCandidateSignedB = candSigned;
      s.peakCandidateHeldB = candHeld;
      s.peakCandidateUrlB = candUrl;
      s.peakSignedB = confirmedPeak;
      keyFramesRef.current.peakUrlB = confirmedUrl;
    }
  }, [
    isCalibratedRotation,
    calibrationLocked,
    grabBiomechFrame,
    bodyPart,
    movementId,
    side,
    isMergedMovement,
    isMergedShoulderRotation,
    isMergedShoulderAbAd,
    isMergedShoulderFlexExt,
    isMergedKneeFE,
    isMergedNeckFE,
    isMergedNeckLateral,
    isMergedHipRotation,
    isNeckRotation,
    isHipFlexion,
    isHipExtension,
    isAnkleFlexion,
    isAnkleExtension,
  ]);

  // Clears every per-trial peak-tracking field on the state ref —
  // both the primary slot AND the secondary slot for merged
  // movements. Used by calibration-lock callbacks (so a freshly
  // locked baseline starts a clean trial) and by the Reset Peak
  // button. The keyFramesRef screenshots are NOT touched here; they
  // are reset separately in resetPeak so calibration-lock keeps the
  // earlier neutral screenshot when re-baselining.
  const resetPeakState = (s: typeof stateRef.current) => {
    s.peakSigned = null;
    s.peakCandidateSigned = null;
    s.peakCandidateHeld = 0;
    s.peakCandidateUrl = null;
    s.prevAngleForDelta = null;
    s.peakSignedB = null;
    s.peakCandidateSignedB = null;
    s.peakCandidateHeldB = 0;
    s.peakCandidateUrlB = null;
    s.prevAngleForDeltaB = null;
    s.currentDirection = null;
    s.validFrames = 0;
    s.totalFrames = 0;
    s.current = null;
    s.repsPrimary = 0;
    s.repsSecondary = 0;
    s.repPhasePrimary = "neutral";
    s.repPhaseSecondary = "neutral";
    s.repBaselineMag = null;
    s.repBaselineMagB = null;
  };

  // Per-frame smoothed keypoints — used by calibration phases for
  // neck rotation and shoulder external/internal rotation. Detects
  // the "neutral pose" specific to the test (face camera straight
  // for neck; tucked elbow with forearm pointing forward for
  // shoulder), then after CALIBRATION_STABLE_MS auto-snapshots the
  // baseline. Other body parts / movements ignore this callback.
  const onSmoothedKeypoints = useCallback(
    (kp: Keypoint[]) => {
      if (!isCalibratedRotation) return;
      // Auto-flow gate — the calibration step (stable-pose timer +
      // auto-lock) starts at the countdown→live transition, keeping
      // the existing sequence: countdown → lock baseline → measure.
      // The operator's manual "Lock baseline now" button is not
      // affected (it reads latestKpRef directly).
      if (!armedRef.current) return;
      // Already calibrated — nothing to do.
      if (isNeckRotation && baseline) return;
      if (isShoulderRotation && shoulderBaseline) return;

      const s = stateRef.current;
      const tNow = Date.now();

      // Pose check varies by movement; baseline capture varies too.
      const sideOrRight = side ?? "right";
      const inNeutral = isNeckRotation
        ? isStableFacingForward(kp)
        : isShoulderRotationNeutral(kp, sideOrRight);
      s.calibFacingForward = inNeutral;

      if (inNeutral) {
        if (s.calibStableSinceMs === null) s.calibStableSinceMs = tNow;
        if (tNow - s.calibStableSinceMs >= CALIBRATION_STABLE_MS) {
          if (isNeckRotation) {
            const cal = captureNeckRotationBaseline(kp);
            if (cal) {
              setBaseline(cal);
              resetPeakState(s);
            }
          } else if (isShoulderRotation) {
            const cal = captureShoulderRotationBaseline(kp, sideOrRight);
            if (cal) {
              setShoulderBaseline(cal);
              resetPeakState(s);
            }
          }
        }
      } else {
        s.calibStableSinceMs = null;
      }
    },
    [isCalibratedRotation, isNeckRotation, isShoulderRotation, baseline, shoulderBaseline, side],
  );

  const onError = useCallback((msg: string) => {
    stateRef.current.apiError = msg;
  }, []);

  // Manual override — operator clicks "Lock baseline" without waiting
  // for the auto-stable window. Captures from the most recent frame's
  // current_angle... no, we need the keypoints. Stash the latest
  // smoothed keypoints in a ref so the manual button can reach them.
  const latestKpRef = useRef<Keypoint[] | null>(null);
  const onSmoothedKeypointsTracked = useCallback(
    (kp: Keypoint[]) => {
      latestKpRef.current = kp;
      onSmoothedKeypoints(kp);
    },
    [onSmoothedKeypoints],
  );

  function lockBaselineNow() {
    const kp = latestKpRef.current;
    if (!kp) return;
    const sideOrRight = side ?? "right";
    if (isNeckRotation) {
      const cal = captureNeckRotationBaseline(kp);
      if (!cal) return;
      setBaseline(cal);
    } else if (isShoulderRotation) {
      const cal = captureShoulderRotationBaseline(kp, sideOrRight);
      if (!cal) return;
      setShoulderBaseline(cal);
    } else {
      return;
    }
    resetPeakState(stateRef.current);
  }

  function resetPeak() {
    const s = stateRef.current;
    resetPeakState(s);
    autoCompleteFiredRef.current = false;
    s.calibStableSinceMs = null;
    s.calibFacingForward = false;
    s.postureHint = null;
    keyFramesRef.current.neutralUrl = null;
    keyFramesRef.current.peakUrl = null;
    keyFramesRef.current.peakUrlB = null;
    setShowResult(false);
    // For calibrated-rotation tests, "Reset Peak" also clears the
    // calibration so the operator can re-baseline (e.g. patient
    // shifted position between attempts).
    if (isNeckRotation) setBaseline(null);
    if (isShoulderRotation) setShoulderBaseline(null);
    setVersion((v) => v + 1);
  }

  // ── Auto-flow countdown ────────────────────────────────────────
  // Starts only once the fullscreen shell is open AND the camera
  // stream is actually live — otherwise the 3-2-1 would eat the
  // getUserMedia permission delay. There is no separate framing gate
  // today (framing hints are advisory; capture is continuous), so
  // camActive is the only readiness condition. onLive arms capture:
  // peaks / screenshots / calibration all start from a clean slate
  // at that instant.
  const {
    phase: flowPhase,
    countdown,
    skipCountdown,
  } = useRehabAutoFlow(liveFullscreen && camActive, () => {
    armedRef.current = true;
    autoCompleteFiredRef.current = false;
    const s = stateRef.current;
    resetPeakState(s);
    s.calibStableSinceMs = null;
    s.calibFacingForward = false;
    keyFramesRef.current.neutralUrl = null;
    keyFramesRef.current.peakUrl = null;
    keyFramesRef.current.peakUrlB = null;
  });

  // Enter the fullscreen auto-flow shell (the single click of the
  // live mode). Camera auto-starts inside; countdown → capture.
  function enterLive() {
    armedRef.current = false;
    autoCompleteFiredRef.current = false;
    setShowResult(false);
    setCamActive(false);
    setLiveFullscreen(true);
  }

  function exitLive() {
    armedRef.current = false;
    resetPeak();
    setLiveFullscreen(false);
    setCamActive(false);
  }

  // "Show Analysis" — ends the live session exactly as before, just
  // also leaving the fullscreen shell so the report renders in the
  // normal page flow. Auto Mode's onCompleted fires here too (every
  // completion path — rep auto-complete, manual click, forced —
  // funnels through this function).
  const onCompletedRef = useRef(onCompleted);
  onCompletedRef.current = onCompleted;
  function showAnalysis() {
    armedRef.current = false;
    setShowResult(true);
    setLiveFullscreen(false);
    setCamActive(false);
    onCompletedRef.current?.();
  }

  // Auto Mode: enter the fullscreen shell immediately on mount —
  // the sequence runner already counted the patient in. Fired once
  // (StrictMode-safe).
  const autoEnteredRef = useRef(false);
  useEffect(() => {
    if (!autoEnter || autoEnteredRef.current) return;
    autoEnteredRef.current = true;
    enterLive();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoEnter]);

  // Auto Mode safety timeout: when the runner bumps the signal,
  // complete with whatever has been captured. Nothing captured →
  // no-op (the runner skips the step itself after a grace period).
  const prevForceSignalRef = useRef(forceCompleteSignal);
  useEffect(() => {
    if (forceCompleteSignal === undefined) return;
    if (prevForceSignalRef.current === forceCompleteSignal) return;
    prevForceSignalRef.current = forceCompleteSignal;
    if (showResult) return;
    const s = stateRef.current;
    const anyPeak =
      s.peakSigned !== null || (isMergedMovement && s.peakSignedB !== null);
    if (anyPeak) showAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forceCompleteSignal, showResult]);

  // ── derived render values ────────────────────────────────────
  const {
    current,
    peakSigned,
    peakSignedB,
    validFrames,
    totalFrames,
    status,
    apiError,
    postureHint,
    currentDirection,
    repsPrimary,
    repsSecondary,
  } = stateRef.current;
  // Merged direction-routed tests need the rep target in BOTH
  // directions; knee FE counts bend/straighten cycles on the primary
  // slot only (extension is captured passively at each return).
  const repsNeedBoth = !!isMergedMovement && !isMergedKneeFE;
  // For knee merged, the secondary peak is the MIN angle reached
  // (residual flexion at full extension). A value of 0° is a
  // legitimate measurement — it means the knee straightened
  // perfectly — so we can't filter it out with the usual `> 0`
  // shoulder convention. Derive "has captured a peak" from the
  // signed peak being non-null instead.
  const peakMag = peakSigned !== null ? Math.abs(peakSigned) : 0;
  const peakMagB = peakSignedB !== null ? Math.abs(peakSignedB) : 0;
  const hasPeak = peakSigned !== null;
  const hasPeakB = peakSignedB !== null;
  const hasAnyPeak = hasPeak || (isMergedMovement && hasPeakB);

  const instructions = getInstructions(bodyPart, movementId);
  // Reference illustration — shown for any joint whose movement
  // entry carries an imageUrl. Returns null for joints/movements
  // without an asset, so the render stays a no-op there.
  const movementImageUrl = resolveMovement(bodyPart, movementId)?.imageUrl ?? null;

  const statusPresentation: Record<
    PostureStatus,
    { color: string; text: string; Icon: typeof CheckCircle2 }
  > = {
    idle: { color: "text-muted", text: "Waiting for camera…", Icon: AlertCircle },
    good: {
      color: "text-accent",
      text: "✓ Subject visible — capturing",
      Icon: CheckCircle2,
    },
    low_visibility: {
      color: "text-warning",
      text: "Required landmarks below visibility threshold",
      Icon: AlertTriangle,
    },
    no_landmarks: {
      color: "text-error",
      text: "No subject detected — check position / lighting",
      Icon: AlertCircle,
    },
  };
  const sp = statusPresentation[status];

  let resultStatus: "good" | "fair" | "poor" = "poor";
  if (peakMag >= target[0] && peakMag <= target[1]) resultStatus = "good";
  else if (peakMag >= target[0] * 0.8 && peakMag <= target[1] * 1.1)
    resultStatus = "fair";

  const resultStyles =
    resultStatus === "good"
      ? "border-accent/40 bg-accent/5"
      : resultStatus === "fair"
        ? "border-warning/40 bg-warning/5"
        : "border-error/40 bg-error/5";
  const resultText =
    resultStatus === "good"
      ? "Within normal range"
      : resultStatus === "fair"
        ? "Near normal range"
        : "Below normal range";
  const resultColor =
    resultStatus === "good"
      ? "text-accent"
      : resultStatus === "fair"
        ? "text-warning"
        : "text-error";

  const inCalibrationUi = isCalibratedRotation && !calibrationLocked;

  // Positioning cue under the sidebar countdown — for rotation tests
  // it primes the patient for the baseline step that follows.
  const countdownHint = isNeckRotation
    ? "Patient: face the camera straight ahead, head still — the baseline step starts right after."
    : isShoulderRotation
      ? "Patient: tuck the elbow at your side, forearm pointing forward — the baseline step starts right after."
      : "Stand ~2 m (6 ft) from the camera with the whole limb in frame.";

  // ─── Pre-fullscreen intro — movement context + ONE click ───────
  const introView = (
    <div className="space-y-10">
      {/* ─── CENTERED TITLE BLOCK ─────────────────────────────── */}
      <div className="text-center">
        <p className="eyebrow">Current movement</p>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight md:text-4xl">
          {movementLabel}
        </h2>
        <p className="mt-2 text-sm text-muted">{description}</p>
        {side && (
          <p className="mt-2 text-xs uppercase tracking-[0.12em] text-subtle">
            Side: <span className="text-foreground">{side}</span>
          </p>
        )}
      </div>

      {/* ─── 2-column layout (instructions | one-click start) ──── */}
      <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
        {/* ─── LEFT ────────────────────────────────────────────── */}
        <div className="space-y-5">
          {instructions.length > 0 && (
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Movement instructions
            </p>
            {movementImageUrl && (
              <div className="mt-3 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={movementImageUrl}
                  alt=""
                  aria-hidden="true"
                  loading="lazy"
                  className="block w-full object-contain"
                  style={{ maxHeight: 280 }}
                />
              </div>
            )}
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {instructions.map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
        </div>

        {/* ─── RIGHT: one-click start ──────────────────────────── */}
        <div className="space-y-4">
          <div className="rounded-card border border-border bg-surface p-6 text-center">
            <p className="text-sm text-muted">
              One click — the camera opens fullscreen, a 3-2-1 countdown
              runs, and capture starts by itself.{" "}
              {isCalibratedRotation
                ? `The baseline step runs first (hold the neutral pose until it locks), then perform the movement ${TARGET_REP_COUNT} times — the analysis opens and saves by itself.`
                : `Perform the movement ${TARGET_REP_COUNT} times — the analysis opens and saves by itself after the final rep.`}
            </p>
            <div className="mt-4 flex justify-center">
              <Button onClick={enterLive}>
                <Camera className="h-4 w-4" />
                Start Assessment
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* ─── Unified report disclaimer ──────────────────────────── */}
      <ReportDisclaimer />
    </div>
  );

  // ─── Fullscreen sidebar — countdown / live status / controls ────
  const fullscreenSidebar = (
    <>
      {flowPhase === "countdown" && countdown !== null && (
        <AutoFlowCountdownCard
          countdown={countdown}
          onSkip={skipCountdown}
          hint={countdownHint}
        />
      )}

        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Live status
          </p>

          <div className="mt-3 flex items-center gap-2">
            <sp.Icon className={`h-4 w-4 ${sp.color}`} />
            <span className={`text-sm ${sp.color}`}>{sp.text}</span>
          </div>

          {/* Actionable framing guidance — only shown while the
              subject isn't correctly positioned. Clears itself the
              moment the framing core is visible and capture is good. */}
          {postureHint && status !== "good" && (
            <p className="mt-2 rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning">
              ↳ {postureHint}
            </p>
          )}

          {apiError && (
            <p className="mt-2 text-xs text-error">⚠ {apiError}</p>
          )}

          <div className="mt-5 grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.12em] text-subtle">
                Current
              </p>
              <p className="mt-1 tabular text-3xl font-semibold leading-none text-foreground">
                {/* Show magnitude only — some movement formulas return
                    signed values where the sign just encodes direction
                    (e.g. neck flex/ext is positive for one orientation,
                    negative for the other; shoulder rotation is signed
                    in the calibrated formula). The operator-facing
                    Current readout should always read as a positive
                    angle; direction is conveyed by the tag below. */}
                {current !== null ? `${fmt(Math.abs(current), 1)}°` : "—"}
              </p>
              {isMergedMovement && currentDirection && (
                <p className="mt-1 text-[10px] uppercase tracking-[0.1em] text-accent">
                  {currentDirection === "primary"
                    ? primaryLabel ?? "Primary"
                    : secondaryLabel ?? "Secondary"}
                </p>
              )}
            </div>
            {isMergedMovement ? (
              <>
                <div>
                  <p className="text-xs uppercase tracking-[0.12em] text-subtle">
                    {primaryLabel ?? "Peak A"}
                  </p>
                  <p className="mt-1 tabular text-2xl font-semibold leading-none text-accent">
                    {hasPeak ? `${fmt(peakMag, 1)}°` : "—"}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.12em] text-subtle">
                    {secondaryLabel ?? "Peak B"}
                  </p>
                  <p className="mt-1 tabular text-2xl font-semibold leading-none text-accent">
                    {hasPeakB ? `${fmt(peakMagB, 1)}°` : "—"}
                  </p>
                </div>
              </>
            ) : (
              <>
                <div>
                  <p className="text-xs uppercase tracking-[0.12em] text-subtle">Peak</p>
                  <p className="mt-1 tabular text-3xl font-semibold leading-none text-accent">
                    {hasPeak ? `${fmt(peakMag, 1)}°` : "—"}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.12em] text-subtle">Frames</p>
                  <p className="mt-1 tabular text-2xl font-semibold leading-none text-foreground">
                    {validFrames}
                    <span className="text-sm text-subtle">/{totalFrames}</span>
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Rep-cycle progress — the session ends itself once the
              patient completes TARGET_REP_COUNT full cycles (both
              directions for routed merged tests). */}
          <div className="mt-4 rounded-md border border-accent/25 bg-accent/5 px-3 py-2">
            <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">
              Reps · auto-completes at {TARGET_REP_COUNT}
            </p>
            <p className="mt-1 tabular text-sm font-semibold text-foreground">
              {repsNeedBoth ? (
                <>
                  {primaryLabel ?? "A"}{" "}
                  <span className="text-accent">
                    {Math.min(repsPrimary, TARGET_REP_COUNT)}/{TARGET_REP_COUNT}
                  </span>
                  <span className="mx-2 text-subtle">·</span>
                  {secondaryLabel ?? "B"}{" "}
                  <span className="text-accent">
                    {Math.min(repsSecondary, TARGET_REP_COUNT)}/{TARGET_REP_COUNT}
                  </span>
                </>
              ) : (
                <span className="text-accent">
                  {Math.min(repsPrimary, TARGET_REP_COUNT)}/{TARGET_REP_COUNT}
                </span>
              )}
            </p>
          </div>

          {/* Real-time compensation warning banner. Renders for any
              movement that runs a compensation tracker. Disappears
              as soon as the patient corrects form. */}
          {(
            isMergedShoulderFlexExt ||
            isMergedShoulderAbAd ||
            isMergedShoulderRotation ||
            isMergedNeckFE ||
            isMergedNeckLateral ||
            isNeckRotation ||
            isMergedKneeFE ||
            isMergedHipRotation ||
            isHipFlexion ||
            isHipExtension ||
            isAnkleFlexion ||
            isAnkleExtension
          ) && stateRef.current.currentCompensations.length > 0 && (
            <div className="mt-4 rounded-md border border-warning/40 bg-warning/5 px-3 py-2">
              <p className="text-[10px] uppercase tracking-[0.12em] text-warning">
                Compensation detected
              </p>
              <ul className="mt-1 space-y-0.5 text-xs text-foreground">
                {stateRef.current.currentCompensations.map((c) => (
                  <li key={c.type} className="flex items-center gap-2">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        c.severity === "high" ? "bg-error" : "bg-warning"
                      }`}
                    />
                    <span>{c.label}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="mt-5 text-xs text-muted">
            {isMergedMovement ? (
              <>
                Targets:{" "}
                <span className="tabular text-foreground">
                  {primaryLabel ?? "A"} {target[0]}°–{target[1]}°
                </span>
                {secondaryTarget && (
                  <>
                    {" · "}
                    <span className="tabular text-foreground">
                      {secondaryLabel ?? "B"} {secondaryTarget[0]}°–{secondaryTarget[1]}°
                    </span>
                  </>
                )}
                . Perform {TARGET_REP_COUNT} reps in each direction — the
                analysis opens by itself after the final rep ({" "}
                <span className="text-foreground">Show Analysis</span>{" "}
                ends it early).
              </>
            ) : (
              <>
                Target range:{" "}
                <span className="tabular text-foreground">
                  {target[0]}°–{target[1]}°
                </span>
                . Capture is continuous — perform the movement{" "}
                {TARGET_REP_COUNT} times and the analysis opens by itself ({" "}
                <span className="text-foreground">Show Analysis</span>{" "}
                ends it early).
              </>
            )}
          </p>

          {/* Calibration banner. Renders for any rotation test that
              uses a baseline-locked formula (neck + shoulder
              ext/int). Hides automatically once baseline is locked. */}
          {isCalibratedRotation && !calibrationLocked && (
            <div className="mt-5 rounded-card border border-accent/40 bg-accent/5 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-accent">
                Step 1 of 2 — Lock baseline
              </p>
              <p className="mt-2 text-sm text-foreground">
                {isNeckRotation
                  ? "Patient: face the camera straight ahead with your head still."
                  : "Patient: tuck the elbow against your side, bend it 90°, and point the forearm straight forward at the camera. Hold still."}
                {" "}The baseline auto-locks once you&apos;re stable for{" "}
                {(CALIBRATION_STABLE_MS / 1000).toFixed(1)} s.
              </p>
              <div className="mt-3 flex items-center gap-2 text-xs">
                <span
                  className={
                    stateRef.current.calibFacingForward
                      ? "inline-flex items-center gap-1 text-accent"
                      : "inline-flex items-center gap-1 text-muted"
                  }
                >
                  <span
                    className={`h-2 w-2 rounded-full ${
                      stateRef.current.calibFacingForward
                        ? "bg-accent animate-pulse"
                        : "bg-border"
                    }`}
                  />
                  {stateRef.current.calibFacingForward
                    ? "Holding steady…"
                    : isNeckRotation
                      ? "Waiting for facing-forward pose"
                      : "Waiting for tucked-elbow pose"}
                </span>
              </div>
              <div className="mt-3">
                <Button
                  onClick={lockBaselineNow}
                  variant="secondary"
                  size="sm"
                  disabled={status !== "good" && status !== "low_visibility"}
                >
                  <Lock className="h-4 w-4" />
                  Lock baseline now
                </Button>
              </div>
            </div>
          )}

          {isCalibratedRotation && calibrationLocked && (
            <div className="mt-5 rounded-card border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs">
              <p className="inline-flex items-center gap-1.5 font-medium text-foreground">
                <Lock className="h-3.5 w-3.5 text-emerald-600" />
                Step 2 of 2 — Baseline locked.
                {isNeckRotation
                  ? " Rotate the head toward the shoulder."
                  : movementId === "rotation"
                    ? " Rotate the forearm outward, return to centre, then inward — both peaks are captured."
                    : movementId === "external_rotation"
                      ? " Rotate the forearm outward to the peak."
                      : " Rotate the forearm inward to the peak."}
              </p>
            </div>
          )}

          <div className="mt-5 flex gap-2">
            <Button
              onClick={showAnalysis}
              disabled={!hasAnyPeak || (isCalibratedRotation && !calibrationLocked)}
              className="flex-1"
            >
              <Eye className="h-4 w-4" />
              Show Analysis
            </Button>
            <Button
              variant="secondary"
              onClick={resetPeak}
              disabled={!hasAnyPeak && !(isCalibratedRotation && calibrationLocked)}
            >
              <RotateCcw className="h-4 w-4" />
              {isCalibratedRotation && calibrationLocked ? "Re-baseline" : "Reset Peak"}
            </Button>
          </div>
        </div>
    </>
  );

  // ─── Fullscreen live shell — camera left, status sidebar right ──
  const fullscreenView = (
    <LiveModeLayout
      title={movementLabel}
      subtitle={side ? `${description} · Side: ${side}` : description}
      onExit={exitLive}
      camera={(
        <LiveBiomechCamera
          bodyPart={bodyPart}
          movement={movementId}
          side={side}
          onResult={onResult}
          onError={onError}
          neckRotationBaseline={isNeckRotation ? baseline : null}
          shoulderRotationBaseline={isShoulderRotation ? shoulderBaseline : null}
          onSmoothedKeypoints={isCalibratedRotation ? onSmoothedKeypointsTracked : undefined}
          autoStart
          hideControls
          fill
          onActiveChange={setCamActive}
        >
          {!camActive && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="rounded-full bg-black/60 px-4 py-2 text-sm text-white/80">
                Starting camera…
              </p>
            </div>
          )}
          {flowPhase === "countdown" && countdown !== null && (
            <AutoFlowCountdownOverlay
              countdown={countdown}
              label={isCalibratedRotation ? "Baseline step in" : "Capture starts in"}
            />
          )}
          {flowPhase === "live" && (
            <div className="absolute left-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
              <p className="text-[10px] uppercase tracking-[0.14em] text-emerald-300">
                {inCalibrationUi ? "Step 1 — lock baseline" : "● Capturing"}
              </p>
              <p className="tabular text-2xl font-semibold text-white">
                {current !== null ? `${fmt(Math.abs(current), 1)}°` : "—"}
              </p>
              {!inCalibrationUi && (
                <p className="tabular text-[10px] text-white/70">
                  {isMergedMovement
                    ? `${primaryLabel ?? "A"} ${hasPeak ? `${fmt(peakMag, 1)}°` : "—"} · ${secondaryLabel ?? "B"} ${hasPeakB ? `${fmt(peakMagB, 1)}°` : "—"}`
                    : `Peak ${hasPeak ? `${fmt(peakMag, 1)}°` : "—"}`}
                </p>
              )}
              {!inCalibrationUi && (
                <p className="tabular text-[10px] font-semibold text-emerald-300">
                  {repsNeedBoth
                    ? `Reps ${Math.min(repsPrimary, TARGET_REP_COUNT)}/${TARGET_REP_COUNT} · ${Math.min(repsSecondary, TARGET_REP_COUNT)}/${TARGET_REP_COUNT}`
                    : `Rep ${Math.min(repsPrimary, TARGET_REP_COUNT)}/${TARGET_REP_COUNT}`}
                </p>
              )}
            </div>
          )}
        </LiveBiomechCamera>
      )}
      sidebar={fullscreenSidebar}
    />
  );

  if (showResult && hasAnyPeak) {
    // Assemble the keyFrames array for the report. Only include
    // entries we actually captured (skip nulls so the section is
    // hidden cleanly when capture fell through, e.g. very short
    // attempts where the rAF loop hadn't yet stored a smoothed
    // landmark snapshot).
    const liveKeyFrames: Array<{
      label: string;
      frame_index: number;
      image_data_url: string;
    }> = [];
    if (keyFramesRef.current.neutralUrl) {
      liveKeyFrames.push({
        label: "Neutral — start",
        frame_index: 0,
        image_data_url: keyFramesRef.current.neutralUrl,
      });
    }
    if (keyFramesRef.current.peakUrl && hasPeak) {
      liveKeyFrames.push({
        label: isMergedMovement
          ? `${primaryLabel ?? "Peak A"} (${peakMag.toFixed(1)}°)`
          : `Peak ${reportName} (${peakMag.toFixed(1)}°)`,
        frame_index: 1,
        image_data_url: keyFramesRef.current.peakUrl,
      });
    }
    if (isMergedMovement && keyFramesRef.current.peakUrlB && hasPeakB) {
      liveKeyFrames.push({
        label: `${secondaryLabel ?? "Peak B"} (${peakMagB.toFixed(1)}°)`,
        frame_index: 2,
        image_data_url: keyFramesRef.current.peakUrlB,
      });
    }

    return (
      <div className="space-y-8">
        <AssessmentReport
          bodyPart={bodyPart}
          movementName={isMergedMovement ? (primaryLabel ?? reportName) : reportName}
          movementId={movementId}
          measured={peakMag}
          target={target}
          side={side}
          keyFrames={liveKeyFrames}
          /* For merged tests we ALWAYS pass the secondary props so
             the report renders two rows, two chart bars, and two
             interpretation sentences. The measured value is
             undefined when the patient didn't perform that
             direction — the report shows "Not detected" for it
             instead of silently collapsing to a single row. */
          secondaryMovementName={isMergedMovement ? secondaryLabel : undefined}
          secondaryMeasured={
            isMergedMovement && hasPeakB ? peakMagB : undefined
          }
          secondaryTarget={isMergedMovement ? secondaryTarget : undefined}
          /* Final compensations summary — resolve from whichever
             tracker matches the active merged movement (shoulder
             flex+ext / ab+ad / rotation, or neck flex+ext /
             lateral_flexion / rotation). Only one tracker is ever
             populated per trial, so at most one finish() result is
             non-undefined. Every other (bodyPart, movement) pair
             leaves all trackers null and the report's section stays
             hidden. */
          compensations={
            isMergedShoulderFlexExt
              ? compTrackerRef.current?.finish()
              : isMergedShoulderAbAd
                ? compAbAdTrackerRef.current?.finish()
                : isMergedShoulderRotation
                  ? compRotationTrackerRef.current?.finish()
                  : isMergedNeckFE
                    ? compNeckFlexExtTrackerRef.current?.finish()
                    : isMergedNeckLateral
                      ? compNeckLateralTrackerRef.current?.finish()
                      : isNeckRotation
                        ? compNeckRotationTrackerRef.current?.finish()
                        : isMergedKneeFE
                          ? compKneeFETrackerRef.current?.finish()
                          : isMergedHipRotation
                            ? compHipRotationTrackerRef.current?.finish()
                            : isHipFlexion
                              ? compHipFlexionTrackerRef.current?.finish()
                              : isHipExtension
                                ? compHipExtensionTrackerRef.current?.finish()
                                : isAnkleFlexion
                                  ? compAnkleFlexionTrackerRef.current?.finish()
                                  : isAnkleExtension
                                    ? compAnkleExtensionTrackerRef.current?.finish()
                                    : undefined
          }
        />

        {/* Live sessions auto-save in the doctor flow (toast with a
            10s undo). Fires once on mount of the report view — i.e.
            exactly when today's manual save became available. No-op
            in the public flow. Payload identical to the previous
            SaveToPatientButton. */}
        <AutoSaveToast
          onSaved={onSaved}
          buildPayload={() => ({
            module: "biomech",
            body_part: bodyPart,
            movement: movementId,
            side,
            metrics: {
              peak_magnitude: peakMag,
              peak_signed: peakSigned,
              target,
              // Persist merged-test metadata for the saved-report
              // viewer. Labels + secondary target are saved whenever
              // the test is merged, so the viewer can render the
              // dual-row layout (with "Not detected" placeholder for
              // a missing direction) even when only one direction
              // was actually captured. The numeric secondary peak is
              // saved only when it was actually measured.
              ...(isMergedMovement
                ? {
                    primary_label: primaryLabel,
                    secondary_label: secondaryLabel,
                    secondary_target: secondaryTarget,
                    ...(hasPeakB
                      ? {
                          secondary_peak_magnitude: peakMagB,
                          secondary_peak_signed: peakSignedB,
                        }
                      : {}),
                  }
                : {}),
              valid_frames: validFrames,
              total_frames: totalFrames,
              // Persist annotated screenshots so the saved-report
              // viewer can show the same key-frame strip later.
              key_frames: liveKeyFrames,
              // Persist compensatory-movement findings from whichever
              // tracker ran during the trial. Only one is non-null
              // per recording (only one merged movement active at a
              // time across shoulder + neck). Resolution chain
              // mirrors the AssessmentReport prop above.
              ...(() => {
                const c = isMergedShoulderFlexExt
                  ? compTrackerRef.current?.finish()
                  : isMergedShoulderAbAd
                    ? compAbAdTrackerRef.current?.finish()
                    : isMergedShoulderRotation
                      ? compRotationTrackerRef.current?.finish()
                      : isMergedNeckFE
                        ? compNeckFlexExtTrackerRef.current?.finish()
                        : isMergedNeckLateral
                          ? compNeckLateralTrackerRef.current?.finish()
                          : isNeckRotation
                            ? compNeckRotationTrackerRef.current?.finish()
                            : isMergedKneeFE
                              ? compKneeFETrackerRef.current?.finish()
                              : isMergedHipRotation
                                ? compHipRotationTrackerRef.current?.finish()
                                : isHipFlexion
                                  ? compHipFlexionTrackerRef.current?.finish()
                                  : isHipExtension
                                    ? compHipExtensionTrackerRef.current?.finish()
                                    : isAnkleFlexion
                                      ? compAnkleFlexionTrackerRef.current?.finish()
                                      : isAnkleExtension
                                        ? compAnkleExtensionTrackerRef.current?.finish()
                                        : null;
                return c && c.length > 0 ? { compensations: c } : {};
              })(),
            },
          })}
        />

        <div className="flex justify-center gap-3 border-t border-border pt-6">
          {/* Auto Mode delegates the retest to the sequence runner —
              it deletes this attempt's auto-saved report and remounts
              the step fresh (camera auto-starts again). Standalone
              mode keeps the in-place reset. */}
          <Button
            variant="secondary"
            onClick={() => (onRetest ? onRetest() : resetPeak())}
          >
            <RotateCcw className="h-4 w-4" />
            {onRetest ? "Retest (discard this attempt)" : "Try again"}
          </Button>
        </div>
      </div>
    );
  }

  // Pre-result view — fullscreen capture shell once started; intro
  // (instructions + one-click start, with the doctor-flow reminder
  // banner) otherwise.
  if (liveFullscreen) return fullscreenView;
  return (
    <>
      {isDoctorFlow && (
        <div className="mb-6">
          <SaveStatusBanner patient={patient} saveStatus={null} />
        </div>
      )}
      {introView}
    </>
  );
}
