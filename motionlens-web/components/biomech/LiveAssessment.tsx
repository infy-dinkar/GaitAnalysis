"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Eye,
  Lock,
  RotateCcw,
} from "lucide-react";
import { LiveBiomechCamera } from "@/components/biomech/LiveBiomechCamera";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { Button } from "@/components/ui/Button";
import { usePatientContext } from "@/hooks/usePatientContext";
import { fmt } from "@/lib/utils";
import { getInstructions } from "@/lib/biomech/instructions";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import {
  captureNeckRotationBaseline,
  isStableFacingForward,
  type NeckRotationCalibration,
} from "@/lib/biomech/neck";
import {
  captureShoulderRotationBaseline,
  isShoulderRotationNeutral,
  type ShoulderRotationCalibration,
} from "@/lib/biomech/shoulder";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveBiomechFrameDataDTO } from "@/lib/api";
import { LM } from "@/lib/pose/landmarks";

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
      return [LM.NOSE, LM.LEFT_EAR, LM.RIGHT_EAR, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER];
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
  const isShoulderRotation =
    bodyPart === "shoulder" &&
    (movementId === "external_rotation" || movementId === "internal_rotation");
  const isCalibratedRotation = isNeckRotation || isShoulderRotation;

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
  });

  // Annotated-frame screenshots for the report. Captured on the
  // composite canvas exposed by LiveBiomechCamera (via
  // window.__biomechCapture).
  //   - neutralUrl: first usable frame after movement starts → the
  //     patient's neutral / starting position.
  //   - peakUrl:    re-captured every time a new peak ROM is reached.
  // Both reset on "Reset Peak" so a re-attempt starts fresh.
  const keyFramesRef = useRef<{ neutralUrl: string | null; peakUrl: string | null }>({
    neutralUrl: null,
    peakUrl: null,
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

  // Single source of truth for "calibration done" — used by gating
  // logic so the same code path handles both neck and shoulder.
  const calibrationLocked = isNeckRotation
    ? !!baseline
    : isShoulderRotation
      ? !!shoulderBaseline
      : true;

  // Doctor-flow context (no-op when accessed publicly). Save happens
  // explicitly via the SaveToPatientButton inside the report view.
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
      return;
    }

    const angle = data.current_angle;
    s.current = angle;
    s.validFrames += 1;

    // First usable frame after the start of measurement →
    // capture the patient's neutral / starting position once.
    if (keyFramesRef.current.neutralUrl === null) {
      keyFramesRef.current.neutralUrl = grabBiomechFrame();
    }

    // ── Peak-screenshot gates ───────────────────────────────
    // Gate 1: visibility — the primary joints driving this body
    // part's angle math must all be confidently visible. When the
    // patient disengages at end-of-trial, keypoint scores collapse
    // and the angle formula can produce phantom highs; gating here
    // stops those frames from rewriting the peak screenshot.
    const relevant = relevantJointIndices(bodyPart, movementId, side);
    let minVis = 1;
    for (const i of relevant) {
      const lm = data.landmarks[i];
      const v = lm?.visibility ?? 0;
      if (v < minVis) minVis = v;
    }
    const visOk = minVis >= PEAK_VIS_GATE;

    // Gate 2: per-frame delta ceiling. A single frame whose angle
    // changed by more than PEAK_MAX_JUMP_DEG is almost always a
    // formula glitch (saturation near 90°, sign flip) or a keypoint
    // switch — limbs physically can't rotate that fast between
    // frames at 30-60 FPS. Drop it from peak consideration but keep
    // updating `prevAngle` so the next frame's delta is measured
    // from the current reading (we don't want to permanently shadow
    // a legitimate fast movement).
    const prev = s.prevAngleForDelta;
    const deltaOk = prev === null || Math.abs(angle - prev) <= PEAK_MAX_JUMP_DEG;
    s.prevAngleForDelta = angle;

    if (!visOk || !deltaOk) return;

    // ── Held-candidate confirmation ─────────────────────────
    // Candidate is the largest |angle| seen in the current peak
    // attempt. It only commits to `peakSigned` + `peakUrl` once it
    // has been sustained (within band) for PEAK_HOLD_FRAMES frames.
    const absA = Math.abs(angle);
    const cand = s.peakCandidateSigned;

    if (cand === null) {
      s.peakCandidateSigned = angle;
      s.peakCandidateHeld = 1;
      s.peakCandidateUrl = grabBiomechFrame();
    } else if (absA > Math.abs(cand)) {
      // New higher candidate — reset hold counter and re-capture
      // the screenshot at this fresh peak position.
      s.peakCandidateSigned = angle;
      s.peakCandidateHeld = 1;
      s.peakCandidateUrl = grabBiomechFrame();
    } else if (absA >= Math.abs(cand) - PEAK_HOLD_BAND_DEG) {
      // Within band of candidate — patient is holding the peak.
      s.peakCandidateHeld += 1;
    } else {
      // Dropped clearly below candidate (patient came down off
      // peak). If the candidate was already held long enough, this
      // is a strong signal that the previously-tracked peak was
      // real — commit it before resetting the candidate to the new
      // (lower) angle.
      if (
        s.peakCandidateHeld >= PEAK_HOLD_FRAMES &&
        (s.peakSigned === null || Math.abs(cand) > Math.abs(s.peakSigned))
      ) {
        s.peakSigned = cand;
        keyFramesRef.current.peakUrl = s.peakCandidateUrl;
      }
      s.peakCandidateSigned = angle;
      s.peakCandidateHeld = 1;
      s.peakCandidateUrl = grabBiomechFrame();
    }

    // Confirm in-place once the candidate has been held the full
    // window — handles the common case where the patient holds at
    // peak while the operator clicks "Show Analysis".
    if (
      s.peakCandidateHeld >= PEAK_HOLD_FRAMES &&
      s.peakCandidateSigned !== null &&
      (s.peakSigned === null ||
        Math.abs(s.peakCandidateSigned) > Math.abs(s.peakSigned))
    ) {
      s.peakSigned = s.peakCandidateSigned;
      keyFramesRef.current.peakUrl = s.peakCandidateUrl;
    }
  }, [isCalibratedRotation, calibrationLocked, grabBiomechFrame, bodyPart, movementId, side]);

  // Per-frame smoothed keypoints — used by calibration phases for
  // neck rotation and shoulder external/internal rotation. Detects
  // the "neutral pose" specific to the test (face camera straight
  // for neck; tucked elbow with forearm pointing forward for
  // shoulder), then after CALIBRATION_STABLE_MS auto-snapshots the
  // baseline. Other body parts / movements ignore this callback.
  const onSmoothedKeypoints = useCallback(
    (kp: Keypoint[]) => {
      if (!isCalibratedRotation) return;
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
              s.peakSigned = null;
              s.validFrames = 0;
              s.totalFrames = 0;
              s.current = null;
              s.peakCandidateSigned = null;
              s.peakCandidateHeld = 0;
              s.peakCandidateUrl = null;
              s.prevAngleForDelta = null;
            }
          } else if (isShoulderRotation) {
            const cal = captureShoulderRotationBaseline(kp, sideOrRight);
            if (cal) {
              setShoulderBaseline(cal);
              s.peakSigned = null;
              s.validFrames = 0;
              s.totalFrames = 0;
              s.current = null;
              s.peakCandidateSigned = null;
              s.peakCandidateHeld = 0;
              s.peakCandidateUrl = null;
              s.prevAngleForDelta = null;
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
    const s = stateRef.current;
    s.peakSigned = null;
    s.validFrames = 0;
    s.totalFrames = 0;
    s.current = null;
    s.peakCandidateSigned = null;
    s.peakCandidateHeld = 0;
    s.peakCandidateUrl = null;
    s.prevAngleForDelta = null;
  }

  function resetPeak() {
    const s = stateRef.current;
    s.peakSigned = null;
    s.validFrames = 0;
    s.totalFrames = 0;
    s.calibStableSinceMs = null;
    s.calibFacingForward = false;
    s.peakCandidateSigned = null;
    s.peakCandidateHeld = 0;
    s.peakCandidateUrl = null;
    s.prevAngleForDelta = null;
    s.postureHint = null;
    keyFramesRef.current.neutralUrl = null;
    keyFramesRef.current.peakUrl = null;
    setShowResult(false);
    // For calibrated-rotation tests, "Reset Peak" also clears the
    // calibration so the operator can re-baseline (e.g. patient
    // shifted position between attempts).
    if (isNeckRotation) setBaseline(null);
    if (isShoulderRotation) setShoulderBaseline(null);
    setVersion((v) => v + 1);
  }

  // ── derived render values ────────────────────────────────────
  const { current, peakSigned, validFrames, totalFrames, status, apiError, postureHint } =
    stateRef.current;
  const peakMag = peakSigned !== null ? Math.abs(peakSigned) : 0;
  const hasPeak = peakMag > 0;

  const instructions = getInstructions(bodyPart, movementId);

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

  const liveLayout = (
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

      {/* ─── 2-column layout (instructions+status | camera) ──── */}
      <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
        {/* ─── LEFT ────────────────────────────────────────────── */}
        <div className="space-y-5">
          {instructions.length > 0 && (
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Movement instructions
            </p>
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
                {current !== null ? `${fmt(current, 1)}°` : "—"}
              </p>
            </div>
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
          </div>

          <p className="mt-5 text-xs text-muted">
            Target range:{" "}
            <span className="tabular text-foreground">
              {target[0]}°–{target[1]}°
            </span>
            . Capture is continuous — perform the movement, then click{" "}
            <span className="text-foreground">Show Analysis</span> at the peak.
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
                  : movementId === "external_rotation"
                    ? " Rotate the forearm outward to the peak."
                    : " Rotate the forearm inward to the peak."}
              </p>
            </div>
          )}

          <div className="mt-5 flex gap-2">
            <Button
              onClick={() => setShowResult(true)}
              disabled={!hasPeak || (isCalibratedRotation && !calibrationLocked)}
              className="flex-1"
            >
              <Eye className="h-4 w-4" />
              Show Analysis
            </Button>
            <Button
              variant="secondary"
              onClick={resetPeak}
              disabled={!hasPeak && !(isCalibratedRotation && calibrationLocked)}
            >
              <RotateCcw className="h-4 w-4" />
              {isCalibratedRotation && calibrationLocked ? "Re-baseline" : "Reset Peak"}
            </Button>
          </div>
        </div>

      </div>

      {/* ─── RIGHT: camera + skeleton (sticky so it tracks the user as they scroll) ─── */}
      <div className="lg:sticky lg:top-28">
        <LiveBiomechCamera
          bodyPart={bodyPart}
          movement={movementId}
          side={side}
          onResult={onResult}
          onError={onError}
          neckRotationBaseline={isNeckRotation ? baseline : null}
          shoulderRotationBaseline={isShoulderRotation ? shoulderBaseline : null}
          onSmoothedKeypoints={isCalibratedRotation ? onSmoothedKeypointsTracked : undefined}
        />
        <p className="mt-3 text-xs text-subtle">
          Start the camera and perform the movement. The on-screen skeleton tracks
          your joints in real time — keep the relevant limbs inside the frame.
        </p>
        </div>
      </div>

      {/* ─── Unified report disclaimer ──────────────────────────── */}
      <ReportDisclaimer />
    </div>
  );

  if (showResult && hasPeak) {
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
    if (keyFramesRef.current.peakUrl) {
      liveKeyFrames.push({
        label: `Peak ${reportName} (${peakMag.toFixed(1)}°)`,
        frame_index: 1,
        image_data_url: keyFramesRef.current.peakUrl,
      });
    }

    return (
      <div className="space-y-8">
        <AssessmentReport
          bodyPart={bodyPart}
          movementName={reportName}
          movementId={movementId}
          measured={peakMag}
          target={target}
          side={side}
          keyFrames={liveKeyFrames}
        />

        {/* Explicit save button — only renders in doctor flow */}
        <SaveToPatientButton
          buildPayload={() => ({
            module: "biomech",
            body_part: bodyPart,
            movement: movementId,
            side,
            metrics: {
              peak_magnitude: peakMag,
              peak_signed: peakSigned,
              target,
              valid_frames: validFrames,
              total_frames: totalFrames,
              // Persist annotated screenshots so the saved-report
              // viewer can show the same key-frame strip later.
              key_frames: liveKeyFrames,
            },
          })}
        />

        <div className="flex justify-center gap-3 border-t border-border pt-6">
          <Button variant="secondary" onClick={resetPeak}>
            <RotateCcw className="h-4 w-4" />
            Try again
          </Button>
        </div>
      </div>
    );
  }

  // Pre-result view — gentle reminder banner if doctor flow
  return (
    <>
      {isDoctorFlow && (
        <div className="mb-6">
          <SaveStatusBanner patient={patient} saveStatus={null} />
        </div>
      )}
      {liveLayout}
    </>
  );
}
