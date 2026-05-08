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

type PostureStatus = "idle" | "good" | "low_visibility" | "no_landmarks";

// Auto-calibration timing for neck rotation. The patient has to be
// stably facing the camera for this long continuously before the
// baseline is auto-locked. Operator can also click "Lock baseline"
// at any time to skip the auto-detect.
const CALIBRATION_STABLE_MS = 1500;

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
  });

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
      return;
    }
    s.status = data.status as PostureStatus;
    s.apiError = null;
    if (data.status === "good" && data.current_angle !== null) {
      // Suppress angle / peak tracking during the calibration phase
      // for rotation tests. Until baseline is locked, the camera is
      // running the legacy fallback formula whose readings are
      // meaningless (and can saturate well past the anatomical max)
      // — surfacing them as Current / Peak pollutes the trial's
      // actual peak measurement.
      const inCalibration = isCalibratedRotation && !calibrationLocked;
      if (inCalibration) {
        s.current = null;
        return;
      }
      s.current = data.current_angle;
      s.validFrames += 1;
      if (
        s.peakSigned === null ||
        Math.abs(data.current_angle) > Math.abs(s.peakSigned)
      ) {
        s.peakSigned = data.current_angle;
      }
    } else {
      s.current = null;
    }
  }, [isCalibratedRotation, calibrationLocked]);

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
            }
          } else if (isShoulderRotation) {
            const cal = captureShoulderRotationBaseline(kp, sideOrRight);
            if (cal) {
              setShoulderBaseline(cal);
              s.peakSigned = null;
              s.validFrames = 0;
              s.totalFrames = 0;
              s.current = null;
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
  }

  function resetPeak() {
    const s = stateRef.current;
    s.peakSigned = null;
    s.validFrames = 0;
    s.totalFrames = 0;
    s.calibStableSinceMs = null;
    s.calibFacingForward = false;
    setShowResult(false);
    // For calibrated-rotation tests, "Reset Peak" also clears the
    // calibration so the operator can re-baseline (e.g. patient
    // shifted position between attempts).
    if (isNeckRotation) setBaseline(null);
    if (isShoulderRotation) setShoulderBaseline(null);
    setVersion((v) => v + 1);
  }

  // ── derived render values ────────────────────────────────────
  const { current, peakSigned, validFrames, totalFrames, status, apiError } =
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
    return (
      <div className="space-y-8">
        <AssessmentReport
          bodyPart={bodyPart}
          movementName={reportName}
          movementId={movementId}
          measured={peakMag}
          target={target}
          side={side}
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
