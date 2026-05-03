"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Eye,
  RotateCcw,
} from "lucide-react";
import { LiveBiomechCamera } from "@/components/biomech/LiveBiomechCamera";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { Button } from "@/components/ui/Button";
import { fmt } from "@/lib/utils";
import { getInstructions, isRotationMovement } from "@/lib/biomech/instructions";
import type { LiveBiomechFrameDataDTO } from "@/lib/api";

type PostureStatus = "idle" | "good" | "low_visibility" | "no_landmarks";

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
  const stateRef = useRef({
    current: null as number | null,
    peakSigned: null as number | null,
    validFrames: 0,
    totalFrames: 0,
    status: "idle" as PostureStatus,
    apiError: null as string | null,
  });

  const [, setVersion] = useState(0);
  const [showResult, setShowResult] = useState(false);

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
  }, []);

  const onError = useCallback((msg: string) => {
    stateRef.current.apiError = msg;
  }, []);

  function resetPeak() {
    const s = stateRef.current;
    s.peakSigned = null;
    s.validFrames = 0;
    s.totalFrames = 0;
    setShowResult(false);
    setVersion((v) => v + 1);
  }

  // ── derived render values ────────────────────────────────────
  const { current, peakSigned, validFrames, totalFrames, status, apiError } =
    stateRef.current;
  const peakMag = peakSigned !== null ? Math.abs(peakSigned) : 0;
  const hasPeak = peakMag > 0;

  const instructions = getInstructions(bodyPart, movementId);
  const isRotation = isRotationMovement(bodyPart, movementId);

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

          <div className="mt-5 flex gap-2">
            <Button
              onClick={() => setShowResult(true)}
              disabled={!hasPeak}
              className="flex-1"
            >
              <Eye className="h-4 w-4" />
              Show Analysis
            </Button>
            <Button variant="secondary" onClick={resetPeak} disabled={!hasPeak}>
              <RotateCcw className="h-4 w-4" />
              Reset Peak
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
        />
        <p className="mt-3 text-xs text-subtle">
          Start the camera and perform the movement. The on-screen skeleton tracks
          your joints in real time — keep the relevant limbs inside the frame.
        </p>
        </div>
      </div>

      {/* ─── Subtle disclaimer footer ─────────────────────────── */}
      <p className="border-t border-border/60 pt-4 text-center text-[11px] leading-relaxed text-subtle/80">
        Measurements are estimated from a single 2D camera and are intended for
        movement tracking and self-screening
        {isRotation ? " — rotation values are approximate. " : ". "}
        For clinical-grade range-of-motion assessment, please consult a qualified
        practitioner.
      </p>
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

        <div className="flex justify-center gap-3 border-t border-border pt-6">
          <Button variant="secondary" onClick={resetPeak}>
            <RotateCcw className="h-4 w-4" />
            Try again
          </Button>
        </div>
      </div>
    );
  }

  return liveLayout;
}
