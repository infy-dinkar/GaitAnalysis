"use client";
// SPPB Component 1 (Balance) — MANUAL data entry.
//
// Replaces the previous auto-detection (SPPBBalanceRecorder) which
// relied on backend MediaPipe pose analysis. Real-world video
// reliability proved too variable — patient stance geometry,
// camera angle, lighting, and clothing all produced edge cases the
// classifier couldn't handle. Manual entry by the operator is the
// authoritative SPPB protocol anyway (a clinician with a stopwatch
// is the spec instrument).
//
// Operator enters the hold time for each stage in seconds. The
// scoring rule is the SPPB spec:
//   - hold >= 10 s → Pass
//   - hold <  10 s → Fail
//   - Stage 1 fail → Stages 2 + 3 "not attempted" (protocol stops)
//   - Stage 2 fail → Stage 3 "not attempted"
//
// Emits the same {1?: StageResult, 2?: StageResult, 3?: StageResult}
// shape the orchestrator's `buildBalanceComponent()` consumes, so
// the composite SPPB scoring downstream is unchanged.

import { useState } from "react";
import { CheckCircle2, ChevronRight, XCircle, Info } from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { StageResult } from "@/lib/orthopedic/fourStageBalance";

const STAGE_HOLD_REQUIRED_SEC = 10.0;

interface Props {
  /** Fired once the operator submits. Stages 2/3 are absent
   *  ("not attempted") when a preceding stage failed. */
  onComplete: (stages: {
    1?: StageResult;
    2?: StageResult;
    3?: StageResult;
  }) => void;
}

function buildStageResult(
  stage: 1 | 2 | 3,
  durationSeconds: number,
): StageResult {
  const passed = durationSeconds >= STAGE_HOLD_REQUIRED_SEC;
  return {
    stage,
    outcome: passed ? "pass" : "fail",
    // Cap the displayed hold at 10 s (spec convention). Keep
    // duration_seconds at the actual measured time so the report
    // shows the operator-entered value.
    hold_seconds: Math.min(durationSeconds, STAGE_HOLD_REQUIRED_SEC),
    duration_seconds: durationSeconds,
    failure_mode: passed ? null : "stopped",
    sway_path_px: 0,
    sway_95_ellipse_px2: 0,
    hip_path: [],
    samples: [],
    keypoints: [],
    screenshot_data_url: null,
  };
}

export function SPPBBalanceManualEntry({ onComplete }: Props) {
  const [s1, setS1] = useState<string>("");
  const [s2, setS2] = useState<string>("");
  const [s3, setS3] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Parsed numeric values + per-stage validity
  const s1Num = s1.trim() === "" ? NaN : Number(s1);
  const s2Num = s2.trim() === "" ? NaN : Number(s2);
  const s3Num = s3.trim() === "" ? NaN : Number(s3);

  const s1Pass = !Number.isNaN(s1Num) && s1Num >= STAGE_HOLD_REQUIRED_SEC;
  const s2Pass = !Number.isNaN(s2Num) && s2Num >= STAGE_HOLD_REQUIRED_SEC;

  // SPPB protocol: only collect data for the next stage if the
  // previous one passed. Reset downstream values when a stage flips
  // from pass to fail by updating its input.
  const stage2Available = s1Pass;
  const stage3Available = s1Pass && s2Pass;

  function handleSubmit() {
    setError(null);

    if (s1.trim() === "" || Number.isNaN(s1Num) || s1Num < 0) {
      setError("Enter Stage 1 hold time (a non-negative number in seconds).");
      return;
    }
    if (s1Num > 60) {
      setError("Stage 1 hold time looks unreasonably large (over 60 s).");
      return;
    }
    if (stage2Available) {
      if (s2.trim() === "" || Number.isNaN(s2Num) || s2Num < 0) {
        setError("Enter Stage 2 hold time.");
        return;
      }
      if (s2Num > 60) {
        setError("Stage 2 hold time looks unreasonably large (over 60 s).");
        return;
      }
    }
    if (stage3Available) {
      if (s3.trim() === "" || Number.isNaN(s3Num) || s3Num < 0) {
        setError("Enter Stage 3 hold time.");
        return;
      }
      if (s3Num > 60) {
        setError("Stage 3 hold time looks unreasonably large (over 60 s).");
        return;
      }
    }

    const stages: {
      1?: StageResult;
      2?: StageResult;
      3?: StageResult;
    } = {
      1: buildStageResult(1, s1Num),
    };
    if (stage2Available) {
      stages[2] = buildStageResult(2, s2Num);
      if (stage3Available) {
        stages[3] = buildStageResult(3, s3Num);
      }
    }
    onComplete(stages);
  }

  return (
    <div className="space-y-5">
      <div className="rounded-card border border-border bg-surface p-4 text-sm">
        <p className="flex items-start gap-2 text-foreground">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
          <span>
            Time each stage with a stopwatch while the patient performs
            it. Enter the hold duration in seconds for each. A stage
            held for <strong>{STAGE_HOLD_REQUIRED_SEC} s or more</strong>{" "}
            counts as a <strong>Pass</strong>. Per SPPB protocol, the test
            stops at the first failed stage — fields for subsequent
            stages will lock automatically.
          </span>
        </p>
      </div>

      <StageInput
        stage={1}
        label="Stage 1 — Side-by-side stance"
        description="Patient stands with both feet next to each other, toes and heels in line."
        value={s1}
        onChange={(v) => {
          setS1(v);
          setError(null);
        }}
        disabled={false}
      />

      <StageInput
        stage={2}
        label="Stage 2 — Semi-tandem stance"
        description="One foot half-step forward; the heel of the moved foot beside the big toe of the other foot."
        value={s2}
        onChange={(v) => {
          setS2(v);
          setError(null);
        }}
        disabled={!stage2Available}
        disabledReason={!s1Pass ? "Stage 1 must hold ≥ 10 s before Stage 2 is attempted." : undefined}
      />

      <StageInput
        stage={3}
        label="Stage 3 — Tandem stance (heel-to-toe)"
        description="One foot directly in front of the other in a single line. Heel of front foot touches toes of back foot."
        value={s3}
        onChange={(v) => {
          setS3(v);
          setError(null);
        }}
        disabled={!stage3Available}
        disabledReason={
          !s1Pass
            ? "Stage 1 must hold ≥ 10 s before Stage 3 is attempted."
            : !s2Pass
              ? "Stage 2 must hold ≥ 10 s before Stage 3 is attempted."
              : undefined
        }
      />

      {error && (
        <div className="rounded-card border border-error/40 bg-error/5 p-3 text-xs text-error">
          {error}
        </div>
      )}

      <div>
        <Button onClick={handleSubmit}>
          <ChevronRight className="h-4 w-4" />
          Submit balance results
        </Button>
      </div>
    </div>
  );
}

function StageInput({
  stage,
  label,
  description,
  value,
  onChange,
  disabled,
  disabledReason,
}: {
  stage: 1 | 2 | 3;
  label: string;
  description: string;
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
  disabledReason?: string;
}) {
  const num = value.trim() === "" ? NaN : Number(value);
  const valid = !Number.isNaN(num) && num >= 0;
  const passed = valid && num >= STAGE_HOLD_REQUIRED_SEC;
  const failed = valid && num < STAGE_HOLD_REQUIRED_SEC;

  const borderTone = disabled
    ? "border-border opacity-50"
    : passed
      ? "border-emerald-500/40 bg-emerald-500/5"
      : failed
        ? "border-red-500/40 bg-red-500/5"
        : "border-border bg-surface";

  return (
    <div className={`rounded-card border p-4 ${borderTone}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Stage {stage}
          </p>
          <p className="mt-1 text-sm font-medium text-foreground">{label}</p>
          <p className="mt-1 text-xs text-muted">{description}</p>
        </div>
        {passed && (
          <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
            <CheckCircle2 className="h-3 w-3" />
            Pass
          </span>
        )}
        {failed && (
          <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-red-500/15 px-2 py-0.5 text-[11px] font-medium text-red-700 dark:text-red-400">
            <XCircle className="h-3 w-3" />
            Fail
          </span>
        )}
      </div>
      <div className="mt-3 flex items-center gap-3">
        <input
          type="number"
          inputMode="decimal"
          step="0.1"
          min="0"
          max="60"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="Hold time"
          className="w-32 rounded-md border border-border bg-background px-3 py-2 text-sm tabular text-foreground placeholder:text-subtle focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent disabled:cursor-not-allowed disabled:opacity-60"
        />
        <span className="text-xs text-muted">seconds</span>
      </div>
      {disabled && disabledReason && (
        <p className="mt-2 text-[11px] italic text-subtle">{disabledReason}</p>
      )}
    </div>
  );
}
