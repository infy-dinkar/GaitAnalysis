"use client";
// Saved-report viewer for the rehab module. One saved report =
// one rehab session (K1 Squat, K5 Wall Sit, ...). The metrics blob
// carries a provider-agnostic shape (see app/rehab/squat/page.tsx's
// buildRehabPayload for the schema); this component unpacks it and
// renders a clean single-session detail card.
//
// Signature mirrors every other SavedXxxReport (SavedSingleLegSquatReport
// et al.) so the dispatch in reports/[id]/page.tsx stays uniform. The
// whole render is DOM-only — PDF export via html2canvas + jsPDF works
// out of the box.
//
// Extension path: the mechanic_id switch below currently handles
// rep_count only (the mechanic K1 uses). Adding another mechanic
// (hold_in_zone, target_reach, ...) is a new case in the switch —
// no schema change.

import { useMemo } from "react";
import { Award, Dumbbell, Flame, Target } from "lucide-react";
import {
  drawSkeletonFromLandmarks,
  type SavedLandmark,
  type SourceFrame,
} from "@/lib/pose/skeletonDraw";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  patientName: string | null;
  patient?: PatientDTO | null;
  metrics: Record<string, unknown>;
  observations: Record<string, unknown>;
}

// ─── Shape helpers (defensive — old saved sessions may be partial) ─
interface SignalBlock {
  name: string;
  unit: string;
  value_at_peak: number;
  target_band?: { min: number; max: number };
}
interface ScoreBlock {
  points: number;
  streak: number;
  bestStreak: number;
}

function pickNumber(o: unknown, key: string): number | null {
  if (!o || typeof o !== "object") return null;
  const v = (o as Record<string, unknown>)[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function pickString(o: unknown, key: string): string | null {
  if (!o || typeof o !== "object") return null;
  const v = (o as Record<string, unknown>)[key];
  return typeof v === "string" && v.length > 0 ? v : null;
}

function pickObject(o: unknown, key: string): Record<string, unknown> | null {
  if (!o || typeof o !== "object") return null;
  const v = (o as Record<string, unknown>)[key];
  return v && typeof v === "object" ? (v as Record<string, unknown>) : null;
}

function humanExerciseName(slug: string | null): string {
  if (!slug) return "Rehab exercise";
  return slug
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function humanMechanicName(id: string | null): string {
  switch (id) {
    case "rep_count": return "Rep-Count";
    case "hold_in_zone": return "Hold-in-Zone";
    case "target_reach": return "Target-Reach";
    case "trace": return "Trace";
    case "weight_shift": return "Weight-Shift";
    case "match_pose": return "Match-Pose";
    case "metronome": return "Metronome";
    default: return "Rehab mechanic";
  }
}

function formatDuration(sec: number): string {
  if (!Number.isFinite(sec) || sec <= 0) return "—";
  if (sec < 60) return `${sec.toFixed(0)} s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m} min ${s.toString().padStart(2, "0")} s`;
}

export function SavedRehabReport({
  patientName,
  patient: _patient,
  metrics,
  observations,
}: Props) {
  // Unpack — every field is defensive so a partial older payload
  // still renders instead of crashing the report route.
  const exerciseSlug = pickString(metrics, "exercise_slug");
  const mechanicId = pickString(metrics, "mechanic_id");
  const durationSec = pickNumber(metrics, "duration_sec") ?? 0;
  const targetReps = pickNumber(metrics, "target_reps");

  const scoreRaw = pickObject(metrics, "score");
  const score: ScoreBlock = {
    points: pickNumber(scoreRaw, "points") ?? 0,
    streak: pickNumber(scoreRaw, "streak") ?? 0,
    bestStreak: pickNumber(scoreRaw, "bestStreak") ?? 0,
  };

  const signalRaw = pickObject(metrics, "signal");
  const signal: SignalBlock | null = signalRaw
    ? {
        name: pickString(signalRaw, "name") ?? "Metric",
        unit: pickString(signalRaw, "unit") ?? "",
        value_at_peak: pickNumber(signalRaw, "value_at_peak") ?? 0,
        target_band: (() => {
          const b = pickObject(signalRaw, "target_band");
          const mn = pickNumber(b, "min");
          const mx = pickNumber(b, "max");
          return mn !== null && mx !== null ? { min: mn, max: mx } : undefined;
        })(),
      }
    : null;

  const mechanicState = pickObject(metrics, "mechanic_state");
  const interpretation = pickString(observations, "interpretation");

  const exerciseName = humanExerciseName(exerciseSlug);
  const mechanicName = humanMechanicName(mechanicId);

  return (
    <div className="space-y-8">
      {/* Header block */}
      <div className="rounded-card border border-border bg-surface p-5">
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-card bg-orange-500/10 text-orange-700 dark:text-orange-400">
            <Dumbbell className="h-6 w-6" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="eyebrow">{mechanicName} mechanic</p>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight">
              {exerciseName}
            </h2>
            <p className="mt-1 text-sm text-muted">
              {patientName?.trim() || "Anonymous patient"} · session duration{" "}
              {formatDuration(durationSec)}
            </p>
          </div>
        </div>
      </div>

      {/* Score card — game-side summary */}
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard
          icon={Award}
          label="Points"
          value={score.points.toFixed(0)}
          tone="text-amber-600 dark:text-amber-400"
        />
        <StatCard
          icon={Flame}
          label="Best streak"
          value={score.bestStreak.toFixed(0)}
          tone="text-orange-600 dark:text-orange-400"
        />
        <StatCard
          icon={Target}
          label="Final streak"
          value={score.streak.toFixed(0)}
          tone="text-emerald-600 dark:text-emerald-400"
        />
      </div>

      {/* Mechanic-specific summary */}
      <MechanicSummary
        mechanicId={mechanicId}
        state={mechanicState}
        targetReps={targetReps}
      />

      {/* Clinical metric card */}
      {signal && (
        <ClinicalMetricCard
          signalName={signal.name}
          unit={signal.unit}
          valueAtPeak={signal.value_at_peak}
          band={signal.target_band}
        />
      )}

      {/* Best-rep skeleton — redrawn from saved landmark coords onto
          a stand-alone canvas so it's centered + full-body, not a
          screenshot of whatever framing the live camera happened to
          use. Not wrapped in .no-pdf so it survives the PDF export. */}
      <SkeletonPoseCard raw={pickObject(metrics, "skeleton_pose")} />

      {/* Interpretation */}
      {interpretation && (
        <section>
          <h3 className="text-base font-semibold tracking-tight">
            Session interpretation
          </h3>
          <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
            {interpretation}
          </div>
        </section>
      )}
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────────

interface StatCardProps {
  icon: typeof Award;
  label: string;
  value: string;
  tone: string;
}

function StatCard({ icon: Icon, label, value, tone }: StatCardProps) {
  return (
    <div className="rounded-card border border-border bg-surface p-5">
      <div className="flex items-center gap-2">
        <Icon className={`h-4 w-4 ${tone}`} />
        <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
          {label}
        </p>
      </div>
      <p className="mt-2 tabular text-3xl font-semibold text-foreground">
        {value}
      </p>
    </div>
  );
}

interface MechanicSummaryProps {
  mechanicId: string | null;
  state: Record<string, unknown> | null;
  targetReps: number | null;
}

function MechanicSummary({
  mechanicId,
  state,
  targetReps,
}: MechanicSummaryProps) {
  // Switch on mechanic_id — one case per mechanic; every rehab
  // exercise routes to exactly one of these seven.
  switch (mechanicId) {
    case "rep_count":
      return <RepCountSummary state={state} targetReps={targetReps} />;
    // The other 6 mechanic summaries land here as they get wired in
    // subsequent turns (hold_in_zone → dwell time, target_reach →
    // hits/misses, etc.). Falling through to a generic block keeps
    // older / unknown mechanic payloads viewable.
    default:
      return (
        <div className="rounded-card border border-border bg-surface p-5 text-sm text-muted">
          Detailed per-mechanic breakdown not available for this session.
        </div>
      );
  }
}

function RepCountSummary({
  state,
  targetReps,
}: {
  state: Record<string, unknown> | null;
  targetReps: number | null;
}) {
  const reps = pickNumber(state, "reps") ?? 0;
  const goodReps = pickNumber(state, "goodReps") ?? 0;
  const flagged = Math.max(0, reps - goodReps);
  const goalPct = targetReps && targetReps > 0
    ? Math.min(100, (reps / targetReps) * 100)
    : null;

  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Reps summary</h3>
      <div className="mt-3 grid gap-4 sm:grid-cols-3">
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Reps completed
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {reps}
            {targetReps !== null && (
              <span className="ml-1 text-base font-normal text-muted">
                / {targetReps}
              </span>
            )}
          </p>
          {goalPct !== null && (
            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
              <div
                className="h-full bg-accent transition-all"
                style={{ width: `${goalPct}%` }}
              />
            </div>
          )}
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Clean reps
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-emerald-600 dark:text-emerald-400">
            {goodReps}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Flagged reps
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-rose-600 dark:text-rose-400">
            {flagged}
          </p>
          <p className="mt-1 text-[11px] text-muted">
            shallow / jerky / off-target
          </p>
        </div>
      </div>
    </section>
  );
}

interface SkeletonPoseCardProps {
  raw: Record<string, unknown> | null;
}

// Target canvas dims — portrait, full-body friendly, matches
// clinical skeleton illustrations. Rendered at 2× DPR then
// CSS-scaled for retina sharpness.
const POSE_CANVAS_CSS_W = 320;
const POSE_CANVAS_CSS_H = 480;
const POSE_CANVAS_DPR = 2;

function SkeletonPoseCard({ raw }: SkeletonPoseCardProps) {
  const dataUrl = useMemo<string | null>(() => {
    if (typeof document === "undefined" || !raw) return null;
    const landmarks = raw.landmarks as SavedLandmark[] | undefined;
    const sourceFrame = raw.source_frame as SourceFrame | undefined;
    if (
      !Array.isArray(landmarks) || landmarks.length === 0
      || !sourceFrame
      || typeof sourceFrame.width !== "number"
      || typeof sourceFrame.height !== "number"
      || sourceFrame.width <= 0 || sourceFrame.height <= 0
    ) {
      return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = POSE_CANVAS_CSS_W * POSE_CANVAS_DPR;
    canvas.height = POSE_CANVAS_CSS_H * POSE_CANVAS_DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(POSE_CANVAS_DPR, 0, 0, POSE_CANVAS_DPR, 0, 0);
    drawSkeletonFromLandmarks(
      ctx,
      landmarks,
      sourceFrame,
      POSE_CANVAS_CSS_W,
      POSE_CANVAS_CSS_H,
      {
        // Transparent background — the surrounding card's own
        // surface shows through, so the report reads clean on both
        // light + dark themes. Warm palette (red bones + orange
        // dots) pops against the dark-theme card surface where the
        // previous slate-on-slate combo was invisible.
        background: null,
        boneColor: "#EF4444",
        boneWidth: 4,
        dotColor: "#F97316",
        dotRadius: 6,
      },
    );
    return canvas.toDataURL("image/png");
  }, [raw]);

  if (!dataUrl) return null;
  const label = pickString(raw, "label");

  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Best-rep skeleton
      </h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <div className="flex justify-center bg-surface p-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={dataUrl}
            alt={label ?? "Skeleton redrawn from best-rep landmarks"}
            width={POSE_CANVAS_CSS_W}
            height={POSE_CANVAS_CSS_H}
            style={{ width: POSE_CANVAS_CSS_W, height: POSE_CANVAS_CSS_H }}
            className="block"
          />
        </div>
        {label && (
          <p className="border-t border-border bg-surface px-4 py-2 text-xs text-muted">
            {label}
          </p>
        )}
      </div>
    </section>
  );
}

interface ClinicalMetricCardProps {
  signalName: string;
  unit: string;
  valueAtPeak: number;
  band?: { min: number; max: number };
}

function ClinicalMetricCard({
  signalName,
  unit,
  valueAtPeak,
  band,
}: ClinicalMetricCardProps) {
  const humanName = signalName.replace(/_/g, " ").replace(
    /\b\w/g,
    (c) => c.toUpperCase(),
  );
  const inBand = band
    ? valueAtPeak >= band.min && valueAtPeak <= band.max
    : null;

  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Clinical metric — {humanName}
      </h3>
      <div className="mt-3 rounded-card border border-border bg-surface p-5">
        <div className="flex flex-wrap items-end gap-6">
          <div>
            <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
              Best value
            </p>
            <p className="mt-1 tabular text-4xl font-semibold text-foreground">
              {valueAtPeak.toFixed(1)}
              <span className="ml-1 text-lg font-normal text-muted">
                {unit}
              </span>
            </p>
          </div>
          {band && (
            <div>
              <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
                Target band
              </p>
              <p className="mt-1 tabular text-lg font-medium text-foreground">
                {band.min.toFixed(0)}–{band.max.toFixed(0)} {unit}
              </p>
            </div>
          )}
          {inBand !== null && (
            <span
              className={`ml-auto rounded-full px-3 py-1 text-xs font-semibold ring-1 ${
                inBand
                  ? "bg-emerald-500/15 text-emerald-700 ring-emerald-500/30 dark:text-emerald-300"
                  : "bg-amber-500/15 text-amber-700 ring-amber-500/30 dark:text-amber-300"
              }`}
            >
              {inBand ? "In target band" : "Outside target band"}
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
