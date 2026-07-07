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
import { AlertTriangle, Award, Dumbbell, Flame, Target } from "lucide-react";
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
  const compensationFlags = parseSavedCompensations(metrics);

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

      {/* Compensation flags — biomech-style flagged rows harvested
          during the session and saved to metrics.compensation_flags. */}
      {compensationFlags.length > 0 && (
        <CompensationsCard flags={compensationFlags} />
      )}

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
    case "hold_in_zone":
      return <HoldInZoneSummary state={state} />;
    case "target_reach":
      return <TargetReachSummary state={state} />;
    case "trace":
      return <TraceSummary state={state} />;
    case "weight_shift":
      return <WeightShiftSummary state={state} />;
    case "match_pose":
      return <MatchPoseSummary state={state} />;
    case "metronome":
      return <MetronomeSummary state={state} />;
    default:
      return (
        <div className="rounded-card border border-border bg-surface p-5 text-sm text-muted">
          Detailed per-mechanic breakdown not available for this session.
        </div>
      );
  }
}

function formatDurationSec(sec: number): string {
  if (!Number.isFinite(sec) || sec <= 0) return "—";
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function HoldInZoneSummary({ state }: { state: Record<string, unknown> | null }) {
  const totalMs = pickNumber(state, "totalMsInZone") ?? 0;
  const bestMs = pickNumber(state, "bestDwellMs") ?? 0;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Hold summary</h3>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Cumulative in zone
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {formatDurationSec(totalMs / 1000)}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Longest single hold
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-emerald-600 dark:text-emerald-400">
            {formatDurationSec(bestMs / 1000)}
          </p>
        </div>
      </div>
    </section>
  );
}

function TargetReachSummary({ state }: { state: Record<string, unknown> | null }) {
  const hits = pickNumber(state, "hits") ?? 0;
  const misses = pickNumber(state, "misses") ?? 0;
  const total = hits + misses;
  const rate = total > 0 ? (hits / total) * 100 : 0;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Target reach summary</h3>
      <div className="mt-3 grid gap-4 sm:grid-cols-3">
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">Hits</p>
          <p className="mt-2 tabular text-3xl font-semibold text-emerald-600 dark:text-emerald-400">
            {hits}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">Misses</p>
          <p className="mt-2 tabular text-3xl font-semibold text-rose-600 dark:text-rose-400">
            {misses}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">Hit rate</p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {total > 0 ? `${rate.toFixed(0)}%` : "—"}
          </p>
        </div>
      </div>
    </section>
  );
}

function TraceSummary({ state }: { state: Record<string, unknown> | null }) {
  const samples = pickNumber(state, "samples") ?? 0;
  const accurate = pickNumber(state, "accurateSamples") ?? 0;
  const smooth = pickNumber(state, "smoothSamples") ?? 0;
  const accuracyPct = samples > 0 ? (accurate / samples) * 100 : 0;
  const smoothnessPct = samples > 0 ? (smooth / samples) * 100 : 0;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Trace quality</h3>
      <div className="mt-3 grid gap-4 sm:grid-cols-3">
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">Samples</p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {samples}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">Accuracy</p>
          <p className="mt-2 tabular text-3xl font-semibold text-emerald-600 dark:text-emerald-400">
            {samples > 0 ? `${accuracyPct.toFixed(0)}%` : "—"}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">Smoothness</p>
          <p className="mt-2 tabular text-3xl font-semibold text-sky-600 dark:text-sky-400">
            {samples > 0 ? `${smoothnessPct.toFixed(0)}%` : "—"}
          </p>
        </div>
      </div>
    </section>
  );
}

function WeightShiftSummary({ state }: { state: Record<string, unknown> | null }) {
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Weight-shift summary</h3>
      <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm text-muted">
        {state
          ? "Full zone-capture breakdown recorded in the session payload."
          : "Weight-shift session data not available for this record."}
      </div>
    </section>
  );
}

function MatchPoseSummary({ state }: { state: Record<string, unknown> | null }) {
  const bestMatchPct = pickNumber(state, "bestMatchPct") ?? 0;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Match-pose summary</h3>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Best pose match
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-emerald-600 dark:text-emerald-400">
            {bestMatchPct.toFixed(0)}%
          </p>
        </div>
      </div>
    </section>
  );
}

function MetronomeSummary({ state }: { state: Record<string, unknown> | null }) {
  const lifts = pickNumber(state, "liftCount") ?? 0;
  const perfect = pickNumber(state, "perfectCount") ?? 0;
  const good = pickNumber(state, "goodCount") ?? 0;
  const miss = pickNumber(state, "missCount") ?? 0;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Metronome summary</h3>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Total lifts
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {lifts}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Timing breakdown
          </p>
          <p className="mt-2 text-xs text-muted">
            {perfect + good + miss > 0
              ? `${perfect} perfect · ${good} good · ${miss} missed`
              : "No shell-side beat data recorded."}
          </p>
        </div>
      </div>
    </section>
  );
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

// ─── Compensation flags ─────────────────────────────────────────
// Reads metrics.compensation_flags (added by rehab pages via
// lib/rehab/compensationChecks helpers + reused biomech trackers).
// Same shape as the biomech Compensation type: type/label/severity/
// flagged/details. Only flagged=true entries are rendered.

interface CompensationFlag {
  type: string;
  label: string;
  severity: "low" | "medium" | "high";
  flagged: boolean;
  details?: string;
}

function parseSavedCompensations(
  metrics: Record<string, unknown> | null | undefined,
): CompensationFlag[] {
  if (!metrics || typeof metrics !== "object") return [];
  const raw = (metrics as Record<string, unknown>).compensation_flags;
  if (!Array.isArray(raw)) return [];
  const out: CompensationFlag[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") continue;
    const e = entry as Record<string, unknown>;
    const flagged = Boolean(e.flagged);
    if (!flagged) continue;
    const type = typeof e.type === "string" ? e.type : "compensation";
    const label = typeof e.label === "string" ? e.label : type;
    const rawSeverity = typeof e.severity === "string" ? e.severity : "medium";
    const severity: CompensationFlag["severity"] =
      rawSeverity === "low" || rawSeverity === "high" ? rawSeverity : "medium";
    const details = typeof e.details === "string" ? e.details : undefined;
    out.push({ type, label, severity, flagged: true, details });
  }
  return out;
}

function CompensationsCard({ flags }: { flags: CompensationFlag[] }) {
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Compensation flags
      </h3>
      <p className="mt-1 text-sm text-muted">
        Movement patterns detected during the session that suggest the
        target muscle group wasn't doing all the work.
      </p>
      <ul className="mt-3 space-y-2">
        {flags.map((f, i) => {
          const tone =
            f.severity === "high"
              ? "bg-rose-500/10 text-rose-700 ring-rose-500/30 dark:text-rose-300"
              : f.severity === "low"
                ? "bg-amber-500/10 text-amber-700 ring-amber-500/30 dark:text-amber-300"
                : "bg-orange-500/10 text-orange-700 ring-orange-500/30 dark:text-orange-300";
          return (
            <li
              key={`${f.type}-${i}`}
              className="flex items-start gap-3 rounded-card border border-border bg-surface p-4"
            >
              <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ring-1 ${tone}`}>
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <p className="text-sm font-semibold text-foreground">
                    {f.label}
                  </p>
                  <span className="text-[10px] uppercase tracking-[0.12em] text-subtle">
                    {f.severity}
                  </span>
                </div>
                {f.details && (
                  <p className="mt-1 text-sm text-muted">{f.details}</p>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
