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
// ─── Visual primitives (SVG/Tailwind, PDF-safe) ─────────────────
import { GoalVsActualBar } from "@/components/rehab/report/GoalVsActualBar";
import { PercentRing } from "@/components/rehab/report/PercentRing";
import { TargetVsValue } from "@/components/rehab/report/TargetVsValue";
import { DurationChip } from "@/components/rehab/report/DurationChip";

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

      {/* Mechanic-specific summary — each branch composes the
          visuals relevant to its own metrics (reps + angle for
          rep_count, in-zone % for hold_in_zone, hit rate for
          target_reach, etc.). All primitives fall back gracefully
          when a field is missing so older sessions don't crash. */}
      <MechanicSummary
        mechanicId={mechanicId}
        state={mechanicState}
        targetReps={targetReps}
        targetHoldMs={pickNumber(metrics, "target_hold_ms")}
        durationSec={durationSec}
        signal={signal}
        config={pickObject(metrics, "config")}
      />

      {/* Clinical metric card — target-band vs value visual for any
          mechanic that ships signal + target_band. */}
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
  targetHoldMs: number | null;
  durationSec: number;
  signal: SignalBlock | null;
  config: Record<string, unknown> | null;
}

function MechanicSummary({
  mechanicId,
  state,
  targetReps,
  targetHoldMs,
  durationSec,
  signal,
  config,
}: MechanicSummaryProps) {
  // Switch on mechanic_id — one case per mechanic; every rehab
  // exercise routes to exactly one of these seven.
  switch (mechanicId) {
    case "rep_count":
      return (
        <RepCountSummary
          state={state}
          targetReps={targetReps}
          durationSec={durationSec}
          signal={signal}
        />
      );
    case "hold_in_zone":
      return (
        <HoldInZoneSummary
          state={state}
          targetHoldMs={targetHoldMs}
          durationSec={durationSec}
          signal={signal}
        />
      );
    case "target_reach":
      return (
        <TargetReachSummary
          state={state}
          durationSec={durationSec}
          signal={signal}
        />
      );
    case "trace":
      return <TraceSummary state={state} durationSec={durationSec} />;
    case "weight_shift":
      return (
        <WeightShiftSummary
          state={state}
          durationSec={durationSec}
          config={config}
        />
      );
    case "match_pose":
      return <MatchPoseSummary state={state} durationSec={durationSec} />;
    case "metronome":
      return (
        <MetronomeSummary
          state={state}
          durationSec={durationSec}
          config={config}
        />
      );
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

// ─── Mechanic-summary components ─────────────────────────────────
// Each mechanic composes the primitives that fit its saved metrics.
// Every primitive falls back gracefully so partial/older sessions
// simply hide missing panels instead of crashing.

function RepCountSummary({
  state,
  targetReps,
  durationSec,
  signal,
}: {
  state: Record<string, unknown> | null;
  targetReps: number | null;
  durationSec: number;
  signal: SignalBlock | null;
}) {
  const reps = pickNumber(state, "reps") ?? 0;
  const goodReps = pickNumber(state, "goodReps") ?? 0;
  const cleanPct = reps > 0 ? (goodReps / reps) * 100 : 0;
  const goal = targetReps ?? 0;

  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight">Reps summary</h3>
        <DurationChip seconds={durationSec} />
      </div>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <GoalVsActualBar
          label="Reps completed"
          actual={reps}
          goal={goal}
          unit="reps"
          tone="accent"
          caption={
            goal > 0
              ? `${goodReps} clean · ${Math.max(0, reps - goodReps)} flagged`
              : `${goodReps} clean of ${reps}`
          }
        />
        <PercentRing
          label="Clean-rep rate"
          value={cleanPct}
          tone={cleanPct >= 80 ? "emerald" : cleanPct >= 50 ? "amber" : "rose"}
          subtext={`${goodReps} clean / ${reps} total`}
        />
      </div>
      {signal && (
        <div className="mt-4">
          <TargetVsValue
            label={humanizeSignal(signal.name)}
            value={signal.value_at_peak}
            unit={signal.unit}
            band={signal.target_band}
          />
        </div>
      )}
    </section>
  );
}

function HoldInZoneSummary({
  state,
  targetHoldMs,
  durationSec,
  signal,
}: {
  state: Record<string, unknown> | null;
  targetHoldMs: number | null;
  durationSec: number;
  signal: SignalBlock | null;
}) {
  const totalMs = pickNumber(state, "totalMsInZone") ?? 0;
  const bestMs = pickNumber(state, "bestDwellMs") ?? 0;
  const targetMs = targetHoldMs && targetHoldMs > 0 ? targetHoldMs : null;
  const inZonePct = targetMs
    ? Math.min(100, (totalMs / targetMs) * 100)
    : durationSec > 0
      ? Math.min(100, (totalMs / (durationSec * 1000)) * 100)
      : 0;
  const totalSec = totalMs / 1000;
  const bestSec = bestMs / 1000;
  const goalSec = targetMs ? targetMs / 1000 : 0;
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight">Hold summary</h3>
        <DurationChip seconds={durationSec} />
      </div>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <PercentRing
          label={targetMs ? "Hold completion" : "In-zone share"}
          value={inZonePct}
          tone={inZonePct >= 80 ? "emerald" : inZonePct >= 40 ? "amber" : "rose"}
          subtext={
            targetMs
              ? `${totalSec.toFixed(1)}s / ${goalSec.toFixed(0)}s target`
              : `${totalSec.toFixed(1)}s in zone`
          }
        />
        <GoalVsActualBar
          label="Longest single hold"
          actual={bestSec}
          goal={goalSec}
          unit="s"
          tone="emerald"
        />
      </div>
      {signal && (
        <div className="mt-4">
          <TargetVsValue
            label={humanizeSignal(signal.name)}
            value={signal.value_at_peak}
            unit={signal.unit}
            band={signal.target_band}
          />
        </div>
      )}
    </section>
  );
}

function TargetReachSummary({
  state,
  durationSec,
  signal,
}: {
  state: Record<string, unknown> | null;
  durationSec: number;
  signal: SignalBlock | null;
}) {
  const hits = pickNumber(state, "hits") ?? 0;
  const misses = pickNumber(state, "misses") ?? 0;
  const total = pickNumber(state, "totalTargets") ?? hits + misses;
  const hitRate = total > 0 ? (hits / total) * 100 : 0;
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight">Target reach summary</h3>
        <DurationChip seconds={durationSec} />
      </div>
      {total > 0 ? (
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <GoalVsActualBar
            label="Targets hit"
            actual={hits}
            goal={total}
            unit="targets"
            tone="emerald"
            caption={`${misses} missed`}
          />
          <PercentRing
            label="Hit rate"
            value={hitRate}
            tone={hitRate >= 70 ? "emerald" : hitRate >= 40 ? "amber" : "rose"}
            subtext={`${hits} of ${total}`}
          />
        </div>
      ) : (
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm text-muted">
          Hit / miss counts weren&apos;t recorded on this older session — see peak reach below.
        </div>
      )}
      {signal && (
        <div className="mt-4">
          <TargetVsValue
            label={humanizeSignal(signal.name)}
            value={signal.value_at_peak}
            unit={signal.unit}
            band={signal.target_band}
          />
        </div>
      )}
    </section>
  );
}

function TraceSummary({
  state,
  durationSec,
}: {
  state: Record<string, unknown> | null;
  durationSec: number;
}) {
  const samples = pickNumber(state, "samples") ?? 0;
  const accurate = pickNumber(state, "accurateSamples") ?? 0;
  const smooth = pickNumber(state, "smoothSamples") ?? 0;
  const accuracyPct = samples > 0 ? (accurate / samples) * 100 : 0;
  const smoothnessPct = samples > 0 ? (smooth / samples) * 100 : 0;
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight">Trace quality</h3>
        <DurationChip seconds={durationSec} />
      </div>
      {samples > 0 ? (
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <PercentRing
            label="Accuracy"
            value={accuracyPct}
            tone={accuracyPct >= 70 ? "emerald" : accuracyPct >= 40 ? "amber" : "rose"}
            subtext={`${accurate.toFixed(0)} on-path / ${samples} samples`}
          />
          <PercentRing
            label="Smoothness"
            value={smoothnessPct}
            tone={smoothnessPct >= 70 ? "sky" : "amber"}
            subtext={`${smooth.toFixed(0)} smooth / ${samples} samples`}
          />
        </div>
      ) : (
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm text-muted">
          No trace samples recorded for this session.
        </div>
      )}
    </section>
  );
}

function WeightShiftSummary({
  state,
  durationSec,
  config,
}: {
  state: Record<string, unknown> | null;
  durationSec: number;
  config: Record<string, unknown> | null;
}) {
  const captured = pickNumber(state, "zonesCaptured");
  const totalFromState = pickNumber(state, "totalZones");
  const totalFromConfig = Array.isArray(config?.zones)
    ? (config?.zones as unknown[]).length
    : null;
  const total = totalFromState ?? totalFromConfig ?? 0;
  const maxExcursion = pickNumber(state, "maxExcursion") ?? 0;
  const stepCount = pickNumber(state, "stepCount") ?? 0;
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight">Weight-shift summary</h3>
        <DurationChip seconds={durationSec} />
      </div>
      {captured !== null && total > 0 ? (
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <GoalVsActualBar
            label="Zones captured"
            actual={captured}
            goal={total}
            unit="zones"
            tone="emerald"
          />
          <div className="grid gap-3">
            <div className="rounded-card border border-border bg-surface p-4">
              <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
                Max lateral excursion
              </p>
              <p className="mt-1 tabular text-2xl font-semibold text-foreground">
                {(maxExcursion * 100).toFixed(0)}
                <span className="ml-1 text-sm text-muted">% of range</span>
              </p>
            </div>
            <div className="rounded-card border border-border bg-surface p-4">
              <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
                Steps detected
              </p>
              <p className="mt-1 tabular text-2xl font-semibold text-foreground">
                {stepCount}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm text-muted">
          Zone-capture breakdown not available for this session.
        </div>
      )}
    </section>
  );
}

function MatchPoseSummary({
  state,
  durationSec,
}: {
  state: Record<string, unknown> | null;
  durationSec: number;
}) {
  const bestMatchPct = pickNumber(state, "bestMatchPct") ?? 0;
  const bestDwellMs = pickNumber(state, "bestDwellMs");
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight">Match-pose summary</h3>
        <DurationChip seconds={durationSec} />
      </div>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        <PercentRing
          label="Best pose match"
          value={bestMatchPct}
          tone={bestMatchPct >= 70 ? "emerald" : bestMatchPct >= 50 ? "amber" : "rose"}
          subtext="Peak alignment score"
        />
        {bestDwellMs !== null && (
          <div className="rounded-card border border-border bg-surface p-4">
            <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
              Longest held match
            </p>
            <p className="mt-1 tabular text-2xl font-semibold text-foreground">
              {(bestDwellMs / 1000).toFixed(1)}
              <span className="ml-1 text-sm text-muted">s</span>
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

function MetronomeSummary({
  state,
  durationSec,
  config,
}: {
  state: Record<string, unknown> | null;
  durationSec: number;
  config: Record<string, unknown> | null;
}) {
  const perfect = pickNumber(state, "perfect") ?? pickNumber(state, "perfectCount") ?? 0;
  const good = pickNumber(state, "good") ?? pickNumber(state, "goodCount") ?? 0;
  const miss = pickNumber(state, "miss") ?? pickNumber(state, "missCount") ?? 0;
  const total =
    pickNumber(state, "totalBeats") ?? perfect + good + miss;
  const onBeatPct =
    pickNumber(state, "onBeatPct")
    ?? (total > 0 ? ((perfect + good) / total) * 100 : 0);
  const lifts = pickNumber(state, "liftCount") ?? 0;
  const bpm = pickNumber(config, "bpm");
  const perfectPct = total > 0 ? (perfect / total) * 100 : 0;
  const goodPct = total > 0 ? (good / total) * 100 : 0;
  const missPct = total > 0 ? (miss / total) * 100 : 0;
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight">Metronome summary</h3>
        <DurationChip seconds={durationSec} />
      </div>
      {total > 0 ? (
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <PercentRing
            label="On-beat rate"
            value={onBeatPct}
            tone={onBeatPct >= 70 ? "emerald" : onBeatPct >= 40 ? "amber" : "rose"}
            subtext={
              bpm !== null
                ? `${lifts} lifts · ${bpm.toFixed(0)} bpm target`
                : `${lifts} lifts across ${total} beats`
            }
          />
          <div className="rounded-card border border-border bg-surface p-4">
            <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
              Timing breakdown
            </p>
            <div className="mt-3 flex h-2 w-full overflow-hidden rounded-full bg-elevated">
              <div
                className="h-full bg-emerald-500"
                style={{ width: `${perfectPct}%` }}
                title={`${perfect} perfect`}
              />
              <div
                className="h-full bg-amber-500"
                style={{ width: `${goodPct}%` }}
                title={`${good} good`}
              />
              <div
                className="h-full bg-rose-500"
                style={{ width: `${missPct}%` }}
                title={`${miss} missed`}
              />
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
              <div>
                <p className="tabular font-semibold text-emerald-600 dark:text-emerald-400">
                  {perfect}
                </p>
                <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">
                  Perfect
                </p>
              </div>
              <div>
                <p className="tabular font-semibold text-amber-600 dark:text-amber-400">
                  {good}
                </p>
                <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">
                  Good
                </p>
              </div>
              <div>
                <p className="tabular font-semibold text-rose-600 dark:text-rose-400">
                  {miss}
                </p>
                <p className="text-[10px] uppercase tracking-[0.12em] text-subtle">
                  Missed
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
              Total lifts
            </p>
            <p className="mt-2 tabular text-3xl font-semibold text-foreground">
              {lifts}
            </p>
            <p className="mt-1 text-[11px] text-muted">
              Beat-timing data not recorded on this older session.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}

function humanizeSignal(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
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

  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">
        Clinical metric — {humanName}
      </h3>
      <div className="mt-3 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        {/* Numeric summary on the left — preserves the previous
            "Best value / Target band" readout for at-a-glance
            quantitative reference. */}
        <div className="rounded-card border border-border bg-surface p-5">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Best value
          </p>
          <p className="mt-1 tabular text-4xl font-semibold text-foreground">
            {valueAtPeak.toFixed(1)}
            <span className="ml-1 text-lg font-normal text-muted">
              {unit}
            </span>
          </p>
          {band && (
            <div className="mt-3">
              <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
                Target band
              </p>
              <p className="mt-1 tabular text-sm font-medium text-foreground">
                {band.min.toFixed(0)}–{band.max.toFixed(0)} {unit}
              </p>
            </div>
          )}
        </div>
        {/* Visual: target band + patient value on a number line —
            replaces the plain "in band / outside" pill. */}
        <TargetVsValue
          label={humanName}
          value={valueAtPeak}
          unit={unit}
          band={band}
        />
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
