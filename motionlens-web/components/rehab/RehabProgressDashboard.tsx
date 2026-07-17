"use client";
// Rehab progress dashboard — Kemtai / ViFive style.
//
// Fetches the patient's saved rehab sessions (via listPatientReports
// then N × getReport for full metrics), aggregates client-side, and
// renders:
//
//   1. Streak badge (🔥 current + best) — reuses RehabStreakBadge.
//   2. Summary cards: Workout Days · Total Sessions · Avg Completion
//      · Total Duration.
//   3. Sessions-per-day bar chart.
//   4. Duration-per-day bar chart.
//   5. Completion-rate-over-time line.
//   6. Clinical-metric progress trend (line, per-exercise picker).
//   7. Top exercises horizontal bar (goal vs completed reps/holds).
//
// Zero backend / engine / biomech / save-wiring changes. Pure
// client-side reduction of the payloads every rehab page already
// writes to `metrics.{exercise_slug, mechanic_id, score, signal,
// duration_sec, mechanic_state, config, target_reps, ...}`.

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CalendarDays,
  ChartLine,
  Dumbbell,
  Loader2,
  Percent,
  Timer,
  Trash2,
  Trophy,
} from "lucide-react";
import Link from "next/link";
import { PlotlyChart } from "@/components/gait/PlotlyChart";
import { RehabStreakBadge } from "@/components/rehab/RehabStreakBadge";
import { computeStreak, type StreakResult } from "@/lib/rehab/streak";
import { useRecommendations } from "@/lib/rehab/useRecommendations";
import {
  completedInLastDays,
  computeCompletionByWeek,
} from "@/lib/rehab/adherence";
import {
  toDisplayKneeAngle,
  KNEE_ANGLE_DISPLAY_LABEL,
} from "@/lib/rehab/kneeAngleDisplay";
import {
  deleteReport,
  getReport,
  listPatientReports,
  type ReportDTO,
  type ReportSummaryDTO,
} from "@/lib/reports";

// ═════════════════════════════════════════════════════════════════
// Types (loosely derived from what the 24 rehab pages actually save)
// ═════════════════════════════════════════════════════════════════

interface RehabSessionMetrics {
  exerciseSlug: string;
  mechanicId: string;
  points: number;
  bestStreak: number;
  durationSec: number;
  reps: number;                // RepCount only, else 0
  goodReps: number;            // RepCount only, else 0
  targetReps: number | null;   // RepCount only
  totalMsInZone: number;       // HoldInZone only, else 0
  targetHoldMs: number | null; // HoldInZone only
  hits: number;                // TargetReach only, else 0
  misses: number;              // TargetReach only, else 0
  bestMatchPct: number;        // MatchPose only, else 0
  liftCount: number;           // Metronome only, else 0
  signalName: string | null;
  signalUnit: string;
  signalValueAtPeak: number | null;
  completionPct: number;       // 0..100 (derived per-mechanic)
}

interface EnrichedSession {
  id: string;
  createdAtIso: string;
  createdDateLocal: string;    // YYYY-MM-DD local
  /** Full metrics — null when getReport failed for this row.
   *  Summary-only sessions still contribute to date-based charts. */
  metrics: RehabSessionMetrics | null;
  /** Fallback exercise slug when metrics are missing (from
   *  ReportSummary.movement — every rehab save writes the slug
   *  there too). */
  fallbackExerciseSlug: string;
}

/** Narrowed row type for metric-dependent charts. Sub-components
 *  that need `.metrics` receive this so the type never widens back
 *  to `RehabSessionMetrics | null`. */
type MetricSession = EnrichedSession & { metrics: RehabSessionMetrics };

// ═════════════════════════════════════════════════════════════════
// Defensive metric extraction
// ═════════════════════════════════════════════════════════════════

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

function toLocalIsoDate(iso: string): string {
  const d = new Date(iso);
  if (!Number.isNaN(d.getTime())) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }
  // Fallback: FastAPI's datetime serializer can emit fractional
  // seconds with more than 3 digits, which some engines refuse to
  // parse. Slice the leading YYYY-MM-DD directly so date grouping
  // still works.
  const m10 = iso.match(/^(\d{4}-\d{2}-\d{2})/);
  return m10 ? m10[1] : "";
}

function computeCompletion(
  mechanicId: string | null,
  state: Record<string, unknown> | null,
  targetReps: number | null,
  targetHoldMs: number | null,
): number {
  if (mechanicId === "rep_count") {
    const reps = pickNumber(state, "reps") ?? 0;
    if (!targetReps || targetReps <= 0) return reps > 0 ? 100 : 0;
    return Math.min(100, (reps / targetReps) * 100);
  }
  if (mechanicId === "hold_in_zone") {
    const total = pickNumber(state, "totalMsInZone") ?? 0;
    if (!targetHoldMs || targetHoldMs <= 0) return total > 0 ? 100 : 0;
    return Math.min(100, (total / targetHoldMs) * 100);
  }
  if (mechanicId === "target_reach") {
    const hits = pickNumber(state, "hits") ?? 0;
    const misses = pickNumber(state, "misses") ?? 0;
    const total = hits + misses;
    return total > 0 ? (hits / total) * 100 : 0;
  }
  if (mechanicId === "match_pose") {
    return pickNumber(state, "bestMatchPct") ?? 0;
  }
  if (mechanicId === "metronome") {
    const lifts = pickNumber(state, "liftCount") ?? 0;
    // No natural "target" for metronome — cap at 20 lifts = 100 %.
    return Math.min(100, lifts * 5);
  }
  // Trace / weight_shift — session-based; treat as "did any work".
  return 100;
}

function extractMetrics(dto: ReportDTO): RehabSessionMetrics | null {
  const m = dto.metrics;
  const exerciseSlug =
    pickString(m, "exercise_slug") ?? dto.movement ?? "unknown";
  const mechanicId = pickString(m, "mechanic_id") ?? "unknown";
  const score = pickObject(m, "score");
  const state = pickObject(m, "mechanic_state");
  const signal = pickObject(m, "signal");
  const config = pickObject(m, "config");

  const durationSec = pickNumber(m, "duration_sec") ?? 0;
  const points = pickNumber(score, "points") ?? 0;
  const bestStreak = pickNumber(score, "bestStreak") ?? 0;

  const reps = pickNumber(state, "reps") ?? 0;
  const goodReps = pickNumber(state, "goodReps") ?? 0;
  const targetReps = pickNumber(m, "target_reps");
  const totalMsInZone = pickNumber(state, "totalMsInZone") ?? 0;
  const targetHoldMs =
    pickNumber(m, "target_hold_ms") ?? pickNumber(config, "targetHoldMs");
  const hits = pickNumber(state, "hits") ?? 0;
  const misses = pickNumber(state, "misses") ?? 0;
  const bestMatchPct = pickNumber(state, "bestMatchPct") ?? 0;
  const liftCount = pickNumber(state, "liftCount") ?? 0;

  const signalName = pickString(signal, "name");
  const signalUnit = pickString(signal, "unit") ?? "";
  const signalValueAtPeak = pickNumber(signal, "value_at_peak");

  const completionPct = computeCompletion(
    mechanicId,
    state,
    targetReps,
    targetHoldMs,
  );

  return {
    exerciseSlug,
    mechanicId,
    points,
    bestStreak,
    durationSec,
    reps,
    goodReps,
    targetReps,
    totalMsInZone,
    targetHoldMs,
    hits,
    misses,
    bestMatchPct,
    liftCount,
    signalName,
    signalUnit,
    signalValueAtPeak,
    completionPct,
  };
}

function humanExerciseName(slug: string): string {
  return slug
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatHms(totalSec: number): string {
  if (!Number.isFinite(totalSec) || totalSec <= 0) return "0:00";
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec - h * 3600) / 60);
  const s = Math.floor(totalSec - h * 3600 - m * 60);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

// ═════════════════════════════════════════════════════════════════
// Component
// ═════════════════════════════════════════════════════════════════

interface Props {
  patientId: string;
  /** Optional cap on how many rehab reports to hydrate. Prevents
   *  runaway fetches for very active patients. Defaults to 60 —
   *  enough for a couple of months at daily sessions. */
  maxSessions?: number;
}

export function RehabProgressDashboard({
  patientId,
  maxSessions = 60,
}: Props) {
  const [state, setState] = useState<
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; sessions: EnrichedSession[] }
  >({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    listPatientReports(patientId)
      .then(async (res) => {
        if (cancelled) return;
        const rehabSummaries = res.data.filter(
          (r: ReportSummaryDTO) => r.module === "rehab",
        );
        if (rehabSummaries.length === 0) {
          setState({ status: "ready", sessions: [] });
          return;
        }
        const sorted = [...rehabSummaries].sort((a, b) =>
          b.created_at.localeCompare(a.created_at),
        );
        const capped = sorted.slice(0, maxSessions);

        // Seed the session list from the summaries so at LEAST
        // date-based aggregates (streak, workout days, total
        // sessions, sessions-per-day) always render — even if the
        // parallel getReport batch fails on some or all rows.
        const seeded: EnrichedSession[] = capped.map((s) => ({
          id: s.id,
          createdAtIso: s.created_at,
          createdDateLocal: toLocalIsoDate(s.created_at),
          metrics: null,
          fallbackExerciseSlug: s.movement ?? "unknown",
        }));
        seeded.sort((a, b) =>
          a.createdAtIso.localeCompare(b.createdAtIso),
        );

        // Try to enrich each with full metrics. Failures are
        // non-fatal — the seeded row stays with metrics: null and
        // the summary-only slice still contributes to date charts.
        const fulls = await Promise.all(
          capped.map((s) => getReport(s.id).catch(() => null)),
        );
        if (cancelled) return;
        const enrichedById = new Map<string, RehabSessionMetrics>();
        for (const dto of fulls) {
          if (!dto) continue;
          const metrics = extractMetrics(dto);
          if (metrics) enrichedById.set(dto.id, metrics);
        }
        const merged: EnrichedSession[] = seeded.map((row) => ({
          ...row,
          metrics: enrichedById.get(row.id) ?? null,
        }));
        setState({ status: "ready", sessions: merged });
      })
      .catch((e) => {
        if (cancelled) return;
        const msg =
          e instanceof Error ? e.message : "Could not load rehab history.";
        setState({ status: "error", message: msg });
      });
    return () => {
      cancelled = true;
    };
  }, [patientId, maxSessions]);

  // Section shell — always renders so the "Rehab progress" area is
  // visible + clearly labelled regardless of data state.
  const header = (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div>
        <p className="eyebrow">Rehab progress</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight md:text-3xl">
          {state.status === "ready"
            ? state.sessions.length > 0
              ? `${state.sessions.length} session${state.sessions.length === 1 ? "" : "s"} tracked`
              : "Progress dashboard"
            : "Progress dashboard"}
        </h2>
      </div>
    </div>
  );

  if (state.status === "loading") {
    return (
      <div className="space-y-6">
        {header}
        <div className="flex min-h-[240px] items-center justify-center rounded-card border border-border bg-surface">
          <Loader2 className="h-6 w-6 animate-spin text-accent" />
        </div>
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="space-y-6">
        {header}
        <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
          {state.message}
        </div>
      </div>
    );
  }

  const handleDeleted = (id: string) => {
    setState((prev) =>
      prev.status === "ready"
        ? { ...prev, sessions: prev.sessions.filter((s) => s.id !== id) }
        : prev,
    );
  };

  return (
    <Ready
      sessions={state.sessions}
      onDeleted={handleDeleted}
      patientId={patientId}
    />
  );
}

// ═════════════════════════════════════════════════════════════════
// Ready render — aggregates + charts
// ═════════════════════════════════════════════════════════════════

interface DailyBucket {
  date: string;              // YYYY-MM-DD local
  sessions: EnrichedSession[];
}

function Ready({
  sessions,
  onDeleted,
  patientId,
}: {
  sessions: EnrichedSession[];
  onDeleted: (id: string) => void;
  patientId: string;
}) {
  const hasSessions = sessions.length > 0;
  const sessionsWithMetrics = useMemo<MetricSession[]>(
    () =>
      sessions.filter(
        (s): s is MetricSession => s.metrics !== null,
      ),
    [sessions],
  );
  const hasMetrics = sessionsWithMetrics.length > 0;

  // Streak + workout-day counts come from `created_at` alone, so
  // they always populate even when getReport failed downstream.
  const streak = useMemo<StreakResult>(
    () => computeStreak(sessions.map((s) => s.createdAtIso)),
    [sessions],
  );

  const byDate = useMemo<DailyBucket[]>(() => {
    const map = new Map<string, EnrichedSession[]>();
    for (const s of sessions) {
      const arr = map.get(s.createdDateLocal) ?? [];
      arr.push(s);
      map.set(s.createdDateLocal, arr);
    }
    return Array.from(map.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, sess]) => ({ date, sessions: sess }));
  }, [sessions]);

  // Metric-dependent aggregates. Denominators use
  // sessionsWithMetrics so a partial-hydration state doesn't skew
  // the averages toward zero.
  const totalDurationSec = useMemo(
    () =>
      sessionsWithMetrics.reduce(
        (sum, s) => sum + (s.metrics.durationSec || 0),
        0,
      ),
    [sessionsWithMetrics],
  );

  const avgCompletionPct = useMemo(() => {
    if (sessionsWithMetrics.length === 0) return 0;
    const sum = sessionsWithMetrics.reduce(
      (s, x) => s + (x.metrics.completionPct || 0),
      0,
    );
    return sum / sessionsWithMetrics.length;
  }, [sessionsWithMetrics]);

  const exerciseSlugs = useMemo<string[]>(() => {
    const set = new Set<string>();
    for (const s of sessions) {
      const slug = s.metrics?.exerciseSlug ?? s.fallbackExerciseSlug;
      if (slug && slug !== "unknown") set.add(slug);
    }
    return Array.from(set).sort();
  }, [sessions]);

  const [pickedSlug, setPickedSlug] = useState<string>("");
  useEffect(() => {
    if (exerciseSlugs.length > 0 && !exerciseSlugs.includes(pickedSlug)) {
      setPickedSlug(exerciseSlugs[0]);
    }
  }, [exerciseSlugs, pickedSlug]);

  const trendSessions = useMemo<MetricSession[]>(
    () =>
      sessionsWithMetrics.filter(
        (s) => s.metrics.exerciseSlug === pickedSlug,
      ),
    [sessionsWithMetrics, pickedSlug],
  );

  return (
    <div className="space-y-8">
      {/* Section header — always visible so the section is
          impossible to miss even before any sessions land. */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Rehab progress</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight md:text-3xl">
            {hasSessions
              ? `${sessions.length} session${sessions.length === 1 ? "" : "s"} across ${byDate.length} day${byDate.length === 1 ? "" : "s"}`
              : "Progress dashboard"}
          </h2>
          {!hasSessions && (
            <p className="mt-1 text-sm text-muted">
              No rehab sessions saved for this patient yet. Play any
              exercise from the catalogue above — session data
              populates this dashboard automatically.
            </p>
          )}
        </div>
        <RehabStreakBadge streak={streak} />
      </div>

      {/* ── Summary cards ─────────────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={CalendarDays}
          label="Workout days"
          value={byDate.length.toString()}
          tone="text-sky-600 dark:text-sky-400"
        />
        <StatCard
          icon={Activity}
          label="Total sessions"
          value={sessions.length.toString()}
          tone="text-emerald-600 dark:text-emerald-400"
        />
        <StatCard
          icon={Percent}
          label="Avg completion"
          value={hasMetrics ? `${avgCompletionPct.toFixed(0)}%` : "—"}
          tone="text-amber-600 dark:text-amber-400"
        />
        <StatCard
          icon={Timer}
          label="Total duration"
          value={hasMetrics ? formatHms(totalDurationSec) : "—"}
          tone="text-orange-600 dark:text-orange-400"
        />
      </div>

      {/* ── Sessions per day (bar) + Duration per day (bar) ───── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard
          icon={BarChart3}
          title="Sessions per day"
          subtitle="Kemtai-style activity histogram"
        >
          {byDate.length > 0
            ? <SessionsPerDayChart buckets={byDate} />
            : <ChartEmpty message="No sessions to chart yet." />}
        </ChartCard>
        <ChartCard
          icon={Timer}
          title="Duration per day"
          subtitle="Cumulative minutes each active day"
        >
          {hasMetrics
            ? <DurationPerDayChart buckets={byDate} />
            : <ChartEmpty message={hasSessions
                ? "Session metrics still loading or unavailable."
                : "No sessions to chart yet."} />}
        </ChartCard>
      </div>

      {/* ── Completion trend + Clinical-metric progress ─────── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard
          icon={Percent}
          title="Completion rate over time"
          subtitle="Average completion % per active day"
        >
          {hasMetrics
            ? <CompletionOverTimeChart buckets={byDate} />
            : <ChartEmpty message={hasSessions
                ? "Session metrics still loading or unavailable."
                : "No sessions to chart yet."} />}
        </ChartCard>
        <ChartCard
          icon={ChartLine}
          title="Clinical metric progress"
          subtitle="Day-by-day peak signal — the ViFive improvement view"
          headerRight={
            exerciseSlugs.length > 1 && (
              <select
                value={pickedSlug}
                onChange={(e) => setPickedSlug(e.target.value)}
                className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground outline-none focus:border-accent"
              >
                {exerciseSlugs.map((slug) => (
                  <option key={slug} value={slug}>
                    {humanExerciseName(slug)}
                  </option>
                ))}
              </select>
            )
          }
        >
          {exerciseSlugs.length === 0
            ? <ChartEmpty message="No sessions to chart yet." />
            : <ProgressTrendChart sessions={trendSessions} />}
        </ChartCard>
      </div>

      {/* ── Adherence — completed side + prescribed side. Prescribed
             is resolved via getPrescribedSet (today: auto-recommended;
             later: doctor-saved prescription preferred). */}
      <AdherenceCard sessions={sessions} patientId={patientId} />

      {/* ── Top exercises (goal vs completed) ──────────────────── */}
      <ChartCard
        icon={Trophy}
        title="Top exercises"
        subtitle="Goal vs completed reps / holds across every session"
      >
        {hasMetrics
          ? <TopExercisesChart sessions={sessionsWithMetrics} />
          : <ChartEmpty message={hasSessions
              ? "Session metrics still loading or unavailable."
              : "Complete a rehab session to populate this leaderboard."} />}
      </ChartCard>

      {/* ── Recent sessions with per-row delete ────────────────── */}
      {hasSessions && (
        <RecentSessionsCard sessions={sessions} onDeleted={onDeleted} />
      )}
    </div>
  );
}

// Adherence card — shows sessions completed this week + last 30d
// alongside the "prescribed" side, resolved from getPrescribedSet.
// Today prescribed = auto-recommended; later a doctor-saved
// prescription preempts the auto set (see lib/rehab/recommendation.js
// -> getPrescribedSet). This card doesn't care which — it always
// reads the same seam.
function AdherenceCard({
  sessions,
  patientId,
}: {
  sessions: EnrichedSession[];
  patientId: string;
}) {
  const recs = useRecommendations(patientId);
  // Filter out empty / aborted sessions — a session where the patient
  // never actually did the movement (0 reps for RepCount, 0 ms in
  // zone for HoldInZone, etc.) should NOT count toward adherence.
  // Use per-mechanic activity signals; fall back to points > 0 when
  // full metrics are missing.
  const summaries = useMemo(
    () =>
      sessions
        .filter((s) => sessionHadActivity(s))
        .map((s) => ({
          id: s.id,
          module: "rehab",
          movement: s.metrics?.exerciseSlug ?? s.fallbackExerciseSlug,
          created_at: s.createdAtIso,
        })),
    [sessions],
  );

  const perWeek = useMemo(
    () => computeCompletionByWeek(summaries),
    [summaries],
  );
  const last30 = useMemo(
    () => completedInLastDays(summaries, 30),
    [summaries],
  );
  const last7 = useMemo(
    () => completedInLastDays(summaries, 7),
    [summaries],
  );
  const thisWeek = perWeek.length > 0 ? perWeek[perWeek.length - 1] : null;

  const prescribedCount = recs.slugs.size;
  // How many DISTINCT prescribed slugs the patient has actually done
  // this week? That's the natural "prescribed adherence" numerator.
  const distinctPrescribedThisWeekDone = useMemo(() => {
    if (prescribedCount === 0) return 0;
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
    const seen = new Set<string>();
    for (const s of summaries) {
      const created = new Date(s.created_at).getTime();
      if (!Number.isFinite(created) || created < cutoff) continue;
      const slug = typeof s.movement === "string" ? s.movement : null;
      if (slug && recs.slugs.has(slug)) seen.add(slug);
    }
    return seen.size;
  }, [summaries, recs.slugs, prescribedCount]);

  const prescribedDenominator = prescribedCount > 0 ? String(prescribedCount) : "—";
  const sourceLabel =
    recs.source === "doctor" ? "clinician-prescribed" : "auto-recommended";

  return (
    <ChartCard
      icon={CalendarDays}
      title="Adherence"
      subtitle={
        prescribedCount > 0
          ? `Completed sessions vs ${prescribedCount} ${sourceLabel} exercise${prescribedCount === 1 ? "" : "s"}`
          : "Completed sessions — no prescription set yet"
      }
    >
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-card border border-border bg-surface p-4">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            This week
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {thisWeek?.total ?? 0}
            <span className="ml-1 text-xs font-normal text-muted">
              / {prescribedDenominator}
            </span>
          </p>
          <p className="mt-1 text-[11px] text-muted">
            {thisWeek
              ? `${thisWeek.days} active day${thisWeek.days === 1 ? "" : "s"}`
              : "No sessions yet"}
            {prescribedCount > 0
              ? ` · ${distinctPrescribedThisWeekDone}/${prescribedCount} prescribed touched`
              : ""}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-4">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Last 7 days
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {last7}
          </p>
        </div>
        <div className="rounded-card border border-border bg-surface p-4">
          <p className="text-[10px] uppercase tracking-[0.14em] text-subtle">
            Last 30 days
          </p>
          <p className="mt-2 tabular text-3xl font-semibold text-foreground">
            {last30}
          </p>
        </div>
      </div>
      <p className="mt-4 text-xs text-muted">
        Prescribed:{" "}
        <span className="tabular text-foreground">
          {prescribedCount > 0 ? `${prescribedCount} exercise${prescribedCount === 1 ? "" : "s"}` : "—"}
        </span>
        {prescribedCount > 0
          ? ` · ${sourceLabel} from ${recs.assessmentsUsed} recent assessment${recs.assessmentsUsed === 1 ? "" : "s"}.`
          : " · will populate once a rehab prescription is assigned or the patient completes an assessment."}
      </p>
    </ChartCard>
  );
}

// Recent sessions list — one row per saved session with a link to
// the full report and a per-row delete button. Deletion goes through
// the shared deleteReport() API; on success we call onDeleted so the
// dashboard removes the row and every chart / adherence count above
// re-computes without a page reload.
function RecentSessionsCard({
  sessions,
  onDeleted,
}: {
  sessions: EnrichedSession[];
  onDeleted: (id: string) => void;
}) {
  // Newest first — sessions came in oldest-first for chart plotting.
  const rows = useMemo(
    () =>
      [...sessions].sort((a, b) =>
        b.createdAtIso.localeCompare(a.createdAtIso),
      ),
    [sessions],
  );
  return (
    <ChartCard
      icon={Dumbbell}
      title="Recent sessions"
      subtitle="All saved rehab sessions. Delete removes a session from the record and every chart above."
    >
      <ul className="space-y-2">
        {rows.map((s) => (
          <RecentSessionRow
            key={s.id}
            session={s}
            onDeleted={onDeleted}
          />
        ))}
      </ul>
    </ChartCard>
  );
}

function RecentSessionRow({
  session,
  onDeleted,
}: {
  session: EnrichedSession;
  onDeleted: (id: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const slug = session.metrics?.exerciseSlug ?? session.fallbackExerciseSlug;
  const title = humanExerciseName(slug);
  const dateStr = new Date(session.createdAtIso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
  const summary = summarizeMechanic(session.metrics);

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (busy) return;
    if (!confirm(`Delete this ${title} session? This cannot be undone.`)) return;
    setBusy(true);
    setErr(null);
    try {
      await deleteReport(session.id);
      onDeleted(session.id);
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : "Delete failed");
      setBusy(false);
    }
  }

  return (
    <li>
      <div className="group flex items-center gap-3 rounded-card border border-border bg-surface p-3 transition hover:border-accent">
        <Link
          href={`/dashboard/reports/${session.id}`}
          className="flex flex-1 items-center gap-3 min-w-0"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-card bg-orange-500/10 text-orange-700 dark:text-orange-400">
            <Dumbbell className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-foreground">
              {title}
            </p>
            <p className="truncate text-xs text-muted">
              {summary}
            </p>
          </div>
          <p className="shrink-0 text-xs tabular text-muted">{dateStr}</p>
        </Link>
        <button
          type="button"
          onClick={handleDelete}
          disabled={busy}
          className="shrink-0 rounded-md p-1.5 text-muted transition hover:bg-error/10 hover:text-error disabled:opacity-50"
          aria-label={`Delete ${title} session`}
        >
          {busy
            ? <Loader2 className="h-4 w-4 animate-spin" />
            : <Trash2 className="h-4 w-4" />}
        </button>
      </div>
      {err && <p className="mt-1 pl-3 text-xs text-error">{err}</p>}
    </li>
  );
}

function summarizeMechanic(m: RehabSessionMetrics | null): string {
  if (!m) return "Session details unavailable";
  switch (m.mechanicId) {
    case "rep_count":
      return `${m.reps} reps · ${m.goodReps} clean · ${m.points} pts`;
    case "hold_in_zone": {
      const s = (m.totalMsInZone / 1000).toFixed(1);
      return `${s} s in zone · ${m.points} pts`;
    }
    case "target_reach":
      return `${m.hits} hits · ${m.misses} misses · ${m.points} pts`;
    case "match_pose":
      return `Best match ${m.bestMatchPct.toFixed(0)}% · ${m.points} pts`;
    case "metronome":
      return `${m.liftCount} lifts · ${m.points} pts`;
    default:
      return `${m.points} pts · ${m.durationSec.toFixed(0)}s`;
  }
}

// True when the session actually recorded work — filters out
// aborted / camera-only sessions that would otherwise inflate the
// adherence count. Uses whichever mechanic-appropriate signal fits;
// falls back to points > 0 when full metrics didn't hydrate.
function sessionHadActivity(s: EnrichedSession): boolean {
  const m = s.metrics;
  if (!m) return false;
  if (m.reps > 0) return true;
  if (m.totalMsInZone > 0) return true;
  if (m.hits > 0 || m.misses > 0) return true;
  if (m.bestMatchPct > 0) return true;
  if (m.liftCount > 0) return true;
  if (m.completionPct > 0) return true;
  if (m.points > 0) return true;
  return false;
}

// Compact placeholder rendered INSIDE a ChartCard when the
// underlying data set is empty. Keeps the card sized the same as a
// populated one so the dashboard grid never collapses.
function ChartEmpty({ message }: { message: string }) {
  return (
    <div className="flex min-h-[220px] items-center justify-center rounded-md border border-dashed border-border bg-surface px-4 text-center text-xs text-muted">
      {message}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════
// Sub-components
// ═════════════════════════════════════════════════════════════════

interface StatCardProps {
  icon: typeof Activity;
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

interface ChartCardProps {
  icon: typeof Activity;
  title: string;
  subtitle?: string;
  headerRight?: React.ReactNode;
  children: React.ReactNode;
}

function ChartCard({
  icon: Icon,
  title,
  subtitle,
  headerRight,
  children,
}: ChartCardProps) {
  return (
    <div className="rounded-card border border-border bg-surface p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Icon className="h-4 w-4 text-accent" />
            <h3 className="text-sm font-semibold tracking-tight text-foreground">
              {title}
            </h3>
          </div>
          {subtitle && (
            <p className="mt-1 text-xs text-muted">{subtitle}</p>
          )}
        </div>
        {headerRight}
      </div>
      <div className="mt-4">{children}</div>
    </div>
  );
}

// ── Charts ────────────────────────────────────────────────────

const TEAL = "#14B8A6";
const CYAN = "#06B6D4";
const ORANGE = "#F97316";
const RED = "#EF4444";
const SLATE = "#64748B";

function SessionsPerDayChart({ buckets }: { buckets: DailyBucket[] }) {
  const x = buckets.map((b) => b.date);
  const y = buckets.map((b) => b.sessions.length);
  return (
    <PlotlyChart
      height={240}
      data={[
        {
          type: "bar",
          x,
          y,
          marker: { color: TEAL },
          hovertemplate: "%{x}<br>%{y} sessions<extra></extra>",
        },
      ]}
      layout={{
        yaxis: { title: { text: "Sessions" }, rangemode: "tozero", dtick: 1 },
        xaxis: { type: "category" },
      }}
    />
  );
}

function DurationPerDayChart({ buckets }: { buckets: DailyBucket[] }) {
  const x = buckets.map((b) => b.date);
  const y = buckets.map((b) =>
    b.sessions.reduce(
      (s, x2) => s + (x2.metrics?.durationSec ?? 0),
      0,
    ) / 60,
  );
  return (
    <PlotlyChart
      height={240}
      data={[
        {
          type: "bar",
          x,
          y,
          marker: { color: ORANGE },
          hovertemplate: "%{x}<br>%{y:.1f} min<extra></extra>",
        },
      ]}
      layout={{
        yaxis: { title: { text: "Minutes" }, rangemode: "tozero" },
        xaxis: { type: "category" },
      }}
    />
  );
}

function CompletionOverTimeChart({ buckets }: { buckets: DailyBucket[] }) {
  const x = buckets.map((b) => b.date);
  const y = buckets.map((b) => {
    const scored = b.sessions.filter((s) => s.metrics !== null);
    if (scored.length === 0) return 0;
    const sum = scored.reduce(
      (s, x2) => s + (x2.metrics?.completionPct ?? 0),
      0,
    );
    return sum / scored.length;
  });
  return (
    <PlotlyChart
      height={240}
      data={[
        {
          type: "scatter",
          mode: "lines+markers",
          x,
          y,
          line: { color: CYAN, width: 2.5 },
          marker: { color: CYAN, size: 7 },
          hovertemplate: "%{x}<br>%{y:.0f}%<extra></extra>",
        },
      ]}
      layout={{
        yaxis: { title: { text: "Completion (%)" }, range: [0, 105] },
        xaxis: { type: "category" },
      }}
    />
  );
}

function ProgressTrendChart({ sessions }: { sessions: MetricSession[] }) {
  // Group by date; average signal.value_at_peak per day.
  //
  // CRITICAL: normalise each session's signal through
  // `toDisplayKneeAngle` BEFORE bucketing. Sessions written before
  // this migration hold interior values (~110 for a deep squat);
  // sessions written after hold flexion values (~70). Averaging the
  // raw numbers would draw a spurious cliff in the trend on the day
  // the migration shipped. Normalising both to the flexion display
  // convention keeps the line smooth. Non-knee signals fall through
  // with their original values so unrelated exercises stay
  // untouched.
  const map = new Map<string, { sum: number; count: number; unit: string; name: string | null }>();
  for (const s of sessions) {
    const rawV = s.metrics.signalValueAtPeak;
    const rawName = s.metrics.signalName;
    const rawUnit = s.metrics.signalUnit;
    const knee = rawName
      ? toDisplayKneeAngle({
          name: rawName,
          unit: rawUnit,
          value_at_peak: rawV ?? undefined,
        })
      : null;
    const v = knee ? knee.value : rawV;
    if (v === null) continue;
    const displayName = knee ? KNEE_ANGLE_DISPLAY_LABEL : rawName;
    const bucket = map.get(s.createdDateLocal) ?? {
      sum: 0, count: 0, unit: rawUnit, name: displayName,
    };
    bucket.sum += v;
    bucket.count += 1;
    bucket.unit = rawUnit || bucket.unit;
    bucket.name = displayName || bucket.name;
    map.set(s.createdDateLocal, bucket);
  }
  const entries = Array.from(map.entries()).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) {
    return (
      <div className="flex min-h-[200px] items-center justify-center rounded-md border border-dashed border-border bg-surface text-xs text-muted">
        No clinical-metric data recorded for this exercise yet.
      </div>
    );
  }
  const x = entries.map(([d]) => d);
  const y = entries.map(([, b]) => b.sum / b.count);
  const unit = entries[0][1].unit || "";
  const name = entries[0][1].name
    ? entries[0][1].name!.replace(/_/g, " ")
    : "signal";
  return (
    <PlotlyChart
      height={240}
      data={[
        {
          type: "scatter",
          mode: "lines+markers",
          x,
          y,
          line: { color: TEAL, width: 2.5 },
          marker: { color: TEAL, size: 7 },
          hovertemplate: `%{x}<br>%{y:.1f} ${unit}<extra></extra>`,
          name,
        },
      ]}
      layout={{
        yaxis: { title: { text: `${name} (${unit})` } },
        xaxis: { type: "category" },
      }}
    />
  );
}

function TopExercisesChart({ sessions }: { sessions: MetricSession[] }) {
  // Aggregate per exercise slug: sum completed vs summed target.
  interface Row {
    slug: string;
    completedUnits: number;
    goalUnits: number;
    unitLabel: string;
    sessionCount: number;
  }
  const map = new Map<string, Row>();
  for (const s of sessions) {
    const slug = s.metrics.exerciseSlug;
    const row = map.get(slug) ?? {
      slug,
      completedUnits: 0,
      goalUnits: 0,
      unitLabel: "reps",
      sessionCount: 0,
    };
    row.sessionCount += 1;
    if (s.metrics.mechanicId === "rep_count") {
      row.completedUnits += s.metrics.reps;
      row.goalUnits += s.metrics.targetReps ?? 0;
      row.unitLabel = "reps";
    } else if (s.metrics.mechanicId === "hold_in_zone") {
      row.completedUnits += Math.round(s.metrics.totalMsInZone / 1000);
      row.goalUnits += Math.round((s.metrics.targetHoldMs ?? 0) / 1000);
      row.unitLabel = "s in-zone";
    } else if (s.metrics.mechanicId === "target_reach") {
      row.completedUnits += s.metrics.hits;
      row.goalUnits += s.metrics.hits + s.metrics.misses;
      row.unitLabel = "hits";
    } else if (s.metrics.mechanicId === "metronome") {
      row.completedUnits += s.metrics.liftCount;
      row.goalUnits += Math.max(row.goalUnits, s.metrics.liftCount);
      row.unitLabel = "lifts";
    } else {
      // trace / weight_shift / match_pose — use points as a proxy.
      row.completedUnits += Math.round(s.metrics.bestMatchPct);
      row.goalUnits += 100;
      row.unitLabel = "%";
    }
    map.set(slug, row);
  }
  const rows = Array.from(map.values()).sort(
    (a, b) => b.completedUnits - a.completedUnits,
  );
  if (rows.length === 0) {
    return (
      <div className="flex min-h-[200px] items-center justify-center rounded-md border border-dashed border-border bg-surface text-xs text-muted">
        No exercise data yet.
      </div>
    );
  }
  const yLabels = rows.map((r) => humanExerciseName(r.slug));
  const completed = rows.map((r) => r.completedUnits);
  const goal = rows.map((r) => Math.max(r.goalUnits, r.completedUnits));
  // Chart height scales with row count so all bars are readable.
  const chartHeight = Math.max(240, rows.length * 42 + 60);
  return (
    <PlotlyChart
      height={chartHeight}
      data={[
        {
          type: "bar",
          orientation: "h",
          x: goal,
          y: yLabels,
          name: "Goal",
          marker: { color: SLATE },
          opacity: 0.35,
          hovertemplate: "%{y}<br>Goal: %{x}<extra></extra>",
        },
        {
          type: "bar",
          orientation: "h",
          x: completed,
          y: yLabels,
          name: "Completed",
          marker: { color: TEAL },
          hovertemplate: "%{y}<br>Completed: %{x}<extra></extra>",
        },
      ]}
      layout={{
        barmode: "overlay",
        xaxis: { title: { text: "Units (reps / seconds / hits / %)" } },
        margin: { l: 140, r: 24, t: 20, b: 44 },
        legend: { orientation: "h", y: -0.15 },
      }}
    />
  );
}

// Suppress unused-import warning — kept for the "Trophy stat card"
// future placement; currently the Trophy icon is only used above.
export const __RED_UNUSED = RED;
