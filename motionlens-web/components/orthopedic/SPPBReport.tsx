"use client";
// SPPB Composite Report.
//
// Renders all 8 sections per spec:
//   1. Patient header
//   2. Primary outcome card (total score 0-12 + classification)
//   3. Component scores breakdown (3 sub-cards)
//   4. Component details (balance, gait, chair-stand)
//   5. Plain-language interpretation
//   6. Trend chart (only when previous sessions are available —
//      passed in via the `previousScores` prop)
//   7. Clinical recommendations
//   8. Unified disclaimer

import dynamic from "next/dynamic";
import { AlertTriangle, ArrowDown, ArrowUp, CheckCircle2, XCircle } from "lucide-react";

import { PatientHeader } from "@/components/dashboard/PatientHeader";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import {
  SPPB_CLASSIFICATION_LABEL,
  SPPB_CLASSIFICATION_TONE,
  type SPPBResult,
} from "@/lib/orthopedic/sppb";
import { STAGE_LABEL } from "@/lib/orthopedic/fourStageBalance";
import type { PatientDTO } from "@/lib/patients";

const PlotlyChart = dynamic(
  () => import("@/components/gait/PlotlyChart").then((m) => m.PlotlyChart),
  { ssr: false },
);

export interface SPPBHistoryEntry {
  date: string;            // ISO date for X axis
  total_score: number;     // 0-12
}

interface Props {
  result: SPPBResult;
  patient?: PatientDTO | null;
  patientName?: string | null;
  /** Optional historical SPPB scores for trend chart. Pass an empty
   *  array (or omit) when this is the first session. */
  previousScores?: SPPBHistoryEntry[];
}

export function SPPBReport({
  result, patient, patientName, previousScores = [],
}: Props) {
  const tone = SPPB_CLASSIFICATION_TONE[result.classification];
  const classLabel = SPPB_CLASSIFICATION_LABEL[result.classification];

  return (
    <div className="space-y-10">
      {/* 1. Patient header */}
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Short Physical Performance Battery (SPPB) · composite geriatric screen"
      />

      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Short Physical Performance Battery
        </h2>
      </div>

      {/* 2. Primary outcome */}
      <section className={`rounded-card border-0 p-6 ${tone}`}>
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] opacity-80">
              Composite SPPB score
            </p>
            <p className="mt-1 tabular text-5xl font-semibold tracking-tight">
              {result.total_score}/12
            </p>
            <p className="mt-1 text-sm font-medium opacity-90">{classLabel}</p>
          </div>
          <div className="text-right text-sm opacity-80">
            <p>Balance: <span className="tabular font-semibold">{result.balance.score}/4</span></p>
            <p>Gait speed: <span className="tabular font-semibold">{result.gait_speed.score}/4</span></p>
            <p>Chair stand: <span className="tabular font-semibold">{result.chair_stand.score}/4</span></p>
          </div>
        </div>
      </section>

      {/* 3. Component scores breakdown */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Component scores</h3>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <ScoreCard
            label="Balance"
            score={result.balance.score}
            outcome={
              result.balance.final_stage_completed === 3
                ? `Held tandem ${result.balance.stage3_hold_seconds.toFixed(1)} s`
                : `Passed Stage ${result.balance.final_stage_completed}`
            }
          />
          <ScoreCard
            label="Gait speed"
            score={result.gait_speed.score}
            outcome={
              result.gait_speed.speed_mps !== null
                ? `${result.gait_speed.speed_mps.toFixed(2)} m/s`
                : "Not completed"
            }
          />
          <ScoreCard
            label="Chair stand"
            score={result.chair_stand.score}
            outcome={
              result.chair_stand.trial.termination === "completed"
                ? `${result.chair_stand.trial.total_time_seconds.toFixed(2)} s for 5 reps`
                : "Not completed"
            }
          />
        </div>
      </section>

      {/* 4. Component details */}
      <BalanceDetails result={result} />
      <GaitDetails result={result} />
      <ChairDetails result={result} />

      {/* 5. Interpretation */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Clinical interpretation</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-sm leading-relaxed text-foreground">
          {result.interpretation}
        </div>
      </section>

      {/* 6. Trend chart */}
      {previousScores.length > 0 && (
        <TrendChart previousScores={previousScores} currentScore={result.total_score} />
      )}

      {/* 7. Recommendation */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Recommendation</h3>
        <div className="mt-3 rounded-card border border-accent/30 bg-accent/5 p-5 text-sm leading-relaxed text-foreground">
          {result.recommendation}
        </div>
      </section>

      {/* 8. Reference cutoffs */}
      <section>
        <h3 className="text-base font-semibold tracking-tight">Reference cutoffs</h3>
        <div className="mt-3 rounded-card border border-border bg-surface p-5 text-xs leading-relaxed text-muted">
          <p>
            <span className="font-medium text-foreground">Total score:</span>{" "}
            10-12 minimal/mild · 7-9 moderate · 0-6 severe lower-extremity limitation.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Balance:</span>{" "}
            0 = failed Stage 1 · 1 = passed Stage 1 only · 2 = passed Stage 2 OR tandem &lt; 3 s ·
            3 = tandem 3-10 s · 4 = tandem 10 s.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Gait speed (m/s):</span>{" "}
            &lt; 0.43 → 1 · 0.43-0.60 → 2 · 0.60-0.77 → 3 · ≥ 0.77 → 4.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Chair stand (5 reps, seconds):</span>{" "}
            &gt; 16.7 → 1 · 13.7-16.7 → 2 · 11.2-13.7 → 3 · ≤ 11.2 → 4.
          </p>
          <p className="mt-1">
            <span className="font-medium text-foreground">Trend:</span>{" "}
            ≥ 1-point drop between visits = clinically meaningful decline.
          </p>
        </div>
      </section>

      <ReportDisclaimer />
    </div>
  );
}

// ─── Score card ────────────────────────────────────────────

function ScoreCard({ label, score, outcome }: { label: string; score: number; outcome: string }) {
  const tone =
    score === 4
      ? "border-emerald-500/40 bg-emerald-500/5"
      : score >= 2
        ? "border-amber-500/40 bg-amber-500/5"
        : "border-red-500/40 bg-red-500/5";
  return (
    <div className={`rounded-card border p-4 ${tone}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {label}
      </p>
      <p className="mt-1 tabular text-3xl font-semibold text-foreground">
        {score}<span className="text-base text-muted">/4</span>
      </p>
      <p className="mt-1 text-xs text-muted">{outcome}</p>
    </div>
  );
}

// ─── Per-component details ──────────────────────────────────

function BalanceDetails({ result }: { result: SPPBResult }) {
  const s = result.balance.stages;
  const rows: Array<{ stage: 1 | 2 | 3; label: string; outcome: string; time: string }> = [];
  for (const idx of [1, 2, 3] as const) {
    const r = s[idx];
    if (r) {
      // Three distinct outcomes — don't collapse "not_attempted"
      // into "Failed". A stage the engine couldn't detect ANY
      // frames for is a coverage/setup issue, not a balance
      // failure; surfacing the difference helps the operator
      // decide whether to re-record.
      const outcomeLabel =
        r.outcome === "pass" ? "Passed"
        : r.outcome === "fail" ? "Failed"
        : "Not attempted";
      rows.push({
        stage: idx,
        label: STAGE_LABEL[idx],
        outcome: outcomeLabel,
        time: r.outcome === "not_attempted" ? "—" : `${r.duration_seconds.toFixed(2)} s`,
      });
    } else {
      rows.push({
        stage: idx,
        label: STAGE_LABEL[idx],
        outcome: "Not attempted",
        time: "—",
      });
    }
  }
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Component 1 — Balance</h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-5 py-3 font-medium">Stage</th>
              <th className="px-5 py-3 font-medium">Outcome</th>
              <th className="px-5 py-3 text-right font-medium">Time held</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.stage} className="border-b border-border/50 last:border-b-0">
                <td className="px-5 py-3 text-foreground">{r.label}</td>
                <td className="px-5 py-3 text-muted">
                  {r.outcome === "Passed" && <CheckCircle2 className="mr-1 inline h-3.5 w-3.5 text-emerald-600" />}
                  {r.outcome === "Failed" && <XCircle className="mr-1 inline h-3.5 w-3.5 text-red-600" />}
                  {r.outcome}
                </td>
                <td className="px-5 py-3 text-right tabular text-foreground">{r.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function GaitDetails({ result }: { result: SPPBResult }) {
  const { trial1, trial2, best_time_sec, speed_mps } = result.gait_speed;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Component 2 — Gait speed</h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <tbody>
            <Row label="Trial 1 time" value={trial1 ? `${trial1.duration_sec.toFixed(2)} s` : "—"} />
            <Row label="Trial 2 time" value={trial2 ? `${trial2.duration_sec.toFixed(2)} s` : "—"} />
            <Row label="Better-of-two (used)" value={best_time_sec !== null ? `${best_time_sec.toFixed(2)} s` : "—"} />
            <Row label="Walking speed" value={speed_mps !== null ? `${speed_mps.toFixed(2)} m/s` : "—"} />
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ChairDetails({ result }: { result: SPPBResult }) {
  const t = result.chair_stand.trial;
  const meanDur =
    t.rep_durations.length > 0
      ? t.rep_durations.reduce((a, b) => a + b, 0) / t.rep_durations.length
      : 0;
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">Component 3 — Chair stand</h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <tbody>
            <Row label="Reps completed" value={`${t.reps.length} / 5`} />
            <Row label="Total time (5 reps)" value={`${t.total_time_seconds.toFixed(2)} s`} />
            <Row label="Mean rep duration" value={`${meanDur.toFixed(2)} s`} />
            <Row label="Termination" value={t.termination === "completed" ? "Completed normally" : t.termination === "timeout" ? "30 s timeout" : "Stopped early"} />
            {t.arm_uncrossed_flag && (
              <tr className="border-b border-border/50 last:border-b-0">
                <td className="px-5 py-3 text-foreground">Arm uncross flag</td>
                <td className="px-5 py-3 text-right">
                  <span className="inline-flex items-center gap-1 text-xs text-warning">
                    <AlertTriangle className="h-3 w-3" />
                    Arms uncrossed at some point — may inflate the score
                  </span>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr className="border-b border-border/50 last:border-b-0">
      <td className="px-5 py-3 text-foreground">{label}</td>
      <td className="px-5 py-3 text-right tabular text-foreground">{value}</td>
    </tr>
  );
}

// ─── Trend chart ────────────────────────────────────────────

function TrendChart({
  previousScores,
  currentScore,
}: {
  previousScores: SPPBHistoryEntry[];
  currentScore: number;
}) {
  const allDates = [...previousScores.map((e) => e.date), new Date().toISOString().slice(0, 10)];
  const allScores = [...previousScores.map((e) => e.total_score), currentScore];

  // Detect change since previous visit (≥1 point = clinically meaningful)
  const prev = previousScores[previousScores.length - 1]?.total_score ?? null;
  const delta = prev !== null ? currentScore - prev : 0;
  const deltaTone =
    delta >= 1
      ? "text-emerald-600 dark:text-emerald-400"
      : delta <= -1
        ? "text-red-600 dark:text-red-400"
        : "text-muted";
  const DeltaIcon = delta >= 1 ? ArrowUp : delta <= -1 ? ArrowDown : null;

  return (
    <section>
      <div className="flex items-baseline justify-between">
        <h3 className="text-base font-semibold tracking-tight">Trend over visits</h3>
        {prev !== null && (
          <p className={`inline-flex items-center gap-1 text-sm font-medium ${deltaTone}`}>
            {DeltaIcon && <DeltaIcon className="h-3.5 w-3.5" />}
            {delta > 0 ? "+" : ""}{delta} since last visit
            {Math.abs(delta) >= 1 && (
              <span className="ml-1 text-xs text-muted">
                ({delta >= 1 ? "improvement" : "decline"} — clinically meaningful)
              </span>
            )}
          </p>
        )}
      </div>
      <div className="mt-3 rounded-card border border-border bg-surface p-4">
        <PlotlyChart
          height={260}
          data={[
            {
              type: "scatter",
              mode: "lines+markers",
              x: allDates,
              y: allScores,
              line: { color: "#EA580C", width: 2 },
              marker: { color: "#EA580C", size: 9 },
              name: "SPPB total",
            },
          ]}
          layout={{
            xaxis: { title: { text: "Visit date" } },
            yaxis: { title: { text: "SPPB score" }, range: [0, 12] },
            margin: { l: 56, r: 24, t: 12, b: 44 },
            showlegend: false,
          }}
        />
      </div>
    </section>
  );
}
