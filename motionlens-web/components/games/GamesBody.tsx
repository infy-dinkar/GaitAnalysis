"use client";
// Saved-report body for module === "games".
//
// Shows the clinician half of the result screen, the level and hand,
// and the calibration the round was measured against. No debug data —
// frame counts, palm sources and the rest exist for tuning the game,
// not for a clinical record.
//
// Plain DOM and no canvas, so the existing html2canvas PDF export
// captures it like any other report body.

import type { ReportDTO } from "@/lib/reports";

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

const GAME_LABELS: Record<string, string> = {
  fruit_harvest: "Fruit Harvest",
};

export function GamesBody({ report }: { report: ReportDTO }) {
  const m = (report.metrics ?? {}) as Record<string, unknown>;
  const game = str(m.game) ?? report.movement ?? "";
  const title = GAME_LABELS[game] ?? "Camera game";
  const level = num(m.level);
  const side = str(report.side);
  const lowConf = m.abduction_low_confidence === true;

  const harvested = num(m.harvested);
  const missed = num(m.missed);
  const accuracy = num(m.accuracy_pct);
  const avg = num(m.avg_collect_sec);
  const abd = num(m.max_abduction_deg);
  // max_adduction_deg is deliberately NOT read. Rounds saved before
  // this change still carry the field; it is a horizontal-plane value
  // that a frontal camera cannot measure and that is not comparable
  // with the biomech adduction range, so it is not shown for any
  // report, old or new.
  const h1 = num(m.first_half_accuracy_pct);
  const h2 = num(m.second_half_accuracy_pct);
  const zones = (m.zone_hits ?? null) as Record<string, unknown> | null;
  const cal = (m.calibration ?? null) as Record<string, unknown> | null;
  const duration = num(m.duration_sec);

  return (
    <div className="space-y-6">
      {/* Session */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Session
        </h2>
        <div className="mt-3 flex flex-wrap gap-x-8 gap-y-2 text-sm">
          <Fact label="Game" value={title} />
          {level !== null && <Fact label="Level" value={String(level)} />}
          {side && <Fact label="Hand" value={cap(side)} />}
          {duration !== null && <Fact label="Duration" value={`${duration} s`} />}
        </div>
      </section>

      {/* Performance */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Performance
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Harvested" value={harvested === null ? "—" : String(harvested)} />
          <Stat label="Missed" value={missed === null ? "—" : String(missed)} />
          <Stat label="Accuracy" value={accuracy === null ? "—" : `${accuracy}%`} />
          <Stat
            label="Avg per fruit"
            value={avg === null ? "—" : `${avg.toFixed(2)} s`}
          />
        </div>
      </section>

      {/* Shoulder range */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Shoulder range
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Stat
            label="Max abduction"
            value={abd === null ? "—" : `${abd}°`}
          />
          <Stat
            label="Reaches across body"
            value={String(num(zones?.adduction) ?? "—")}
          />
        </div>
        {lowConf && (
          <p className="mt-3 text-sm text-warning">
            Lower confidence: the hip was not visible for part of this
            round, so the trunk axis fell back to screen vertical.
          </p>
        )}
        {/* One short line. `break-words` so it wraps rather than
            overflowing the card on a narrow column or in the PDF. */}
        <p className="mt-3 break-words text-sm text-muted">
          Game-based estimate. Use Biomechanics for clinical range of
          motion.
        </p>
      </section>

      {/* Movement pattern */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Movement pattern
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Stat
            label="Reaches out / up"
            value={String(num(zones?.abduction) ?? "—")}
          />
          <Stat
            label="Accuracy, first half"
            value={h1 === null ? "—" : `${h1}%`}
          />
          <Stat
            label="Accuracy, second half"
            value={h2 === null ? "—" : `${h2}%`}
          />
        </div>
        {h1 !== null && h2 !== null && h2 < h1 - 15 && (
          <p className="mt-3 text-sm text-warning">
            Accuracy fell {h1 - h2} points between halves — possible fatigue.
          </p>
        )}
      </section>

      {/* Calibration */}
      {cal && (
        <section className="rounded-card border border-border bg-surface p-5">
          <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Calibration
          </h2>
          <p className="mt-1 text-sm text-muted">
            Reach measured before play, in arm lengths — comparable
            between sessions and between patients.
          </p>
          <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Up" value={fmtArm(cal.up)} />
            <Stat label="Out to side" value={fmtArm(cal.side)} />
            <Stat label="Across body" value={fmtArm(cal.across)} />
            <Stat label="Headroom" value={fmtArm(cal.headroom_ratio)} />
          </div>
        </section>
      )}
    </div>
  );
}

function fmtArm(v: unknown): string {
  const n = num(v);
  return n === null ? "—" : `${n.toFixed(2)}×`;
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <span className="text-muted">{label}: </span>
      <span className="font-medium text-foreground">{value}</span>
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.1em] text-subtle">{label}</p>
      <p className="mt-1 text-xl font-semibold tabular text-foreground">
        {value}
      </p>
    </div>
  );
}
