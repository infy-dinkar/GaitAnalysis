"use client";
// Saved-report body for module === "games".
//
// Shows the clinician half of the result screen, the level and hand,
// and the calibration the round was measured against. No debug data —
// frame counts, palm sources and the rest exist for tuning the game,
// not for a clinical record.
//
// TWO SECTIONS ARE SHARED and two are per-game. Session and Calibration
// read keys every game writes, and the calibration comes from the same
// three holds whichever game was played. Everything between them is the
// game's own, chosen by `metrics.game`.
//
// OLD REPORTS. The branch keys on `metrics.game` with a fallback to
// `report.movement`, and every Fruit Harvest round ever saved carries
// game: "fruit_harvest", so none of them can be routed anywhere else.
// The Fruit Harvest body below is unchanged from before Cloudburst
// existed.
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
  cloudburst: "Cloudburst",
  kite_flying: "Kite Flying",
};

type Metrics = Record<string, unknown>;

export function GamesBody({ report }: { report: ReportDTO }) {
  const m = (report.metrics ?? {}) as Metrics;
  const game = str(m.game) ?? report.movement ?? "";
  const title = GAME_LABELS[game] ?? "Camera game";
  const level = num(m.level);
  const side = str(report.side);
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

      {game === "cloudburst" ? (
        <CloudburstBody m={m} />
      ) : game === "kite_flying" ? (
        <KiteFlyingBody m={m} />
      ) : (
        <FruitHarvestBody m={m} />
      )}

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

// ─── Fruit Harvest ────────────────────────────────────────────────
// Unchanged. Old reports render exactly as they did before Cloudburst.

function FruitHarvestBody({ m }: { m: Metrics }) {
  const harvested = num(m.harvested);
  const missed = num(m.missed);
  const accuracy = num(m.accuracy_pct);
  const avg = num(m.avg_collect_sec);
  // Neither max_abduction_deg nor max_adduction_deg is read. Both are
  // still SAVED — abduction deliberately, for possible later use — but
  // a frontal 2-D camera during a reaching game cannot produce a
  // degree figure worth putting in front of a clinician next to the
  // Biomechanics module's. The reach counts are what this game can
  // honestly report, so they are all it shows.
  // max_adduction_deg is deliberately NOT read. Rounds saved before
  // this change still carry the field; it is a horizontal-plane value
  // that a frontal camera cannot measure and that is not comparable
  // with the biomech adduction range, so it is not shown for any
  // report, old or new.
  const h1 = num(m.first_half_accuracy_pct);
  const h2 = num(m.second_half_accuracy_pct);
  const zones = (m.zone_hits ?? null) as Record<string, unknown> | null;

  return (
    <>
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

      {/* Reach — counts only. No degree figure is shown: see the
          note on max_abduction_deg above. */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Reach
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Stat
            label="Reaches out / up"
            value={String(num(zones?.abduction) ?? "—")}
          />
          <Stat
            label="Reaches across body"
            value={String(num(zones?.adduction) ?? "—")}
          />
        </div>
      </section>

      {/* Movement pattern */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Movement pattern
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4">
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
    </>
  );
}

// ─── Cloudburst ───────────────────────────────────────────────────

function CloudburstBody({ m }: { m: Metrics }) {
  const caught = num(m.drops_caught);
  const missed = num(m.drops_missed);
  const accuracy = num(m.catch_accuracy_pct);
  const reaction = num(m.avg_reaction_sec);
  const touched = num(m.lightning_touched);
  const avoided = num(m.lightning_avoided);
  const avoidance = num(m.avoidance_pct);
  const h1 = num(m.first_half_catch_accuracy_pct);
  const h2 = num(m.second_half_catch_accuracy_pct);
  const zones = (m.zone_hits ?? null) as Record<string, unknown> | null;
  // Rounds saved before the big strike existed carry none of these;
  // the section below is skipped entirely for them rather than shown
  // as a row of dashes.
  const strikes = num(m.big_strikes_total);
  const dodged = num(m.big_strikes_dodged);
  const struck = num(m.big_strikes_hit);
  const dodgeSec = num(m.avg_dodge_sec);
  // max_abduction_deg is saved but not read, for the same reason as in
  // Fruit Harvest: a frontal camera during a moving task cannot produce
  // a degree figure fit to sit beside the Biomechanics module's.

  return (
    <>
      {/* Catching */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Catching
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Drops caught" value={caught === null ? "—" : String(caught)} />
          <Stat label="Drops missed" value={missed === null ? "—" : String(missed)} />
          <Stat label="Accuracy" value={accuracy === null ? "—" : `${accuracy}%`} />
          <Stat
            label="Avg reaction"
            value={reaction === null ? "—" : `${reaction.toFixed(2)} s`}
          />
        </div>
      </section>

      {/* Avoiding. Kept in its own section, never averaged with the
          numbers above: catching is a commission task and avoiding is
          an inhibition one, and a patient can be intact at one and
          impaired at the other. */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Avoiding
        </h2>
        <p className="mt-1 text-sm text-muted">
          Lightning was to be left alone. Touching it is an error of
          commission, not a missed catch.
        </p>
        <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-3">
          <Stat
            label="Lightning touched"
            value={touched === null ? "—" : String(touched)}
          />
          <Stat
            label="Lightning avoided"
            value={avoided === null ? "—" : String(avoided)}
          />
          <Stat
            label="Avoidance"
            value={avoidance === null ? "—" : `${avoidance}%`}
          />
        </div>
      </section>

      {/* Big centre strikes. A whole-body dodge out of a marked band,
          rather than the hand movement the bolts above ask for — so it
          gets its own section rather than being mixed in. */}
      {strikes !== null && strikes > 0 && (
        <section className="rounded-card border border-border bg-surface p-5">
          <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Storm strikes
          </h2>
          <p className="mt-1 text-sm text-muted">
            A warning marked the middle of the reach for 1.5 s before each
            strike. Moving clear of it is a dodge.
          </p>
          <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Stat
              label="Strikes dodged"
              value={`${dodged === null ? "—" : dodged} of ${strikes}`}
            />
            <Stat label="Struck" value={struck === null ? "—" : String(struck)} />
            {/* Blank when every strike found the hand already outside
                the band: there was no dodge to time. */}
            <Stat
              label="Avg dodge time"
              value={dodgeSec === null ? "—" : `${dodgeSec.toFixed(2)} s`}
            />
          </div>
        </section>
      )}

      {/* Reach */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Reach
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Stat
            label="Catches on own side"
            value={String(num(zones?.same_side) ?? "—")}
          />
          <Stat
            label="Catches across body"
            value={String(num(zones?.across) ?? "—")}
          />
        </div>
      </section>

      {/* Movement pattern. The fall speed rises through the round, so
          the two halves are not the same task — a drop between them is
          where speed started to cost accuracy. */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Movement pattern
        </h2>
        <p className="mt-1 text-sm text-muted">
          Items fall 1.0× to 1.6× faster across the round.
        </p>
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Stat
            label="Catch accuracy, first half"
            value={h1 === null ? "—" : `${h1}%`}
          />
          <Stat
            label="Catch accuracy, second half"
            value={h2 === null ? "—" : `${h2}%`}
          />
        </div>
        {h1 !== null && h2 !== null && h2 < h1 - 15 && (
          <p className="mt-3 text-sm text-warning">
            Catch accuracy fell {h1 - h2} points as the speed rose.
          </p>
        )}
      </section>
    </>
  );
}

// ─── Kite Flying ──────────────────────────────────────────────────

function KiteFlyingBody({ m }: { m: Metrics }) {
  const inside = num(m.time_in_corridor_pct);
  const dev = num(m.mean_deviation_pct);
  const falls = num(m.kite_falls);
  const peaks = num(m.velocity_peaks_per_sec);
  const smooth = num(m.smoothness_score);
  const h1 = num(m.first_half_time_in_corridor_pct);
  const h2 = num(m.second_half_time_in_corridor_pct);
  // Rounds saved before the corridor's width started varying carry
  // neither; the row is skipped for them rather than shown as dashes.
  const narrow = num(m.narrow_time_in_corridor_pct);
  const wide = num(m.wide_time_in_corridor_pct);
  // max_abduction_deg is saved but not read, for the same reason as in
  // the other two games.

  return (
    <>
      {/* Holding the line */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Holding the line
        </h2>
        <p className="mt-1 text-sm text-muted">
          A ribbon of wind drifted across the sky; the task was to keep the
          kite inside it. Deviation is the average distance from the middle
          of the ribbon, as a share of its half-width — 100% is the edge.
        </p>
        <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-3">
          <Stat
            label="Time in the wind"
            value={inside === null ? "—" : `${inside}%`}
          />
          <Stat
            label="Avg deviation"
            value={dev === null ? "—" : `${dev}%`}
          />
          <Stat label="Kite falls" value={falls === null ? "—" : String(falls)} />
        </div>
        {/* The lane breathes between the level's two width factors.
            Holding the line through a narrow stretch is a harder thing
            than holding it through a wide one, so the two are reported
            apart — a patient who keeps the wide stretches and loses the
            narrow ones has a precision problem, not a tracking one. */}
        {narrow !== null && (
          <div className="mt-4 grid grid-cols-2 gap-4 border-t border-border pt-4">
            <Stat label="In narrow stretches" value={`${narrow}%`} />
            <Stat
              label="In wide stretches"
              value={wide === null ? "—" : `${wide}%`}
            />
          </div>
        )}
      </section>

      {/* Movement quality. This is what Kite Flying measures that the
          other two games do not: not how many targets were reached but
          how steadily the hand moved between them. */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Movement quality
        </h2>
        <p className="mt-1 text-sm text-muted">
          Measured from the unsmoothed hand position. Velocity peaks count
          the separate corrections a movement was broken into; smoothness is
          an ordinal 0–100 score derived from them, for comparing a patient
          with themselves across sessions.
        </p>
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Stat
            label="Smoothness"
            value={smooth === null ? "—" : `${smooth} / 100`}
          />
          <Stat
            label="Velocity peaks"
            value={peaks === null ? "—" : `${peaks.toFixed(2)} / s`}
          />
        </div>
        {smooth === null && (
          <p className="mt-3 text-sm text-muted">
            Not scored — the hand did not move enough during the round for
            the measure to mean anything.
          </p>
        )}
      </section>

      {/* Endurance */}
      <section className="rounded-card border border-border bg-surface p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Endurance
        </h2>
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Stat
            label="In the wind, first half"
            value={h1 === null ? "—" : `${h1}%`}
          />
          <Stat
            label="In the wind, second half"
            value={h2 === null ? "—" : `${h2}%`}
          />
        </div>
        {h1 !== null && h2 !== null && h2 < h1 - 15 && (
          <p className="mt-3 text-sm text-warning">
            Time in the wind fell {h1 - h2} points between halves — possible
            fatigue.
          </p>
        )}
      </section>
    </>
  );
}

// ─── Shared bits ──────────────────────────────────────────────────

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
