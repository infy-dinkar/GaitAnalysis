"use client";
// Kite Flying — what makes it Kite Flying, and nothing else.
//
// The round itself (hand pick, setup, calibration, countdowns, play,
// result, save) is GameShell's. This file is the definition it takes.

import { GameShell } from "@/components/games/GameShell";
import { Figure, ResultStat } from "@/components/games/gameUi";
import { MetricsRecorder } from "@/lib/games/gameMetrics";
import type { Hand } from "@/lib/games/handTracker";
import {
  buildKiteMetrics,
  pct,
  type KiteResult,
} from "@/lib/games/kiteMetrics";
import { smoothnessScore } from "@/lib/games/kiteMovement";
import {
  DEFAULT_LEVEL,
  LEVELS,
  ROUND_MS,
  levelById,
} from "@/lib/games/kiteLevels";
import type { GameDefinition } from "@/lib/games/gameDefinition";
import type { KiteControl } from "@/lib/games/kiteControl";

/**
 * Which hand this round is being played with.
 *
 * The shell passes the hand to makeRecorder() but not to buildControl(),
 * and the scene needs it — the kite's string is tied on the side the
 * patient is NOT using, and the hand goes in the report. The shell
 * calls makeRecorder first and buildControl immediately after, for the
 * same round, so stashing it here is safe. There is one shell per page.
 */
let handForRound: Hand = "right";

const KITE_FLYING: GameDefinition<KiteResult> = {
  slug: "kite_flying",
  title: "Kite Flying",
  sampleGlyph: "🪁",
  sampleGlyphLabel: "sample kite",

  levels: LEVELS,
  defaultLevel: DEFAULT_LEVEL,
  roundMs: ROUND_MS,

  // Dynamic: the only path from the React tree to the scene, and
  // therefore to Phaser.
  loadScene: async () => {
    const mod = await import("@/lib/games/kiteScene");
    return { key: "kite-flying", SceneClass: mod.KiteScene };
  },

  makeRecorder: (hand, startedAtMs) => {
    handForRound = hand;
    return new MetricsRecorder(hand, startedAtMs);
  },

  buildControl: ({ base, levelId, onFinish }) => {
    const control: KiteControl = {
      ...base,
      level: levelById(levelId),
      hand: handForRound,
      insidePct: 0,
      falls: 0,
      onFinish,
    };
    return control;
  },

  buildMetrics: buildKiteMetrics,

  renderHeadline: (result) => (
    <div className="mt-6 flex gap-12 text-center">
      <Figure
        value={`${pct(result.insideMs, result.measuredMs)}%`}
        label="Time in the wind"
        tone="text-sky-300"
      />
      <Figure value={result.falls} label="Kite falls" tone="text-white" />
    </div>
  ),

  renderClinical: (_stats, result) => <ClinicalBlock r={result} />,
};

export function KiteFlyingGame() {
  return <GameShell def={KITE_FLYING} />;
}

/** The clinician half of the result screen. No degree figures — see
 *  lib/games/kiteMetrics.ts. */
function ClinicalBlock({ r }: { r: KiteResult }) {
  const pps = r.peaksPerSec;
  const dev = r.devCount > 0 ? Math.round((r.devSum / r.devCount) * 100) : null;
  const rows: [string, string][] = [
    [
      "Velocity peaks",
      pps === null ? "—" : `${pps.toFixed(2)} / s of movement`,
    ],
    [
      "Average deviation",
      dev === null ? "—" : `${dev}% of half-width`,
    ],
    [
      "In the wind, 1st / 2nd half",
      `${pct(r.firstHalf.inside, r.firstHalf.total)}%`
      + ` / ${pct(r.secondHalf.inside, r.secondHalf.total)}%`,
    ],
  ];
  return (
    <div className="mt-6 w-full max-w-md rounded-card bg-black/35 px-5 py-3 text-left">
      {/* Smoothness leads, because it is the thing this game measures
          that the other two do not. It is blank rather than zero when
          the patient barely moved — a 0 would read as "severely
          impaired" when it means "did not play". */}
      <p className="text-xs uppercase tracking-[0.12em] text-white/40">
        Movement quality
      </p>
      <div className="mt-2 grid grid-cols-2 gap-4">
        <ResultStat
          label="Smoothness"
          value={pps === null ? "—" : `${smoothnessScore(pps)} / 100`}
        />
        <ResultStat label="Kite falls" value={String(r.falls)} />
      </div>

      <div className="mt-3 border-t border-white/10 pt-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 py-1 text-base">
            <span className="min-w-0 break-words text-white/55">{k}</span>
            <span className="tabular shrink-0 text-white">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
