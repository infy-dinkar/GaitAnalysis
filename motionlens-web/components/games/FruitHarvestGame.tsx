"use client";
// Fruit Harvest — what makes it Fruit Harvest, and nothing else.
//
// The round itself (hand pick, setup, calibration, countdowns, play,
// result, save) is GameShell's; this file is the definition it takes.
// The flow that used to live here moved there unchanged, so the only
// things below are the ones another game would answer differently:
// which scene to load, what a level means, what the two big numbers on
// the result screen are, and what goes into the report.

import { GameShell } from "@/components/games/GameShell";
import { Figure, ResultStat } from "@/components/games/gameUi";
import {
  MetricsRecorder,
  meanCollectSec,
  pct,
  type RoundMetrics,
} from "@/lib/games/gameMetrics";
import { buildFruitHarvestMetrics, type FruitHarvestResult } from "@/lib/games/fruitHarvestMetrics";
import {
  DEFAULT_LEVEL,
  LEVELS,
  ROUND_MS,
  levelById,
} from "@/lib/games/levels";
import type { GameDefinition } from "@/lib/games/gameDefinition";
import type { FruitHarvestControl } from "@/lib/games/fruitHarvestControl";

const FRUIT_HARVEST: GameDefinition<FruitHarvestResult> = {
  slug: "fruit_harvest",
  title: "Fruit Harvest",
  sampleGlyph: "🍎",
  sampleGlyphLabel: "sample fruit",

  levels: LEVELS,
  defaultLevel: DEFAULT_LEVEL,
  roundMs: ROUND_MS,

  // Dynamic: this is the ONLY path from the React tree to the scene,
  // and therefore to Phaser. A static import here would put 1.3 MB in
  // the shell's chunk and make the split pointless.
  loadScene: async () => {
    const mod = await import("@/lib/games/fruitHarvestScene");
    return { key: "fruit-harvest", SceneClass: mod.FruitHarvestScene };
  },

  makeRecorder: (hand, startedAtMs) => new MetricsRecorder(hand, startedAtMs),

  buildControl: ({ base, levelId, onFinish }) => {
    const control: FruitHarvestControl = {
      ...base,
      level: levelById(levelId),
      harvested: 0,
      missed: 0,
      onFinish,
    };
    return control;
  },

  buildMetrics: buildFruitHarvestMetrics,

  renderHeadline: (result) => {
    const total = result.harvested + result.missed;
    const accuracy = total > 0
      ? Math.round((result.harvested / total) * 100)
      : 0;
    return (
      <div className="mt-6 flex gap-12 text-center">
        <Figure value={result.harvested} label="Harvested" tone="text-lime-300" />
        <Figure value={`${accuracy}%`} label="Accuracy" tone="text-white" />
      </div>
    );
  },

  renderClinical: (stats, result) => (
    <ClinicalBlock m={stats} missed={result.missed} />
  ),
};

export function FruitHarvestGame() {
  return <GameShell def={FRUIT_HARVEST} />;
}

/** The clinician half of the result screen. Deliberately quieter than
 *  the two big patient numbers above it. */
function ClinicalBlock({
  m,
  missed,
}: {
  m: RoundMetrics | null;
  missed: number;
}) {
  if (!m) return null;
  const avg = meanCollectSec(m.collectSecs);
  const rows: [string, string][] = [
    ["Avg time per fruit", avg === null ? "—" : `${avg.toFixed(2)} s`],
    [
      "Accuracy 1st / 2nd half",
      `${pct(m.firstHalf.hit, m.firstHalf.total)}% / ${pct(m.secondHalf.hit, m.secondHalf.total)}%`,
    ],
    ["Missed", String(missed)],
  ];
  return (
    <div className="mt-6 w-full max-w-md rounded-card bg-black/35 px-5 py-3 text-left">
      {/* Reach — counts only, the same two-value layout the saved
          report uses, so the clinician sees one thing in both places.
          max_abduction_deg is still recorded and saved; it is simply
          not shown. */}
      <p className="text-xs uppercase tracking-[0.12em] text-white/40">
        Reach
      </p>
      <div className="mt-2 grid grid-cols-2 gap-4">
        <ResultStat
          label="Reaches out / up"
          value={String(m.zoneHits.abduction)}
        />
        <ResultStat
          label="Reaches across body"
          value={String(m.zoneHits.adduction)}
        />
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
