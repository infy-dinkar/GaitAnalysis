"use client";
// Cloudburst — what makes it Cloudburst, and nothing else.
//
// The round itself (hand pick, setup, calibration, countdowns, play,
// result, save) is GameShell's. This file is the definition it takes,
// and it is the whole cost of adding a second game to the platform.

import { GameShell } from "@/components/games/GameShell";
import { Figure, ResultStat } from "@/components/games/gameUi";
import { MetricsRecorder } from "@/lib/games/gameMetrics";
import {
  buildCloudburstMetrics,
  meanSec,
  pct,
  type CloudburstResult,
} from "@/lib/games/cloudburstMetrics";
import {
  DEFAULT_LEVEL,
  LEVELS,
  ROUND_MS,
  levelById,
} from "@/lib/games/cloudburstLevels";
import type { GameDefinition } from "@/lib/games/gameDefinition";
import type { CloudburstControl } from "@/lib/games/cloudburstControl";

const CLOUDBURST: GameDefinition<CloudburstResult> = {
  slug: "cloudburst",
  title: "Cloudburst",
  sampleGlyph: "💧",
  sampleGlyphLabel: "sample water drop",

  levels: LEVELS,
  defaultLevel: DEFAULT_LEVEL,
  roundMs: ROUND_MS,

  // Dynamic: the only path from the React tree to the scene, and
  // therefore to Phaser.
  loadScene: async () => {
    const mod = await import("@/lib/games/cloudburstScene");
    return { key: "cloudburst", SceneClass: mod.CloudburstScene };
  },

  // The SHARED recorder, unchanged. Its sample() reads pose, not
  // gameplay, so the shoulder angle is measured identically in both
  // games; Cloudburst's own counters ride on the result object instead.
  makeRecorder: (hand, startedAtMs) => new MetricsRecorder(hand, startedAtMs),

  buildControl: ({ base, levelId, onFinish }) => {
    const control: CloudburstControl = {
      ...base,
      level: levelById(levelId),
      caught: 0,
      lightningTouched: 0,
      onFinish,
    };
    return control;
  },

  buildMetrics: buildCloudburstMetrics,

  renderHeadline: (result) => {
    const seen = result.caught + result.dropsMissed;
    return (
      <div className="mt-6 flex gap-12 text-center">
        <Figure value={result.caught} label="Drops caught" tone="text-sky-300" />
        <Figure
          value={`${pct(result.caught, seen)}%`}
          label="Catch accuracy"
          tone="text-white"
        />
      </div>
    );
  },

  renderClinical: (_stats, result) => <ClinicalBlock r={result} />,
};

export function CloudburstGame() {
  return <GameShell def={CLOUDBURST} />;
}

/** The clinician half of the result screen. Deliberately quieter than
 *  the two big patient numbers above it, and with no degree figures —
 *  see lib/games/cloudburstMetrics.ts. */
function ClinicalBlock({ r }: { r: CloudburstResult }) {
  const avg = meanSec(r.reactionSecs);
  const dodge = meanSec(r.dodgeSecs);
  const rows: [string, string][] = [
    [
      "Lightning touched",
      `${r.lightningTouched} of ${r.lightningTouched + r.lightningAvoided}`,
    ],
    ["Strikes dodged", `${r.bigStrikesDodged} of ${r.bigStrikes}`],
    // Blank when every strike found the hand already outside the band:
    // there was no dodge to time. See cloudburstMetrics.ts.
    ["Average dodge time", dodge === null ? "—" : `${dodge.toFixed(2)} s`],
    ["Avg reaction", avg === null ? "—" : `${avg.toFixed(2)} s`],
    [
      "Catch accuracy 1st / 2nd half",
      `${pct(r.firstHalf.hit, r.firstHalf.total)}% / ${pct(r.secondHalf.hit, r.secondHalf.total)}%`,
    ],
  ];
  return (
    <div className="mt-6 w-full max-w-md rounded-card bg-black/35 px-5 py-3 text-left">
      {/* Avoiding is a separate ability from catching, so it gets its
          own headline number rather than being folded into accuracy. */}
      <p className="text-xs uppercase tracking-[0.12em] text-white/40">
        Control
      </p>
      <div className="mt-2 grid grid-cols-2 gap-4">
        <ResultStat
          label="Lightning avoided"
          value={`${pct(r.lightningAvoided, r.lightningAvoided + r.lightningTouched)}%`}
        />
        <ResultStat
          label="Reaches across body"
          value={String(r.zoneHits.across)}
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
