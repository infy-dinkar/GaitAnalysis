// What GameShell needs to know about a game.
//
// The shell owns everything that is the same for every camera game:
// the camera, the pose loop, the hand and level pick, the setup check,
// the three calibration holds with their redo/verify branch, both
// countdowns, fullscreen, the canvas resize, the report save (envelope,
// dedupe, retry) and the result layout. A game supplies only what
// differs, through the seams below.
//
// NOTHING HERE MAY IMPORT PHASER, even as a value. `loadScene()` is a
// dynamic import for exactly that reason: the scene — and the ~1.3 MB
// of Phaser behind it — must not be pulled into the shell's chunk. The
// one reference to Phaser's types below is written as `import("phaser")`
// in a TYPE position, which TypeScript erases.

import type { ReactNode } from "react";
import type { ReportCreatePayload } from "@/lib/reports";
import type { HandState, Hand } from "@/lib/games/handTracker";
import type { ReachBox } from "@/lib/games/calibration";
import type { GameAudio } from "@/lib/games/gameAudio";
import type {
  CalibrationSummary,
  MetricsRecorder,
  RoundMetrics,
} from "@/lib/games/gameMetrics";
import type { GameDebugCore } from "@/lib/games/gameDebug";

/** Games ship two difficulties. Widening this is a shell change, not a
 *  per-game one, so it stays here rather than in any game's own file. */
export type LevelId = 1 | 2;

/** The only thing the shell reads from a level: what to put on the
 *  pick button and the result screen. Everything else in a level
 *  config is the game's business. */
export interface BaseLevel {
  id: LevelId;
  label: string;
}

/** The minimum a round result must carry. Games extend it with their
 *  own counters, which only their own renderers read. */
export interface GameResult {
  level: LevelId;
}

/**
 * The generic half of the object handed to the scene.
 *
 * The scene mutates `remainingMs` and `finished`; the React pose loop
 * mutates `state` and feeds `metrics`. A game's control type extends
 * this with its own level config, counters and `onFinish`.
 */
export interface BaseControl {
  /** Live hand state, mutated by the React pose loop. */
  state: HandState;
  box: ReachBox;
  /** 1 = normal, >1 when the patient asked for bigger visuals. */
  visualScale: number;
  /** Collects the round's clinical numbers. The pose loop feeds it
   *  shoulder angles; the scene feeds it the gameplay events. */
  metrics: MetricsRecorder;
  audio: GameAudio;
  /** Round length. Fixed per game, not per level. */
  roundMs: number;
  remainingMs: number;
  finished: boolean;
  debug: GameDebugCore;
}

/** What `loadScene()` resolves to. `key` must match the string the
 *  scene passes to `super(...)` in its constructor. */
export interface GameSceneModule {
  key: string;
  SceneClass: new () => import("phaser").Scene;
}

/** Everything a game needs to turn a finished round into report
 *  metrics. `calibration` is already in arm lengths — the shell derives
 *  it from the same three holds for every game. */
export interface MetricsContext<TResult extends GameResult = GameResult> {
  m: RoundMetrics;
  result: TResult;
  roundMs: number;
  calibration: CalibrationSummary | null;
}

export interface GameDefinition<TResult extends GameResult = GameResult> {
  // ── Identity
  /** Saved as the report's `movement`, and as `metrics.game`. */
  slug: string;
  title: string;
  /** Shown on the setup screen's "is this clearly visible?" check, at
   *  the same scale the game's own items will be drawn. */
  sampleGlyph: string;
  sampleGlyphLabel: string;

  // ── Levels
  levels: Record<LevelId, BaseLevel>;
  defaultLevel: LevelId;

  /** Fixed round length in ms. */
  roundMs: number;

  // ── Play
  /** Dynamic import of the scene. Keeps Phaser out of every chunk but
   *  the one the round actually needs. */
  loadScene(): Promise<GameSceneModule>;
  makeRecorder(hand: Hand, startedAtMs: number): MetricsRecorder;
  /** Wrap the generic control in the game's own, adding its level
   *  config, its counters and its `onFinish`. */
  buildControl(args: {
    base: BaseControl;
    levelId: LevelId;
    onFinish: (r: TResult) => void;
  }): BaseControl;

  // ── Reporting
  buildMetrics(ctx: MetricsContext<TResult>): Record<string, unknown>;
  /** The two big patient-facing numbers. */
  renderHeadline(result: TResult): ReactNode;
  /** The quieter clinician block under them. `stats` is the snapshot
   *  taken when the round ended, or null if the recorder was gone. */
  renderClinical(stats: RoundMetrics | null, result: TResult): ReactNode;
}

/**
 * The report envelope every game shares.
 *
 * `module` is the single literal "games" for all of them; the game's
 * identity travels in `movement`, matching how the 24 rehab exercises
 * share `module: "rehab"` (models/report_models.py:62).
 */
export function buildGameReport(args: {
  slug: string;
  side: Hand;
  metrics: Record<string, unknown>;
}): ReportCreatePayload {
  return {
    module: "games" as const,
    movement: args.slug,
    side: args.side,
    metrics: args.metrics,
  };
}
