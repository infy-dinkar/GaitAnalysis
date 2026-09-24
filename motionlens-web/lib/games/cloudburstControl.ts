// The object the React layer hands Cloudburst's scene.
//
// Phaser-free, for the same reason as fruitHarvestControl.ts: the
// definition in components/games/CloudburstGame.tsx builds one, and
// that file must not reach into a module that imports Phaser.

import type { BaseControl } from "@/lib/games/gameDefinition";
import type { CloudburstLevel } from "@/lib/games/cloudburstLevels";
import type { CloudburstResult } from "@/lib/games/cloudburstMetrics";

export interface CloudburstControl extends BaseControl {
  level: CloudburstLevel;
  /** Live counters, so the scene's HUD and the result agree. */
  caught: number;
  lightningTouched: number;
  onFinish: (r: CloudburstResult) => void;
}
