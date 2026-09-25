// The object the React layer hands Kite Flying's scene.
//
// Phaser-free, for the same reason as the other two games' control
// types: the definition in components/games/KiteFlyingGame.tsx builds
// one, and that file must not reach into a module that imports Phaser.

import type { BaseControl } from "@/lib/games/gameDefinition";
import type { KiteLevel } from "@/lib/games/kiteLevels";
import type { KiteResult } from "@/lib/games/kiteMetrics";
import type { Hand } from "@/lib/games/handTracker";

export interface KiteControl extends BaseControl {
  level: KiteLevel;
  /** Which hand the round is being played with. The scene needs it for
   *  the result; the shell already has it but does not pass it on. */
  hand: Hand;
  /** Live counters, so the HUD and the result agree. */
  insidePct: number;
  falls: number;
  onFinish: (r: KiteResult) => void;
}
