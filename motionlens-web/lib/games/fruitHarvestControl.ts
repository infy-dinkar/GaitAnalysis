// The object the React layer hands Fruit Harvest's scene.
//
// It lives apart from the scene itself because the definition in
// components/games/FruitHarvestGame.tsx builds one, and that file must
// not reach into a module that imports Phaser — a type-only import
// would be erased, but a file with no Phaser in it at all is a
// guarantee rather than a compiler setting.

import type { BaseControl } from "@/lib/games/gameDefinition";
import type { LevelConfig } from "@/lib/games/levels";
import type { FruitHarvestResult } from "@/lib/games/fruitHarvestMetrics";

export interface FruitHarvestControl extends BaseControl {
  /** Everything that differs between levels. */
  level: LevelConfig;
  /** Counters the React layer reads back through onFinish. */
  harvested: number;
  missed: number;
  onFinish: (r: FruitHarvestResult) => void;
}
