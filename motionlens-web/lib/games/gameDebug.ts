// Diagnostics shared by every camera game (?gamedebug=1).
//
// Written by the scene every frame, polled by the React shell. Keep it
// cheap — no allocation, no formatting beyond what the panel needs.
//
// This file deliberately imports NO Phaser. It used to live inside
// fruitHarvestScene.ts, which meant the React component that reads
// `GameDebug` pulled Phaser in statically and the scene's dynamic
// import could never actually defer anything.
//
// Everything here is game-agnostic. A game's own counters go in
// `extra`, a plain string map the debug panel prints verbatim, so a
// second game does not have to widen this type to add a row.

import type { PalmSource } from "@/lib/games/handTracker";

export interface GameDebugCore {
  phaserCreated: boolean;
  sceneState: string;
  canvasW: number;
  canvasH: number;
  canvasZ: string;
  /** Reach box projected into canvas pixels, as the scene sees it. */
  boxPx: { x0: number; x1: number; y0: number; y1: number } | null;
  boxN: { x0: number; x1: number; y0: number; y1: number } | null;
  handLive: boolean;
  handInFrame: boolean;
  cursorX: number;
  cursorY: number;
  elapsedMs: number;
  remainingMs: number;
  texturesOk: boolean;
  error: string | null;
  /** Smoothed framerate from Phaser's own loop. */
  fps: number;
  /** Worst framerate seen since the round began — where a costly
   *  effect would show up. Ignores the first second of warm-up. */
  fpsMin: number;
  /** Live tween count; the animation cost in one number. */
  tweens: number;
  /** Display-list size, so a leak would be visible as steady growth. */
  objects: number;
  /** Detector callbacks per second — the freshness ceiling. */
  poseHz: number;
  /** One-euro cutoff actually applied this frame, in Hz. */
  cutoffHz: number;
  /** Distance from the raw mapped palm to the drawn cursor, px. */
  lagPx: number;
  /** Where the palm came from this frame. */
  palmSource: PalmSource;
  /** Frames per source SINCE THE ROUND STARTED, so setup and
   *  calibration do not pollute the tally. */
  palmCounts: { hand: number; elbow: number; wrist: number };
  /** Shoulder -> elbow -> wrist, canvas px. */
  armLenPx: number;
  /** Space above the chosen shoulder on the VISIBLE canvas, px. */
  headroomPx: number;
  /** headroomPx / armLenPx. Below ~1.1 a raised arm cannot be seen. */
  headroomRatio: number;
  /** Round is held because the hand has not been seen for a second. */
  handLost: boolean;
  /** Shoulder-angle diagnostics: the current angle, the direction the
   *  biomech rule returned, which branch decided it, and where the
   *  elbow and wrist actually are. */
  angleDeg: number | null;
  angleDir: string;
  angleDecidedBy: string;
  elbowDx: number;
  elbowDy: number;
  wristDx: number;
  elbowFromMid: number;
  wristFromMid: number;
  maxAbductionDeg: number;
  /** Whether this frame fed the abduction peak, and if not why. */
  angleCounted: string;
  /**
   * The game's own rows, label -> value, printed in insertion order.
   *
   * A scene should create every key it will ever write in create(), so
   * the panel's row order is fixed rather than depending on which
   * event happened first.
   */
  extra: Record<string, string>;
}

export function createGameDebug(roundMs: number): GameDebugCore {
  return {
    phaserCreated: false,
    sceneState: "not started",
    canvasW: 0,
    canvasH: 0,
    canvasZ: "—",
    boxPx: null,
    boxN: null,
    handLive: false,
    handInFrame: false,
    cursorX: 0,
    cursorY: 0,
    elapsedMs: 0,
    remainingMs: roundMs,
    texturesOk: false,
    error: null,
    fps: 0,
    fpsMin: 0,
    tweens: 0,
    objects: 0,
    poseHz: 0,
    cutoffHz: 0,
    lagPx: 0,
    palmSource: "wrist",
    palmCounts: { hand: 0, elbow: 0, wrist: 0 },
    armLenPx: 0,
    headroomPx: 0,
    headroomRatio: 0,
    handLost: false,
    angleDeg: null,
    angleDir: "-",
    angleDecidedBy: "-",
    elbowDx: 0,
    elbowDy: 0,
    wristDx: 0,
    elbowFromMid: 0,
    wristFromMid: 0,
    maxAbductionDeg: 0,
    angleCounted: "-",
    extra: {},
  };
}
