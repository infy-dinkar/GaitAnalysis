// The parts of a camera-game scene that are not about the game.
//
// Four things every round needs, extracted from FruitHarvestScene
// unchanged: the frame clock, the hand-lost pause, the palm cursor, and
// the round timer. Each was a bug at least once, and none of them is
// worth getting wrong twice, so they live here and a game's scene calls
// them in order from its own update().
//
// The base deliberately does NOT create display objects. A scene builds
// its own cursor, timer and banners in create() exactly where it wants
// them in the draw order, and assigns them to the protected fields
// below; the base only drives them. That keeps a game's create() — and
// therefore its depth ordering — entirely its own.
//
// SIZING RULE, inherited by every game: dimensions are fractions of
// `unit` (the shortest canvas edge), never fixed pixels. A 2 m viewing
// distance and an unknown display size make absolute pixels meaningless.

import Phaser from "phaser";
import { OneEuro2D } from "@/lib/games/oneEuro";
import type { BaseControl } from "@/lib/games/gameDefinition";

// ── Cursor responsiveness.
//
// Measured against a simulated 1500 px/s sweep at pose 22 Hz / render
// 60 fps. The headline finding: the filter is NOT the main source of
// trailing. Half a pose interval alone is ~34 px of lag at that speed,
// and opening the filter right up (beta 0.02 -> 0.25) only moved the
// total from 45 px to 37 px. Short prediction is what actually helps —
// it halves the lag to ~19 px.
/** Cutoff at zero speed. As low as it can go without visible jitter. */
export const CURSOR_MIN_CUTOFF = 0.9;
/** Opens the cutoff with speed. 6x the old value; costs ~0.2 px of
 *  jitter at rest and removes the filter as the limiting factor. */
export const CURSOR_BETA = 0.12;
/** Extrapolate forward by this fraction of a pose interval. */
export const PREDICT_HORIZON = 0.5;
/** Hard cap on the extrapolation, as a fraction of the canvas unit.
 *  Sized so a hard stop overshoots ~25 px — under half the hit radius
 *  (0.075 unit). Without a cap a fast stop overshoots much further. */
export const PREDICT_CAP = 0.035;
/** Below this speed (canvas units/second) prediction is switched off
 *  entirely, so a still hand cannot drift. */
export const PREDICT_DEADBAND = 0.25;
/** Assumed pose rate before a real measurement arrives. */
export const FALLBACK_POSE_HZ = 20;

/** How long the hand may be missing before the round pauses. The
 *  patient can no longer see themselves, so they need telling. */
export const HAND_LOST_MS = 1000;

/** What beginFrame() hands to the rest of the frame. */
export interface FrameInfo {
  /** update()'s own timestamp. The ONLY clock in play — see below. */
  time: number;
  dtMs: number;
  dt: number;
  /** The hand has not been seen for HAND_LOST_MS; the round is held. */
  lost: boolean;
  /** Round time actually played, with held time already subtracted. */
  elapsedMs: number;
}

export abstract class GameSceneBase extends Phaser.Scene {
  protected control!: BaseControl;

  // ── Display objects the base drives. The SCENE creates these.
  /** The palm cursor. Required. */
  protected cursor!: Phaser.GameObjects.Image;
  /** Seconds remaining. Required. */
  protected timerText!: Phaser.GameObjects.Text;
  /** "We can't see your hand" banner. Optional. */
  protected lostBand: Phaser.GameObjects.Rectangle | null = null;
  protected lostText: Phaser.GameObjects.Text | null = null;
  /** In-canvas diagnostics line, only built under ?gamedebug=1. */
  protected fpsText: Phaser.GameObjects.Text | null = null;

  protected filter = new OneEuro2D({
    minCutoff: CURSOR_MIN_CUTOFF,
    beta: CURSOR_BETA,
  });

  /**
   * -1 until the first update frame seeds it.
   *
   * It MUST come from update()'s own `time` argument and nothing else.
   * Phaser has two unrelated clocks: `time` there is the raw rAF
   * timestamp (ms since the PAGE loaded), while `this.time.now` is
   * `game.loop.time`, which starts at 0 when THIS game boots. Seeding
   * this from `this.time.now` and comparing it against `time` measured
   * the age of the page, so the round ended on frame one with nothing
   * spawned. Seed from the same clock you compare against.
   */
  protected startedAt = -1;
  protected lastFrameAt = -1;
  protected cursorSeeded = false;
  /** Milliseconds the hand has been missing, and the total time the
   *  round clock has been held for. */
  protected handLostMs = 0;
  protected pausedMs = 0;
  protected fpsMin = Infinity;
  /** Palm-source tallies as they stood when the round began. */
  protected palmBase = { hand: 0, elbow: 0, wrist: 0 };
  /** Canvas size at the last layout(), so a fullscreen change is seen. */
  protected canvasSize = { w: 0, h: 0 };

  init(data: { control: BaseControl }) {
    this.control = data.control;
  }

  /** Shortest canvas edge — the unit every size is expressed in. */
  protected get unit(): number {
    return Math.min(this.scale.width, this.scale.height);
  }

  /** Position and size everything anchored to the canvas. Called by the
   *  scene at the end of create(), and by syncCanvasSize() on resize.
   *  Implementations must set `this.canvasSize`. */
  protected abstract layout(): void;

  /** End the round: tear down anything long-lived and call the
   *  control's onFinish with the game's own result shape. */
  protected abstract finishRound(): void;

  /** Called once, on the frame that seeds the clocks, so a scene can
   *  anchor its own timers to the same base. */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected onClockSeeded(time: number): void {}

  /**
   * Frame 1 of the update loop: clocks, then the hand-lost pause.
   *
   * Returns null when the round is already over, which the caller must
   * treat as "return immediately".
   */
  protected beginFrame(time: number): FrameInfo | null {
    const c = this.control;
    if (c.finished) return null;

    // First frame: anchor every clock to update()'s own time base.
    if (this.startedAt < 0) {
      this.startedAt = time;
      this.lastFrameAt = time;
      this.onClockSeeded(time);
      c.debug.sceneState = "running";
    }

    const dtMs = Math.max(0, Math.min(200, time - this.lastFrameAt));
    const dt = Math.max(0.001, (time - this.lastFrameAt) / 1000);
    this.lastFrameAt = time;

    // ── Hand lost. With the camera hidden the patient cannot tell why
    //    nothing is happening, so say so, and hold the round rather
    //    than burning their time.
    if (c.state.usable) {
      this.handLostMs = 0;
    } else {
      this.handLostMs += dtMs;
    }
    const lost = this.handLostMs > HAND_LOST_MS;
    if (lost) this.pausedMs += dtMs;
    this.lostBand?.setVisible(lost);
    this.lostText?.setVisible(lost);
    c.debug.handLost = lost;

    return {
      time,
      dtMs,
      dt,
      lost,
      // Paused time is subtracted, so a round always gives the patient
      // the full roundMs of play.
      elapsedMs: time - this.startedAt - this.pausedMs,
    };
  }

  /**
   * Entering or leaving fullscreen changes the canvas under us.
   * Everything positioned in create() has to be laid out again, or the
   * HUD stays where it was on the old canvas.
   */
  protected syncCanvasSize(): void {
    if (
      this.canvasSize.w !== this.scale.width
      || this.canvasSize.h !== this.scale.height
    ) {
      this.layout();
    }
  }

  /** The round clock. Returns true when the round has just ended, which
   *  the caller must treat as "return immediately". */
  protected advanceRoundClock(f: FrameInfo): boolean {
    const c = this.control;
    c.remainingMs = Math.max(0, c.roundMs - f.elapsedMs);
    this.timerText.setText(String(Math.ceil(c.remainingMs / 1000)));
    if (c.remainingMs <= 0) {
      c.finished = true;
      this.finishRound();
      return true;
    }
    return false;
  }

  /**
   * The palm cursor.
   *
   * Map FIRST, then smooth: filtering in canvas space keeps the cutoff
   * in the same units as the on-screen motion the patient sees, and
   * survives a resize without a jump. The target is the PALM, not the
   * wrist — the glyph reads as a palm and a game's hit test uses the
   * same drawn point, so what the patient sees is what collects.
   */
  protected updateCursor(f: FrameInfo): void {
    const c = this.control;
    if (c.state.usable) {
      const rawX = c.state.palmX;
      const rawY = c.state.palmY;
      if (!this.cursorSeeded) {
        // Snap on the first good frame instead of sliding in from the
        // middle of the screen.
        this.filter.reset();
        this.cursor.setPosition(rawX, rawY);
        this.cursor.setAlpha(1);
        this.cursorSeeded = true;
        this.filter.filter(rawX, rawY, f.dt);
      } else {
        const p = this.filter.filter(rawX, rawY, f.dt);

        // Light prediction. Uses the filter's OWN low-passed velocity
        // rather than differencing the output again, which would put
        // back the noise the filter just removed.
        const hz = c.state.poseHz > 1 ? c.state.poseHz : FALLBACK_POSE_HZ;
        const ahead = (PREDICT_HORIZON / hz);
        const { vx, vy } = this.filter.velocity;
        const speed = Math.hypot(vx, vy);
        let ex = 0;
        let ey = 0;
        if (speed > PREDICT_DEADBAND * this.unit) {
          const cap = PREDICT_CAP * this.unit;
          const dx = vx * ahead;
          const dy = vy * ahead;
          const mag = Math.hypot(dx, dy);
          // Scale the vector as a whole so capping cannot bend its
          // direction the way clamping each axis would.
          const k = mag > cap ? cap / mag : 1;
          ex = dx * k;
          ey = dy * k;
        }
        this.cursor.setPosition(p.x + ex, p.y + ey);
        this.cursor.setAlpha(1);
      }
      // Lag: raw mapped palm vs where the cursor is actually drawn.
      c.debug.lagPx = Math.round(
        Math.hypot(this.cursor.x - rawX, this.cursor.y - rawY),
      );
    } else {
      this.cursor.setAlpha(0.25);
    }
  }

  /** Every diagnostic the base can fill in on its own. A scene writes
   *  its own rows into `debug.extra` alongside this. */
  protected writeCoreDebug(f: FrameInfo): void {
    const c = this.control;
    const d = c.debug;
    d.elapsedMs = f.elapsedMs;
    d.remainingMs = c.remainingMs;
    d.handLive = c.state.live;
    d.handInFrame = c.state.inFrame;
    d.cursorX = Math.round(this.cursor.x);
    d.cursorY = Math.round(this.cursor.y);
    d.canvasW = Math.round(this.scale.width);
    d.canvasH = Math.round(this.scale.height);

    const fps = this.game.loop.actualFps;
    d.fps = Math.round(fps);
    // Skip the first second: the very first frames are always slow
    // (shader compile, texture upload) and would mask a real dip.
    if (f.elapsedMs > 1000 && fps > 0) this.fpsMin = Math.min(this.fpsMin, fps);
    d.fpsMin = Number.isFinite(this.fpsMin) ? Math.round(this.fpsMin) : 0;
    d.tweens = this.tweens.getTweens().length;
    d.objects = this.children.list.length;
    d.poseHz = Math.round(c.state.poseHz * 10) / 10;
    d.cutoffHz = Math.round(this.filter.lastCutoff * 100) / 100;
    d.palmSource = c.state.palmSource;
    // Report deltas against the baseline taken in create(), so the
    // tally covers this round only.
    d.palmCounts = {
      hand: c.state.palmCounts.hand - this.palmBase.hand,
      elbow: c.state.palmCounts.elbow - this.palmBase.elbow,
      wrist: c.state.palmCounts.wrist - this.palmBase.wrist,
    };
    d.armLenPx = Math.round(c.state.armLenPx);
    d.headroomPx = Math.round(c.state.headroomPx);
    d.headroomRatio = Math.round(c.state.headroomRatio * 100) / 100;
    const am = c.metrics.m;
    d.angleDeg = am.dbg.angleDeg;
    d.angleDir = am.dbg.dir;
    d.angleDecidedBy = am.dbg.decidedBy;
    d.elbowDx = am.dbg.elbowDx;
    d.elbowDy = am.dbg.elbowDy;
    d.wristDx = am.dbg.wristDx;
    d.elbowFromMid = am.dbg.elbowFromMid;
    d.wristFromMid = am.dbg.wristFromMid;
    d.maxAbductionDeg = am.maxAbductionDeg;
    d.angleCounted = am.dbg.counted;

    const cover = c.state.cover;
    if (cover.dispW > 0) {
      d.boxPx = {
        x0: Math.round(cover.offX + c.box.xLo * cover.dispW),
        x1: Math.round(cover.offX + c.box.xHi * cover.dispW),
        y0: Math.round(cover.offY + c.box.yLo * cover.dispH),
        y1: Math.round(cover.offY + c.box.yHi * cover.dispH),
      };
    }
  }
}
