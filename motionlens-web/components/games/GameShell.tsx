"use client";
// One camera game round, from hand pick to saved report.
//
// This is Fruit Harvest's flow with the game taken out of it. Structure
// mirrors the rehab exercise pages: the camera and the pose loop live
// here, every phase before and after play is plain DOM drawn over the
// video, and Phaser is dynamically imported for the round only —
// through the definition's loadScene(), so this file never names a
// scene and never pulls Phaser into its own chunk.
//
// The pose loop writes into a MUTABLE HandState (a ref) rather than
// React state. A 30 Hz setState would re-render the tree for nothing.
// Low-frequency UI — the readiness ticks and the calibration ring —
// is copied out on a timer instead.
//
// What the game supplies is in lib/games/gameDefinition.ts. What the
// shell owns is everything here: camera, pose loop, hand and level
// pick, setup auto-advance, the three calibration holds with redo and
// verify, both countdowns, fullscreen, canvas resize, the save
// (envelope, per-round dedupe, retry) and the result layout.

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  ChevronRight,
  Gamepad2,
  Hand as HandIcon,
  Maximize2,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { useCamera } from "@/hooks/useCamera";
import { usePoseDetectionLive } from "@/hooks/usePoseDetectionLive";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  createHandState,
  updateHandState,
  type Hand,
  type HandState,
} from "@/lib/games/handTracker";
import {
  HOLDS,
  HOLD_MESSAGE,
  HoldTracker,
  buildReachBox,
  upReachLooksShort,
  checkHoldPose,
  screenDir,
  recordedHoldFails,
  reachGeometry,
  type HoldId,
  type Point,
  type ReachBox,
} from "@/lib/games/calibration";
import { HEADROOM_RATIO_MIN } from "@/lib/games/handTracker";
import {
  enterFullscreen,
  exitFullscreen,
  isFullscreen,
  onFullscreenChange,
} from "@/lib/games/fullscreen";
import { GameAudio } from "@/lib/games/gameAudio";
import type { MetricsRecorder, RoundMetrics } from "@/lib/games/gameMetrics";
import { createGameDebug, type GameDebugCore } from "@/lib/games/gameDebug";
import {
  buildGameReport,
  type BaseControl,
  type GameDefinition,
  type GameResult,
  type LevelId,
} from "@/lib/games/gameDefinition";
import type { ReportCreatePayload } from "@/lib/reports";
import {
  BLANK_LIVE,
  DebugPanel,
  Overlay,
  HeadroomBar,
  HoldFigure,
  ProgressRing,
  SaveStatus,
  SetupDebugPanel,
  Tick,
  type Live,
  type SaveState,
} from "@/components/games/gameUi";

type Phase =
  | "hand"
  | "setup"
  | "countdown-calib"
  | "calibrate"
  | "verify"
  | "countdown-play"
  | "play"
  | "result";

/** Consecutive passing pose frames before the setup screen advances.
 *  At ~20 Hz this is about a quarter of a second — long enough that a
 *  single flickering landmark cannot trigger it, short enough that it
 *  still feels immediate. */
const SETUP_OK_FRAMES = 5;

/**
 * Why the calibration ring is currently held, or null to let it fill.
 *
 * Order matters. The patient can only act on one instruction, so the
 * most fundamental problem wins: no view of the body, then not enough
 * room, then the hand out of frame, then the wrong hand, then the
 * pose itself.
 */
function holdBlockReason(
  s: HandState,
  hold: HoldId,
  hand: Hand,
): string | null {
  if (!s.shoulderOk || s.armLenPx < 1) return "Step into view of the camera";

  // Re-checked every frame, not just at setup: a patient who moves
  // closer after the setup screen loses the room for a raised arm, and
  // nothing downstream would ever notice.
  if (s.headroomRatio < HEADROOM_RATIO_MIN) {
    return "Step back — we need room above your head";
  }
  if (!s.usable) return "Hand out of view — move back or lower the camera";

  const geom = {
    wx: s.x,
    wy: s.y,
    sx: s.shoulderX,
    sy: s.shoulderY,
    midX: s.midShoulderX,
    arm: s.armLenPx,
    dir: screenDir(hand),
  };
  const mine = checkHoldPose(hold, geom);
  if (mine.ok) return null;

  // Wrong hand: the chosen arm is not in the pose but the other one
  // is. Saying "raise your arm higher" at someone whose other arm is
  // already up is the least helpful thing we could do.
  if (s.otherUsable && s.otherShoulderOk) {
    const theirs = checkHoldPose(hold, {
      wx: s.otherX,
      wy: s.otherY,
      sx: s.otherShoulderX,
      sy: s.otherShoulderY,
      midX: s.midShoulderX,
      arm: s.armLenPx,
      dir: screenDir(hand === "left" ? "right" : "left"),
    });
    if (theirs.ok) {
      return `Use your ${hand === "left" ? "LEFT" : "RIGHT"} hand`;
    }
  }
  return mine.message;
}

/** Signed reach of one wrist along the axis `hold` cares about, in arm
 *  lengths. Debug only. */
function reachFor(
  s: HandState,
  hold: HoldId,
  mine: boolean,
  hand: Hand | null,
): number {
  if (!hand || s.armLenPx < 1) return 0;
  const side = mine ? hand : hand === "left" ? "right" : "left";
  const v = checkHoldPose(hold, {
    wx: mine ? s.x : s.otherX,
    wy: mine ? s.y : s.otherY,
    sx: mine ? s.shoulderX : s.otherShoulderX,
    sy: mine ? s.shoulderY : s.otherShoulderY,
    midX: s.midShoulderX,
    arm: s.armLenPx,
    dir: screenDir(side),
  });
  return v.reach;
}

export function GameShell<TResult extends GameResult>({
  def,
}: {
  def: GameDefinition<TResult>;
}) {
  const search = useSearchParams();
  const debugOn = search.get("gamedebug") === "1";
  // ?handswap=1 flips which BlazePose side the chosen hand reads.
  //
  // The code path says no flip should be needed: the detector is fed
  // the raw <video> (CSS transforms do not touch its pixels) and no
  // selfieMode is set, so its left/right labels should already be
  // anatomical. On camera they came back reversed. Rather than guess,
  // this makes it switchable so one round settles it.
  const swapOn = search.get("handswap") === "1";
  const { patientId, patient, saveReport } = usePatientContext();
  const { videoRef, active, error: camError, start } = useCamera();
  const { ready: poseReady, error: poseError, detect } = usePoseDetectionLive();

  const [phase, setPhase] = useState<Phase>("hand");
  const [hand, setHand] = useState<Hand | null>(null);
  const [visualScale, setVisualScale] = useState(1);
  // Chosen on the hand screen, where the clinician is still at the
  // device. Kept in a ref too so the play effect reads the current
  // value without re-running when it changes mid-flow.
  const [level, setLevel] = useState<LevelId>(def.defaultLevel);
  const [count, setCount] = useState(3);
  const [holdIndex, setHoldIndex] = useState(0);
  const [live, setLive] = useState<Live>(BLANK_LIVE);
  const [result, setResult] = useState<TResult | null>(null);
  const [upCheck, setUpCheck] = useState<
    { rUp: number; rSide: number; ratio: number } | null
  >(null);
  const [fs, setFs] = useState(false);
  const [dbg, setDbg] = useState<GameDebugCore | null>(null);
  const [saveState, setSaveState] = useState<SaveState>({ status: "idle" });
  // Snapshot taken when the round ends. The recorder itself is a ref
  // (written ~20x/s by the pose loop); React must not read it during
  // render, so the finished numbers are copied out once.
  const [roundStats, setRoundStats] = useState<RoundMetrics | null>(null);
  const debugRef = useRef<GameDebugCore>(createGameDebug(def.roundMs));

  const stageRef = useRef<HTMLDivElement | null>(null);
  const phaserHostRef = useRef<HTMLDivElement | null>(null);
  const stateRef = useRef<HandState>(createHandState());
  const handRef = useRef<Hand | null>(null);
  const phaseRef = useRef<Phase>("hand");
  const holdRef = useRef(new HoldTracker());
  const pointsRef = useRef<Partial<Record<string, Point>>>({});
  const boxRef = useRef<ReachBox | null>(null);
  const audioRef = useRef<GameAudio | null>(null);
  const holdIndexRef = useRef(0);
  const lastPoseAtRef = useRef(0);
  // "all" walks the three holds; a HoldId repeats just that one, used
  // by Redo and by the post-hold sanity check.
  const calibModeRef = useRef<"all" | HoldId>("all");
  const poseMsgRef = useRef("");
  // Consecutive pose frames with the setup fully passing. The setup
  // screen advances on its own once this is reached, so one noisy
  // frame cannot launch the countdown while the patient is still
  // getting into position.
  const setupOkFramesRef = useRef(0);
  const autoAdvanceRef = useRef<() => void>(() => {});
  const gameRef = useRef<import("phaser").Game | null>(null);
  const metricsRef = useRef<MetricsRecorder | null>(null);
  const pendingSaveRef = useRef<{ key: string; body: ReportCreatePayload } | null>(null);
  const savedKeysRef = useRef<Set<string>>(new Set());
  const onHoldDoneRef = useRef<(p: Point) => void>(() => {});

  // ── Camera. Started from the hand-pick click so the permission
  //    prompt is tied to a gesture; this effect only retries if the
  //    component mounted with a phase that needs it.
  const beginWithHand = useCallback(
    (h: Hand) => {
      handRef.current = h;
      setHand(h);
      if (!audioRef.current) audioRef.current = new GameAudio();
      audioRef.current.prime();
      // Fullscreen must ride on a user gesture, and this click is the
      // first one in the flow — it cannot be done when the countdown
      // ends. Fire and forget: a refusal is not allowed to stop play.
      void enterFullscreen(stageRef.current);
      void start();
      setPhase("setup");
      phaseRef.current = "setup";
    },
    [start],
  );

  // ── Pose loop. One rAF chain for the whole session; it writes the
  //    mutable HandState and drives the calibration hold tracker.
  useEffect(() => {
    if (!active || !poseReady) return;
    let cancelled = false;
    let raf = 0;

    const tick = async () => {
      if (cancelled) return;
      const video = videoRef.current;
      const stage = stageRef.current;
      const h = handRef.current;
      if (!video || !stage || !h || video.readyState < 2 || !video.videoWidth) {
        raf = requestAnimationFrame(() => void tick());
        return;
      }
      try {
        const pose = await detect(video);
        if (cancelled) return;
        const now = performance.now();

        // Pose rate. detect() returns null when it dropped the frame
        // (one send already in flight), so only real results count.
        // This is the freshness ceiling for the cursor: no amount of
        // filtering can show the hand sooner than the detector reports
        // it, which is why it is on the debug overlay.
        if (pose) {
          const prev = lastPoseAtRef.current;
          if (prev > 0) {
            const gap = now - prev;
            if (gap > 1 && gap < 1000) {
              const hz = 1000 / gap;
              const s = stateRef.current;
              s.poseHz = s.poseHz > 0 ? s.poseHz * 0.85 + hz * 0.15 : hz;
            }
          }
          lastPoseAtRef.current = now;
        }

        updateHandState(
          stateRef.current,
          pose?.keypoints ?? null,
          h,
          swapOn,
          video.videoWidth,
          video.videoHeight,
          stage.clientWidth,
          stage.clientHeight,
          now,
        );

        // Clinical sampling. Raw, un-mirrored keypoints, exactly what
        // the biomech formulas expect.
        if (phaseRef.current === "play") {
          metricsRef.current?.sample(pose?.keypoints ?? null, now);
        }

        // Setup advances by itself: no click, so the patient never has
        // to walk back to the machine mid-framing.
        if (phaseRef.current === "setup") {
          if (stateRef.current.setupOk) {
            setupOkFramesRef.current += 1;
            if (setupOkFramesRef.current >= SETUP_OK_FRAMES) {
              setupOkFramesRef.current = 0;
              autoAdvanceRef.current();
            }
          } else {
            setupOkFramesRef.current = 0;
          }
        }

        if (phaseRef.current === "calibrate") {
          const s = stateRef.current;
          const hold = HOLDS[holdIndexRef.current];
          const block = holdBlockReason(s, hold.id, h);
          poseMsgRef.current = block ?? "";
          const done = holdRef.current.feed(s.nx, s.ny, block, now);
          if (done) onHoldDoneRef.current(done);
        }
      } catch {
        // A dropped frame is not worth interrupting the round.
      }
      if (!cancelled) raf = requestAnimationFrame(() => void tick());
    };

    raf = requestAnimationFrame(() => void tick());
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [active, poseReady, detect, videoRef, swapOn]);

  // ── Low-frequency mirror of the mutable state into React, for the
  //    setup ticks and the calibration ring.
  useEffect(() => {
    if (phase !== "setup" && phase !== "calibrate") return;
    const id = window.setInterval(() => {
      const s = stateRef.current;
      const t = holdRef.current;
      setLive({
        noseOk: s.noseOk,
        shouldersOk: s.shouldersOk,
        elbowOk: s.elbowOk,
        wristOk: s.wristOk,
        headroomOk: s.headroomOk,
        setupOk: s.setupOk,
        armLenPx: Math.round(s.armLenPx),
        headroomPx: Math.round(s.headroomPx),
        headroomRatio: Math.round(s.headroomRatio * 100) / 100,
        progress: t.progress,
        holding: t.holding,
        holdStatus: t.status,
        blockReason: t.blockReason,
        myReach: reachFor(s, HOLDS[holdIndexRef.current].id, true, handRef.current),
        otherReach: reachFor(s, HOLDS[holdIndexRef.current].id, false, handRef.current),
        l15: { ...s.probeL15 },
        r16: { ...s.probeR16 },
        usingWrist: s.usingWrist,
        swapped: s.swapped,
      });
    }, 60);
    return () => window.clearInterval(id);
  }, [phase]);

  // ── Countdown driver, shared by both 3-2-1s.
  useEffect(() => {
    if (phase !== "countdown-calib" && phase !== "countdown-play") return;
    const next: Phase = phase === "countdown-calib" ? "calibrate" : "play";
    const id = window.setInterval(() => {
      setCount((c) => {
        if (c <= 1) {
          window.clearInterval(id);
          setPhase(next);
          phaseRef.current = next;
          return 3;
        }
        return c - 1;
      });
    }, 1000);
    return () => window.clearInterval(id);
  }, [phase]);

  // ── Calibration: advance through the three holds, then build the box.
  const redoHold = useCallback((id: HoldId) => {
    const idx = HOLDS.findIndex((h) => h.id === id);
    calibModeRef.current = id;
    holdIndexRef.current = idx < 0 ? 0 : idx;
    setHoldIndex(holdIndexRef.current);
    delete pointsRef.current[id];
    holdRef.current.reset();
    setUpCheck(null);
    setPhase("calibrate");
    phaseRef.current = "calibrate";
  }, []);
  const redoUpHold = useCallback(() => redoHold("up"), [redoHold]);

  // Build the box from whatever holds are recorded, then check the "up"
  // hold against the "side" hold — it is the same arm, so a much
  // shorter overhead reach means the arm never really went up.
  const finishCalibration = useCallback(() => {
    const up = pointsRef.current.up;
    const side = pointsRef.current.side;
    const across = pointsRef.current.across;
    const h = handRef.current;
    const s = stateRef.current;
    holdRef.current.reset();

    if (!up || !side || !across || !h) {
      setCount(3);
      setPhase("countdown-play");
      phaseRef.current = "countdown-play";
      return;
    }

    // Sanity check before the box is trusted. The live gate should
    // have made this impossible, but a hold recorded at the very edge
    // of tolerance, or with the body drifting, can still land outside
    // its pose — and one bad hold poisons every spawn for the round.
    if (s.shoulderOk && s.armLenPx > 1) {
      const pairs: [HoldId, Point][] = [
        ["up", up],
        ["side", side],
        ["across", across],
      ];
      for (const [id, pt] of pairs) {
        const bad = recordedHoldFails(
          id,
          pt,
          h,
          s.shoulderX,
          s.shoulderY,
          s.midShoulderX,
          s.armLenPx,
          s.cover,
        );
        if (bad) {
          redoHold(id);
          return;
        }
      }
    }

    boxRef.current = buildReachBox(
      up,
      side,
      across,
      s.midXValid ? s.midX : 0.5,
      h,
    );

    const chk = s.shoulderOk
      ? upReachLooksShort(up, side, s.shoulderX, s.shoulderY, s.cover)
      : null;
    if (chk?.short) {
      setUpCheck({
        rUp: Math.round(chk.rUp),
        rSide: Math.round(chk.rSide),
        ratio: Math.round(chk.ratio * 100),
      });
      setPhase("verify");
      phaseRef.current = "verify";
      return;
    }
    setCount(3);
    setPhase("countdown-play");
    phaseRef.current = "countdown-play";
  }, [redoHold]);

  useEffect(() => {
    onHoldDoneRef.current = (p: Point) => {
      const idx = holdIndexRef.current;
      pointsRef.current[HOLDS[idx].id] = p;
      holdRef.current.reset();

      // Redo mode repeats only the "up" hold, so go straight back to
      // the check rather than walking the other two again.
      if (calibModeRef.current !== "all") {
        calibModeRef.current = "all";
        finishCalibration();
        return;
      }
      if (idx < HOLDS.length - 1) {
        holdIndexRef.current = idx + 1;
        setHoldIndex(idx + 1);
        return;
      }
      finishCalibration();
    };
  }, [finishCalibration]);

  // ── Play: mount Phaser, tear it down on exit. The definition's
  //    loadScene() is a dynamic import, which keeps Phaser out of every
  //    other route's bundle — and out of this file's chunk.
  useEffect(() => {
    if (phase !== "play") return;
    const host = phaserHostRef.current;
    const box = boxRef.current;
    const h = handRef.current;
    if (!host || !box || !h) return;

    let game: import("phaser").Game | null = null;
    let cancelled = false;

    const debug = createGameDebug(def.roundMs);
    debugRef.current = debug;
    debug.boxN = { x0: box.xLo, x1: box.xHi, y0: box.yLo, y1: box.yHi };

    const base: BaseControl = {
      state: stateRef.current,
      box,
      visualScale,
      metrics: (metricsRef.current = def.makeRecorder(h, Date.now())),
      audio: audioRef.current ?? new GameAudio(),
      roundMs: def.roundMs,
      remainingMs: def.roundMs,
      finished: false,
      debug,
    };
    const control = def.buildControl({
      base,
      levelId: level,
      onFinish: (r) => {
        if (cancelled) return;
        setRoundStats(metricsRef.current ? { ...metricsRef.current.m } : null);
        setResult(r);
        setPhase("result");
        phaseRef.current = "result";
      },
    });

    void (async () => {
      try {
        debug.sceneState = "importing phaser";
        const [Phaser, mod] = await Promise.all([
          import("phaser"),
          def.loadScene(),
        ]);
        if (cancelled) return;
        debug.sceneState = "creating game";
        game = new Phaser.Game({
          type: Phaser.AUTO,
          parent: host,
          transparent: true,
          scale: {
            mode: Phaser.Scale.RESIZE,
            width: host.clientWidth,
            height: host.clientHeight,
          },
          scene: [mod.SceneClass],
        });
        // Called before the game has booted, so the SceneManager parks
        // this in its holding pattern and injects `data` at bootQueue
        // (SceneManager.js:236). That is what gets `control` into
        // init() — the config array auto-starts scene 0 with no data.
        gameRef.current = game;
        game.scene.start(mod.key, { control });
        debug.phaserCreated = true;
        debug.sceneState = "booting";
        // Canvas geometry, once Phaser has appended it.
        window.setTimeout(() => {
          const cv = host.querySelector("canvas");
          if (!cv) {
            debug.error = "no <canvas> was appended to the host element";
            return;
          }
          debug.canvasW = cv.width;
          debug.canvasH = cv.height;
          debug.canvasZ = window.getComputedStyle(cv).zIndex;
        }, 120);
      } catch (e) {
        debug.error = e instanceof Error ? e.message : String(e);
        debug.sceneState = "threw during creation";
      }
    })();

    return () => {
      cancelled = true;
      control.finished = true;
      gameRef.current = null;
      game?.destroy(true);
    };
  }, [phase, visualScale, level, def]);

  useEffect(() => {
    const audio = audioRef.current;
    return () => audio?.close();
  }, []);

  // ── Fullscreen state, including the patient pressing Esc.
  useEffect(() => onFullscreenChange(() => setFs(isFullscreen())), []);

  // ── Keep Phaser's canvas matched to the stage.
  //
  // Scale.RESIZE follows window resizes, but entering fullscreen
  // changes the PARENT's box without necessarily resizing the window,
  // so watch the element itself. The scene re-lays-out its own HUD when
  // it sees the new size, and items carry normalised coordinates so
  // they stay reachable on their own.
  useEffect(() => {
    const host = phaserHostRef.current;
    if (!host || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      const g = gameRef.current;
      if (!g) return;
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (w > 0 && h > 0) g.scale.resize(w, h);
    });
    ro.observe(host);
    return () => ro.disconnect();
  }, []);

  // ── Debug overlay poll. Only while ?gamedebug=1 and only during play.
  useEffect(() => {
    if (!debugOn || phase !== "play") return;
    const id = window.setInterval(() => {
      setDbg({ ...debugRef.current });
    }, 200);
    return () => window.clearInterval(id);
  }, [debugOn, phase]);

  const startCalibration = useCallback(() => {
    holdIndexRef.current = 0;
    setHoldIndex(0);
    pointsRef.current = {};
    holdRef.current.reset();
    setCount(3);
    setPhase("countdown-calib");
    phaseRef.current = "countdown-calib";
  }, []);

  // The pose loop calls this through a ref, so the loop does not have
  // to be torn down and rebuilt when the callback identity changes.
  // The phase guard makes a second call during the same setup a no-op.
  useEffect(() => {
    autoAdvanceRef.current = () => {
      if (phaseRef.current !== "setup") return;
      startCalibration();
    };
  }, [startCalibration]);

  /** Repeat the "arm up" hold only; side and across are kept. */

  const acceptCalibration = useCallback(() => {
    setUpCheck(null);
    setCount(3);
    setPhase("countdown-play");
    phaseRef.current = "countdown-play";
  }, []);

  // ── Save the finished round.
  //
  // One report per round, guarded by the round's own start timestamp:
  // "Play again" and "Next level" each begin a new round with a new
  // startedAtMs, so they save separately and can never resave the old
  // one. A failed save leaves the result on screen untouched — Retry
  // re-posts the SAME payload rather than re-deriving it.
  const saveRoundRef = useRef<() => void>(() => {});
  useEffect(() => {
    saveRoundRef.current = () => {
      const payload = pendingSaveRef.current;
      if (!payload) return;
      if (!patientId) {
        setSaveState({ status: "none" });
        return;
      }
      if (savedKeysRef.current.has(payload.key)) return;
      setSaveState({ status: "saving" });
      void saveReport(payload.body)
        .then((out) => {
          if (out.ok) {
            savedKeysRef.current.add(payload.key);
            setSaveState({ status: "saved" });
          } else {
            setSaveState({
              status: "error",
              message: out.message || "Could not save the report.",
            });
          }
        })
        .catch((e: unknown) => {
          setSaveState({
            status: "error",
            message: e instanceof Error ? e.message : "Could not save the report.",
          });
        });
    };
  }, [patientId, saveReport]);

  // Build the payload the moment a round finishes, then save it.
  useEffect(() => {
    if (phase !== "result" || !result) return;
    const rec = metricsRef.current;
    const h = handRef.current;
    if (!rec || !h) return;

    const m = rec.m;
    const box = boxRef.current;
    const st = stateRef.current;
    const arm = st.armLenPx > 1 ? st.armLenPx : 1;
    const cal = box && st.shoulderOk
      ? reachGeometry(box, st.shoulderX, st.shoulderY, st.cover, 0)
      : null;

    pendingSaveRef.current = {
      // Identity of THIS round. A new round gets a new key, so the
      // guard blocks a double-save without blocking the next round.
      key: `${m.startedAtMs}`,
      body: buildGameReport({
        slug: def.slug,
        side: h,
        metrics: def.buildMetrics({
          m,
          result,
          roundMs: def.roundMs,
          calibration: cal
            ? {
              // Arm lengths, so the numbers mean the same thing for
              // any body size at any distance from the camera.
              up: Math.round((cal.rUp / arm) * 100) / 100,
              side: Math.round((cal.rSide / arm) * 100) / 100,
              across: Math.round((cal.rAcross / arm) * 100) / 100,
              headroom_ratio: Math.round(st.headroomRatio * 100) / 100,
            }
            : null,
        }),
      }),
    };
    saveRoundRef.current();
  }, [phase, result, def]);

  /** Same hand, same calibration, next level. */
  const nextLevel = useCallback(() => {
    setLevel(2);
    setResult(null);
    setCount(3);
    setPhase("countdown-play");
    phaseRef.current = "countdown-play";
  }, []);

  const playAgain = useCallback(() => {
    setResult(null);
    setCount(3);
    setPhase("countdown-play");
    phaseRef.current = "countdown-play";
  }, []);

  const backHref = patientId
    ? `/dashboard/patients/${patientId}/games`
    : "/";

  const hold = HOLDS[holdIndex];

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-6">
      <div className="mb-4 flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-card bg-stone-500/10 text-stone-700 dark:text-stone-400">
          <Gamepad2 className="h-5 w-5" />
        </span>
        <div>
          <p className="eyebrow">Games</p>
          <h1 className="text-xl font-semibold tracking-tight">{def.title}</h1>
        </div>
        {patient && (
          <span className="ml-auto text-sm text-muted">
            Session for {patient.name}
          </span>
        )}
      </div>

      {/* In fullscreen the stage IS the fullscreen element, so the
          browser sizes it to the screen — drop the aspect ratio and
          the rounding, or it letterboxes itself inside the display.
          Nav, footer and page padding are ancestors, so they are not
          rendered at all while this element is in the top layer. */}
      <div
        ref={stageRef}
        className={
          fs
            ? "relative h-full w-full overflow-hidden bg-black"
            : "relative aspect-video w-full overflow-hidden rounded-card bg-black"
        }
      >
        {/* The camera is HIDDEN during play — the patient sees the
            game and their hand, nothing else. The element stays
            mounted and playing either way, because pose detection
            reads frames from it; only its presentation changes.
            `opacity-0` rather than `hidden`, so the browser keeps
            decoding it. With ?gamedebug=1 it shrinks to a corner PiP
            instead of disappearing. */}
        <video
          ref={videoRef}
          playsInline
          muted
          className={
            phase !== "play"
              ? "h-full w-full scale-x-[-1] object-cover"
              : debugOn
                ? "absolute bottom-2 right-2 z-30 w-1/5 scale-x-[-1] rounded-md border border-white/30 object-cover opacity-90"
                : "h-full w-full scale-x-[-1] object-cover opacity-0"
          }
        />
        {/* Phaser draws here during play only. z-5 puts it above the
            <video> (which has its own stacking context from the mirror
            transform) and below the z-10 phase overlays. */}
        <div
          ref={phaserHostRef}
          className={`absolute inset-0 z-[5] ${phase === "play" ? "" : "hidden"}`}
        />

        {/* Dropped out of fullscreen mid-flow (Esc, usually). Offer the
            way back without interrupting anything — the round keeps
            running behind this. */}
        {!fs && phase !== "hand" && (
          <button
            type="button"
            onClick={() => void enterFullscreen(stageRef.current)}
            className="absolute right-2 top-2 z-40 rounded-md border border-white/30 bg-black/60 px-3 py-1.5 text-sm text-white hover:bg-black/80"
          >
            <Maximize2 className="mr-1 inline h-4 w-4" />
            Full screen
          </button>
        )}

        {debugOn && phase === "play" && dbg && <DebugPanel d={dbg} />}
        {debugOn && (phase === "setup" || phase === "calibrate") && (
          <SetupDebugPanel live={live} holdId={hold.id} />
        )}

        {(camError || poseError) && (
          <Overlay>
            <p className="text-lg text-white">
              {camError ?? poseError}
            </p>
          </Overlay>
        )}

        {phase === "hand" && (
          <Overlay>
            <h2 className="text-3xl font-semibold text-white">
              Which hand will you play with?
            </h2>
            <p className="mt-2 text-lg text-white/70">
              Only that hand controls the game. The other one is ignored.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
              <Button size="lg" onClick={() => beginWithHand("left")}>
                <HandIcon className="h-5 w-5 scale-x-[-1]" />
                Left hand
              </Button>
              <Button size="lg" onClick={() => beginWithHand("right")}>
                <HandIcon className="h-5 w-5" />
                Right hand
              </Button>
              <span className="mx-1 h-8 w-px bg-white/25" aria-hidden />
              {([1, 2] as LevelId[]).map((id) => (
                <Button
                  key={id}
                  size="lg"
                  variant={level === id ? "primary" : "secondary"}
                  onClick={() => setLevel(id)}
                >
                  {def.levels[id].label}
                </Button>
              ))}
            </div>
          </Overlay>
        )}

        {phase === "setup" && (
          <Overlay>
            <h2 className="text-2xl font-semibold text-white">
              Head and shoulders in frame, with room above your head
            </h2>
            <p className="mt-1 text-base text-white/60">
              Your legs are not needed for this game.
            </p>
            <ul className="mt-5 space-y-2 text-lg">
              <Tick ok={live.noseOk} label="Head visible" />
              <Tick ok={live.shouldersOk} label="Both shoulders visible" />
              <Tick
                ok={live.elbowOk && live.wristOk}
                label={`${hand === "left" ? "Left" : "Right"} elbow and hand visible`}
              />
            </ul>

            <HeadroomBar live={live} />

            <div className="mt-8 flex flex-col items-center">
              <p className="text-lg text-white/80">Is this clearly visible?</p>
              <div
                className="mt-3 flex items-center justify-center"
                style={{ fontSize: `${3.2 * visualScale}rem`, lineHeight: 1 }}
              >
                <span role="img" aria-label={def.sampleGlyphLabel}>
                  {def.sampleGlyph}
                </span>
              </div>
              <div className="mt-4 flex gap-3">
                <Button
                  variant="secondary"
                  onClick={() => setVisualScale((v) => Math.min(2.2, v * 1.35))}
                >
                  Make bigger
                </Button>
              </div>
            </div>
          </Overlay>
        )}

        {(phase === "countdown-calib" || phase === "countdown-play") && (
          <Overlay>
            <p className="text-[9rem] font-bold leading-none text-white">
              {count}
            </p>
            <p className="mt-2 text-xl text-white/70">
              {phase === "countdown-calib" ? "Calibration" : "Get ready"}
            </p>
          </Overlay>
        )}

        {phase === "calibrate" && (
          <Overlay>
            <p className="text-lg text-white/60">
              Hold {holdIndex + 1} of {HOLDS.length}
            </p>
            <h2 className="mt-1 text-3xl font-semibold text-white">
              {hold.title}
            </h2>
            <p className="mt-2 text-lg text-white/70">{hold.instruction}</p>

            <div className="mt-6 flex items-center gap-8">
              <HoldFigure id={hold.id} hand={hand ?? "right"} />
              <ProgressRing progress={live.progress} active={live.holding} />
            </div>
            {/* The ring never changes silently: paused, drifted and
                idle each say what is happening and what to do. */}
            <p
              className={`mt-4 max-w-xl ${
                live.holdStatus === "blocked"
                  ? "text-3xl font-semibold text-amber-300"
                  : "text-lg text-white/70"
              }`}
            >
              {live.holdStatus === "blocked"
                ? live.blockReason
                : HOLD_MESSAGE[live.holdStatus]}
            </p>
            {live.holdStatus === "blocked" && live.progress > 0 && (
              <p className="mt-1 text-sm text-white/50">
                Timer paused at {Math.round(live.progress * 100)}% — it will
                carry on from here.
              </p>
            )}
          </Overlay>
        )}

        {phase === "verify" && upCheck && (
          <Overlay>
            <h2 className="text-3xl font-semibold text-white">
              Arm didn&apos;t go fully up
            </h2>
            <p className="mt-3 max-w-xl text-lg text-white/70">
              Your overhead reach measured {upCheck.ratio}% of your sideways
              reach. It is the same arm, so those should be close. If the
              raised hand left the top of the picture, move back or tilt the
              camera down and try that hold again.
            </p>
            <div className="mt-8 flex gap-3">
              <Button size="lg" onClick={redoUpHold}>
                <RotateCcw className="h-5 w-5" />
                Redo the &quot;arm up&quot; hold
              </Button>
              <Button size="lg" variant="secondary" onClick={acceptCalibration}>
                Continue anyway
              </Button>
            </div>
          </Overlay>
        )}

        {phase === "result" && result && (
          <Overlay>
            <h2 className="text-3xl font-semibold text-white">Round complete</h2>
            <p className="mt-1 text-lg text-white/60">
              {def.levels[result.level].label}
            </p>
            {/* Patient-facing: big, two numbers, nothing else. */}
            {def.renderHeadline(result)}

            {/* Clinician-facing: smaller, below the fold of attention. */}
            {def.renderClinical(roundStats, result)}

            <SaveStatus
              state={saveState}
              patientName={patient?.name ?? null}
              onRetry={() => saveRoundRef.current()}
            />
            <div className="mt-10 flex gap-3">
              <Button size="lg" onClick={playAgain}>
                <RotateCcw className="h-5 w-5" />
                Play again
              </Button>
              {result.level === 1 && (
                <Button size="lg" onClick={nextLevel}>
                  <ChevronRight className="h-5 w-5" />
                  Next level
                </Button>
              )}
              {/* Leaving the game leaves fullscreen. "Play again"
                  deliberately stays in it. */}
              <Link href={backHref} onClick={() => void exitFullscreen()}>
                <Button size="lg" variant="secondary">
                  Back to games
                </Button>
              </Link>
            </div>
          </Overlay>
        )}
      </div>

    </div>
  );
}
