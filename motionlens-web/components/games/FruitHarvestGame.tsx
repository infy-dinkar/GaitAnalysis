"use client";
// Fruit Harvest — the whole round, from hand pick to result.
//
// Structure mirrors the rehab exercise pages: the camera and the pose
// loop live here, every phase before and after play is plain DOM drawn
// over the video, and Phaser is dynamically imported for the 60 s round
// only. Nothing in this file writes a report — step 1 has no metrics
// and no save.
//
// The pose loop writes into a MUTABLE HandState (a ref) rather than
// React state. A 30 Hz setState would re-render the tree for nothing.
// Low-frequency UI — the readiness ticks and the calibration ring —
// is copied out on a timer instead.

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Check,
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
  type Probe,
} from "@/lib/games/handTracker";
import {
  HOLDS,
  HOLD_MESSAGE,
  HOLD_MS,
  HoldTracker,
  buildReachBox,
  upReachLooksShort,
  checkHoldPose,
  screenDir,
  recordedHoldFails,
  type HoldId,
  type HoldStatus,
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
import {
  ROUND_MS,
  createGameDebug,
  type FruitHarvestControl,
  type GameDebug,
} from "@/lib/games/fruitHarvestScene";

type Phase =
  | "hand"
  | "setup"
  | "countdown-calib"
  | "calibrate"
  | "verify"
  | "countdown-play"
  | "play"
  | "result";

interface Live {
  noseOk: boolean;
  shouldersOk: boolean;
  elbowOk: boolean;
  wristOk: boolean;
  headroomOk: boolean;
  setupOk: boolean;
  armLenPx: number;
  headroomPx: number;
  headroomRatio: number;
  progress: number;
  holding: boolean;
  holdStatus: HoldStatus;
  /** Raw readouts for both wrists, to settle which label follows which
   *  physical hand. */
  l15: Probe;
  r16: Probe;
  usingWrist: number;
  swapped: boolean;
  blockReason: string;
  /** Signed reach of each wrist along the axis this hold cares about,
   *  in arm lengths. Near-misses are invisible without it. */
  myReach: number;
  otherReach: number;
}

const BLANK_PROBE: Probe = { x: 0, y: 0, vis: 0, inFrame: false };

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

const BLANK_LIVE: Live = {
  noseOk: false,
  shouldersOk: false,
  elbowOk: false,
  wristOk: false,
  headroomOk: false,
  setupOk: false,
  armLenPx: 0,
  headroomPx: 0,
  headroomRatio: 0,
  progress: 0,
  holding: false,
  holdStatus: "idle",
  l15: BLANK_PROBE,
  r16: BLANK_PROBE,
  usingWrist: 16,
  swapped: false,
  blockReason: "",
  myReach: 0,
  otherReach: 0,
};

export function FruitHarvestGame() {
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
  const { patientId, patient } = usePatientContext();
  const { videoRef, active, error: camError, start } = useCamera();
  const { ready: poseReady, error: poseError, detect } = usePoseDetectionLive();

  const [phase, setPhase] = useState<Phase>("hand");
  const [hand, setHand] = useState<Hand | null>(null);
  const [visualScale, setVisualScale] = useState(1);
  const [count, setCount] = useState(3);
  const [holdIndex, setHoldIndex] = useState(0);
  const [live, setLive] = useState<Live>(BLANK_LIVE);
  const [result, setResult] = useState<{ harvested: number; missed: number } | null>(
    null,
  );
  const [upCheck, setUpCheck] = useState<
    { rUp: number; rSide: number; ratio: number } | null
  >(null);
  const [fs, setFs] = useState(false);
  const [dbg, setDbg] = useState<GameDebug | null>(null);
  const debugRef = useRef<GameDebug>(createGameDebug());

  const stageRef = useRef<HTMLDivElement | null>(null);
  const phaserHostRef = useRef<HTMLDivElement | null>(null);
  const stateRef = useRef<HandState>(createHandState());
  const handRef = useRef<Hand | null>(null);
  const phaseRef = useRef<Phase>("hand");
  const holdRef = useRef(new HoldTracker());
  const pointsRef = useRef<Partial<Record<string, Point>>>({});
  const boxRef = useRef<ReachBox | null>(null);
  const controlRef = useRef<FruitHarvestControl | null>(null);
  const audioRef = useRef<GameAudio | null>(null);
  const holdIndexRef = useRef(0);
  const lastPoseAtRef = useRef(0);
  // "all" walks the three holds; a HoldId repeats just that one, used
  // by Redo and by the post-hold sanity check.
  const calibModeRef = useRef<"all" | HoldId>("all");
  const poseMsgRef = useRef("");
  const gameRef = useRef<import("phaser").Game | null>(null);
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

  // ── Play: mount Phaser, tear it down on exit. Dynamic import keeps
  //    Phaser out of every other route's bundle.
  useEffect(() => {
    if (phase !== "play") return;
    const host = phaserHostRef.current;
    const box = boxRef.current;
    const h = handRef.current;
    if (!host || !box || !h) return;

    let game: import("phaser").Game | null = null;
    let cancelled = false;

    const debug = createGameDebug();
    debugRef.current = debug;
    debug.boxN = { x0: box.xLo, x1: box.xHi, y0: box.yLo, y1: box.yHi };

    const control: FruitHarvestControl = {
      state: stateRef.current,
      box,
      visualScale,
      audio: audioRef.current ?? new GameAudio(),
      harvested: 0,
      missed: 0,
      remainingMs: ROUND_MS,
      finished: false,
      debug,
      onFinish: (r) => {
        if (cancelled) return;
        setResult(r);
        setPhase("result");
        phaseRef.current = "result";
      },
    };
    controlRef.current = control;

    void (async () => {
      try {
        debug.sceneState = "importing phaser";
        const [Phaser, mod] = await Promise.all([
          import("phaser"),
          import("@/lib/games/fruitHarvestScene"),
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
          scene: [mod.FruitHarvestScene],
        });
        // Called before the game has booted, so the SceneManager parks
        // this in its holding pattern and injects `data` at bootQueue
        // (SceneManager.js:236). That is what gets `control` into
        // init() — the config array auto-starts scene 0 with no data.
        gameRef.current = game;
        game.scene.start("fruit-harvest", { control });
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
  }, [phase, visualScale]);

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
  // so watch the element itself. The scene re-lays-out the HUD and
  // basket when it sees the new size, and fruit carry normalised
  // coordinates so they stay reachable on their own.
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

  /** Repeat the "arm up" hold only; side and across are kept. */

  const acceptCalibration = useCallback(() => {
    setUpCheck(null);
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
  const accuracy =
    result && result.harvested + result.missed > 0
      ? Math.round((result.harvested / (result.harvested + result.missed)) * 100)
      : 0;

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-6">
      <div className="mb-4 flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-card bg-stone-500/10 text-stone-700 dark:text-stone-400">
          <Gamepad2 className="h-5 w-5" />
        </span>
        <div>
          <p className="eyebrow">Games</p>
          <h1 className="text-xl font-semibold tracking-tight">Fruit Harvest</h1>
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
            orchard and their hand, nothing else. The element stays
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
            <div className="mt-8 flex gap-4">
              <Button size="lg" onClick={() => beginWithHand("left")}>
                <HandIcon className="h-5 w-5 scale-x-[-1]" />
                Left hand
              </Button>
              <Button size="lg" onClick={() => beginWithHand("right")}>
                <HandIcon className="h-5 w-5" />
                Right hand
              </Button>
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
                <span role="img" aria-label="sample fruit">🍎</span>
              </div>
              <div className="mt-4 flex gap-3">
                <Button
                  variant="secondary"
                  onClick={() => setVisualScale((v) => Math.min(2.2, v * 1.35))}
                >
                  Make bigger
                </Button>
                <Button onClick={startCalibration} disabled={!live.setupOk}>
                  Yes — continue
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
            <div className="mt-8 flex gap-10 text-center">
              <Figure value={result.harvested} label="Harvested" tone="text-lime-300" />
              <Figure value={result.missed} label="Missed" tone="text-rose-300" />
              <Figure value={`${accuracy}%`} label="Accuracy" tone="text-white" />
            </div>
            <div className="mt-10 flex gap-3">
              <Button size="lg" onClick={playAgain}>
                <RotateCcw className="h-5 w-5" />
                Play again
              </Button>
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

      <p className="mt-3 text-center text-sm text-muted">
        Results are not saved yet.
      </p>
    </div>
  );
}

/** ?gamedebug=1 during setup and calibration — the play-phase panel
 *  only exists once Phaser is running, and these are the numbers that
 *  decide whether the calibration is worth anything. */
function SetupDebugPanel({ live, holdId }: { live: Live; holdId: string }) {
  const rows: [string, string][] = [
    ["arm length", `${live.armLenPx} px`],
    ["headroom", `${live.headroomPx} px`],
    [
      "ratio",
      `${live.headroomRatio.toFixed(2)}  (need >= ${HEADROOM_RATIO_MIN})`,
    ],
    [
      "landmarks",
      `nose ${live.noseOk ? "Y" : "N"} · sh ${live.shouldersOk ? "Y" : "N"}`
      + ` · elb ${live.elbowOk ? "Y" : "N"} · wr ${live.wristOk ? "Y" : "N"}`,
    ],
    ["setup ok", live.setupOk ? "YES" : "NO"],
    [
      "L15 (labelled left)",
      `x${live.l15.x} y${live.l15.y} vis ${live.l15.vis.toFixed(2)}`
      + `${live.l15.inFrame ? "" : " OUT"}`,
    ],
    [
      "R16 (labelled right)",
      `x${live.r16.x} y${live.r16.y} vis ${live.r16.vis.toFixed(2)}`
      + `${live.r16.inFrame ? "" : " OUT"}`,
    ],
    [
      "game is using",
      `${live.usingWrist === 15 ? "L15" : "R16"}`
      + `${live.swapped ? "  (handswap=1)" : ""}`,
    ],
    ["hold", holdId],
    ["ring status", live.holdStatus],
    [
      "pose met",
      live.holdStatus === "blocked" ? `NO — ${live.blockReason}` : "yes",
    ],
    [
      "reach (arm lengths)",
      `chosen ${live.myReach.toFixed(2)} · other ${live.otherReach.toFixed(2)}`
      + "  (need >= 0.60)",
    ],
    ["ring progress", `${Math.round(live.progress * 100)}%`],
  ];
  return (
    <div className="pointer-events-none absolute left-2 top-2 z-20 max-w-[24rem] rounded-md bg-black/80 p-3 font-mono text-[11px] leading-relaxed text-lime-300">
      <p className="mb-1 font-bold text-white">gamedebug · setup</p>
      {rows.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <span className="w-28 shrink-0 text-white/50">{k}</span>
          <span className="break-all">{v}</span>
        </div>
      ))}
    </div>
  );
}

/** ?gamedebug=1 — live scene diagnostics, drawn above the canvas. */
function DebugPanel({ d }: { d: GameDebug }) {
  const box = d.boxPx;
  const rows: [string, string][] = [
    ["phaser created", d.phaserCreated ? "YES" : "NO"],
    ["scene state", d.sceneState],
    ["textures ok", d.texturesOk ? "YES" : "NO"],
    ["canvas", `${d.canvasW} x ${d.canvasH}  z-index ${d.canvasZ}`],
    [
      "reach box (px)",
      box
        ? `x ${box.x0}..${box.x1}  y ${box.y0}..${box.y1}`
        : "not projected yet",
    ],
    [
      "reach box (norm)",
      d.boxN
        ? `x ${d.boxN.x0.toFixed(2)}..${d.boxN.x1.toFixed(2)}  `
          + `y ${d.boxN.y0.toFixed(2)}..${d.boxN.y1.toFixed(2)}`
        : "none",
    ],
    ["hand lost", d.handLost ? "YES — round held" : "no"],
    ["arm length", `${d.armLenPx} px`],
    ["headroom", `${d.headroomPx} px  (ratio ${d.headroomRatio})`],
    ["pose rate", `${d.poseHz} Hz`],
    ["render fps", `${d.fps}  (min ${d.fpsMin})`],
    ["filter cutoff", `${d.cutoffHz} Hz`],
    ["lag px", `${d.lagPx}  (raw palm -> drawn cursor)`],
    [
      "palm from",
      d.palmSource === "hand"
        ? "hand landmarks"
        : d.palmSource === "elbow"
          ? "elbow (projected)"
          : "WRIST — cursor is short of the hand",
    ],
    [
      "palm frames",
      `hand ${d.palmCounts.hand} · elbow ${d.palmCounts.elbow} · `
      + `wrist ${d.palmCounts.wrist}`,
    ],
    ["tweens / objects", `${d.tweens} / ${d.objects}`],
    ["fruit spawned", String(d.spawnedTotal)],
    ["fruit on screen", String(d.onScreen)],
    ["last spawn", d.lastSpawn],
    [
      "hand",
      `live ${d.handLive ? "Y" : "N"} · in frame ${d.handInFrame ? "Y" : "N"}`,
    ],
    ["cursor", `${d.cursorX}, ${d.cursorY}`],
    [
      "clock",
      `elapsed ${(d.elapsedMs / 1000).toFixed(1)}s · `
        + `remaining ${(d.remainingMs / 1000).toFixed(1)}s`,
    ],
  ];
  return (
    <div className="pointer-events-none absolute left-2 top-2 z-20 max-w-[24rem] rounded-md bg-black/80 p-3 font-mono text-[11px] leading-relaxed text-lime-300">
      <p className="mb-1 font-bold text-white">gamedebug</p>
      {rows.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <span className="w-28 shrink-0 text-white/50">{k}</span>
          {/* The wrist fallback means the cursor is drawn short of the
              hand — call it out rather than let it pass as normal. */}
          <span
            className={`break-all ${
              v.startsWith("WRIST") ? "font-bold text-rose-400" : ""
            }`}
          >
            {v}
          </span>
        </div>
      ))}
      {d.error && (
        <p className="mt-2 text-rose-400">error: {d.error}</p>
      )}
    </div>
  );
}

function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/55 px-6 text-center backdrop-blur-[2px]">
      {children}
    </div>
  );
}

/**
 * Live headroom gauge.
 *
 * A raised arm puts the wrist about one arm length above the shoulder,
 * so the bar is scaled in arm lengths with the pass mark at
 * HEADROOM_RATIO_MIN. The patient can watch it move as they step back
 * or the camera tilts, which is the whole point — the old check gave
 * no way to tell you were short until calibration silently failed.
 */
function HeadroomBar({ live }: { live: Live }) {
  const target = HEADROOM_RATIO_MIN;
  const pct = Math.min(100, (live.headroomRatio / (target * 1.4)) * 100);
  const markPct = (target / (target * 1.4)) * 100;
  const ok = live.headroomOk;
  const shortBy = Math.max(0, Math.round(target * live.armLenPx - live.headroomPx));

  return (
    <div className="mt-6 w-full max-w-md">
      <div className="flex items-baseline justify-between text-base">
        <span className={ok ? "text-lime-300" : "text-amber-300"}>
          Room above your head
        </span>
        <span className="font-mono text-sm text-white/60">
          {live.headroomRatio.toFixed(2)} / {target.toFixed(2)} arm lengths
        </span>
      </div>
      <div className="relative mt-2 h-5 overflow-hidden rounded-full bg-white/15">
        <div
          className={`h-full transition-[width] duration-150 ${
            ok ? "bg-lime-400" : "bg-amber-400"
          }`}
          style={{ width: `${pct}%` }}
        />
        {/* Pass mark */}
        <div
          className="absolute inset-y-0 w-0.5 bg-white"
          style={{ left: `${markPct}%` }}
        />
      </div>
      {!ok && (
        <p className="mt-3 text-2xl font-semibold text-amber-300">
          {live.headroomRatio > 0.75
            ? "Move back a little"
            : "Tilt the camera down"}
        </p>
      )}
      {!ok && live.armLenPx > 0 && (
        <p className="mt-1 text-sm text-white/60">
          About {shortBy} px more room needed above your shoulder.
        </p>
      )}
      {ok && (
        <p className="mt-3 text-lg text-lime-300">
          Good — your raised arm will be in view.
        </p>
      )}
    </div>
  );
}

function Tick({ ok, label }: { ok: boolean; label: string }) {
  return (
    <li className="flex items-center justify-center gap-3 text-white">
      <span
        className={`flex h-7 w-7 items-center justify-center rounded-full ${
          ok ? "bg-lime-500 text-black" : "bg-white/15 text-white/40"
        }`}
      >
        <Check className="h-4 w-4" />
      </span>
      <span className={ok ? "text-white" : "text-white/50"}>{label}</span>
    </li>
  );
}

function Figure({
  value,
  label,
  tone,
}: {
  value: number | string;
  label: string;
  tone: string;
}) {
  return (
    <div>
      <p className={`text-6xl font-bold leading-none ${tone}`}>{value}</p>
      <p className="mt-2 text-base text-white/60">{label}</p>
    </div>
  );
}

/** 5 s ring. Fills only while the wrist is inside the still tolerance;
 *  any drift resets `progress` in the tracker and the ring follows. */
function ProgressRing({ progress, active }: { progress: number; active: boolean }) {
  const R = 54;
  const C = 2 * Math.PI * R;
  const secs = Math.max(0, HOLD_MS / 1000 - progress * (HOLD_MS / 1000));
  return (
    <div className="relative h-32 w-32">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={R} className="fill-none stroke-white/15" strokeWidth="10" />
        <circle
          cx="64"
          cy="64"
          r={R}
          className={`fill-none ${active ? "stroke-lime-400" : "stroke-white/30"}`}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={C}
          strokeDashoffset={C * (1 - progress)}
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-3xl font-bold text-white">
        {Math.ceil(secs)}
      </span>
    </div>
  );
}

/** Large stick figure showing the pose to copy. Mirrored to match the
 *  selfie view, so the drawn arm is on the same side of the screen as
 *  the patient's own. */
function HoldFigure({ id, hand }: { id: string; hand: Hand }) {
  // Screen-space arm direction. In the mirrored view the patient's
  // right hand appears on the right, so "same side" points right for a
  // right-handed session.
  const sameRight = hand === "right";
  const arm =
    id === "up"
      ? { x: sameRight ? 74 : 54, y: 22 }
      : id === "side"
        ? { x: sameRight ? 112 : 16, y: 52 }
        : { x: sameRight ? 26 : 102, y: 46 };

  return (
    <svg viewBox="0 0 128 128" className="h-32 w-32" aria-hidden>
      <circle cx="64" cy="22" r="11" className="fill-white/85" />
      <line x1="64" y1="33" x2="64" y2="80" className="stroke-white/85" strokeWidth="7" strokeLinecap="round" />
      <line x1="64" y1="80" x2="48" y2="118" className="stroke-white/85" strokeWidth="7" strokeLinecap="round" />
      <line x1="64" y1="80" x2="80" y2="118" className="stroke-white/85" strokeWidth="7" strokeLinecap="round" />
      {/* resting arm */}
      <line
        x1="64"
        y1="46"
        x2={sameRight ? 40 : 88}
        y2="74"
        className="stroke-white/35"
        strokeWidth="7"
        strokeLinecap="round"
      />
      {/* the arm being held */}
      <line
        x1="64"
        y1="46"
        x2={arm.x}
        y2={arm.y}
        className="stroke-lime-400"
        strokeWidth="8"
        strokeLinecap="round"
      />
      <circle cx={arm.x} cy={arm.y} r="7" className="fill-lime-400" />
    </svg>
  );
}
