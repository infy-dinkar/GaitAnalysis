"use client";
// Presentation shared by every camera game's flow.
//
// All of it moved out of FruitHarvestGame.tsx unchanged. None of it
// knows what game is being played: these are the overlay, the setup
// ticks, the calibration ring and figure, the two big result numbers
// and the save status line — the furniture around whatever the game
// itself draws.

import { Check } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { HOLD_MS, type HoldStatus } from "@/lib/games/calibration";
import {
  HEADROOM_RATIO_MIN,
  type Hand,
  type Probe,
} from "@/lib/games/handTracker";
import type { GameDebugCore } from "@/lib/games/gameDebug";

/** Low-frequency copy of the mutable HandState, for the setup ticks and
 *  the calibration ring. The pose loop writes a ref at ~20 Hz; this is
 *  what React is allowed to see. */
export interface Live {
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

export const BLANK_PROBE: Probe = { x: 0, y: 0, vis: 0, inFrame: false };

export const BLANK_LIVE: Live = {
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

export type SaveState =
  | { status: "idle" }
  | { status: "saving" }
  | { status: "saved" }
  | { status: "none" }
  | { status: "error"; message: string };

export function Overlay({ children }: { children: React.ReactNode }) {
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
export function HeadroomBar({ live }: { live: Live }) {
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

export function Tick({ ok, label }: { ok: boolean; label: string }) {
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

/** One of the two big patient-facing numbers on the result screen. */
export function Figure({
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

/** Stacked label-over-value, matching GamesBody's Stat. */
export function ResultStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <p className="break-words text-xs uppercase tracking-[0.1em] text-white/40">
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold tabular text-white">{value}</p>
    </div>
  );
}

/** 5 s ring. Fills only while the wrist is inside the still tolerance;
 *  any drift resets `progress` in the tracker and the ring follows. */
export function ProgressRing({
  progress,
  active,
}: {
  progress: number;
  active: boolean;
}) {
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
export function HoldFigure({ id, hand }: { id: string; hand: Hand }) {
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

export function SaveStatus({
  state,
  patientName,
  onRetry,
}: {
  state: SaveState;
  patientName: string | null;
  onRetry: () => void;
}) {
  if (state.status === "idle") return null;
  if (state.status === "saving") {
    return <p className="mt-3 text-sm text-white/60">Saving…</p>;
  }
  if (state.status === "saved") {
    return (
      <p className="mt-3 text-sm text-lime-300">
        Saved to {patientName ? `${patientName}'s` : "the patient's"} reports
      </p>
    );
  }
  if (state.status === "none") {
    return (
      <p className="mt-3 text-sm text-white/50">
        Not saved (no patient selected)
      </p>
    );
  }
  return (
    <div className="mt-3 flex items-center gap-3">
      <p className="text-sm text-amber-300">{state.message}</p>
      <Button size="sm" variant="secondary" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}

/** ?gamedebug=1 during setup and calibration — the play-phase panel
 *  only exists once Phaser is running, and these are the numbers that
 *  decide whether the calibration is worth anything. */
export function SetupDebugPanel({
  live,
  holdId,
}: {
  live: Live;
  holdId: string;
}) {
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

/**
 * ?gamedebug=1 — live scene diagnostics, drawn above the canvas.
 *
 * The rows the shell knows about are fixed; the game's own rows come
 * from `d.extra` and are printed where the fruit counters used to sit,
 * between the object tally and the hand state.
 */
export function DebugPanel({ d }: { d: GameDebugCore }) {
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
    [
      "angle",
      `${d.angleDeg === null ? "—" : `${d.angleDeg}°`}  dir ${d.angleDir}`,
    ],
    ["decided by", d.angleDecidedBy],
    [
      "elbow (shoulder widths)",
      `dx ${d.elbowDx}  dy ${d.elbowDy}  fromMid ${d.elbowFromMid}`,
    ],
    [
      "wrist (shoulder widths)",
      `dx ${d.wristDx}  fromMid ${d.wristFromMid}`,
    ],
    ["counted this frame", d.angleCounted],
    ["running max abduction", `${d.maxAbductionDeg}°`],
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
    ...Object.entries(d.extra),
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
