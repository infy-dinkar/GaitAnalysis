"use client";
// useRehabAutoFlow — shared session phase machine for the reduced-
// click rehab flow, extracted from the validated Controlled Squat
// prototype (app/rehab/squat/page.tsx) so every exercise page gets
// the identical UX without duplicating the effect chains:
//
//   started=false          → phase null (side picker / start gate)
//   started flips true     → phase "countdown", 3-2-1 tick
//   countdown reaches 0    → phase "live" (+ onLive() so the page can
//                            reset its session-scoped refs / anchors)
//   markComplete() called  → phase "complete" (only from "live", so
//                            re-entrant snapshot callbacks can't
//                            re-fire it)
//   started flips false    → phase null again (Change side / Exit)
//
// Space / Escape (or the skipCountdown callback wired to a sidebar
// button) cancels the countdown and goes live immediately — same
// escape hatch the squat prototype shipped with.

import { useCallback, useEffect, useRef, useState } from "react";

export type RehabAutoFlowPhase = "countdown" | "live" | "complete";

const COUNTDOWN_START_SEC = 3;

export function useRehabAutoFlow(
  started: boolean,
  /** Called at the countdown→live (or skip→live) transition — reset
   *  session-scoped refs (sessionStartRef, peak trackers, snapshot
   *  refs) here so pre-session framing noise never leaks into the
   *  saved payload. */
  onLive?: () => void,
): {
  phase: RehabAutoFlowPhase | null;
  countdown: number | null;
  skipCountdown: () => void;
  markComplete: () => void;
} {
  const [phase, setPhase] = useState<RehabAutoFlowPhase | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);

  // Latest onLive in a ref so the tick effect depends only on the
  // countdown value (same pattern as RepCountShell's onSnapshot ref).
  const onLiveRef = useRef(onLive);
  onLiveRef.current = onLive;

  const goLive = useCallback(() => {
    setCountdown(null);
    setPhase("live");
    onLiveRef.current?.();
  }, []);

  // Seed / reset the machine when the start gate flips.
  useEffect(() => {
    if (!started) {
      setPhase(null);
      setCountdown(null);
      return;
    }
    setPhase("countdown");
    setCountdown(COUNTDOWN_START_SEC);
  }, [started]);

  // Countdown tick — flips to live at 0.
  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) {
      goLive();
      return;
    }
    const id = window.setTimeout(() => {
      setCountdown((c) => (c === null ? null : c - 1));
    }, 1000);
    return () => window.clearTimeout(id);
  }, [countdown, goLive]);

  // Space / Escape skips the countdown.
  useEffect(() => {
    if (countdown === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === " " || e.code === "Space") {
        e.preventDefault();
        goLive();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [countdown, goLive]);

  const markComplete = useCallback(() => {
    setPhase((prev) => (prev === "live" ? "complete" : prev));
  }, []);

  return { phase, countdown, skipCountdown: goLive, markComplete };
}
