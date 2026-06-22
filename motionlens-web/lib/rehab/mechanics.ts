// Rehab module — 7 pure scoring engines, one per game mechanic.
//
// Each engine is a function: (priorState, latestValue(s), config, nowMs)
// → MechanicResult<NewState>. They are PURE — no imports from React,
// no DOM, no camera, no exercise-specific knowledge. Testable in
// isolation. The UI shells in components/rehab/mechanics/ wrap these
// with useState/useRef and drive them with the input signal of the
// exercise that's been plugged in.
//
// All engines are session-cumulative: the caller initialises with
// emptyXxxState() and feeds in the latest value every frame; the
// engine integrates time, accumulates score, and emits a one-off
// MechanicEvent on noteworthy transitions. The caller can render
// score / status / events as it sees fit.

import {
  type GameStatus,
  type HoldInZoneConfig,
  type HoldInZoneState,
  type MatchPoseConfig,
  type MatchPoseState,
  type MechanicEvent,
  type MechanicResult,
  type MetronomeBeatRecord,
  type MetronomeConfig,
  type MetronomeGrade,
  type MetronomeState,
  type ReachTarget,
  type RepCountConfig,
  type RepCountState,
  type RepPhase,
  type Score,
  type TargetReachConfig,
  type TargetReachState,
  type TraceConfig,
  type TracePathPoint,
  type TraceState,
  type WeightShiftConfig,
  type WeightShiftState,
} from "@/lib/rehab/gameState";

// ─── Helpers ────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function dist2D(
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function updateBestStreak(score: Score): Score {
  return score.streak > score.bestStreak
    ? { ...score, bestStreak: score.streak }
    : score;
}

// ─── 1. Hold-in-Zone ────────────────────────────────────────────

export function holdInZoneStep(
  prior: HoldInZoneState,
  score: Score,
  value: number,
  config: HoldInZoneConfig,
  nowMs: number,
): MechanicResult<HoldInZoneState> {
  const hysteresis = config.hysteresis ?? 0;
  const min = config.min;
  const max = config.max;

  // Hysteresis: once in-zone, stay in-zone until we cross the
  // band ± hysteresis. Stops chatter on noisy signals.
  let inZone = prior.inZone;
  if (inZone) {
    if (value < min - hysteresis || value > max + hysteresis) {
      inZone = false;
    }
  } else {
    if (value >= min && value <= max) {
      inZone = true;
    }
  }

  let dtMs = 0;
  if (prior.lastUpdatedAt !== null) {
    dtMs = Math.max(0, nowMs - prior.lastUpdatedAt);
  }

  let currentDwellMs = prior.currentDwellMs;
  let totalMsInZone = prior.totalMsInZone;
  let bestDwellMs = prior.bestDwellMs;
  let event: MechanicEvent | null = null;
  let nextScore = score;

  if (inZone) {
    currentDwellMs += dtMs;
    totalMsInZone += dtMs;
    if (currentDwellMs > bestDwellMs) bestDwellMs = currentDwellMs;
    const ppms = config.pointsPerMs ?? 0.01;
    nextScore = {
      ...nextScore,
      points: nextScore.points + dtMs * ppms,
    };
    if (!prior.inZone) {
      event = { kind: "entered_zone", at: nowMs };
    }
  } else {
    if (prior.inZone) {
      event = {
        kind: "exited_zone",
        at: nowMs,
        payload: { dwellMs: prior.currentDwellMs },
      };
    }
    currentDwellMs = 0;
  }

  // Streak = current consecutive dwell expressed as a coarse rep
  // (every full required hold = +1 streak). Keeps the HUD lively.
  const newStreak = Math.floor(currentDwellMs / Math.max(100, config.targetHoldMs / 4));
  if (newStreak !== nextScore.streak) {
    nextScore = updateBestStreak({ ...nextScore, streak: newStreak });
  }

  const complete = totalMsInZone >= config.targetHoldMs;
  const status: GameStatus = complete ? "complete" : "running";

  return {
    state: {
      totalMsInZone,
      currentDwellMs,
      bestDwellMs,
      inZone,
      lastValue: value,
      lastUpdatedAt: nowMs,
    },
    score: nextScore,
    status,
    event,
  };
}

// ─── 2. Rep Count Gate ──────────────────────────────────────────

export function repCountStep(
  prior: RepCountState,
  score: Score,
  value: number,
  config: RepCountConfig,
  nowMs: number,
): MechanicResult<RepCountState> {
  // State machine — one rep = above_top → descending → below_depth
  // → ascending → above_top. Min/max tracked per rep so the gate
  // can check amplitude when the rep closes.
  let phase: RepPhase = prior.phase;
  let reps = prior.reps;
  let goodReps = prior.goodReps;
  let currentRepMin = prior.currentRepMin;
  let currentRepMax = prior.currentRepMax;
  let currentRepPeakJerk = prior.currentRepPeakJerk;
  let lastRepDowngrade = prior.lastRepDowngrade;
  let event: MechanicEvent | null = null;
  let nextScore = score;

  if (value < currentRepMin) currentRepMin = value;
  if (value > currentRepMax) currentRepMax = value;

  // Jerk = |Δvalue / Δt| (units per ms).
  if (
    prior.lastValue !== null
    && prior.lastUpdatedAt !== null
    && nowMs > prior.lastUpdatedAt
  ) {
    const jerk = Math.abs(value - prior.lastValue) / (nowMs - prior.lastUpdatedAt);
    if (jerk > currentRepPeakJerk) currentRepPeakJerk = jerk;
  }

  if (phase === "init") {
    // Wait for the first crossing above topThreshold so we have a
    // known starting point.
    if (value >= config.topThreshold) {
      phase = "above_top";
    }
  } else if (phase === "above_top") {
    if (value < config.topThreshold) {
      phase = "descending";
    }
  } else if (phase === "descending") {
    if (value < config.depthThreshold) {
      phase = "below_depth";
    } else if (value >= config.topThreshold) {
      // Bounced back up without reaching depth — reset cycle bounds.
      phase = "above_top";
      currentRepMin = value;
      currentRepMax = value;
      currentRepPeakJerk = 0;
    }
  } else if (phase === "below_depth") {
    if (value > config.depthThreshold) {
      phase = "ascending";
    }
  } else if (phase === "ascending") {
    if (value >= config.topThreshold) {
      // Rep closes here. Apply quality gates.
      const amplitude = currentRepMax - currentRepMin;
      const shallow = amplitude < config.minAmplitude;
      const jerky =
        config.maxJerk != null && currentRepPeakJerk > config.maxJerk;
      const good = !shallow && !jerky;

      reps += 1;
      if (good) {
        goodReps += 1;
        nextScore = {
          ...nextScore,
          points: nextScore.points + (config.pointsPerRep ?? 10),
          streak: nextScore.streak + 1,
        };
        nextScore = updateBestStreak(nextScore);
      } else {
        nextScore = { ...nextScore, streak: 0 };
      }
      lastRepDowngrade = shallow
        ? "shallow"
        : jerky
        ? "jerky"
        : null;

      event = {
        kind: "rep_counted",
        at: nowMs,
        payload: {
          repIndex: reps,
          good,
          amplitude,
          peakJerk: currentRepPeakJerk,
          downgrade: lastRepDowngrade,
        },
      };

      // Reset cycle bounds for the next rep.
      currentRepMin = value;
      currentRepMax = value;
      currentRepPeakJerk = 0;
      phase = "above_top";
    } else if (value < config.depthThreshold) {
      // Dipped back down without finishing the rep — stay in
      // descending phase but don't break out.
      phase = "below_depth";
    }
  }

  return {
    state: {
      reps,
      goodReps,
      phase,
      currentRepMin,
      currentRepMax,
      lastValue: value,
      lastUpdatedAt: nowMs,
      currentRepPeakJerk,
      lastRepDowngrade,
    },
    score: nextScore,
    status: "running",
    event,
  };
}

// ─── 3. Target Reach ────────────────────────────────────────────

export function targetReachStep(
  prior: TargetReachState,
  score: Score,
  cursor: { x: number; y: number },
  config: TargetReachConfig,
  nowMs: number,
): MechanicResult<TargetReachState> {
  // Hit-test the cursor against every active target. Hits remove
  // the target; misses (TTL expiry) remove and apply a penalty.
  const hitMult = config.hitRadiusMultiplier ?? 1.0;
  const ppHit = config.pointsPerHit ?? 10;
  const ppMiss = config.pointsPerMiss ?? -2;

  let hits = prior.hits;
  let misses = prior.misses;
  let event: MechanicEvent | null = null;
  let nextScore = score;

  const remaining: ReachTarget[] = [];
  for (const t of prior.targets) {
    const reach = dist2D(cursor, t);
    if (reach <= t.radius * hitMult) {
      hits += 1;
      nextScore = {
        ...nextScore,
        points: nextScore.points + ppHit,
        streak: nextScore.streak + 1,
      };
      nextScore = updateBestStreak(nextScore);
      event = {
        kind: "hit",
        at: nowMs,
        payload: { targetId: t.id, reach },
      };
      continue;
    }
    if (
      t.ttlMs != null
      && t.ttlMs > 0
      && nowMs - t.spawnedAt >= t.ttlMs
    ) {
      misses += 1;
      nextScore = {
        ...nextScore,
        points: nextScore.points + ppMiss,
        streak: 0,
      };
      event = event ?? {
        kind: "miss",
        at: nowMs,
        payload: { targetId: t.id },
      };
      continue;
    }
    remaining.push(t);
  }

  // Excursion = how far the cursor has reached from screen centre.
  // Useful clinical signal — patients with small reach show small
  // excursion regardless of which targets they hit.
  const excursion = dist2D(cursor, { x: 0.5, y: 0.5 });
  const maxExcursion = excursion > prior.maxExcursion ? excursion : prior.maxExcursion;

  return {
    state: {
      targets: remaining,
      hits,
      misses,
      maxExcursion,
      lastUpdatedAt: nowMs,
    },
    score: nextScore,
    status: "running",
    event,
  };
}

/** Helper for spawning a new target. Pure — caller manages timing. */
export function spawnReachTarget(
  state: TargetReachState,
  target: ReachTarget,
): TargetReachState {
  return { ...state, targets: [...state.targets, target] };
}

// ─── 4. Trace ───────────────────────────────────────────────────

export function traceStep(
  prior: TraceState,
  score: Score,
  cursor: { x: number; y: number },
  pathTarget: TracePathPoint,
  config: TraceConfig,
  nowMs: number,
): MechanicResult<TraceState> {
  // Distance from path target → accuracy. Second derivative
  // (Δvelocity / Δt) of cursor position → jerk → smoothness.
  const dev = dist2D(cursor, pathTarget);

  let jerk = 0;
  if (
    prior.prevCursor !== null
    && prior.prevVelocity !== null
    && prior.prevUpdatedAt !== null
    && nowMs > prior.prevUpdatedAt
  ) {
    const dt = nowMs - prior.prevUpdatedAt;
    const vx = (cursor.x - prior.prevCursor.x) / dt;
    const vy = (cursor.y - prior.prevCursor.y) / dt;
    const ax = (vx - prior.prevVelocity.x) / dt;
    const ay = (vy - prior.prevVelocity.y) / dt;
    jerk = Math.hypot(ax, ay);
  }
  const velocity =
    prior.prevCursor !== null && prior.prevUpdatedAt !== null && nowMs > prior.prevUpdatedAt
      ? {
          x: (cursor.x - prior.prevCursor.x) / (nowMs - prior.prevUpdatedAt),
          y: (cursor.y - prior.prevCursor.y) / (nowMs - prior.prevUpdatedAt),
        }
      : { x: 0, y: 0 };

  const accurate = dev <= config.accuracyTolerance;
  const smooth = jerk <= config.smoothnessTolerance;

  const samples = prior.samples + 1;
  const accurateSamples = prior.accurateSamples + (accurate ? 1 : 0);
  const smoothSamples = prior.smoothSamples + (smooth ? 1 : 0);
  const meanDeviation =
    (prior.meanDeviation * prior.samples + dev) / samples;
  const meanJerk =
    (prior.meanJerk * prior.samples + jerk) / samples;

  let nextScore = score;
  let event: MechanicEvent | null = null;
  if (accurate && smooth) {
    nextScore = {
      ...nextScore,
      points: nextScore.points + (config.pointsPerSample ?? 1),
      streak: nextScore.streak + 1,
    };
    nextScore = updateBestStreak(nextScore);
  } else {
    if (nextScore.streak > 0) {
      event = { kind: "broke_streak", at: nowMs };
    }
    nextScore = { ...nextScore, streak: 0 };
  }

  return {
    state: {
      samples,
      accurateSamples,
      smoothSamples,
      meanDeviation,
      meanJerk,
      prevCursor: cursor,
      prevVelocity: velocity,
      prevUpdatedAt: nowMs,
    },
    score: nextScore,
    status: "running",
    event,
  };
}

// ─── 5. Weight Shift ────────────────────────────────────────────

export function weightShiftStep(
  prior: WeightShiftState,
  score: Score,
  shift: number,
  stepDetected: boolean,
  config: WeightShiftConfig,
  nowMs: number,
): MechanicResult<WeightShiftState> {
  // Find which zone (if any) contains the cursor.
  const cursor = clamp(shift, -1, 1);
  let currentZoneId: string | null = null;
  for (const z of config.zones) {
    if (Math.abs(cursor - z.centre) <= z.halfWidth) {
      currentZoneId = z.id;
      break;
    }
  }

  let dtMs = 0;
  if (prior.lastUpdatedAt !== null) {
    dtMs = Math.max(0, nowMs - prior.lastUpdatedAt);
  }

  let dwellMs = prior.dwellMs;
  let stepPausedMs = prior.stepPausedMs;
  let capturedZoneIds = prior.capturedZoneIds;
  let nextScore = score;
  let event: MechanicEvent | null = null;
  const stepPenalty = config.stepPausePenaltyPerMs ?? 0;

  if (stepDetected) {
    stepPausedMs += dtMs;
    if (stepPenalty > 0) {
      nextScore = {
        ...nextScore,
        points: nextScore.points - dtMs * stepPenalty,
      };
    }
    // Game auto-pauses dwell accumulation while stepping.
    dwellMs = 0;
    if (!prior.stepPaused) {
      event = { kind: "step_paused", at: nowMs };
    }
  } else if (currentZoneId === null) {
    // Outside all zones — reset dwell.
    if (prior.currentZoneId !== null) {
      event = {
        kind: "exited_zone",
        at: nowMs,
        payload: { zoneId: prior.currentZoneId },
      };
    }
    dwellMs = 0;
  } else if (currentZoneId !== prior.currentZoneId) {
    // Switched zones — reset dwell to start counting fresh.
    dwellMs = 0;
    event = {
      kind: "entered_zone",
      at: nowMs,
      payload: { zoneId: currentZoneId },
    };
  } else {
    dwellMs += dtMs;
    const zone = config.zones.find((z) => z.id === currentZoneId);
    if (
      zone
      && dwellMs >= zone.dwellMs
      && !capturedZoneIds.includes(currentZoneId)
    ) {
      capturedZoneIds = [...capturedZoneIds, currentZoneId];
      nextScore = {
        ...nextScore,
        points: nextScore.points + (config.pointsPerCapture ?? 25),
        streak: nextScore.streak + 1,
      };
      nextScore = updateBestStreak(nextScore);
      event = {
        kind: "zone_captured",
        at: nowMs,
        payload: { zoneId: currentZoneId },
      };
    }
  }

  const status: GameStatus =
    capturedZoneIds.length >= config.zones.length ? "complete" : "running";

  return {
    state: {
      cursor,
      currentZoneId,
      dwellMs,
      capturedZoneIds,
      stepPaused: stepDetected,
      stepPausedMs,
      lastUpdatedAt: nowMs,
    },
    score: nextScore,
    status,
    event,
  };
}

// ─── 6. Match Pose ──────────────────────────────────────────────

export function matchPoseStep(
  prior: MatchPoseState,
  score: Score,
  currentAngles: Record<string, number>,
  config: MatchPoseConfig,
  nowMs: number,
): MechanicResult<MatchPoseState> {
  const perJoint: Record<string, number> = {};
  let weightedSum = 0;
  let totalWeight = 0;

  for (const [joint, target] of Object.entries(config.pose)) {
    const current = currentAngles[joint];
    if (current === undefined || !Number.isFinite(current)) {
      perJoint[joint] = 0;
      continue;
    }
    const delta = Math.abs(current - target.value);
    // 0 → tolerance maps to 100 %; tolerance → 2 × tolerance maps
    // linearly to 0 %. Beyond 2 × tolerance stays at 0.
    let pct = 0;
    if (delta <= target.tolerance) {
      pct = 100;
    } else if (delta < 2 * target.tolerance) {
      pct = 100 * (1 - (delta - target.tolerance) / target.tolerance);
    }
    const weight = target.weight ?? 1;
    perJoint[joint] = pct;
    weightedSum += pct * weight;
    totalWeight += weight;
  }

  const matchPct = totalWeight > 0 ? weightedSum / totalWeight : 0;
  const achieved = matchPct >= config.achievedThresholdPct;

  let dtMs = 0;
  if (prior.lastUpdatedAt !== null) {
    dtMs = Math.max(0, nowMs - prior.lastUpdatedAt);
  }

  let achievedDwellMs = prior.achievedDwellMs;
  let bestDwellMs = prior.bestDwellMs;
  let nextScore = score;
  let event: MechanicEvent | null = null;

  if (achieved) {
    achievedDwellMs += dtMs;
    if (achievedDwellMs > bestDwellMs) bestDwellMs = achievedDwellMs;
    const ppms = config.pointsPerMs ?? 0.05;
    nextScore = { ...nextScore, points: nextScore.points + dtMs * ppms };
    if (!prior.achieved) {
      event = { kind: "pose_achieved", at: nowMs };
    }
  } else {
    if (prior.achieved) {
      event = {
        kind: "pose_lost",
        at: nowMs,
        payload: { dwellMs: prior.achievedDwellMs },
      };
    }
    achievedDwellMs = 0;
  }

  const status: GameStatus =
    achievedDwellMs >= config.requiredHoldMs ? "complete" : "running";

  return {
    state: {
      matchPct,
      perJoint,
      achievedDwellMs,
      bestDwellMs,
      achieved,
      lastUpdatedAt: nowMs,
    },
    score: nextScore,
    status,
    event,
  };
}

// ─── 7. Metronome ───────────────────────────────────────────────

/** Compute the scheduled beat time for beat index N (ms since
 *  session start, where session start = beat 0). */
export function metronomeBeatTime(beatIndex: number, bpm: number): number {
  return (beatIndex * 60_000) / bpm;
}

/** Find the closest expected beat to the given event time. */
function nearestBeatIndex(eventMs: number, bpm: number): number {
  return Math.max(0, Math.round((eventMs * bpm) / 60_000));
}

export function metronomeStep(
  prior: MetronomeState,
  score: Score,
  /** The event time in ms since performance.now() origin. The
   *  engine converts this to "ms since session start" internally
   *  using sessionStartedAt. */
  eventAtMs: number,
  config: MetronomeConfig,
  nowMs: number,
): MechanicResult<MetronomeState> {
  const sessionStartedAt = prior.sessionStartedAt ?? eventAtMs;
  const eventSessionMs = eventAtMs - sessionStartedAt;

  const beatIndex = nearestBeatIndex(eventSessionMs, config.bpm);
  const beatAt = metronomeBeatTime(beatIndex, config.bpm);
  const dev = eventSessionMs - beatAt;
  const absDev = Math.abs(dev);

  let grade: MetronomeGrade = "miss";
  if (absDev <= config.perfectWindowMs) grade = "perfect";
  else if (absDev <= config.goodWindowMs) grade = "good";

  const beat: MetronomeBeatRecord = {
    beatIndex,
    beatAt,
    eventAt: eventSessionMs,
    deviationMs: dev,
    grade,
  };

  const beats = [...prior.beats, beat];
  let perfectCount = prior.perfectCount;
  let goodCount = prior.goodCount;
  let missCount = prior.missCount;
  let currentStreak = prior.currentStreak;
  let nextScore = score;

  if (grade === "perfect") {
    perfectCount += 1;
    currentStreak += 1;
    nextScore = {
      ...nextScore,
      points: nextScore.points + (config.pointsPerfect ?? 10),
      streak: currentStreak,
    };
    nextScore = updateBestStreak(nextScore);
  } else if (grade === "good") {
    goodCount += 1;
    currentStreak += 1;
    nextScore = {
      ...nextScore,
      points: nextScore.points + (config.pointsGood ?? 5),
      streak: currentStreak,
    };
    nextScore = updateBestStreak(nextScore);
  } else {
    missCount += 1;
    currentStreak = 0;
    nextScore = { ...nextScore, streak: 0 };
  }

  // Running mean |dev| over graded (non-miss) beats only.
  const graded = perfectCount + goodCount;
  const meanAbsDeviationMs =
    grade === "miss"
      ? prior.meanAbsDeviationMs
      : graded === 1
      ? absDev
      : (prior.meanAbsDeviationMs * (graded - 1) + absDev) / graded;

  const bestStreak =
    currentStreak > prior.bestStreak ? currentStreak : prior.bestStreak;

  const event: MechanicEvent = {
    kind: `beat_${grade}`,
    at: nowMs,
    payload: { beatIndex, deviationMs: dev },
  };

  return {
    state: {
      sessionStartedAt,
      beats,
      perfectCount,
      goodCount,
      missCount,
      meanAbsDeviationMs,
      bestStreak,
      currentStreak,
    },
    score: nextScore,
    status: "running",
    event,
  };
}

/** Compute which beats are upcoming so the UI can render a
 *  countdown / pulse. Pure. */
export function metronomeUpcomingBeats(
  prior: MetronomeState,
  config: MetronomeConfig,
  nowMs: number,
  lookaheadMs: number,
): Array<{ beatIndex: number; beatAt: number; tMinusMs: number }> {
  const sessionStartedAt = prior.sessionStartedAt ?? nowMs;
  const sessionMs = nowMs - sessionStartedAt;
  const startBeat = Math.max(0, nearestBeatIndex(sessionMs, config.bpm));
  const beatPeriod = 60_000 / config.bpm;
  const count = Math.max(1, Math.ceil(lookaheadMs / beatPeriod));
  const out: Array<{ beatIndex: number; beatAt: number; tMinusMs: number }> = [];
  for (let i = 0; i < count; i++) {
    const idx = startBeat + i;
    const at = metronomeBeatTime(idx, config.bpm);
    out.push({ beatIndex: idx, beatAt: at, tMinusMs: at - sessionMs });
  }
  return out;
}
