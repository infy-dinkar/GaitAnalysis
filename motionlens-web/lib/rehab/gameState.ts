// Rehab module — shared types for game mechanics + UI shells.
//
// Each of the 7 mechanic engines in lib/rehab/mechanics.ts is a PURE
// function: it takes a prior state, the latest signal value, and a
// config object, and returns the next state. The caller (typically
// the UI shell) holds the state via useState/useRef and re-invokes
// the engine on every input update. This file just defines the
// types those engines share.
//
// Nothing in here imports from the existing modules — keeping the
// rehab module self-contained so it can be reused / extracted /
// removed independently without ripple effects.

// ─── Common ─────────────────────────────────────────────────────

export type GameStatus =
  | "idle"
  | "ready"
  | "running"
  | "paused"
  | "complete"
  | "failed";

export interface Score {
  /** Total points accumulated this session. */
  points: number;
  /** Current consecutive successful events (reps / hits / on-beats). */
  streak: number;
  /** Best streak seen this session. */
  bestStreak: number;
}

export function emptyScore(): Score {
  return { points: 0, streak: 0, bestStreak: 0 };
}

/** Numeric difficulty knob applied uniformly across mechanics. 0 =
 *  easiest, 1 = default, 2+ = progressively harder. Each engine
 *  decides what the level means for it (tighter tolerance, faster
 *  spawn, etc.). */
export interface Difficulty {
  level: number;
  /** Optional per-mechanic override map — keys are mechanic ids,
   *  values are 0-2+ overrides for that mechanic. Lets a single
   *  session bump rep-count's depth gate while leaving hold-in-
   *  zone's tolerance loose. */
  overrides?: Partial<Record<MechanicId, number>>;
}

export const DEFAULT_DIFFICULTY: Difficulty = { level: 1 };

export type MechanicId =
  | "hold_in_zone"
  | "rep_count"
  | "target_reach"
  | "trace"
  | "weight_shift"
  | "match_pose"
  | "metronome";

/** Generic per-event feedback emitted by a mechanic engine when
 *  something noteworthy happened on the latest step (rep counted,
 *  target hit, beat missed, etc.). UI shells render these as
 *  toast-style overlays. */
export interface MechanicEvent {
  kind: string;       // mechanic-specific: "rep_counted" | "hit" | "miss" | ...
  payload?: Record<string, unknown>;
  at: number;         // performance.now() ms when emitted
}

/** Shared envelope returned by every mechanic step. */
export interface MechanicResult<S> {
  state: S;
  score: Score;
  status: GameStatus;
  /** Zero or one event per step — null when nothing notable. */
  event: MechanicEvent | null;
}

// ─── 1. Hold-in-Zone ────────────────────────────────────────────

export interface HoldInZoneConfig {
  /** Inclusive lower bound for the input signal to be "in zone". */
  min: number;
  /** Inclusive upper bound. */
  max: number;
  /** Total ms the patient must accumulate inside the band to
   *  succeed. */
  targetHoldMs: number;
  /** Score awarded per ms inside the zone (the simple per-tick
   *  rule — tune in the difficulty later). */
  pointsPerMs?: number;
  /** Hysteresis (units of the input signal) — once in-zone, the
   *  patient stays in-zone until they drift past band ± hysteresis.
   *  Stops a noisy signal from chattering across the boundary. */
  hysteresis?: number;
}

export interface HoldInZoneState {
  /** Cumulative ms inside the band, across the whole session. */
  totalMsInZone: number;
  /** Continuous in-zone time since the last exit. */
  currentDwellMs: number;
  /** Longest single dwell achieved this session. */
  bestDwellMs: number;
  /** Latest classification of the signal. */
  inZone: boolean;
  /** Last signal value fed in (so the shell can rerender without
   *  needing its own copy). */
  lastValue: number | null;
  /** Last update timestamp — used to integrate Δt. */
  lastUpdatedAt: number | null;
}

export function emptyHoldInZoneState(): HoldInZoneState {
  return {
    totalMsInZone: 0,
    currentDwellMs: 0,
    bestDwellMs: 0,
    inZone: false,
    lastValue: null,
    lastUpdatedAt: null,
  };
}

// ─── 2. Rep Count Gate ──────────────────────────────────────────

export interface RepCountConfig {
  /** Signal value below which the patient is in the "deep" half
   *  of the rep (e.g. squat bottom). */
  depthThreshold: number;
  /** Signal value above which the patient is in the "extended"
   *  half (e.g. squat top). depthThreshold < topThreshold. */
  topThreshold: number;
  /** Minimum amplitude the rep must reach to count as good (gate
   *  on shallow reps). max - min of the rep cycle. */
  minAmplitude: number;
  /** Optional maximum frame-to-frame jerk (signal Δ per ms) above
   *  which the rep is marked low-quality. Skip with null. */
  maxJerk?: number | null;
  /** Score per good rep. */
  pointsPerRep?: number;
}

export type RepPhase = "init" | "above_top" | "descending" | "below_depth" | "ascending";

export interface RepCountState {
  reps: number;
  goodReps: number;
  phase: RepPhase;
  /** Min seen in the current descending+below_depth window. */
  currentRepMin: number;
  /** Max seen in the current ascending+above_top window. */
  currentRepMax: number;
  lastValue: number | null;
  lastUpdatedAt: number | null;
  /** Peak jerk magnitude seen during the current rep. */
  currentRepPeakJerk: number;
  /** Reason the last rep was downgraded ("shallow" | "jerky" | null). */
  lastRepDowngrade: string | null;
}

export function emptyRepCountState(): RepCountState {
  return {
    reps: 0,
    goodReps: 0,
    phase: "init",
    currentRepMin: Infinity,
    currentRepMax: -Infinity,
    lastValue: null,
    lastUpdatedAt: null,
    currentRepPeakJerk: 0,
    lastRepDowngrade: null,
  };
}

// ─── 3. Target Reach ────────────────────────────────────────────

export interface ReachTarget {
  id: string;
  /** Normalised coords [0, 1] inside the play area. The shell
   *  scales them to its canvas dims. */
  x: number;
  y: number;
  /** Radius (normalised). */
  radius: number;
  /** Optional max time-to-live in ms — target disappears if not hit
   *  in time. null = persistent. */
  ttlMs?: number | null;
  /** When the target was spawned (performance.now()). */
  spawnedAt: number;
}

export interface TargetReachConfig {
  /** Pixel-space tolerance multiplier applied to each target's radius
   *  when testing for a hit. Difficulty knob. */
  hitRadiusMultiplier?: number;
  /** Score per target hit. */
  pointsPerHit?: number;
  /** Penalty for letting a target expire untouched. */
  pointsPerMiss?: number;
}

export interface TargetReachState {
  targets: ReachTarget[];
  hits: number;
  misses: number;
  /** Maximum normalised distance the cursor has reached from the
   *  origin (centre 0.5, 0.5). Lets clinicians see excursion. */
  maxExcursion: number;
  lastUpdatedAt: number | null;
}

export function emptyTargetReachState(): TargetReachState {
  return {
    targets: [],
    hits: 0,
    misses: 0,
    maxExcursion: 0,
    lastUpdatedAt: null,
  };
}

// ─── 4. Trace ───────────────────────────────────────────────────

export interface TracePathPoint {
  /** Normalised coordinates [0, 1]. */
  x: number;
  y: number;
}

export interface TraceConfig {
  /** Acceptable distance (normalised) of cursor from path target —
   *  beyond this the sample counts as off-path. */
  accuracyTolerance: number;
  /** Max acceptable jerk (Δ² distance / ms²). Beyond this the
   *  sample counts as jerky. */
  smoothnessTolerance: number;
  /** Score per accurate-AND-smooth sample. */
  pointsPerSample?: number;
}

export interface TraceState {
  /** Total samples evaluated. */
  samples: number;
  /** Samples where |cursor - path| <= accuracyTolerance. */
  accurateSamples: number;
  /** Samples below smoothnessTolerance jerk. */
  smoothSamples: number;
  /** Running mean deviation from the path. */
  meanDeviation: number;
  /** Running mean cursor jerk. */
  meanJerk: number;
  /** Previous cursor + velocity for jerk computation. */
  prevCursor: { x: number; y: number } | null;
  prevVelocity: { x: number; y: number } | null;
  prevUpdatedAt: number | null;
}

export function emptyTraceState(): TraceState {
  return {
    samples: 0,
    accurateSamples: 0,
    smoothSamples: 0,
    meanDeviation: 0,
    meanJerk: 0,
    prevCursor: null,
    prevVelocity: null,
    prevUpdatedAt: null,
  };
}

// ─── 5. Weight Shift ────────────────────────────────────────────

export interface WeightShiftZone {
  id: string;
  /** Normalised lateral position [−1, +1] this zone occupies. */
  centre: number;
  /** Half-width — patient must drive cursor within centre ± halfWidth. */
  halfWidth: number;
  /** Required dwell time inside the zone for a "capture". */
  dwellMs: number;
}

export interface WeightShiftConfig {
  zones: WeightShiftZone[];
  /** Score per captured zone. */
  pointsPerCapture?: number;
  /** Penalty applied per ms the patient is stepping (no points
   *  awarded during step pauses regardless). */
  stepPausePenaltyPerMs?: number;
}

export interface WeightShiftState {
  /** Patient's current lateral shift in [-1, +1]. */
  cursor: number;
  /** Zone the cursor is currently inside, or null. */
  currentZoneId: string | null;
  /** Ms accumulated in the current zone since entry. */
  dwellMs: number;
  /** IDs of all captured zones this session. */
  capturedZoneIds: string[];
  /** True while a step was detected — game auto-pauses dwell
   *  accumulation. */
  stepPaused: boolean;
  /** Cumulative ms spent in step-paused state. */
  stepPausedMs: number;
  lastUpdatedAt: number | null;
}

export function emptyWeightShiftState(): WeightShiftState {
  return {
    cursor: 0,
    currentZoneId: null,
    dwellMs: 0,
    capturedZoneIds: [],
    stepPaused: false,
    stepPausedMs: 0,
    lastUpdatedAt: null,
  };
}

// ─── 6. Match Pose ──────────────────────────────────────────────

export interface MatchPoseTargetJoint {
  /** Target value (degrees or whatever unit the source produces). */
  value: number;
  /** Tolerance — abs(current - target) <= tolerance scores 100 % on
   *  that joint; linearly decaying to 0 % at 2 × tolerance. */
  tolerance: number;
  /** Weight in the overall match — higher means this joint counts
   *  more toward the aggregate %. Defaults to 1. */
  weight?: number;
}

export interface MatchPoseConfig {
  /** Map of joint name → target. Keys are arbitrary; the consumer
   *  decides what "knee" / "elbow" / etc. mean. */
  pose: Record<string, MatchPoseTargetJoint>;
  /** Match % above which the pose counts as "achieved" for hold-
   *  time accumulation. */
  achievedThresholdPct: number;
  /** Total ms the pose must be achieved continuously to succeed. */
  requiredHoldMs: number;
  /** Score per ms of achieved hold. */
  pointsPerMs?: number;
}

export interface MatchPoseState {
  /** Most recent overall match % (0..100). */
  matchPct: number;
  /** Per-joint match % for the last update. */
  perJoint: Record<string, number>;
  /** Ms continuously above achievedThresholdPct. */
  achievedDwellMs: number;
  /** Longest achieved dwell across the session. */
  bestDwellMs: number;
  achieved: boolean;
  lastUpdatedAt: number | null;
}

export function emptyMatchPoseState(): MatchPoseState {
  return {
    matchPct: 0,
    perJoint: {},
    achievedDwellMs: 0,
    bestDwellMs: 0,
    achieved: false,
    lastUpdatedAt: null,
  };
}

// ─── 7. Metronome ───────────────────────────────────────────────

export interface MetronomeConfig {
  /** Beats per minute. */
  bpm: number;
  /** Acceptable |event - beat| in ms for a "perfect" beat. */
  perfectWindowMs: number;
  /** Wider window — within this counts as "good", outside as "miss". */
  goodWindowMs: number;
  /** Score for perfect / good. */
  pointsPerfect?: number;
  pointsGood?: number;
}

export type MetronomeGrade = "perfect" | "good" | "miss";

export interface MetronomeBeatRecord {
  /** Beat index since session start (0, 1, 2, ...). */
  beatIndex: number;
  /** Scheduled time of this beat (ms since session start). */
  beatAt: number;
  /** Actual event time the patient produced (ms since session
   *  start), or null if they missed it entirely. */
  eventAt: number | null;
  /** |eventAt - beatAt| in ms, or null when missed. */
  deviationMs: number | null;
  grade: MetronomeGrade;
}

export interface MetronomeState {
  /** Session start (performance.now() ms). Set on first event. */
  sessionStartedAt: number | null;
  beats: MetronomeBeatRecord[];
  perfectCount: number;
  goodCount: number;
  missCount: number;
  /** Running mean |deviation| across all graded beats (perfect+good). */
  meanAbsDeviationMs: number;
  /** Highest consecutive perfect/good streak. */
  bestStreak: number;
  /** Current streak. */
  currentStreak: number;
}

export function emptyMetronomeState(): MetronomeState {
  return {
    sessionStartedAt: null,
    beats: [],
    perfectCount: 0,
    goodCount: 0,
    missCount: 0,
    meanAbsDeviationMs: 0,
    bestStreak: 0,
    currentStreak: 0,
  };
}
