// Rehab knee-angle display conversion — pure JS, no React.
//
// WHY THIS EXISTS
// ─────────────────
// Historically the squat-family rep-count exercises (squat,
// mini-squat, single-leg-squat, step-up, lateral-step) feed the
// rep-count engine with `interior = 180 − flexion`. The engine
// requires the signal to be HIGH at the top of a rep (standing) and
// LOW at the bottom (deep squat), so this convention was baked in
// on purpose. Interior values were ALSO what got persisted into
// the report payload as `signal.value_at_peak` with
// `signal.name === "knee_interior"`.
//
// Physios and patients read joint angles in FLEXION convention
// (standing ≈ 0°, deep squat ≈ 70°). Interior (standing 180°, deep
// squat 110°) reads backwards to them.
//
// This helper is a **display-only** conversion applied at every
// render site:
//
//   • Old sessions (name === "knee_interior") → subtract from 180
//     for the value AND for BOTH ends of the target_band, then swap
//     the band ends so min < max. Do NOT skip the swap: without it
//     min > max and every downstream consumer (TargetVsValue,
//     ClinicalMetricCard, ProgressTrendChart) breaks silently.
//   • New sessions (name === "knee_flexion") → the payload was
//     written in flexion convention already; passthrough.
//   • Anything else (bridge, wall-sit, marching, hip-abduction, …)
//     → return null so the caller falls through to its existing
//     render path unchanged.
//
// ⚠️  DO NOT USE THIS ON THE REP-COUNT ENGINE FEED. The engine
//     still receives `interior` directly from each page's
//     handleFrame. Running the flexion conversion on that input
//     inverts the state machine's polarity and breaks rep counting
//     on every one of the 10 rep-count exercises.

const LABEL = "Knee angle (°)";
const HINT = "higher = deeper";

/**
 * @typedef {{ min: number, max: number }} Band
 * @typedef {{ name?: string, unit?: string,
 *   value_at_peak?: number | null,
 *   target_band?: Band }} SignalLike
 * @typedef {{ label: string, value: number|null, band: Band|null,
 *   hint: string }} DisplaySignal
 */

function isFiniteNumber(x) {
  return typeof x === "number" && Number.isFinite(x);
}

function readBand(raw) {
  if (!raw || typeof raw !== "object") return null;
  const mn = isFiniteNumber(raw.min) ? raw.min : null;
  const mx = isFiniteNumber(raw.max) ? raw.max : null;
  if (mn === null || mx === null) return null;
  return { min: mn, max: mx };
}

/**
 * Convert a persisted `signal` object into display units.
 * @param {SignalLike|null|undefined} signal
 * @returns {DisplaySignal|null}
 */
export function toDisplayKneeAngle(signal) {
  if (!signal || typeof signal !== "object") return null;
  const name = typeof signal.name === "string" ? signal.name : null;
  if (name !== "knee_interior" && name !== "knee_flexion") return null;

  const rawValue = isFiniteNumber(signal.value_at_peak)
    ? signal.value_at_peak
    : null;
  const rawBand = readBand(signal.target_band);

  if (name === "knee_interior") {
    // Interior → flexion: subtract from 180, and INVERT the band
    // (swap ends so min < max stays true after the numerical flip).
    const value = rawValue !== null ? 180 - rawValue : null;
    const band = rawBand
      ? { min: 180 - rawBand.max, max: 180 - rawBand.min }
      : null;
    return { label: LABEL, value, band, hint: HINT };
  }

  // knee_flexion — passthrough (already in display convention).
  //
  // POISONED-RECORD RESCUE: a short-lived intermediate build wrote
  // `name: "knee_flexion"` + converted band (e.g. {20,70}) but fed
  // `value_at_peak` from the RAW interior tracker (e.g. 123.6 for a
  // normal squat whose true flexion is 56.4). Those records claim
  // flexion but hold interior, and by name alone they're
  // indistinguishable from good ones.
  //
  // Heuristic — deliberately BAND-AWARE so it never touches
  // exercises whose legit flexion lives high (wall-sit saves
  // knee_flexion with band {80,100} and real values ~90°):
  // rescue ONLY when the band itself is a squat-family converted
  // band (max ≤ 70°) AND the value sits implausibly far above it
  // (> 90°). Only the poisoned squat-family records match both.
  const looksPoisoned =
    rawValue !== null
    && rawValue > 90
    && rawBand !== null
    && rawBand.max <= 70;
  const value = looksPoisoned ? 180 - rawValue : rawValue;
  return { label: LABEL, value, band: rawBand, hint: HINT };
}

/**
 * Convert a LIVE interior-angle scalar into its flexion display
 * value. Used by RepCountShell + squat-family pages when they need
 * to relabel a running number without disturbing the engine feed.
 *
 * ⚠️  Never feed the return value back into `repCountStep` or any
 *     `RepCountShell.signal` prop — the engine expects interior.
 *
 * @param {number|null|undefined} rawInterior
 * @returns {number|null}
 */
export function interiorToDisplayFlexion(rawInterior) {
  if (!isFiniteNumber(rawInterior)) return null;
  return 180 - rawInterior;
}

export const KNEE_ANGLE_DISPLAY_LABEL = LABEL;
export const KNEE_ANGLE_DISPLAY_HINT = HINT;
