// Clean-% helper — reads a saved rehab session's metrics blob and
// returns a 0-100 "quality" score per the spec's Section 8 rules.
// Used by useProgressionLevel to decide advance/hold/regress.

/**
 * @typedef {"rep_count" | "hold_in_zone" | "target_reach" | "trace" | "weight_shift" | "match_pose" | "metronome"} MechanicId
 */

/**
 * @param {MechanicId | string | null} mechanicId
 * @param {any} metrics — the saved rehab report's metrics blob
 * @returns {number} 0..100
 */
export function computeCleanPct(mechanicId, metrics) {
  if (!metrics || typeof metrics !== "object") return 0;
  const state = metrics.mechanic_state || {};
  const config = metrics.config || {};
  switch (mechanicId) {
    case "rep_count": {
      const reps = num(state.reps);
      const good = num(state.goodReps);
      if (reps <= 0) return 0;
      return clampPct((good / reps) * 100);
    }
    case "hold_in_zone": {
      const inZone = num(state.totalMsInZone);
      const target = num(metrics.target_hold_ms ?? config.targetHoldMs);
      if (target <= 0) return inZone > 0 ? 100 : 0;
      return clampPct((inZone / target) * 100);
    }
    case "target_reach": {
      const hits = num(state.hits);
      const misses = num(state.misses);
      const total = hits + misses;
      if (total <= 0) return 0;
      return clampPct((hits / total) * 100);
    }
    case "trace": {
      const samples = num(state.samples);
      const accurate = num(state.accurateSamples);
      const smooth = num(state.smoothSamples);
      if (samples <= 0) return 0;
      const accPct = (accurate / samples) * 100;
      const smoothPct = (smooth / samples) * 100;
      return clampPct((accPct + smoothPct) / 2);
    }
    case "match_pose": {
      const best = num(state.bestMatchPct);
      return clampPct(best);
    }
    case "metronome": {
      const perfect = num(state.perfectCount);
      const good = num(state.goodCount);
      const miss = num(state.missCount);
      const lifts = num(state.liftCount);
      const total = perfect + good + miss;
      if (total > 0) return clampPct(((perfect + good) / total) * 100);
      // Fallback for pages that only count lifts (e.g. marching):
      return clampPct(Math.min(100, lifts * 5));
    }
    case "weight_shift": {
      const captured = Array.isArray(state.capturedZoneIds)
        ? state.capturedZoneIds.length
        : 0;
      const totalZones = Array.isArray(config.zones)
        ? config.zones.length
        : 4;
      if (totalZones <= 0) return 0;
      return clampPct((captured / totalZones) * 100);
    }
    default:
      return 0;
  }
}

function num(v) {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}
function clampPct(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(100, v));
}
