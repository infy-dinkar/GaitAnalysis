// Proof-of-Progress artifact builder — pure client-side.
//
// Assembles the improvement graph for ONE (patient, exercise) pair:
//   • baseline     — earliest assessment matching the exercise's
//                    sharedMetric.assessmentModule (+ filters)
//   • trend        — rehab sessions of this exercise, sorted asc, each
//                    contributing signal.value_at_peak (optionally
//                    transformed) as a Y-value on the same axis
//   • reassessment — latest assessment AFTER the first rehab session
//                    (i.e. the "did rehab actually help" datapoint)
//
// Everything on ONE axis in the SAME units → this is the PRD closed-
// loop USP. `betterDirection` in the returned shape lets the chart
// draw the improving-arrow correctly.
//
// Fetch pattern is identical to lib/rehab/recommendation.js:
// listPatientReports → filter → sort → Promise.all(getReport().catch).
// No backend or engine touched.

import { listPatientReports, getReport } from "@/lib/reports";
import { indicationOf } from "@/lib/rehab/exerciseIndications";

const HISTORY_CAP = 60;

/** @typedef {import("@/lib/rehab/exerciseIndications").SharedMetric} SharedMetric */

/** @typedef {object} ProofPoint
 *  @property {string} date        ISO timestamp of the report
 *  @property {number} value       Y-axis value in `unit`
 *  @property {string} reportId    Link-back to the source report
 */

/** @typedef {object} ProofArtifact
 *  @property {string} slug              The exercise slug
 *  @property {string} unit
 *  @property {string} label
 *  @property {"higher"|"lower"} betterDirection
 *  @property {string=} caveat
 *  @property {ProofPoint | null} baseline
 *  @property {ProofPoint[]} trend
 *  @property {ProofPoint | null} reassessment
 *  @property {number=} improvementDelta   baseline.value → reassessment.value delta (positive = improved per direction)
 *  @property {number} assessmentsFound
 *  @property {number} sessionsFound
 */

/** Extract the scalar Y value for an ASSESSMENT report per the
 *  sharedMetric declaration. Returns null when the shape doesn't
 *  contain the field.
 *  @param {any} report
 *  @param {SharedMetric} shared */
function readAssessmentValue(report, shared) {
  const metrics = report?.metrics;
  if (!metrics || typeof metrics !== "object") return null;
  switch (shared.assessmentModule) {
    case "trendelenburg": {
      // Worse-side max_drop_deg — the clinically relevant scalar.
      const left = pickPathNum(metrics, "left.max_drop_deg");
      const right = pickPathNum(metrics, "right.max_drop_deg");
      if (left == null && right == null) return null;
      const worse = Math.max(left ?? -Infinity, right ?? -Infinity);
      return Number.isFinite(worse) ? worse : null;
    }
    case "ake": {
      const left = pickPathNum(metrics, "left.deficit_deg");
      const right = pickPathNum(metrics, "right.deficit_deg");
      if (left == null && right == null) return null;
      const worse = Math.max(left ?? -Infinity, right ?? -Infinity);
      return Number.isFinite(worse) ? worse : null;
    }
    case "biomech": {
      // Filter check happened at the summary-filter layer; here we
      // just read the top-level peak_magnitude.
      const v = pickPathNum(metrics, "peak_magnitude");
      return v ?? null;
    }
    case "gait": {
      const gm = shared.gaitMetric ?? "cadence";
      // Prefer clean metrics, fall back to total.
      return (
        pickPathNum(metrics, `metrics_clean.${gm}`)
        ?? pickPathNum(metrics, `metrics_total.${gm}`)
        ?? null
      );
    }
    default:
      return null;
  }
}

/** Extract the scalar Y value from a REHAB report. Applies the
 *  optional rehabTransform. */
function readRehabValue(report, shared) {
  const peak = pickPathNum(report?.metrics, "signal.value_at_peak");
  if (peak == null) return null;
  if (shared.rehabTransform === "deficit_from_180") {
    return 180 - peak;
  }
  return peak;
}

/** Summary-level filter — does this assessment summary match the
 *  sharedMetric's module + biomech filters? */
function matchesAssessmentSummary(summary, shared) {
  if (summary.module !== shared.assessmentModule) return false;
  if (shared.assessmentModule === "biomech") {
    if (shared.assessmentBodyPart && summary.body_part !== shared.assessmentBodyPart) {
      return false;
    }
    if (Array.isArray(shared.assessmentMovements) && shared.assessmentMovements.length > 0) {
      if (!shared.assessmentMovements.includes(summary.movement ?? "")) return false;
    }
  }
  return true;
}

/** Rehab-session summary filter — must be this exercise's slug. */
function matchesRehabSummary(summary, slug) {
  return summary.module === "rehab" && summary.movement === slug;
}

/**
 * Build the proof artifact for one (patient, exercise).
 *
 * Returns null when the exercise has no sharedMetric OR when neither
 * baseline nor any trend / re-assessment values could be recovered.
 * That way callers can `Array.filter(Boolean)` cleanly.
 *
 * @param {string} patientId
 * @param {string} slug
 * @returns {Promise<ProofArtifact | null>}
 */
export async function computeProofArtifact(patientId, slug) {
  if (!patientId || !slug) return null;
  const ind = indicationOf(slug);
  const shared = ind?.sharedMetric;
  if (!shared) return null;

  let list;
  try {
    list = await listPatientReports(patientId);
  } catch {
    return null;
  }
  const rows = Array.isArray(list?.data) ? list.data : [];
  const assessmentSummaries = rows.filter((r) => matchesAssessmentSummary(r, shared));
  const rehabSummaries = rows.filter((r) => matchesRehabSummary(r, slug));

  if (assessmentSummaries.length === 0 && rehabSummaries.length === 0) {
    return null;
  }

  // Cap total hydration so a very active patient doesn't runaway
  // fetch. Assessment set is small; rehab set is capped.
  const capped = [
    ...assessmentSummaries,
    ...rehabSummaries.slice(0, HISTORY_CAP),
  ];
  const fulls = await Promise.all(
    capped.map((s) => getReport(s.id).catch(() => null)),
  );

  const assessmentPoints = [];
  const trendPoints = [];
  for (const dto of fulls) {
    if (!dto) continue;
    if (matchesAssessmentSummary(dto, shared)) {
      const v = readAssessmentValue(dto, shared);
      if (v != null && Number.isFinite(v)) {
        assessmentPoints.push({
          date: dto.created_at,
          value: v,
          reportId: dto.id,
        });
      }
    } else if (matchesRehabSummary(dto, slug)) {
      const v = readRehabValue(dto, shared);
      if (v != null && Number.isFinite(v)) {
        trendPoints.push({
          date: dto.created_at,
          value: v,
          reportId: dto.id,
        });
      }
    }
  }

  assessmentPoints.sort((a, b) => a.date.localeCompare(b.date));
  trendPoints.sort((a, b) => a.date.localeCompare(b.date));

  const baseline = assessmentPoints[0] ?? null;
  const firstTrendDate = trendPoints[0]?.date ?? null;
  // Re-assessment = latest assessment AFTER the first rehab session.
  // If no rehab session yet, the "reassessment" concept doesn't apply;
  // we still return the latest assessment as informational context
  // via `baseline` (already earliest) — no reassessment.
  let reassessment = null;
  if (firstTrendDate && assessmentPoints.length > 0) {
    const laterAssessments = assessmentPoints.filter(
      (p) => p.date > firstTrendDate,
    );
    if (laterAssessments.length > 0) {
      reassessment = laterAssessments[laterAssessments.length - 1];
    }
  }

  let improvementDelta;
  if (baseline && reassessment) {
    const raw = reassessment.value - baseline.value;
    // Higher-better → improvement = positive delta.
    // Lower-better  → improvement = NEGATIVE delta, which we flip so
    // the UI always shows "positive number = better".
    improvementDelta = shared.betterDirection === "lower" ? -raw : raw;
  }

  if (!baseline && trendPoints.length === 0 && !reassessment) {
    return null;
  }

  return {
    slug,
    unit: shared.unit,
    label: shared.label,
    betterDirection: shared.betterDirection,
    caveat: shared.caveat,
    baseline,
    trend: trendPoints,
    reassessment,
    improvementDelta,
    assessmentsFound: assessmentPoints.length,
    sessionsFound: trendPoints.length,
  };
}
// ─── Internal helpers ─────────────────────────────────────────────

/** Dot-path number reader. Returns null if the value at path isn't a finite number. */
function pickPathNum(obj, path) {
  if (!obj || !path) return null;
  const parts = path.split(".");
  let cur = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return null;
    cur = /** @type {any} */ (cur)[p];
  }
  return typeof cur === "number" && Number.isFinite(cur) ? cur : null;
}
