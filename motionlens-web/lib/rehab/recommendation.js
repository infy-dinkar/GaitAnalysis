// Rehab exercise recommendation engine.
//
// Pipeline:
//   1. listPatientReports(patientId)
//   2. filter to assessments (module !== "rehab")
//   3. sort desc created_at, cap ~30
//   4. hydrate via Promise.all(getReport().catch(() => null))
//   5. dedupe latest-per-(module, body_part, side)
//   6. evaluate each exercise's deficitRules against the reports
//   7. rank matched exercises by (priority × severity × rule weight)
//
// Return shape is stable — future doctor-edit / prescription store
// can wrap this without altering the engine. See getPrescribedSet at
// the bottom of the file for the seam.

import { listPatientReports, getReport } from "@/lib/reports";
import { EXERCISE_INDICATIONS, INDICATED_SLUGS } from "@/lib/rehab/exerciseIndications";
import { loadPrescription } from "@/lib/rehab/prescriptions";

const HISTORY_CAP = 30;

/** Assessment reports are anything module != "rehab". */
function isAssessmentSummary(r) {
  return r && r.module && r.module !== "rehab";
}

/** Dedupe key groups the "same" assessment across time. Trendelenburg
 *  left+right survive independently (side is a dimension); biomech
 *  shoulder-flexion left is separate from shoulder-flexion right. */
function reportKey(r) {
  return [r.module, r.body_part ?? "", r.movement ?? "", r.side ?? ""].join("|");
}

/** Fetch + hydrate the patient's assessment set. Soft-fails per-row so
 *  partial hydration still surfaces recommendations. */
async function fetchAssessments(patientId) {
  let listRes;
  try {
    listRes = await listPatientReports(patientId);
  } catch {
    return [];
  }
  const rows = (listRes?.data ?? [])
    .filter(isAssessmentSummary)
    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))
    .slice(0, HISTORY_CAP);
  if (rows.length === 0) return [];
  const fulls = await Promise.all(
    rows.map((r) => getReport(r.id).catch(() => null)),
  );
  const hydrated = [];
  for (const f of fulls) {
    if (f && typeof f === "object") hydrated.push(f);
  }
  // Latest-per-key: rows already come newest-first, so first seen wins.
  const seen = new Set();
  const deduped = [];
  for (const r of hydrated) {
    const k = reportKey(r);
    if (seen.has(k)) continue;
    seen.add(k);
    deduped.push(r);
  }
  return deduped;
}

// ─── Rule evaluator ──────────────────────────────────────────────
// Each evaluator returns { hit: boolean, severity: number, reason: string }.
// Severity is a normalised 0..1 float; the recommender multiplies it
// by priority weight to produce the final exercise score.

const SEVERITY_ORDER = { low: 0.3, medium: 0.6, high: 1.0 };

/** Walk a dot-path through an object. Returns undefined if missing. */
function getPath(obj, path) {
  if (!obj || !path) return undefined;
  const parts = path.split(".");
  let cur = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = cur[p];
  }
  return cur;
}

/** Read a classification field from a report's metrics blob. Handles
 *  the two common layouts:
 *    • metrics.result.classification
 *    • metrics.left / metrics.right / metrics.trial / metrics.session
 *      each carrying a classification field
 *  A single positive side is enough to fire. */
function readClassifications(metrics, path) {
  if (!metrics || typeof metrics !== "object") return [];
  const out = [];
  if (path) {
    const v = getPath(metrics, path);
    if (typeof v === "string") out.push(v);
  }
  // Common fallbacks — safe to add even when a path was specified,
  // since the engine `equals` filter drops non-matches.
  const candidates = ["classification", "class", "result_class"];
  const containers = ["result", "left", "right", "trial", "session"];
  for (const key of candidates) {
    const v = metrics[key];
    if (typeof v === "string") out.push(v);
  }
  for (const c of containers) {
    const cont = metrics[c];
    if (cont && typeof cont === "object") {
      for (const key of candidates) {
        const v = cont[key];
        if (typeof v === "string") out.push(v);
      }
      // TUG variant: lsi_class lives at metrics.lsi_class, not
      // metrics.left.classification — pick it up explicitly.
      if (typeof cont.lsi_class === "string") out.push(cont.lsi_class);
    }
  }
  if (typeof metrics.lsi_class === "string") out.push(metrics.lsi_class);
  return out;
}

function evalClassification(rule, report) {
  const wanted = rule.equals ?? [];
  if (wanted.length === 0) return null;
  const classes = readClassifications(report.metrics, rule.path);
  for (const c of classes) {
    if (wanted.includes(c)) {
      // Heuristic severity: known "worst" tokens map high, otherwise 0.6.
      const sev = /(severe|very_high|significant|deficit|positive|valgus|weakness|high|below_norm|fail)/i.test(c)
        ? 1.0
        : /(moderate|elevated|warning|compensated|borderline|hesitant|mild|below|tight)/i.test(c)
          ? 0.65
          : 0.5;
      return { hit: true, severity: sev, reason: `${report.module}: ${c}` };
    }
  }
  return null;
}

/** Biomech target-shortfall — peak_magnitude below target[0] by
 *  shortfall_deg. Rules filter by body_part + movement. */
function evalTargetShortfall(rule, report) {
  if (report.module !== "biomech") return null;
  if (rule.body_part && report.body_part !== rule.body_part) return null;
  const mvSet = rule.movement
    ? Array.isArray(rule.movement) ? rule.movement : [rule.movement]
    : null;
  if (mvSet && !mvSet.includes(report.movement ?? "")) return null;
  const m = report.metrics ?? {};
  const peak = typeof m.peak_magnitude === "number" ? m.peak_magnitude : null;
  const target = Array.isArray(m.target) && m.target.length >= 2 ? m.target : null;
  if (peak == null || target == null) return null;
  const [lo] = target;
  if (typeof lo !== "number") return null;
  const shortfall = rule.shortfall_deg ?? 5;
  const gap = lo - peak;
  if (gap < shortfall) return null;
  const sev = Math.max(0.4, Math.min(1.0, gap / (lo || 1)));
  return {
    hit: true,
    severity: sev,
    reason: `${report.body_part} ${report.movement}: ${peak.toFixed(0)}° vs target ≥${lo}°`,
  };
}

/** Biomech compensation — any flagged compensation whose type is in
 *  the rule's `types` set and severity >= min_severity. */
function evalCompensation(rule, report) {
  if (report.module !== "biomech") return null;
  if (rule.body_part && report.body_part !== rule.body_part) return null;
  const m = report.metrics ?? {};
  const comps = Array.isArray(m.compensations) ? m.compensations : [];
  const wantedTypes = rule.types ?? [];
  const minSev = SEVERITY_ORDER[rule.min_severity ?? "medium"];
  let best = null;
  for (const c of comps) {
    if (!c || typeof c !== "object") continue;
    if (!c.flagged) continue;
    if (wantedTypes.length > 0 && !wantedTypes.includes(c.type)) continue;
    const sev = SEVERITY_ORDER[c.severity] ?? 0.5;
    if (sev < minSev) continue;
    if (!best || sev > best.severity) {
      best = { hit: true, severity: sev, reason: `${report.body_part} compensation: ${c.label ?? c.type}` };
    }
  }
  return best;
}

/** Gait metric out of NORMAL_RANGES. We consume the metrics blob's
 *  `metrics_total`/`metrics_clean` (whichever exists) as the summary
 *  numbers; NORMAL_RANGES bounds are inlined so we don't couple to
 *  lib/gait/metrics.ts imports here. */
const GAIT_NORMAL_RANGES = {
  cadence: [100, 130],
  strideTime: [1.0, 1.2],
  walkingSpeed: [1.2, 1.5],
  symmetry: [0.95, 1.0],
  kneeFlexion: [55, 70],
  hipFlexion: [25, 40],
  kneeRom: [55, 75],
  hipRom: [40, 55],
};

function evalGaitNorm(rule, report) {
  if (report.module !== "gait") return null;
  const m = report.metrics ?? {};
  const summary = m.metrics_clean ?? m.metrics_total ?? m;
  const value = summary && typeof summary === "object" ? summary[rule.metric] : undefined;
  if (typeof value !== "number") return null;
  const range = GAIT_NORMAL_RANGES[rule.metric];
  if (!range) return null;
  const [lo, hi] = range;
  const isBelow = rule.direction === "below_low" && value < lo;
  const isAbove = rule.direction === "above_high" && value > hi;
  if (!isBelow && !isAbove) return null;
  const bound = isBelow ? lo : hi;
  const gap = Math.abs(value - bound);
  const sev = Math.max(0.4, Math.min(1.0, gap / (bound || 1)));
  return {
    hit: true,
    severity: sev,
    reason: `gait ${rule.metric} ${value.toFixed(2)} vs normal ${lo}–${hi}`,
  };
}

/** Gait observation tone — a saved Observation whose tone matches. */
function evalObservationTone(rule, report) {
  if (report.module !== "gait") return null;
  const obs = report.observations;
  if (!obs) return null;
  const list = Array.isArray(obs) ? obs : Array.isArray(obs?.items) ? obs.items : null;
  if (!list) return null;
  const wanted = rule.tone ?? "bad";
  for (const o of list) {
    if (o && typeof o === "object" && o.tone === wanted) {
      const text = typeof o.text === "string" ? o.text : "gait deficit";
      return { hit: true, severity: wanted === "bad" ? 0.8 : 0.55, reason: text };
    }
  }
  return null;
}

function evaluateRule(rule, report) {
  if (rule.module !== report.module) return null;
  switch (rule.when) {
    case "classification":       return evalClassification(rule, report);
    case "target_shortfall":     return evalTargetShortfall(rule, report);
    case "compensation":         return evalCompensation(rule, report);
    case "gait_norm":            return evalGaitNorm(rule, report);
    case "observation_tone":     return evalObservationTone(rule, report);
    default:                     return null;
  }
}

// ─── Public API ──────────────────────────────────────────────────

const PRIORITY_WEIGHT = { 1: 1.5, 2: 1.0, 3: 0.7 };

/**
 * @typedef {object} RecommendationReason
 * @property {string} reportId
 * @property {string} module
 * @property {string} reason
 * @property {number} severity
 */

/**
 * @typedef {object} Recommendation
 * @property {string} slug
 * @property {number} score            Composite ranking score
 * @property {RecommendationReason[]} reasons
 * @property {string} note             Prose from the indication entry
 * @property {1|2|3} priority
 */

/**
 * @typedef {object} RecommendationResult
 * @property {Recommendation[]} recommended      Ranked, non-empty subset
 * @property {number} deficitsFound              Total distinct rule-hits
 * @property {number} assessmentsUsed            Reports fed into the engine
 * @property {boolean} loadedPartially           True when some getReport() failed
 * @property {AbortSignal=} _n                   (reserved for future)
 */

/**
 * Score every indicated exercise against the patient's most recent
 * assessments. Highest score first. Returns { recommended: [] } when
 * no assessments hit any rules — the caller can render an empty
 * state.
 *
 * @param {string} patientId
 * @returns {Promise<RecommendationResult>}
 */
export async function computeRecommendations(patientId) {
  if (!patientId) {
    return { recommended: [], deficitsFound: 0, assessmentsUsed: 0, loadedPartially: false };
  }
  const reports = await fetchAssessments(patientId);
  const results = [];
  let totalHits = 0;
  for (const slug of INDICATED_SLUGS) {
    const ind = EXERCISE_INDICATIONS[slug];
    if (!ind) continue;
    const priority = ind.priority ?? 2;
    const pw = PRIORITY_WEIGHT[priority] ?? 1.0;
    const reasons = [];
    let score = 0;
    for (const rule of ind.deficitRules ?? []) {
      const ruleWeight = typeof rule.weight === "number" ? rule.weight : 1.0;
      for (const rep of reports) {
        const evalRes = evaluateRule(rule, rep);
        if (!evalRes) continue;
        score += evalRes.severity * ruleWeight * pw;
        reasons.push({
          reportId: rep.id,
          module: rep.module,
          reason: evalRes.reason,
          severity: evalRes.severity,
        });
        totalHits += 1;
      }
    }
    if (reasons.length > 0) {
      results.push({
        slug,
        score,
        reasons,
        note: ind.note ?? "",
        priority,
      });
    }
  }
  results.sort((a, b) => b.score - a.score || a.priority - b.priority);
  return {
    recommended: results,
    deficitsFound: totalHits,
    assessmentsUsed: reports.length,
    loadedPartially: false,
  };
}

// ─── Prescribed-set seam ─────────────────────────────────────────
//
// getPrescribedSet is the SINGLE source of truth the UI reads when it
// needs to know "which exercises are prescribed for this patient".
// Today prescribed === auto-recommended.
//
// FUTURE (doctor-edit layer): when a doctor-saved prescription lands,
// the ONLY change here will be to check a prescription store first —
// e.g.
//     const saved = await loadPrescription(patientId);
//     if (saved && !saved.expired) {
//       return { source: "doctor", slugs: new Set(saved.slugs), … };
//     }
// and fall back to the auto set below. No engine changes, no UI
// changes: adherence, launcher chip, sort order all read from this
// same function. That is the "clean seam" the PRD asked for.
//
// The returned shape intentionally carries `source` so a future UI
// can render "Prescribed by Dr. X on 2026-06-10" vs "Auto-recommended
// from your last assessments" with zero refactor.

/**
 * @typedef {object} PrescribedSetResult
 * @property {"auto"|"doctor"} source
 * @property {Set<string>} slugs              Prescribed / recommended exercise slugs
 * @property {Recommendation[]} recommended   Ranked list with reasons (auto tier)
 * @property {number} assessmentsUsed
 * @property {number} deficitsFound
 */

/**
 * Resolve the prescribed set for a patient.
 *
 *   Doctor-saved prescription is preferred when present. Otherwise
 *   falls back to the auto recommender.
 *
 * @param {string} patientId
 * @returns {Promise<PrescribedSetResult>}
 */
export async function getPrescribedSet(patientId) {
  // ─── DOCTOR-OVERRIDE SEAM ──────────────────────────────────────
  // Doctor-saved prescription wins when present. We ALSO run the auto
  // recommender because the UI still surfaces auto-derived reasons
  // (e.g. "why was pelvic-hold in the doctor's list — because the
  // patient's trendelenburg was positive"). Reasons are looked up per
  // slug from the auto result and attached back to doctor-picked
  // entries so the recommended-strip UI reads uniformly.
  // ────────────────────────────────────────────────────────────────
  const [saved, auto] = await Promise.all([
    loadPrescriptionSafely(patientId),
    computeRecommendations(patientId),
  ]);

  if (saved && Array.isArray(saved.slugs)) {
    const autoBySlug = new Map();
    for (const r of auto.recommended) autoBySlug.set(r.slug, r);
    const slugs = new Set(saved.slugs);
    // Build a Recommendation-shaped array for the prescribed set so
    // downstream UI (strip, badges) is source-agnostic.
    const recommended = [];
    for (const slug of saved.slugs) {
      const ind = EXERCISE_INDICATIONS[slug];
      const autoRec = autoBySlug.get(slug);
      recommended.push({
        slug,
        score: autoRec?.score ?? 0,
        reasons: autoRec?.reasons ?? [],
        note: ind?.note ?? "",
        priority: ind?.priority ?? 2,
      });
    }
    return {
      source: "doctor",
      slugs,
      recommended,
      assessmentsUsed: auto.assessmentsUsed,
      deficitsFound: auto.deficitsFound,
    };
  }

  return {
    source: "auto",
    slugs: new Set(auto.recommended.map((r) => r.slug)),
    recommended: auto.recommended,
    assessmentsUsed: auto.assessmentsUsed,
    deficitsFound: auto.deficitsFound,
  };
}

/** loadPrescription wrapped to soft-fail (network / 404-analog).
 *  Returning null cleanly falls back to auto. */
async function loadPrescriptionSafely(patientId) {
  try {
    return await loadPrescription(patientId);
  } catch {
    return null;
  }
}
