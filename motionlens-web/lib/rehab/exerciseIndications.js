// Exercise -> Assessment-Deficit Indications
//
// Keyed by rehab exercise slug. Each entry lists:
//   • linkedAssessments — the assessment modules whose reports feed
//     this exercise's recommendation. Purely informational / used for
//     tooltips and the "why recommended" hint.
//   • deficitRules — a data-driven rule list the recommendation engine
//     evaluates against actual report metrics/observations. Rules are
//     interpreted generically by lib/rehab/recommendation.js — no
//     rule-specific code lives here.
//   • priority — base recommendation weight when a rule fires. Higher
//     priority (1..3) exercises rank first at equal severity. Serves
//     as the tie-breaker between "same deficit, different exercise
//     candidates" and downweights coarse-proxy exercises (e.g. S6
//     scapular-set is trend-only per the audit — priority 3).
//
// Rule DSL:
//   {
//     module: "trendelenburg" | "biomech" | "gait" | ...,   // report.module
//     when:   "classification" | "target_shortfall"
//             | "compensation" | "gait_norm" | "observation_tone",
//     // when === "classification":
//     equals?: string[],                 // classification in this set
//     path?:   string,                   // dot-path inside metrics; default "result.classification"
//     // when === "target_shortfall" (biomech):
//     body_part?: string,                // filters biomech report by body_part
//     movement?: string | string[],      // filters biomech report by movement
//     shortfall_deg?: number,            // peak_magnitude must be > shortfall_deg BELOW target[0]
//     // when === "compensation" (biomech):
//     types?: string[],                  // compensation.type in this set
//     min_severity?: "low" | "medium" | "high",
//     // when === "gait_norm":
//     metric: "cadence" | "walkingSpeed" | "symmetry" | ...,
//     direction: "below_low" | "above_high",
//     // when === "observation_tone":
//     tone: "bad" | "warn",
//     // Optional per-rule weight; defaults to 1.0. Amplifies the
//     // severity contribution when this rule matches.
//     weight?: number,
//   }
//
// This file is DATA. No branching, no fetch, no state — the engine in
// recommendation.js does all evaluation. That keeps the rule set
// auditable + patchable without touching the engine.
//
// Coverage (per audit): 4 explicit (H1, H5, K3, S3) + 20 inferred.
// The engine still returns matches on inferred rules because the
// underlying assessments do carry the deficit — the inference is only
// about which exercise TARGETS that deficit, not about the deficit
// signal existing.

/** @typedef {"trendelenburg" | "single_leg_squat" | "sit_to_stand" | "chair_stand_30s" |
 *            "single_leg_stance" | "four_stage_balance" | "tug" | "sppb" | "slr" | "ake" |
 *            "modified_thomas" | "forward_lunge" | "sts_quality" | "tandem_walk" |
 *            "pronator_drift" | "functional_reach" | "single_leg_hop" |
 *            "counter_movement_jump" | "biomech" | "gait" | "posture"} AssessmentModule */

/** @typedef {object} DeficitRule
 *  @property {AssessmentModule} module
 *  @property {"classification"|"target_shortfall"|"compensation"|"gait_norm"|"observation_tone"} when
 *  @property {string[]=} equals
 *  @property {string=} path
 *  @property {string=} body_part
 *  @property {string|string[]=} movement
 *  @property {number=} shortfall_deg
 *  @property {string[]=} types
 *  @property {"low"|"medium"|"high"=} min_severity
 *  @property {string=} metric
 *  @property {"below_low"|"above_high"=} direction
 *  @property {"bad"|"warn"=} tone
 *  @property {number=} weight
 */

/** @typedef {object} ExerciseIndication
 *  @property {string} slug
 *  @property {AssessmentModule[]} linkedAssessments
 *  @property {DeficitRule[]} deficitRules
 *  @property {1|2|3} priority
 *  @property {string=} note      Optional prose shown as fallback tooltip
 */

/** All 24 rehab exercises' indications. Keyed by slug.
 *  @type {Record<string, ExerciseIndication>} */
export const EXERCISE_INDICATIONS = {
  // ── KNEE ─────────────────────────────────────────────────────────
  "squat": {
    slug: "squat",
    linkedAssessments: ["sit_to_stand", "chair_stand_30s", "sppb"],
    priority: 1,
    deficitRules: [
      { module: "sit_to_stand", when: "classification", equals: ["borderline", "weakness"] },
      { module: "chair_stand_30s", when: "classification", equals: ["below_norm"] },
      { module: "sppb", when: "classification", equals: ["severe", "moderate", "mild"] },
    ],
    note: "Full-depth quad + hip strengthening for STS weakness.",
  },
  "mini-squat": {
    slug: "mini-squat",
    linkedAssessments: ["sit_to_stand", "chair_stand_30s", "sppb"],
    priority: 2,
    deficitRules: [
      { module: "sit_to_stand", when: "classification", equals: ["weakness"] },
      { module: "chair_stand_30s", when: "classification", equals: ["below_norm"] },
      { module: "sppb", when: "classification", equals: ["severe", "moderate"] },
    ],
    note: "Early-stage / deconditioned alternative to squat.",
  },
  "knee-extension": {
    slug: "knee-extension",
    linkedAssessments: ["ake"],
    priority: 1,
    deficitRules: [
      { module: "ake", when: "classification", equals: ["mild", "moderate", "severe"], weight: 1.4 },
    ],
    note: "Terminal knee extension for AKE deficit (post-op / quads inhibition).",
  },
  "step-up": {
    slug: "step-up",
    linkedAssessments: ["single_leg_hop", "sppb"],
    priority: 1,
    deficitRules: [
      { module: "single_leg_hop", when: "classification", equals: ["warning", "deficit"], path: "lsi_class" },
      { module: "sppb", when: "classification", equals: ["severe", "moderate"] },
    ],
    note: "Unilateral loading — targets LSI deficit + stair-climb strength.",
  },
  "wall-sit": {
    slug: "wall-sit",
    linkedAssessments: ["sit_to_stand"],
    priority: 2,
    deficitRules: [
      { module: "sit_to_stand", when: "classification", equals: ["borderline", "weakness"] },
    ],
    note: "Isometric quad endurance for STS weakness.",
  },
  "single-leg-squat": {
    slug: "single-leg-squat",
    linkedAssessments: ["single_leg_squat", "single_leg_hop"],
    priority: 1,
    deficitRules: [
      { module: "single_leg_squat", when: "classification", equals: ["borderline", "valgus"] },
      { module: "single_leg_squat", when: "classification", equals: ["moderate", "high"], path: "result.risk_score" },
      { module: "single_leg_hop", when: "classification", equals: ["deficit"], path: "lsi_class" },
    ],
    note: "Targets KFPPA valgus + LSI deficit.",
  },

  // ── HIP ──────────────────────────────────────────────────────────
  "pelvic-hold": {
    slug: "pelvic-hold",
    linkedAssessments: ["trendelenburg"],
    priority: 1,
    deficitRules: [
      { module: "trendelenburg", when: "classification", equals: ["positive", "compensated"], weight: 1.4 },
    ],
    note: "Trendelenburg retraining — hip-drop control.",
  },
  "hip-abduction": {
    slug: "hip-abduction",
    linkedAssessments: ["trendelenburg", "single_leg_squat"],
    priority: 2,
    deficitRules: [
      { module: "trendelenburg", when: "classification", equals: ["positive", "compensated"] },
      { module: "single_leg_squat", when: "classification", equals: ["borderline", "valgus"] },
    ],
    note: "Isolated hip-abductor strengthening.",
  },
  "weight-shift": {
    slug: "weight-shift",
    linkedAssessments: ["functional_reach", "four_stage_balance", "tug"],
    priority: 1,
    deficitRules: [
      { module: "functional_reach", when: "classification", equals: ["moderate", "high", "very_high"] },
      { module: "four_stage_balance", when: "classification", equals: ["borderline", "positive_screen", "fail"] },
      { module: "tug", when: "classification", equals: ["mild_fall_risk", "elevated_fall_risk", "significant_fall_risk"] },
    ],
    note: "Limits-of-stability training for fall-risk / balance deficit.",
  },
  "bridge": {
    slug: "bridge",
    linkedAssessments: ["sit_to_stand", "sppb"],
    priority: 2,
    deficitRules: [
      { module: "sit_to_stand", when: "classification", equals: ["borderline", "weakness"] },
      { module: "sppb", when: "classification", equals: ["severe", "moderate"] },
    ],
    note: "Glute / posterior-chain strengthening.",
  },
  "marching": {
    slug: "marching",
    linkedAssessments: ["gait", "tug"],
    priority: 1,
    deficitRules: [
      { module: "gait", when: "gait_norm", metric: "cadence", direction: "below_low", weight: 1.3 },
      { module: "gait", when: "gait_norm", metric: "symmetry", direction: "below_low", weight: 1.3 },
      { module: "gait", when: "observation_tone", tone: "bad" },
      { module: "tug", when: "classification", equals: ["mild_fall_risk", "elevated_fall_risk", "significant_fall_risk"] },
    ],
    note: "Cadence + symmetry retraining for gait deficit.",
  },
  "lateral-step": {
    slug: "lateral-step",
    linkedAssessments: ["trendelenburg", "single_leg_squat"],
    priority: 2,
    deficitRules: [
      { module: "trendelenburg", when: "classification", equals: ["positive", "compensated"] },
      { module: "single_leg_squat", when: "classification", equals: ["borderline", "valgus"] },
    ],
    note: "Hip-abductor endurance in a functional step pattern.",
  },

  // ── BACK ─────────────────────────────────────────────────────────
  "posture-hold": {
    slug: "posture-hold",
    linkedAssessments: ["biomech"],
    priority: 2,
    deficitRules: [
      {
        module: "biomech", when: "compensation",
        body_part: "neck", types: ["trunk_lean", "shoulder_hike"], min_severity: "medium",
      },
      {
        module: "biomech", when: "target_shortfall",
        body_part: "neck", movement: ["flexion_extension", "lateral_flexion", "rotation"],
        shortfall_deg: 5,
      },
    ],
    note: "Forward-head / cervical posture reset.",
  },
  "back-extension": {
    slug: "back-extension",
    linkedAssessments: ["slr", "modified_thomas"],
    priority: 3,
    deficitRules: [
      { module: "slr", when: "classification", equals: ["positive"] },
      { module: "modified_thomas", when: "classification", equals: ["significant"], path: "hip_classification" },
    ],
    note: "Lumbar extensor endurance (low priority — trend proxy).",
  },
  "side-bend": {
    slug: "side-bend",
    linkedAssessments: ["biomech"],
    priority: 3,
    deficitRules: [
      {
        module: "biomech", when: "target_shortfall",
        body_part: "neck", movement: "lateral_flexion",
        shortfall_deg: 5,
      },
    ],
    note: "Lateral trunk-flexion ROM (uses cervical lateral as proxy).",
  },
  "bird-dog": {
    slug: "bird-dog",
    linkedAssessments: ["sppb", "single_leg_stance"],
    priority: 2,
    deficitRules: [
      { module: "sppb", when: "classification", equals: ["severe", "moderate"] },
      { module: "single_leg_stance", when: "classification", equals: ["fail"] },
    ],
    note: "Contralateral core stability / posterior-chain coordination.",
  },
  "hip-hinge": {
    slug: "hip-hinge",
    linkedAssessments: ["slr", "modified_thomas"],
    priority: 2,
    deficitRules: [
      { module: "slr", when: "classification", equals: ["positive"], weight: 1.2 },
      { module: "modified_thomas", when: "classification", equals: ["significant"], path: "hip_classification" },
      { module: "modified_thomas", when: "classification", equals: ["tight"], path: "knee_classification" },
    ],
    note: "Hip-hinge pattern for SLR positive / hamstring tightness.",
  },
  "cat-cow": {
    slug: "cat-cow",
    linkedAssessments: ["biomech"],
    priority: 3,
    deficitRules: [
      { module: "biomech", when: "target_shortfall", body_part: "neck", movement: "flexion_extension", shortfall_deg: 5 },
    ],
    note: "Spinal mobility flow (low priority — general mobility).",
  },

  // ── SHOULDER ─────────────────────────────────────────────────────
  "shoulder-raise": {
    slug: "shoulder-raise",
    linkedAssessments: ["biomech"],
    priority: 1,
    deficitRules: [
      {
        module: "biomech", when: "target_shortfall",
        body_part: "shoulder", movement: ["abduction_adduction", "abduction"],
        shortfall_deg: 10,
      },
      {
        module: "biomech", when: "compensation",
        body_part: "shoulder", types: ["trunk_lean", "shoulder_elevation"], min_severity: "medium",
      },
    ],
    note: "Shoulder abduction ROM + elevation-compensation retraining.",
  },
  "wall-clock": {
    slug: "wall-clock",
    linkedAssessments: ["biomech"],
    priority: 2,
    deficitRules: [
      {
        module: "biomech", when: "target_shortfall",
        body_part: "shoulder", movement: ["flexion_extension", "abduction_adduction"],
        shortfall_deg: 10,
      },
    ],
    note: "Multi-plane reach ROM.",
  },
  "pendulum": {
    slug: "pendulum",
    linkedAssessments: ["biomech"],
    priority: 1,
    deficitRules: [
      {
        module: "biomech", when: "target_shortfall",
        body_part: "shoulder", movement: ["rotation", "flexion_extension"],
        shortfall_deg: 15, weight: 1.3,
      },
      {
        module: "biomech", when: "compensation",
        body_part: "shoulder", types: ["shoulder_elevation", "trunk_lean"], min_severity: "high",
      },
    ],
    note: "Codman-style passive-motion for early post-op / adhesive capsulitis.",
  },
  "wall-slide": {
    slug: "wall-slide",
    linkedAssessments: ["biomech"],
    priority: 1,
    deficitRules: [
      {
        module: "biomech", when: "target_shortfall",
        body_part: "shoulder", movement: ["flexion_extension", "flexion"],
        shortfall_deg: 15,
      },
      {
        module: "biomech", when: "compensation",
        body_part: "shoulder", types: ["shoulder_elevation"], min_severity: "medium",
      },
    ],
    note: "Overhead flexion ROM + scapulohumeral rhythm.",
  },
  "external-rotation": {
    slug: "external-rotation",
    linkedAssessments: ["biomech"],
    priority: 2,
    deficitRules: [
      {
        module: "biomech", when: "target_shortfall",
        body_part: "shoulder", movement: ["rotation", "external_rotation"],
        shortfall_deg: 10,
      },
    ],
    note: "External-rotation trend (coarser than clinical goniometer).",
  },
  "scapular-set": {
    slug: "scapular-set",
    linkedAssessments: ["biomech"],
    priority: 3,
    deficitRules: [
      {
        module: "biomech", when: "compensation",
        body_part: "shoulder", types: ["shoulder_elevation", "trunk_lean"], min_severity: "medium",
      },
    ],
    note: "Scapular retraction (coarse proxy — trend-only).",
  },
};

/** Convenience list of slugs the recommender iterates through. */
export const INDICATED_SLUGS = Object.keys(EXERCISE_INDICATIONS);

/** Look up an indication entry by slug. Returns null when unknown. */
export function indicationOf(slug) {
  return EXERCISE_INDICATIONS[slug] ?? null;
}
