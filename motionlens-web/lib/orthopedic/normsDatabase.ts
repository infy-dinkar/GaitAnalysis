// Centralised norm-database lookups for orthopedic / geriatric tests.
//
// Per PDF Appendix E ("Norm-database: a single source of truth mapping
// (test, age-band, sex) → reference values"), every test that grades
// the patient against a population norm should lookup its threshold
// here rather than embedding magic numbers in capture / report code.
//
// Today: CDC STEADI 30-second-chair-stand norms.
// Future tests (e.g. 6-minute walk distance, grip strength) should
// add their own helpers + tables here.
//
// Public functions return:
//   - the threshold value
//   - a `comparable` boolean (false when patient demographics are
//     missing, so the report can flag that the comparison is generic)
//   - the band label that was matched (used for the report subtitle)

export type Sex = "male" | "female";

// ─── 30-Second Chair Stand (CDC STEADI / Rikli & Jones 1999) ────
//
// "Below average" thresholds: a patient is considered AT RISK when
// their rep count in 30 s falls BELOW these values. Source: CDC
// STEADI norm tables, copied verbatim from PDF Test C3 spec.
//
// Bands extend below 60 (use the 60-64 row) and above 94 (use the
// 90-94 row) so the lookup never returns null for an out-of-range
// patient — instead it flags `comparable: false` with a "best-effort"
// match. Generic fallback (no demographics on file): men 11, women
// 10 — also flagged non-comparable.
interface ChairStand30sBand {
  readonly ageMin: number;
  readonly ageMax: number;
  readonly male: number;
  readonly female: number;
}

const CHAIR_STAND_30S_BANDS: readonly ChairStand30sBand[] = [
  { ageMin: 60, ageMax: 64, male: 14, female: 12 },
  { ageMin: 65, ageMax: 69, male: 12, female: 11 },
  { ageMin: 70, ageMax: 74, male: 12, female: 10 },
  { ageMin: 75, ageMax: 79, male: 11, female: 10 },
  { ageMin: 80, ageMax: 84, male: 10, female: 9  },
  { ageMin: 85, ageMax: 89, male: 8,  female: 8  },
  { ageMin: 90, ageMax: 94, male: 7,  female: 4  },
];

const CHAIR_STAND_30S_GENERIC_MALE = 11;
const CHAIR_STAND_30S_GENERIC_FEMALE = 10;

export interface ChairStand30sNorm {
  /** Below-this-many-reps = at risk per CDC STEADI. */
  belowAverageThreshold: number;
  /** True when both age + sex were available AND landed inside the
   *  60-94 range covered by the published table. */
  comparable: boolean;
  /** Human-readable description of the matched band, for the report. */
  bandLabel: string;
}

export function getChairStand30sNorm(
  age: number | null | undefined,
  sex: Sex | "other" | null | undefined,
): ChairStand30sNorm {
  // Generic fallback when demographics are missing or non-binary.
  // CDC STEADI is a binary-sex table; "other" patients fall back too.
  if (age === null || age === undefined || sex === null || sex === undefined || sex === "other") {
    return {
      belowAverageThreshold: sex === "female" ? CHAIR_STAND_30S_GENERIC_FEMALE : CHAIR_STAND_30S_GENERIC_MALE,
      comparable: false,
      bandLabel: "generic threshold (patient demographics incomplete)",
    };
  }

  // Out-of-range — clamp to the nearest band but flag non-comparable
  // so the report can call this out.
  if (age < 60) {
    const band = CHAIR_STAND_30S_BANDS[0];
    return {
      belowAverageThreshold: band[sex],
      comparable: false,
      bandLabel: `closest CDC band (60–64) — patient age ${age} is below the published range`,
    };
  }
  if (age > 94) {
    const band = CHAIR_STAND_30S_BANDS[CHAIR_STAND_30S_BANDS.length - 1];
    return {
      belowAverageThreshold: band[sex],
      comparable: false,
      bandLabel: `closest CDC band (90–94) — patient age ${age} is above the published range`,
    };
  }

  for (const band of CHAIR_STAND_30S_BANDS) {
    if (age >= band.ageMin && age <= band.ageMax) {
      return {
        belowAverageThreshold: band[sex],
        comparable: true,
        bandLabel: `CDC STEADI ${band.ageMin}–${band.ageMax} ${sex === "male" ? "men" : "women"}`,
      };
    }
  }

  // Defensive — should be unreachable.
  return {
    belowAverageThreshold: sex === "female" ? CHAIR_STAND_30S_GENERIC_FEMALE : CHAIR_STAND_30S_GENERIC_MALE,
    comparable: false,
    bandLabel: "generic threshold (band lookup failed)",
  };
}

// ─── Single-Leg Stance hold-time thresholds (PDF Test C5) ───────
//
// Minimum hold-time in seconds for a "normal" classification. Below
// the threshold = positive screen for balance impairment / fall
// risk. Eyes-closed is roughly half eyes-open per PDF guidance.
//
// Bands extend below 60 (use the <60 row) and clamp at 70+ for the
// oldest patients, with `comparable: false` on out-of-band age or
// missing demographics.
interface SingleLegStanceBand {
  readonly ageMin: number;
  readonly ageMax: number;
  readonly eyesOpenSec: number;
  readonly eyesClosedSec: number;
}

const SINGLE_LEG_STANCE_BANDS: readonly SingleLegStanceBand[] = [
  { ageMin: 0,   ageMax: 59,  eyesOpenSec: 10, eyesClosedSec: 5   },
  { ageMin: 60,  ageMax: 69,  eyesOpenSec: 7,  eyesClosedSec: 3.5 },
  { ageMin: 70,  ageMax: 200, eyesOpenSec: 5,  eyesClosedSec: 2.5 },
];

export interface SingleLegStanceNorm {
  /** Hold-this-many-seconds-or-more = passing per the matched band. */
  passThresholdSec: number;
  /** True when the patient's age was provided (the table covers all
   *  ages via clamping, so the only failure mode is missing age). */
  comparable: boolean;
  /** Human-readable description of the matched band. */
  bandLabel: string;
}

export function getSingleLegStanceNorm(
  age: number | null | undefined,
  eyesClosed: boolean,
): SingleLegStanceNorm {
  // Missing age — fall back to the strictest band (age <60) so we
  // err toward flagging rather than missing fall risk. Flagged as
  // non-comparable so the report calls it out.
  if (age === null || age === undefined) {
    const band = SINGLE_LEG_STANCE_BANDS[0];
    return {
      passThresholdSec: eyesClosed ? band.eyesClosedSec : band.eyesOpenSec,
      comparable: false,
      bandLabel: "strictest threshold (patient age not available)",
    };
  }

  for (const band of SINGLE_LEG_STANCE_BANDS) {
    if (age >= band.ageMin && age <= band.ageMax) {
      const eo = band.eyesOpenSec;
      const ec = band.eyesClosedSec;
      const cond = eyesClosed ? "eyes-closed" : "eyes-open";
      const ageLabel =
        band.ageMin === 0     ? "under 60" :
        band.ageMax === 200   ? "70+" :
        `${band.ageMin}–${band.ageMax}`;
      return {
        passThresholdSec: eyesClosed ? ec : eo,
        comparable: true,
        bandLabel: `age ${ageLabel}, ${cond}`,
      };
    }
  }

  // Defensive — clamp + flag non-comparable.
  const last = SINGLE_LEG_STANCE_BANDS[SINGLE_LEG_STANCE_BANDS.length - 1];
  return {
    passThresholdSec: eyesClosed ? last.eyesClosedSec : last.eyesOpenSec,
    comparable: false,
    bandLabel: "out-of-range fallback",
  };
}
