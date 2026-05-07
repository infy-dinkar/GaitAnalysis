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
