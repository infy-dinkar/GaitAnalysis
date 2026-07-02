// Rehab day-streak helper.
//
// Pure client-side computation from a list of saved-session dates.
// Consumers pass ReportSummary.created_at strings (or Date objects);
// this file groups them by LOCAL calendar day, computes the current
// and best consecutive-day streaks, and returns the sorted unique
// active-date list.
//
// Design notes:
//   • Uses LOCAL date parts (getFullYear/getMonth/getDate) rather
//     than UTC — "today" must match the patient's real calendar day.
//   • Multiple sessions on the same local day count as one active
//     day (dedupe via Set).
//   • Current streak stays alive when the patient did yesterday but
//     hasn't done today yet — it only breaks when a FULL calendar
//     day is missed. That matches how Kemtai / ViFive / Duolingo
//     treat streaks.
//
// Pure — no side effects, no globals. `today` is injectable for
// testability; production callers omit it and get `new Date()`.

export interface StreakResult {
  /** Consecutive local days ending at (or one before) today. */
  currentStreak: number;
  /** Longest consecutive-day run ever recorded. */
  bestStreak: number;
  /** Most recent active day, or null when no sessions. YYYY-MM-DD. */
  lastActiveDate: string | null;
  /** Unique local-date strings (YYYY-MM-DD) sorted ascending. */
  activeDates: string[];
}

/** Convert a Date to a local YYYY-MM-DD string. Uses local timezone
 *  (not UTC) so the day matches the patient's calendar. */
function toLocalIsoDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Signed calendar-day distance from `a` to `b`. Both YYYY-MM-DD.
 *  Positive when `b` is later. Works across month/year boundaries. */
function daysBetween(a: string, b: string): number {
  const [ay, am, ad] = a.split("-").map(Number);
  const [by, bm, bd] = b.split("-").map(Number);
  // Local-noon anchors avoid DST edge cases where midnight-to-midnight
  // isn't exactly 24 hours; a 12h offset absorbs the ±1h shift.
  const A = new Date(ay, am - 1, ad, 12, 0, 0).getTime();
  const B = new Date(by, bm - 1, bd, 12, 0, 0).getTime();
  return Math.round((B - A) / (24 * 60 * 60 * 1000));
}

/** Compute the current + best streak from a list of session dates.
 *
 *  @param inputs   Raw Date objects OR ISO strings (e.g.
 *                  ReportSummary.created_at). Invalid entries
 *                  are silently skipped.
 *  @param today    Optional "today" reference for testability.
 *                  Defaults to `new Date()` at call time.
 */
export function computeStreak(
  inputs: Array<string | Date>,
  today?: Date,
): StreakResult {
  const todayIso = toLocalIsoDate(today ?? new Date());

  // Normalise → local YYYY-MM-DD → dedupe → sort ascending.
  const set = new Set<string>();
  for (const raw of inputs) {
    const d = typeof raw === "string" ? new Date(raw) : raw;
    if (!(d instanceof Date) || Number.isNaN(d.getTime())) continue;
    set.add(toLocalIsoDate(d));
  }
  const activeDates = Array.from(set).sort();

  if (activeDates.length === 0) {
    return {
      currentStreak: 0,
      bestStreak: 0,
      lastActiveDate: null,
      activeDates: [],
    };
  }

  // Best streak — one pass counting consecutive-day runs.
  let bestStreak = 1;
  let runLength = 1;
  for (let i = 1; i < activeDates.length; i++) {
    if (daysBetween(activeDates[i - 1], activeDates[i]) === 1) {
      runLength++;
      if (runLength > bestStreak) bestStreak = runLength;
    } else {
      runLength = 1;
    }
  }

  // Current streak — walk backwards from the most recent active day.
  // Alive when the last session was today OR yesterday; broken
  // otherwise (a full day missed).
  const lastActiveDate = activeDates[activeDates.length - 1];
  const daysSinceLast = daysBetween(lastActiveDate, todayIso);

  let currentStreak = 0;
  if (daysSinceLast === 0 || daysSinceLast === 1) {
    currentStreak = 1;
    for (let i = activeDates.length - 2; i >= 0; i--) {
      if (daysBetween(activeDates[i], activeDates[i + 1]) === 1) {
        currentStreak++;
      } else {
        break;
      }
    }
  }

  return {
    currentStreak,
    bestStreak,
    lastActiveDate,
    activeDates,
  };
}
