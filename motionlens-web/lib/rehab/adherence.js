// Adherence helpers — Phase 1 (completed side only).
//
// Prescriptions (the "goal" half) don't have a persistent store yet.
// This file only handles the DERIVED-COMPLETED side: given the
// patient's saved rehab sessions, compute what they actually did per
// day and per exercise, in local calendar time.

/**
 * @typedef {object} ReportSummary
 * @property {string} id
 * @property {string} module
 * @property {string | null} movement
 * @property {string} created_at
 */

/**
 * @typedef {object} DailyCompletion
 * @property {string} date        YYYY-MM-DD local
 * @property {number} total       total rehab sessions that day
 * @property {Record<string, number>} bySlug  count per exercise slug
 */

/**
 * @typedef {object} WeeklyCompletion
 * @property {string} weekStart   YYYY-MM-DD (Monday, local)
 * @property {number} total
 * @property {Record<string, number>} bySlug
 * @property {number} days        distinct active days in the week (1..7)
 */

function toLocalIsoDate(d) {
  const dt = d instanceof Date ? d : new Date(d);
  if (Number.isNaN(dt.getTime())) return "";
  const y = dt.getFullYear();
  const m = String(dt.getMonth() + 1).padStart(2, "0");
  const day = String(dt.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/**
 * Monday-anchored week bucket for a local date. Uses local weekday so
 * "this week" matches the patient's calendar, not UTC.
 * @param {string} iso YYYY-MM-DD
 * @returns {string} weekStart YYYY-MM-DD
 */
function weekStartOf(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  const dt = new Date(y, m - 1, d, 12, 0, 0);
  // getDay: 0 = Sunday, 1 = Monday. Shift so Monday = 0.
  const dow = (dt.getDay() + 6) % 7;
  dt.setDate(dt.getDate() - dow);
  return toLocalIsoDate(dt);
}

/**
 * Bucket the patient's rehab sessions by local calendar day, plus a
 * per-exercise slug count within each day.
 * @param {ReportSummary[]} reports
 * @returns {DailyCompletion[]}
 */
export function computeCompletionByDay(reports) {
  /** @type {Map<string, DailyCompletion>} */
  const byDate = new Map();
  for (const r of reports ?? []) {
    if (r.module !== "rehab") continue;
    const date = toLocalIsoDate(r.created_at);
    if (!date) continue;
    const slug = r.movement || "unknown";
    const bucket = byDate.get(date) ?? { date, total: 0, bySlug: {} };
    bucket.total += 1;
    bucket.bySlug[slug] = (bucket.bySlug[slug] ?? 0) + 1;
    byDate.set(date, bucket);
  }
  return Array.from(byDate.values()).sort((a, b) => a.date.localeCompare(b.date));
}

/**
 * Roll daily completion up into Monday-anchored weeks. Sorted ascending.
 * @param {ReportSummary[]} reports
 * @returns {WeeklyCompletion[]}
 */
export function computeCompletionByWeek(reports) {
  const daily = computeCompletionByDay(reports);
  /** @type {Map<string, WeeklyCompletion>} */
  const byWeek = new Map();
  for (const d of daily) {
    const ws = weekStartOf(d.date);
    const wk = byWeek.get(ws) ?? { weekStart: ws, total: 0, bySlug: {}, days: 0 };
    wk.total += d.total;
    wk.days += 1;
    for (const [slug, count] of Object.entries(d.bySlug)) {
      wk.bySlug[slug] = (wk.bySlug[slug] ?? 0) + count;
    }
    byWeek.set(ws, wk);
  }
  return Array.from(byWeek.values()).sort((a, b) => a.weekStart.localeCompare(b.weekStart));
}

/**
 * Sessions completed in the last N days (inclusive of today, local time).
 * @param {ReportSummary[]} reports
 * @param {number} days
 * @param {Date} [now]
 * @returns {number}
 */
export function completedInLastDays(reports, days, now) {
  const nowIso = toLocalIsoDate(now ?? new Date());
  if (!nowIso) return 0;
  const [y, m, d] = nowIso.split("-").map(Number);
  const cutoff = new Date(y, m - 1, d - (days - 1), 12, 0, 0);
  let count = 0;
  for (const r of reports ?? []) {
    if (r.module !== "rehab") continue;
    const t = new Date(r.created_at).getTime();
    if (!Number.isFinite(t)) continue;
    if (t >= cutoff.getTime()) count += 1;
  }
  return count;
}
