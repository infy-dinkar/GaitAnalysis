"use client";
// useProgressionLevel — Section 8 auto-progression hook.
//
// Reads the patient's saved rehab sessions filtered to the given
// exercise slug and derives the CURRENT level:
//   • last 3 sessions at the current level all ≥ 90 % clean → +1
//   • last 2 sessions at the current level all < 60 % clean  → −1
//   • else                                                    → hold
//
// Optional override + lock flags (F3F design placeholder — no
// prescription store yet, both default to undefined):
//   • level_override:      forces the level, ignoring history
//   • progression_locked:  disables +1/-1; hint returns "locked"
//
// Additive: no backend changes, no schema changes. Each save
// payload writes metrics.level_index so future runs can derive
// the current level from history alone.

import { useEffect, useMemo, useState } from "react";
import { listPatientReports, getReport } from "@/lib/reports";
import { computeCleanPct } from "@/lib/rehab/cleanPct";
import { DEFAULT_LEVEL_INDEX, defaultLevelConfig } from "@/lib/rehab/progressionLadders";

const ADVANCE_THRESHOLD = 90;
const REGRESS_THRESHOLD = 60;
const ADVANCE_STREAK = 3;
const REGRESS_STREAK = 2;
const HISTORY_CAP = 20;

/**
 * @typedef {object} ProgressionResult
 * @property {number} level         current recommended level index
 * @property {any} config           ladder[level] — spread into page's _CONFIG
 * @property {"advance"|"hold"|"regress"|"locked"|"insufficient"} reason
 * @property {string} hint          human-readable one-liner for the UI chip
 * @property {boolean} loading
 * @property {boolean} lockedByProtocol
 */

/**
 * @param {string | null | undefined} patientId
 * @param {string} slug
 * @param {any[]} ladder
 * @param {{ level_override?: number, progression_locked?: boolean }} [opts]
 * @returns {ProgressionResult}
 */
export function useProgressionLevel(patientId, slug, ladder, opts) {
  const [loading, setLoading] = useState(Boolean(patientId));
  const [sessions, setSessions] = useState(/** @type {any[]} */ ([]));

  useEffect(() => {
    let cancelled = false;
    if (!patientId) {
      setLoading(false);
      return;
    }
    setLoading(true);
    listPatientReports(patientId)
      .then(async (res) => {
        if (cancelled) return;
        const rehab = res.data.filter(
          (r) => r.module === "rehab" && r.movement === slug,
        );
        const capped = [...rehab]
          .sort((a, b) => b.created_at.localeCompare(a.created_at))
          .slice(0, HISTORY_CAP);
        const fulls = await Promise.all(
          capped.map((s) => getReport(s.id).catch(() => null)),
        );
        if (cancelled) return;
        setSessions(fulls.filter(Boolean));
      })
      .catch(() => {
        if (!cancelled) setSessions([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [patientId, slug]);

  return useMemo(() => {
    const lockedByProtocol = Boolean(opts?.progression_locked);
    const override = typeof opts?.level_override === "number"
      ? clampLevel(opts.level_override, ladder)
      : null;
    if (override !== null) {
      return {
        level: override,
        config: ladder[override] ?? defaultLevelConfig(ladder),
        reason: "hold",
        hint: `Clinician set level ${override + 1}.`,
        loading,
        lockedByProtocol,
      };
    }
    // Newest first for the streak checks below.
    const ordered = sessions
      .map((s) => ({
        cleanPct: computeCleanPct(
          pickMechanicId(s),
          s.metrics,
        ),
        levelIndex: pickLevelIndex(s.metrics),
      }))
      .filter((x) => x.levelIndex !== null);
    const currentLevel = ordered.length > 0
      ? ordered[0].levelIndex
      : DEFAULT_LEVEL_INDEX;
    const clamped = clampLevel(currentLevel, ladder);

    if (lockedByProtocol) {
      return {
        level: clamped,
        config: ladder[clamped] ?? defaultLevelConfig(ladder),
        reason: "locked",
        hint: "Post-op protocol lock — auto-progression disabled.",
        loading,
        lockedByProtocol: true,
      };
    }

    const atCurrent = ordered.filter((x) => x.levelIndex === clamped);
    const lastAdvance = atCurrent.slice(0, ADVANCE_STREAK);
    const canAdvance =
      lastAdvance.length >= ADVANCE_STREAK
      && lastAdvance.every((x) => x.cleanPct >= ADVANCE_THRESHOLD);
    const lastRegress = atCurrent.slice(0, REGRESS_STREAK);
    const canRegress =
      lastRegress.length >= REGRESS_STREAK
      && lastRegress.every((x) => x.cleanPct < REGRESS_THRESHOLD);

    if (canAdvance) {
      const next = clampLevel(clamped + 1, ladder);
      return {
        level: next,
        config: ladder[next] ?? defaultLevelConfig(ladder),
        reason: "advance",
        hint: `${ADVANCE_STREAK} clean sessions in a row — advanced to level ${next + 1}.`,
        loading,
        lockedByProtocol: false,
      };
    }
    if (canRegress) {
      const prev = clampLevel(clamped - 1, ladder);
      return {
        level: prev,
        config: ladder[prev] ?? defaultLevelConfig(ladder),
        reason: "regress",
        hint: `${REGRESS_STREAK} rough sessions — stepped back to level ${prev + 1}.`,
        loading,
        lockedByProtocol: false,
      };
    }

    const reason = atCurrent.length === 0 ? "insufficient" : "hold";
    return {
      level: clamped,
      config: ladder[clamped] ?? defaultLevelConfig(ladder),
      reason,
      hint: reason === "insufficient"
        ? `Starting at level ${clamped + 1}.`
        : `Holding at level ${clamped + 1}.`,
      loading,
      lockedByProtocol: false,
    };
  }, [sessions, ladder, opts, loading]);
}

function clampLevel(idx, ladder) {
  if (!ladder || ladder.length === 0) return 0;
  if (typeof idx !== "number" || !Number.isFinite(idx)) return DEFAULT_LEVEL_INDEX;
  return Math.max(0, Math.min(ladder.length - 1, Math.floor(idx)));
}

function pickMechanicId(dto) {
  if (!dto?.metrics || typeof dto.metrics !== "object") return null;
  const v = dto.metrics.mechanic_id;
  return typeof v === "string" ? v : null;
}

function pickLevelIndex(metrics) {
  if (!metrics || typeof metrics !== "object") return null;
  const v = metrics.level_index;
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
