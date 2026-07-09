"use client";
// useRecommendations — React hook that fetches the prescribed set for
// a patient. Reads getPrescribedSet — that function prefers a
// doctor-saved prescription and falls back to the auto recommender.
//
// The hook also exposes save() and reset() so the launcher's edit UI
// can persist a doctor-authored slug list and revert to auto without
// touching the API client directly.

import { useCallback, useEffect, useState } from "react";
import { getPrescribedSet } from "@/lib/rehab/recommendation";
import { savePrescription, clearPrescription } from "@/lib/rehab/prescriptions";

/**
 * @typedef {import("@/lib/rehab/recommendation").Recommendation} Recommendation
 */

/**
 * @typedef {object} UseRecommendationsResult
 * @property {"loading"|"ready"|"empty"|"error"} status
 * @property {Set<string>} slugs
 * @property {"auto"|"doctor"} source
 * @property {Recommendation[]} recommended
 * @property {number} assessmentsUsed
 * @property {number} deficitsFound
 * @property {Map<string, Recommendation>} bySlug
 * @property {(slugs: string[]) => Promise<void>} save
 * @property {() => Promise<void>} reset
 * @property {boolean} saving
 */

/**
 * @param {string | null | undefined} patientId
 * @returns {UseRecommendationsResult}
 */
export function useRecommendations(patientId) {
  const [state, setState] = useState({
    status: patientId ? "loading" : "empty",
    slugs: new Set(),
    source: "auto",
    recommended: [],
    assessmentsUsed: 0,
    deficitsFound: 0,
    bySlug: new Map(),
  });
  const [saving, setSaving] = useState(false);
  // Bump this to force the effect below to re-run after a save/reset
  // so the UI reflects the new prescribed set without a page reload.
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    if (!patientId) {
      setState({
        status: "empty",
        slugs: new Set(),
        source: "auto",
        recommended: [],
        assessmentsUsed: 0,
        deficitsFound: 0,
        bySlug: new Map(),
      });
      return;
    }
    let cancelled = false;
    setState((prev) => ({ ...prev, status: "loading" }));
    getPrescribedSet(patientId)
      .then((res) => {
        if (cancelled) return;
        const bySlug = new Map();
        for (const r of res.recommended) bySlug.set(r.slug, r);
        setState({
          status: res.recommended.length === 0 ? "empty" : "ready",
          slugs: res.slugs,
          source: res.source,
          recommended: res.recommended,
          assessmentsUsed: res.assessmentsUsed,
          deficitsFound: res.deficitsFound,
          bySlug,
        });
      })
      .catch(() => {
        if (cancelled) return;
        setState({
          status: "error",
          slugs: new Set(),
          source: "auto",
          recommended: [],
          assessmentsUsed: 0,
          deficitsFound: 0,
          bySlug: new Map(),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [patientId, refreshTick]);

  const save = useCallback(
    async (slugs) => {
      if (!patientId) return;
      setSaving(true);
      try {
        await savePrescription(patientId, slugs);
      } finally {
        setSaving(false);
        setRefreshTick((t) => t + 1);
      }
    },
    [patientId],
  );

  const reset = useCallback(async () => {
    if (!patientId) return;
    setSaving(true);
    try {
      await clearPrescription(patientId);
    } finally {
      setSaving(false);
      setRefreshTick((t) => t + 1);
    }
  }, [patientId]);

  return { ...state, save, reset, saving };
}
