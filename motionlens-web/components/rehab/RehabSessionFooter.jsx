"use client";
// Rehab session footer — owns:
//   1. Supervised / Unsupervised toggle (defaults to unsupervised)
//   2. The SaveToPatientButton (only renders in doctor flow)
//
// Threads the supervised flag into the caller-supplied buildPayload
// so each page's payload shape stays local while the toggle lives
// in one shared place. Additive: replaces the per-page
// <div className="no-pdf"><SaveToPatientButton ... /></div> block
// with a single component call.

import { useCallback, useState } from "react";
import { UserCheck, User } from "lucide-react";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";

/**
 * @typedef {object} SupervisedToggleProps
 * @property {boolean} supervised
 * @property {(next: boolean) => void} onChange
 */

/**
 * @param {SupervisedToggleProps} props
 */
function SupervisedToggle({ supervised, onChange }) {
  return (
    <div
      className="inline-flex items-center gap-1 rounded-full border border-border bg-surface p-1"
      role="radiogroup"
      aria-label="Session supervision"
    >
      <button
        type="button"
        onClick={() => onChange(false)}
        role="radio"
        aria-checked={!supervised}
        className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition ${
          !supervised
            ? "bg-accent/15 text-accent"
            : "text-muted hover:text-foreground"
        }`}
      >
        <User className="h-3.5 w-3.5" />
        Unsupervised
      </button>
      <button
        type="button"
        onClick={() => onChange(true)}
        role="radio"
        aria-checked={supervised}
        className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition ${
          supervised
            ? "bg-accent/15 text-accent"
            : "text-muted hover:text-foreground"
        }`}
      >
        <UserCheck className="h-3.5 w-3.5" />
        Supervised
      </button>
    </div>
  );
}

/**
 * @typedef {object} RehabSessionFooterProps
 * @property {(supervised: boolean) => (object | null)} buildPayload
 *   The page's buildRehabPayload. Receives the supervised flag so
 *   the page can merge it into metrics.supervised.
 * @property {string} [label]
 */

/**
 * Session footer — supervision toggle + save button in one row.
 * Marked no-pdf so it never leaks into a saved-report PDF export.
 *
 * @param {RehabSessionFooterProps} props
 */
export function RehabSessionFooter({
  buildPayload,
  label = "Save rehab session",
}) {
  const { isDoctorFlow } = usePatientContext();
  const [supervised, setSupervised] = useState(false);
  const wrappedBuild = useCallback(
    () => buildPayload(supervised),
    [buildPayload, supervised],
  );
  // Public flow: no patient to save to, so hide the whole footer —
  // matches the pre-F1A behavior (SaveToPatientButton returned null).
  if (!isDoctorFlow) return null;
  return (
    <div className="no-pdf space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <SupervisedToggle
          supervised={supervised}
          onChange={setSupervised}
        />
        <p className="text-xs text-muted">
          {supervised
            ? "Clinician is watching this session — flagged in the record."
            : "Patient is playing on their own — flagged as unsupervised."}
        </p>
      </div>
      <SaveToPatientButton
        buildPayload={wrappedBuild}
        label={label}
      />
    </div>
  );
}
