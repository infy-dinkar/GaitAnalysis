"use client";
// Rehab session footer — thin wrapper around SaveToPatientButton.
// Renders nothing outside the doctor flow so the public /rehab
// catalogue view never shows a stray save UI.

import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";

/**
 * @typedef {object} RehabSessionFooterProps
 * @property {() => (object | null)} buildPayload
 *   The page's buildRehabPayload — called on Save click to produce
 *   the report payload.
 * @property {string} [label]
 * @property {boolean} [compact]  Slim variant for space-constrained
 *   shells like the live-mode sidebar. Renders just the button.
 */

/**
 * @param {RehabSessionFooterProps} props
 */
export function RehabSessionFooter({
  buildPayload,
  label = "Save rehab session",
  compact = false,
}) {
  const { isDoctorFlow } = usePatientContext();
  if (!isDoctorFlow) return null;
  return (
    <div className="no-pdf">
      <SaveToPatientButton
        buildPayload={buildPayload}
        label={label}
        compact={compact}
      />
    </div>
  );
}
