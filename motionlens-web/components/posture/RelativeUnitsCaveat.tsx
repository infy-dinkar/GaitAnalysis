// Spec (Appendix C): without an absolute calibration step, distance
// outputs must be labelled "relative units only — distance not
// calibrated". Posture reports horizontal shifts as a percentage of
// body height, so this caveat is rendered below any %-distance table.

export function RelativeUnitsCaveat({ className = "" }: { className?: string }) {
  return (
    <p className={`mt-2 text-xs italic text-gray-500 ${className}`}>
      ⓘ Distances are reported as % of body height — not calibrated to
      absolute units (cm). Suitable for trend tracking; for clinical-grade
      measurements, use a calibrated reference.
    </p>
  );
}
