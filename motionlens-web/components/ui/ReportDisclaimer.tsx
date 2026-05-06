// Shared disclaimer block rendered at the very bottom of every
// results page (gait, biomech, posture). Wording lives in
// `lib/disclaimer.ts` so the PDF generator can pull the same string.

import { REPORT_DISCLAIMER } from "@/lib/disclaimer";

export function ReportDisclaimer({ className = "" }: { className?: string }) {
  return (
    <p
      className={
        "mt-8 rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm " +
        "italic text-gray-600 " +
        className
      }
    >
      {REPORT_DISCLAIMER}
    </p>
  );
}
