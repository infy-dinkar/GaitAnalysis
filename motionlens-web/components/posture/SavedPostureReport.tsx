"use client";
// Read-only posture report for the saved-report viewer.
// Same findings tables as PostureReport, but rebuilt from saved metrics
// (no annotated images — those were never persisted to the DB).

import { useMemo } from "react";
import {
  buildFrontFindings,
  buildSideFindings,
  type PostureFinding,
  type FrontMeasurements,
  type SideMeasurements,
} from "@/lib/posture/measurements";

interface Props {
  front: FrontMeasurements | null;
  side: SideMeasurements | null;
  /** Pre-built findings from observations (preferred — matches what the
   *  doctor saw at save time). Falls back to recomputing from metrics. */
  frontFindings?: PostureFinding[] | null;
  sideFindings?: PostureFinding[] | null;
}

export function SavedPostureReport({
  front,
  side,
  frontFindings,
  sideFindings,
}: Props) {
  const fFindings = useMemo<PostureFinding[]>(() => {
    if (frontFindings && frontFindings.length > 0) return frontFindings;
    return front ? buildFrontFindings(front) : [];
  }, [front, frontFindings]);

  const sFindings = useMemo<PostureFinding[]>(() => {
    if (sideFindings && sideFindings.length > 0) return sideFindings;
    return side ? buildSideFindings(side) : [];
  }, [side, sideFindings]);

  const all = [...fFindings, ...sFindings];
  const hasNotable = all.some((f) => f.severity === "notable");
  const hasMild = all.some((f) => f.severity === "mild");

  const summary = hasNotable
    ? "One or more notable postural deviations were detected. Review the findings below and consider further evaluation."
    : hasMild
      ? "Mild postural asymmetries detected. These may be benign — review and re-test if persistent."
      : "Posture appears well aligned across the assessed views.";

  return (
    <div className="space-y-10">
      <div className="text-center">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Posture Assessment
        </h2>
        <p className="mt-3 text-sm text-muted">{summary}</p>
      </div>

      {fFindings.length > 0 && (
        <FindingsTable title="Front view findings" findings={fFindings} />
      )}
      {sFindings.length > 0 && (
        <FindingsTable title="Side view findings" findings={sFindings} />
      )}

      {fFindings.length === 0 && sFindings.length === 0 && (
        <p className="rounded-card border border-border bg-surface p-5 text-center text-sm text-muted">
          No measurements were available for this saved report.
        </p>
      )}

      <p className="border-t border-border/60 pt-4 text-center text-[11px] leading-relaxed text-subtle/80">
        Postural measurements are estimated from 2D pose detection and are
        intended for screening only. For a clinical-grade postural assessment,
        please consult a qualified practitioner.
      </p>
    </div>
  );
}

const SEVERITY_STYLES: Record<PostureFinding["severity"], string> = {
  ok:      "bg-accent/10 text-accent",
  mild:    "bg-warning/10 text-warning",
  notable: "bg-error/10 text-error",
};

const SEVERITY_LABEL: Record<PostureFinding["severity"], string> = {
  ok: "OK",
  mild: "Mild",
  notable: "Notable",
};

function FindingsTable({
  title,
  findings,
}: {
  title: string;
  findings: PostureFinding[];
}) {
  return (
    <section>
      <h3 className="text-base font-semibold tracking-tight">{title}</h3>
      <div className="mt-3 overflow-hidden rounded-card border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border bg-elevated text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-5 py-3 font-medium">Measurement</th>
              <th className="px-5 py-3 text-center font-medium">Value</th>
              <th className="px-5 py-3 text-center font-medium">Status</th>
              <th className="px-5 py-3 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody>
            {findings.map((f, i) => (
              <tr key={i} className="border-b border-border/50 last:border-b-0">
                <td className="px-5 py-3 text-foreground">{f.label}</td>
                <td className="px-5 py-3 text-center tabular text-foreground">
                  {f.value}
                </td>
                <td className="px-5 py-3 text-center">
                  <span
                    className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-medium ${SEVERITY_STYLES[f.severity]}`}
                  >
                    {SEVERITY_LABEL[f.severity]}
                  </span>
                </td>
                <td className="px-5 py-3 text-muted">{f.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
