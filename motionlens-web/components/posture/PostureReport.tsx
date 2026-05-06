"use client";
// Combined posture analysis report — annotated front/side images +
// finding tables with severity color coding + summary interpretation.

import { useMemo } from "react";
import { PostureImageOverlay } from "@/components/posture/PostureImageOverlay";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import {
  buildFrontFindings,
  buildSideFindings,
  type PostureFinding,
} from "@/lib/posture/measurements";
import type { PostureAnalysisResult } from "@/lib/posture/analyzer";

interface Props {
  front: PostureAnalysisResult | null;
  side: PostureAnalysisResult | null;
}

export function PostureReport({ front, side }: Props) {
  const frontFindings = useMemo<PostureFinding[]>(
    () => (front?.front ? buildFrontFindings(front.front) : []),
    [front],
  );
  const sideFindings = useMemo<PostureFinding[]>(
    () => (side?.side ? buildSideFindings(side.side) : []),
    [side],
  );

  const hasNotable =
    [...frontFindings, ...sideFindings].some((f) => f.severity === "notable");
  const hasMild =
    [...frontFindings, ...sideFindings].some((f) => f.severity === "mild");

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

      {/* ── Annotated images ─────────────────────────────────── */}
      <div className="grid gap-6 md:grid-cols-2">
        {front && (
          <ViewBlock title="Front view" view={front} />
        )}
        {side && (
          <ViewBlock title="Side view" view={side} />
        )}
      </div>

      {/* ── Findings tables ──────────────────────────────────── */}
      {frontFindings.length > 0 && (
        <FindingsTable
          title="Front view findings"
          findings={frontFindings}
        />
      )}
      {sideFindings.length > 0 && (
        <FindingsTable
          title="Side view findings"
          findings={sideFindings}
        />
      )}

      {/* ── Unified report disclaimer ──────────────────────────── */}
      <ReportDisclaimer />
    </div>
  );
}

function ViewBlock({
  title,
  view,
}: {
  title: string;
  view: PostureAnalysisResult;
}) {
  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {title}
      </p>
      <PostureImageOverlay
        view={view.view}
        imageUrl={view.imageUrl}
        imageWidth={view.imageWidth}
        imageHeight={view.imageHeight}
        keypoints={view.keypoints}
        front={view.front}
        side={view.side}
      />
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
