"use client";
// Combined posture analysis report — annotated front/side images +
// finding tables with severity color coding + summary interpretation.

import { useMemo } from "react";
import { PostureImageOverlay } from "@/components/posture/PostureImageOverlay";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { RelativeUnitsCaveat } from "@/components/posture/RelativeUnitsCaveat";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import {
  buildFrontFindings,
  buildSideFindings,
  type PostureFinding,
} from "@/lib/posture/measurements";
import type {
  PostureAnalysisResult,
  PostureBackResult,
  PostureExplicitSideResult,
  PostureViewError,
} from "@/lib/posture/analyzer";
import { isPostureViewError } from "@/lib/posture/analyzer";
import type { PatientDTO } from "@/lib/patients";

interface Props {
  front: PostureAnalysisResult | null;
  side: PostureAnalysisResult | null;
  patient?: PatientDTO | null;
  patientName?: string | null;
  // ── Additive multi-view props (4-view expansion). Absent for
  // old callers — the existing front+side blocks render unchanged.
  back?: PostureBackResult | PostureViewError | null;
  leftSide?: PostureExplicitSideResult | PostureViewError | null;
  rightSide?: PostureExplicitSideResult | PostureViewError | null;
}

export function PostureReport({
  front,
  side,
  patient,
  patientName,
  back = null,
  leftSide = null,
  rightSide = null,
}: Props) {
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
      <PatientHeader
        patient={patient ?? null}
        fallbackName={patientName}
        subtitle="Posture Assessment · static-photo screening"
      />

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
        <div>
          <FindingsTable
            title="Side view findings"
            findings={sideFindings}
          />
          <RelativeUnitsCaveat />
        </div>
      )}

      {/* ── Additive multi-view blocks (only render when present) ── */}
      {back && <BackViewBlock back={back} />}
      {leftSide && (
        <ExplicitSideBlock
          side={leftSide}
          title="Left-side view (explicit)"
        />
      )}
      {rightSide && (
        <ExplicitSideBlock
          side={rightSide}
          title="Right-side view (explicit)"
        />
      )}

      {/* ── Unified report disclaimer ──────────────────────────── */}
      <ReportDisclaimer />
    </div>
  );
}

// ─── Back-view block (metrics + honest not_assessed list) ─────
function BackViewBlock({
  back,
}: {
  back: PostureBackResult | PostureViewError;
}) {
  if (isPostureViewError(back)) {
    return (
      <section className="rounded-card border border-warning/40 bg-warning/5 p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-warning">
          Back view
        </p>
        <p className="mt-2 text-sm text-foreground">
          Back view analysis failed: {friendlyError(back.error)}
        </p>
      </section>
    );
  }
  const m = back.back;
  const backFindings = back.findings ?? [];
  return (
    <section className="space-y-4">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          Back view
        </p>
        <h3 className="mt-1 text-base font-semibold tracking-tight">
          Frontal-plane items measured from behind
        </h3>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-3">
          <PostureImageOverlay
            view="front"
            imageUrl={back.imageUrl}
            imageWidth={back.imageWidth}
            imageHeight={back.imageHeight}
            keypoints={back.keypoints}
          />
          {back.lr_swap_applied && (
            <p className="rounded-md border border-border bg-surface px-3 py-2 text-[11px] text-muted">
              L/R keypoint swap applied — MediaPipe from behind was
              treated as screen-mirrored. See engineer note in
              posture_engine_multi_view.py.
            </p>
          )}
        </div>
        <div className="space-y-3 text-sm">
          {m && (
            <dl className="grid grid-cols-[auto_auto] gap-x-4 gap-y-1 tabular">
              <dt className="text-muted">Shoulder tilt</dt>
              <dd className="text-foreground">
                {m.shoulderTilt !== null ? `${m.shoulderTilt.toFixed(1)}°` : "—"}
              </dd>
              <dt className="text-muted">Hip tilt</dt>
              <dd className="text-foreground">
                {m.hipTilt !== null ? `${m.hipTilt.toFixed(1)}°` : "—"}
              </dd>
              <dt className="text-muted">Lateral trunk shift</dt>
              <dd className="text-foreground">
                {m.lateralTrunkShiftPct !== null
                  ? `${m.lateralTrunkShiftPct.toFixed(1)}%`
                  : "—"}
              </dd>
              <dt className="text-muted">Left knee alignment</dt>
              <dd className="text-foreground">
                {m.leftKneeAlignment !== null
                  ? `${(180 - m.leftKneeAlignment).toFixed(1)}° dev`
                  : "—"}
              </dd>
              <dt className="text-muted">Right knee alignment</dt>
              <dd className="text-foreground">
                {m.rightKneeAlignment !== null
                  ? `${(180 - m.rightKneeAlignment).toFixed(1)}° dev`
                  : "—"}
              </dd>
            </dl>
          )}
          {back.not_assessed && back.not_assessed.length > 0 && (
            <div>
              <p className="mt-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
                Not assessed from this view
              </p>
              <ul className="mt-2 space-y-2 text-[12px]">
                {back.not_assessed.map((na) => (
                  <li
                    key={na.label}
                    className="rounded-md border border-border bg-surface px-3 py-2"
                  >
                    <p className="font-medium text-foreground">{na.label}</p>
                    <p className="mt-0.5 text-muted">{na.reason}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
      {backFindings.length > 0 && (
        <FindingsTable
          title="Back view findings"
          findings={backFindings}
        />
      )}
    </section>
  );
}

// ─── Explicit-side block (left_side / right_side) ─────────────
function ExplicitSideBlock({
  side,
  title,
}: {
  side: PostureExplicitSideResult | PostureViewError;
  title: string;
}) {
  if (isPostureViewError(side)) {
    return (
      <section className="rounded-card border border-warning/40 bg-warning/5 p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-warning">
          {title}
        </p>
        <p className="mt-2 text-sm text-foreground">
          Analysis failed: {friendlyError(side.error)}
        </p>
      </section>
    );
  }
  const findings = side.findings ?? [];
  return (
    <section className="space-y-4">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          {title}
        </p>
        <h3 className="mt-1 text-base font-semibold tracking-tight">
          {side.explicit_side === "left" ? "Patient's left" : "Patient's right"} side
          (side declared, not auto-picked)
        </h3>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <PostureImageOverlay
          view="side"
          imageUrl={side.imageUrl}
          imageWidth={side.imageWidth}
          imageHeight={side.imageHeight}
          keypoints={side.keypoints}
          side={side.side}
        />
      </div>
      {findings.length > 0 && (
        <FindingsTable
          title={`${title} findings`}
          findings={findings}
        />
      )}
    </section>
  );
}

function friendlyError(code: string): string {
  if (code === "poor_visibility") {
    return "Person not clearly visible. Please retake with the full body in frame.";
  }
  if (code === "invalid_image") {
    return "Invalid image — could not decode.";
  }
  return code || "unknown error";
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
