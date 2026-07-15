"use client";
// Read-only posture report for the saved-report viewer.
// Renders the same annotated-image overlays + findings tables the
// doctor saw at capture time. Image overlays only show when the saved
// payload includes the compressed source photos (added in a later
// schema revision); older posture reports gracefully fall back to
// findings-only rendering.

import { useMemo } from "react";
import { ReportDisclaimer } from "@/components/ui/ReportDisclaimer";
import { RelativeUnitsCaveat } from "@/components/posture/RelativeUnitsCaveat";
import { PatientHeader } from "@/components/dashboard/PatientHeader";
import { PostureImageOverlay } from "@/components/posture/PostureImageOverlay";
import {
  buildFrontFindings,
  buildSideFindings,
  type PostureFinding,
  type FrontMeasurements,
  type SideMeasurements,
} from "@/lib/posture/measurements";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { PatientDTO } from "@/lib/patients";
import type {
  BackMeasurements,
  PostureNotAssessed,
} from "@/lib/posture/analyzer";

/** Compressed source-photo blob persisted into the report's `metrics`
 *  field by PostureCapture. Width/height are the actual pixel
 *  dimensions of the (resized) image so the overlay scales correctly. */
export interface SavedPostureImage {
  data_url: string;
  width: number;
  height: number;
}

interface Props {
  front: FrontMeasurements | null;
  side: SideMeasurements | null;
  /** Pre-built findings from observations (preferred — matches what the
   *  doctor saw at save time). Falls back to recomputing from metrics. */
  frontFindings?: PostureFinding[] | null;
  sideFindings?: PostureFinding[] | null;
  /** Saved source photo + landmarks for the front view. Both must be
   *  present for the overlay to render. */
  frontImage?: SavedPostureImage | null;
  sideImage?: SavedPostureImage | null;
  frontKeypoints?: Keypoint[] | null;
  sideKeypoints?: Keypoint[] | null;
  patient?: PatientDTO | null;
  patientName?: string | null;
  // ── Additive multi-view (4-view expansion). All optional — old
  // saved reports (front+side only) render byte-identical.
  back?: BackMeasurements | null;
  backFindings?: PostureFinding[] | null;
  backImage?: SavedPostureImage | null;
  backKeypoints?: Keypoint[] | null;
  backNotAssessed?: PostureNotAssessed[] | null;
  backLrSwapApplied?: boolean | null;
  leftSide?: SideMeasurements | null;
  leftSideFindings?: PostureFinding[] | null;
  leftSideImage?: SavedPostureImage | null;
  leftSideKeypoints?: Keypoint[] | null;
  rightSide?: SideMeasurements | null;
  rightSideFindings?: PostureFinding[] | null;
  rightSideImage?: SavedPostureImage | null;
  rightSideKeypoints?: Keypoint[] | null;
}

export function SavedPostureReport({
  front,
  side,
  frontFindings,
  sideFindings,
  frontImage,
  sideImage,
  frontKeypoints,
  sideKeypoints,
  patient,
  patientName,
  back,
  backFindings,
  backImage,
  backKeypoints,
  backNotAssessed,
  backLrSwapApplied,
  leftSide,
  leftSideFindings,
  leftSideImage,
  leftSideKeypoints,
  rightSide,
  rightSideFindings,
  rightSideImage,
  rightSideKeypoints,
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

      {/* ── Annotated images ─────────────────────────────────────
          Only renders for reports saved AFTER images were added to
          the payload schema. Legacy posture reports (findings-only)
          skip this block and go straight to the findings tables. */}
      {(frontImage || sideImage) && (
        <div className="grid gap-6 md:grid-cols-2">
          {frontImage && frontKeypoints && (
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Front view
              </p>
              <PostureImageOverlay
                view="front"
                imageUrl={frontImage.data_url}
                imageWidth={frontImage.width}
                imageHeight={frontImage.height}
                keypoints={frontKeypoints}
                front={front ?? undefined}
              />
            </div>
          )}
          {sideImage && sideKeypoints && (
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
                Side view
              </p>
              <PostureImageOverlay
                view="side"
                imageUrl={sideImage.data_url}
                imageWidth={sideImage.width}
                imageHeight={sideImage.height}
                keypoints={sideKeypoints}
                side={side ?? undefined}
              />
            </div>
          )}
        </div>
      )}

      {fFindings.length > 0 && (
        <FindingsTable title="Front view findings" findings={fFindings} />
      )}
      {sFindings.length > 0 && (
        <div>
          <FindingsTable title="Side view findings" findings={sFindings} />
          <RelativeUnitsCaveat />
        </div>
      )}

      {/* ── Additive multi-view saved blocks — render only when present ── */}
      {(back || backImage || (backFindings && backFindings.length > 0)) && (
        <SavedBackBlock
          back={back ?? null}
          backFindings={backFindings ?? null}
          backImage={backImage ?? null}
          backKeypoints={backKeypoints ?? null}
          backNotAssessed={backNotAssessed ?? null}
          backLrSwapApplied={backLrSwapApplied ?? null}
        />
      )}
      {(leftSide || leftSideImage
        || (leftSideFindings && leftSideFindings.length > 0)) && (
        <SavedExplicitSideBlock
          title="Left-side view (explicit)"
          side={leftSide ?? null}
          findings={leftSideFindings ?? null}
          image={leftSideImage ?? null}
          keypoints={leftSideKeypoints ?? null}
        />
      )}
      {(rightSide || rightSideImage
        || (rightSideFindings && rightSideFindings.length > 0)) && (
        <SavedExplicitSideBlock
          title="Right-side view (explicit)"
          side={rightSide ?? null}
          findings={rightSideFindings ?? null}
          image={rightSideImage ?? null}
          keypoints={rightSideKeypoints ?? null}
        />
      )}

      {fFindings.length === 0 && sFindings.length === 0 && (
        <p className="rounded-card border border-border bg-surface p-5 text-center text-sm text-muted">
          No measurements were available for this saved report.
        </p>
      )}

      <ReportDisclaimer />
    </div>
  );
}

// ─── Saved-report back-view block ─────────────────────────────
function SavedBackBlock({
  back,
  backFindings,
  backImage,
  backKeypoints,
  backNotAssessed,
  backLrSwapApplied,
}: {
  back: BackMeasurements | null;
  backFindings: PostureFinding[] | null;
  backImage: SavedPostureImage | null;
  backKeypoints: Keypoint[] | null;
  backNotAssessed: PostureNotAssessed[] | null;
  backLrSwapApplied: boolean | null;
}) {
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
        {backImage && backKeypoints && (
          <div className="space-y-3">
            <PostureImageOverlay
              view="front"
              imageUrl={backImage.data_url}
              imageWidth={backImage.width}
              imageHeight={backImage.height}
              keypoints={backKeypoints}
            />
            {backLrSwapApplied && (
              <p className="rounded-md border border-border bg-surface px-3 py-2 text-[11px] text-muted">
                L/R keypoint swap was applied on this analysis.
              </p>
            )}
          </div>
        )}
        <div className="space-y-3 text-sm">
          {back && (
            <dl className="grid grid-cols-[auto_auto] gap-x-4 gap-y-1 tabular">
              <dt className="text-muted">Shoulder tilt</dt>
              <dd className="text-foreground">
                {back.shoulderTilt !== null ? `${back.shoulderTilt.toFixed(1)}°` : "—"}
              </dd>
              <dt className="text-muted">Hip tilt</dt>
              <dd className="text-foreground">
                {back.hipTilt !== null ? `${back.hipTilt.toFixed(1)}°` : "—"}
              </dd>
              <dt className="text-muted">Lateral trunk shift</dt>
              <dd className="text-foreground">
                {back.lateralTrunkShiftPct !== null
                  ? `${back.lateralTrunkShiftPct.toFixed(1)}%`
                  : "—"}
              </dd>
              <dt className="text-muted">Left knee alignment</dt>
              <dd className="text-foreground">
                {back.leftKneeAlignment !== null
                  ? `${(180 - back.leftKneeAlignment).toFixed(1)}° dev`
                  : "—"}
              </dd>
              <dt className="text-muted">Right knee alignment</dt>
              <dd className="text-foreground">
                {back.rightKneeAlignment !== null
                  ? `${(180 - back.rightKneeAlignment).toFixed(1)}° dev`
                  : "—"}
              </dd>
            </dl>
          )}
          {backNotAssessed && backNotAssessed.length > 0 && (
            <div>
              <p className="mt-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-subtle">
                Not assessed from this view
              </p>
              <ul className="mt-2 space-y-2 text-[12px]">
                {backNotAssessed.map((na) => (
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
      {backFindings && backFindings.length > 0 && (
        <FindingsTable
          title="Back view findings"
          findings={backFindings}
        />
      )}
    </section>
  );
}

// ─── Saved-report explicit-side block ─────────────────────────
function SavedExplicitSideBlock({
  title,
  side,
  findings,
  image,
  keypoints,
}: {
  title: string;
  side: SideMeasurements | null;
  findings: PostureFinding[] | null;
  image: SavedPostureImage | null;
  keypoints: Keypoint[] | null;
}) {
  return (
    <section className="space-y-4">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
          {title}
        </p>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        {image && keypoints && (
          <PostureImageOverlay
            view="side"
            imageUrl={image.data_url}
            imageWidth={image.width}
            imageHeight={image.height}
            keypoints={keypoints}
            side={side ?? undefined}
          />
        )}
        <div className="rounded-md border border-border bg-surface p-3 text-xs text-muted">
          Only the camera-facing (near-side) leg is analysed. The
          far-side limb is occluded and its keypoints are unreliable
          from this view.
        </div>
      </div>
      {findings && findings.length > 0 && (
        <FindingsTable
          title={`${title} findings`}
          findings={findings}
        />
      )}
    </section>
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
