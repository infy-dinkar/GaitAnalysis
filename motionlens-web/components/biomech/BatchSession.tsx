"use client";
// Batch biomechanics session — upload one video per selected
// joint+movement, see the per-joint report render as each completes,
// then save all to the patient + download a single multi-joint PDF.
//
// Architecture:
//   • Reuses analyzeBiomechVideo() from lib/biomech/uploadAnalyze.ts
//     directly (the same function ApiUploadAssessment calls). Zero
//     touch to the existing upload-mode component or the backend.
//   • Reuses <AssessmentReport> per completed item — one instance
//     per joint+movement, stacked vertically.
//   • Save All → ONE createReport() call with all per-item results
//     packed into metrics.items. The saved-report viewer detects the
//     `is_batch` sentinel and renders the same stacked layout. Patient
//     history shows a single "Biomechanics · Batch" row instead of N
//     individual rows. Zero backend schema changes — body_part is
//     Optional and movement is Optional<str max 64> on the Pydantic
//     model, so "batch" + null pass validation as-is.
//   • Download PDF → exportReportPdf() rasterises the entire stacked
//     report container into a multi-page A4 PDF.
//
// State machine: "select" → "running". The running phase keeps the
// queue mutable so the operator can upload videos in any order; as
// each item finishes its analysis the inline <AssessmentReport>
// replaces its file-picker card. The Save All + Download PDF buttons
// at the bottom enable only when every queued item is `done`.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  Loader2,
  RotateCcw,
  Save,
  Upload,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { AssessmentReport } from "@/components/biomech/AssessmentReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { usePatientContext } from "@/hooks/usePatientContext";
import { exportReportPdf } from "@/lib/pdf/exportReportPdf";
import { createReport, type ReportCreatePayload } from "@/lib/reports";
import { analyzeBiomechVideo } from "@/lib/biomech/uploadAnalyze";
import { SHOULDER_MOVEMENTS } from "@/lib/biomech/shoulder";
import { NECK_MOVEMENTS } from "@/lib/biomech/neck";
import { KNEE_MOVEMENTS } from "@/lib/biomech/knee";
import { HIP_MOVEMENTS } from "@/lib/biomech/hip";
import { ANKLE_MOVEMENTS } from "@/lib/biomech/ankle";
import type { BiomechDataDTO } from "@/lib/api";

type BodyPart = "shoulder" | "neck" | "knee" | "hip" | "ankle";

interface MovementOption {
  bodyPart: BodyPart;
  id: string;
  label: string;
  target: [number, number];
  /** Neck has no per-side concept (uses both ears + shoulder midline)
   *  — its movements show up with no side selector. */
  perSide: boolean;
  /** Reference illustration. When set, selecting this movement pops
   *  up a modal preview; deselecting does not re-open it. */
  imageUrl?: string;
}

/** Build the flat catalogue of movements the operator can pick from.
 *  Filters out the legacy single-direction entries that are hidden
 *  from the regular biomech chooser — batch mode only exposes the
 *  movements the current UI surfaces. */
function getCatalogue(): MovementOption[] {
  const out: MovementOption[] = [];
  for (const m of SHOULDER_MOVEMENTS) {
    if (m.hidden) continue;
    out.push({
      bodyPart: "shoulder",
      id: m.id,
      label: m.label,
      target: m.target,
      perSide: true,
      imageUrl: m.imageUrl,
    });
  }
  for (const m of NECK_MOVEMENTS) {
    if (m.hidden) continue;
    out.push({
      bodyPart: "neck",
      id: m.id,
      label: m.label,
      target: m.target,
      perSide: false,
      imageUrl: m.imageUrl,
    });
  }
  for (const m of KNEE_MOVEMENTS) {
    if (m.hidden) continue;
    out.push({
      bodyPart: "knee",
      id: m.id,
      label: m.label,
      target: m.target,
      perSide: true,
      imageUrl: m.imageUrl,
    });
  }
  for (const m of HIP_MOVEMENTS) {
    if (m.hidden) continue;
    out.push({
      bodyPart: "hip",
      id: m.id,
      label: m.label,
      target: m.target,
      perSide: true,
      imageUrl: m.imageUrl,
    });
  }
  for (const m of ANKLE_MOVEMENTS) {
    out.push({
      bodyPart: "ankle",
      id: m.id,
      label: m.label,
      target: m.target,
      perSide: true,
      imageUrl: m.imageUrl,
    });
  }
  return out;
}

type Status = "pending" | "uploading" | "done" | "error";

interface BatchItem {
  uid: string;
  option: MovementOption;
  side: "left" | "right" | null;
  status: Status;
  /** 0..1 during uploading. */
  progress: number;
  file: File | null;
  result: BiomechDataDTO | null;
  errorMsg: string | null;
  /** True after this item's payload has been saved to the patient's
   *  history via createReport. Visible as a small "saved" pill on
   *  the report card. */
  saved: boolean;
}

function bodyPartLabel(b: BodyPart): string {
  return b.charAt(0).toUpperCase() + b.slice(1);
}

function sideLabel(s: "left" | "right" | null): string {
  if (!s) return "";
  return s === "left" ? "Left" : "Right";
}

/** Flatten one completed batch item into the metrics-items entry shape
 *  the saved-report viewer reads. Same keys ApiUploadAssessment's
 *  payload uses, just nested inside the batch metrics blob instead of
 *  being the top-level payload. */
function buildBatchItemEntry(item: BatchItem): Record<string, unknown> {
  const r = item.result!;
  const isMerged =
    !!r.primary_label &&
    !!r.secondary_label &&
    !!r.secondary_reference_range;
  return {
    body_part: item.option.bodyPart,
    movement: item.option.id,
    side: item.side ?? null,
    peak_angle: r.peak_angle,
    peak_magnitude: r.peak_magnitude,
    reference_range: r.reference_range,
    target: r.target,
    percentage: r.percentage,
    status: r.status,
    valid_frames: r.valid_frames,
    total_frames: r.total_frames,
    fps: r.fps,
    interpretation: r.interpretation,
    video_filename: item.file?.name,
    video_size_bytes: item.file?.size,
    ...(isMerged
      ? {
          primary_label: r.primary_label,
          secondary_label: r.secondary_label,
          secondary_peak_angle: r.secondary_peak_angle,
          secondary_peak_magnitude: r.secondary_peak_magnitude,
          secondary_reference_range: r.secondary_reference_range,
        }
      : {}),
    ...(r.key_frames && r.key_frames.length > 0
      ? { key_frames: r.key_frames }
      : {}),
  };
}

// Side selection in the chooser. "both" expands into TWO BatchItems
// (one left + one right) on startSession(); BatchItem.side itself
// only ever holds "left" | "right" | null.
type SideSelection = "left" | "right" | "both" | null;

export function BatchSession() {
  const { isDoctorFlow, patient, patientId } = usePatientContext();

  const [phase, setPhase] = useState<"select" | "running">("select");
  // Selected movements from the catalogue. Keyed by `${bodyPart}.${id}`.
  const [selected, setSelected] = useState<Record<string, SideSelection>>({});
  // Once "Start session" is clicked, the selection is frozen into
  // this ordered queue of mutable BatchItem objects.
  const [queue, setQueue] = useState<BatchItem[]>([]);
  // Reference-image preview popup. Opened when the operator newly
  // selects a movement that has an imageUrl; closed via the X button,
  // the backdrop click, or the Escape key. Deselect does not re-open
  // the popup.
  const [popupImage, setPopupImage] = useState<{ src: string; label: string } | null>(null);

  const catalogue = useMemo(() => getCatalogue(), []);
  const reportRootRef = useRef<HTMLDivElement | null>(null);

  // ── Selection phase helpers ──────────────────────────────────
  const keyOf = (bodyPart: BodyPart, id: string) => `${bodyPart}.${id}`;
  function toggleMovement(opt: MovementOption) {
    const k = keyOf(opt.bodyPart, opt.id);
    const wasOn = k in selected;
    setSelected((prev) => {
      const next = { ...prev };
      if (k in next) {
        delete next[k];
      } else {
        next[k] = opt.perSide ? "right" : null;
      }
      return next;
    });
    // Pop up the reference image only when newly SELECTING (not on
    // deselect). Skipped silently if this movement has no asset.
    if (!wasOn && opt.imageUrl) {
      setPopupImage({ src: opt.imageUrl, label: opt.label });
    }
  }

  // Close the reference-image popup on Escape (in addition to the X
  // button and the backdrop click).
  useEffect(() => {
    if (!popupImage) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setPopupImage(null);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [popupImage]);
  function setItemSide(opt: MovementOption, side: "left" | "right" | "both") {
    const k = keyOf(opt.bodyPart, opt.id);
    setSelected((prev) => (k in prev ? { ...prev, [k]: side } : prev));
  }

  function startSession() {
    const makeItem = (
      opt: MovementOption,
      side: "left" | "right" | null,
    ): BatchItem => ({
      uid: `${opt.bodyPart}-${opt.id}-${side ?? "x"}-${Math.random().toString(36).slice(2, 8)}`,
      option: opt,
      side,
      status: "pending",
      progress: 0,
      file: null,
      result: null,
      errorMsg: null,
      saved: false,
    });

    const items: BatchItem[] = [];
    for (const opt of catalogue) {
      const k = keyOf(opt.bodyPart, opt.id);
      if (!(k in selected)) continue;
      const sel = selected[k];
      if (sel === "both") {
        // Expand into two queued items — one per side. The operator
        // uploads ONE video per item; backend analysis is per-side
        // so each video produces its own report.
        items.push(makeItem(opt, "left"));
        items.push(makeItem(opt, "right"));
      } else {
        items.push(makeItem(opt, sel));
      }
    }
    if (items.length === 0) return;
    setQueue(items);
    setPhase("running");
  }

  // ── Running-phase helpers ────────────────────────────────────
  const updateItem = useCallback(
    (uid: string, patch: Partial<BatchItem>) => {
      setQueue((prev) =>
        prev.map((it) => (it.uid === uid ? { ...it, ...patch } : it)),
      );
    },
    [],
  );

  const onFilePicked = useCallback(
    async (uid: string, file: File) => {
      updateItem(uid, {
        file,
        status: "uploading",
        progress: 0,
        errorMsg: null,
        result: null,
      });
      const item = queueRef.current.find((q) => q.uid === uid);
      if (!item) return;
      try {
        const data = await analyzeBiomechVideo({
          file,
          bodyPart: item.option.bodyPart,
          movement: item.option.id,
          side: item.side ?? undefined,
          onProgress: (frac) => updateItem(uid, { progress: frac }),
        });
        updateItem(uid, {
          result: data,
          status: "done",
          progress: 1,
        });
      } catch (e) {
        updateItem(uid, {
          status: "error",
          errorMsg: e instanceof Error ? e.message : "Analysis failed",
        });
      }
    },
    [updateItem],
  );

  // Ref-mirror of queue so the analyseBiomechVideo callback sees the
  // current item state when it captures the closure. Without this,
  // the picked-file's analysis path uses the stale snapshot at the
  // time the click handler was bound.
  const queueRef = useRef<BatchItem[]>(queue);
  queueRef.current = queue;

  function retryItem(uid: string) {
    updateItem(uid, {
      file: null,
      status: "pending",
      progress: 0,
      errorMsg: null,
      result: null,
    });
  }

  function resetSession() {
    setQueue([]);
    setSelected({});
    setPhase("select");
  }

  // ── Save All + PDF ─────────────────────────────────────────────
  const [saveState, setSaveState] = useState<
    | { kind: "idle" }
    | { kind: "saving"; done: number; total: number }
    | { kind: "saved"; count: number }
    | { kind: "error"; msg: string }
  >({ kind: "idle" });

  const allDone =
    queue.length > 0 && queue.every((it) => it.status === "done");
  const someErrored = queue.some((it) => it.status === "error");
  const anySaved = queue.some((it) => it.saved);

  async function saveAll() {
    if (!patientId || !allDone) return;
    const completed = queue.filter((it) => it.status === "done");
    if (completed.length === 0 || queue.some((it) => it.saved)) return;
    setSaveState({ kind: "saving", done: 0, total: 1 });
    // ONE createReport call with all per-item entries packed into
    // metrics.items + an `is_batch` sentinel so the saved-report
    // viewer can detect the batch shape and render every entry as a
    // stacked AssessmentReport. body_part / movement / side stay null
    // on the wrapper row — per-item identity lives inside each entry.
    const payload: ReportCreatePayload = {
      module: "biomech",
      movement: "batch",
      metrics: {
        is_batch: true,
        items: completed.map(buildBatchItemEntry),
      },
    };
    try {
      await createReport(patientId, payload);
      setQueue((prev) =>
        prev.map((it) =>
          it.status === "done" ? { ...it, saved: true } : it,
        ),
      );
      setSaveState({ kind: "saved", count: completed.length });
    } catch (e) {
      setSaveState({
        kind: "error",
        msg: e instanceof Error ? e.message : "Could not save batch report.",
      });
    }
  }

  const [pdfBusy, setPdfBusy] = useState(false);
  async function downloadPdf() {
    const root = reportRootRef.current;
    if (!root) return;
    setPdfBusy(true);
    try {
      const date = new Date()
        .toISOString()
        .slice(0, 10)
        .replace(/-/g, "");
      const patientPart = patient?.name
        ? `-${patient.name.replace(/\W+/g, "_")}`
        : "";
      await exportReportPdf(root, {
        filename: `biomech-batch${patientPart}-${date}.pdf`,
      });
    } finally {
      setPdfBusy(false);
    }
  }

  // ─── Render ──────────────────────────────────────────────────

  if (phase === "select") {
    const selectedCount = Object.keys(selected).length;
    return (
      <div className="space-y-8">
        {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

        <div>
          <p className="eyebrow">Batch assessment</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">
            Pick the movements to assess
          </h2>
          <p className="mt-2 max-w-xl text-sm text-muted">
            Select any combination of joints + movements. You&apos;ll upload
            one video per selected movement on the next screen, then save
            all results or download a combined PDF.
          </p>
        </div>

        <div className="space-y-6">
          {(["shoulder", "neck", "knee", "hip", "ankle"] as BodyPart[]).map(
            (bp) => {
              const opts = catalogue.filter((o) => o.bodyPart === bp);
              if (opts.length === 0) return null;
              return (
                <section
                  key={bp}
                  className="rounded-card border border-border bg-surface p-5"
                >
                  <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-foreground">
                    {bodyPartLabel(bp)}
                  </h3>
                  <div className="mt-3 space-y-3">
                    {opts.map((opt) => {
                      const k = keyOf(opt.bodyPart, opt.id);
                      const isOn = k in selected;
                      return (
                        <div
                          key={opt.id}
                          className={`flex flex-col gap-3 rounded-card border px-4 py-3 transition sm:flex-row sm:items-center sm:justify-between ${
                            isOn
                              ? "border-accent/50 bg-accent/5"
                              : "border-border bg-background hover:border-accent/30"
                          }`}
                        >
                          <label className="flex cursor-pointer items-center gap-3 text-sm">
                            <input
                              type="checkbox"
                              className="h-4 w-4 shrink-0 rounded border-border accent-accent"
                              checked={isOn}
                              onChange={() => toggleMovement(opt)}
                            />
                            <span className="font-medium text-foreground">
                              {opt.label}
                            </span>
                            <span className="text-xs text-muted">
                              target {opt.target[0]}°–{opt.target[1]}°
                            </span>
                          </label>
                          {opt.perSide && isOn && (
                            <div className="flex shrink-0 gap-2">
                              {(["left", "right", "both"] as const).map((s) => (
                                <button
                                  key={s}
                                  type="button"
                                  onClick={() => setItemSide(opt, s)}
                                  className={`rounded-md border px-3 py-1 text-xs font-medium transition ${
                                    selected[k] === s
                                      ? "border-accent bg-accent text-white"
                                      : "border-border bg-surface text-muted hover:border-accent/50"
                                  }`}
                                >
                                  {s === "both" ? "Left + Right" : sideLabel(s)}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </section>
              );
            },
          )}
        </div>

        <div className="flex flex-col items-stretch gap-3 border-t border-border pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm text-muted">
            {selectedCount === 0
              ? "Select at least one movement to start."
              : `${selectedCount} movement${selectedCount === 1 ? "" : "s"} selected.`}
          </p>
          <Button
            onClick={startSession}
            disabled={selectedCount === 0}
            className="w-full sm:w-auto"
          >
            Start session
          </Button>
        </div>

        {popupImage && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
            onClick={() => setPopupImage(null)}
            role="dialog"
            aria-modal="true"
            aria-label={`${popupImage.label} reference`}
          >
            <div
              className="relative w-full max-w-2xl overflow-hidden rounded-card border border-border bg-surface shadow-glow-sm"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={() => setPopupImage(null)}
                aria-label="Close reference image"
                className="absolute right-3 top-3 z-10 flex h-9 w-9 items-center justify-center rounded-full bg-background/90 text-foreground transition hover:bg-background"
              >
                <X className="h-4 w-4" />
              </button>
              <div className="bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={popupImage.src}
                  alt=""
                  aria-hidden="true"
                  className="block w-full object-contain"
                  style={{ maxHeight: "70vh" }}
                />
              </div>
              <div className="border-t border-border bg-surface p-4">
                <p className="text-sm font-semibold text-foreground">
                  {popupImage.label}
                </p>
                <p className="mt-1 text-xs text-muted">
                  Reference illustration. Press Esc, click the X, or tap outside to close.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── Running phase ────────────────────────────────────────────
  return (
    <div className="space-y-8">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Batch session</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">
            Upload videos & combined report
          </h2>
          <p className="mt-2 text-sm text-muted">
            Upload one video per queued movement. Each report renders below
            as soon as its analysis finishes.
          </p>
        </div>
        <Button variant="ghost" onClick={resetSession} className="no-pdf">
          <RotateCcw className="h-4 w-4" />
          Start over
        </Button>
      </div>

      {/* ── Upload queue ─────────────────────────────────────── */}
      <div className="space-y-4 no-pdf">
        {queue.map((item, idx) => (
          <BatchItemCard
            key={item.uid}
            item={item}
            index={idx + 1}
            onFilePicked={onFilePicked}
            onRetry={retryItem}
          />
        ))}
      </div>

      {/* ── Combined report — rendered into a single ref so the
              PDF exporter can rasterise everything in one shot ──── */}
      <div ref={reportRootRef} className="space-y-12">
        {queue.some((it) => it.status === "done") && (
          <div className="rounded-card border border-border bg-surface px-5 py-4 text-sm">
            <p className="font-semibold text-foreground">
              {patient?.name?.trim() || "Anonymous patient"}
            </p>
            <p className="mt-0.5 text-xs text-muted">
              Combined biomechanics report ·{" "}
              {queue.filter((it) => it.status === "done").length} of{" "}
              {queue.length} movement{queue.length === 1 ? "" : "s"} captured ·{" "}
              {new Date().toLocaleDateString()}
            </p>
          </div>
        )}

        {queue
          .filter((it) => it.status === "done" && it.result)
          .map((item) => {
            const r = item.result!;
            const isMerged =
              !!r.primary_label &&
              !!r.secondary_label &&
              !!r.secondary_reference_range;
            const hasSecondaryValue =
              isMerged && typeof r.secondary_peak_magnitude === "number";
            return (
              <div
                key={item.uid}
                className="border-t border-border pt-8 first:border-t-0 first:pt-0"
              >
                <AssessmentReport
                  bodyPart={item.option.bodyPart}
                  movementName={r.primary_label ?? item.option.label}
                  movementId={item.option.id}
                  measured={r.peak_magnitude}
                  target={[r.reference_range[0], r.reference_range[1]]}
                  side={item.side ?? undefined}
                  keyFrames={r.key_frames}
                  secondaryMovementName={
                    isMerged ? r.secondary_label : undefined
                  }
                  secondaryMeasured={
                    hasSecondaryValue ? r.secondary_peak_magnitude : undefined
                  }
                  secondaryTarget={
                    isMerged ? r.secondary_reference_range : undefined
                  }
                />
              </div>
            );
          })}
      </div>

      {/* ── Action buttons (excluded from PDF via no-pdf) ───── */}
      {queue.length > 0 && (
        <div className="no-pdf space-y-3 border-t border-border pt-6">
          {someErrored && (
            <div className="flex items-start gap-2 rounded-card border border-error/30 bg-error/5 px-4 py-3 text-sm">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">
                Some uploads failed. Retry them before saving or generating
                the PDF — only completed items will be included.
              </p>
            </div>
          )}

          {saveState.kind === "saving" && (
            <div className="flex items-center gap-2 rounded-card border border-accent/30 bg-accent/5 px-4 py-3 text-sm">
              <Loader2 className="h-4 w-4 animate-spin text-accent" />
              Saving {saveState.done} of {saveState.total}…
            </div>
          )}
          {saveState.kind === "saved" && (
            <div className="flex items-center gap-2 rounded-card border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-sm">
              <CheckCircle2 className="h-4 w-4 text-emerald-600" />
              Saved {saveState.count} report{saveState.count === 1 ? "" : "s"}{" "}
              to {patient?.name || "patient"}&apos;s history.
            </div>
          )}
          {saveState.kind === "error" && (
            <div className="flex items-start gap-2 rounded-card border border-error/30 bg-error/5 px-4 py-3 text-sm">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
              <p className="text-foreground">{saveState.msg}</p>
            </div>
          )}

          <div className="flex flex-col gap-3 sm:flex-row sm:justify-end">
            {isDoctorFlow && (
              <Button
                onClick={saveAll}
                disabled={
                  !allDone ||
                  saveState.kind === "saving" ||
                  (anySaved && saveState.kind === "saved")
                }
              >
                {saveState.kind === "saving" ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Saving…
                  </>
                ) : saveState.kind === "saved" ? (
                  <>
                    <CheckCircle2 className="h-4 w-4" />
                    Saved
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Save all to patient history
                  </>
                )}
              </Button>
            )}
            <Button
              variant="secondary"
              onClick={downloadPdf}
              disabled={!allDone || pdfBusy}
            >
              {pdfBusy ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating PDF…
                </>
              ) : (
                <>
                  <Download className="h-4 w-4" />
                  Download PDF
                </>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Single-item upload card ─────────────────────────────────────

interface CardProps {
  item: BatchItem;
  index: number;
  onFilePicked: (uid: string, file: File) => void;
  onRetry: (uid: string) => void;
}

function BatchItemCard({ item, index, onFilePicked, onRetry }: CardProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const title = `${bodyPartLabel(item.option.bodyPart)} · ${item.option.label}${
    item.side ? ` (${sideLabel(item.side)})` : ""
  }`;
  const handleClick = () => inputRef.current?.click();
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) onFilePicked(item.uid, f);
  };

  // Once the analysis succeeds, the upload card hides and the inline
  // <AssessmentReport> renders below. We keep a compact "done" status
  // strip in place of the card so the operator can scroll the queue
  // and see at a glance which items completed.
  if (item.status === "done") {
    return (
      <div className="flex items-center gap-3 rounded-card border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-sm">
        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
        <span className="font-medium text-foreground">{title}</span>
        <span className="ml-auto text-xs text-muted">
          peak {item.result?.peak_magnitude.toFixed(1)}°
        </span>
        {item.saved && (
          <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">
            saved
          </span>
        )}
      </div>
    );
  }

  if (item.status === "error") {
    return (
      <div className="rounded-card border border-error/30 bg-error/5 p-4 text-sm">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <div className="flex-1">
            <p className="font-medium text-foreground">{title}</p>
            <p className="mt-1 text-muted">{item.errorMsg}</p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onRetry(item.uid)}
          >
            <RotateCcw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-card border border-border bg-surface p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-[0.12em] text-subtle">
            Movement {index}
          </p>
          <p className="mt-0.5 truncate text-sm font-medium text-foreground">
            {title}
          </p>
          <p className="mt-0.5 text-xs text-muted">
            Target {item.option.target[0]}°–{item.option.target[1]}°
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          {item.status === "uploading" ? (
            <div className="flex min-w-[180px] items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin text-accent" />
              <div className="flex-1">
                <div className="h-1.5 overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full bg-accent transition-all"
                    style={{ width: `${Math.round(item.progress * 100)}%` }}
                  />
                </div>
                <p className="mt-1 text-[10px] text-subtle">
                  {item.progress < 1
                    ? `Uploading ${Math.round(item.progress * 100)}%`
                    : "Analysing…"}
                </p>
              </div>
            </div>
          ) : (
            <>
              <Button variant="secondary" onClick={handleClick}>
                <Upload className="h-4 w-4" />
                {item.file ? "Re-pick" : "Upload video"}
              </Button>
              {item.file && (
                <span className="hidden max-w-[160px] truncate text-xs text-muted sm:inline">
                  {item.file.name}
                </span>
              )}
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleChange}
          />
        </div>
      </div>
    </div>
  );
}
