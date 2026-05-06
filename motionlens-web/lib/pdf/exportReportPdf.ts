"use client";
// Capture a DOM element (a rendered saved-report body) and emit a
// multi-page A4 PDF. Uses html2canvas to rasterize the element — this
// gives us pixel-perfect output for Plotly charts, the posture image
// overlay (canvas), tables, and any other dynamic visuals exactly as
// they appear on screen.
//
// Pagination strategy:
//   1. Render the whole element to a single tall canvas at scale 2x
//      (sharper text, manageable file size).
//   2. Slice the canvas into A4-page-sized vertical slabs.
//   3. Add each slab as a separate jsPDF page so multi-screen reports
//      flow naturally instead of getting compressed onto one page.

// html2canvas-pro is the maintained fork that supports modern CSS
// color functions (oklch / oklab / lab / color()). Tailwind v4 emits
// oklch by default — original html2canvas throws "unsupported color
// function 'oklab'" the moment it encounters one of those values.
import html2canvas from "html2canvas-pro";
import jsPDF from "jspdf";

interface ExportOptions {
  filename: string;
  /** Optional title rendered in the top-left of every page. */
  title?: string;
}

const A4_WIDTH_PT = 595.28;   // jsPDF default A4 size in points
const A4_HEIGHT_PT = 841.89;
const PAGE_MARGIN = 24;       // leave a small margin around the slab

export async function exportReportPdf(
  element: HTMLElement,
  opts: ExportOptions,
): Promise<void> {
  // Scale up for sharper text / chart strokes — A4 at 96 DPI is ~794px
  // wide, so a 2x scale ≈ retina rendering.
  const scale = Math.min(2, Math.max(1, window.devicePixelRatio || 1));

  // Some chart libraries (Plotly) attach event listeners that mutate
  // the DOM after html2canvas snapshots it; passing `useCORS: true`
  // lets canvas elements with loaded images (e.g. PostureImageOverlay
  // backdrop) get captured rather than triggering a security taint.
  const canvas = await html2canvas(element, {
    scale,
    useCORS: true,
    backgroundColor: "#ffffff",
    logging: false,
    // html2canvas mis-renders gradient borders + box-shadow in some
    // Tailwind classes; turning these off gives a cleaner print look
    // without changing what's on-screen.
    ignoreElements: (el) => el.classList?.contains("no-pdf") ?? false,
  });

  const pdf = new jsPDF({ unit: "pt", format: "a4", orientation: "portrait" });
  const usableWidth = A4_WIDTH_PT - PAGE_MARGIN * 2;
  const usableHeight = A4_HEIGHT_PT - PAGE_MARGIN * 2;

  // Each PDF page shows a vertical slice of the source canvas. The
  // slice height in canvas pixels is computed from the canvas → page
  // scale ratio so it lands on whole pages.
  const pageCanvasHeight = Math.floor((canvas.width * usableHeight) / usableWidth);
  const totalPages = Math.max(1, Math.ceil(canvas.height / pageCanvasHeight));

  for (let i = 0; i < totalPages; i++) {
    if (i > 0) pdf.addPage();

    const sourceY = i * pageCanvasHeight;
    const sliceHeight = Math.min(pageCanvasHeight, canvas.height - sourceY);

    // Slice the source canvas onto a per-page canvas and toDataURL it.
    const slice = document.createElement("canvas");
    slice.width = canvas.width;
    slice.height = sliceHeight;
    const ctx = slice.getContext("2d");
    if (!ctx) throw new Error("Could not allocate canvas slice for PDF export.");
    ctx.drawImage(
      canvas,
      0, sourceY, canvas.width, sliceHeight,
      0, 0,         canvas.width, sliceHeight,
    );
    const data = slice.toDataURL("image/png");
    const renderedHeight = (sliceHeight * usableWidth) / canvas.width;

    pdf.addImage(
      data,
      "PNG",
      PAGE_MARGIN,
      PAGE_MARGIN,
      usableWidth,
      renderedHeight,
      undefined,
      "FAST",
    );

    if (opts.title) {
      pdf.setFontSize(8);
      pdf.setTextColor(120);
      pdf.text(opts.title, PAGE_MARGIN, 16);
      pdf.text(
        `Page ${i + 1} / ${totalPages}`,
        A4_WIDTH_PT - PAGE_MARGIN,
        16,
        { align: "right" },
      );
    }
  }

  pdf.save(opts.filename);
}
