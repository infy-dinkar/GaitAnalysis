"use client";
// LiveModeLayout — shared full-viewport 50/50 split shell for live
// biomech + rehab pages.
//
// Layout contract:
//   • Left half (50%)  — camera view slot (`camera` prop).
//   • Right half (50%) — instructions / controls / reference image
//                        slot (`sidebar` prop).
//   • Header strip spanning the full width with the exercise title,
//     side chip, and an Exit / Minimise button.
//
// Two size modes are available:
//   • "expanded" (default when consumer sets `active`) —
//        position: fixed, inset: 0. Occupies the whole viewport,
//        floats over the page's Nav / Footer / setup help. Escape
//        key or the Exit button closes it back to `active = false`.
//   • "inline" — renders in normal flow (used e.g. when the layout
//        is embedded inside an already-fullscreen sequence runner
//        like Auto Mode where the outer chrome is already hidden).
//
// Fullscreen browser API is optional and gated behind an explicit
// toggle button. The default expanded mode is viewport-full without
// requiring the browser's fullscreen permission.

import {
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { Maximize2, Minimize2, X } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface LiveModeLayoutProps {
  /** Exercise / test title shown in the header. */
  title: string;
  /** Optional subtitle line under the title (side / description). */
  subtitle?: string;
  /** Camera slot — fills the left 50 % of the split. */
  camera: ReactNode;
  /** Sidebar slot — fills the right 50 % of the split. */
  sidebar: ReactNode;
  /** Called when the user clicks Exit / presses Escape. */
  onExit: () => void;
  /**
   * Size mode. "expanded" pins the layout to the viewport and hides
   * the page chrome around it. "inline" renders in normal flow.
   * Default: "expanded".
   */
  mode?: "expanded" | "inline";
  /** Optional right-side header slot for buttons like Save / Reset. */
  headerRight?: ReactNode;
}

export function LiveModeLayout({
  title,
  subtitle,
  camera,
  sidebar,
  onExit,
  mode = "expanded",
  headerRight,
}: LiveModeLayoutProps) {
  const [browserFs, setBrowserFs] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Escape closes the expanded mode. Handled at document level so
  // it works even if focus is inside a nested control.
  useEffect(() => {
    if (mode !== "expanded") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !browserFs) {
        onExit();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [mode, onExit, browserFs]);

  // Track native fullscreen state so the toggle icon flips.
  useEffect(() => {
    const sync = () => setBrowserFs(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, []);

  // Auto-enter native fullscreen on mount. Consumers typically mount
  // this layout in response to a user click (e.g. "Left leg" side
  // picker button), so the fullscreen request is still within the
  // browser's user-activation window and succeeds. If the request is
  // rejected (Safari, embedded iframes, some browser policies), the
  // layout still fills the viewport via `fixed inset-0` — the
  // Maximize button in the header lets the operator retry.
  useEffect(() => {
    if (mode !== "expanded") return;
    if (document.fullscreenElement) return;
    const el = containerRef.current;
    if (!el) return;
    const r = el.requestFullscreen?.();
    if (r && typeof r.catch === "function") r.catch(() => {});
  }, [mode]);

  // On unmount, exit browser fullscreen so the parent page comes
  // back in a clean state.
  useEffect(() => {
    return () => {
      if (document.fullscreenElement) {
        document.exitFullscreen?.().catch(() => {});
      }
    };
  }, []);

  const toggleFs = useCallback(async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else if (containerRef.current) {
        await containerRef.current.requestFullscreen();
      }
    } catch {
      // Some browsers reject the fullscreen request silently.
    }
  }, []);

  const rootClass =
    mode === "expanded"
      ? "fixed inset-0 z-50 flex flex-col bg-background"
      : "relative flex min-h-[85vh] flex-col rounded-hero border border-border bg-background";

  return (
    <div ref={containerRef} className={rootClass}>
      {/* ── Header strip ────────────────────────────────────── */}
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-surface/95 px-4 py-3 backdrop-blur">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold tracking-tight text-foreground md:text-base">
            {title}
          </p>
          {subtitle && (
            <p className="truncate text-[11px] text-muted md:text-xs">
              {subtitle}
            </p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {headerRight}
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleFs}
            aria-label={browserFs ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {browserFs ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={onExit}
            className="text-error hover:bg-error/10"
          >
            <X className="h-4 w-4" />
            Exit
          </Button>
        </div>
      </div>

      {/* ── Split body — camera dominant, sidebar compact. ───
             Camera gets ~68 % of the viewport width, sidebar ~32 %.
             Both halves are hard-capped to their share so the
             sidebar never scrolls — content sits in a compact
             vertical stack sized to fit the available height. */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden md:flex-row">
        <div className="relative flex min-h-0 flex-1 items-stretch overflow-hidden bg-black md:basis-[68%]">
          {/* Inject `fill` into the camera child so the RehabCameraShell
              (and the LiveBiomechCamera on the biomech side, if wired)
              drops its intrinsic aspect-video ratio and stretches to
              fill the entire left half. Camera slots that don't accept
              the prop simply ignore it. */}
          <div className="flex min-h-0 w-full items-stretch justify-stretch p-2">
            {isValidElement(camera)
              ? cloneElement(camera as ReactElement<{ fill?: boolean }>, { fill: true })
              : camera}
          </div>
        </div>
        <div className="flex min-h-0 shrink-0 flex-col overflow-hidden border-t border-border md:basis-[32%] md:border-l md:border-t-0">
          {/* Fill the full sidebar column height — any child marked
              with `flex-1` (typically the live stats card) will grow
              to absorb the leftover space so we never leave an empty
              gap at the bottom of the sidebar. */}
          <div className="flex min-h-0 flex-1 flex-col gap-3 p-3 md:p-4">
            {sidebar}
          </div>
        </div>
      </div>
    </div>
  );
}
