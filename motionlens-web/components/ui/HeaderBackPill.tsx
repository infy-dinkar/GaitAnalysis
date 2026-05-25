"use client";
// Accent-coloured "Back" pill designed to be dropped inside an existing
// header (Nav.tsx for public pages, DashboardShell topbar for dashboard
// pages). NOT fixed-positioned — that's the whole point. Floating
// versions kept colliding with per-page action buttons ("New
// assessment", "Compare", "Delete patient") on every page that had a
// right-aligned button cluster. Putting it inside the header puts it
// in the natural top-right of the page without ever overlapping page
// content.
//
// Hidden on `/` (the landing page) since there's no meaningful back
// target from the root. Falls back to /dashboard (if signed in) or /
// when the browser has no session history (direct-link arrivals).

import { usePathname, useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

export function HeaderBackPill() {
  const router = useRouter();
  const pathname = usePathname();
  const { doctor } = useAuth();

  if (pathname === "/") return null;

  function handleBack() {
    if (typeof window !== "undefined" && window.history.length > 1) {
      router.back();
      return;
    }
    router.push(doctor ? "/dashboard" : "/");
  }

  return (
    <button
      type="button"
      onClick={handleBack}
      aria-label="Go back"
      className="inline-flex items-center gap-1.5 rounded-full bg-accent px-3 py-1.5 text-sm font-medium text-white shadow-md shadow-accent/25 transition hover:bg-accent/90 active:scale-95 sm:px-4"
    >
      <ArrowLeft className="h-4 w-4" />
      {/* Icon-only on the narrowest viewports so the navbar doesn't
          get crammed; full label from sm: up. */}
      <span className="hidden sm:inline">Back</span>
    </button>
  );
}
