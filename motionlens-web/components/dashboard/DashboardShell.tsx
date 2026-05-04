"use client";
// Sticky sidebar + top bar layout for all /dashboard/* pages.

import { useState, type ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  ArrowLeft,
  LayoutDashboard,
  LogOut,
  Menu,
  Stethoscope,
  Users,
  X,
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/dashboard/patients", label: "Patients", icon: Users },
];

interface Props {
  children: ReactNode;
  /** Optional back link (rendered next to page title in topbar). */
  backHref?: string;
  backLabel?: string;
  /** Page-specific title shown in topbar. */
  title?: string;
}

export function DashboardShell({ children, backHref, backLabel, title }: Props) {
  const { doctor, signOut } = useAuth();
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background">
      {/* ─── Sidebar ──────────────────────────────────────── */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border bg-elevated transition-transform lg:static lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Logo */}
        <Link
          href="/dashboard"
          className="flex h-16 items-center gap-0.5 border-b border-border px-6 text-base font-semibold tracking-tight"
        >
          <Stethoscope className="mr-2 h-5 w-5 text-accent" />
          <span>MotionLens</span>
          <span className="text-accent">.</span>
        </Link>

        {/* Nav */}
        <nav className="flex-1 px-3 py-6">
          <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-subtle">
            Workspace
          </p>
          <ul className="space-y-1">
            {NAV.map((item) => {
              const active =
                pathname === item.href ||
                (item.href !== "/dashboard" && pathname?.startsWith(item.href));
              const Icon = item.icon;
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className={cn(
                      "flex items-center gap-3 rounded-card px-3 py-2 text-sm transition",
                      active
                        ? "bg-accent/10 font-medium text-accent"
                        : "text-muted hover:bg-surface hover:text-foreground",
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Doctor card + signout */}
        <div className="border-t border-border p-4">
          <div className="rounded-card bg-surface p-3">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent/15 text-sm font-semibold text-accent">
                {doctor?.name?.[0]?.toUpperCase() ?? "?"}
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-foreground">
                  {doctor?.name ?? "—"}
                </p>
                <p className="truncate text-xs text-muted">{doctor?.email ?? ""}</p>
              </div>
            </div>
            <button
              type="button"
              onClick={signOut}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-md border border-border bg-background py-1.5 text-xs text-muted transition hover:border-error/40 hover:text-error"
            >
              <LogOut className="h-3.5 w-3.5" />
              Sign out
            </button>
          </div>
        </div>
      </aside>

      {/* ─── Mobile backdrop ─────────────────────────────── */}
      {mobileOpen && (
        <button
          type="button"
          aria-label="Close menu"
          className="fixed inset-0 z-30 bg-black/30 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ─── Main column ─────────────────────────────────── */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Topbar */}
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-border bg-background/80 px-4 backdrop-blur md:px-6">
          <button
            type="button"
            className="rounded-md p-2 text-muted hover:bg-surface lg:hidden"
            onClick={() => setMobileOpen((v) => !v)}
            aria-label="Open menu"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>

          {backHref && (
            <Link
              href={backHref}
              className="flex items-center gap-1.5 rounded-md px-2 py-1 text-sm text-muted transition hover:bg-surface hover:text-foreground"
            >
              <ArrowLeft className="h-4 w-4" />
              {backLabel ?? "Back"}
            </Link>
          )}

          {title && (
            <h1 className="ml-1 text-sm font-medium text-foreground">{title}</h1>
          )}

          <div className="ml-auto flex items-center gap-2 text-xs text-subtle">
            <span className="inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
            <Activity className="h-3.5 w-3.5" />
            Live
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-x-hidden">
          <div className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
