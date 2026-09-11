"use client";
import { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface TabDef<TId extends string = string> {
  id: TId;
  label: string;
}

interface JointTabsProps<TId extends string> {
  tabs: TabDef<TId>[];
  active: TId;
  onChange: (id: TId) => void;
  children: ReactNode;
  /** Extra classes for the tab BAR only — the content wrapper below is
   *  never affected. Added so the saved-report view can mark the bar
   *  "no-pdf": the tabs are interactive, and only the active tab's
   *  children are ever mounted, so a printed bar advertises seven tabs
   *  whose content is not in the document. Undefined by default, which
   *  leaves the rendered markup byte-identical for every other caller. */
  barClassName?: string;
}

export function JointTabs<TId extends string>({
  tabs,
  active,
  onChange,
  children,
  barClassName,
}: JointTabsProps<TId>) {
  return (
    <div>
      <div className={cn("flex flex-wrap gap-1 border-b border-border", barClassName)}>
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onChange(t.id)}
            className={cn(
              "relative -mb-px px-4 py-2.5 text-xs font-semibold uppercase tracking-[0.12em] transition",
              active === t.id
                ? "text-accent"
                : "text-muted hover:text-foreground",
            )}
          >
            {t.label}
            {active === t.id && (
              <span
                className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-accent"
                aria-hidden
              />
            )}
          </button>
        ))}
      </div>
      <div className="mt-6">{children}</div>
    </div>
  );
}
