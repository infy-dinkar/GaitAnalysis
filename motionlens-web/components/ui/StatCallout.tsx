import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatCalloutProps {
  value: string;
  label: string;
  icon?: LucideIcon;
  className?: string;
}

export function StatCallout({ value, label, icon: Icon, className }: StatCalloutProps) {
  return (
    <div className={cn("flex items-center gap-3", className)}>
      {Icon && (
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
          <Icon className="h-5 w-5" />
        </span>
      )}
      <div>
        <div className="tabular text-2xl font-semibold leading-none text-foreground">
          {value}
        </div>
        <div className="mt-1 text-xs text-muted">{label}</div>
      </div>
    </div>
  );
}
