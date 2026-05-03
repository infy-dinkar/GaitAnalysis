import { Info } from "lucide-react";

interface InfoBoxProps {
  title?: string;
  children: React.ReactNode;
}

/** Blue "Understanding this graph" explainer box. */
export function InfoBox({ title = "Understanding this graph", children }: InfoBoxProps) {
  return (
    <div className="rounded-card border border-data-2/30 bg-data-2/5 p-4">
      <div className="flex items-start gap-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-data-2" />
        <div className="text-sm text-muted">
          <p className="mb-1 text-xs font-semibold uppercase tracking-[0.12em] text-data-2">
            {title}
          </p>
          <div className="leading-relaxed">{children}</div>
        </div>
      </div>
    </div>
  );
}
