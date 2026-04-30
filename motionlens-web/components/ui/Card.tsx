import { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type Variant = "default" | "elevated" | "interactive";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: Variant;
}

const variants: Record<Variant, string> = {
  default: "border border-border bg-surface",
  elevated: "border border-border bg-elevated",
  interactive:
    "border border-border bg-surface transition-all duration-200 ease-out hover:border-accent hover:shadow-glow-sm hover:-translate-y-0.5",
};

export function Card({ className, variant = "default", ...rest }: CardProps) {
  return <div className={cn("rounded-card p-6", variants[variant], className)} {...rest} />;
}
