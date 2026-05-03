import { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Badge({ className, ...rest }: HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn("eyebrow", className)} {...rest} />;
}
