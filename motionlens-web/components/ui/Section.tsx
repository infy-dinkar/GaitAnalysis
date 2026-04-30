import { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface SectionProps extends HTMLAttributes<HTMLElement> {
  as?: keyof React.JSX.IntrinsicElements;
  containerClassName?: string;
}

export function Section({
  className,
  containerClassName,
  children,
  ...rest
}: SectionProps) {
  return (
    <section className={cn("relative w-full py-24 md:py-32", className)} {...rest}>
      <div className={cn("mx-auto w-full max-w-7xl px-6 md:px-10", containerClassName)}>
        {children}
      </div>
    </section>
  );
}
