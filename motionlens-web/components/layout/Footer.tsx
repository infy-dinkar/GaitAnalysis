import Link from "next/link";

const COLUMNS: { title: string; links: { label: string; href: string }[] }[] = [
  {
    title: "Product",
    links: [
      { label: "Gait Analysis", href: "/gait" },
      { label: "Biomechanics", href: "/biomech" },
    ],
  },
  {
    title: "Company",
    links: [
      { label: "About", href: "#about" },
      { label: "Contact", href: "#contact" },
    ],
  },
  {
    title: "Legal",
    links: [
      { label: "Privacy", href: "#privacy" },
      { label: "Terms", href: "#terms" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto w-full max-w-7xl px-6 py-8 md:px-10 md:py-10">
        <div className="grid gap-8 md:grid-cols-[2fr_1fr_1fr_1fr]">
          <div className="max-w-xs">
            <Link
              href="/"
              className="flex items-center gap-0.5 text-base font-semibold tracking-tight"
            >
              <span>MotionLens</span>
              <span className="text-accent">.</span>
            </Link>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              Clinical-grade biomechanics from any video. Markerless, lab-free.
            </p>
          </div>
          {COLUMNS.map((col) => (
            <div key={col.title}>
              <h4 className="text-xs font-semibold uppercase tracking-[0.12em] text-foreground">
                {col.title}
              </h4>
              <ul className="mt-3 space-y-1.5">
                {col.links.map((link) => (
                  <li key={link.label}>
                    <Link
                      href={link.href}
                      className="text-xs text-muted transition hover:text-foreground"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-6 flex flex-col gap-1 border-t border-border pt-4 text-[11px] text-subtle md:flex-row md:items-center md:justify-between">
          <p>© {new Date().getFullYear()} MotionLens. All rights reserved.</p>
          <p>Built for clinicians, researchers, and movement scientists.</p>
        </div>
      </div>
    </footer>
  );
}
