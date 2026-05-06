"use client";
// Light/dark theme toggle. State lives on the <html> element via
// `data-theme="dark"`/"light" (set by the FOUC-prevention script in
// app/layout.tsx). The toggle reads + flips that attribute and
// persists the choice to localStorage under "motionlens.theme".

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

type Theme = "light" | "dark";

const STORAGE_KEY = "motionlens.theme";

function readCurrentTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export function ThemeToggle({ className = "" }: { className?: string }) {
  // Start as null on the server / first client render so the icon
  // doesn't flash the wrong glyph. We populate on mount.
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    setTheme(readCurrentTheme());
  }, []);

  function toggle() {
    const next: Theme = (theme ?? readCurrentTheme()) === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // localStorage unavailable — toggle still works for the session.
    }
    setTheme(next);
  }

  // Render a neutral placeholder during the brief pre-hydration window
  // so the button keeps its slot in the layout without a glyph flash.
  const isDark = theme === "dark";
  const label = isDark ? "Switch to light theme" : "Switch to dark theme";

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={label}
      title={label}
      className={
        "inline-flex h-9 w-9 items-center justify-center rounded-full " +
        "border border-border bg-elevated text-muted transition " +
        "hover:border-accent hover:text-accent " +
        className
      }
    >
      {theme === null ? (
        <span className="block h-4 w-4" aria-hidden />
      ) : isDark ? (
        <Sun className="h-4 w-4" />
      ) : (
        <Moon className="h-4 w-4" />
      )}
    </button>
  );
}
