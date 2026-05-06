import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MotionLens — AI-Powered Motion Analysis",
  description:
    "Extract clinical-grade biomechanics from any video. Markerless gait and joint analysis powered by browser-based pose estimation.",
};

// Runs before React hydrates — sets data-theme on <html> from
// localStorage (or prefers-color-scheme on first visit) so the page
// never paints with the wrong theme. The try/catch keeps SSR + the
// rare browsers without localStorage from throwing.
const themeBootstrapScript = `
(function () {
  try {
    var saved = window.localStorage.getItem('motionlens.theme');
    var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var theme = saved === 'dark' || saved === 'light' ? saved : (prefersDark ? 'dark' : 'light');
    document.documentElement.dataset.theme = theme;
  } catch (e) {
    document.documentElement.dataset.theme = 'light';
  }
})();
`.trim();

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} antialiased`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootstrapScript }} />
      </head>
      <body className="flex min-h-screen flex-col">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
