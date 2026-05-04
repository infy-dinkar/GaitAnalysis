"use client";
// Auth-page visual — reuses the SkeletonHero from the landing page so
// the brand looks consistent across all surfaces. Adds a calm warm
// gradient + clinical graph grid + tagline.

import { motion } from "framer-motion";
import { SkeletonHero } from "@/components/visuals/SkeletonHero";

export function MedicalAnimations() {
  return (
    <div className="relative h-full w-full overflow-hidden bg-gradient-to-br from-orange-50/60 via-amber-50/40 to-rose-50/30 dark:from-slate-900 dark:via-slate-900 dark:to-orange-950/40">
      {/* Clinical graph-paper grid */}
      <div
        aria-hidden
        className="absolute inset-0 opacity-30 dark:opacity-15"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(234,88,12,0.12) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(234,88,12,0.12) 1px, transparent 1px)
          `,
          backgroundSize: "32px 32px",
        }}
      />

      {/* Slow warm radial glow drifting around */}
      <motion.div
        aria-hidden
        className="absolute inset-0"
        animate={{
          background: [
            "radial-gradient(circle at 30% 30%, rgba(234,88,12,0.14) 0%, transparent 60%)",
            "radial-gradient(circle at 60% 50%, rgba(234,88,12,0.10) 0%, transparent 60%)",
            "radial-gradient(circle at 30% 30%, rgba(234,88,12,0.14) 0%, transparent 60%)",
          ],
        }}
        transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
      />

      {/* Centered: same SkeletonHero used on the landing page */}
      <div className="absolute inset-0 flex items-center justify-center px-8">
        <div className="w-full max-w-md">
          <SkeletonHero />
        </div>
      </div>

      {/* Tagline at bottom */}
      <div className="pointer-events-none absolute bottom-10 left-0 right-0 px-10 text-center">
        <motion.h2
          className="text-2xl font-semibold tracking-tight text-slate-900 dark:text-slate-100 md:text-3xl"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
        >
          Movement, measured<span className="text-accent">.</span>
        </motion.h2>
        <motion.p
          className="mt-2 text-sm text-slate-600 dark:text-slate-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.6 }}
        >
          Markerless biomechanics for clinics, labs, and rehab.
        </motion.p>
      </div>
    </div>
  );
}
