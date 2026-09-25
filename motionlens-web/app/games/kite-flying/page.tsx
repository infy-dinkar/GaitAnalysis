"use client";
// Kite Flying — activity page.
//
// Placed OUTSIDE /dashboard, exactly like the other two games and the
// rehab exercise pages: the doctor launches it from the dashboard and
// the patient id arrives as `?patientId=…`, which usePatientContext
// picks up. No AuthGuard here, matching app/games/cloudburst/page.tsx.

import { Suspense } from "react";
import dynamic from "next/dynamic";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";

// Client-only: the component owns a camera, a pose detector and (during
// play) a Phaser canvas, none of which can be server-rendered.
const KiteFlyingGame = dynamic(
  () => import("@/components/games/KiteFlyingGame").then((m) => m.KiteFlyingGame),
  { ssr: false },
);

export default function KiteFlyingPage() {
  // Next.js 16 requires routes that use useSearchParams (via
  // usePatientContext inside the game) to sit under a Suspense
  // boundary, or the route's static prerender bails at build time.
  return (
    <>
      <Nav />
      <main className="flex-1">
        <Suspense fallback={null}>
          <KiteFlyingGame />
        </Suspense>
      </main>
      <Footer />
    </>
  );
}
