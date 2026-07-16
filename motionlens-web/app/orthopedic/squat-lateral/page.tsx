"use client";
import Link from "next/link";
import { Suspense } from "react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SquatLateralCapture } from "@/components/orthopedic/SquatLateralCapture";

export default function SquatLateralPage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between">
            <div className="max-w-2xl">
              <Badge>Physio squat · lateral</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Squat (Lateral)<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Sagittal-plane squat screen — side-on single camera.
                Patient stands with the analysed leg toward the camera
                and performs 3-6 slow squats to about parallel depth.
                Five metrics reported at the deepest rep: peak knee
                flexion, peak hip flexion, trunk lean, hip:knee ratio,
                heel rise. Ankle dorsiflexion is intentionally not
                scored — near-side ankle keypoints jitter under
                occlusion in a lateral view, and BlazePose foot
                landmarks aren&apos;t reliable for a 2D dorsi/plantar
                angle.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Frontal-plane items honestly not assessed:</strong>{" "}
                knee valgus / pelvic drop are frontal-plane findings
                that a lateral camera cannot see. Use Overhead Squat or
                Single-Leg Squat for those. All caveats are surfaced in
                the report.
              </p>
              <p className="mt-3 text-xs text-muted">
                <strong>Scale calibration:</strong> the system measures
                body pixel height during the standing window and combines
                it with the entered standing height to give the
                heel-rise threshold in centimetres. Skipping the height
                step is allowed — heel-rise falls back to a
                fraction-of-leg-length threshold and classification
                stays valid.
              </p>
            </div>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">← Dashboard</Button>
            </Link>
          </div>

          <div className="mt-10">
            <SquatLateralCapture />
          </div>

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <p className="mt-2 text-sm font-medium text-foreground">
              LATERAL view, full body in frame.
            </p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at hip height, ~2.5-3 m away, perpendicular to
                the patient. The declared leg is CLOSER to the camera.
              </li>
              <li>
                Feet shoulder-width apart, arms relaxed. Frame must
                show the full body: head, shoulders, hips, knees, and
                BOTH feet.
              </li>
              <li>
                Hold a ~1 s static standing pose BEFORE the first squat
                — this locks the baseline hip Y + heel Y (heel-rise
                reference).
              </li>
              <li>
                Perform 3-6 slow squats to about parallel depth.
                Return fully to standing between reps.
              </li>
              <li>
                To assess the other leg, turn the patient around and
                re-record.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
