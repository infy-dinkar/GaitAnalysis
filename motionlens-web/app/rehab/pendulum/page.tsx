"use client";
// S3 — Pendulum / Circle Trace.
//
// Mechanic: Trace (lib/rehab/mechanics.ts traceStep). Cursor is
// in normalised [0..1] × [0..1] CSS-y-down coords — same space
// TraceShell expects.
//
// Cursor source — the WRIST landmark directly (no biomech math
// needed; the wrist x/y IS the game control):
//   cursor.x ← 1 − wrist.x / video.videoWidth   (mirrored for selfie
//                                                view, same direction
//                                                as the on-screen
//                                                skeleton)
//   cursor.y ← wrist.y / video.videoHeight
//
// Path: gentle counter-clockwise circle, centre (0.5, 0.5), radius
// 0.25 of the play area, 8 s per revolution. Tuned for early-stage
// frozen-shoulder / post-op pendulum mobility — slow enough that the
// patient can stay glued to the lead target without rushing.
//
// Reuses (no modifications):
//   • RehabCameraShell, TraceShell, traceStep — rehab mechanic library
//   • LM_LIVE wrist indices — lib/pose/landmarks-live
//   • usePoseDetectionLive, useCamera (via RehabCameraShell)
//   • usePatientContext for ?patientId doctor flow
// NO biomech file imported or touched — wrist position is the raw signal.

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { TraceShell } from "@/components/rehab/mechanics/TraceShell";
import { LM_LIVE as LM } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { TracePathPoint } from "@/lib/rehab/gameState";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";

const WRIST_VIS_THRESHOLD = 0.3;

// Counter-clockwise circle centred on the play area. Radius 0.25
// of the normalised play width means the wrist arcs through 50 %
// of the canvas span at apex — a comfortable pendulum range for
// the typical clinic setup. Slow 8 s loop period.
function circlePath(t: number): TracePathPoint {
  const angle = t * 2 * Math.PI;
  const radius = 0.25;
  return {
    x: 0.5 + radius * Math.cos(angle),
    y: 0.5 + radius * Math.sin(angle),
  };
}
const LOOP_DURATION_MS = 8_000;

const TRACE_CONFIG = {
  // Distance from path target (normalised units) that still counts
  // as "on-path". 6 % of the play width — about a finger-width
  // tolerance at typical clinic camera framing.
  accuracyTolerance: 0.06,
  // Maximum acceptable cursor jerk (Δ² distance / ms²) for a sample
  // to count as smooth. Empirically the value most patients hit
  // when moving deliberately; jerky guard / spasm moves blow past it.
  smoothnessTolerance: 0.001,
  pointsPerSample: 1,
};

export default function PendulumExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [side, setSide] = useState<Side | null>(null);
  // Default cursor at centre — patient will pick it up when they
  // raise the arm into frame. We don't want a jarring jump from a
  // (0,0) origin on the first valid frame.
  const [cursor, setCursor] = useState<{ x: number; y: number }>({
    x: 0.5,
    y: 0.5,
  });
  // Track whether the wrist has been successfully read at least
  // once — purely for the on-screen "tracking" indicator.
  const wristSeenRef = useRef(false);
  const [tracking, setTracking] = useState(false);

  const { patient, isDoctorFlow } = usePatientContext();

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!side) return;
      const wristIdx = side === "right" ? LM.RIGHT_WRIST : LM.LEFT_WRIST;
      const wrist = kp[wristIdx];
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (
        !wrist
        || (wrist.score ?? 0) < WRIST_VIS_THRESHOLD
        || vw <= 0
        || vh <= 0
      ) {
        // Sticky last position — don't snap cursor on dropouts.
        return;
      }
      // Mirror x to match the selfie-mirror skeleton view (the
      // shell renders the camera feed with -scale-x-100 and its
      // skeleton overlay does the same flip, so the cursor must
      // too for natural left-right correspondence).
      const cx = Math.max(0, Math.min(1, 1 - wrist.x / vw));
      const cy = Math.max(0, Math.min(1, wrist.y / vh));
      setCursor({ x: cx, y: cy });
      if (!wristSeenRef.current) {
        wristSeenRef.current = true;
        setTracking(true);
      }
    },
    [side],
  );

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>S3 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Pendulum / Circle Trace<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Gentle shoulder mobility — patient leans forward,
                lets the arm hang, and traces a slow circle following
                a moving lead target. Designed for frozen-shoulder
                management and early post-op range work. Cursor is
                the wrist position directly; score = path accuracy +
                cursor smoothness (low jerk). Powered by the Trace
                mechanic.
              </p>
              {isDoctorFlow && patient && (
                <p className="mt-3 text-xs text-muted">
                  Connected to{" "}
                  <span className="font-semibold text-foreground">
                    {patient.name}
                  </span>
                  &apos;s record.
                </p>
              )}
            </div>
            <Link href="/rehab">
              <Button variant="ghost" size="sm">← Catalogue</Button>
            </Link>
          </div>

          {!side ? (
            <SidePicker onPick={setSide} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-purple-500/15 px-3 py-1 text-xs font-semibold text-purple-200 ring-1 ring-purple-400/40">
                  Testing: {side === "left" ? "Left" : "Right"} arm
                </span>
                <span
                  className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ring-1 ${
                    tracking
                      ? "bg-emerald-500/15 text-emerald-200 ring-emerald-400/40"
                      : "bg-amber-500/15 text-amber-200 ring-amber-400/40"
                  }`}
                >
                  {tracking ? "Wrist tracking" : "Waiting for wrist…"}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSide(null)}
                >
                  Change side
                </Button>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  <RehabCameraShell onFrame={handleFrame}>
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        {side === "left" ? "Left" : "Right"} wrist
                      </p>
                      <p className="tabular text-sm font-semibold text-white">
                        x {cursor.x.toFixed(2)} · y {cursor.y.toFixed(2)}
                      </p>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <TraceShell
                    cursor={cursor}
                    pathFn={circlePath}
                    loopDurationMs={LOOP_DURATION_MS}
                    config={TRACE_CONFIG}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Setup help */}
          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at chest height, ~2 m away, perpendicular to
                the patient — they face the camera (frontal view).
              </li>
              <li>
                Patient leans forward slightly so the test arm hangs
                free in front of the chest — classic pendulum
                position.
              </li>
              <li>
                Make sure the wrist on the test side stays in frame
                across the full circle. The &quot;Wrist tracking&quot;
                badge above turns green once the landmark is locked.
              </li>
              <li>
                A lead target moves around a slow circle (one loop
                every{" "}
                <strong>{(LOOP_DURATION_MS / 1000).toFixed(0)} s</strong>).
                Keep the wrist on the lead — accuracy + smoothness
                are scored per frame.
              </li>
              <li>
                Both accurate AND smooth samples count toward the
                score; jerky or guarded movement breaks the streak.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function SidePicker({ onPick }: { onPick: (s: Side) => void }) {
  return (
    <div className="mt-10 max-w-xl">
      {REHAB_EXERCISE_IMAGES["pendulum"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["pendulum"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the test arm
      </h2>
      <p className="mt-2 text-sm text-muted">
        Pick the arm the patient will pendulum-trace. We track that
        side&apos;s wrist landmark directly.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick("left")}>Left arm</Button>
        <Button onClick={() => onPick("right")}>Right arm</Button>
      </div>
    </div>
  );
}
