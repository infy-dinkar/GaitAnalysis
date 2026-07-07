"use client";
// B4 — Bird-Dog (★ FIRST MATCH-POSE USE).
//
// Mechanic: Match-Pose (lib/rehab/mechanics.ts matchPoseStep).
// The mechanic was scaffolded + bug-fixed earlier but never wired
// to a real exercise — Bird-Dog is its first driver.
//
// Quadruped pose: patient on hands and knees, extends ONE arm
// forward + the OPPOSITE leg backward, holds for the required
// duration. The system tracks three joint angles simultaneously
// and the engine scores overall pose match.
//
// Joint targets (currentAngles ↔ config.pose keys MUST align):
//
//   arm   = computeShoulderAngle("flexion", kp, combo.armSide)
//           Target 180° — extended arm in line with the trunk axis
//           (quadruped trunk is horizontal; arm-forward = full
//           flexion vs the trunk-down reference).
//
//   leg   = 180 − computeHipAngle("flexion", kp, combo.legSide)
//           The biomech helper returns 0° when the thigh is in
//           line with the trunk DOWNWARD direction. In bird-dog
//           the thigh extends backward (in line with trunk going
//           BACKWARD), so the helper returns 0 — we flip to 180
//           so the target matches the arm convention.
//
//   trunk = computeTrunkAngleFromHorizontal(kp)
//           Target 0° — trunk perfectly horizontal (no sag, no
//           pike). Weighted slightly higher because spinal
//           neutrality is the clinical priority.
//
// Combo handling — bird-dog uses OPPOSITE arm + leg:
//   • "Right arm + Left leg"
//   • "Left arm + Right leg"
// Side picker stores both. One limb is camera-side (visible);
// the other is far-side (likely partially occluded). The
// per-frame currentAngles update is INCREMENTAL — if a joint's
// math returns null for that frame, we keep the previous value
// instead of zeroing the joint, so transient far-side occlusion
// doesn't crash the patient's score.
//
// Match-Pose engine contract handling:
//   • currentAngles keys MUST exist in config.pose. Mismatched
//     keys would compute 0 % for that joint silently.
//   • A null/NaN value scores 0 % for that joint (engine handles).
//     We avoid passing NaN by skipping the per-key update on null.
//
// Reuses (no modifications):
//   • MatchPoseShell, matchPoseStep, RehabCameraShell
//   • computeShoulderAngle, computeHipAngle (biomech — IMPORT ONLY)
//   • computeTrunkAngleFromHorizontal — NEW in poseMetrics
//   • usePatientContext

import { Suspense, useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { RehabCameraShell } from "@/components/rehab/mechanics/RehabCameraShell";
import { MatchPoseShell } from "@/components/rehab/mechanics/MatchPoseShell";
import { RehabSessionFooter } from "@/components/rehab/RehabSessionFooter";
import {
  buildSkeletonPosePayload,
  elapsedSecondsSince,
  kpToPoseSnapshot,
  type BestPoseSnapshot,
  type PoseSnapshot,
} from "@/lib/rehab/sessionHelpers";
import { computeShoulderAngle } from "@/lib/biomech/shoulder-live";
import { computeHipAngle } from "@/lib/biomech/hip-live";
import { computeTrunkAngleFromHorizontal } from "@/lib/rehab/poseMetrics";
import { DEFAULT_LEVEL_INDEX } from "@/lib/rehab/progressionLadders";
import { LM_LIVE } from "@/lib/pose/landmarks-live";
import { usePatientContext } from "@/hooks/usePatientContext";
import type { Keypoint } from "@tensorflow-models/pose-detection";
import type { LiveKeypoint } from "@/hooks/usePoseDetectionLive";
import { REHAB_EXERCISE_IMAGES } from "@/lib/rehab/exerciseImages";

type Side = "left" | "right";
interface Combo {
  armSide: Side;
  legSide: Side;
}

const COMBO_RIGHT_LEFT: Combo = { armSide: "right", legSide: "left" };
const COMBO_LEFT_RIGHT: Combo = { armSide: "left", legSide: "right" };

const BIRD_DOG_CONFIG = {
  pose: {
    arm:   { value: 180, tolerance: 20, weight: 1 },
    leg:   { value: 180, tolerance: 20, weight: 1 },
    trunk: { value: 0,   tolerance: 15, weight: 1.5 },
  },
  // 70 % aggregate match required to count as "achieved"
  achievedThresholdPct: 70,
  // 4 s cumulative hold target — challenging but reachable
  requiredHoldMs: 4_000,
  pointsPerMs: 0.05,
};

export default function BirdDogExercisePage() {
  return (
    <Suspense fallback={null}>
      <Inner />
    </Suspense>
  );
}

function Inner() {
  const [combo, setCombo] = useState<Combo | null>(null);
  // currentAngles starts empty — Match-Pose engine treats missing
  // keys as 0 % score, which is what we want before the patient
  // assumes the pose.
  const [currentAngles, setCurrentAngles] = useState<Record<string, number>>(
    {},
  );

  const { patient, isDoctorFlow } = usePatientContext();

  const sessionStartRef = useRef<number>(performance.now());
  const bestPoseRef = useRef<BestPoseSnapshot | null>(null);
  const lastKpRef = useRef<PoseSnapshot | null>(null);
  const bestMatchRef = useRef<number>(0);

  const handleFrame = useCallback(
    (kp: Keypoint[], video: HTMLVideoElement) => {
      if (!combo) return;
      const snap = kpToPoseSnapshot(kp, video.videoWidth, video.videoHeight);
      if (snap) lastKpRef.current = snap;
      const liveKp = kp as unknown as LiveKeypoint[];
      const arm = computeShoulderAngle("flexion", liveKp, combo.armSide);
      const hipFlex = computeHipAngle("flexion", liveKp, combo.legSide);
      const trunk = computeTrunkAngleFromHorizontal(liveKp);

      // Incremental update — skip per-key if math returned null so
      // transient occlusion of the far-side limb doesn't destroy
      // the patient's match score. Engine treats stale value as
      // last-known good.
      setCurrentAngles((prev) => {
        const next = { ...prev };
        if (arm !== null) next.arm = arm;
        if (hipFlex !== null) next.leg = 180 - hipFlex;
        if (trunk !== null) next.trunk = trunk;
        // Approximate match % from proximity to targets (180 arm/leg, 0 trunk).
        // Higher score = closer to pose.
        const legVal = next.leg ?? 0;
        const armVal = next.arm ?? 0;
        const trunkVal = Math.abs(next.trunk ?? 90);
        const armScore = Math.max(0, 100 - Math.abs(180 - armVal) * 2);
        const legScore = Math.max(0, 100 - Math.abs(180 - legVal) * 2);
        const trunkScore = Math.max(0, 100 - trunkVal * 2);
        const matchPct = (armScore + legScore + trunkScore * 1.5) / 3.5;
        if (matchPct > bestMatchRef.current && lastKpRef.current) {
          bestMatchRef.current = matchPct;
          if (matchPct >= 60) {
            bestPoseRef.current = {
              landmarks: lastKpRef.current.landmarks,
              source_frame: lastKpRef.current.source_frame,
              angle: matchPct,
              capturedAtMs: performance.now(),
            };
          }
        }
        return next;
      });
    },
    [combo],
  );

  const buildRehabPayload = useCallback(() => {
    if (!combo) return null;
    const bestMatch = bestMatchRef.current;
    const interpretation =
      `Bird-Dog session — best pose match ${bestMatch.toFixed(0)}% (arm ${combo.armSide}, leg ${combo.legSide}).`;
    const skeletonPose = buildSkeletonPosePayload(
      bestPoseRef.current,
      lastKpRef.current,
      bestMatch,
      combo.legSide,
      `Best bird-dog match — ${bestMatch.toFixed(0)}%`,
    );
    return {
      module: "rehab" as const,
      movement: "bird-dog",
      side: combo.legSide,
      metrics: {
        exercise_slug: "bird-dog",
        mechanic_id: "match_pose",
        started_at_ms: sessionStartRef.current,
        duration_sec: elapsedSecondsSince(sessionStartRef.current),
        score: { points: 0, streak: 0, bestStreak: 0 },
        mechanic_state: {
          bestMatchPct: bestMatch,
          finalAngles: currentAngles,
        },
        signal: {
          name: "match_pct",
          unit: "%",
          value_at_peak: bestMatch,
          target_band: { min: 70, max: 100 },
        },
        combo: { armSide: combo.armSide, legSide: combo.legSide },
        config: BIRD_DOG_CONFIG,
        level_index: DEFAULT_LEVEL_INDEX,
        skeleton_pose: skeletonPose,
      },
      observations: { interpretation },
    };
  }, [combo, currentAngles]);

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div className="max-w-2xl">
              <Badge>B4 · Rehab game</Badge>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight md:text-5xl">
                Bird-Dog<span className="text-accent">.</span>
              </h1>
              <p className="mt-5 text-lg text-muted">
                Core-stability + posterior-chain coordination drill —
                quadruped position, extend ONE arm forward + the
                OPPOSITE leg backward, hold a horizontal arm-trunk-
                leg line. Three joint angles tracked simultaneously
                (arm, leg, trunk); pose-match aggregate ≥ 70 % for
                ≥ 4 s. Powered by the Match-Pose mechanic.
              </p>
              <div className="mt-5 rounded-card border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                <p className="font-semibold uppercase tracking-[0.14em] text-amber-200 text-[10px]">
                  Lateral view + opposite-limb visibility caveat
                </p>
                <p className="mt-1 text-xs leading-relaxed">
                  Bird-dog uses opposite-arm-opposite-leg, but lateral
                  camera view always has ONE of those two limbs partly
                  occluded by the body. The page handles this
                  gracefully — the score is computed only from
                  recently-visible angles, so brief occlusion doesn&apos;t
                  destroy the patient&apos;s hold. For best results, use
                  a slightly oblique (~30° off lateral) camera angle
                  so both extended limbs stay in frame.
                </p>
              </div>
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

          {!combo ? (
            <ComboPicker onPick={setCombo} />
          ) : (
            <div className="mt-10 space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-pink-500/15 px-3 py-1 text-xs font-semibold text-pink-200 ring-1 ring-pink-400/40">
                  {combo.armSide === "right" ? "Right" : "Left"} arm
                  {" + "}
                  {combo.legSide === "right" ? "Right" : "Left"} leg
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setCombo(null)}
                >
                  Change combo
                </Button>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  <RehabCameraShell
                    onFrame={handleFrame}
                    angleArc={{
                      vertex: combo.legSide === "left" ? LM_LIVE.LEFT_HIP : LM_LIVE.RIGHT_HIP,
                      armA: combo.legSide === "left" ? LM_LIVE.LEFT_SHOULDER : LM_LIVE.RIGHT_SHOULDER,
                      armB: combo.legSide === "left" ? LM_LIVE.LEFT_KNEE : LM_LIVE.RIGHT_KNEE,
                      currentDeg: currentAngles.leg ?? 0,
                      band: { min: 160, max: 200 },
                    }}
                  >
                    <div className="absolute right-3 top-3 rounded-lg border border-white/15 bg-black/70 px-3 py-2 backdrop-blur">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                        Live joint angles
                      </p>
                      <div className="tabular text-[11px] text-zinc-200 space-y-0.5">
                        <p>
                          arm{" "}
                          <span className="font-semibold text-white">
                            {(currentAngles.arm ?? 0).toFixed(0)}°
                          </span>{" "}
                          / 180
                        </p>
                        <p>
                          leg{" "}
                          <span className="font-semibold text-white">
                            {(currentAngles.leg ?? 0).toFixed(0)}°
                          </span>{" "}
                          / 180
                        </p>
                        <p>
                          trunk{" "}
                          <span className="font-semibold text-white">
                            {(currentAngles.trunk ?? 0).toFixed(0)}°
                          </span>{" "}
                          / 0
                        </p>
                      </div>
                    </div>
                  </RehabCameraShell>
                </div>

                <div>
                  <MatchPoseShell
                    currentAngles={currentAngles}
                    config={BIRD_DOG_CONFIG}
                  />
                </div>
              </div>

              <div className="no-pdf">
                <RehabSessionFooter
                  buildPayload={buildRehabPayload}
                  label="Save rehab session"
                />
              </div>
            </div>
          )}

          <div className="mt-16 rounded-card border border-border bg-surface p-5 text-sm text-muted">
            <p className="font-semibold text-foreground">Camera setup</p>
            <ul className="mt-3 list-disc space-y-1.5 pl-5">
              <li>
                Camera at quadruped height (~50 cm), ~2.5 m away,
                <strong> slightly oblique to lateral</strong> (~30°
                off) so both the extended arm AND the extended leg
                stay in frame.
              </li>
              <li>
                Set up on hands and knees — wrists under shoulders,
                knees under hips, spine neutral (trunk horizontal).
              </li>
              <li>
                Extend the chosen arm forward and the opposite leg
                backward — aim for a straight horizontal line from
                fingertips → arm → trunk → leg → toes.
              </li>
              <li>
                Hold the pose. Aggregate match ≥{" "}
                <strong>{BIRD_DOG_CONFIG.achievedThresholdPct} %</strong>{" "}
                for ≥{" "}
                <strong>
                  {(BIRD_DOG_CONFIG.requiredHoldMs / 1000).toFixed(0)} s
                </strong>{" "}
                clears the round.
              </li>
              <li>
                Per-joint tolerances are generous (±15-20°) — focus
                on the OVERALL line, not perfect angles. Trunk
                neutrality is weighted highest because spinal
                position is the clinical priority.
              </li>
            </ul>
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}

function ComboPicker({ onPick }: { onPick: (c: Combo) => void }) {
  return (
    <div className="mt-10 max-w-xl">
      {REHAB_EXERCISE_IMAGES["bird-dog"] && (
        <div className="mb-6 mx-auto max-w-md overflow-hidden rounded-md border border-border bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={REHAB_EXERCISE_IMAGES["bird-dog"]}
            alt=""
            aria-hidden="true"
            loading="lazy"
            className="block w-full object-contain"
            style={{ maxHeight: 240 }}
          />
        </div>
      )}
      <h2 className="text-2xl font-semibold tracking-tight">
        Choose the arm + leg combo
      </h2>
      <p className="mt-2 text-sm text-muted">
        Bird-dog uses OPPOSITE arm + leg. Pick which combo the
        patient is doing — the system tracks those limbs&apos;
        angles + the trunk for the pose-match score.
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <Button onClick={() => onPick(COMBO_RIGHT_LEFT)}>
          Right arm + Left leg
        </Button>
        <Button onClick={() => onPick(COMBO_LEFT_RIGHT)}>
          Left arm + Right leg
        </Button>
      </div>
    </div>
  );
}
