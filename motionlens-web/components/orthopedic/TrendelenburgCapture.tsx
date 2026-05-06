"use client";
// Trendelenburg test capture flow.
//
// State machine:
//   idle → ready_left → recording_left → done_left →
//          ready_right → recording_right → done_right →
//          done
//
// During `recording_*`:
//   - Sample pelvic-tilt + trunk-lean at 10 Hz
//   - Save per-frame keypoints
//   - Track running max-drop and capture a screenshot at the moment
//     of max drop
//   - Auto-terminate if the lifted foot returns to the ground OR
//     pelvic tilt spikes past PELVIC_SPIKE_TERMINATION_DEG
//   - End normally after TARGET_HOLD_SECONDS (30s)

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Play, RotateCcw } from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { TrendelenburgLiveCamera } from "@/components/orthopedic/TrendelenburgLiveCamera";
import { TrendelenburgReport } from "@/components/orthopedic/TrendelenburgReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  COMPENSATORY_TRUNK_LEAN_DEG,
  PELVIC_SPIKE_TERMINATION_DEG,
  SAMPLE_INTERVAL_MS,
  TARGET_HOLD_SECONDS,
  buildInterpretation,
  computePelvicTilt,
  computeTrunkLean,
  detectStanceSide,
  dropForStance,
  summarizeSide,
  type Side,
  type TrendelenburgFrameSample,
  type TrendelenburgFullResult,
  type TrendelenburgSideResult,
} from "@/lib/orthopedic/trendelenburg";

type Phase =
  | "idle"
  | "ready"
  | "recording"
  | "done";

// Once the patient has reached single-leg stance, a brief loss of
// the detected stance (foot wobble) is allowed before the test
// terminates as a foot-touch. Was 1s; bumped to 2s so a momentary
// dip doesn't end the recording.
const POST_LIFT_LOSS_GRACE_SEC = 2.0;
// Hard ceiling on "armed but never lifted" — if the patient hasn't
// achieved stance after this many seconds we still terminate so the
// UI doesn't sit indefinitely.
const PRE_LIFT_TIMEOUT_SEC = 12.0;

interface RecordingState {
  side: Side;
  startedAt: number;
  samples: TrendelenburgFrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  lastSampleAt: number;
  maxDropSoFar: number;
  peakScreenshotDataUrl: string | null;
  /** Set the first time `detectStanceSide` returns the expected side.
   *  Determines whether subsequent stance loss is "foot wobble" (post-
   *  lift, terminate) or "still getting ready" (pre-lift, wait). */
  firstStanceAt: number | null;
  /** Most recent timestamp at which `detectStanceSide` returned a
   *  non-null value (used for the post-lift loss grace window). */
  lastStanceSeenAt: number | null;
}

export function TrendelenburgCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrendelenburgFullResult>({ left: null, right: null });
  const [now, setNow] = useState<number>(0); // re-render driver for the countdown
  // Live coaching message shown to the operator + patient during the
  // hold. Updated from `handleFrame` only when the message changes
  // (otherwise React would re-render on every detection tick).
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachMsgRef = useRef<string>("");

  // Active-recording state lives in a ref because the per-frame
  // callback fires from a rAF loop and can't depend on React state
  // closures without going stale.
  const recordingRef = useRef<RecordingState | null>(null);
  // Which side(s) are still pending so the user can do them in
  // any order. After one side completes, the OTHER side is offered.
  const [completedSides, setCompletedSides] = useState<Set<Side>>(new Set());
  // Currently-armed side (set when user clicks "Start <side>-leg stance").
  const [armedSide, setArmedSide] = useState<Side | null>(null);

  // 250 ms tick for the live UI (countdown timer + current-tilt readout).
  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, [phase]);

  const finishSide = useCallback((termination: TrendelenburgSideResult["termination"]) => {
    const rec = recordingRef.current;
    if (!rec) return;
    const summary = summarizeSide(
      rec.side,
      rec.startedAt,
      Date.now(),
      termination,
      rec.samples,
      rec.keypoints,
      rec.peakScreenshotDataUrl,
    );
    setResult((prev) => ({ ...prev, [rec.side]: summary }));
    setCompletedSides((prev) => {
      const next = new Set(prev);
      next.add(rec.side);
      return next;
    });
    recordingRef.current = null;
    setArmedSide(null);
    setCoachMsg("");
    lastCoachMsgRef.current = "";
    // If both sides done → render report. Else go back to ready.
    setPhase((prevPhase) => {
      const bothDone = completedSidesIncluding(prevPhase, rec.side);
      return bothDone ? "done" : "ready";
    });
  }, []);

  // Helper: check whether THE OTHER side has already been recorded
  // by inspecting the current `result` state at the moment of the
  // finishSide call. Captured via a separate ref to avoid a stale
  // closure.
  const resultRef = useRef(result);
  useEffect(() => { resultRef.current = result; }, [result]);

  function completedSidesIncluding(_prev: Phase, justFinished: Side): boolean {
    const r = resultRef.current;
    const otherDone = justFinished === "left" ? r.right !== null : r.left !== null;
    return otherDone;
  }

  // Helper: publish a coaching message to the UI without spamming
  // setState on every frame.
  const setCoachIfChanged = useCallback((msg: string) => {
    if (lastCoachMsgRef.current === msg) return;
    lastCoachMsgRef.current = msg;
    setCoachMsg(msg);
  }, []);

  // ── Per-frame callback from the live camera ────────────────────
  const handleFrame = useCallback((kp: Keypoint[], video: HTMLVideoElement) => {
    // PRE-RECORD: just show coaching info — no recording in this branch.
    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    const expectedSide = rec.side;
    const expectedStr = expectedSide === "left" ? "left" : "right";
    const liftedStr   = expectedSide === "left" ? "right" : "left";
    const elapsedSec  = (tNow - rec.startedAt) / 1000;

    // Auto-terminate guards — but only AFTER stance has been achieved
    // at least once. Before that we sit in a "waiting for lift" state
    // and surface a coaching prompt instead of cutting the test short.
    const detectedStance = detectStanceSide(kp);

    if (detectedStance === expectedSide) {
      // Good stance. Mark first-detection + last-detection.
      if (rec.firstStanceAt === null) rec.firstStanceAt = tNow;
      rec.lastStanceSeenAt = tNow;
    } else {
      // Either no detection, or the wrong leg is on the ground.
      if (rec.firstStanceAt === null) {
        // Pre-lift: never reached stance yet. Coach the patient and
        // give them up to PRE_LIFT_TIMEOUT_SEC before bailing.
        if (detectedStance === null) {
          setCoachIfChanged(`Lift your ${liftedStr} leg — keep the ${expectedStr} foot planted.`);
        } else {
          setCoachIfChanged(`Switch sides — please stand on the ${expectedStr} leg.`);
        }
        if (elapsedSec > PRE_LIFT_TIMEOUT_SEC) {
          finishSide("foot_touch");
        }
        return; // don't sample / advance until the lift is detected
      }
      // Post-lift: stance was achieved before — this is a foot wobble
      // or full touch-down. Allow POST_LIFT_LOSS_GRACE_SEC of slack
      // before terminating.
      const sinceLastStance = rec.lastStanceSeenAt
        ? (tNow - rec.lastStanceSeenAt) / 1000
        : 0;
      if (sinceLastStance > POST_LIFT_LOSS_GRACE_SEC) {
        finishSide("foot_touch");
        return;
      }
      setCoachIfChanged(
        detectedStance === null
          ? "Stay up — keep that leg lifted."
          : `Wrong leg on the ground — return to standing on the ${expectedStr} leg.`,
      );
    }

    const pelvic = computePelvicTilt(kp);
    const lean = computeTrunkLean(kp);

    if (pelvic !== null && Math.abs(pelvic) > PELVIC_SPIKE_TERMINATION_DEG) {
      finishSide("spike");
      return;
    }

    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      pelvic_tilt_deg: pelvic,
      trunk_lean_deg: lean,
    });
    rec.keypoints.push(
      kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })),
    );

    // Track max drop for the peak-frame screenshot.
    if (pelvic !== null) {
      const drop = dropForStance(pelvic, rec.side);
      if (drop > rec.maxDropSoFar) {
        rec.maxDropSoFar = drop;
        const grab = (window as unknown as {
          __trendelenburgCapture?: () => string | null;
        }).__trendelenburgCapture;
        if (grab) {
          const url = grab();
          if (url) rec.peakScreenshotDataUrl = url;
        }
      }
    }

    // Live coaching while in good stance — based on the current pelvic
    // drop magnitude.
    if (rec.firstStanceAt !== null && pelvic !== null) {
      const drop = dropForStance(pelvic, rec.side);
      const absLean = lean !== null ? Math.abs(lean) : 0;
      if (drop > 5) {
        setCoachIfChanged("Pelvic drop is high — try to level your hips.");
      } else if (drop > 2) {
        setCoachIfChanged("Slight pelvic drop — engage the stance-side hip.");
      } else if (absLean > COMPENSATORY_TRUNK_LEAN_DEG) {
        setCoachIfChanged("Trunk leaning — try to keep the upper body upright.");
      } else {
        setCoachIfChanged("Holding well — stay steady.");
      }
    }

    // Voluntary 30-second completion.
    if (elapsedSec >= TARGET_HOLD_SECONDS) {
      finishSide("completed");
    }

    // Avoid TS "video unused" warning — we actually only need the
    // <video> reference for the parent's screenshot path, which goes
    // through the global captureFrame helper exposed by the camera.
    void video;
  }, [phase, finishSide, setCoachIfChanged]);

  // ── Arming + start ─────────────────────────────────────────────
  function arm(side: Side) {
    setError(null);
    setArmedSide(side);
    setPhase("ready");
  }

  function startRecording() {
    if (!armedSide) return;
    recordingRef.current = {
      side: armedSide,
      startedAt: Date.now(),
      samples: [],
      keypoints: [],
      lastSampleAt: 0,
      maxDropSoFar: 0,
      peakScreenshotDataUrl: null,
      firstStanceAt: null,
      lastStanceSeenAt: null,
    };
    lastCoachMsgRef.current = "";
    setCoachMsg(`Lift your ${armedSide === "left" ? "right" : "left"} leg to begin.`);
    setPhase("recording");
  }

  function stopEarly() {
    finishSide("foot_touch");
  }

  function reset() {
    recordingRef.current = null;
    setResult({ left: null, right: null });
    setCompletedSides(new Set());
    setArmedSide(null);
    setCoachMsg("");
    lastCoachMsgRef.current = "";
    setPhase("idle");
    setError(null);
  }

  // ── Done view ──────────────────────────────────────────────────
  if (phase === "done") {
    const interpretation = buildInterpretation(result);
    return (
      <div className="space-y-8">
        <TrendelenburgReport
          patientName={patient?.name ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "trendelenburg",
            // Per-frame keypoints + samples + screenshot data-URL all
            // live inside each per-side TrendelenburgSideResult on the
            // metrics blob (PDF Section 2 (a) compliance). The top-
            // level `keypoints` field is reserved for single-snapshot
            // modules (posture) and is left unset here.
            metrics: {
              left:  result.left,
              right: result.right,
            },
            observations: { interpretation },
          })}
        />

        <div className="flex justify-center border-t border-border pt-6">
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
            Run again
          </Button>
        </div>
      </div>
    );
  }

  // ── Capture view ───────────────────────────────────────────────
  const elapsed =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remaining = Math.max(0, TARGET_HOLD_SECONDS - elapsed);
  const liveSide = recordingRef.current?.side ?? armedSide ?? null;
  const sidesRemaining: Side[] = (["left", "right"] as Side[]).filter(
    (s) => !completedSides.has(s),
  );

  return (
    <div className="space-y-6">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      <div className="rounded-card border border-accent/30 bg-accent/5 p-4 text-sm">
        <p className="font-medium text-foreground">
          Trendelenburg single-leg-stance test
        </p>
        <p className="mt-1 text-muted">
          Patient stands barefoot facing the camera, both hips visible.
          Lift one leg by flexing hip + knee to roughly 90° and hold for{" "}
          {TARGET_HOLD_SECONDS} seconds. The system records pelvic tilt
          and trunk lean for both sides.
        </p>
      </div>

      <TrendelenburgLiveCamera onFrame={handleFrame} onError={setError} />

      {/* Status / control panel */}
      {phase === "recording" && recordingRef.current && (
        <div className="rounded-card border border-accent/40 bg-accent/5 p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-foreground">
              {recordingRef.current.firstStanceAt === null
                ? `Waiting for stance — ${liveSide === "left" ? "Left" : "Right"}-leg test`
                : `Recording — ${liveSide === "left" ? "Left" : "Right"}-leg stance`}
            </p>
            <p className="tabular text-2xl font-semibold text-accent">
              {recordingRef.current.firstStanceAt === null
                ? "—"
                : `${remaining.toFixed(1)}s`}
            </p>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
            <div
              className="h-full bg-accent transition-all"
              style={{
                width: `${Math.min(
                  100,
                  recordingRef.current.firstStanceAt === null
                    ? 0
                    : (elapsed / TARGET_HOLD_SECONDS) * 100,
                )}%`,
              }}
            />
          </div>

          {/* Live coaching message — updates as stance / drop changes */}
          {coachMsg && (
            <p className="mt-3 rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
              {coachMsg}
            </p>
          )}

          <p className="mt-3 text-xs text-muted">
            Recording auto-stops if the lifted foot touches down for
            longer than {POST_LIFT_LOSS_GRACE_SEC.toFixed(0)} s, or if
            pelvic tilt exceeds {PELVIC_SPIKE_TERMINATION_DEG}°.
            Compensatory trunk lean beyond {COMPENSATORY_TRUNK_LEAN_DEG}°
            toward the stance side will be flagged on the report.
          </p>
          <div className="mt-3 flex justify-end">
            <Button variant="ghost" size="sm" onClick={stopEarly}>
              Stop early
            </Button>
          </div>
        </div>
      )}

      {phase !== "recording" && (
        <div className="rounded-card border border-border bg-surface p-5">
          {sidesRemaining.length === 0 ? (
            <p className="text-sm text-muted">
              Both sides recorded. Compiling the report…
            </p>
          ) : armedSide ? (
            <div className="space-y-3">
              <p className="text-sm">
                Ready to record:{" "}
                <span className="font-medium text-foreground">
                  {armedSide === "left" ? "Left" : "Right"}-leg stance
                </span>
                .
              </p>
              <p className="text-xs text-muted">
                The patient should be standing on the{" "}
                {armedSide === "left" ? "left" : "right"} leg with the{" "}
                opposite leg lifted. Click <em>Start hold</em> when ready;
                a {TARGET_HOLD_SECONDS}-second timer will begin.
              </p>
              <div className="flex gap-2">
                <Button onClick={startRecording}>
                  <Play className="h-4 w-4" />
                  Start hold ({armedSide})
                </Button>
                <Button variant="ghost" onClick={() => setArmedSide(null)}>
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm font-medium">
                Choose which side to record next:
              </p>
              <div className="flex flex-wrap gap-3">
                {sidesRemaining.map((s) => (
                  <Button key={s} onClick={() => arm(s)}>
                    {s === "left" ? "Left" : "Right"}-leg stance
                  </Button>
                ))}
              </div>
              {completedSides.size > 0 && (
                <p className="inline-flex items-center gap-1 text-xs text-emerald-600">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {completedSides.size === 1
                    ? "1 side recorded"
                    : "Both sides recorded"}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}
    </div>
  );
}
