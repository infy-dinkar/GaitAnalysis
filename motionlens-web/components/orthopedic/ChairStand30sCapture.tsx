"use client";
// 30-Second Chair Stand capture flow.
//
// Single-trial, timer-driven test (NOT rep-driven like 5xSTS).
// State machine: idle → armed → recording (30 s timer) → done.
// The trial only ends when the timer expires (or the operator stops
// early). Reps continue to be detected throughout, but a partial
// rep at the moment of timer expiry does NOT count toward the total.
//
// Reuses the per-frame math + sit↔stand detector from
// `lib/orthopedic/chairStand30s.ts` (which mirrors 5xSTS but has
// different aggregation + classification rules).

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  FileVideo,
  Loader2,
  Play,
  RotateCcw,
  Upload,
  Video,
} from "lucide-react";
import type { Keypoint } from "@tensorflow-models/pose-detection";

import { Button } from "@/components/ui/Button";
import { ChairStand30sLiveCamera } from "@/components/orthopedic/ChairStand30sLiveCamera";
import { ChairStand30sReport } from "@/components/orthopedic/ChairStand30sReport";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { SaveToPatientButton } from "@/components/dashboard/SaveToPatientButton";
import { usePatientContext } from "@/hooks/usePatientContext";
import {
  SAMPLE_INTERVAL_MS,
  TRIAL_DURATION_SEC,
  analyzeChairStand30sUpload,
  areArmsCrossed,
  buildInterpretation,
  computeHipMidY,
  computeKneeAngle,
  computeLegLengthPx,
  newRepDetector,
  stepRepDetector,
  summarizeTrial,
  type ChairStand30sResult,
  type FrameSample,
  type RepDetectorState,
  type RepMetrics,
  type Termination,
} from "@/lib/orthopedic/chairStand30s";
import type { Sex } from "@/lib/orthopedic/normsDatabase";

type Mode = "live" | "upload";
type Phase = "idle" | "armed" | "recording" | "done";
type UploadPhase = "idle" | "analyzing" | "done" | "error";

const MAX_FILE_MB = 100;
const ACCEPTED_VIDEO_TYPES = [
  "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
];

function errorMessage(e: unknown): string | null {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  return null;
}

interface RecordingState {
  startedAt: number;
  lastSampleAt: number;
  samples: FrameSample[];
  keypoints: Array<Array<{ x: number; y: number; score?: number }>>;
  reps: RepMetrics[];
  detector: RepDetectorState;
  armsUncrossedSeen: boolean;
  prevSitMs: number;
  deepestKneeSoFar: number;
  lastRepScreenshot: string | null;
}

export function ChairStand30sCapture() {
  const { isDoctorFlow, patient } = usePatientContext();

  const [mode, setMode] = useState<Mode>("live");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ChairStand30sResult | null>(null);
  const [now, setNow] = useState<number>(0);
  const [coachMsg, setCoachMsg] = useState<string>("");
  const lastCoachRef = useRef<string>("");

  const recordingRef = useRef<RecordingState | null>(null);

  useEffect(() => {
    if (phase !== "recording") return;
    const id = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [phase]);

  const setCoachIfChanged = useCallback((msg: string) => {
    if (lastCoachRef.current === msg) return;
    lastCoachRef.current = msg;
    setCoachMsg(msg);
  }, []);

  // Snapshot the patient's age + sex at trial-end so the saved
  // report carries demographics even if the patient profile is
  // edited later. Falls back to null when the test is run from the
  // public flow (no patient context).
  function snapshotDemographics(): { age: number | null; sex: Sex | "other" | null } {
    if (!patient) return { age: null, sex: null };
    const sex: Sex | "other" =
      patient.gender === "male" ? "male"
      : patient.gender === "female" ? "female"
      : "other";
    return { age: patient.age, sex };
  }

  const finishTrial = useCallback((termination: Termination) => {
    const rec = recordingRef.current;
    if (!rec) return;
    // Fallback screenshot — same pattern as the other tests so a
    // saved report always has at least one frame.
    if (!rec.lastRepScreenshot) {
      const grab = (window as unknown as {
        __chairStand30sCapture?: () => string | null;
      }).__chairStand30sCapture;
      if (grab) {
        const url = grab();
        if (url) rec.lastRepScreenshot = url;
      }
    }
    const demo = snapshotDemographics();
    const summary = summarizeTrial(
      rec.startedAt,
      Date.now(),
      termination,
      rec.reps,
      rec.armsUncrossedSeen,
      rec.samples,
      rec.keypoints,
      rec.lastRepScreenshot,
      demo.age,
      demo.sex,
    );
    setResult(summary);
    recordingRef.current = null;
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("done");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient]);

  // Per-frame callback ----------------------------------------------
  const handleFrame = useCallback((kp: Keypoint[], _video: HTMLVideoElement) => {
    if (phase !== "recording" || !recordingRef.current) return;

    const rec = recordingRef.current;
    const tNow = Date.now();
    if (tNow - rec.lastSampleAt < SAMPLE_INTERVAL_MS) return;
    rec.lastSampleAt = tNow;

    const elapsed = (tNow - rec.startedAt) / 1000;
    if (elapsed >= TRIAL_DURATION_SEC) {
      finishTrial("completed");
      return;
    }

    const hipMidY = computeHipMidY(kp);
    const kneeAngle = computeKneeAngle(kp);
    const armsOk = areArmsCrossed(kp);

    if (rec.detector.legLengthPx === null) {
      rec.detector.legLengthPx = computeLegLengthPx(kp);
    }

    rec.samples.push({
      t_ms: tNow - rec.startedAt,
      hip_mid_y: hipMidY,
      knee_angle_deg: kneeAngle,
      arms_crossed: armsOk,
    });
    rec.keypoints.push(kp.map((p) => ({ x: p.x, y: p.y, score: p.score ?? 0 })));

    if (kneeAngle !== null && kneeAngle < rec.deepestKneeSoFar) {
      rec.deepestKneeSoFar = kneeAngle;
      const grab = (window as unknown as {
        __chairStand30sCapture?: () => string | null;
      }).__chairStand30sCapture;
      if (grab) {
        const url = grab();
        if (url) rec.lastRepScreenshot = url;
      }
    }

    if (!armsOk && !rec.armsUncrossedSeen) {
      rec.armsUncrossedSeen = true;
    }

    const tMsRel = tNow - rec.startedAt;
    const stepResult = stepRepDetector(rec.detector, hipMidY, kneeAngle, tMsRel);

    if (rec.armsUncrossedSeen) {
      setCoachIfChanged("Arms uncrossed — keep both hands across the chest.");
    } else {
      const remaining = Math.max(0, TRIAL_DURATION_SEC - elapsed);
      setCoachIfChanged(
        `Reps so far: ${rec.reps.length}. ${remaining.toFixed(0)} s remaining — keep going.`,
      );
    }

    if (stepResult.completedRep) {
      const sitEvents = rec.detector.sitEvents;
      const startMs = sitEvents[sitEvents.length - 2] ?? 0;
      const endMs   = sitEvents[sitEvents.length - 1];
      const duration = (endMs - startMs) / 1000;
      const newRep: RepMetrics = {
        rep_index: rec.reps.length + 1,
        duration_seconds: duration,
        min_knee_angle_deg: rec.detector.currentMinKneeAngle,
      };
      rec.reps.push(newRep);
      rec.prevSitMs = endMs;
      rec.detector.currentMinKneeAngle = 180;
    }
  }, [phase, finishTrial, setCoachIfChanged]);

  // ── Upload-mode state ──────────────────────────────────────────
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [filePickError, setFilePickError] = useState<string | null>(null);

  function handleFilePick(file: File | null) {
    setFilePickError(null);
    if (!file) {
      setUploadFile(null);
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFilePickError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_MB} MB.`);
      return;
    }
    if (file.type && !ACCEPTED_VIDEO_TYPES.includes(file.type)) {
      setFilePickError(`Unsupported file type (${file.type}). Use MP4, WebM, MOV, or MKV.`);
      return;
    }
    setUploadFile(file);
  }

  async function analyzeUpload() {
    if (!uploadFile) return;
    setUploadPhase("analyzing");
    setUploadProgress(0);
    setUploadError(null);
    setError(null);
    const demo = snapshotDemographics();
    try {
      const trialResult = await analyzeChairStand30sUpload(
        uploadFile, demo.age, demo.sex, setUploadProgress,
      );
      setResult(trialResult);
      setUploadPhase("done");
    } catch (e) {
      setUploadError(errorMessage(e) ?? "Analysis failed.");
      setUploadPhase("error");
    }
  }

  function resetUpload() {
    setUploadFile(null);
    setUploadProgress(0);
    setUploadError(null);
    setFilePickError(null);
    setUploadPhase("idle");
    setResult(null);
    setError(null);
  }

  function switchMode(next: Mode) {
    if (next === mode) return;
    if (phase === "recording" || uploadPhase === "analyzing") return;
    recordingRef.current = null;
    setResult(null);
    setCoachMsg("");
    lastCoachRef.current = "";
    setPhase("idle");
    setError(null);
    resetUpload();
    setMode(next);
  }

  function arm() {
    setError(null);
    setPhase("armed");
    setCoachMsg(
      "Patient should be seated, back against the chair, feet flat, arms crossed at chest. " +
      "Click Start when ready — 30-second timer begins immediately and patient should stand up.",
    );
  }

  function startRecording() {
    setError(null);
    recordingRef.current = {
      startedAt: Date.now(),
      lastSampleAt: 0,
      samples: [],
      keypoints: [],
      reps: [],
      detector: newRepDetector(),
      armsUncrossedSeen: false,
      prevSitMs: 0,
      deepestKneeSoFar: 180,
      lastRepScreenshot: null,
    };
    lastCoachRef.current = "";
    setCoachMsg("Stand up to start — as many full sit-to-stand cycles as possible in 30 seconds.");
    setPhase("recording");
  }

  function stopEarly() {
    finishTrial("stopped");
  }

  function reset() {
    recordingRef.current = null;
    setResult(null);
    setPhase("idle");
    setError(null);
    setCoachMsg("");
    lastCoachRef.current = "";
  }

  // Done view ---------------------------------------------------------
  const isLiveDone = phase === "done" && result !== null;
  const isUploadDone = uploadPhase === "done" && result !== null;
  if ((isLiveDone || isUploadDone) && result) {
    const interpretation = buildInterpretation(result);
    const onRunAgain = isUploadDone ? () => { resetUpload(); } : reset;
    return (
      <div className="space-y-8">
        <ChairStand30sReport
          patientName={patient?.name ?? null}
          patient={patient ?? null}
          result={result}
          interpretation={interpretation}
        />

        <SaveToPatientButton
          buildPayload={() => ({
            module: "chair_stand_30s",
            metrics: { trial: result },
            observations: { interpretation },
          })}
        />

        <div className="flex justify-center border-t border-border pt-6">
          <Button variant="secondary" onClick={onRunAgain}>
            <RotateCcw className="h-4 w-4" />
            Run again
          </Button>
        </div>
      </div>
    );
  }

  // Capture view ------------------------------------------------------
  const elapsedSec =
    phase === "recording" && recordingRef.current
      ? (now - recordingRef.current.startedAt) / 1000
      : 0;
  const remainingSec = Math.max(0, TRIAL_DURATION_SEC - elapsedSec);
  const repsSoFar = recordingRef.current?.reps.length ?? 0;
  const armsFlag = recordingRef.current?.armsUncrossedSeen ?? false;

  // Patient demographics warning — surface BEFORE the trial so the
  // operator can fill them in if missing. Generic norms are still
  // applied if they don't, but the report flags it.
  const demographicsMissing =
    !!patient && (patient.age === null || patient.age === undefined ||
                   patient.gender === null || patient.gender === undefined);

  const modeSwitchDisabled =
    phase === "recording" || uploadPhase === "analyzing";

  return (
    <div className="space-y-10">
      {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

      {/* ─── Mode toggle: Live Camera vs Upload Video ───────────── */}
      <div className="inline-flex rounded-card border border-border bg-surface p-1">
        <button
          type="button"
          onClick={() => switchMode("live")}
          disabled={modeSwitchDisabled}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "live"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${modeSwitchDisabled ? "opacity-50" : ""}`}
        >
          <Camera className="h-4 w-4" />
          Live camera
        </button>
        <button
          type="button"
          onClick={() => switchMode("upload")}
          disabled={modeSwitchDisabled}
          className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
            mode === "upload"
              ? "bg-accent text-white shadow-sm"
              : "text-muted hover:text-foreground"
          } ${modeSwitchDisabled ? "opacity-50" : ""}`}
        >
          <Upload className="h-4 w-4" />
          Upload video
        </button>
      </div>

      {/* ─── UPLOAD MODE ────────────────────────────────────────── */}
      {mode === "upload" && (
        <div className="space-y-6">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Setup checklist
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Sturdy chair without armrests, back against the backrest, feet flat.",
                "Arms crossed over the chest throughout the test.",
                "Camera to the SIDE of the patient (lateral view).",
                "Record the patient performing 30 seconds of sit-to-stand cycles.",
                "Trim the clip to start with the patient seated and end at the 30-second mark.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Upload video
            </p>
            <p className="mt-1 text-xs text-muted">
              MP4, WebM, MOV, or MKV · max {MAX_FILE_MB} MB
            </p>

            {!uploadFile && uploadPhase === "idle" && (
              <label className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-card border border-dashed border-border bg-elevated p-8 text-center transition hover:border-accent/60">
                <FileVideo className="h-10 w-10 text-muted" />
                <p className="text-sm font-medium text-foreground">Choose a video file</p>
                <input
                  type="file"
                  accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
                  className="hidden"
                  onChange={(e) => handleFilePick(e.target.files?.[0] ?? null)}
                />
              </label>
            )}

            {uploadFile && uploadPhase !== "analyzing" && (
              <div className="mt-3 flex items-center gap-3 rounded-md bg-elevated p-3 text-sm">
                <Video className="h-5 w-5 shrink-0 text-accent" />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-foreground">{uploadFile.name}</p>
                  <p className="text-xs text-muted">
                    {(uploadFile.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
                <button type="button" onClick={() => handleFilePick(null)} className="text-xs text-muted hover:text-error">
                  remove
                </button>
              </div>
            )}

            {filePickError && (
              <div className="mt-3 flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-3 text-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                <p className="text-foreground">{filePickError}</p>
              </div>
            )}

            {uploadPhase === "idle" && uploadFile && (
              <div className="mt-4">
                <Button onClick={analyzeUpload}>
                  <Upload className="h-4 w-4" />
                  Analyse video
                </Button>
              </div>
            )}

            {uploadPhase === "analyzing" && uploadFile && (
              <div className="mt-4 space-y-2">
                <div className="flex items-center gap-3 rounded-md bg-elevated p-3 text-sm">
                  <Video className="h-5 w-5 shrink-0 text-accent" />
                  <p className="min-w-0 flex-1 truncate font-medium text-foreground">{uploadFile.name}</p>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div className="h-full bg-accent transition-all" style={{ width: `${Math.min(100, Math.max(0, uploadProgress))}%` }} />
                </div>
                <p className="inline-flex items-center gap-1.5 text-xs text-muted">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Analysing — {Math.round(uploadProgress)}%
                </p>
              </div>
            )}

            {uploadPhase === "error" && (
              <div className="mt-4 space-y-3">
                <div className="flex items-start gap-3 rounded-md border border-error/40 bg-error/5 p-3 text-sm">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                  <p className="text-foreground">{uploadError ?? "Analysis failed."}</p>
                </div>
                <Button variant="secondary" onClick={resetUpload}>
                  <RotateCcw className="h-4 w-4" />
                  Try again
                </Button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ─── LIVE MODE (unchanged behaviour) ─────────────────────── */}
      {mode === "live" && (
      <>
      {/* Demographics / public-mode warnings stay full-width above the
          grid since they're conditional setup blockers, not part of
          the active capture flow. */}
      {!isDoctorFlow && (
        <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
          <p className="font-medium text-foreground">Demo / public mode</p>
          <p className="mt-1 text-muted">
            No patient context — the report will use a generic CDC threshold
            (men 11, women 10). For an accurate norm comparison, run this
            test from a patient profile in the dashboard.
          </p>
        </div>
      )}
      {isDoctorFlow && demographicsMissing && (
        <div className="rounded-card border border-warning/40 bg-warning/5 p-4 text-sm">
          <p className="font-medium text-foreground">Patient demographics incomplete</p>
          <p className="mt-1 text-muted">
            Age and/or sex are missing on the patient profile. The report will
            fall back to a generic CDC threshold and flag the comparison as
            non-comparable.
          </p>
        </div>
      )}

      {/* ─── 2-column layout (instructions+status | camera) ─────── */}
      <div className="grid items-start gap-8 lg:grid-cols-[2fr_3fr]">
        {/* LEFT — instructions + controls */}
        <div className="space-y-5">
          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Movement instructions
            </p>
            <ol className="mt-3 space-y-2.5 text-sm text-foreground">
              {[
                "Sit on a sturdy chair with no armrests. Back against the backrest.",
                "Turn so the camera sees your side (sideways / side view).",
                "Both feet flat on the floor, shoulder-width apart.",
                "Cross both arms over your chest — keep them crossed for the whole test.",
                `On Start, stand up fully and sit back down — keep going as fast as you safely can for ${TRIAL_DURATION_SEC} seconds.`,
                "Don't stop until the timer ends. The system counts each completed sit-to-stand cycle.",
              ].map((s, i) => (
                <li key={i} className="flex gap-2.5">
                  <span className="tabular shrink-0 text-accent">{i + 1}.</span>
                  <span className="leading-relaxed">{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="rounded-card border border-border bg-surface p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
              Live status
            </p>

            {phase === "recording" && recordingRef.current && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground">Recording</p>
                  <p className="tabular text-2xl font-semibold text-accent">
                    {repsSoFar} reps · {remainingSec.toFixed(1)}s
                  </p>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full bg-accent transition-all"
                    style={{ width: `${(elapsedSec / TRIAL_DURATION_SEC) * 100}%` }}
                  />
                </div>
                {coachMsg && (
                  <p className="rounded-md border border-accent/30 bg-background/60 px-3 py-2 text-sm font-medium text-foreground">
                    {coachMsg}
                  </p>
                )}
                {armsFlag ? (
                  <p className="inline-flex items-center gap-1 rounded-full bg-warning/10 px-2.5 py-1 text-[11px] font-medium text-warning">
                    <AlertTriangle className="h-3 w-3" />
                    Arm uncrossed during trial — will be flagged
                  </p>
                ) : (
                  <p className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-600">
                    <CheckCircle2 className="h-3 w-3" />
                    Arms crossed — good
                  </p>
                )}
                <div className="flex justify-end">
                  <Button variant="ghost" size="sm" onClick={stopEarly}>Stop early</Button>
                </div>
              </div>
            )}

            {phase !== "recording" && (
              <div className="mt-3">
                {phase === "armed" ? (
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-foreground">Ready to record</p>
                    <p className="text-xs text-muted">
                      Patient seated, back against backrest, feet flat, arms crossed at
                      chest. The {TRIAL_DURATION_SEC}-second timer starts the moment
                      you click <em>Start</em> — instruct the patient to begin standing
                      immediately and continue until the timer expires.
                    </p>
                    {coachMsg && (
                      <p className="rounded-md bg-background/40 px-3 py-2 text-sm text-foreground">
                        {coachMsg}
                      </p>
                    )}
                    <div className="flex gap-2">
                      <Button onClick={startRecording}>
                        <Play className="h-4 w-4" />
                        Start trial
                      </Button>
                      <Button variant="ghost" onClick={() => setPhase("idle")}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <p className="text-sm font-medium text-foreground">Begin a new trial</p>
                    <Button onClick={arm}>Start</Button>
                  </div>
                )}
              </div>
            )}

            <p className="mt-4 text-xs text-muted">
              Cutoffs (CDC STEADI): age- and sex-matched rep-count thresholds —
              falling below the published norm for the patient&apos;s band is a
              positive screen for fall risk. Last incomplete rep at timer expiry
              does not count.
            </p>
          </div>
        </div>

        {/* RIGHT — sticky camera */}
        <div className="lg:sticky lg:top-28">
          <ChairStand30sLiveCamera onFrame={handleFrame} onError={setError} />
          <p className="mt-3 text-xs text-subtle">
            Start the camera and have the patient seated in profile to the
            lens. The on-screen skeleton tracks the knee and hip in real
            time — keep the full body in side-view frame.
          </p>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-3 rounded-card border border-error/40 bg-error/5 p-4 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
          <p className="text-foreground">{error}</p>
        </div>
      )}
      </>
      )}
    </div>
  );
}
