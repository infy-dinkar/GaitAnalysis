"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { Play, RotateCcw, AlertCircle, Upload, Video } from "lucide-react";
import { Nav } from "@/components/layout/Nav";
import { Footer } from "@/components/layout/Footer";
import { Section } from "@/components/ui/Section";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { VideoUpload } from "@/components/analysis/VideoUpload";
import { GaitRecordCapture } from "@/components/gait/GaitRecordCapture";
import { analyzeGait } from "@/lib/api";
import { SaveStatusBanner } from "@/components/dashboard/SaveStatusBanner";
import { usePatientContext } from "@/hooks/usePatientContext";

const STORAGE_KEY = "motionlens.gait_api_result";
const HEIGHT_KEY = "motionlens.height_cm";

type Phase = "idle" | "uploading" | "analysing" | "error";
type Mode = "upload" | "record";

export default function GaitUploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  // Mode toggle — "upload" keeps the existing file-picker flow exactly
  // as before; "record" reveals the in-browser camera + recorder and
  // auto-uploads the recorded blob through the same analyzeGait
  // pipeline. Defaults to "upload" so existing users land on the
  // unchanged flow.
  const [mode, setMode] = useState<Mode>("upload");

  const { isDoctorFlow, patientId, patient } = usePatientContext();

  const onSelect = useCallback((f: File) => {
    setFile(f);
    setError(null);
    setProgress(0);
    setPhase("idle");
  }, []);

  async function run() {
    if (!file) return;
    setError(null);
    setProgress(0);
    setPhase("uploading");

    const heightStr = sessionStorage.getItem(HEIGHT_KEY);
    const heightCm = heightStr ? parseFloat(heightStr) : (patient?.height_cm ?? 170);
    const patientName = patient?.name?.trim() || null;

    try {
      const res = await analyzeGait(
        { video: file, heightCm, patientName },
        (loaded, total) => {
          setProgress(total > 0 ? loaded / total : 0);
          if (loaded >= total) setPhase("analysing");
        },
      );

      if (!res.success || !res.data) {
        setPhase("error");
        setError(res.error || "Analysis failed");
        return;
      }

      sessionStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          ...res.data,
          patient_info: {
            ...res.data.patient_info,
            name: patientName,
          },
          // Carry the source video metadata — used by the results page
          // to build the report payload when the doctor clicks "Save".
          _video_filename: file.name,
          _video_size_bytes: file.size,
        }),
      );

      // Forward patientId through to /gait/results so the results page
      // can show the doctor an explicit "Save to patient history" button
      // alongside the full charts + metrics view.
      const url = isDoctorFlow
        ? `/gait/results?patientId=${patientId}`
        : "/gait/results";
      router.push(url);
    } catch (e) {
      setPhase("error");
      setError(
        e instanceof Error
          ? `${e.message}. Please try again or contact support.`
          : "Network error",
      );
    }
  }

  // Record-mode handler. Called by <GaitRecordCapture> when the
  // operator stops the recording. Mirrors the upload-mode `run()`
  // flow but with the recorded File + the wall-clock duration so the
  // backend can repair a MediaRecorder WebM with broken duration
  // header before its FPS gate runs. We don't go through `run()` here
  // because that would require waiting for setFile() to commit before
  // reading state — passing the File explicitly is simpler and keeps
  // `run()` untouched.
  async function handleRecorded(recordedFile: File, recDurationMs: number) {
    setFile(recordedFile);
    setError(null);
    setProgress(0);
    setPhase("uploading");

    const heightStr = sessionStorage.getItem(HEIGHT_KEY);
    const heightCm = heightStr ? parseFloat(heightStr) : (patient?.height_cm ?? 170);
    const patientName = patient?.name?.trim() || null;

    try {
      const res = await analyzeGait(
        {
          video: recordedFile,
          heightCm,
          patientName,
          recordingDurationMs: recDurationMs,
        },
        (loaded, total) => {
          setProgress(total > 0 ? loaded / total : 0);
          if (loaded >= total) setPhase("analysing");
        },
      );

      if (!res.success || !res.data) {
        setPhase("error");
        setError(res.error || "Analysis failed");
        return;
      }

      sessionStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          ...res.data,
          patient_info: {
            ...res.data.patient_info,
            name: patientName,
          },
          _video_filename: recordedFile.name,
          _video_size_bytes: recordedFile.size,
        }),
      );

      const url = isDoctorFlow
        ? `/gait/results?patientId=${patientId}`
        : "/gait/results";
      router.push(url);
    } catch (e) {
      setPhase("error");
      setError(
        e instanceof Error
          ? `${e.message}. Please try again or contact support.`
          : "Network error",
      );
    }
  }

  function reset() {
    setFile(null);
    setError(null);
    setProgress(0);
    setPhase("idle");
  }

  return (
    <>
      <Nav />
      <main className="flex flex-col">
        <Section className="pt-32 md:pt-40">
          <div className="flex items-start justify-between gap-4">
            <div>
              <Badge>Step 4 — capture</Badge>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
                Upload the walking clip
              </h1>
              <p className="mt-3 text-sm text-muted">
                Drop a side-on walking clip — the MotionLens engine extracts cadence,
                stride symmetry, joint kinematics, and the full gait-cycle report.
              </p>
            </div>
            <Link href="/gait">
              <Button variant="ghost" size="sm">← Back</Button>
            </Link>
          </div>

          <div className="mt-10 space-y-8">
            {isDoctorFlow && <SaveStatusBanner patient={patient} saveStatus={null} />}

            {/* Mode toggle — Upload (existing flow) vs Record (new in-
                browser camera + recorder). The progress + error UI
                further down is shared, so only the input mechanism
                differs between the two modes. */}
            <div className="inline-flex rounded-card border border-border bg-surface p-1">
              <button
                type="button"
                onClick={() => setMode("upload")}
                disabled={phase === "uploading" || phase === "analysing"}
                className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
                  mode === "upload"
                    ? "bg-accent text-white shadow-sm"
                    : "text-muted hover:text-foreground"
                }`}
              >
                <Upload className="h-4 w-4" />
                Upload video
              </button>
              <button
                type="button"
                onClick={() => setMode("record")}
                disabled={phase === "uploading" || phase === "analysing"}
                className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition ${
                  mode === "record"
                    ? "bg-accent text-white shadow-sm"
                    : "text-muted hover:text-foreground"
                }`}
              >
                <Video className="h-4 w-4" />
                Record live
              </button>
            </div>

            {mode === "upload" && <VideoUpload onSelect={onSelect} />}
            {mode === "record" && (
              <GaitRecordCapture
                onRecorded={handleRecorded}
                disabled={phase === "uploading" || phase === "analysing"}
              />
            )}

            {mode === "upload" && file && phase === "idle" && (
              <Button onClick={run}>
                <Play className="h-4 w-4" />
                Analyse on server
              </Button>
            )}

            {(phase === "uploading" || phase === "analysing") && (
              <div className="rounded-card border border-border bg-surface p-5">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-[0.12em] text-subtle">
                    {phase === "uploading" ? "Uploading video" : "Analysing on server"}
                  </span>
                  <span className="tabular text-sm text-foreground">
                    {phase === "uploading"
                      ? `${Math.round(progress * 100)}%`
                      : "running pose model + computing metrics…"}
                  </span>
                </div>
                <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-elevated">
                  <div
                    className={`h-full bg-accent transition-all ${
                      phase === "analysing" ? "animate-pulse" : "duration-200"
                    }`}
                    style={{
                      width:
                        phase === "uploading" ? `${progress * 100}%` : "100%",
                    }}
                  />
                </div>
                {phase === "analysing" && (
                  <p className="mt-3 text-xs text-muted">
                    Analysis can take 20–60 seconds depending on clip length.
                  </p>
                )}
              </div>
            )}

            {phase === "error" && error && (
              <div className="rounded-card border border-error/40 bg-error/5 p-5">
                <div className="flex items-start gap-3">
                  <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-error" />
                  <div className="text-sm">
                    <p className="font-medium text-foreground">Analysis failed</p>
                    <p className="mt-1 text-muted">{error}</p>
                    <Button variant="secondary" size="sm" className="mt-4" onClick={reset}>
                      <RotateCcw className="h-4 w-4" />
                      Try another video
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Section>
      </main>
      <Footer />
    </>
  );
}
