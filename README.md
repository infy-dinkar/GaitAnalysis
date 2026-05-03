---
title: MotionLens API
emoji: 🦴
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Markerless gait + biomech ROM API (FastAPI + MediaPipe)
---

# MotionLens

> **AI-powered movement assessment platform** — markerless gait analysis, multi-joint biomechanical range-of-motion measurement, and static posture screening from a single video or photo, accessible through a Streamlit clinical tool or a modern Next.js web app with browser-side pose detection.

🌐 **Live demo:** [https://gait-analysis123.vercel.app/](https://gait-analysis123.vercel.app/)
🔧 **Backend API:** [https://thakurdinkar-motionlens-api.hf.space/api/health](https://thakurdinkar-motionlens-api.hf.space/api/health)

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![TensorFlow.js](https://img.shields.io/badge/TF.js-MoveNet-FF6F00.svg)](https://www.tensorflow.org/js)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-orange.svg)](https://developers.google.com/mediapipe)
[![Vercel](https://img.shields.io/badge/Vercel-Frontend-black.svg)](https://gait-analysis123.vercel.app/)
[![HF Spaces](https://img.shields.io/badge/HF%20Spaces-Backend-yellow.svg)](https://thakurdinkar-motionlens-api.hf.space/api/health)
[![License](https://img.shields.io/badge/license-Educational%20Prototype-lightgrey.svg)](#disclaimer)

---

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Usage](#usage)
9. [API reference](#api-reference)
10. [Project structure](#project-structure)
11. [Pipeline — how it works](#pipeline--how-it-works)
12. [Deployment](#deployment)
13. [Troubleshooting](#troubleshooting)
14. [Disclaimer](#disclaimer)

---

## Overview

MotionLens is a **three-module movement assessment platform**:

1. **Gait analysis** — clinical-grade biomechanics (cadence, stride symmetry, joint kinematics, gait-cycle curves) from a single side-view walking video.
2. **Biomechanics** — range-of-motion assessment across **5 joints** (shoulder, neck, knee, hip, ankle) covering **18 movements** in both live-camera and video-upload modes.
3. **Posture** — static posture screening from front + side photos (head/shoulder/hip tilt, plumb-line shifts, knee alignment).

All without markers, motion-capture suits, or specialised hardware.

### Hybrid ML architecture

The Next.js production stack runs ML in **two places**, optimising each path for what it needs most:

| Mode | Where ML runs | Model | Why |
|---|---|---|---|
| Live biomech (5 joints) | **Browser** (TF.js + WebGL) | MoveNet Lightning (17 kp) | Sub-20 ms per frame, 50+ FPS, zero network |
| Biomech video upload | **Browser** (TF.js + WebGL) | MoveNet Lightning | Privacy — video never leaves device |
| Posture (front + side photo) | **Browser** (TF.js + WebGL) | MoveNet Lightning | Same |
| **Gait video upload** | **Server** (FastAPI + MediaPipe) | BlazePose (33 kp) | Needs heel + toe-index for stride detection |

The Streamlit front-end keeps the legacy server-side flow for everything (single-process clinical tool).

### Front-ends

- **Streamlit** — single-process clinical tool, dark-themed, suitable for lab use.
- **Next.js + FastAPI** — modern product UX, deployable as a SaaS (Vercel + Hugging Face Spaces).

Both UIs call the **identical** Python engine modules. The Next.js frontend additionally has TypeScript ports of the shoulder/neck/knee/hip/ankle math so live mode and biomech upload can run entirely in the browser without a network round-trip.

---

## Features

### Gait analysis

#### Spatiotemporal metrics — computed twice (whole-video **Total** + steady-state **Clean**)
- Step count, cadence (steps/min), gait symmetry (0–1)
- Average knee flexion, stride variation (CV %)
- Step length anatomically calibrated to the subject's height (m or px)
- Step timing intervals
- Torso lean (signed: + forward / − backward in walking direction)
- Ankle and heel trajectories
- Walking direction (L→R / R→L / Bidirectional with pass counts)

#### Cycle-normalised joint kinematics
- Per-stride **hip / knee / ankle** angles resampled to 0–100 % of the gait cycle
- Mean ± 1 SD curves per leg overlaid on **healthy-adult reference bands** (Perry & Burnfield, Winter, Kadaba)
- Stance-phase shading (0–60 %) + toe-off marker (60 %)
- Per-leg cycle-detection metadata (accepted / rejected by amplitude / rejected as too long / too short)

#### Robustness layers
- Direction-aware **pass segmentation** — handles bidirectional walking videos
- Anatomical **pixel-to-meter scaling** from user-supplied height (no on-screen ruler)
- Per-leg **ankle baseline correction** (auto-detected static stance, running-median fallback)
- Heel-strike **amplitude floor** — rejects foot-drag events with low heel clearance
- **Cycle duration filter** — rejects spans outside 0.7×–1.5× the leg's median stride duration
- Static-stance detection restricted to clean walking windows (prevents turning-point poisoning of the baseline)

### Biomechanics — 5-joint range-of-motion (18 movements)

| Joint | Movements | Per-side |
|---|---|---|
| **Shoulder** | Flexion · Extension · Abduction · Adduction · External Rotation · Internal Rotation | ✓ |
| **Neck** | Flexion · Extension · Lateral Flexion · Rotation | — |
| **Knee** | Flexion · Extension | ✓ |
| **Hip** | Flexion · Extension · Internal Rotation · External Rotation | ✓ |
| **Ankle** | Dorsiflexion · Plantarflexion | ✓ |

- **Live mode** — continuous-capture webcam tracking via TF.js MoveNet (browser GPU), on-canvas skeleton overlay with EMA-smoothed keypoints, current/peak/frame counters, **PiP camera preview** (toggleable), configurable side selector
- **Upload mode** — drag-and-drop a recorded clip, **fully client-side** frame-by-frame analysis using `requestVideoFrameCallback` at 4× playback speed (~5-10 sec for a 10-sec video), peak angle + status
- Reference ranges from AAOS / APTA — `good` ≥ 90 %, `fair` ≥ 75 %, `poor` < 75 % of normal target
- **Plain-language per-movement instructions** — 5-step numbered guides written for non-clinical users
- Auto-generated **Assessment Report** with patient header, results table, Plotly bar chart (measured vs normal range), clinical interpretation, educational block, and subtle disclaimer footer

### Posture — static screening from photos

- **Two-photo workflow** — patient uploads front + side view (both required)
- **Frontal-plane measurements** — head tilt, shoulder tilt, hip tilt, knee Q-angle (left + right)
- **Sagittal-plane measurements** — forward head shift, shoulder shift, hip shift, knee shift (all as % body height), trunk lean
- **Annotated images** — keypoint dots, reference plumb line, horizontal bands, badge labels rendered on canvas
- **Severity-graded findings** — OK / Mild / Notable color-coded per measurement with detail text
- 100 % browser-side — photos never leave the user's device

### UX layer

| | Streamlit UI | Next.js + FastAPI |
|---|---|---|
| Audience | Clinicians, lab use, validation | Modern product UX, frontend-as-a-service |
| Deploy | Streamlit Cloud (`packages.txt`) | Vercel + Hugging Face Spaces (Dockerfile) |
| Live biomech | streamlit-webrtc (WebRTC) | **TF.js MoveNet in browser, 50+ FPS** |
| Biomech upload | Server-side MediaPipe | **TF.js MoveNet in browser, no upload** |
| Gait upload | Server-side MediaPipe | Server-side MediaPipe (unchanged) |
| Posture | — | TF.js MoveNet, two-photo flow |
| Charts | Matplotlib + Plotly | Plotly.js |
| Theme | Dark slate medical | Dark with lime accent (clinical-tech) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                        │
│   USER (browser)                                                       │
│      │                                                                  │
│      ▼                                                                  │
│   Next.js 16 frontend (Vercel)                                         │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │  TypeScript ports (live + biomech upload + posture):          │    │
│   │  • lib/biomech/{shoulder,neck,knee,hip,ankle}.ts              │    │
│   │  • lib/posture/measurements.ts                                │    │
│   │                                                                │    │
│   │  Pose detector — TensorFlow.js + MoveNet Lightning            │    │
│   │  Runtime — WebGL (GPU acceleration via browser)               │    │
│   │  Output — 17 keypoints @ 50 FPS                               │    │
│   └──────────────────────────────────────────────────────────────┘    │
│      │                                                                  │
│      │ ONLY for /gait/upload (gait needs heel + toe)                   │
│      ▼ multipart HTTPS POST                                            │
│   FastAPI backend (Hugging Face Spaces · Docker)                       │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │  Python engines (source of truth):                            │    │
│   │  • gait_engine.py    — kinematic pipeline                     │    │
│   │  • gait_cycle.py     — heel-strike + cycle extraction         │    │
│   │  • gait_plots.py     — Plotly figure builder                  │    │
│   │  • shoulder_engine.py / neck_engine.py — ROM math             │    │
│   │  • biomech_flow.py   — orchestrator                           │    │
│   │                                                                │    │
│   │  Pose detector — MediaPipe Tasks (BlazePose, 33 keypoints)    │    │
│   │  Runtime — Python C-extensions, server CPU                    │    │
│   └──────────────────────────────────────────────────────────────┘    │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘

           Auto-deploy CI/CD
           ─────────────────
           git push main → GitHub Actions
                              │
                              ├─→ Vercel rebuild (frontend, ~2 min)
                              └─→ HF Spaces auto-push + Docker rebuild (~5 min)
```

The Python **engine modules remain the source of truth**. The FastAPI layer (`api.py`, `api_helpers.py`, `api_models.py`) only handles file I/O, JSON sanitisation, and CORS. The TypeScript shoulder/neck math (`motionlens-web/lib/biomech/*.ts`) is a **direct port** of the Python engines — both produce byte-identical results so live and upload measurements stay consistent. The Streamlit UI continues to call the Python engines directly via Python imports.

---

## Tech stack

### Backend (Python 3.11)

| Concern | Library |
|---|---|
| Pose extraction | `mediapipe.tasks.vision.PoseLandmarker` (modern Task API, IMAGE & VIDEO running modes) |
| Computer vision | `opencv-python-headless` |
| Signal processing | `numpy`, `scipy` (Savitzky-Golay smoothing, `find_peaks`, MAD filtering) |
| REST API | `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` v2 |
| Streamlit charts | `matplotlib` (Agg backend), `plotly` |
| Streamlit UI | `streamlit`, `streamlit-webrtc`, `av` |

### Frontend (Next.js 16 + TypeScript)

| Concern | Library |
|---|---|
| Framework | Next.js 16 (App Router, Turbopack), React 19, TypeScript 5 |
| Styling | Tailwind CSS v4 (CSS-first `@theme` tokens, no JS config) |
| **Pose detection** | `@tensorflow/tfjs` + `@tensorflow-models/pose-detection` (MoveNet Lightning) |
| **GPU acceleration** | WebGL backend (browser GPU — works on integrated graphics too) |
| Charts | Plotly.js via `react-plotly.js` |
| Animations | `framer-motion` |
| Icons | `lucide-react` |

### Deployment

| Component | Target | URL |
|---|---|---|
| **Frontend** | Vercel (Hobby — free) | [gait-analysis123.vercel.app](https://gait-analysis123.vercel.app/) |
| **Backend** | Hugging Face Spaces (CPU Basic — free) | [thakurdinkar-motionlens-api.hf.space](https://thakurdinkar-motionlens-api.hf.space/api/health) |
| Streamlit (legacy) | Streamlit Cloud (`packages.txt` for apt deps) | — |
| Auto-deploy CI/CD | GitHub Actions (`.github/workflows/deploy-hf.yml`) | Triggers HF rebuild on every `main` push |

📘 **Step-by-step deployment guide:** See **[DEPLOYMENT.md](DEPLOYMENT.md)** (~30 min from `git push` to live URL).

---

## Prerequisites

- **Operating system** — Windows, macOS, or Linux
- **Python 3.11** — the MediaPipe wheel is only published for 3.11 at time of writing (`.python-version` is pinned)
- **Node.js 18+** — only required to run the Next.js frontend
- **Webcam** — only required for live biomech mode
- A virtual environment is strongly recommended (conda or venv)

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Gait_Analysis
```

### 2. Python dependencies

```bash
# create + activate a virtual environment (example: conda)
conda create -n motionlens python=3.11 -y
conda activate motionlens

# install engine + Streamlit deps
pip install -r requirements.txt

# install API deps (only needed if running the FastAPI backend)
pip install -r requirements_api.txt
```

### 3. Frontend dependencies (only if running the Next.js app)

```bash
cd motionlens-web
npm install
cp .env.local.example .env.local        # PowerShell: Copy-Item .env.local.example .env.local
```

---

## Configuration

### Frontend environment

`motionlens-web/.env.local`:

```bash
# Local development
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Production example
# NEXT_PUBLIC_API_BASE_URL=https://your-space.hf.space
```

### Backend CORS

Allowed origins default to `http://localhost:3000`, `http://127.0.0.1:3000`, and `*`. Tighten for production via env var:

```bash
export MOTIONLENS_ALLOWED_ORIGINS="https://app.example.com,https://staging.example.com"
```

---

## Usage

### Option A — Streamlit (single command, single port)

```bash
streamlit run app.py
```

Open **http://localhost:8501**, upload a side-profile walking video, enter the subject's height, and watch metrics + charts populate.

### Option B — Next.js + FastAPI (modern UX, two terminals)

**Terminal 1 — backend:**

```bash
cd /path/to/Gait_Analysis
conda activate motionlens
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Verify at **http://localhost:8000/docs** — Swagger UI lists 5 endpoints.

**Terminal 2 — frontend:**

```bash
cd motionlens-web
npm run dev
```

Open **http://localhost:3000**.

> The Streamlit app and the FastAPI app share the engine modules but run as independent processes. You can run all three (Streamlit on 8501, FastAPI on 8000, Next.js on 3000) simultaneously.

### Production build (frontend)

```bash
cd motionlens-web
npm run build
npm start
```

---

## API reference

Base URL: `http://localhost:8000` (or your deployed Hugging Face Space). Auto-generated Swagger UI at `/docs`, ReDoc at `/redoc`.

### `GET /api/health`

```json
{ "status": "healthy", "service": "MotionLens API", "version": "1.0.0" }
```

### `POST /api/analyze-gait`

Multipart form-data:

| Field | Type | Default | Notes |
|---|---|---|---|
| `video` | file | — | mp4 / mov / avi / mkv / webm |
| `height_cm` | float | 170 | drives stride-length calibration |
| `patient_name` | string | — | optional, surfaced in the report header |

```bash
curl -X POST http://localhost:8000/api/analyze-gait \
  -F "video=@walk.mp4" \
  -F "height_cm=172" \
  -F "patient_name=Priya Sharma"
```

Response shape (full-feature parity with Streamlit):

```jsonc
{
  "success": true,
  "data": {
    "patient_info":         { "name", "height_cm" },
    "video_info":           { "duration_sec", "fps", "total_frames",
                              "calibration_mm_per_px",
                              "valid_passes", "frames_used",
                              "ankle_baseline_left/right/method/n_frames" },
    "walking_direction":    "Bidirectional (3 L→R, 3 R→L)",
    "metrics_total":        { "step_count", "cadence", "symmetry", "knee_peak",
                              "stride_cv", "step_length", "torso_lean", "step_time", … },
    "metrics_clean":        { …same as metrics_total + "validated_passes",
                              "video_coverage_pct" },
    "joint_angles":         { "left_knee", "right_knee", "left_hip", "right_hip",
                              "left_ankle", "right_ankle"
                              — each with peak/min/rom/mean/time_series },
    "gait_cycle_data":      { "left", "right", "normal_reference",
                              "stance_phase_pct" },
    "normalized_overview":  { "leg_angle", "knee_flexion", "hip_flexion",
                              "ankle_deflection", "time_axis" },
    "tabs_data":            { "heel_position", "step_length", "timing",
                              "torso_lean", "ankle_trajectory", "pass_segments" },
    "observations":         { "hip", "knee", "ankle", "overall", "suggestions" }
  },
  "error": null
}
```

### `POST /api/analyze-shoulder`

Multipart form-data:

| Field | Type | Default | Allowed values |
|---|---|---|---|
| `video` | file | — | — |
| `movement_type` | string | — | `flexion`, `extension`, `abduction`, `adduction`, `external_rotation`, `internal_rotation` |
| `side` | string | `right` | `left`, `right` |
| `patient_name` | string | — | optional |

```bash
curl -X POST http://localhost:8000/api/analyze-shoulder \
  -F "video=@flexion.mp4" \
  -F "movement_type=flexion" \
  -F "side=right"
```

### `POST /api/analyze-neck`

Multipart form-data:

| Field | Type | Allowed values |
|---|---|---|
| `video` | file | — |
| `movement_type` | string | `flexion`, `extension`, `lateral_flexion`, `rotation` |
| `patient_name` | string | optional |

```bash
curl -X POST http://localhost:8000/api/analyze-neck \
  -F "video=@neck_flex.mp4" \
  -F "movement_type=flexion"
```

> ⚠ The neck engine has **no per-side parameter** — `lateral_flexion` and `rotation` are computed from both ears + shoulder midline regardless of which side the patient tilts/turns toward.

### `POST /api/live/biomech-frame` *(deprecated)*

> ⚠ This endpoint exists only for backward compatibility with the original Streamlit + WebRTC pipeline. The Next.js frontend no longer calls it — all live biomech detection runs in the browser via TensorFlow.js + MoveNet. Kept in the codebase as a fallback path; will likely be removed in a future cleanup.

### Common response envelope

All POST endpoints return `{ success: bool, data: T | null, error: string | null }`. Engine-level failures (no clean walking detected, low visibility, etc.) return `success: true` with explanatory fields in `data`. Network/parse failures return `success: false` with a populated `error`.

---

## Project structure

```
Gait_Analysis/
├── app.py                    # Streamlit UI — layout, observations panel
├── biomech_flow.py           # Streamlit biomech sub-flow + reusable helpers
│                             #   _run_biomech_upload_analysis,
│                             #   _create_image_pose_landmarker,
│                             #   _ensure_pose_model_file,
│                             #   _LandmarkAdapter, _wrap_landmarks
│
├── gait_engine.py            # Master gait pipeline:
│                             #   extract_poses → build_time_series →
│                             #   segment_passes → compute_meters_per_pixel →
│                             #   compute_metrics → compute_all_features →
│                             #   interpret
├── gait_cycle.py             # Heel-strike detection, cycle extraction,
│                             # duration filter, ensemble stats
├── gait_plots.py             # Matplotlib figures + healthy-adult reference
│                             # curves (used by both Streamlit and the API)
├── shoulder_engine.py        # 6 shoulder ROM movements + visibility gate
├── neck_engine.py            # 4 cervical ROM movements + visibility gate
│
├── api.py                    # FastAPI app — 5 endpoints
│                             # + cached PoseLandmarker for live mode
├── api_helpers.py            # File I/O, NumPy/NaN sanitisation,
│                             # response formatters (Gait + Biomech)
├── api_models.py             # Pydantic v2 response models
│
├── requirements.txt          # Engines + Streamlit deps
├── requirements_api.txt      # Extra: fastapi, uvicorn, python-multipart, pydantic
├── packages.txt              # apt deps for Streamlit Cloud (libGL, ffmpeg, …)
├── .python-version           # Pinned 3.11
│
├── Dockerfile                # HF Spaces deploy (FastAPI image)
├── .dockerignore
├── README.md                 # this file
│
└── motionlens-web/           # Next.js frontend (deployed on Vercel)
    ├── app/
    │   ├── layout.tsx
    │   ├── globals.css       # Tailwind v4 @theme tokens + custom utilities
    │   ├── page.tsx          # Landing — Hero + 3 modules + use cases
    │   ├── gait/
    │   │   ├── page.tsx      # Patient + height + capture mode
    │   │   ├── upload/page.tsx
    │   │   └── results/page.tsx  # Total + Clean metrics, joint tabs,
    │   │                          # gait-cycle charts (server-backed)
    │   ├── biomech/                # 5 joints, 18 movements
    │   │   ├── page.tsx            # Joint chooser
    │   │   ├── shoulder/{page,live,upload}/page.tsx
    │   │   ├── neck/{page,live,upload}/page.tsx
    │   │   ├── knee/{page,live,upload}/page.tsx
    │   │   ├── hip/{page,live,upload}/page.tsx
    │   │   └── ankle/{page,live,upload}/page.tsx
    │   └── posture/
    │       └── page.tsx            # Front + side photo upload + report
    │       └── neck/{page,live,upload}/page.tsx
    ├── components/
    │   ├── layout/{Nav,Footer}.tsx
    │   ├── landing/{Hero,Features,ProductShowcase,UseCases}.tsx
    │   ├── ui/{Button,Card,Section,Badge,FadeIn,…}.tsx
    │   ├── analysis/{VideoUpload,ChartView,ResultsTable}.tsx
    │   ├── biomech/{PatientForm,MovementGrid,
    │   │            LiveAssessment,LiveBiomechCamera,
    │   │            ApiUploadAssessment,AssessmentReport}.tsx
    │   ├── gait/{CalibrationHeader,MetricsSection,JointTabs,
    │   │         GaitCycleSection,PlotlyChart,InfoBox}.tsx
    │   └── visuals/SkeletonHero.tsx
    ├── hooks/useCamera.ts
    ├── lib/
    │   ├── api.ts            # XHR + fetch client for FastAPI
    │   ├── utils.ts
    │   ├── biomech/{shoulder,neck,instructions}.ts
    │   ├── gait/{cycleDetection,metrics}.ts
    │   └── pose/{landmarks,mediapipe-stub}.ts
    ├── public/
    ├── next.config.ts
    └── package.json
```

---

## Pipeline — how it works

### Gait analysis — 9 stages

1. **Pose extraction** — `extract_poses()` runs MediaPipe Pose on every frame, captures `(x, y, visibility, fps, total_frames, frame_w, frame_h)`.
2. **Pre-processing** — `build_time_series()` interpolates NaN gaps, removes z-score outliers, applies Savitzky-Golay smoothing per landmark.
3. **Pass segmentation** — `segment_passes()` smooths the hip-midpoint trajectory, computes velocity, suppresses turning frames, and emits stable-direction passes with **trimmed cores** (acceleration / deceleration excluded).
4. **Anatomical scale** — `compute_meters_per_pixel()` uses `(height_cm × 0.53) / median(leg_length_px)` over stance frames inside passes.
5. **Per-frame angles** — hip, knee, ankle in **pixel** coordinates with sign convention + per-leg ankle baseline correction.
6. **Heel-strike detection** — heel-to-heel horizontal-separation peaks (for spatiotemporal metrics) + heel-y peaks with amplitude floor (for cycle plots).
7. **Metrics** — `compute_metrics()` runs once over all frames (**Total**) and once over the steady-state mask (**Clean**).
8. **Cycle assembly** — `extract_cycles` → duration filter → MAD filter → `ensemble_statistics` per leg per joint, producing length-101 mean ± SD curves.
9. **Rule-based interpretation** — `interpret()` reads `clean_metrics` and emits observation + suggestion strings.

### Live biomech — per-frame loop (Next.js, browser-only)

1. `<video>` element holds the webcam stream (`getUserMedia`, mirrored selfie view).
2. `requestAnimationFrame` loop runs at the browser's vsync rate (~60 Hz max).
3. Each tick: `tf.browser.fromPixels(video)` uploads the current frame to a WebGL texture (GPU memory) — ~2 ms.
4. **MoveNet Lightning** runs entirely on the GPU as a chain of GLSL fragment shaders (MobileNetV2 backbone → FPN → heatmap heads → argmax + offset refinement) — ~12 ms.
5. `gl.readPixels` returns 17 keypoints to CPU (~3 ms).
6. **EMA smoothing** (α = 0.4) applied per-keypoint to suppress jitter.
7. Joint angle computed by the relevant TS module (`shoulder.ts` / `neck.ts` / `knee.ts` / `hip.ts` / `ankle.ts`).
8. Skeleton drawn directly to `<canvas>` via `ctx.beginPath / fill / stroke` (bypassing React reconciliation for zero-state-churn).
9. Result handed to `LiveAssessment` via a ref — separate 10 Hz `setInterval` triggers React re-render to update Current/Peak/Frames readouts.

**Total per-frame: ~17-22 ms = sustained 45-55 FPS on integrated GPU. Zero network calls.**

### Biomech video upload — browser-only

1. User picks a video file → `URL.createObjectURL` → off-screen `<video>` element.
2. `requestVideoFrameCallback` fires per rendered frame (Chrome/Edge/Safari 15+); fallback to seek-by-time at 15 FPS sample rate on older browsers.
3. Video plays at **4× speed** so a 10-sec clip processes in ~2.5 sec of wall time + inference.
4. Each frame → MoveNet → angle math → time-series accumulator.
5. Peak detection on the time series → `BiomechDataDTO` with peak angle, magnitude, status, percentage, interpretation.
6. Same `AssessmentReport` component renders the result.

### Posture screening — single-image flow

1. Two `<input type="file">` slots: front view + side view (both required).
2. Each file → `Image` element → `analyzePostureImage(file, view)` runs MoveNet once.
3. **Front measurements** — head/shoulder/hip tilts (signed angle from horizontal) + left/right knee Q-angle (interior angle of hip-knee-ankle).
4. **Side measurements** — pick the side with higher keypoint confidence, then compute horizontal offsets (head/shoulder/hip/knee shifts) as percentage of body height (shoulder→ankle vertical span as proxy).
5. `PostureImageOverlay` draws the original image to a canvas, overlays plumb line + horizontal reference bands + keypoint dots + measurement badges — looks like a clinical screening shot.
6. Severity-graded `FindingsTable` (OK / Mild / Notable) renders below.

---

## Deployment

> 📘 **Step-by-step beginner guide:** See **[DEPLOYMENT.md](DEPLOYMENT.md)** for a full walk-through — push to GitHub, deploy backend to Hugging Face Spaces, deploy frontend to Vercel, wire them together, and verify. Total time ~30 minutes.

### Quick reference

| Component | Target | Key files |
|---|---|---|
| FastAPI backend | Hugging Face Spaces (Docker) | `Dockerfile`, `.dockerignore`, `requirements.txt`, `requirements_api.txt`, README frontmatter |
| Next.js frontend | Vercel | `motionlens-web/.env.production.example` (template), Vercel auto-detects Next.js |
| Streamlit (legacy) | Streamlit Cloud | `packages.txt`, `.python-version` |

### Required environment variables

**Backend** — set in HF Space → Settings → Variables:

```
MOTIONLENS_ALLOWED_ORIGINS=https://your-app.vercel.app
```

**Frontend** — set in Vercel → Project Settings → Environment Variables:

```
NEXT_PUBLIC_API_BASE_URL=https://your-username-motionlens-api.hf.space
```

### What requires the backend?

Only **gait analysis** (video upload). Live biomech, biomech upload, and posture analysis all run **fully in the browser** via TensorFlow.js + MoveNet — they need only the frontend deploy.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: api` when starting uvicorn | Wrong working directory | `cd /path/to/Gait_Analysis` first, then run uvicorn |
| `pip` complains about `fastmcp` packaging conflict | Unrelated MCP package on your system | Ignore — install line says `Successfully installed fastapi …` |
| `http://0.0.0.0:8000` "site can't be reached" | `0.0.0.0` is a bind address, not a browseable URL | Use `http://localhost:8000` |
| Live biomech: camera permission denied / `getUserMedia is undefined` | Browser blocks camera on plain HTTP from non-localhost origin | Use `https://` (Vercel auto-HTTPS) or `localhost:3000` for local dev. For LAN testing run `npm run dev:https`. |
| Live biomech: "Required landmarks below visibility threshold" | Patient partially out of frame, or pose model just loaded | Stand 6 ft back, full upper body in shot. First inference is slower (shader compile). |
| Live biomech: skeleton doesn't draw | TF.js failed to load WebGL backend | Open browser console, check `tf.getBackend()` — should be `"webgl"`. Falls back to CPU automatically (~10 FPS instead of 50). |
| Live biomech: NaN coordinates / detection fails silently | Older `BlazePose-tfjs` issue | We use MoveNet now, not BlazePose-tfjs — should not happen. If it does, check `lib/pose/detector.ts` |
| Biomech video upload hangs | Video file unsupported codec or corrupt | Try a different MP4 (H.264 + AAC works best) |
| Posture: "No person detected" | Patient not fully in frame, or low contrast clothing | Plain background, full body in frame, contrasting clothing |
| Gait upload network error | `NEXT_PUBLIC_API_BASE_URL` missing in Vercel | Set env var, **then redeploy** (Next.js bakes env at build time) |
| Vercel build fails: missing `@/components/gait/...` on case-insensitive systems | Root `.gitignore` rule like `Gait/` matches lowercase `gait/` | Anchor to root: `/Gait/` not `Gait/` |
| Gait cycle tab: "not enough clean cycles" | Clip too short / too few strides per leg in passes | Use a 10–15 s side-on clip with ≥4 full cycles |
| Frontend `npm run dev` shows API errors | `NEXT_PUBLIC_API_BASE_URL` mis-set or backend down | Check `.env.local`, verify `/api/health` |
| Build error: `@mediapipe/pose` missing exports | Stale `node_modules` | The `next.config.ts` aliases this to a local stub; ensure that alias is intact and rebuild |

---

## Disclaimer

MotionLens is an **educational / research prototype**. All outputs are derived from 2D pose estimation and **do not replace** professional 3D clinical gait laboratories or medical diagnosis. Auto-generated observations are descriptive only — clinical interpretation belongs to a qualified professional considering the full clinical picture, medical history, and other assessments. Rotation measurements are especially sensitive to camera angle and should be considered indicative only.

---

## References

- Perry J, Burnfield JM. *Gait Analysis: Normal and Pathological Function.* 2nd ed., 2010.
- Winter DA. *Biomechanics and Motor Control of Human Movement.* 4th ed., 2009.
- Kadaba MP et al. "Measurement of lower extremity kinematics during level walking." *J Orthop Res* 1990;8(3):383–392.
- AAOS / APTA standard ROM positioning guidance (shoulder, cervical spine).
- MediaPipe Pose — Google Research, 2023.



