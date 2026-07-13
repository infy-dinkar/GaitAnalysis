"""
api.py
FastAPI wrapper for the MotionLens Python engines.

Routes
------
GET  /api/health              → service heartbeat
POST /api/analyze-gait        → multipart {video, height_cm, patient_name?}
POST /api/analyze-shoulder    → multipart {video, movement_type, side?, patient_name?}
POST /api/analyze-neck        → multipart {video, movement_type, patient_name?}

The endpoints call the existing engine functions WITHOUT modification.
Streamlit's app.py continues to work in parallel — the API is purely
additive.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Swagger UI:  http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Engine imports (no modifications)
from engines.gait_engine import (
    extract_poses,
    build_time_series,
    compute_all_features,
    interpret,
)
from engines.biomech.shoulder_engine import (
    SHOULDER_NORMAL_RANGES,
    analyze_shoulder as analyze_shoulder_engine,
    compute_shoulder_angle,
)
from engines.biomech.neck_engine import NECK_NORMAL_RANGES, compute_neck_angle
from engines.biomech_flow import (
    _run_biomech_upload_analysis,
    _ensure_pose_model_file,
    _LandmarkAdapter,
    _wrap_landmarks,
)

from utils.api_helpers import (
    save_uploaded_video,
    cleanup_temp_file,
    format_gait_response,
    format_biomech_response,
)
from models.api_models import (
    BiomechResponse,
    GaitResponse,
    HealthResponse,
    LandmarkPoint,
    LiveBiomechFrameData,
    LiveBiomechFrameResponse,
)

# ─── Timed Up and Go (TUG) — reuses the gait MediaPipe pipeline ───
from engines.orthopedic.tug_engine import analyze_tug
from engines.orthopedic.tug_models import TUGResponse

# ─── Trendelenburg — single-leg stance upload mode ───────────────
# Reuses the gait MediaPipe pipeline + per-side single-leg-stance
# math in trendelenburg_engine.py. Live mode (browser BlazePose
# WASM) and upload mode produce the same TrendelenburgSideResult
# shape so the existing TrendelenburgReport renders both without
# translation.
from engines.orthopedic.trendelenburg_engine import analyze_trendelenburg

# ─── Single-Leg Squat — Test B1 upload mode ──────────────────────
# Same per-side, parallel-analysis pattern as Trendelenburg.
# Engine math mirrors lib/orthopedic/singleLegSquat.ts so the
# existing SingleLegSquatReport renders live + upload identically.
from engines.orthopedic.single_leg_squat_engine import analyze_single_leg_squat
from engines.orthopedic.slr_engine import analyze_slr
from engines.orthopedic.ake_engine import analyze_ake
from engines.orthopedic.modified_thomas_engine import analyze_modified_thomas
from engines.orthopedic.forward_lunge_engine import analyze_forward_lunge
from engines.orthopedic.sts_quality_engine import analyze_sts_quality
from engines.orthopedic.tandem_walk_engine import analyze_tandem_walk
from engines.orthopedic.pronator_drift_engine import analyze_pronator_drift
from engines.orthopedic.functional_reach_engine import analyze_functional_reach
from engines.orthopedic.single_leg_hop_engine import analyze_single_leg_hop
from engines.orthopedic.counter_movement_jump_engine import analyze_counter_movement_jump
from engines.orthopedic.tuck_jump_engine import analyze_tuck_jump
from engines.orthopedic.overhead_squat_engine import analyze_overhead_squat

# ─── 5x Sit-to-Stand — Test C2 upload mode ───────────────────────
# SINGLE trial (no L/R split). Engine math mirrors
# lib/orthopedic/sitToStand.ts so live + upload produce the same
# SitToStandResult shape and the existing SitToStandReport renders
# both without translation.
from engines.orthopedic.sit_to_stand_engine import analyze_sit_to_stand

# ─── 30-Second Chair Stand — Test C3 upload mode ─────────────────
# Single trial, timer-driven (30s). Same sit↔stand state machine
# as 5xSTS; differs in termination (30s timer) and primary outcome
# (rep COUNT vs total time). CDC STEADI age + sex norm comparison.
from engines.orthopedic.chair_stand_30s_engine import analyze_chair_stand_30s

# ─── Single-Leg Stance — Test C5 upload mode ─────────────────────
# Per-trial analysis (side + condition). Up to 4 trials per session
# (left_open, right_open, left_closed, right_closed). Frontend
# parallelises uploads via Promise.allSettled.
from engines.orthopedic.single_leg_stance_engine import analyze_single_leg_stance

# ─── 4-Stage Balance — Test C4 upload mode ───────────────────────
# Per-stage analysis. Frontend uploads up to 4 stages in parallel
# and applies the stop-at-first-failure rule client-side.
from engines.orthopedic.four_stage_balance_engine import analyze_four_stage_balance

# ─── SPPB Component 2 (Gait Speed) — Test C7 upload mode ─────────
# 4-metre walk gait-speed detection on the backend. Other two SPPB
# components reuse existing endpoints.
from engines.sppb.sppb_gait_speed_engine import analyze_sppb_gait_speed

# ─── Ankle (dorsi/plantar) — reuses gait MediaPipe pipeline ──────
from engines.biomech.ankle_engine import analyze_ankle as analyze_ankle_engine

# ─── Knee (merged flex+ext) — reuses gait MediaPipe pipeline ─────
from engines.biomech.knee_engine import analyze_knee as analyze_knee_engine

# ─── Neck (merged flex+ext) — reuses gait MediaPipe pipeline ─────
from engines.biomech.neck_engine import analyze_neck as analyze_neck_engine

# ─── Hip (flexion) — reuses gait MediaPipe pipeline ──────────────
from engines.biomech.hip_engine import analyze_hip as analyze_hip_engine

# ─── Posture — IMAGE-mode MediaPipe (separate pipeline) ─────────
from engines.posture_engine import (
    analyze_posture_combined as analyze_posture_combined_engine,
)

# ─── SPPB Component 1 (Balance) — reuses gait MediaPipe pipeline ─
from engines.sppb.sppb_balance_engine import analyze_sppb_balance

# ─── Auth + database (Phase 1) ─────────────────────────────────────
from utils import db as db_module
from routes.auth_routes import router as auth_router

# ─── Patient + report endpoints (Phase 2) ──────────────────────────
from routes.patient_routes import router as patient_router
from routes.report_routes import (
    patient_reports_router,
    reports_router,
)
from routes.prescription_routes import router as prescription_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("motionlens.api")

# ─── Gait upload validation limits ────────────────────────────────
# Per MotionLens Test Battery spec v1.0. Clinical gait analysis only
# needs ~10 sec of footage; anything beyond 60 sec is noise and risks
# OOM. Sub-24 FPS uploads silently produce wrong cadence/step-time
# metrics, so reject them outright; 24-29 FPS gets a soft warning.
MAX_GAIT_FILE_SIZE_MB = 100
MAX_GAIT_DURATION_SEC = 60
MIN_GAIT_DURATION_SEC = 5
MIN_REQUIRED_FPS = 24
RECOMMENDED_FPS = 30


app = FastAPI(
    title="MotionLens API",
    description="REST endpoints exposing the gait + biomech engines.",
    version="1.0.0",
)

# CORS: allow the Next.js dev server + any origin in dev. Tighten in prod.
ALLOWED_ORIGINS = os.environ.get(
    "MOTIONLENS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Database lifecycle ────────────────────────────────────────────
@app.on_event("startup")
async def _on_startup() -> None:
    """Connect to MongoDB once at process start.

    If MONGODB_URI isn't set, we log a warning and continue — non-DB
    endpoints (gait/biomech analysis) keep working. Only auth/patient/
    report endpoints will fail until the env var is provided.
    """
    try:
        await db_module.connect()
    except Exception as e:
        log.warning("MongoDB unavailable at startup: %s", e)
        log.warning("Auth + patient + report endpoints will return 500 until MONGODB_URI is configured.")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    await db_module.disconnect()


# ─── Routers ───────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(patient_reports_router)
app.include_router(reports_router)
app.include_router(prescription_router)


# ══════════════════════════════════════════════════════════════════════
# Pose-model setup (module-level, reused across requests)
# ══════════════════════════════════════════════════════════════════════
# Using Full variant for clinical-grade landmark accuracy
# (BlazePose Full per MotionLens Test Battery spec v1.0).
# Lite was previously used; upgraded for Module D readiness.
def _build_gait_pose_options():
    """Construct PoseLandmarkerOptions for VIDEO running mode — exactly
    matching app.py's load_pose_model_options() construction."""
    model_path = _ensure_pose_model_file()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )


@app.on_event("startup")
async def _warmup() -> None:
    try:
        _ensure_pose_model_file()
        log.info("Pose model file ready.")
    except Exception as e:
        log.warning("Pose model preload failed (will retry on first call): %s", e)


# ══════════════════════════════════════════════════════════════════════
# Live-mode pose landmarker pool — VIDEO running mode for stable tracking
# ══════════════════════════════════════════════════════════════════════
# Why VIDEO instead of IMAGE: in IMAGE mode each frame is detected from
# scratch, so per-frame jitter is fully visible. VIDEO mode maintains
# internal tracking state across frames (the same state Google's mobile
# pose apps use), giving much smoother landmarks. The one constraint is
# that timestamps must be strictly monotonically increasing per
# landmarker instance — `_StatefulVideoLandmarker` tracks that.
#
# We keep a small POOL of landmarkers (not thread-safe individually, but
# parallel across instances) so FastAPI's threadpool can process up to
# N frames concurrently — roughly 2× throughput on a 4-core CPU at
# pool size 2. Pool size is env-configurable.
# Default 1 for VIDEO mode: a single landmarker sees the full frame
# stream, so the internal tracker's state stays continuous across every
# frame — maximum stability. Bump higher only if you have multiple
# concurrent users (then trade tracking continuity for throughput).
_LIVE_POOL_SIZE = max(1, int(os.environ.get("MOTIONLENS_LIVE_POOL_SIZE", "1")))
_live_pool: "queue.Queue | None" = None
_live_pool_init_lock = threading.Lock()

# Each landmarker bumps its frame timestamp by this many ms. Any positive
# step works (only monotonicity matters), but ~42 ms matches our typical
# 24 fps live rate and gives MediaPipe's internal tracker a realistic
# dt for its velocity model.
_TS_STEP_MS = 42


class _StatefulVideoLandmarker:
    """Wraps a PoseLandmarker (VIDEO mode) with its own monotonic
    timestamp counter + a per-landmark EMA smoother for the *visualised*
    landmarks. The smoother only affects what's drawn on screen — the
    raw landmarks are still used for the engine's angle math, so peak /
    current measurements are unaffected.

    EMA alpha is tunable via env var. Default 0.45:
      smoothed.x = 0.45 * raw.x + 0.55 * prev_smoothed.x
    Lighter than 0.375 — at 20 fps the higher frame rate already
    gives more averaging in real time, so per-frame alpha can lift
    without losing visual stability."""

    __slots__ = ("_landmarker", "_next_ts_ms", "_smooth_alpha", "_prev_xy")

    def __init__(self, landmarker, smooth_alpha: float = 0.45):
        self._landmarker = landmarker
        self._next_ts_ms = 0
        self._smooth_alpha = smooth_alpha
        # list of (x, y) per landmark, populated lazily on first frame
        self._prev_xy: list[tuple[float, float]] | None = None

    def detect(self, mp_image):
        ts = self._next_ts_ms
        self._next_ts_ms += _TS_STEP_MS
        return self._landmarker.detect_for_video(mp_image, ts)

    def smooth_for_overlay(self, raw_landmarks) -> list[tuple[float, float, float]]:
        """Apply EMA smoothing for the response payload only. Returns a
        list of (x, y, visibility) tuples — same length as raw_landmarks."""
        a = self._smooth_alpha
        prev = self._prev_xy
        if prev is None or len(prev) != len(raw_landmarks):
            self._prev_xy = [(lm.x, lm.y) for lm in raw_landmarks]
            return [
                (lm.x, lm.y, float(lm.visibility) if lm.visibility is not None else 0.0)
                for lm in raw_landmarks
            ]
        out: list[tuple[float, float, float]] = []
        new_prev: list[tuple[float, float]] = []
        for i, lm in enumerate(raw_landmarks):
            px, py = prev[i]
            sx = a * lm.x + (1 - a) * px
            sy = a * lm.y + (1 - a) * py
            new_prev.append((sx, sy))
            vis = float(lm.visibility) if lm.visibility is not None else 0.0
            out.append((sx, sy, vis))
        self._prev_xy = new_prev
        return out

    def reset_smoother(self):
        self._prev_xy = None

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


_OVERLAY_SMOOTH_ALPHA = float(
    os.environ.get("MOTIONLENS_OVERLAY_SMOOTH_ALPHA", "0.45")
)


# Using Full variant for clinical-grade landmark accuracy
# (BlazePose Full per MotionLens Test Battery spec v1.0).
# Lite was previously used; upgraded for Module D readiness.
def _build_video_landmarker() -> _StatefulVideoLandmarker:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    model_path = _ensure_pose_model_file()
    raw = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
        )
    )
    return _StatefulVideoLandmarker(raw, smooth_alpha=_OVERLAY_SMOOTH_ALPHA)


def _ensure_live_pool() -> "queue.Queue":
    global _live_pool
    if _live_pool is not None:
        return _live_pool
    with _live_pool_init_lock:
        if _live_pool is not None:
            return _live_pool
        pool: "queue.Queue" = queue.Queue(maxsize=_LIVE_POOL_SIZE)
        for _ in range(_LIVE_POOL_SIZE):
            pool.put(_build_video_landmarker())
        _live_pool = pool
        log.info(
            "Live landmarker pool initialised (size=%d, mode=VIDEO).",
            _LIVE_POOL_SIZE,
        )
        return pool


@contextmanager
def _live_landmarker():
    """Acquire a landmarker from the pool, release back on exit."""
    pool = _ensure_live_pool()
    lmk = pool.get()
    try:
        yield lmk
    finally:
        pool.put(lmk)


# ══════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


# Build-info endpoint — useful for confirming which commit the
# deployed backend is actually running. Reads from git at startup
# (if available) and falls back to an env var that CI/CD can set.
import subprocess as _subprocess
import os as _os

def _get_build_info() -> dict:
    info: dict = {}
    # Try git directly first.
    try:
        sha = _subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
        if sha:
            info["commit"] = sha
    except Exception:
        pass
    # Fall back to env var (set by hosting platforms — Render sets
    # RENDER_GIT_COMMIT, Railway sets RAILWAY_GIT_COMMIT_SHA, etc.).
    for env_key in (
        "BACKEND_COMMIT_SHA",
        "RENDER_GIT_COMMIT",
        "RAILWAY_GIT_COMMIT_SHA",
        "K_REVISION",       # Cloud Run
        "SOURCE_VERSION",   # Heroku
    ):
        v = _os.environ.get(env_key)
        if v:
            info.setdefault("commit", v[:12])
            info["source"] = env_key
            break
    if "commit" not in info:
        info["commit"] = "unknown"
    return info


_BUILD_INFO = _get_build_info()


@app.get("/api/version")
async def version() -> dict:
    """Returns the commit SHA the backend is running. Curl this to
    confirm a deploy actually picked up new code:
        curl https://your-backend.example.com/api/version
    """
    return _BUILD_INFO


@app.post("/api/analyze-gait", response_model=GaitResponse)
async def analyze_gait(
    video: UploadFile = File(...),
    height_cm: float = Form(170.0),
    patient_name: Optional[str] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> GaitResponse:
    """Run the full gait pipeline on an uploaded clip.

    `recording_duration_ms` is supplied by the in-browser "record then
    upload" path on the frontend — wall-clock duration between
    MediaRecorder.start() and stop(). MediaRecorder-produced WebMs
    often ship with broken / missing duration metadata, which makes
    cv2's CAP_PROP_FPS probe return 0 (or a bogus low frame count)
    and would falsely reject otherwise-valid live recordings at the
    FPS gate. When set we re-mux the file with a clean header via
    tug_engine._ensure_decodable_video BEFORE the gates run. Normal
    file uploads pass recording_duration_ms=None and the helper is
    a safe no-op (returns the original path unchanged).
    """
    from engines.orthopedic.tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # ── 1) FILE SIZE GATE — reject before touching the disk ──────
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB. "
                    "Please trim or compress the video."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "video.mp4")
        log.info(
            "gait: file=%s size=%.2f MB height=%scm recording_ms=%s",
            video.filename, size_mb, height_cm, recording_duration_ms,
        )

        # ── 1.5) Repair MediaRecorder WebMs with broken duration
        # headers BEFORE the cv2 FPS probe — otherwise the probe
        # returns 0 and we'd reject otherwise-valid live recordings.
        # For normal uploads (recording_duration_ms=None) this is a
        # no-op and returns (tmp_path, None).
        processed_path, fixed_path_cleanup = _ensure_decodable_video(
            tmp_path, recording_duration_ms,
        )

        # ── 2) FPS + DURATION GATE — peek at the container before the
        # full pose-extraction pass. A bad clip should fail in <100 ms,
        # not after 30 sec of MediaPipe work.
        probe = cv2.VideoCapture(processed_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required for gait "
                    f"analysis. Recommended: {RECOMMENDED_FPS}+ FPS. "
                    "Please re-record at higher frame rate."
                ),
            )

        fps_warning: Optional[str] = None
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
            log.warning("gait: low fps %.1f (below recommended)", probe_fps)

        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video length. "
                    "Please upload a different file."
                ),
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed. "
                    "Please trim to capture 4-6 gait cycles "
                    "(~10 seconds is ideal)."
                ),
            )

        duration_warning: Optional[str] = None
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds (4 gait cycles) "
                "recommended for reliable metrics."
            )
            log.warning(
                "gait: short clip %.1fs (below recommended)", duration_seconds,
            )

        # ── 3) Validation passed — run the full pipeline ─────────────
        pose_options = _build_gait_pose_options()
        raw, fps, total_frames = extract_poses(processed_path, pose_options)
        ts = build_time_series(raw)
        features = compute_all_features(
            ts, fps, total_frames, user_height_cm=float(height_cm),
        )
        insights = interpret(features)

        data = format_gait_response(
            features=features,
            insights=insights,
            ts=ts,
            height_cm=float(height_cm),
            patient_name=patient_name,
        )
        return GaitResponse(
            success=True,
            data=data,
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("gait validation: %s", e)
        return GaitResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("gait analysis failed")
        return GaitResponse(success=False, data=None, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
        # _ensure_decodable_video returns a second path only when it
        # actually wrote a repaired file — clean that up too.
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


# ══════════════════════════════════════════════════════════════════════
# Timed Up and Go (TUG) — record-then-analyze
# ══════════════════════════════════════════════════════════════════════
# Reuses the gait module's MediaPipe BlazePose Full pipeline + the
# same upload-validation gate (size, FPS, duration). Phase detection
# and TUG-specific metrics live in tug_engine.py.
@app.post("/api/analyze-tug", response_model=TUGResponse)
async def analyze_tug_endpoint(
    video: UploadFile = File(...),
    patient_age: Optional[int] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> TUGResponse:
    """Run the TUG pipeline on an uploaded side-view clip.

    `recording_duration_ms` is supplied by the live-record path on the
    frontend — wall-clock time between MediaRecorder.start() and stop().
    The WebM container produced by MediaRecorder often has missing
    duration metadata, which makes cv2's CAP_PROP_FPS probe return 0;
    in that case we use the client duration to compute FPS in
    tug_engine._ensure_decodable_video.
    """
    tmp_path: str | None = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # 1) File-size gate — same threshold as gait (100 MB)
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "tug.mp4")
        log.info(
            "tug: file=%s size=%.2f MB age=%s recording_ms=%s",
            video.filename, size_mb, patient_age, recording_duration_ms,
        )

        # 2) FPS + duration gate. cv2's CAP_PROP_FPS is sometimes 0
        # for MediaRecorder-produced WebM. When that happens, fall
        # back to the client-supplied recording duration. The engine
        # will re-encode the file with a proper FPS header below.
        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        # If header is broken AND we have a client duration, defer
        # the FPS / duration validation to the engine — it'll count
        # actual frames during the rewrite step and compute the
        # effective FPS from there.
        deferred_fps_check = (
            probe_fps <= 0 and recording_duration_ms and recording_duration_ms > 0
        )

        if probe_fps <= 0 and not deferred_fps_check:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "If this was a live recording, please retry; otherwise "
                    "upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None
        duration_seconds: float = 0.0

        if probe_fps > 0:
            if probe_fps < MIN_REQUIRED_FPS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Video frame rate too low ({probe_fps:.1f} FPS). "
                        f"Minimum {MIN_REQUIRED_FPS} FPS required for TUG analysis. "
                        f"Recommended: {RECOMMENDED_FPS}+ FPS."
                    ),
                )
            if probe_fps < RECOMMENDED_FPS:
                fps_warning = (
                    f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                    f"{RECOMMENDED_FPS} FPS — results may be less accurate."
                )
            if probe_total_frames <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine video length. Please upload a different file.",
                )
            duration_seconds = probe_total_frames / probe_fps
        else:
            # FPS was deferred — use client duration for the upper-/
            # lower-bound checks against MAX/MIN_GAIT_DURATION_SEC.
            duration_seconds = (recording_duration_ms or 0) / 1000.0

        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds and duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended for a complete TUG."
            )

        # 3) Validation passed — run the TUG pipeline (reuses gait
        #    MediaPipe options).
        pose_options = _build_gait_pose_options()
        result = analyze_tug(
            video_path=tmp_path,
            pose_options=pose_options,
            patient_age=patient_age,
            recording_duration_ms=recording_duration_ms,
        )

        return TUGResponse(
            success=True,
            data=result,
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("tug validation: %s", e)
        return TUGResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("tug analysis failed")
        return TUGResponse(success=False, data=None, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Trendelenburg — single-leg stance, per-side upload analysis
# ══════════════════════════════════════════════════════════════════════
# One POST per side. The frontend uploads the left-stance and right-
# stance clips in parallel (Promise.allSettled) and assembles the
# combined TrendelenburgFullResult { left, right } client-side. Math
# + classification cutoffs mirror lib/orthopedic/trendelenburg.ts so
# live + upload produce identical reports.
class TrendelenburgFrameSampleDTO(BaseModel):
    t_ms: float
    pelvic_tilt_deg: Optional[float] = None
    trunk_lean_deg: Optional[float] = None


class TrendelenburgKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class TrendelenburgSideResultDTO(BaseModel):
    side_tested: str
    hold_seconds: float
    max_drop_deg: float
    mean_drop_deg: float
    max_compensatory_lean_deg: float
    classification: str
    short_hold: bool
    trendelenburg_gait_pattern: bool
    termination: str
    samples: List[TrendelenburgFrameSampleDTO]
    keypoints: List[List[TrendelenburgKeypointDTO]]
    peak_screenshot_data_url: Optional[str] = None


class TrendelenburgResponse(BaseModel):
    success: bool
    data: Optional[TrendelenburgSideResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-trendelenburg", response_model=TrendelenburgResponse)
async def analyze_trendelenburg_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),                 # "left" or "right" stance
    patient_age: Optional[int] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> TrendelenburgResponse:
    """Run the Trendelenburg pipeline on an uploaded single-leg-stance
    clip. `side` is the STANCE leg (the leg the patient is standing
    on). Validation gates mirror the TUG endpoint (file size, FPS,
    duration) — both use the same gait MediaPipe pipeline downstream.

    `recording_duration_ms` is reserved for a future record-mode
    fallback when MediaRecorder WebMs ship with broken duration
    headers — currently accepted but unused in this pipeline.
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # 1) File-size gate
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "trendelenburg.mp4")
        log.info(
            "trendelenburg: file=%s side=%s size=%.2f MB age=%s",
            video.filename, side, size_mb, patient_age,
        )

        # 2) FPS + duration gate
        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended for a complete "
                f"single-leg-stance hold."
            )

        # 3) Run the analysis (reuses gait MediaPipe options).
        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_trendelenburg(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
        )

        return TrendelenburgResponse(
            success=True,
            data=TrendelenburgSideResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("trendelenburg validation: %s", e)
        return TrendelenburgResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("trendelenburg analysis failed")
        return TrendelenburgResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Single-Leg Squat — Test B1, per-side upload analysis
# ══════════════════════════════════════════════════════════════════════
# One POST per side. The frontend uploads the left-stance and right-
# stance clips in parallel (Promise.allSettled) and assembles the
# combined SingleLegSquatFullResult { left, right } client-side.
# Math + classification cutoffs mirror lib/orthopedic/singleLegSquat.ts
# so live + upload produce identical reports.
class SLSRepDTO(BaseModel):
    rep_index: int
    t_ms: float
    kfppa_deg: Optional[float] = None
    pelvic_drop_deg: Optional[float] = None
    trunk_lean_deg: Optional[float] = None
    depth_pct: Optional[float] = None


class SLSFrameSampleDTO(BaseModel):
    t_ms: float
    hip_mid_y: Optional[float] = None
    kfppa_deg: Optional[float] = None
    pelvic_drop_deg: Optional[float] = None
    trunk_lean_deg: Optional[float] = None


class SLSKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class SingleLegSquatSideResultDTO(BaseModel):
    side_tested: str
    reps: List[SLSRepDTO]
    worst_rep_index: Optional[int] = None
    worst_kfppa_deg: float
    mean_pelvic_drop_deg: float
    mean_trunk_lean_deg: float
    mean_depth_pct: float
    classification: str
    risk_score: str
    duration_seconds: float
    termination: str
    incomplete: bool
    samples: List[SLSFrameSampleDTO]
    keypoints: List[List[SLSKeypointDTO]]
    worst_rep_screenshot_data_url: Optional[str] = None
    # Extras — not in the strict TS SingleLegSquatSideResult shape but
    # surfaced for the response envelope's per-side annotations.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None
    camera_squareness_warning: Optional[bool] = None
    median_shoulder_tilt_deg: Optional[float] = None


class SingleLegSquatResponse(BaseModel):
    success: bool
    data: Optional[SingleLegSquatSideResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None
    squareness_warning: Optional[str] = None


@app.post("/api/analyze-single-leg-squat", response_model=SingleLegSquatResponse)
async def analyze_single_leg_squat_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),                 # "left" or "right" stance
    patient_age: Optional[int] = Form(None),
) -> SingleLegSquatResponse:
    """Run the Single-Leg Squat (Test B1) pipeline on an uploaded
    clip. `side` is the STANCE leg (the leg the patient is standing
    on during the squat). Validation gates mirror the TUG /
    Trendelenburg endpoints — same gait MediaPipe pipeline downstream.
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # 1) File-size gate
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "single_leg_squat.mp4",
        )
        log.info(
            "single_leg_squat: file=%s side=%s size=%.2f MB age=%s",
            video.filename, side, size_mb, patient_age,
        )

        # 2) FPS + duration gate
        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended for a complete "
                f"single-leg-squat set."
            )

        # 3) Run the analysis (reuses gait MediaPipe options).
        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_single_leg_squat(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
        )

        squareness_warning: Optional[str] = None
        if result.get("camera_squareness_warning"):
            tilt = result.get("median_shoulder_tilt_deg") or 0.0
            squareness_warning = (
                f"Patient was rotated by a median {abs(tilt):.1f}° — "
                "KFPPA accuracy degrades. For best results, have the "
                "patient face the camera squarely with both shoulders "
                "level in frame."
            )

        return SingleLegSquatResponse(
            success=True,
            data=SingleLegSquatSideResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
            squareness_warning=squareness_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("single_leg_squat validation: %s", e)
        return SingleLegSquatResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("single_leg_squat analysis failed")
        return SingleLegSquatResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Straight Leg Raise (SLR) — per-side upload analysis
# ══════════════════════════════════════════════════════════════════════
# One POST per side. The frontend uploads left and right clips in
# parallel (Promise.allSettled) and assembles SLRFullResult { left,
# right } client-side. Math + classification cutoffs mirror
# lib/orthopedic/slr.ts so live + upload produce identical reports.
class SLRFrameSampleDTO(BaseModel):
    t_ms: float
    raise_angle_deg: Optional[float] = None
    knee_angle_deg: Optional[float] = None
    knee_straight: bool


class SLRKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class SLRSideResultDTO(BaseModel):
    side_tested: str
    max_raise_angle_deg: float
    max_raise_sample_index: Optional[int] = None
    knee_angle_at_peak_deg: Optional[float] = None
    classification: str
    duration_seconds: float
    termination: str
    knee_straight_fraction: float
    samples: List[SLRFrameSampleDTO]
    keypoints: List[List[SLRKeypointDTO]]
    peak_screenshot_data_url: Optional[str] = None
    # Extras — not in the strict TS SLRSideResult shape but surfaced
    # for the response envelope's per-side annotations.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class SLRResponse(BaseModel):
    success: bool
    data: Optional[SLRSideResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-slr", response_model=SLRResponse)
async def analyze_slr_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
) -> SLRResponse:
    """Run the Straight Leg Raise pipeline on an uploaded clip. `side`
    is the LEG being raised (the camera should sit on that same side
    of the patient for the lateral view). Validation gates mirror the
    Single-Leg-Squat / TUG endpoints — same gait MediaPipe pipeline
    downstream.
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # 1) File-size gate
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "slr.mp4",
        )
        log.info(
            "slr: file=%s side=%s size=%.2f MB",
            video.filename, side, size_mb,
        )

        # 2) FPS + duration gate
        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine can capture "
                f"the full raise + a brief hold."
            )

        # 3) Run the analysis (reuses gait MediaPipe options).
        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_slr(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
        )

        return SLRResponse(
            success=True,
            data=SLRSideResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("slr validation: %s", e)
        return SLRResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("slr analysis failed")
        return SLRResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Active Knee Extension (AKE) — per-side upload analysis
# ══════════════════════════════════════════════════════════════════════
# One POST per side. The frontend uploads left and right clips
# sequentially (same cold-worker mitigation as SLR) and assembles
# AKEFullResult { left, right } client-side. Math + classification
# cutoffs mirror lib/orthopedic/ake.ts so live + upload produce
# identical reports.
class AKEFrameSampleDTO(BaseModel):
    t_ms: float
    knee_angle_deg: Optional[float] = None
    hip_flex_angle_deg: Optional[float] = None
    thigh_held: bool


class AKEKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class AKESideResultDTO(BaseModel):
    side_tested: str
    max_knee_angle_deg: float
    deficit_deg: float
    max_knee_sample_index: Optional[int] = None
    hip_flex_angle_at_peak_deg: Optional[float] = None
    classification: str
    duration_seconds: float
    termination: str
    thigh_held_fraction: float
    samples: List[AKEFrameSampleDTO]
    keypoints: List[List[AKEKeypointDTO]]
    peak_screenshot_data_url: Optional[str] = None
    # Extras — not in the strict TS AKESideResult shape but surfaced
    # for the response envelope's per-side annotations.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class AKEResponse(BaseModel):
    success: bool
    data: Optional[AKESideResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-ake", response_model=AKEResponse)
async def analyze_ake_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
) -> AKEResponse:
    """Run the Active Knee Extension pipeline on an uploaded clip.
    `side` is the LEG being tested (the camera should sit on that same
    side of the patient for the lateral view). Validation gates mirror
    the SLR / Single-Leg-Squat endpoints — same gait MediaPipe pipeline
    downstream.
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # 1) File-size gate
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "ake.mp4",
        )
        log.info(
            "ake: file=%s side=%s size=%.2f MB",
            video.filename, side, size_mb,
        )

        # 2) FPS + duration gate
        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine can capture "
                f"the full knee extension."
            )

        # 3) Run the analysis (reuses gait MediaPipe options).
        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_ake(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
        )

        return AKEResponse(
            success=True,
            data=AKESideResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("ake validation: %s", e)
        return AKEResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("ake analysis failed")
        return AKEResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Modified Thomas Test (MTT) — per-side upload analysis
# ══════════════════════════════════════════════════════════════════════
# One POST per side. The frontend uploads left and right clips
# sequentially (same cold-worker mitigation as SLR / AKE) and
# assembles ModifiedThomasFullResult { left, right } client-side. Math
# + classification cutoffs mirror lib/orthopedic/modifiedThomas.ts so
# live + upload produce identical reports.
#
# This test is a STATIC HOLD — the engine looks for the longest
# stable window in the clip and returns the median hip + knee angles
# from that window. If no window meets the jitter gate it falls back
# to the median of the last 1.5 s and flags low_confidence=True.
class MTTFrameSampleDTO(BaseModel):
    t_ms: float
    hip_angle_deg: Optional[float] = None
    knee_angle_deg: Optional[float] = None
    stable: bool


class MTTKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class ModifiedThomasSideResultDTO(BaseModel):
    side_tested: str
    hip_angle_deg: float
    knee_angle_deg: float
    hip_classification: str
    knee_classification: str
    hip_angle_stddev_deg: float
    knee_angle_stddev_deg: float
    low_confidence: bool
    capture_sample_index: Optional[int] = None
    duration_seconds: float
    termination: str
    samples: List[MTTFrameSampleDTO]
    keypoints: List[List[MTTKeypointDTO]]
    capture_screenshot_data_url: Optional[str] = None
    # Extras — not in the strict TS ModifiedThomasSideResult shape but
    # surfaced for the response envelope's per-side annotations.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class ModifiedThomasResponse(BaseModel):
    success: bool
    data: Optional[ModifiedThomasSideResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-modified-thomas", response_model=ModifiedThomasResponse)
async def analyze_modified_thomas_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
) -> ModifiedThomasResponse:
    """Run the Modified Thomas Test pipeline on an uploaded clip.
    `side` is the HANGING (test) leg (the camera should sit on that
    same side of the patient for the lateral view). Validation gates
    mirror the SLR / AKE endpoints — same gait MediaPipe pipeline
    downstream.
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # 1) File-size gate
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "modified_thomas.mp4",
        )
        log.info(
            "modified_thomas: file=%s side=%s size=%.2f MB",
            video.filename, side, size_mb,
        )

        # 2) FPS + duration gate
        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine can find "
                f"a stable hold window."
            )

        # 3) Run the analysis (reuses gait MediaPipe options).
        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_modified_thomas(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
        )

        return ModifiedThomasResponse(
            success=True,
            data=ModifiedThomasSideResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("modified_thomas validation: %s", e)
        return ModifiedThomasResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("modified_thomas analysis failed")
        return ModifiedThomasResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Forward Lunge (B3) — per-side upload analysis
# ══════════════════════════════════════════════════════════════════════
# One POST per side. The frontend uploads left and right clips
# sequentially (same cold-worker mitigation as SLR / AKE / MTT) and
# assembles ForwardLungeFullResult { left, right } client-side. Math
# + classification cutoffs mirror lib/orthopedic/forwardLunge.ts so
# live + upload produce identical reports.
#
# Rep detection uses scipy.signal.find_peaks on the TEST-side hip Y
# trajectory with distance + prominence gates bit-identical to the
# SLS engine — see forward_lunge_engine.py for the full pipeline.
class FLRepDTO(BaseModel):
    rep_index: int
    t_ms: float
    knee_angle_at_bottom_deg: Optional[float] = None
    knee_over_toe_ratio: Optional[float] = None
    trunk_lean_deg: Optional[float] = None


class FLFrameSampleDTO(BaseModel):
    t_ms: float
    hip_y: Optional[float] = None
    knee_angle_deg: Optional[float] = None
    knee_over_toe_ratio: Optional[float] = None
    trunk_lean_deg: Optional[float] = None


class FLKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class ForwardLungeSideResultDTO(BaseModel):
    side_tested: str
    reps: List[FLRepDTO]
    worst_rep_index: Optional[int] = None
    worst_rep_knee_angle_deg: float
    worst_rep_kot_ratio: float
    worst_rep_trunk_lean_deg: float
    mean_knee_angle_deg: float
    depth_variation_deg: float
    depth_out_of_band: bool
    kot_flagged: bool
    trunk_lean_flagged: bool
    fatigue_flagged: bool
    classification: str
    duration_seconds: float
    termination: str
    incomplete: bool
    samples: List[FLFrameSampleDTO]
    keypoints: List[List[FLKeypointDTO]]
    worst_rep_screenshot_data_url: Optional[str] = None
    # Extras — not in the strict TS ForwardLungeSideResult shape but
    # surfaced for the response envelope's per-side annotations.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class ForwardLungeResponse(BaseModel):
    success: bool
    data: Optional[ForwardLungeSideResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-forward-lunge", response_model=ForwardLungeResponse)
async def analyze_forward_lunge_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
) -> ForwardLungeResponse:
    """Run the Forward Lunge (B3) pipeline on an uploaded clip. `side`
    is the FRONT (test) leg (the camera should sit on that same side
    of the patient for the lateral view). Validation gates mirror the
    SLR / AKE / MTT endpoints — same gait MediaPipe pipeline
    downstream.
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "forward_lunge.mp4",
        )
        log.info(
            "forward_lunge: file=%s side=%s size=%.2f MB",
            video.filename, side, size_mb,
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine "
                f"can capture 5 reps."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_forward_lunge(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
        )

        return ForwardLungeResponse(
            success=True,
            data=ForwardLungeSideResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("forward_lunge validation: %s", e)
        return ForwardLungeResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("forward_lunge analysis failed")
        return ForwardLungeResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Sit-to-Stand QUALITY (B4) — single-trial upload analysis
# ══════════════════════════════════════════════════════════════════════
# COMPLETELY SEPARATE from the 5x Sit-to-Stand (C2) speed test below.
# This module measures QUALITY (phase timing, smoothness, hand-use
# compensation) rather than total cycle time. Uses ONE clip with a
# camera-facing `side` + optional chair seat height. 3 reps. Math
# mirrors lib/orthopedic/stsQuality.ts.
class STSQRepDTO(BaseModel):
    rep_index: int
    seat_off_t_ms: Optional[float] = None
    top_of_stand_t_ms: float
    start_of_descent_t_ms: Optional[float] = None
    re_seated_t_ms: Optional[float] = None
    sit_to_stand_ms: Optional[float] = None
    pause_ms: Optional[float] = None
    stand_to_sit_ms: Optional[float] = None
    trunk_lean_at_seat_off_deg: Optional[float] = None
    knee_angle_at_seat_off_deg: Optional[float] = None
    smoothness_score: Optional[float] = None
    hand_use_detected: bool


class STSQFrameSampleDTO(BaseModel):
    t_ms: float
    hip_y: Optional[float] = None
    knee_angle_deg: Optional[float] = None
    trunk_lean_deg: Optional[float] = None
    wrist_y: Optional[float] = None
    shoulder_y: Optional[float] = None
    leg_length_px: Optional[float] = None


class STSQKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class STSQualityResultDTO(BaseModel):
    camera_side: str
    chair_seat_height_cm: Optional[float] = None
    reps: List[STSQRepDTO]
    median_sit_to_stand_ms: Optional[float] = None
    median_pause_ms: Optional[float] = None
    median_stand_to_sit_ms: Optional[float] = None
    median_trunk_lean_deg: Optional[float] = None
    median_knee_angle_deg: Optional[float] = None
    median_smoothness_score: Optional[float] = None
    hand_use_count: int
    any_hand_use: bool
    classification: str
    duration_seconds: float
    termination: str
    incomplete: bool
    samples: List[STSQFrameSampleDTO]
    keypoints: List[List[STSQKeypointDTO]]
    worst_rep_screenshot_data_url: Optional[str] = None
    # Diagnostic extras — not in the strict TS shape but surfaced for
    # the response envelope's annotations.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class STSQualityResponse(BaseModel):
    success: bool
    data: Optional[STSQualityResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-sts-quality", response_model=STSQualityResponse)
async def analyze_sts_quality_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
    chair_seat_height_cm: Optional[float] = Form(None),
) -> STSQualityResponse:
    """Run the B4 Sit-to-Stand Quality pipeline on an uploaded clip.
    `side` is the CAMERA-FACING side (lateral view). `chair_seat_height_cm`
    is optional context metadata recorded verbatim. Validation gates
    mirror the FL / SLR / AKE / MTT endpoints.
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "sts_quality.mp4",
        )
        log.info(
            "sts_quality: file=%s side=%s seat_height=%s size=%.2f MB",
            video.filename, side, chair_seat_height_cm, size_mb,
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine "
                f"can capture 3 reps."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_sts_quality(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
            chair_seat_height_cm=chair_seat_height_cm,
        )

        return STSQualityResponse(
            success=True,
            data=STSQualityResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("sts_quality validation: %s", e)
        return STSQualityResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("sts_quality analysis failed")
        return STSQualityResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Tandem Walk (E1) — single-trial upload analysis
# ══════════════════════════════════════════════════════════════════════
# Frontal-view heel-to-toe gait screen. Single clip — patient walks
# 10 steps along a taped line toward the camera. Math mirrors
# lib/orthopedic/tandemWalk.ts: per-foot velocity-threshold step
# detector, least-squares walking-line fit on hip-midpoint, lateral
# deviation per footstrike normalised by shoulder width × 40 cm.
class TWStepDTO(BaseModel):
    step_index: int
    side: str
    t_ms: float
    foot_x: float
    foot_y: float
    deviation_px: Optional[float] = None
    # raw deviation in cm (no tolerance subtracted) — kept for
    # transparency / future re-classification with different cutoffs.
    raw_deviation_cm: Optional[float] = None
    # effective deviation = max(0, raw - DEVIATION_TOLERANCE_CM).
    # Mean / max aggregates, is_misstep, and classification all use
    # this value.
    deviation_cm: Optional[float] = None
    is_misstep: bool
    shoulder_width_px: Optional[float] = None


class TWFrameSampleDTO(BaseModel):
    t_ms: float
    hip_mid_x: Optional[float] = None
    hip_mid_y: Optional[float] = None
    shoulder_mid_x: Optional[float] = None
    shoulder_width_px: Optional[float] = None
    left_foot_x: Optional[float] = None
    left_foot_y: Optional[float] = None
    right_foot_x: Optional[float] = None
    right_foot_y: Optional[float] = None
    left_arm_abduction_deg: Optional[float] = None
    right_arm_abduction_deg: Optional[float] = None


class TWKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class TWLineDTO(BaseModel):
    a: float
    b: float


class TandemWalkResultDTO(BaseModel):
    steps: List[TWStepDTO]
    misstep_count: int
    arm_grab_count: int
    mean_deviation_cm: float
    max_deviation_cm: float
    step_time_mean_ms: float
    step_time_stddev_ms: float
    step_time_cv: float
    trunk_sway_range_px: float
    trunk_sway_range_cm: float
    classification: str
    duration_seconds: float
    termination: str
    incomplete: bool
    walking_line: Optional[TWLineDTO] = None
    mean_shoulder_width_px: float
    samples: List[TWFrameSampleDTO]
    keypoints: List[List[TWKeypointDTO]]
    capture_screenshot_data_url: Optional[str] = None
    patient_age: Optional[int] = None
    # Diagnostic extras.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class TandemWalkResponse(BaseModel):
    success: bool
    data: Optional[TandemWalkResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-tandem-walk", response_model=TandemWalkResponse)
async def analyze_tandem_walk_endpoint(
    video: UploadFile = File(...),
    patient_age: Optional[int] = Form(None),
) -> TandemWalkResponse:
    """Run the E1 Tandem Walk pipeline on an uploaded clip. The
    patient walks heel-to-toe toward the camera for 10 steps along a
    taped line. Validation gates mirror the other multi-rep modules.
    """
    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "tandem_walk.mp4",
        )
        log.info(
            "tandem_walk: file=%s age=%s size=%.2f MB",
            video.filename, patient_age, size_mb,
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine "
                f"can capture 10 footstrikes."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_tandem_walk(
            video_path=tmp_path,
            pose_options=pose_options,
            patient_age=patient_age,
        )

        return TandemWalkResponse(
            success=True,
            data=TandemWalkResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("tandem_walk validation: %s", e)
        return TandemWalkResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("tandem_walk analysis failed")
        return TandemWalkResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Pronator Drift (E2) — single-trial upload analysis
# ══════════════════════════════════════════════════════════════════════
# Frontal-view static hold (~20 s). Patient holds both arms extended
# forward at shoulder height, eyes closed; engine tracks per-arm
# vertical wrist drift from a baseline taken in the first stable
# second. Math mirrors lib/orthopedic/pronatorDrift.ts.
#
# 2D LIMITATION (carried through to the frontend report banner):
# the third classical pronator-drift sign — forearm rotation as
# the arm drops — is NOT measurable by a single 2D camera.
class PDArmDriftDTO(BaseModel):
    baseline_wrist_y_px: Optional[float] = None
    final_wrist_y_px: Optional[float] = None
    drift_px: Optional[float] = None
    drift_cm: Optional[float] = None
    drift_velocity_cm_per_sec: Optional[float] = None
    drift_cm_series: List[Optional[float]] = []


class PDFrameSampleDTO(BaseModel):
    t_ms: float
    left_wrist_y: Optional[float] = None
    right_wrist_y: Optional[float] = None
    left_shoulder_y: Optional[float] = None
    right_shoulder_y: Optional[float] = None
    shoulder_width_px: Optional[float] = None


class PDKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class PronatorDriftResultDTO(BaseModel):
    hold_duration_seconds: float
    mean_shoulder_width_px: float
    left: PDArmDriftDTO
    right: PDArmDriftDTO
    t_seconds_series: List[float]
    max_downward_drift_cm: float
    min_downward_drift_cm: float
    asymmetry_ratio: float
    asymmetry_absolute_cm: float
    classification: str
    termination: str
    incomplete: bool
    samples: List[PDFrameSampleDTO]
    keypoints: List[List[PDKeypointDTO]]
    capture_screenshot_data_url: Optional[str] = None
    # Diagnostic extras.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class PronatorDriftResponse(BaseModel):
    success: bool
    data: Optional[PronatorDriftResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-pronator-drift", response_model=PronatorDriftResponse)
async def analyze_pronator_drift_endpoint(
    video: UploadFile = File(...),
) -> PronatorDriftResponse:
    """Run the E2 Pronator Drift pipeline on an uploaded clip.
    Patient holds both arms extended forward at shoulder height with
    eyes closed; engine measures per-arm vertical drift. 2D system —
    rotation is not assessed (caveat surfaced in the report).
    """
    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "pronator_drift.mp4",
        )
        log.info(
            "pronator_drift: file=%s size=%.2f MB",
            video.filename, size_mb,
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended; the spec calls "
                f"for a 20 s hold."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_pronator_drift(
            video_path=tmp_path,
            pose_options=pose_options,
        )

        return PronatorDriftResponse(
            success=True,
            data=PronatorDriftResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("pronator_drift validation: %s", e)
        return PronatorDriftResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("pronator_drift analysis failed")
        return PronatorDriftResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Functional Reach (C6) — single-trial upload analysis
# ══════════════════════════════════════════════════════════════════════
# Lateral view, ~30 s clip. Patient stands side-on, raises the near
# arm to shoulder height, and reaches forward 3 times. Math mirrors
# lib/orthopedic/functionalReach.ts so live + upload produce
# identical reports.
#
# `calibration` is an optional JSON string carrying the frontend's
# height-based CalibrationResult (produced by HeightCalibrationStep).
# When omitted, the engine reports distances in relative pixel units
# only — no fall-risk classification (the explicit graceful-
# degradation path the spec calls for).
class FRFrameSampleDTO(BaseModel):
    t_ms: float
    wrist_x_px: Optional[float] = None
    wrist_y_px: Optional[float] = None
    shoulder_y_px: Optional[float] = None
    ankle_x_px: Optional[float] = None
    heel_y_px: Optional[float] = None
    foot_index_y_px: Optional[float] = None
    trunk_angle_deg: Optional[float] = None
    arm_raised: bool


class FRKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class FRTrialDTO(BaseModel):
    trial_index: int
    peak_sample_index: int
    peak_t_ms: float
    signed_displacement_px: float
    reach_px: float
    reach_cm: Optional[float] = None
    trunk_angle_at_peak_deg: Optional[float] = None
    validity: str
    invalidity_detail: Optional[str] = None
    window_start_index: int
    window_end_index: int
    max_heel_drift_px: float
    max_heel_drift_cm: Optional[float] = None
    max_ankle_drift_px: float
    max_ankle_drift_cm: Optional[float] = None


class FunctionalReachResultDTO(BaseModel):
    side_tested: str
    baseline_locked: bool
    baseline_locked_at_index: Optional[int] = None
    baseline_wrist_x_px: Optional[float] = None
    baseline_ankle_x_px: Optional[float] = None
    baseline_heel_y_px: Optional[float] = None
    trials: List[FRTrialDTO]
    best_valid_trial_index: Optional[int] = None
    best_valid_reach_px: Optional[float] = None
    best_valid_reach_cm: Optional[float] = None
    classification: Optional[str] = None
    # Provider-agnostic calibration dict — populated with the
    # frontend's height-based CalibrationResult when present, else
    # null (relative-units mode).
    calibration: Optional[Dict[str, Any]] = None
    duration_seconds: float
    termination: str
    samples: List[FRFrameSampleDTO]
    keypoints: List[List[FRKeypointDTO]]
    peak_screenshot_data_url: Optional[str] = None
    # Diagnostic extras.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class FunctionalReachResponse(BaseModel):
    success: bool
    data: Optional[FunctionalReachResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-functional-reach", response_model=FunctionalReachResponse)
async def analyze_functional_reach_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
    calibration: Optional[str] = Form(None),
    patient_height_cm: Optional[float] = Form(None),
) -> FunctionalReachResponse:
    """Run the C6 Functional Reach pipeline on an uploaded clip.

    `side` is the test arm (the arm nearest the camera in the
    lateral view).

    Calibration is supplied in one of two ways:
      1. `calibration` — a JSON-encoded CalibrationResult (live mode
         already locked one in via HeightCalibrationStep).
      2. `patient_height_cm` — the patient's standing height; the
         engine measures body pixel height from the standing window
         of the clip and derives pixels_per_cm itself.
    Neither is required; absent both the test runs in relative-units
    mode (no fall-risk classification).
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    # Parse optional client-supplied calibration. Accept anything
    # dict-shaped with a positive pixels_per_cm; the engine ignores
    # unknown keys.
    parsed_calibration: Optional[Dict[str, Any]] = None
    if calibration:
        import json as _json
        try:
            parsed_calibration = _json.loads(calibration)
            if not isinstance(parsed_calibration, dict):
                parsed_calibration = None
            else:
                ppc = parsed_calibration.get("pixels_per_cm")
                if not isinstance(ppc, (int, float)) or ppc <= 0:
                    parsed_calibration = None
        except (ValueError, TypeError):
            parsed_calibration = None

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "functional_reach.mp4",
        )
        log.info(
            "functional_reach: file=%s side=%s size=%.2f MB calibration=%s",
            video.filename, side, size_mb,
            "client" if parsed_calibration else "server-auto",
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine "
                f"can capture a baseline + 3 reaches."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_functional_reach(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
            calibration=parsed_calibration,
            patient_height_cm=patient_height_cm,
        )

        return FunctionalReachResponse(
            success=True,
            data=FunctionalReachResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("functional_reach validation: %s", e)
        return FunctionalReachResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("functional_reach analysis failed")
        return FunctionalReachResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# D3 Single-Leg Hop — forward hop for distance (per-leg)
# ══════════════════════════════════════════════════════════════════════
# Per-leg endpoint — caller invokes once per side. LSI (left vs
# right) is computed CLIENT-side after both calls return. Mirrors
# the Functional Reach endpoint architecturally: multipart form
# with `side` + optional `calibration` JSON + optional
# `patient_height_cm`; provider-agnostic CalibrationResult pass-
# through; reuses the engines.calibration module without
# duplication.
class SLHTrialDTO(BaseModel):
    trial_index: int
    takeoff_frame_index: int
    landing_frame_index: int
    takeoff_t_ms: float
    landing_t_ms: float
    hop_distance_px: float
    hop_distance_cm: Optional[float] = None
    valid: bool
    invalidation_reason: Optional[str] = None


class SingleLegHopResultDTO(BaseModel):
    side_tested: str
    patient_height_cm: Optional[float] = None
    calibration: Optional[Dict[str, Any]] = None
    baseline_ankle_y_px: float
    leg_length_px: float
    trials: List[SLHTrialDTO]
    best_valid_trial_index: Optional[int] = None
    best_valid_hop_px: Optional[float] = None
    best_valid_hop_cm: Optional[float] = None
    peak_screenshot_data_url: Optional[str] = None
    duration_seconds: float
    termination: str
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class SingleLegHopResponse(BaseModel):
    success: bool
    data: Optional[SingleLegHopResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-single-leg-hop", response_model=SingleLegHopResponse)
async def analyze_single_leg_hop_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
    calibration: Optional[str] = Form(None),
    patient_height_cm: Optional[float] = Form(None),
) -> SingleLegHopResponse:
    """Run the D3 Single-Leg Hop pipeline on an uploaded clip.

    `side` is the test (hopping) leg. The endpoint is per-leg —
    the caller invokes it once per side, then computes the Limb
    Symmetry Index client-side from the two best-valid distances.

    Calibration is supplied in one of two ways:
      1. `calibration` — a JSON-encoded CalibrationResult that
         the live-mode HeightCalibrationStep already locked in.
      2. `patient_height_cm` — the patient's standing height;
         the engine measures body pixel height from the standing
         window of the clip itself and derives pixels_per_cm.
    With neither, the test runs in relative-units mode (pixel
    distances only — no cm value and no LSI classification).
    """
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="Side must be 'left' or 'right'.",
        )

    parsed_calibration: Optional[Dict[str, Any]] = None
    if calibration:
        import json as _json
        try:
            parsed_calibration = _json.loads(calibration)
            if not isinstance(parsed_calibration, dict):
                parsed_calibration = None
            else:
                ppc = parsed_calibration.get("pixels_per_cm")
                if not isinstance(ppc, (int, float)) or ppc <= 0:
                    parsed_calibration = None
        except (ValueError, TypeError):
            parsed_calibration = None

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "single_leg_hop.mp4",
        )
        log.info(
            "single_leg_hop: file=%s side=%s size=%.2f MB calibration=%s",
            video.filename, side, size_mb,
            "client" if parsed_calibration else "server-auto",
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — hop event timing may be coarse."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine "
                f"can capture a stance + up to 3 hops."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_single_leg_hop(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
            calibration=parsed_calibration,
            patient_height_cm=patient_height_cm,
        )

        return SingleLegHopResponse(
            success=True,
            data=SingleLegHopResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("single_leg_hop validation: %s", e)
        return SingleLegHopResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("single_leg_hop analysis failed")
        return SingleLegHopResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# D4 Counter-Movement Jump — vertical jump for power (both legs)
# ══════════════════════════════════════════════════════════════════════
# Both legs together — NO side parameter, NO LSI. Single recording
# captures up to 3 jumps. Primary outcome is jump height (cm via
# height calibration); secondary is flight time (s) plus a physics
# cross-check on the height (h = g·t²/8). Mirrors the D3 endpoint
# architecture (multipart form, optional calibration + height,
# provider-agnostic CalibrationResult pass-through) minus the
# per-leg structure.
class CMJTrialDTO(BaseModel):
    trial_index: int
    takeoff_frame_index: int
    apex_frame_index: int
    landing_frame_index: int
    takeoff_t_ms: float
    landing_t_ms: float
    flight_time_sec: float
    jump_height_px: float
    jump_height_cm: Optional[float] = None
    physics_height_cm: float
    valid: bool
    invalidation_reason: Optional[str] = None


class CMJResultDTO(BaseModel):
    patient_height_cm: Optional[float] = None
    calibration: Optional[Dict[str, Any]] = None
    baseline_hip_y_px: float
    baseline_left_ankle_y_px: float
    baseline_right_ankle_y_px: float
    leg_length_px: float
    trials: List[CMJTrialDTO]
    best_valid_trial_index: Optional[int] = None
    best_valid_jump_px: Optional[float] = None
    best_valid_jump_cm: Optional[float] = None
    best_valid_flight_sec: Optional[float] = None
    mean_valid_jump_cm: Optional[float] = None
    mean_valid_flight_sec: Optional[float] = None
    peak_screenshot_data_url: Optional[str] = None
    duration_seconds: float
    termination: str
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class CMJResponse(BaseModel):
    success: bool
    data: Optional[CMJResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post(
    "/api/analyze-counter-movement-jump",
    response_model=CMJResponse,
)
async def analyze_counter_movement_jump_endpoint(
    video: UploadFile = File(...),
    calibration: Optional[str] = Form(None),
    patient_height_cm: Optional[float] = Form(None),
) -> CMJResponse:
    """Run the D4 CMJ pipeline on an uploaded clip. Both legs
    together — single recording, no side parameter.
    """
    parsed_calibration: Optional[Dict[str, Any]] = None
    if calibration:
        import json as _json
        try:
            parsed_calibration = _json.loads(calibration)
            if not isinstance(parsed_calibration, dict):
                parsed_calibration = None
            else:
                ppc = parsed_calibration.get("pixels_per_cm")
                if not isinstance(ppc, (int, float)) or ppc <= 0:
                    parsed_calibration = None
        except (ValueError, TypeError):
            parsed_calibration = None

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "counter_movement_jump.mp4",
        )
        log.info(
            "cmj: file=%s size=%.2f MB calibration=%s",
            video.filename, size_mb,
            "client" if parsed_calibration else "server-auto",
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — jump event timing may be coarse."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended so the engine "
                f"can capture a stance + up to 3 jumps."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_counter_movement_jump(
            video_path=tmp_path,
            pose_options=pose_options,
            calibration=parsed_calibration,
            patient_height_cm=patient_height_cm,
        )

        return CMJResponse(
            success=True,
            data=CMJResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("cmj validation: %s", e)
        return CMJResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("cmj analysis failed")
        return CMJResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# D2 Tuck Jump — Myer's Tuck Jump Assessment (injury-risk screen)
# ══════════════════════════════════════════════════════════════════════
# Frontal single-camera. Patient performs ~10 s of continuous tuck
# jumps. Scored against Myer's 10-item checklist; classification is
# good / moderate / poor based on the count of measurable fails.
# Items 5 (foot yaw) and 7 (contact noise) are honestly marked
# "not_assessed" — a single frontal camera cannot resolve them.
class TuckJumpChecklistItemDTO(BaseModel):
    index: int
    label: str
    status: Literal["pass", "fail", "not_assessed"]
    detail: Optional[str] = None


class TuckJumpJumpDTO(BaseModel):
    jump_index: int
    takeoff_frame_index: int
    apex_frame_index: int
    landing_frame_index: int
    takeoff_t_ms: float
    landing_t_ms: float
    flight_time_sec: float
    jump_height_px: float
    jump_height_cm: Optional[float] = None
    physics_height_cm: float
    landing_kfppa_left_deg: float
    landing_kfppa_right_deg: float
    landing_kfppa_worse_deg: float
    landing_ank_spread_px: float
    landing_ank_spread_ratio: float
    apex_l_thigh_rise_px: float
    apex_r_thigh_rise_px: float
    takeoff_side_delta_ms: float
    landing_side_delta_ms: float
    grounded_since_prev_ms: Optional[float] = None
    landing_ank_left_x_px: float
    landing_ank_right_x_px: float


class TuckJumpResultDTO(BaseModel):
    patient_height_cm: Optional[float] = None
    calibration: Optional[Dict[str, Any]] = None
    baseline_hip_y_px: float
    baseline_left_ankle_y_px: float
    baseline_right_ankle_y_px: float
    baseline_ank_spread_px: float
    baseline_shoulder_hip_span_px: float
    leg_length_px: float
    jumps: List[TuckJumpJumpDTO]
    jump_count: int
    mean_jump_height_px: float
    mean_jump_height_cm: Optional[float] = None
    mean_valgus_worse_deg: float
    max_valgus_worse_deg: float
    height_fade_frac: float
    valgus_growth_deg: float
    footprint_drift_frac: float
    pause_gap_max_ms: float
    duration_seconds: float
    checklist: List[TuckJumpChecklistItemDTO]
    measurable_fails: int
    classification: Literal["good", "moderate", "poor"]
    peak_screenshot_data_url: Optional[str] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class TuckJumpResponse(BaseModel):
    success: bool
    data: Optional[TuckJumpResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post(
    "/api/analyze-tuck-jump",
    response_model=TuckJumpResponse,
)
async def analyze_tuck_jump_endpoint(
    video: UploadFile = File(...),
    calibration: Optional[str] = Form(None),
    patient_height_cm: Optional[float] = Form(None),
) -> TuckJumpResponse:
    """Run the D2 Tuck Jump pipeline on an uploaded clip. Continuous
    tuck-jump session (~10 s) — no side parameter.
    """
    parsed_calibration: Optional[Dict[str, Any]] = None
    if calibration:
        import json as _json
        try:
            parsed_calibration = _json.loads(calibration)
            if not isinstance(parsed_calibration, dict):
                parsed_calibration = None
            else:
                ppc = parsed_calibration.get("pixels_per_cm")
                if not isinstance(ppc, (int, float)) or ppc <= 0:
                    parsed_calibration = None
        except (ValueError, TypeError):
            parsed_calibration = None

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "tuck_jump.mp4",
        )
        log.info(
            "tuck_jump: file=%s size=%.2f MB calibration=%s",
            video.filename, size_mb,
            "client" if parsed_calibration else "server-auto",
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — jump event timing may be coarse."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — Myer's TJA "
                f"recommends a continuous ~10 s tuck-jump session for full "
                f"fatigue-effect scoring."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_tuck_jump(
            video_path=tmp_path,
            pose_options=pose_options,
            calibration=parsed_calibration,
            patient_height_cm=patient_height_cm,
        )

        return TuckJumpResponse(
            success=True,
            data=TuckJumpResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("tuck_jump validation: %s", e)
        return TuckJumpResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("tuck_jump analysis failed")
        return TuckJumpResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# B2 Overhead Squat — NASM/FMS-style overhead squat assessment
# ══════════════════════════════════════════════════════════════════════
# Frontal single-camera. Patient performs 3-5 slow overhead squats
# (arms straight overhead) to about parallel depth. Scored against a
# 7-item checklist (5 measurable + 2 not_assessed — torso lean and
# heel rise require sagittal / feet close-up). Classification is
# good / moderate / poor based on measurable fails.
class OverheadSquatChecklistItemDTO(BaseModel):
    index: int
    label: str
    status: Literal["pass", "fail", "not_assessed"]
    detail: Optional[str] = None


class OverheadSquatRepDTO(BaseModel):
    rep_index: int
    descent_start_frame_index: int
    bottom_frame_index: int
    return_frame_index: int
    descent_start_t_ms: float
    bottom_t_ms: float
    return_t_ms: float
    depth_px: float
    depth_cm: Optional[float] = None
    depth_frac_of_leg: float
    bottom_kfppa_left_deg: float
    bottom_kfppa_right_deg: float
    bottom_kfppa_worse_deg: float
    bottom_pelvic_tilt_px: float
    bottom_pelvic_tilt_frac: float
    bottom_ank_spread_px: float
    bottom_ank_spread_ratio: float
    bottom_l_arm_overhead_frac: Optional[float] = None
    bottom_r_arm_overhead_frac: Optional[float] = None
    bottom_worst_arm_overhead_frac: Optional[float] = None


class OverheadSquatResultDTO(BaseModel):
    patient_height_cm: Optional[float] = None
    calibration: Optional[Dict[str, Any]] = None
    baseline_hip_y_px: float
    baseline_shoulder_y_px: float
    baseline_wrist_y_px: Optional[float] = None
    baseline_ank_spread_px: float
    baseline_hip_span_px: float
    leg_length_px: float
    trunk_length_px: float
    reps: List[OverheadSquatRepDTO]
    rep_count: int
    mean_depth_frac: float
    max_depth_frac: float
    max_depth_cm: Optional[float] = None
    mean_valgus_worse_deg: float
    max_valgus_worse_deg: float
    max_pelvic_tilt_frac: float
    mean_ank_spread_ratio: float
    min_arm_overhead_frac: Optional[float] = None
    duration_seconds: float
    checklist: List[OverheadSquatChecklistItemDTO]
    measurable_fails: int
    classification: Literal["good", "moderate", "poor"]
    peak_screenshot_data_url: Optional[str] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class OverheadSquatResponse(BaseModel):
    success: bool
    data: Optional[OverheadSquatResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post(
    "/api/analyze-overhead-squat",
    response_model=OverheadSquatResponse,
)
async def analyze_overhead_squat_endpoint(
    video: UploadFile = File(...),
    calibration: Optional[str] = Form(None),
    patient_height_cm: Optional[float] = Form(None),
) -> OverheadSquatResponse:
    """Run the B2 Overhead Squat pipeline on an uploaded clip. 3-5
    slow squats, arms overhead — no side parameter.
    """
    parsed_calibration: Optional[Dict[str, Any]] = None
    if calibration:
        import json as _json
        try:
            parsed_calibration = _json.loads(calibration)
            if not isinstance(parsed_calibration, dict):
                parsed_calibration = None
            else:
                ppc = parsed_calibration.get("pixels_per_cm")
                if not isinstance(ppc, (int, float)) or ppc <= 0:
                    parsed_calibration = None
        except (ValueError, TypeError):
            parsed_calibration = None

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "overhead_squat.mp4",
        )
        log.info(
            "overhead_squat: file=%s size=%.2f MB calibration=%s",
            video.filename, size_mb,
            "client" if parsed_calibration else "server-auto",
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — rep-bottom timing may be coarse."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — the overhead "
                f"squat assessment recommends 3-5 slow reps (~10-15 s total)."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_overhead_squat(
            video_path=tmp_path,
            pose_options=pose_options,
            calibration=parsed_calibration,
            patient_height_cm=patient_height_cm,
        )

        return OverheadSquatResponse(
            success=True,
            data=OverheadSquatResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("overhead_squat validation: %s", e)
        return OverheadSquatResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("overhead_squat analysis failed")
        return OverheadSquatResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# 5x Sit-to-Stand — Test C2, single-trial upload analysis
# ══════════════════════════════════════════════════════════════════════
# SINGLE trial — no `side` form field. The frontend uploads ONE
# video and renders a flat SitToStandResult (no L/R split). Math +
# classification cutoffs mirror lib/orthopedic/sitToStand.ts so
# live + upload produce identical reports.
class STSRepDTO(BaseModel):
    rep_index: int
    duration_seconds: float
    min_knee_angle_deg: float


class STSFrameSampleDTO(BaseModel):
    t_ms: float
    hip_mid_y: Optional[float] = None
    knee_angle_deg: Optional[float] = None
    arms_crossed: bool


class STSKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class SitToStandResultDTO(BaseModel):
    total_time_seconds: float
    reps: List[STSRepDTO]
    rep_durations: List[float]
    cv_percent: float
    classification: str
    fatigue_flag: bool
    arm_uncrossed_flag: bool
    termination: str
    incomplete: bool
    trial_duration_seconds: float
    samples: List[STSFrameSampleDTO]
    keypoints: List[List[STSKeypointDTO]]
    last_rep_screenshot_data_url: Optional[str] = None
    # Extras — not in the strict TS SitToStandResult shape but useful
    # for the response envelope + future report enrichment.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None


class SitToStandResponse(BaseModel):
    success: bool
    data: Optional[SitToStandResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-sit-to-stand", response_model=SitToStandResponse)
async def analyze_sit_to_stand_endpoint(
    video: UploadFile = File(...),
    patient_age: Optional[int] = Form(None),
) -> SitToStandResponse:
    """Run the 5x Sit-to-Stand (Test C2) pipeline on an uploaded
    clip. Single trial — no `side` form field. Validation gates
    mirror the TUG / Trendelenburg / SLS endpoints (same gait
    MediaPipe pipeline downstream).
    """
    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        # 1) File-size gate
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(
            contents, video.filename or "sit_to_stand.mp4",
        )
        log.info(
            "sit_to_stand: file=%s size=%.2f MB age=%s",
            video.filename, size_mb, patient_age,
        )

        # 2) FPS + duration gate
        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended for a complete "
                f"5x sit-to-stand trial."
            )

        # 3) Run the analysis (reuses gait MediaPipe options).
        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_sit_to_stand(
            video_path=tmp_path,
            pose_options=pose_options,
        )

        return SitToStandResponse(
            success=True,
            data=SitToStandResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("sit_to_stand validation: %s", e)
        return SitToStandResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("sit_to_stand analysis failed")
        return SitToStandResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# 30-Second Chair Stand — Test C3, single-trial upload analysis
# ══════════════════════════════════════════════════════════════════════
# Single trial, timer-driven. Sit↔stand 2-state machine + CDC STEADI
# age+sex norm lookup. Frontend uploads ONE video; backend analyzes
# the first 30s and returns rep count + per-rep metrics + classification.
class CS30RepDTO(BaseModel):
    rep_index: int
    duration_seconds: float
    min_knee_angle_deg: float


class CS30FrameSampleDTO(BaseModel):
    t_ms: float
    hip_mid_y: Optional[float] = None
    knee_angle_deg: Optional[float] = None
    arms_crossed: bool


class CS30KeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class ChairStand30sResultDTO(BaseModel):
    rep_count: int
    reps: List[CS30RepDTO]
    rep_durations: List[float]
    mean_rep_duration_sec: float
    depth_sd_deg: float
    fatigue_slope_sec_per_rep: float
    norm_threshold: int
    norm_band_label: str
    norm_comparable: bool
    classification: str
    arm_uncrossed_flag: bool
    termination: str
    trial_duration_seconds: float
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    samples: List[CS30FrameSampleDTO]
    keypoints: List[List[CS30KeypointDTO]]
    last_rep_screenshot_data_url: Optional[str] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    valid_frames: Optional[int] = None
    interpretation: Optional[str] = None
    norm_passed: Optional[bool] = None


class ChairStand30sResponse(BaseModel):
    success: bool
    data: Optional[ChairStand30sResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-chair-stand-30s", response_model=ChairStand30sResponse)
async def analyze_chair_stand_30s_endpoint(
    video: UploadFile = File(...),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
) -> ChairStand30sResponse:
    """Run the 30-Second Chair Stand (Test C3) pipeline. Single
    trial — no `side` form field. CDC STEADI age + sex norm
    comparison applied server-side."""
    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "chair_stand_30s.mp4")
        log.info(
            "chair_stand_30s: file=%s size=%.2f MB age=%s sex=%s",
            video.filename, size_mb, patient_age, patient_sex,
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not determine video frame rate. "
                    "Please upload a different file."
                ),
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None

        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required. "
                    f"Recommended: {RECOMMENDED_FPS}+ FPS."
                ),
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video length. Please upload a different file.",
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended for a complete "
                f"30-second chair stand trial."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_chair_stand_30s(
            video_path=tmp_path,
            pose_options=pose_options,
            patient_age=patient_age,
            patient_sex=patient_sex,
        )

        return ChairStand30sResponse(
            success=True,
            data=ChairStand30sResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("chair_stand_30s validation: %s", e)
        return ChairStand30sResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("chair_stand_30s analysis failed")
        return ChairStand30sResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Single-Leg Stance — Test C5, per-trial upload analysis
# ══════════════════════════════════════════════════════════════════════
class SLSStanceFrameSampleDTO(BaseModel):
    t_ms: float
    hip_x: Optional[float] = None
    hip_y: Optional[float] = None
    trunk_lean_deg: Optional[float] = None


class SLSStanceKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class SLSStancePointDTO(BaseModel):
    x: float
    y: float


class SingleLegStanceTrialResultDTO(BaseModel):
    side: str
    condition: str
    hold_seconds: float
    hold_capped_at: float
    termination: str
    norm_threshold_sec: float
    norm_band_label: str
    norm_comparable: bool
    classification: str
    sway_path_px: float
    sway_95_ellipse_px2: float
    mean_trunk_lean_deg: float
    max_trunk_lean_deg: float
    hip_path: List[SLSStancePointDTO]
    samples: List[SLSStanceFrameSampleDTO]
    keypoints: List[List[SLSStanceKeypointDTO]]
    screenshot_data_url: Optional[str] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    interpretation: Optional[str] = None


class SingleLegStanceResponse(BaseModel):
    success: bool
    data: Optional[SingleLegStanceTrialResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-single-leg-stance", response_model=SingleLegStanceResponse)
async def analyze_single_leg_stance_endpoint(
    video: UploadFile = File(...),
    side: str = Form(...),
    condition: str = Form(...),
    patient_age: Optional[int] = Form(None),
) -> SingleLegStanceResponse:
    """Run the Single-Leg Stance pipeline on ONE (side, condition)
    trial. `side` ∈ {'left', 'right'} (stance leg). `condition` ∈
    {'eyes_open', 'eyes_closed'}."""
    if side not in ("left", "right"):
        raise HTTPException(status_code=400, detail="Side must be 'left' or 'right'.")
    if condition not in ("eyes_open", "eyes_closed"):
        raise HTTPException(status_code=400, detail="Condition must be 'eyes_open' or 'eyes_closed'.")

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB.",
            )

        tmp_path = save_uploaded_video(contents, video.filename or "single_leg_stance.mp4")
        log.info(
            "single_leg_stance: file=%s side=%s condition=%s size=%.2f MB age=%s",
            video.filename, side, condition, size_mb, patient_age,
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine video frame rate. Please upload a different file.",
            )

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=f"Video frame rate too low ({probe_fps:.1f} FPS). Minimum {MIN_REQUIRED_FPS} FPS required.",
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = (
                f"Note: Video is {probe_fps:.1f} FPS, below recommended "
                f"{RECOMMENDED_FPS} FPS — results may be less accurate."
            )
        if probe_total_frames <= 0:
            raise HTTPException(status_code=400, detail="Could not determine video length.")

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long ({duration_seconds:.1f}s). Maximum {MAX_GAIT_DURATION_SEC} seconds allowed.",
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = (
                f"Video is short ({duration_seconds:.1f}s) — at least "
                f"{MIN_GAIT_DURATION_SEC} seconds recommended."
            )

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_single_leg_stance(
            video_path=tmp_path,
            pose_options=pose_options,
            side=side,
            condition=condition,
            patient_age=patient_age,
        )

        return SingleLegStanceResponse(
            success=True,
            data=SingleLegStanceTrialResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("single_leg_stance validation: %s", e)
        return SingleLegStanceResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("single_leg_stance analysis failed")
        return SingleLegStanceResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# 4-Stage Balance — Test C4, per-stage upload analysis
# ══════════════════════════════════════════════════════════════════════
class FSBFrameSampleDTO(BaseModel):
    t_ms: float
    hip_x: Optional[float] = None
    hip_y: Optional[float] = None
    ankle_l_x: Optional[float] = None
    ankle_l_y: Optional[float] = None
    ankle_r_x: Optional[float] = None
    ankle_r_y: Optional[float] = None


class FSBKeypointDTO(BaseModel):
    x: float
    y: float
    score: float


class FSBPointDTO(BaseModel):
    x: float
    y: float


class FourStageBalanceStageResultDTO(BaseModel):
    stage: int
    outcome: str
    hold_seconds: float
    failure_mode: Optional[str] = None
    sway_path_px: float
    sway_95_ellipse_px2: float
    hip_path: List[FSBPointDTO]
    samples: List[FSBFrameSampleDTO]
    keypoints: List[List[FSBKeypointDTO]]
    screenshot_data_url: Optional[str] = None
    duration_seconds: float
    fps: Optional[float] = None
    total_frames: Optional[int] = None


class FourStageBalanceResponse(BaseModel):
    success: bool
    data: Optional[FourStageBalanceStageResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-four-stage-balance", response_model=FourStageBalanceResponse)
async def analyze_four_stage_balance_endpoint(
    video: UploadFile = File(...),
    stage: int = Form(...),
) -> FourStageBalanceResponse:
    """Run the 4-Stage Balance (Test C4) pipeline on ONE stage. The
    frontend uploads stages 1-4 separately and assembles the
    SessionResult client-side, applying the CDC stop-at-first-failure
    rule for downstream stages."""
    if stage not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="Stage must be 1, 2, 3, or 4.")

    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_GAIT_FILE_SIZE_MB} MB.",
            )

        tmp_path = save_uploaded_video(contents, video.filename or "four_stage_balance.mp4")
        log.info(
            "four_stage_balance: file=%s stage=%d size=%.2f MB",
            video.filename, stage, size_mb,
        )

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(status_code=400, detail="Could not determine video frame rate.")

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=f"Video frame rate too low ({probe_fps:.1f} FPS). Minimum {MIN_REQUIRED_FPS} FPS required.",
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = f"Note: Video is {probe_fps:.1f} FPS, below recommended {RECOMMENDED_FPS} FPS."
        if probe_total_frames <= 0:
            raise HTTPException(status_code=400, detail="Could not determine video length.")

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long ({duration_seconds:.1f}s). Maximum {MAX_GAIT_DURATION_SEC} seconds allowed.",
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = f"Video is short ({duration_seconds:.1f}s)."

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_four_stage_balance(
            video_path=tmp_path,
            pose_options=pose_options,
            stage=stage,
        )

        return FourStageBalanceResponse(
            success=True,
            data=FourStageBalanceStageResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("four_stage_balance validation: %s", e)
        return FourStageBalanceResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("four_stage_balance analysis failed")
        return FourStageBalanceResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# SPPB Component 2 — 4-metre Gait Speed upload analysis
# ══════════════════════════════════════════════════════════════════════
class SPPBGaitSpeedResultDTO(BaseModel):
    duration_sec: float
    speed_mps: float
    score: int
    completed: bool
    started_at_ms: int
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    interpretation: Optional[str] = None


class SPPBGaitSpeedResponse(BaseModel):
    success: bool
    data: Optional[SPPBGaitSpeedResultDTO] = None
    error: Optional[str] = None
    fps_warning: Optional[str] = None
    duration_warning: Optional[str] = None


@app.post("/api/analyze-sppb-gait-speed", response_model=SPPBGaitSpeedResponse)
async def analyze_sppb_gait_speed_endpoint(
    video: UploadFile = File(...),
) -> SPPBGaitSpeedResponse:
    """Run the SPPB 4-metre gait-speed pipeline on an uploaded clip."""
    tmp_path: Optional[str] = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_GAIT_FILE_SIZE_MB} MB.",
            )

        tmp_path = save_uploaded_video(contents, video.filename or "sppb_gait_speed.mp4")
        log.info("sppb_gait_speed: file=%s size=%.2f MB", video.filename, size_mb)

        probe = cv2.VideoCapture(tmp_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0:
            raise HTTPException(status_code=400, detail="Could not determine video frame rate.")

        fps_warning: Optional[str] = None
        duration_warning: Optional[str] = None
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=f"Video frame rate too low ({probe_fps:.1f} FPS). Minimum {MIN_REQUIRED_FPS} FPS required.",
            )
        if probe_fps < RECOMMENDED_FPS:
            fps_warning = f"Note: Video is {probe_fps:.1f} FPS, below recommended {RECOMMENDED_FPS} FPS."
        if probe_total_frames <= 0:
            raise HTTPException(status_code=400, detail="Could not determine video length.")

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long ({duration_seconds:.1f}s). Maximum {MAX_GAIT_DURATION_SEC} seconds.",
            )
        if duration_seconds < MIN_GAIT_DURATION_SEC:
            duration_warning = f"Video is short ({duration_seconds:.1f}s)."

        pose_options = _build_gait_pose_options()
        result: Dict[str, Any] = analyze_sppb_gait_speed(
            video_path=tmp_path,
            pose_options=pose_options,
        )

        return SPPBGaitSpeedResponse(
            success=True,
            data=SPPBGaitSpeedResultDTO(**result),
            error=None,
            fps_warning=fps_warning,
            duration_warning=duration_warning,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("sppb_gait_speed validation: %s", e)
        return SPPBGaitSpeedResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("sppb_gait_speed analysis failed")
        return SPPBGaitSpeedResponse(
            success=False, data=None, error=f"Analysis failed: {e}",
        )
    finally:
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Ankle (dorsiflexion + plantarflexion) — backend MediaPipe
# ══════════════════════════════════════════════════════════════════════
# All ankle biomech goes through the backend MediaPipe pipeline because
# the foot landmarks (heel + foot_index, MediaPipe kp 29-32) are
# required for an accurate ankle-joint angle — MoveNet's 17-keypoint
# set doesn't have them, so the previous browser-side pipeline could
# only measure shin-from-vertical (returning ~90° for seated patients).
# This endpoint reuses gait's pose pipeline + new shin/foot-vector
# math in ankle_engine.py.
@app.post("/api/analyze-ankle", response_model=BiomechResponse)
async def analyze_ankle(
    video: UploadFile = File(...),
    movement_type: str = Form(...),    # "flexion" (dorsi) or "extension" (plantar)
    side: str = Form("right"),         # "left" or "right"
    patient_name: Optional[str] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> BiomechResponse:
    """Run the backend MediaPipe ankle pipeline on an uploaded clip.

    `recording_duration_ms` is supplied by the live-record path on the
    frontend. MediaRecorder-produced WebMs often have broken duration
    headers, so we use TUG's `_ensure_decodable_video` helper to rewrite
    the container with a proper FPS computed from frames/duration.
    """
    # Import locally to avoid a top-level cycle and to keep the helper
    # owned by tug_engine (single source of truth for the WebM repair).
    from engines.orthopedic.tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "ankle.mp4")
        log.info(
            "ankle: file=%s size=%.2f MB movement=%s side=%s recording_ms=%s",
            video.filename, size_mb, movement_type, side, recording_duration_ms,
        )

        # Repair MediaRecorder WebMs with broken duration headers BEFORE
        # the cv2 FPS probe — otherwise the probe returns 0 and we'd
        # reject otherwise-valid live recordings.
        processed_path, fixed_path_cleanup = _ensure_decodable_video(
            tmp_path, recording_duration_ms,
        )

        probe = cv2.VideoCapture(processed_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0 or probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read video metadata. "
                    "If this was a live recording, please retry; otherwise "
                    "upload a different file."
                ),
            )
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required."
                ),
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )

        # Suppress unused warning for the patient name (kept on the
        # API for parity with other biomech endpoints; not used by
        # the report renderer because the patient header is set on
        # the frontend from usePatientContext).
        _ = patient_name

        pose_options = _build_gait_pose_options()
        result = analyze_ankle_engine(
            video_path=processed_path,
            pose_options=pose_options,
            movement=movement_type,
            side=side,
        )
        return BiomechResponse(success=True, data=result, error=None)

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("ankle validation: %s", e)
        return BiomechResponse(success=False, data=None, error=str(e))
    except Exception as e:
        log.exception("ankle analysis failed")
        return BiomechResponse(success=False, data=None, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
        # _ensure_decodable_video returns a second path only when it
        # actually wrote a repaired file — clean that up too.
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


# ══════════════════════════════════════════════════════════════════════
# SPPB — Component 1 (Balance) — backend MediaPipe
# ══════════════════════════════════════════════════════════════════════
# Only Component 1 of the SPPB battery is processed server-side. The
# spec calls for foot-precise stage detection (side-by-side, semi-
# tandem, tandem) which needs heel + foot_index landmarks — MoveNet
# doesn't provide those. Components 2 (gait speed) and 3 (chair
# stand) of SPPB continue to use the existing MoveNet-based live
# orchestrator path on the frontend, unchanged.
#
# Returns a JSON shape directly consumable by the frontend's
# `buildBalanceComponent()` so the composite 0-12 SPPB scoring
# continues to use the existing scorer.
@app.post("/api/sppb/balance")
async def analyze_sppb_balance_endpoint(
    video: UploadFile = File(...),
    recording_duration_ms: Optional[int] = Form(None),
) -> dict:
    """Run SPPB Balance analysis on an uploaded clip recording all
    three stages in sequence.

    `recording_duration_ms` is supplied by the frontend's MediaRecorder
    path — WebM headers often lack a duration, which makes cv2's
    CAP_PROP_FPS probe return 0. We use the same WebM-repair helper
    TUG uses (tug_engine._ensure_decodable_video)."""
    from engines.orthopedic.tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "sppb_balance.mp4")
        log.info(
            "sppb_balance: file=%s size=%.2f MB recording_ms=%s",
            video.filename, size_mb, recording_duration_ms,
        )

        # WebM-header repair before the cv2 probe so live-recorded
        # MediaRecorder clips with missing duration don't fail validation.
        processed_path, fixed_path_cleanup = _ensure_decodable_video(
            tmp_path, recording_duration_ms,
        )

        probe = cv2.VideoCapture(processed_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0 or probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read video metadata. "
                    "If this was a live recording, please retry."
                ),
            )
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video frame rate too low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required."
                ),
            )
        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Video too long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )

        pose_options = _build_gait_pose_options()
        result = analyze_sppb_balance(
            video_path=processed_path,
            pose_options=pose_options,
        )
        return {"success": True, "data": result, "error": None}

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("sppb_balance validation: %s", e)
        return {"success": False, "data": None, "error": str(e)}
    except Exception as e:
        log.exception("sppb_balance analysis failed")
        return {"success": False, "data": None, "error": f"Analysis failed: {e}"}
    finally:
        cleanup_temp_file(tmp_path)
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


@app.post("/api/analyze-shoulder", response_model=BiomechResponse)
async def analyze_shoulder(
    video: UploadFile = File(...),
    movement_type: str = Form(...),
    side: str = Form("right"),
    patient_name: Optional[str] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> BiomechResponse:
    """Shoulder ROM upload analysis on backend MediaPipe BlazePose Full.

    movement_type accepts:
      • "flexion_extension" — merged test that captures both peaks
        in one trial (returned with secondary_peak_* fields).
      • Any single key from SHOULDER_NORMAL_RANGES (legacy
        single-direction path).
    side ∈ left / right.

    Validation gates (raise HTTPException → frontend maps to user-
    facing error text):
      • file_too_large            (>100 MB)         → 413
      • fps_too_low               (<24 FPS)         → 400
      • duration_too_long         (>60 s)           → 400
      • video_too_short           (<2 s)            → 400
      • poor_visibility           (engine raises)   → 400

    On success returns BiomechResponse with the extended BiomechData
    schema — merged movements populate secondary_peak_* + primary/
    secondary labels for the dual-row report.
    """
    # Imported locally to avoid the top-level circular-import risk that
    # the ankle endpoint dodges the same way.
    from engines.orthopedic.tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        movement = movement_type.lower().strip()
        # Merged movements (one recording, both directions) live as
        # their own movement IDs and aren't part of SHOULDER_NORMAL_
        # RANGES (which is structured for single-direction movements
        # with a single reference range). The engine knows how to
        # dispatch each merged ID on its own.
        _ALLOWED_MERGED = (
            "flexion_extension",
            "abduction_adduction",
            "rotation",
        )
        if movement not in _ALLOWED_MERGED and movement not in SHOULDER_NORMAL_RANGES:
            return BiomechResponse(
                success=False,
                error=(
                    f"Unknown shoulder movement '{movement_type}'. "
                    f"Allowed: {sorted(_ALLOWED_MERGED)} or "
                    f"{sorted(SHOULDER_NORMAL_RANGES.keys())}"
                ),
            )
        side_lc = side.lower().strip()
        if side_lc not in ("left", "right"):
            return BiomechResponse(
                success=False, error=f"side must be 'left' or 'right', got '{side}'.",
            )

        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"file_too_large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "shoulder.mp4")
        log.info(
            "shoulder: file=%s size=%.2f MB movement=%s side=%s recording_ms=%s",
            video.filename, size_mb, movement, side_lc, recording_duration_ms,
        )

        # Repair MediaRecorder WebMs with broken duration headers
        # before the cv2 FPS probe (same as ankle endpoint).
        processed_path, fixed_path_cleanup = _ensure_decodable_video(
            tmp_path, recording_duration_ms,
        )

        probe = cv2.VideoCapture(processed_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0 or probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read video metadata. Please retry, "
                    "or upload a different file."
                ),
            )
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"fps_too_low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required."
                ),
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds < 2.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"video_too_short ({duration_seconds:.1f}s). "
                    f"Minimum 2 seconds required."
                ),
            )
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"duration_too_long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )

        _ = patient_name  # parity with other biomech endpoints; not used

        pose_options = _build_gait_pose_options()
        result = analyze_shoulder_engine(
            video_path=processed_path,
            pose_options=pose_options,
            movement=movement,
            side=side_lc,
        )
        return BiomechResponse(success=True, data=result, error=None)

    except HTTPException:
        raise
    except ValueError as e:
        msg = str(e)
        log.warning("shoulder validation: %s", msg)
        # Engine raises "poor_visibility" directly; surface it 1:1.
        if msg == "poor_visibility":
            raise HTTPException(status_code=400, detail="poor_visibility")
        # Wrong-side selection — surface the user-facing message as
        # the HTTPException detail so the frontend displays it
        # verbatim (matches the poor_visibility pattern).
        if msg.startswith("Requested side"):
            raise HTTPException(status_code=400, detail=msg)
        # Lateral-view rejection for ab/ad test (side profile
        # filmed instead of frontal). Same HTTP 400 surface.
        if msg.startswith("Camera angle"):
            raise HTTPException(status_code=400, detail=msg)
        # Rotation test couldn't lock a calibration baseline — the
        # patient must hold neutral (elbow at 90°, forearm pointing
        # at camera) for ~5 frames at the start of the recording.
        if msg.startswith("Neutral pose"):
            raise HTTPException(status_code=400, detail=msg)
        return BiomechResponse(success=False, error=msg)
    except Exception as e:
        log.exception("shoulder analysis failed")
        return BiomechResponse(success=False, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


@app.post("/api/analyze-knee", response_model=BiomechResponse)
async def analyze_knee(
    video: UploadFile = File(...),
    movement_type: str = Form(...),
    side: str = Form("right"),
    patient_name: Optional[str] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> BiomechResponse:
    """Knee ROM upload analysis on backend MediaPipe BlazePose Full.

    movement_type accepts "flexion_extension" only — the single
    merged knee test surfaced in the UI. Per-frame min/max tracker
    captures peak flexion (max bent amount) and peak extension
    (min bent amount = residual flexion at the patient's
    straightest position). No direction detection, no calibration
    baseline — knee is the simplest of the merged backend tests.

    Validation gates (raise HTTPException → frontend maps to user-
    facing error text):
      • file_too_large            (>100 MB)         → 413
      • fps_too_low               (<24 FPS)         → 400
      • duration_too_long         (>60 s)           → 400
      • video_too_short           (<2 s)            → 400
      • poor_visibility           (engine raises)   → 400
      • Requested side …          (engine raises)   → 400

    `recording_duration_ms` is supplied by the live-record path
    on the frontend. MediaRecorder-produced WebMs often have
    broken duration headers, so we use TUG's `_ensure_decodable_
    video` helper to rewrite the container before the cv2 probe
    (mirrors the ankle / shoulder endpoints).
    """
    from engines.orthopedic.tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        movement = movement_type.lower().strip()
        if movement != "flexion_extension":
            return BiomechResponse(
                success=False,
                error=(
                    f"Unknown knee movement '{movement_type}'. "
                    f"Allowed: 'flexion_extension'."
                ),
            )
        side_lc = side.lower().strip()
        if side_lc not in ("left", "right"):
            return BiomechResponse(
                success=False,
                error=f"side must be 'left' or 'right', got '{side}'.",
            )

        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"file_too_large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "knee.mp4")
        log.info(
            "knee: file=%s size=%.2f MB movement=%s side=%s recording_ms=%s",
            video.filename, size_mb, movement, side_lc, recording_duration_ms,
        )

        # Repair MediaRecorder WebMs with broken duration headers
        # before the cv2 FPS probe (same as ankle / shoulder).
        processed_path, fixed_path_cleanup = _ensure_decodable_video(
            tmp_path, recording_duration_ms,
        )

        probe = cv2.VideoCapture(processed_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0 or probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read video metadata. Please retry, "
                    "or upload a different file."
                ),
            )
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"fps_too_low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required."
                ),
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds < 2.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"video_too_short ({duration_seconds:.1f}s). "
                    f"Minimum 2 seconds required."
                ),
            )
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"duration_too_long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )

        _ = patient_name  # accepted for parity with sibling endpoints

        pose_options = _build_gait_pose_options()
        result = analyze_knee_engine(
            video_path=processed_path,
            pose_options=pose_options,
            movement=movement,
            side=side_lc,
        )
        return BiomechResponse(success=True, data=result, error=None)

    except HTTPException:
        raise
    except ValueError as e:
        msg = str(e)
        log.warning("knee validation: %s", msg)
        if msg == "poor_visibility":
            raise HTTPException(status_code=400, detail="poor_visibility")
        if msg.startswith("Requested side"):
            raise HTTPException(status_code=400, detail=msg)
        return BiomechResponse(success=False, error=msg)
    except Exception as e:
        log.exception("knee analysis failed")
        return BiomechResponse(success=False, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


@app.post("/api/analyze-hip", response_model=BiomechResponse)
async def analyze_hip(
    video: UploadFile = File(...),
    movement_type: str = Form(...),
    side: str = Form("right"),
    patient_name: Optional[str] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> BiomechResponse:
    """Hip ROM upload analysis on backend MediaPipe BlazePose Full.

    movement_type accepts "flexion" only — extension, internal /
    external rotation still run through the browser MoveNet path
    and will migrate in a follow-up. Per-frame max-tracker captures
    peak hip flexion (180° − interior angle between trunk and
    thigh vectors); no direction detection / no calibration —
    same complexity tier as the merged knee test.

    Validation gates (raise HTTPException → frontend maps to
    user-facing error text):
      • file_too_large            (>100 MB)         → 413
      • fps_too_low               (<24 FPS)         → 400
      • duration_too_long         (>60 s)           → 400
      • video_too_short           (<2 s)            → 400
      • poor_visibility           (engine raises)   → 400
      • Requested side …          (engine raises)   → 400

    `recording_duration_ms` is supplied by the live-record path;
    MediaRecorder WebMs often have broken duration headers so we
    repair via tug_engine._ensure_decodable_video (same as ankle /
    shoulder / knee / neck).
    """
    from engines.orthopedic.tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        movement = movement_type.lower().strip()
        _ALLOWED_HIP = ("flexion", "extension", "rotation")
        if movement not in _ALLOWED_HIP:
            return BiomechResponse(
                success=False,
                error=(
                    f"Unknown hip movement '{movement_type}'. "
                    f"Allowed: {sorted(_ALLOWED_HIP)}."
                ),
            )
        side_lc = side.lower().strip()
        if side_lc not in ("left", "right"):
            return BiomechResponse(
                success=False,
                error=f"side must be 'left' or 'right', got '{side}'.",
            )

        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"file_too_large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "hip.mp4")
        log.info(
            "hip: file=%s size=%.2f MB movement=%s side=%s recording_ms=%s",
            video.filename, size_mb, movement, side_lc, recording_duration_ms,
        )

        processed_path, fixed_path_cleanup = _ensure_decodable_video(
            tmp_path, recording_duration_ms,
        )

        probe = cv2.VideoCapture(processed_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0 or probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read video metadata. Please retry, "
                    "or upload a different file."
                ),
            )
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"fps_too_low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required."
                ),
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds < 2.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"video_too_short ({duration_seconds:.1f}s). "
                    f"Minimum 2 seconds required."
                ),
            )
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"duration_too_long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )

        _ = patient_name

        pose_options = _build_gait_pose_options()
        result = analyze_hip_engine(
            video_path=processed_path,
            pose_options=pose_options,
            movement=movement,
            side=side_lc,
        )
        return BiomechResponse(success=True, data=result, error=None)

    except HTTPException:
        raise
    except ValueError as e:
        msg = str(e)
        log.warning("hip validation: %s", msg)
        if msg == "poor_visibility":
            raise HTTPException(status_code=400, detail="poor_visibility")
        if msg.startswith("Requested side"):
            raise HTTPException(status_code=400, detail=msg)
        # Rotation test couldn't lock a calibration baseline at the
        # start of the recording — patient must hold the supine
        # neutral pose (knee at 90°, lower leg pointing toward
        # camera) for ~5 frames before initiating rotation.
        if msg.startswith("Neutral pose"):
            raise HTTPException(status_code=400, detail=msg)
        return BiomechResponse(success=False, error=msg)
    except Exception as e:
        log.exception("hip analysis failed")
        return BiomechResponse(success=False, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


# ──────────────────────────────────────────────────────────────────
# Posture (front + side static-photo analysis)
# ──────────────────────────────────────────────────────────────────
# Posture is the only backend pipeline that takes STATIC PHOTOS
# instead of video — RunningMode.IMAGE rather than VIDEO. Two
# photos arrive in one multipart request (front + side). The
# engine handles EXIF rotation correction (mobile portraits) +
# runs BlazePose Full IMAGE-mode on each photo + returns combined
# metrics/findings + keypoints in the MoveNet-indexed layout so
# the existing frontend overlay and saved-report viewer keep
# rendering without any display changes.
#
# Replaces the browser MoveNet path in
# motionlens-web/lib/posture/analyzer.ts (which now just POSTs
# here and unwraps the response).

# Max per-photo size — posture photos are static stills, smaller
# limit than the 100 MB video cap.
_MAX_POSTURE_PHOTO_MB = 10


@app.post("/api/analyze-posture")
async def analyze_posture(
    front_image: UploadFile = File(...),
    side_image:  UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
):
    """Posture analysis endpoint — accepts two photos in one
    multipart request (front view + side view), runs BlazePose
    Full IMAGE-mode on each after EXIF-correcting orientation,
    returns combined metrics + findings + keypoints (MoveNet-
    indexed 17-element arrays so the existing saved-report
    viewer renders unchanged).

    Validation gates:
      • file_too_large       (>10 MB per photo)        → 413
      • invalid_image        (PIL can't decode)        → 400
      • poor_visibility      (engine raises)           → 400
        → emitted when the engine can't see the trunk anchors
          (both shoulders + both hips) in either photo.

    Response shape:
      {
        "success": true,
        "data": {
          "front": { view, imageWidth, imageHeight, keypoints,
                     front: {...}, findings: [...] },
          "side":  { view, imageWidth, imageHeight, keypoints,
                     side:  {...}, findings: [...] },
          "relative_units": true,
        },
        "error": null,
      }

    Photos are DELETED from disk after analysis (privacy decision
    from Phase B — only metrics + keypoints get persisted to
    MongoDB, never the source photos).
    """
    front_path: Optional[str] = None
    side_path: Optional[str] = None
    try:
        front_bytes = await front_image.read()
        side_bytes  = await side_image.read()
        if not front_bytes or not side_bytes:
            raise HTTPException(
                status_code=400,
                detail="Both front and side photos are required.",
            )
        for tag, payload in (("front", front_bytes), ("side", side_bytes)):
            size_mb = len(payload) / (1024 * 1024)
            if size_mb > _MAX_POSTURE_PHOTO_MB:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"file_too_large ({size_mb:.1f} MB {tag} photo). "
                        f"Maximum {_MAX_POSTURE_PHOTO_MB} MB per photo."
                    ),
                )

        # Save the two photos to disk for PIL/MediaPipe processing.
        # save_uploaded_video validates the suffix against the video-
        # extension allowlist and would reject .png/.jpg here, so we
        # save inline against an image-extension allowlist instead.
        # tempfile.mkstemp gives us a path the engine + the finally-
        # block cleanup both reach via cleanup_temp_file.
        _ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}

        def _save_image(payload: bytes, original_name: str, fallback: str) -> str:
            ext = os.path.splitext(original_name or fallback)[1].lower() or ".jpg"
            if ext not in _ALLOWED_IMAGE_EXTS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"invalid_image (unsupported extension: {ext}). "
                        f"Allowed: {sorted(_ALLOWED_IMAGE_EXTS)}"
                    ),
                )
            fd, path = tempfile.mkstemp(suffix=ext, prefix="motionlens_posture_")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
            except Exception:
                try:
                    os.unlink(path)
                except Exception:
                    pass
                raise
            return path

        front_path = _save_image(
            front_bytes, front_image.filename or "", "posture_front.jpg",
        )
        side_path = _save_image(
            side_bytes, side_image.filename or "", "posture_side.jpg",
        )
        log.info(
            "posture: front=%s (%.2f MB) side=%s (%.2f MB)",
            front_image.filename,
            len(front_bytes) / 1024 / 1024,
            side_image.filename,
            len(side_bytes) / 1024 / 1024,
        )

        _ = patient_name  # accepted for parity; not used here

        result = analyze_posture_combined_engine(
            front_image_path=front_path,
            side_image_path=side_path,
        )
        return {"success": True, "data": result, "error": None}

    except HTTPException:
        raise
    except ValueError as e:
        msg = str(e)
        log.warning("posture validation: %s", msg)
        if msg == "invalid_image":
            raise HTTPException(status_code=400, detail="invalid_image")
        if msg == "poor_visibility":
            raise HTTPException(status_code=400, detail="poor_visibility")
        return {"success": False, "data": None, "error": msg}
    except Exception as e:
        log.exception("posture analysis failed")
        return {"success": False, "data": None, "error": f"Analysis failed: {e}"}
    finally:
        # Phase B privacy decision: photos are NEVER persisted on
        # the backend. Delete temp files whether the analysis
        # succeeded or failed.
        cleanup_temp_file(front_path)
        cleanup_temp_file(side_path)


@app.post("/api/analyze-neck", response_model=BiomechResponse)
async def analyze_neck(
    video: UploadFile = File(...),
    movement_type: str = Form(...),
    # Neck flex/ext doesn't carry a side; the field is accepted
    # for sibling-endpoint parity and ignored by the engine.
    side: Optional[str] = Form(None),
    patient_name: Optional[str] = Form(None),
    recording_duration_ms: Optional[int] = Form(None),
) -> BiomechResponse:
    """Neck merged flexion+extension upload analysis on backend
    MediaPipe BlazePose Full.

    Only `flexion_extension` is routed to backend; the other neck
    movements (lateral_flexion, rotation) still run through the
    browser MoveNet path.

    Math note — the backend uses the ear→nose tilt formula
    (`tilt = atan2(faceVecY, |faceVecX|) − 10° baseline`),
    mirroring motionlens-web/lib/biomech/neck.ts:computeNeckAngle
    so live + upload report the same metric. This requires a
    LATERAL camera profile (the formula collapses in pure
    frontal view); pre-flight rejection surfaces a clear,
    actionable message when the upload is near-frontal.

    Validation gates (raise HTTPException → frontend maps to user-
    facing error text):
      • file_too_large            (>100 MB)         → 413
      • fps_too_low               (<24 FPS)         → 400
      • duration_too_long         (>60 s)           → 400
      • video_too_short           (<2 s)            → 400
      • poor_visibility           (engine raises)   → 400
      • Camera angle …            (engine raises)   → 400
    """
    from engines.orthopedic.tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        movement = movement_type.lower().strip()
        _ALLOWED_NECK = ("flexion_extension", "lateral_flexion", "rotation")
        if movement not in _ALLOWED_NECK:
            return BiomechResponse(
                success=False,
                error=(
                    f"Unknown neck movement '{movement_type}'. "
                    f"Allowed: {sorted(_ALLOWED_NECK)}."
                ),
            )
        _ = side  # accepted for parity, ignored by the engine

        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_GAIT_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"file_too_large ({size_mb:.1f} MB). "
                    f"Maximum allowed: {MAX_GAIT_FILE_SIZE_MB} MB."
                ),
            )

        tmp_path = save_uploaded_video(contents, video.filename or "neck.mp4")
        log.info(
            "neck: file=%s size=%.2f MB movement=%s recording_ms=%s",
            video.filename, size_mb, movement, recording_duration_ms,
        )

        # Repair MediaRecorder WebMs with broken duration headers
        # before the cv2 FPS probe (same as ankle / shoulder / knee).
        processed_path, fixed_path_cleanup = _ensure_decodable_video(
            tmp_path, recording_duration_ms,
        )

        probe = cv2.VideoCapture(processed_path)
        try:
            probe_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            probe_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            probe.release()

        if probe_fps <= 0 or probe_total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read video metadata. Please retry, "
                    "or upload a different file."
                ),
            )
        if probe_fps < MIN_REQUIRED_FPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"fps_too_low ({probe_fps:.1f} FPS). "
                    f"Minimum {MIN_REQUIRED_FPS} FPS required."
                ),
            )

        duration_seconds = probe_total_frames / probe_fps
        if duration_seconds < 2.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"video_too_short ({duration_seconds:.1f}s). "
                    f"Minimum 2 seconds required."
                ),
            )
        if duration_seconds > MAX_GAIT_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"duration_too_long ({duration_seconds:.1f}s). "
                    f"Maximum {MAX_GAIT_DURATION_SEC} seconds allowed."
                ),
            )

        _ = patient_name

        pose_options = _build_gait_pose_options()
        result = analyze_neck_engine(
            video_path=processed_path,
            pose_options=pose_options,
            movement=movement,
        )
        return BiomechResponse(success=True, data=result, error=None)

    except HTTPException:
        raise
    except ValueError as e:
        msg = str(e)
        log.warning("neck validation: %s", msg)
        if msg == "poor_visibility":
            raise HTTPException(status_code=400, detail="poor_visibility")
        # Frontal/lateral view rejection — surfaced as HTTP 400 with
        # the user-facing message preserved verbatim.
        if msg.startswith("Camera angle"):
            raise HTTPException(status_code=400, detail=msg)
        # Rotation test couldn't lock a calibration baseline (patient
        # didn't hold "facing forward" for the calibration window).
        if msg.startswith("Neutral pose"):
            raise HTTPException(status_code=400, detail=msg)
        return BiomechResponse(success=False, error=msg)
    except Exception as e:
        log.exception("neck analysis failed")
        return BiomechResponse(success=False, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


@app.post("/api/live/biomech-frame", response_model=LiveBiomechFrameResponse)
def analyze_live_biomech_frame(
    frame: UploadFile = File(...),
    body_part: str = Form(...),
    movement_type: str = Form(...),
    side: str = Form("right"),
) -> LiveBiomechFrameResponse:
    """Single-frame biomech analysis for live mode.

    Decodes a JPEG, runs MediaPipe pose detection (cached singleton),
    computes the angle via the existing engine helpers, returns angle
    + landmarks for the frontend to draw a skeleton overlay.

    Sync handler — runs on FastAPI's threadpool so the lock-protected
    landmarker call doesn't block the asyncio loop.
    """
    try:
        bp = body_part.lower().strip()
        movement = movement_type.lower().strip()
        side_lc = side.lower().strip()

        if bp not in ("shoulder", "neck"):
            return LiveBiomechFrameResponse(
                success=False, error=f"Unknown body_part '{body_part}'.",
            )
        if bp == "shoulder" and movement not in SHOULDER_NORMAL_RANGES:
            return LiveBiomechFrameResponse(
                success=False, error=f"Unknown shoulder movement '{movement}'.",
            )
        if bp == "neck" and movement not in NECK_NORMAL_RANGES:
            return LiveBiomechFrameResponse(
                success=False, error=f"Unknown neck movement '{movement}'.",
            )
        if bp == "shoulder" and side_lc not in ("left", "right"):
            return LiveBiomechFrameResponse(
                success=False, error="side must be 'left' or 'right'.",
            )

        contents = frame.file.read()
        if not contents:
            return LiveBiomechFrameResponse(success=False, error="Empty frame.")

        arr = np.frombuffer(contents, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return LiveBiomechFrameResponse(success=False, error="Could not decode frame.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        with _live_landmarker() as landmarker:
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks:
                # Reset smoother so the next valid frame doesn't blend
                # with stale state from a different pose.
                landmarker.reset_smoother()
                return LiveBiomechFrameResponse(
                    success=True,
                    data=LiveBiomechFrameData(
                        status="no_landmarks",
                        landmarks=[],
                        current_angle=None,
                        current_magnitude=0.0,
                    ),
                )

            raw = result.pose_landmarks[0]

            # Smoothed coords go to the frontend overlay; raw coords go
            # to the engine math (peak / current measurements).
            smoothed_xy_vis = landmarker.smooth_for_overlay(raw)

        landmarks_dto = [
            LandmarkPoint(x=sx, y=sy, visibility=v)
            for (sx, sy, v) in smoothed_xy_vis
        ]

        wrapped = _wrap_landmarks(raw)
        if bp == "shoulder":
            angle = compute_shoulder_angle(wrapped, side_lc, movement)
        else:
            angle = compute_neck_angle(wrapped, movement)

        if angle is None:
            return LiveBiomechFrameResponse(
                success=True,
                data=LiveBiomechFrameData(
                    status="low_visibility",
                    landmarks=landmarks_dto,
                    current_angle=None,
                    current_magnitude=0.0,
                ),
            )

        return LiveBiomechFrameResponse(
            success=True,
            data=LiveBiomechFrameData(
                status="good",
                landmarks=landmarks_dto,
                current_angle=float(angle),
                current_magnitude=float(abs(angle)),
            ),
        )
    except Exception as e:
        log.exception("live biomech frame analysis failed")
        return LiveBiomechFrameResponse(success=False, error=f"Analysis failed: {e}")


@app.post("/api/analyze-neck", response_model=BiomechResponse)
async def analyze_neck(
    video: UploadFile = File(...),
    movement_type: str = Form(...),
    patient_name: Optional[str] = Form(None),
) -> BiomechResponse:
    """Neck ROM. movement_type ∈ flexion / extension / lateral_flexion /
    rotation. (Engine has no per-side parameter — both lateral_flexion
    and rotation are computed from both ears + shoulder midline.)"""
    tmp_path: str | None = None
    try:
        movement = movement_type.lower().strip()
        if movement not in NECK_NORMAL_RANGES:
            return BiomechResponse(
                success=False,
                error=(
                    f"Unknown neck movement '{movement_type}'. "
                    f"Allowed: {sorted(NECK_NORMAL_RANGES.keys())}"
                ),
            )

        contents = await video.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video upload.")
        tmp_path = save_uploaded_video(contents, video.filename or "video.mp4")
        log.info(
            "neck: file=%s size=%.2f MB movement=%s",
            video.filename, len(contents) / 1024 / 1024, movement,
        )

        raw_result = _run_biomech_upload_analysis(
            tmp_path, "neck", movement, "Right",  # side is unused by neck_engine
        )
        if raw_result.get("valid_frames", 0) == 0:
            return BiomechResponse(
                success=False,
                error=(
                    "No frames had high-confidence pose landmarks. "
                    "Re-record with better lighting or position."
                ),
            )

        normal = NECK_NORMAL_RANGES[movement]
        data = format_biomech_response(
            body_part="neck",
            movement=movement,
            side=None,
            raw_result=raw_result,
            normal_range=normal["range"],
            target=normal["target"],
        )
        return BiomechResponse(success=True, data=data, error=None)

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("neck validation: %s", e)
        return BiomechResponse(success=False, error=str(e))
    except Exception as e:
        log.exception("neck analysis failed")
        return BiomechResponse(success=False, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
