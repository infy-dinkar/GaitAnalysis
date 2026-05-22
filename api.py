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
import threading
from contextlib import contextmanager
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Engine imports (no modifications)
from gait_engine import (
    extract_poses,
    build_time_series,
    compute_all_features,
    interpret,
)
from shoulder_engine import (
    SHOULDER_NORMAL_RANGES,
    analyze_shoulder as analyze_shoulder_engine,
    compute_shoulder_angle,
)
from neck_engine import NECK_NORMAL_RANGES, compute_neck_angle
from biomech_flow import (
    _run_biomech_upload_analysis,
    _ensure_pose_model_file,
    _LandmarkAdapter,
    _wrap_landmarks,
)

from api_helpers import (
    save_uploaded_video,
    cleanup_temp_file,
    format_gait_response,
    format_biomech_response,
)
from api_models import (
    BiomechResponse,
    GaitResponse,
    HealthResponse,
    LandmarkPoint,
    LiveBiomechFrameData,
    LiveBiomechFrameResponse,
)

# ─── Timed Up and Go (TUG) — reuses the gait MediaPipe pipeline ───
from tug_engine import analyze_tug
from tug_models import TUGResponse

# ─── Ankle (dorsi/plantar) — reuses gait MediaPipe pipeline ──────
from ankle_engine import analyze_ankle as analyze_ankle_engine

# ─── Knee (merged flex+ext) — reuses gait MediaPipe pipeline ─────
from knee_engine import analyze_knee as analyze_knee_engine

# ─── Neck (merged flex+ext) — reuses gait MediaPipe pipeline ─────
from neck_engine import analyze_neck as analyze_neck_engine

# ─── Hip (flexion) — reuses gait MediaPipe pipeline ──────────────
from hip_engine import analyze_hip as analyze_hip_engine

# ─── SPPB Component 1 (Balance) — reuses gait MediaPipe pipeline ─
from sppb_balance_engine import analyze_sppb_balance

# ─── Auth + database (Phase 1) ─────────────────────────────────────
import db as db_module
from auth_routes import router as auth_router

# ─── Patient + report endpoints (Phase 2) ──────────────────────────
from patient_routes import router as patient_router
from report_routes import (
    patient_reports_router,
    reports_router,
)

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
    "http://localhost:3000,http://127.0.0.1:3000,*",
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
) -> GaitResponse:
    """Run the full gait pipeline on an uploaded clip."""
    tmp_path: str | None = None
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
            "gait: file=%s size=%.2f MB height=%scm",
            video.filename, size_mb, height_cm,
        )

        # ── 2) FPS + DURATION GATE — peek at the container before the
        # full pose-extraction pass. A bad clip should fail in <100 ms,
        # not after 30 sec of MediaPipe work.
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
        raw, fps, total_frames = extract_poses(tmp_path, pose_options)
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
    from tug_engine import _ensure_decodable_video

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
    from tug_engine import _ensure_decodable_video

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
    from tug_engine import _ensure_decodable_video

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
    from tug_engine import _ensure_decodable_video

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
    from tug_engine import _ensure_decodable_video

    tmp_path: str | None = None
    fixed_path_cleanup: str | None = None
    try:
        movement = movement_type.lower().strip()
        _ALLOWED_HIP = ("flexion", "extension")
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
        return BiomechResponse(success=False, error=msg)
    except Exception as e:
        log.exception("hip analysis failed")
        return BiomechResponse(success=False, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)
        if fixed_path_cleanup and fixed_path_cleanup != tmp_path:
            cleanup_temp_file(fixed_path_cleanup)


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
    from tug_engine import _ensure_decodable_video

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
