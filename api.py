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
from shoulder_engine import SHOULDER_NORMAL_RANGES, compute_shoulder_angle
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


@app.post("/api/analyze-shoulder", response_model=BiomechResponse)
async def analyze_shoulder(
    video: UploadFile = File(...),
    movement_type: str = Form(...),
    side: str = Form("right"),
    patient_name: Optional[str] = Form(None),
) -> BiomechResponse:
    """Shoulder ROM. movement_type ∈ flexion / extension / abduction /
    adduction / external_rotation / internal_rotation. side ∈ left / right."""
    tmp_path: str | None = None
    try:
        movement = movement_type.lower().strip()
        if movement not in SHOULDER_NORMAL_RANGES:
            return BiomechResponse(
                success=False,
                error=(
                    f"Unknown shoulder movement '{movement_type}'. "
                    f"Allowed: {sorted(SHOULDER_NORMAL_RANGES.keys())}"
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
        tmp_path = save_uploaded_video(contents, video.filename or "video.mp4")
        log.info(
            "shoulder: file=%s size=%.2f MB movement=%s side=%s",
            video.filename, len(contents) / 1024 / 1024, movement, side_lc,
        )

        # Reuse the existing engine helper (no logic duplication).
        raw_result = _run_biomech_upload_analysis(
            tmp_path, "shoulder", movement, side_lc.capitalize(),
        )
        if raw_result.get("valid_frames", 0) == 0:
            return BiomechResponse(
                success=False,
                error=(
                    "No frames had high-confidence pose landmarks. "
                    "Re-record with better lighting or position."
                ),
            )

        normal = SHOULDER_NORMAL_RANGES[movement]
        data = format_biomech_response(
            body_part="shoulder",
            movement=movement,
            side=side_lc,
            raw_result=raw_result,
            normal_range=normal["range"],
            target=normal["target"],
        )
        return BiomechResponse(success=True, data=data, error=None)

    except HTTPException:
        raise
    except ValueError as e:
        log.warning("shoulder validation: %s", e)
        return BiomechResponse(success=False, error=str(e))
    except Exception as e:
        log.exception("shoulder analysis failed")
        return BiomechResponse(success=False, error=f"Analysis failed: {e}")
    finally:
        cleanup_temp_file(tmp_path)


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
