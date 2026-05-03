# MotionLens API — Hugging Face Spaces (Docker SDK) compatible.
# Builds a self-contained FastAPI image with MediaPipe + the pose
# landmarker model pre-baked so cold starts don't pay the network fetch.

FROM python:3.11-slim

# ── System libs needed by OpenCV + MediaPipe ──────────────────────
# Use libgl1 (modern replacement for libgl1-mesa-glx, which was dropped
# in Debian Trixie). Other libs still exist with the same names.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps first to leverage Docker layer caching ────
COPY requirements.txt requirements_api.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements_api.txt

# ── Pre-download the pose-landmarker .task at build time ──────────
# Saves ~5 sec on every cold start.
RUN wget -q -O /app/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

# ── App code ──────────────────────────────────────────────────────
# .dockerignore excludes motionlens-web/, sample media, caches, etc.
COPY . /app

# HF Spaces injects $PORT at runtime, defaulting to 7860 elsewhere.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 7860

# Health probe — useful for orchestrators + HF dashboard
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:${PORT:-7860}/api/health || exit 1

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
