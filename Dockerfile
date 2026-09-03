# MotionLens API — HuggingFace Spaces deployment
# Builds a self-contained FastAPI image with MediaPipe + the pose
# landmarker model pre-baked so cold starts don't pay the network fetch.

FROM python:3.11-slim

# ── System libs needed by OpenCV + MediaPipe ──────────────────────
# - libgl1: desktop OpenGL (replaces libgl1-mesa-glx in Debian Trixie)
# - libgles2: OpenGL ES 2.0 — MediaPipe's Tasks API imports this at
#   load-time even when running CPU-only inference
# - libegl1: EGL — needed for GL context creation, paired with libgles2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgles2 \
        libegl1 \
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
# Single unified requirements.txt covering ML + API + DB + auth deps
# (Streamlit + matplotlib dropped — see Phase 2 cleanup notes).
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the pose-landmarker .task files at build time ────
# Saves ~5 sec on every cold start.
# BOTH variants are baked:
#   heavy — every upload / offline analysis path (the default in
#           engines/biomech_flow._ensure_pose_model_file)
#   full  — the real-time stream pool only (api._build_video_landmarker),
#           which is latency-bound and cannot afford heavy's per-frame cost
RUN wget -q -O /app/pose_landmarker_heavy.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task && \
    wget -q -O /app/pose_landmarker_full.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

# ── App code ──────────────────────────────────────────────────────
# .dockerignore excludes motionlens-web/, sample media, caches, etc.
COPY . /app

# HuggingFace Spaces injects $PORT at runtime.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 7860

# HuggingFace Spaces dashboard health probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:${PORT:-7860}/api/health || exit 1

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
