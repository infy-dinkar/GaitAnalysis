# MotionLens API — Hugging Face Spaces (Docker SDK) compatible.
FROM python:3.11-slim

# OpenCV / MediaPipe runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first to leverage Docker layer caching.
COPY requirements.txt requirements_api.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements_api.txt

# Pre-download the pose-landmarker .task at build time so cold starts
# don't pay for the network fetch.
RUN wget -q -O /app/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

# App code (everything we need — Streamlit files come along too,
# they're harmless and let users run `streamlit run app.py` inside
# the container if they want to).
COPY . /app

# HF Spaces injects $PORT at runtime, defaulting to 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
