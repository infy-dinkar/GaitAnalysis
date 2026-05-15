"""Root entry-point shim — keeps `uvicorn api:app` working post-reorg.

The FastAPI application now lives in ``backend/api/main.py``. Hugging Face
Spaces' Dockerfile runs ``uvicorn api:app`` and the HEALTHCHECK probes
``/api/health``; this thin re-export preserves that contract so no deploy
configuration has to change.
"""
from backend.api.main import app  # noqa: F401
