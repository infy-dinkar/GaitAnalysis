"""MongoDB connection module — async client (motor) used by FastAPI.

Loads MONGODB_URI and MONGODB_DB_NAME from the environment (.env in
local dev, HF Spaces secrets in production). Exposes a single shared
client + database handle that the rest of the API imports.

Collection naming convention (lower_snake_case, plural):
    doctors   — registered users (physiotherapists / clinicians)
    patients  — patient records owned by a doctor
    reports   — saved analysis reports linked to a patient + doctor
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# Load .env if present (silently no-op in production where secrets come
# from the platform). Safe to call multiple times.
load_dotenv()

log = logging.getLogger("motionlens.db")

# ─── Module-level singletons ───────────────────────────────────────
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def _require_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(
            f"Environment variable {key!r} is not set. "
            f"Add it to .env (local) or HF Space secrets (production)."
        )
    return val


async def connect() -> None:
    """Initialise the global Mongo client + database handle.

    Called from the FastAPI startup event. Idempotent — calling twice
    is a no-op (just verifies the existing connection is alive).
    """
    global _client, _db

    if _client is not None and _db is not None:
        return  # already connected

    uri = _require_env("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB_NAME", "motionlens").strip() or "motionlens"

    # serverSelectionTimeoutMS keeps app from hanging forever if Atlas
    # is unreachable; 5 sec is a reasonable failure-fast threshold.
    _client = AsyncIOMotorClient(
        uri,
        serverSelectionTimeoutMS=5000,
        appname="motionlens-api",
    )
    _db = _client[db_name]

    # Verify the connection is actually alive (raises on failure).
    try:
        await _client.admin.command("ping")
    except Exception as e:
        log.error("MongoDB connection failed: %s", e)
        # Reset so caller can retry / surface a clear error.
        _client = None
        _db = None
        raise

    log.info("MongoDB connected (db=%s)", db_name)
    await _ensure_indexes(_db)


async def disconnect() -> None:
    """Close the Mongo client (FastAPI shutdown event)."""
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None
    log.info("MongoDB disconnected")


def get_db() -> AsyncIOMotorDatabase:
    """Get the active database handle. Raises if connect() not called."""
    if _db is None:
        raise RuntimeError(
            "Database not initialised. Did you forget to await db.connect() "
            "in the FastAPI startup event?"
        )
    return _db


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes on first startup. Idempotent."""
    # ── doctors ────────────────────────────────────────────────
    # Unique email — cannot have two doctors with the same email.
    await db.doctors.create_index("email", unique=True)

    # ── patients ───────────────────────────────────────────────
    # Index for "list patients owned by doctor X" — primary access pattern
    await db.patients.create_index("doctor_id")
    # Compound index for sorting recent patients per doctor
    await db.patients.create_index([("doctor_id", 1), ("created_at", -1)])

    # ── reports ────────────────────────────────────────────────
    # Index for "list reports for patient X"
    await db.reports.create_index("patient_id")
    # Index for "list all reports owned by doctor X"
    await db.reports.create_index("doctor_id")
    # Compound for sorted recent reports per patient
    await db.reports.create_index([("patient_id", 1), ("created_at", -1)])

    log.info("MongoDB indexes ensured")
