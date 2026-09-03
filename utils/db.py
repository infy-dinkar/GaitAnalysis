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
    # is unreachable; 20 sec covers the first cold-connection round-trip
    # (DNS SRV resolution + TLS handshake + replica-set discovery) over
    # variable home/dev networks. Earlier value (5 sec) was tuned for
    # the HF Space deploy environment but failed on slower local
    # networks even when the credentials and IP allowlist were correct.
    # Overridable via MONGODB_TIMEOUT_MS env var if needed.
    timeout_ms = int(os.environ.get("MONGODB_TIMEOUT_MS", "20000"))
    # tz_aware=True returns tz-aware UTC datetimes from BSON reads — without
    # this, Pydantic emits naive ISO strings that browsers parse as LOCAL
    # time, shifting every saved-report timestamp by the user's tz offset.
    _client = AsyncIOMotorClient(
        uri,
        serverSelectionTimeoutMS=timeout_ms,
        appname="motionlens-api",
        tz_aware=True,
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

    # ── prescriptions ─────────────────────────────────────────
    # One prescription per (patient, doctor) pair — enforced with a
    # unique compound index. Also the primary access pattern for the
    # GET/PUT/DELETE endpoints.
    await db.prescriptions.create_index(
        [("patient_id", 1), ("doctor_id", 1)],
        unique=True,
    )

    log.info("MongoDB indexes ensured")


# ══════════════════════════════════════════════════════════════════════
# PostgreSQL — DORMANT path (Phase 1 of the Mongo→Postgres switch)
# ══════════════════════════════════════════════════════════════════════
# ADDITIVE ONLY. Nothing below touches the Mongo client / _db handle or
# any existing query. It is completely inert while DB_BACKEND=mongo (the
# default): no psycopg import happens, no PG connection opens, and the
# app behaves EXACTLY as before. Query conversion + repository routing
# are later phases — this block only provides a config flag, a dormant
# connection, and a startup health check.

# Module-level async connection POOL (opened lazily on startup when
# DB_BACKEND=postgres). Replaces the earlier single persistent connection,
# which went stale on idle / network blips and then surfaced as
# "psycopg.OperationalError: the connection is closed". The pool
# health-checks each connection on checkout and transparently replaces
# dead ones, so a stale socket never reaches the repositories. Kept
# distinct from _client/_db so the Mongo path is never affected; inert
# while DB_BACKEND=mongo (no psycopg / psycopg_pool import, no socket).
_pg_pool = None  # type: ignore[var-annotated]  # psycopg_pool.AsyncConnectionPool | None


def get_db_backend() -> str:
    """Active DB backend, from the DB_BACKEND env var.

    'mongo'  (default) — today's behaviour, unchanged.
    'postgres'         — enables the dormant PG health check below.

    Any unset / blank / unrecognised value falls back to 'mongo' so the
    app can never accidentally leave the known-good default.
    """
    val = os.environ.get("DB_BACKEND", "mongo").strip().lower()
    return val or "mongo"


def _pg_conn_kwargs() -> dict:
    """Read PG connection details from env (no hardcoded values)."""
    return {
        "host": os.environ.get("PG_HOST", "").strip(),
        "port": int(os.environ.get("PG_PORT", "5432").strip() or "5432"),
        "dbname": os.environ.get("PG_DB", "").strip(),
        "user": os.environ.get("PG_USER", "").strip(),
        "password": os.environ.get("PG_PASSWORD", ""),
    }


async def postgres_healthcheck() -> None:
    """Verify the PostgreSQL connection when DB_BACKEND=postgres.

    When the backend is 'mongo' (default) this is a pure no-op — it
    returns immediately, imports nothing new, and opens no socket, so
    the default configuration is byte-for-byte the same as before.

    When the backend is 'postgres' it opens a real connection using the
    PG_* env vars, runs `SELECT 1`, and stores the connection handle.
    The psycopg import is LAZY (inside this function) so the default
    path never requires the driver to be installed.
    """
    if get_db_backend() != "postgres":
        return  # mongo default → nothing new happens

    global _pg_pool

    try:
        import psycopg  # noqa: F401 — presence check; the pool uses the driver
        from psycopg_pool import AsyncConnectionPool
    except ImportError as e:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "DB_BACKEND=postgres but 'psycopg' / 'psycopg_pool' are not "
            "installed. Install them (see requirements.txt: psycopg[binary], "
            "psycopg_pool) and retry."
        ) from e

    kwargs = _pg_conn_kwargs()
    missing = [k for k in ("host", "dbname", "user") if not kwargs.get(k)]
    if missing:
        raise RuntimeError(
            "DB_BACKEND=postgres but required PG env vars are missing: "
            f"{missing}. Set PG_HOST, PG_DB, PG_USER (and PG_PASSWORD)."
        )

    if _pg_pool is None:
        open_timeout = float(os.environ.get("PG_POOL_OPEN_TIMEOUT", "20"))
        _pg_pool = AsyncConnectionPool(
            # autocommit=True on every pooled connection → each statement
            # commits on its own (mirrors Mongo's no-multi-doc-txn
            # behaviour, keeps the CRUD layer simple). The per-connection
            # UTC session TZ is set in _pg_configure so timestamptz reads
            # come back tz-aware UTC, matching Mongo's tz_aware=True — the
            # API emits identical ISO timestamps on both backends.
            kwargs={**kwargs, "autocommit": True},
            min_size=int(os.environ.get("PG_POOL_MIN", "1")),
            max_size=int(os.environ.get("PG_POOL_MAX", "5")),
            open=False,
            configure=_pg_configure,
            # Liveness check on EVERY checkout: a stale / dropped connection
            # is detected and discarded here and a fresh one handed out —
            # this is the fix for the "connection is closed" errors.
            check=AsyncConnectionPool.check_connection,
            name="motionlens-pg",
        )
        await _pg_pool.open(wait=True, timeout=open_timeout)

    # Verify a live checkout works (SELECT 1 through the pool).
    async with _pg_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            row = await cur.fetchone()
    if not row or row[0] != 1:
        raise RuntimeError("Postgres health check failed: SELECT 1 did not return 1")

    log.info(
        "PostgreSQL pool opened + health check passed (SELECT 1 OK, db=%s, "
        "min=%s max=%s)",
        kwargs.get("dbname"),
        _pg_pool.min_size,
        _pg_pool.max_size,
    )

    await _pg_ensure_schema()


async def _pg_ensure_schema() -> None:
    """Idempotent column migrations for the Postgres backend.

    The base schema was created out-of-band (there is no CREATE TABLE in
    this repo), so additive columns are applied here instead — at
    startup, right after the health check, before any request is served.

    Every statement uses IF NOT EXISTS and a NOT NULL DEFAULT, so:
      • re-running is a no-op,
      • existing rows backfill automatically ('clinician' / TRUE),
      • no downtime window where the column is missing.

    Mongo needs no equivalent — documents are schemaless and
    doctors_insert sets both fields explicitly, while readers use
    .get(field, default) for rows written before this landed.
    """
    if _pg_pool is None:
        return
    statements = (
        "ALTER TABLE doctors ADD COLUMN IF NOT EXISTS "
        "role TEXT NOT NULL DEFAULT 'clinician'",
        "ALTER TABLE doctors ADD COLUMN IF NOT EXISTS "
        "is_active BOOLEAN NOT NULL DEFAULT TRUE",
    )
    try:
        async with _pg_pool.connection() as conn:
            async with conn.cursor() as cur:
                for sql in statements:
                    await cur.execute(sql)
        log.info("PostgreSQL schema ensured (doctors.role, doctors.is_active)")
    except Exception as e:
        # ⚠️ ALTER TABLE requires table ownership. The app connects as
        # PG_USER (motionlens_user), but `doctors` is owned by
        # `postgres`, so this raises InsufficientPrivilege in the
        # current prod setup.
        #
        # Deliberately non-fatal: an auth outage is far worse than a
        # missing feature. The repository layer probes for the columns
        # below and falls back to the legacy column list when they are
        # absent, so login keeps working exactly as before — the admin
        # endpoints are simply inert until a DBA runs the SQL.
        log.warning(
            "Could not apply doctors role/is_active migration (%s). "
            "Admin user-management stays DISABLED until a superuser runs:\n"
            "  ALTER TABLE doctors ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'clinician';\n"
            "  ALTER TABLE doctors ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;\n"
            "(or: ALTER TABLE doctors OWNER TO %s;)",
            e,
            _pg_conn_kwargs().get("user", "motionlens_user"),
        )

    await _pg_detect_doctor_columns()


async def _pg_detect_doctor_columns() -> None:
    """Tell the repository layer whether doctors.role / is_active exist.

    Runs once at startup, after the migration attempt. Without this the
    SELECT column list would name columns that may not be there and
    EVERY doctor lookup — including login — would fail with
    UndefinedColumn. Probing keeps the app fully functional on the old
    schema and lights the new fields up automatically once the ALTER
    has been applied, with no code change or redeploy.
    """
    if _pg_pool is None:
        return
    from utils import repositories as repo
    try:
        async with _pg_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'doctors' "
                    "AND column_name IN ('role', 'is_active')"
                )
                found = {r[0] for r in await cur.fetchall()}
    except Exception as e:
        log.warning("Could not probe doctors columns (%s); assuming legacy schema", e)
        found = set()

    present = {"role", "is_active"} <= found
    repo.set_doctor_role_columns_present(present)
    if present:
        log.info("doctors.role / is_active present — admin features ENABLED")
    else:
        log.warning(
            "doctors.role / is_active MISSING — running in legacy mode: "
            "everyone is treated as an active clinician and /api/admin/* "
            "will reject all callers. Apply the ALTER above to enable."
        )


async def postgres_disconnect() -> None:
    """Close the PG pool if one was opened. No-op otherwise."""
    global _pg_pool
    if _pg_pool is not None:
        try:
            await _pg_pool.close()
        finally:
            _pg_pool = None
        log.info("PostgreSQL pool closed")


async def _pg_configure(conn) -> None:
    """Per-connection setup the pool runs for every new connection: UTC
    session TZ so timestamptz reads come back tz-aware UTC (matching
    Mongo's tz_aware=True). autocommit is applied via the pool's
    connection kwargs (see postgres_healthcheck)."""
    await conn.execute("SET TIME ZONE 'UTC'")


def get_pg_pool():
    """Return the live PG pool (opened by postgres_healthcheck at startup).
    Raises if DB_BACKEND=postgres wasn't active / the pool didn't open.
    Never called on the Mongo path."""
    if _pg_pool is None:
        raise RuntimeError(
            "PostgreSQL pool not available. Is DB_BACKEND=postgres and did "
            "the startup health check succeed?"
        )
    return _pg_pool


def pg_connection():
    """Async context manager that checks out a pooled connection for a
    single operation and returns it to the pool on exit:

        async with pg_connection() as conn:
            async with conn.cursor(...) as cur:
                ...

    The pool health-checks the connection on checkout, so a stale /
    dropped connection is transparently replaced instead of surfacing as
    "connection is closed". Never used on the Mongo path."""
    return get_pg_pool().connection()
