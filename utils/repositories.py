"""Repository routing layer — Phase 2 of the Mongo→Postgres switch.

ADDITIVE. Every function here dispatches on DB_BACKEND (utils.db.
get_db_backend()):

    • "mongo"  (default) — runs the EXACT same motor call the routes ran
      before this layer existed. Wrapped, not rewritten: the Mongo
      branch is behaviour-identical to the pre-Phase-2 code.
    • "postgres"         — runs a parameterised SQL equivalent on a
      connection checked out from the psycopg v3 async pool
      (utils.db.pg_connection()), health-checked on checkout.

Design guarantees so API responses stay byte-identical across backends:
    • ids: the Mongo branch keeps returning ObjectId (so str(_id) in the
      route mappers is unchanged); the Postgres branch stores/returns the
      SAME 24-char ObjectId-hex as TEXT, and str(hex) is a no-op. New rows
      created on Postgres get a fresh ObjectId() hex so the format never
      diverges from Mongo's.
    • id params: the Postgres branch coerces every id argument with str(),
      so callers can keep passing ObjectId tokens (from _parse_object_id /
      current_doctor["_id"]) unchanged. str(ObjectId(hex)) == the TEXT id.
    • JSONB: metrics/figures/observations/keypoints (reports) and
      slugs/notes (prescriptions) are read back by psycopg as native
      Python dict/list (dict_row + default jsonb loader) — the same shape
      Mongo returned — and written with Jsonb(...) wrappers.
    • dates: the PG connection runs at UTC with timestamptz columns, so
      created_at/updated_at come back as tz-aware UTC datetimes, matching
      Mongo's tz_aware=True.

NOT in this phase (Phase 3): auth. get_current_doctor + signup/login stay
on their existing Mongo calls, so protected routes still need Mongo up for
auth even when DB_BACKEND=postgres. The doctors_* repo functions below are
implemented (incl. password_hash on reads) so Phase 3 can flip auth over
without more repo work — they are not yet wired into the auth flow.

NOT in this phase (analytics): the only aggregation in the codebase is the
report-count-per-patient used by the patients list. It is a plain COUNT/
GROUP BY and is implemented here as CRUD support so the list route returns
identical output on Postgres. There is no models/analytics.py; any richer
analytics added later stays Mongo-only until explicitly ported.

psycopg is imported LAZILY inside the Postgres helpers so importing this
module (which happens at app startup via the routers) never requires the
driver on the default Mongo path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from utils import db as dbm
from utils.db import get_db_backend


# ── Schema constants ────────────────────────────────────────────────
# The migrated PostgreSQL schema names the primary key column `id` TEXT
# (holding the ObjectId hex) — NOT `_id`. The rest keep the Mongo field
# names. To keep every existing mapper working (they all read doc["_id"]
# via str(_id)), the Postgres SELECTs alias the real column `id AS _id`
# so the returned row dict carries an `_id` key with the real id value.
# WHERE clauses + INSERT column lists use the real column name `id`.
#   _ID      → real PK column (WHERE / INSERT)
#   _ID_SEL  → SELECT-list form aliasing id → _id for the mappers
# This is the ONLY place table/column names live.
_ID = "id"
_ID_SEL = "id AS _id"

_DOCTORS_BASE_COLS = [
    _ID_SEL, "email", "password_hash", "name", "specialization",
    "license_number", "created_at", "updated_at",
]
_DOCTORS_ROLE_COLS = ["role", "is_active"]

DOCTORS_COLS = _DOCTORS_BASE_COLS + _DOCTORS_ROLE_COLS

# Whether doctors.role / doctors.is_active actually exist in Postgres.
# Set once at startup by db._pg_detect_doctor_columns(). Defaults to
# True so the Mongo backend (schemaless — the keys are simply absent
# and readers use .get() defaults) and any pre-probe call behave
# normally; only a confirmed-legacy Postgres schema flips it off.
#
# ⚠️ When False, every SELECT drops those two columns. Without this the
# app would 500 on login against a database whose ALTER TABLE has not
# been applied — the migration needs table ownership the app role does
# not have. Callers all read via .get(field, default), so a missing
# column degrades to "active clinician" rather than an error.
_DOCTOR_ROLE_COLUMNS_PRESENT = True


def set_doctor_role_columns_present(present: bool) -> None:
    global _DOCTOR_ROLE_COLUMNS_PRESENT
    _DOCTOR_ROLE_COLUMNS_PRESENT = present


def doctor_role_columns_present() -> bool:
    return _DOCTOR_ROLE_COLUMNS_PRESENT


def _doctors_cols(include_password: bool) -> list[str]:
    """The SELECT list for a doctor row, adapted to the live schema."""
    cols = list(_DOCTORS_BASE_COLS)
    if _DOCTOR_ROLE_COLUMNS_PRESENT:
        cols += _DOCTORS_ROLE_COLS
    if not include_password:
        cols = [c for c in cols if c != "password_hash"]
    return cols


DOCTORS_COLS_NO_PW = [c for c in DOCTORS_COLS if c != "password_hash"]

PATIENTS_COLS = [
    _ID_SEL, "doctor_id", "name", "age", "gender", "height_cm", "weight_kg",
    "contact", "medical_notes", "created_at", "updated_at",
]
# Columns a PATCH is allowed to $set (+ updated_at). Whitelist so the
# dynamic UPDATE never interpolates an untrusted column name.
PATIENTS_UPDATABLE = {
    "name", "age", "gender", "height_cm", "weight_kg", "contact",
    "medical_notes", "updated_at",
}

REPORTS_FULL_COLS = [
    _ID_SEL, "patient_id", "doctor_id", "module", "body_part", "movement",
    "side", "metrics", "figures", "observations", "video_filename",
    "video_size_bytes", "keypoints", "created_at",
]
REPORTS_SUMMARY_COLS = [
    _ID_SEL, "patient_id", "doctor_id", "module", "body_part", "movement",
    "side", "created_at",
]
_REPORT_JSONB = ("metrics", "figures", "observations", "keypoints")

PRESCRIPTIONS_COLS = [
    _ID_SEL, "patient_id", "doctor_id", "slugs", "notes",
    "created_at", "updated_at",
]


def _is_pg() -> bool:
    return get_db_backend() == "postgres"


# ── Low-level Postgres helpers (lazy psycopg import) ────────────────
def _jsonb(value: Any):
    """Wrap a dict/list for a JSONB column. None → SQL NULL (not JSON
    'null'), matching how Mongo stores an absent optional."""
    if value is None:
        return None
    from psycopg.types.json import Jsonb
    return Jsonb(value)


async def _pg_fetchone(sql: str, params: list) -> Optional[dict]:
    from psycopg.rows import dict_row
    # One pooled connection per operation — health-checked on checkout, so
    # a stale/dropped connection is transparently replaced by the pool.
    async with dbm.pg_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()


async def _pg_fetchall(sql: str, params: list) -> list[dict]:
    from psycopg.rows import dict_row
    async with dbm.pg_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()


async def _pg_execute(sql: str, params: list) -> int:
    """Run a write; return affected row count."""
    async with dbm.pg_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return cur.rowcount


# ════════════════════════════════════════════════════════════════════
# doctors  (implemented for Phase 3 readiness; NOT wired into auth yet)
# ════════════════════════════════════════════════════════════════════
async def doctors_find_one_by_email(db, email: str) -> Optional[dict]:
    if not _is_pg():
        return await db.doctors.find_one({"email": email})
    sql = f'SELECT {", ".join(_doctors_cols(True))} FROM doctors WHERE email = %s'
    return await _pg_fetchone(sql, [email])


async def doctors_find_one_by_id(
    db, doctor_id, include_password: bool = False
) -> Optional[dict]:
    if not _is_pg():
        proj = None if include_password else {"password_hash": 0}
        return await db.doctors.find_one({"_id": doctor_id}, proj)
    cols = _doctors_cols(include_password)
    sql = f'SELECT {", ".join(cols)} FROM doctors WHERE {_ID} = %s'
    return await _pg_fetchone(sql, [str(doctor_id)])


async def doctors_insert(db, doc: dict):
    """Insert a doctor. Returns the new id (ObjectId on Mongo / hex str
    on Postgres) — str(...) in the mapper makes both identical.

    role / is_active are read with .get() defaults rather than taken as
    required keys, so a caller that omits them lands on the same
    'clinician' / TRUE the Postgres column defaults use. Callers must
    pass role as a server-side literal — never from a request body."""
    if not _is_pg():
        # Mongo is schemaless, so the defaults have to be materialised
        # onto the document here or the field is simply absent.
        doc = {
            **doc,
            "role": doc.get("role", "clinician"),
            "is_active": doc.get("is_active", True),
        }
        res = await db.doctors.insert_one(doc)
        return res.inserted_id
    nid = str(ObjectId())
    cols = [_ID, "email", "password_hash", "name", "specialization",
            "license_number", "created_at", "updated_at"]
    vals = [nid, doc["email"], doc["password_hash"], doc["name"],
            doc.get("specialization"), doc.get("license_number"),
            doc["created_at"], doc["updated_at"]]
    if _DOCTOR_ROLE_COLUMNS_PRESENT:
        cols += ["role", "is_active"]
        vals += [doc.get("role", "clinician"), doc.get("is_active", True)]
    await _pg_execute(
        f'INSERT INTO doctors ({", ".join(cols)}) '
        f'VALUES ({", ".join(["%s"] * len(cols))})',
        vals,
    )
    return nid


async def doctors_list_all(db) -> list[dict]:
    """Every doctor, newest first, WITHOUT password_hash.

    Admin user-management listing only. Deliberately not doctor-scoped —
    unlike patients/reports, which stay owner-scoped and are untouched
    by the admin role."""
    if not _is_pg():
        cursor = db.doctors.find({}, {"password_hash": 0}).sort("created_at", -1)
        return await cursor.to_list(length=1000)
    sql = (
        f'SELECT {", ".join(_doctors_cols(False))} FROM doctors '
        f'ORDER BY created_at DESC LIMIT %s'
    )
    return await _pg_fetchall(sql, [1000])


async def doctors_count_active_admins(db) -> int:
    """Number of doctors with role='admin' AND is_active=true.

    Used by the admin PATCH guard so the last active admin can never be
    demoted or deactivated — that would lock everyone out of user
    management with no recovery path short of DB surgery."""
    if not _is_pg():
        return await db.doctors.count_documents(
            {"role": "admin", "is_active": True}
        )
    if not _DOCTOR_ROLE_COLUMNS_PRESENT:
        # Legacy schema: the column doesn't exist, so no row can be an
        # admin. Returning 0 (rather than querying and blowing up) keeps
        # this callable from the admin guard path.
        return 0
    row = await _pg_fetchone(
        "SELECT count(*) AS n FROM doctors "
        "WHERE role = %s AND is_active = TRUE",
        ["admin"],
    )
    return int(row["n"]) if row else 0


async def doctors_set_role(db, doctor_id, role: str) -> Optional[dict]:
    """THE ONLY write path for doctors.role.

    Centralised so role can never be set from a request body: every
    caller passes a value already checked against auth_utils.ROLES."""
    return await _doctors_set_fields(db, doctor_id, {"role": role})


async def doctors_set_active(db, doctor_id, is_active: bool) -> Optional[dict]:
    """The only write path for doctors.is_active."""
    return await _doctors_set_fields(db, doctor_id, {"is_active": is_active})


async def doctors_set_password_hash(db, doctor_id, password_hash: str) -> Optional[dict]:
    """Admin-initiated password reset. Takes an ALREADY-HASHED value —
    never a plain password."""
    return await _doctors_set_fields(
        db, doctor_id, {"password_hash": password_hash}
    )


# Columns a doctor UPDATE may touch. Whitelisted so the dynamic SQL
# below can never interpolate an untrusted column name.
DOCTORS_UPDATABLE = {"role", "is_active", "password_hash", "updated_at"}


async def _doctors_set_fields(db, doctor_id, fields: dict) -> Optional[dict]:
    """Shared UPDATE helper. Returns the updated doctor (no password
    hash) or None when the id doesn't exist."""
    from datetime import datetime, timezone
    set_fields = {**fields, "updated_at": datetime.now(timezone.utc)}
    bad = [k for k in set_fields if k not in DOCTORS_UPDATABLE]
    if bad:
        raise ValueError(f"doctors update: non-updatable column(s): {bad}")

    # Writing role / is_active against a schema that lacks them would
    # fail with an opaque UndefinedColumn. Fail early with the fix.
    if (
        _is_pg()
        and not _DOCTOR_ROLE_COLUMNS_PRESENT
        and ({"role", "is_active"} & set(fields))
    ):
        raise RuntimeError(
            "doctors.role / doctors.is_active do not exist in this database. "
            "A superuser must run:  ALTER TABLE doctors ADD COLUMN IF NOT "
            "EXISTS role TEXT NOT NULL DEFAULT 'clinician';  ALTER TABLE "
            "doctors ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL "
            "DEFAULT TRUE;"
        )

    if not _is_pg():
        await db.doctors.update_one({"_id": doctor_id}, {"$set": set_fields})
        return await db.doctors.find_one(
            {"_id": doctor_id}, {"password_hash": 0}
        )
    keys = list(set_fields.keys())
    sql = (
        f'UPDATE doctors SET {", ".join(f"{k} = %s" for k in keys)} '
        f'WHERE {_ID} = %s '
        f'RETURNING {", ".join(_doctors_cols(False))}'
    )
    return await _pg_fetchone(sql, [set_fields[k] for k in keys] + [str(doctor_id)])


# ════════════════════════════════════════════════════════════════════
# patients
# ════════════════════════════════════════════════════════════════════
async def patients_insert(db, doc: dict):
    if not _is_pg():
        res = await db.patients.insert_one(doc)
        return res.inserted_id
    nid = str(ObjectId())
    cols = [_ID, "doctor_id", "name", "age", "gender", "height_cm",
            "weight_kg", "contact", "medical_notes", "created_at", "updated_at"]
    vals = [nid, str(doc["doctor_id"]), doc["name"], doc["age"], doc["gender"],
            doc["height_cm"], doc.get("weight_kg"), doc.get("contact"),
            doc.get("medical_notes"), doc["created_at"], doc["updated_at"]]
    await _pg_execute(
        f'INSERT INTO patients ({", ".join(cols)}) '
        f'VALUES ({", ".join(["%s"] * len(cols))})',
        vals,
    )
    return nid


async def patients_list_by_doctor(db, doctor_id) -> list[dict]:
    if not _is_pg():
        cursor = db.patients.find({"doctor_id": doctor_id}).sort("created_at", -1)
        return await cursor.to_list(length=1000)
    sql = (
        f'SELECT {", ".join(PATIENTS_COLS)} FROM patients '
        f'WHERE doctor_id = %s ORDER BY created_at DESC LIMIT %s'
    )
    return await _pg_fetchall(sql, [str(doctor_id), 1000])


async def patients_find_one(db, patient_id, doctor_id) -> Optional[dict]:
    if not _is_pg():
        return await db.patients.find_one(
            {"_id": patient_id, "doctor_id": doctor_id}
        )
    sql = (
        f'SELECT {", ".join(PATIENTS_COLS)} FROM patients '
        f'WHERE {_ID} = %s AND doctor_id = %s'
    )
    return await _pg_fetchone(sql, [str(patient_id), str(doctor_id)])


async def patients_exists(db, patient_id, doctor_id) -> bool:
    """Ownership check (report_routes). True iff the patient exists AND
    belongs to this doctor."""
    if not _is_pg():
        cnt = await db.patients.count_documents(
            {"_id": patient_id, "doctor_id": doctor_id}, limit=1
        )
        return bool(cnt)
    row = await _pg_fetchone(
        f'SELECT 1 AS ok FROM patients WHERE {_ID} = %s AND doctor_id = %s LIMIT 1',
        [str(patient_id), str(doctor_id)],
    )
    return row is not None


async def patients_find_one_min(db, patient_id, doctor_id) -> Optional[dict]:
    """Ownership check that returns a minimal doc (prescription routes)."""
    if not _is_pg():
        return await db.patients.find_one(
            {"_id": patient_id, "doctor_id": doctor_id}, {"_id": 1}
        )
    return await _pg_fetchone(
        f'SELECT {_ID_SEL} FROM patients WHERE {_ID} = %s AND doctor_id = %s',
        [str(patient_id), str(doctor_id)],
    )


async def patients_update(db, patient_id, doctor_id, set_fields: dict) -> Optional[dict]:
    if not _is_pg():
        return await db.patients.find_one_and_update(
            {"_id": patient_id, "doctor_id": doctor_id},
            {"$set": set_fields},
            return_document=True,
        )
    keys = list(set_fields.keys())
    bad = [k for k in keys if k not in PATIENTS_UPDATABLE]
    if bad:
        raise ValueError(f"patients_update: non-updatable column(s): {bad}")
    set_sql = ", ".join(f"{k} = %s" for k in keys)
    params = [set_fields[k] for k in keys] + [str(patient_id), str(doctor_id)]
    sql = (
        f'UPDATE patients SET {set_sql} '
        f'WHERE {_ID} = %s AND doctor_id = %s '
        f'RETURNING {", ".join(PATIENTS_COLS)}'
    )
    return await _pg_fetchone(sql, params)


async def patients_delete(db, patient_id) -> int:
    if not _is_pg():
        res = await db.patients.delete_one({"_id": patient_id})
        return res.deleted_count
    return await _pg_execute(
        f'DELETE FROM patients WHERE {_ID} = %s', [str(patient_id)]
    )


async def patients_delete_cascade(db, patient_id) -> int:
    """Delete a patient plus its child reports; return the reports count
    (for the API's deleted_reports_count).

    Mongo — no FK cascade exists, so the manual child delete runs HERE:
    delete_many(reports) [count from this] then delete_one(patient). This
    is verbatim the pre-Phase-4 two-step flow (prescriptions are left as
    today's Mongo code left them).

    Postgres — the schema's FK ON DELETE CASCADE removes the child reports
    (and prescriptions) when the patient row is deleted, so we DELETE ONLY
    the patient and let the cascade do the rest. The manual reports
    delete_many does NOT run on this branch (no double-cascade). The count
    for the response is taken with a COUNT(*) BEFORE the delete. Requires
    reports.patient_id → patients(id) ON DELETE CASCADE (Phase-5 schema
    check verifies this).
    """
    if not _is_pg():
        res = await db.reports.delete_many({"patient_id": patient_id})
        await db.patients.delete_one({"_id": patient_id})
        return res.deleted_count
    # postgres: count first (response parity), then cascade via patient row
    row = await _pg_fetchone(
        'SELECT COUNT(*) AS n FROM reports WHERE patient_id = %s',
        [str(patient_id)],
    )
    n = int(row["n"]) if row else 0
    await _pg_execute(f'DELETE FROM patients WHERE {_ID} = %s', [str(patient_id)])
    return n


# ════════════════════════════════════════════════════════════════════
# reports
# ════════════════════════════════════════════════════════════════════
async def reports_insert(db, doc: dict):
    if not _is_pg():
        res = await db.reports.insert_one(doc)
        return res.inserted_id
    nid = str(ObjectId())
    row = {
        _ID: nid,
        "patient_id": str(doc["patient_id"]),
        "doctor_id": str(doc["doctor_id"]),
        "module": doc["module"],
        "body_part": doc.get("body_part"),
        "movement": doc.get("movement"),
        "side": doc.get("side"),
        "metrics": _jsonb(doc.get("metrics") or {}),
        "figures": _jsonb(doc.get("figures") or []),
        "observations": _jsonb(doc.get("observations") or {}),
        "video_filename": doc.get("video_filename"),
        "video_size_bytes": doc.get("video_size_bytes"),
        "keypoints": _jsonb(doc.get("keypoints")),  # None → SQL NULL
        "created_at": doc["created_at"],
    }
    cols = list(row.keys())
    await _pg_execute(
        f'INSERT INTO reports ({", ".join(cols)}) '
        f'VALUES ({", ".join(["%s"] * len(cols))})',
        [row[c] for c in cols],
    )
    return nid


async def reports_list_by_patient(db, patient_id) -> list[dict]:
    """List reports (newest first), WITHOUT the heavy figures/metrics/
    observations blobs — mirrors the Mongo projection."""
    if not _is_pg():
        cursor = db.reports.find(
            {"patient_id": patient_id},
            {"figures": 0, "metrics": 0, "observations": 0},
        ).sort("created_at", -1)
        return await cursor.to_list(length=500)
    sql = (
        f'SELECT {", ".join(REPORTS_SUMMARY_COLS)} FROM reports '
        f'WHERE patient_id = %s ORDER BY created_at DESC LIMIT %s'
    )
    return await _pg_fetchall(sql, [str(patient_id), 500])


async def reports_find_one(db, report_id, doctor_id) -> Optional[dict]:
    if not _is_pg():
        return await db.reports.find_one(
            {"_id": report_id, "doctor_id": doctor_id}
        )
    sql = (
        f'SELECT {", ".join(REPORTS_FULL_COLS)} FROM reports '
        f'WHERE {_ID} = %s AND doctor_id = %s'
    )
    return await _pg_fetchone(sql, [str(report_id), str(doctor_id)])


async def reports_delete_one(db, report_id, doctor_id) -> int:
    if not _is_pg():
        res = await db.reports.delete_one(
            {"_id": report_id, "doctor_id": doctor_id}
        )
        return res.deleted_count
    return await _pg_execute(
        f'DELETE FROM reports WHERE {_ID} = %s AND doctor_id = %s',
        [str(report_id), str(doctor_id)],
    )


async def reports_delete_by_patient(db, patient_id) -> int:
    """Cascade helper: delete all of a patient's reports, return the
    count (the count parity keeps the delete-patient response identical).
    On Postgres an FK ON DELETE CASCADE is a backstop, but we still issue
    the explicit DELETE so deleted_reports_count is accurate."""
    if not _is_pg():
        res = await db.reports.delete_many({"patient_id": patient_id})
        return res.deleted_count
    return await _pg_execute(
        'DELETE FROM reports WHERE patient_id = %s', [str(patient_id)]
    )


async def reports_count_by_patient(db, patient_id) -> int:
    if not _is_pg():
        return await db.reports.count_documents({"patient_id": patient_id})
    row = await _pg_fetchone(
        'SELECT COUNT(*) AS n FROM reports WHERE patient_id = %s',
        [str(patient_id)],
    )
    return int(row["n"]) if row else 0


async def reports_count_per_patient(db, patient_ids: list) -> dict:
    """Report count keyed by patient id (patients list view). Simple
    COUNT/GROUP BY — CRUD support, not analytics. Keys match the id type
    of each backend (ObjectId on Mongo, hex str on Postgres) so the
    caller's counts.get(patient["_id"]) resolves on either."""
    if not patient_ids:
        return {}
    if not _is_pg():
        pipeline = [
            {"$match": {"patient_id": {"$in": patient_ids}}},
            {"$group": {"_id": "$patient_id", "count": {"$sum": 1}}},
        ]
        cursor = db.reports.aggregate(pipeline)
        return {c["_id"]: c["count"] async for c in cursor}
    rows = await _pg_fetchall(
        'SELECT patient_id, COUNT(*) AS n FROM reports '
        'WHERE patient_id = ANY(%s) GROUP BY patient_id',
        [[str(x) for x in patient_ids]],
    )
    return {r["patient_id"]: int(r["n"]) for r in rows}


# ════════════════════════════════════════════════════════════════════
# prescriptions
# ════════════════════════════════════════════════════════════════════
async def prescriptions_find(db, patient_id, doctor_id) -> Optional[dict]:
    if not _is_pg():
        return await db.prescriptions.find_one(
            {"patient_id": patient_id, "doctor_id": doctor_id}
        )
    sql = (
        f'SELECT {", ".join(PRESCRIPTIONS_COLS)} FROM prescriptions '
        f'WHERE patient_id = %s AND doctor_id = %s'
    )
    return await _pg_fetchone(sql, [str(patient_id), str(doctor_id)])


async def prescriptions_upsert(db, patient_id, doctor_id, slugs: list, notes: dict) -> None:
    """Idempotent upsert (one per patient+doctor). slugs/notes replaced
    wholesale; created_at set only on insert."""
    now = datetime.now(timezone.utc)
    if not _is_pg():
        update = {
            "$set": {"slugs": slugs, "notes": notes, "updated_at": now},
            "$setOnInsert": {
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "created_at": now,
            },
        }
        await db.prescriptions.update_one(
            {"patient_id": patient_id, "doctor_id": doctor_id},
            update,
            upsert=True,
        )
        return
    nid = str(ObjectId())
    sql = (
        'INSERT INTO prescriptions (id, patient_id, doctor_id, slugs, notes, '
        'created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s) '
        'ON CONFLICT (patient_id, doctor_id) DO UPDATE SET '
        'slugs = EXCLUDED.slugs, notes = EXCLUDED.notes, '
        'updated_at = EXCLUDED.updated_at'
    )
    await _pg_execute(
        sql,
        [nid, str(patient_id), str(doctor_id), _jsonb(slugs), _jsonb(notes), now, now],
    )


async def prescriptions_delete(db, patient_id, doctor_id) -> int:
    if not _is_pg():
        res = await db.prescriptions.delete_one(
            {"patient_id": patient_id, "doctor_id": doctor_id}
        )
        return res.deleted_count
    return await _pg_execute(
        'DELETE FROM prescriptions WHERE patient_id = %s AND doctor_id = %s',
        [str(patient_id), str(doctor_id)],
    )
