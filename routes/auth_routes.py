"""Auth endpoints — sign up, sign in, current user.

Mounted at /api/auth/* by api.py.

Endpoints:
    POST /api/auth/signup   — register a new doctor
    POST /api/auth/login    — exchange email+password for a JWT
    GET  /api/auth/me       — return the current logged-in doctor (protected)

All responses use the same envelope shape the frontend already
expects: { success, data, error }.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status

from models.auth_models import (
    AuthTokenResponse,
    DoctorLoginRequest,
    DoctorPublic,
    DoctorSignupRequest,
)
from utils.auth_utils import (
    create_access_token,
    get_current_doctor,
    hash_password,
    verify_password,
)
from utils.db import get_db
from utils import repositories as repo

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Login rate limiting ───────────────────────────────────────────
# Sliding window over failed login attempts, held in process memory.
#
# ⚠️ PER-WORKER. gunicorn runs 2 workers (see Dockerfile.aws CMD), each
# with its own copy of these dicts, so the EFFECTIVE limit is roughly
# 2x the numbers below (an attacker's requests are distributed across
# workers). That is accepted for now: it still converts an unlimited
# online brute-force into ~20 tries per 15 min, which is the bulk of
# the benefit. A shared Redis counter would make it exact — worth doing
# if the worker count grows or the threat model tightens.
#
# Memory is bounded by pruning expired entries on every check, so the
# dicts only ever hold keys seen inside the current window.
_LOGIN_WINDOW_SECONDS = 15 * 60
_LOGIN_MAX_PER_EMAIL = 10
_LOGIN_MAX_PER_IP = 30

_failed_by_email: dict[str, list[float]] = {}
_failed_by_ip: dict[str, list[float]] = {}


def _prune(bucket: dict[str, list[float]], key: str, now: float) -> list[float]:
    """Drop timestamps outside the window; forget the key entirely when
    it empties, so the dict can't grow without bound."""
    stamps = [t for t in bucket.get(key, []) if now - t < _LOGIN_WINDOW_SECONDS]
    if stamps:
        bucket[key] = stamps
    else:
        bucket.pop(key, None)
    return stamps


def _rate_limit_check(email: str, ip: str) -> None:
    """Raise 429 when either bucket is over its limit. Called BEFORE the
    password is verified so a locked-out attacker burns no bcrypt time."""
    now = time.monotonic()
    if len(_prune(_failed_by_email, email, now)) >= _LOGIN_MAX_PER_EMAIL:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts, try again later",
        )
    if ip and len(_prune(_failed_by_ip, ip, now)) >= _LOGIN_MAX_PER_IP:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts, try again later",
        )


def _record_failure(email: str, ip: str) -> None:
    now = time.monotonic()
    _failed_by_email.setdefault(email, []).append(now)
    if ip:
        _failed_by_ip.setdefault(ip, []).append(now)


def _clear_failures(email: str) -> None:
    """Successful login wipes that email's counter. The IP bucket is
    left alone on purpose — one valid credential shouldn't reset the
    budget for every other account being probed from that address."""
    _failed_by_email.pop(email, None)


def _client_ip(request: Request) -> str:
    """Best-effort client IP. X-Forwarded-For is spoofable, so this is a
    speed bump, not an identity — the per-email limit is the load-bearing
    one. Takes the left-most hop when the header is present."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


# ─── Helpers ───────────────────────────────────────────────────────
def _to_public(doc: dict) -> DoctorPublic:
    """Convert raw Mongo doctor document → DoctorPublic (drops password)."""
    return DoctorPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        name=doc["name"],
        specialization=doc.get("specialization"),
        license_number=doc.get("license_number"),
        # .get() defaults cover rows created before these columns/keys
        # existed — Postgres backfills via the ALTER, Mongo documents
        # simply have no key.
        role=doc.get("role", "clinician"),
        is_active=doc.get("is_active", True),
        created_at=doc["created_at"],
    )


def _expiry_seconds() -> int:
    try:
        return max(60, int(os.environ.get("JWT_EXPIRY_SECONDS", "604800")))
    except ValueError:
        return 604800


# ─── POST /api/auth/signup ─────────────────────────────────────────
@router.post(
    "/signup",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(payload: DoctorSignupRequest):
    """Register a new doctor and issue an auth token.

    Email must be unique. Password is bcrypt-hashed (12 rounds) before
    storage. On success returns a JWT the client can use immediately.
    """
    db = get_db()
    email = payload.email.lower().strip()

    # Reject duplicate emails early — also enforced by the unique index
    # in db._ensure_indexes(), but a friendlier error is nicer than a
    # raw DuplicateKeyError surfacing as a 500.
    existing = await repo.doctors_find_one_by_email(db, email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    now = datetime.now(timezone.utc)
    doc = {
        "email": email,
        "password_hash": hash_password(payload.password),
        "name": payload.name,
        "specialization": payload.specialization,
        "license_number": payload.license_number,
        # SERVER-SIDE LITERALS. Never read from `payload` — that is the
        # whole privilege-escalation guard. DoctorSignupRequest does not
        # declare these fields (so Pydantic drops them from the body),
        # and the doc is built key-by-key rather than splatted, so a
        # client cannot self-promote by POSTing {"role": "admin"}.
        # Role changes go exclusively through repo.doctors_set_role,
        # which is admin-gated.
        "role": "clinician",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    new_id = await repo.doctors_insert(db, doc)
    doc["_id"] = new_id

    token = create_access_token(str(new_id))
    return AuthTokenResponse(
        token=token,
        expires_in=_expiry_seconds(),
        doctor=_to_public(doc),
    )


# ─── POST /api/auth/login ──────────────────────────────────────────
@router.post(
    "/login",
    response_model=AuthTokenResponse,
)
async def login(payload: DoctorLoginRequest, request: Request):
    """Verify credentials and issue an auth token."""
    db = get_db()
    email = payload.email.lower().strip()
    ip = _client_ip(request)

    # Checked first: a rate-limited caller never reaches bcrypt, so a
    # flood can't be used to burn CPU either.
    _rate_limit_check(email, ip)

    doc = await repo.doctors_find_one_by_email(db, email)
    # Use the same generic error for both "no such email" and "wrong
    # password" so attackers can't enumerate registered emails.
    if doc is None or not verify_password(payload.password, doc["password_hash"]):
        _record_failure(email, ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Deactivated accounts get the SAME generic message — telling the
    # caller "this account is deactivated" would confirm the email is
    # registered, defeating the enumeration guard above. The specific
    # "Account is deactivated" wording is only ever returned to a
    # caller who already proved possession of a valid token.
    if not doc.get("is_active", True):
        _record_failure(email, ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    _clear_failures(email)
    token = create_access_token(str(doc["_id"]))
    return AuthTokenResponse(
        token=token,
        expires_in=_expiry_seconds(),
        doctor=_to_public(doc),
    )


# ─── GET /api/auth/me ──────────────────────────────────────────────
@router.get(
    "/me",
    response_model=DoctorPublic,
)
async def get_me(current: dict = Depends(get_current_doctor)):
    """Return the currently-authenticated doctor's profile.

    Useful for the frontend to refresh user state on page load (e.g.
    after a hard refresh, the JWT in localStorage is still valid but
    the AuthContext needs to re-hydrate the doctor object).
    """
    return _to_public(current)
