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
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from auth_models import (
    AuthTokenResponse,
    DoctorLoginRequest,
    DoctorPublic,
    DoctorSignupRequest,
)
from auth_utils import (
    create_access_token,
    get_current_doctor,
    hash_password,
    verify_password,
)
from db import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Helpers ───────────────────────────────────────────────────────
def _to_public(doc: dict) -> DoctorPublic:
    """Convert raw Mongo doctor document → DoctorPublic (drops password)."""
    return DoctorPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        name=doc["name"],
        specialization=doc.get("specialization"),
        license_number=doc.get("license_number"),
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
    existing = await db.doctors.find_one({"email": email})
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
        "created_at": now,
        "updated_at": now,
    }
    result = await db.doctors.insert_one(doc)
    doc["_id"] = result.inserted_id

    token = create_access_token(str(result.inserted_id))
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
async def login(payload: DoctorLoginRequest):
    """Verify credentials and issue an auth token."""
    db = get_db()
    email = payload.email.lower().strip()

    doc = await db.doctors.find_one({"email": email})
    # Use the same generic error for both "no such email" and "wrong
    # password" so attackers can't enumerate registered emails.
    if doc is None or not verify_password(payload.password, doc["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

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
