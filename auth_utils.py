"""Authentication utilities — password hashing + JWT encode/decode.

Used by:
    auth_routes.py      — to issue tokens on signup/login
    patient_routes.py   — via get_current_doctor dependency
    report_routes.py    — same dependency

Design choices:
    • bcrypt with passlib (12 rounds — modern recommended cost).
    • JWT signed with HS256 and a long random JWT_SECRET (env var).
    • Token payload: {"sub": doctor_id, "exp": <unix_ts>}.
    • Tokens are stateless — server doesn't store them. Logout is a
      client-side "forget the token" operation. For revocation, we'd
      need a Redis denylist (out of scope for v1).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from bson import ObjectId
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from db import get_db


# ─── Password hashing ──────────────────────────────────────────────
# bcrypt with 12 rounds — costly enough to slow down brute-force,
# fast enough that signup/login feel snappy (< 200ms typical).
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(plain: str) -> str:
    """Hash a plain-text password for storage."""
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time check of a plain password against a stored hash."""
    return _pwd_ctx.verify(plain, hashed)


# ─── JWT issuance / validation ─────────────────────────────────────
def _jwt_secret() -> str:
    s = os.environ.get("JWT_SECRET", "").strip()
    if not s:
        raise RuntimeError(
            "JWT_SECRET env var is not set. Add a long random string to .env "
            "(generate with `python -c \"import secrets; print(secrets.token_urlsafe(64))\"`)."
        )
    if len(s) < 32:
        raise RuntimeError(
            f"JWT_SECRET is too short ({len(s)} chars). Use at least 32 chars."
        )
    return s


def _jwt_expiry_seconds() -> int:
    raw = os.environ.get("JWT_EXPIRY_SECONDS", "604800")  # 7 days default
    try:
        return max(60, int(raw))  # never less than 1 minute
    except ValueError:
        return 604800


def create_access_token(doctor_id: str) -> str:
    """Issue a signed JWT for the given doctor ID."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": doctor_id,                                   # subject = doctor _id
        "iat": int(now.timestamp()),                        # issued-at
        "exp": int((now + timedelta(seconds=_jwt_expiry_seconds())).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def decode_access_token(token: str) -> str:
    """Validate a JWT and return the doctor_id (subject). Raises 401 on failure."""
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return sub


# ─── FastAPI dependency: get the currently-logged-in doctor ────────
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_doctor(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """Resolve the Authorization header → loaded doctor document.

    Use this as a FastAPI dependency on every protected endpoint:

        @app.post("/api/patients")
        async def create(p: PatientCreate, doctor: dict = Depends(get_current_doctor)):
            ...

    Returns the full doctor document (without password_hash). Raises 401
    if missing token / invalid token / doctor not found.
    """
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header (expected 'Bearer <token>')",
            headers={"WWW-Authenticate": "Bearer"},
        )

    doctor_id_str = decode_access_token(creds.credentials)
    try:
        doctor_id = ObjectId(doctor_id_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid doctor id in token",
        )

    db = get_db()
    doctor = await db.doctors.find_one({"_id": doctor_id}, {"password_hash": 0})
    if doctor is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Doctor account not found",
        )
    return doctor
