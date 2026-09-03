"""Pydantic schemas for authentication endpoints.

Used by:
    auth_routes.py  — request/response validation for /api/auth/*
    admin_routes.py — DoctorPublic as the response model for /api/admin/users

There is no signup request model: public self-registration was removed,
so the only account-creation payload is AdminCreateUserRequest, defined
in routes/admin_routes.py behind require_admin.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ─── Sign in ───────────────────────────────────────────────────────
class DoctorLoginRequest(BaseModel):
    """Payload for POST /api/auth/login."""

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


# ─── Public doctor profile (no password hash) ──────────────────────
class DoctorPublic(BaseModel):
    """Doctor info returned to the frontend — never includes password."""

    id: str = Field(..., description="MongoDB _id as string")
    email: EmailStr
    name: str
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    # Authorisation fields. Response-only: no request model in this
    # file declares them, and the sole creation endpoint
    # (POST /api/admin/users) validates role against the ROLES
    # allowlist before it reaches the repository. Defaults cover doctor
    # rows written before these columns existed (Mongo documents in
    # particular carry no key at all).
    role: str = "clinician"
    is_active: bool = True
    created_at: datetime


# ─── Auth response (login) ─────────────────────────────────────────
class AuthTokenResponse(BaseModel):
    """Returned after a successful login."""

    success: bool = True
    token: str = Field(..., description="JWT bearer token")
    token_type: str = Field(default="Bearer")
    expires_in: int = Field(..., description="Token TTL in seconds")
    doctor: DoctorPublic
