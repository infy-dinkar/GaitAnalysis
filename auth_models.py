"""Pydantic schemas for authentication endpoints.

Used by:
    auth_routes.py — request/response validation for /api/auth/*
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─── Sign up ───────────────────────────────────────────────────────
class DoctorSignupRequest(BaseModel):
    """Payload for POST /api/auth/signup."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=2, max_length=100)
    specialization: Optional[str] = Field(default=None, max_length=100)
    license_number: Optional[str] = Field(default=None, max_length=50)

    @field_validator("name", "specialization", "license_number")
    @classmethod
    def _strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v


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
    created_at: datetime


# ─── Auth response (signup + login both return this) ───────────────
class AuthTokenResponse(BaseModel):
    """Returned after successful signup or login."""

    success: bool = True
    token: str = Field(..., description="JWT bearer token")
    token_type: str = Field(default="Bearer")
    expires_in: int = Field(..., description="Token TTL in seconds")
    doctor: DoctorPublic
