"""Pydantic schemas for patient endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Create / update payloads ──────────────────────────────────────
class PatientCreate(BaseModel):
    """Payload for POST /api/patients."""

    name: str = Field(..., min_length=1, max_length=120)
    age: int = Field(..., ge=0, le=150)
    gender: Literal["male", "female", "other"]
    height_cm: float = Field(..., ge=30, le=250)
    weight_kg: Optional[float] = Field(default=None, ge=2, le=400)
    contact: Optional[str] = Field(default=None, max_length=50)
    medical_notes: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("name", "contact", "medical_notes")
    @classmethod
    def _strip(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v


class PatientUpdate(BaseModel):
    """Payload for PATCH /api/patients/{id} — all fields optional."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    age: Optional[int] = Field(default=None, ge=0, le=150)
    gender: Optional[Literal["male", "female", "other"]] = None
    height_cm: Optional[float] = Field(default=None, ge=30, le=250)
    weight_kg: Optional[float] = Field(default=None, ge=2, le=400)
    contact: Optional[str] = Field(default=None, max_length=50)
    medical_notes: Optional[str] = Field(default=None, max_length=2000)


# ─── Public response shape ─────────────────────────────────────────
class Patient(BaseModel):
    """Patient document returned to the frontend."""

    id: str
    doctor_id: str
    name: str
    age: int
    gender: str
    height_cm: float
    weight_kg: Optional[float] = None
    contact: Optional[str] = None
    medical_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    report_count: int = 0


class PatientListResponse(BaseModel):
    success: bool = True
    data: list[Patient]
    total: int


class PatientDeleteResponse(BaseModel):
    success: bool = True
    deleted_patient_id: str
    deleted_reports_count: int
