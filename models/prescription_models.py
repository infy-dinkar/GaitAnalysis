"""Pydantic schemas for rehab prescription endpoints.

Prescription = a doctor-authored list of rehab exercise slugs the
patient should focus on. Overrides the auto-recommender when present.

One prescription per (doctor, patient). PUT is idempotent upsert;
DELETE removes the doctor override so the auto-recommender reasserts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PrescriptionUpsert(BaseModel):
    """Payload for PUT /api/patients/{patient_id}/prescription."""

    # Rehab exercise slugs (e.g. "squat", "pelvic-hold"). The FE
    # constrains this to the known 24-exercise catalogue at authoring
    # time; the API accepts any string and the recommender is
    # tolerant to unknown slugs (they simply produce no card).
    slugs: list[str] = Field(default_factory=list, min_length=0, max_length=24)

    # Optional per-slug or free-form notes. Kept as a free dict so the
    # UI can layer richer structure (target reps, per-exercise
    # comments) without a schema migration.
    notes: dict = Field(default_factory=dict)


class Prescription(BaseModel):
    """Response model for GET / PUT."""

    id: str
    patient_id: str
    doctor_id: str
    slugs: list[str]
    notes: dict
    created_at: datetime
    updated_at: datetime


class PrescriptionResponse(BaseModel):
    """Envelope so the frontend can distinguish "no prescription set"
    from "prescription empty" cleanly."""

    success: bool = True
    data: Optional[Prescription] = None
