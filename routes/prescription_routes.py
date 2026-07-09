"""Rehab prescription endpoints — one prescription per (doctor, patient).

Mounted at /api/patients/{patient_id}/prescription by api.py.

Endpoints:
    GET    /api/patients/{patient_id}/prescription   — read
    PUT    /api/patients/{patient_id}/prescription   — upsert
    DELETE /api/patients/{patient_id}/prescription   — remove doctor override

Authorization: the patient must be owned by the current doctor
(same rule as patient_routes / report_routes).
"""
from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status

from models.prescription_models import (
    Prescription,
    PrescriptionResponse,
    PrescriptionUpsert,
)
from utils.auth_utils import get_current_doctor
from utils.db import get_db

router = APIRouter(prefix="/api/patients", tags=["prescriptions"])


def _parse_object_id(raw: str, what: str = "id") -> ObjectId:
    try:
        return ObjectId(raw)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {what}: {raw!r}",
        )


async def _ensure_patient_owned(
    db, patient_id: ObjectId, doctor_oid: ObjectId
) -> None:
    """Reject requests for patients not owned by the current doctor."""
    patient = await db.patients.find_one(
        {"_id": patient_id, "doctor_id": doctor_oid},
        {"_id": 1},
    )
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )


def _to_prescription(doc: dict) -> Prescription:
    return Prescription(
        id=str(doc["_id"]),
        patient_id=str(doc["patient_id"]),
        doctor_id=str(doc["doctor_id"]),
        slugs=list(doc.get("slugs", [])),
        notes=dict(doc.get("notes", {})),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


# ─── GET /api/patients/{patient_id}/prescription ──────────────────
@router.get(
    "/{patient_id}/prescription",
    response_model=PrescriptionResponse,
)
async def get_prescription(
    patient_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Return the doctor-saved prescription for a patient.

    Response envelope { success, data } — data is null when no
    prescription has been saved yet (the FE then falls back to the
    auto recommender). This is intentionally NOT a 404, so the
    frontend has a single code path for "prescribed set for X".
    """
    db = get_db()
    patient_oid = _parse_object_id(patient_id, "patient_id")
    doctor_oid = current_doctor["_id"]
    await _ensure_patient_owned(db, patient_oid, doctor_oid)

    doc = await db.prescriptions.find_one(
        {"patient_id": patient_oid, "doctor_id": doctor_oid}
    )
    if doc is None:
        return PrescriptionResponse(success=True, data=None)
    return PrescriptionResponse(success=True, data=_to_prescription(doc))


# ─── PUT /api/patients/{patient_id}/prescription ──────────────────
@router.put(
    "/{patient_id}/prescription",
    response_model=Prescription,
)
async def upsert_prescription(
    patient_id: str,
    payload: PrescriptionUpsert,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Create-or-update the prescription for a patient.

    Idempotent — repeated PUTs replace the slug list wholesale. Notes
    are also replaced wholesale (send the full state, not a diff).
    """
    db = get_db()
    patient_oid = _parse_object_id(patient_id, "patient_id")
    doctor_oid = current_doctor["_id"]
    await _ensure_patient_owned(db, patient_oid, doctor_oid)

    # De-dupe slugs while preserving order — nicer UX than silently
    # accepting duplicates and returning them.
    seen: set[str] = set()
    slugs: list[str] = []
    for s in payload.slugs:
        if not isinstance(s, str) or not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        slugs.append(s)

    now = datetime.now(timezone.utc)
    update = {
        "$set": {
            "slugs": slugs,
            "notes": payload.notes,
            "updated_at": now,
        },
        "$setOnInsert": {
            "patient_id": patient_oid,
            "doctor_id": doctor_oid,
            "created_at": now,
        },
    }
    await db.prescriptions.update_one(
        {"patient_id": patient_oid, "doctor_id": doctor_oid},
        update,
        upsert=True,
    )
    doc = await db.prescriptions.find_one(
        {"patient_id": patient_oid, "doctor_id": doctor_oid}
    )
    if doc is None:
        # Should never happen after a successful upsert, but a
        # defensive branch keeps mypy happy and surfaces the surprise.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prescription upsert did not persist",
        )
    return _to_prescription(doc)


# ─── DELETE /api/patients/{patient_id}/prescription ───────────────
@router.delete(
    "/{patient_id}/prescription",
    response_model=PrescriptionResponse,
)
async def delete_prescription(
    patient_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Remove the doctor prescription so the auto recommender wins."""
    db = get_db()
    patient_oid = _parse_object_id(patient_id, "patient_id")
    doctor_oid = current_doctor["_id"]
    await _ensure_patient_owned(db, patient_oid, doctor_oid)

    await db.prescriptions.delete_one(
        {"patient_id": patient_oid, "doctor_id": doctor_oid}
    )
    return PrescriptionResponse(success=True, data=None)
