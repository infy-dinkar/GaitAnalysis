"""Patient CRUD endpoints — all protected, all doctor-scoped.

Mounted at /api/patients/* by api.py.

Endpoints:
    POST   /api/patients              — create a patient under current doctor
    GET    /api/patients              — list current doctor's patients
    GET    /api/patients/{id}         — single patient (must belong to current doctor)
    PATCH  /api/patients/{id}         — update patient fields
    DELETE /api/patients/{id}         — delete patient + all their reports

Authorization rule: a doctor can only access patients where
patient.doctor_id == current_doctor._id. Mongo queries always include
this filter to prevent IDOR (insecure direct object reference).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status

from auth_utils import get_current_doctor
from db import get_db
from patient_models import (
    Patient,
    PatientCreate,
    PatientDeleteResponse,
    PatientListResponse,
    PatientUpdate,
)

router = APIRouter(prefix="/api/patients", tags=["patients"])


# ─── Helpers ───────────────────────────────────────────────────────
def _to_patient(doc: dict, report_count: int = 0) -> Patient:
    """Convert raw Mongo document → Patient response model."""
    return Patient(
        id=str(doc["_id"]),
        doctor_id=str(doc["doctor_id"]),
        name=doc["name"],
        age=doc["age"],
        gender=doc["gender"],
        height_cm=doc["height_cm"],
        weight_kg=doc.get("weight_kg"),
        contact=doc.get("contact"),
        medical_notes=doc.get("medical_notes"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        report_count=report_count,
    )


def _parse_object_id(raw: str, what: str = "id") -> ObjectId:
    try:
        return ObjectId(raw)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {what}: {raw!r}",
        )


# ─── POST /api/patients ────────────────────────────────────────────
@router.post(
    "",
    response_model=Patient,
    status_code=status.HTTP_201_CREATED,
)
async def create_patient(
    payload: PatientCreate,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Create a new patient under the currently-logged-in doctor.

    The doctor_id is set automatically from the JWT — clients cannot
    create patients on behalf of a different doctor.
    """
    db = get_db()
    now = datetime.now(timezone.utc)
    doc = {
        "doctor_id": current_doctor["_id"],     # auto from JWT
        "name": payload.name,
        "age": payload.age,
        "gender": payload.gender,
        "height_cm": payload.height_cm,
        "weight_kg": payload.weight_kg,
        "contact": payload.contact,
        "medical_notes": payload.medical_notes,
        "created_at": now,
        "updated_at": now,
    }
    res = await db.patients.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _to_patient(doc, report_count=0)


# ─── GET /api/patients ─────────────────────────────────────────────
@router.get(
    "",
    response_model=PatientListResponse,
)
async def list_patients(
    current_doctor: dict = Depends(get_current_doctor),
):
    """List all patients owned by the current doctor (newest first)."""
    db = get_db()
    cursor = (
        db.patients
        .find({"doctor_id": current_doctor["_id"]})
        .sort("created_at", -1)
    )
    docs = await cursor.to_list(length=1000)

    # Aggregate report counts per patient in one query
    if docs:
        patient_ids = [d["_id"] for d in docs]
        pipeline = [
            {"$match": {"patient_id": {"$in": patient_ids}}},
            {"$group": {"_id": "$patient_id", "count": {"$sum": 1}}},
        ]
        counts_cursor = db.reports.aggregate(pipeline)
        counts = {c["_id"]: c["count"] async for c in counts_cursor}
    else:
        counts = {}

    patients = [_to_patient(d, counts.get(d["_id"], 0)) for d in docs]
    return PatientListResponse(data=patients, total=len(patients))


# ─── GET /api/patients/{id} ────────────────────────────────────────
@router.get(
    "/{patient_id}",
    response_model=Patient,
)
async def get_patient(
    patient_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Single patient by ID. 404 if not owned by current doctor."""
    db = get_db()
    pid = _parse_object_id(patient_id, "patient id")
    doc = await db.patients.find_one(
        {"_id": pid, "doctor_id": current_doctor["_id"]}
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )
    report_count = await db.reports.count_documents({"patient_id": pid})
    return _to_patient(doc, report_count)


# ─── PATCH /api/patients/{id} ──────────────────────────────────────
@router.patch(
    "/{patient_id}",
    response_model=Patient,
)
async def update_patient(
    patient_id: str,
    payload: PatientUpdate,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Partially update a patient. Only owned-patient access."""
    db = get_db()
    pid = _parse_object_id(patient_id, "patient id")

    update_fields = {
        k: v for k, v in payload.model_dump(exclude_unset=True).items()
        if v is not None
    }
    if not update_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )
    update_fields["updated_at"] = datetime.now(timezone.utc)

    result = await db.patients.find_one_and_update(
        {"_id": pid, "doctor_id": current_doctor["_id"]},
        {"$set": update_fields},
        return_document=True,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )
    report_count = await db.reports.count_documents({"patient_id": pid})
    return _to_patient(result, report_count)


# ─── DELETE /api/patients/{id} ─────────────────────────────────────
@router.delete(
    "/{patient_id}",
    response_model=PatientDeleteResponse,
)
async def delete_patient(
    patient_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Hard-delete a patient and ALL their reports (cascade).

    Atomic from the doctor's perspective — patient ownership is verified
    before deletion. Reports are removed in a separate query (Mongo M0
    free tier doesn't support multi-document transactions).
    """
    db = get_db()
    pid = _parse_object_id(patient_id, "patient id")

    # Verify ownership before deleting
    existing = await db.patients.find_one(
        {"_id": pid, "doctor_id": current_doctor["_id"]}
    )
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )

    # Cascade-delete reports first
    reports_result = await db.reports.delete_many({"patient_id": pid})
    await db.patients.delete_one({"_id": pid})

    return PatientDeleteResponse(
        deleted_patient_id=patient_id,
        deleted_reports_count=reports_result.deleted_count,
    )
