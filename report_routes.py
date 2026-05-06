"""Report endpoints — save and retrieve analysis reports.

Routes mounted by api.py:
    POST   /api/patients/{patient_id}/reports  — save a new report
    GET    /api/patients/{patient_id}/reports  — list reports for a patient
    GET    /api/reports/{id}                   — full single report (with figures)
    DELETE /api/reports/{id}                   — remove a report

All endpoints require auth + verify the patient/report belongs to the
current doctor.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status

from auth_utils import get_current_doctor
from db import get_db
from report_models import (
    Report,
    ReportCreate,
    ReportDeleteResponse,
    ReportListResponse,
    ReportSummary,
)

# Two routers — one for nested-under-patient endpoints, one for top-level
# /api/reports/{id} lookups. Both included by api.py.
patient_reports_router = APIRouter(
    prefix="/api/patients/{patient_id}/reports",
    tags=["reports"],
)
reports_router = APIRouter(prefix="/api/reports", tags=["reports"])


# ─── Helpers ───────────────────────────────────────────────────────
def _parse_oid(raw: str, what: str) -> ObjectId:
    try:
        return ObjectId(raw)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {what}: {raw!r}",
        )


def _to_report(doc: dict) -> Report:
    return Report(
        id=str(doc["_id"]),
        patient_id=str(doc["patient_id"]),
        doctor_id=str(doc["doctor_id"]),
        module=doc["module"],
        body_part=doc.get("body_part"),
        movement=doc.get("movement"),
        side=doc.get("side"),
        metrics=doc.get("metrics") or {},
        figures=doc.get("figures") or [],
        observations=doc.get("observations") or {},
        video_filename=doc.get("video_filename"),
        video_size_bytes=doc.get("video_size_bytes"),
        # `.get(...)` keeps legacy docs (no keypoints key) returning None
        # rather than KeyError-ing.
        keypoints=doc.get("keypoints"),
        created_at=doc["created_at"],
    )


def _to_report_summary(doc: dict) -> ReportSummary:
    return ReportSummary(
        id=str(doc["_id"]),
        patient_id=str(doc["patient_id"]),
        doctor_id=str(doc["doctor_id"]),
        module=doc["module"],
        body_part=doc.get("body_part"),
        movement=doc.get("movement"),
        side=doc.get("side"),
        created_at=doc["created_at"],
    )


async def _verify_patient_owned(db, patient_id: ObjectId, doctor_id: ObjectId) -> None:
    """Raise 404 if the patient doesn't exist OR isn't owned by this doctor."""
    exists = await db.patients.count_documents(
        {"_id": patient_id, "doctor_id": doctor_id},
        limit=1,
    )
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )


# ─── POST /api/patients/{patient_id}/reports ───────────────────────
@patient_reports_router.post(
    "",
    response_model=Report,
    status_code=status.HTTP_201_CREATED,
)
async def create_report(
    patient_id: str,
    payload: ReportCreate,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Save a new analysis report against a patient."""
    db = get_db()
    pid = _parse_oid(patient_id, "patient id")
    await _verify_patient_owned(db, pid, current_doctor["_id"])

    now = datetime.now(timezone.utc)
    doc = {
        "patient_id": pid,
        "doctor_id": current_doctor["_id"],
        "module": payload.module,
        "body_part": payload.body_part,
        "movement": payload.movement,
        "side": payload.side,
        "metrics": payload.metrics,
        "figures": payload.figures,
        "observations": payload.observations,
        "video_filename": payload.video_filename,
        "video_size_bytes": payload.video_size_bytes,
        # Spec Section 2 (a): raw landmark stream is persisted alongside
        # metrics so saved posture sessions can be re-rendered later.
        "keypoints": payload.keypoints,
        "created_at": now,
    }
    result = await db.reports.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _to_report(doc)


# ─── GET /api/patients/{patient_id}/reports ────────────────────────
@patient_reports_router.get(
    "",
    response_model=ReportListResponse,
)
async def list_patient_reports(
    patient_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    """List all reports for a patient (newest first). Lightweight — no
    figures payload to keep response size small."""
    db = get_db()
    pid = _parse_oid(patient_id, "patient id")
    await _verify_patient_owned(db, pid, current_doctor["_id"])

    cursor = (
        db.reports
        .find(
            {"patient_id": pid},
            {"figures": 0, "metrics": 0, "observations": 0},  # exclude heavy fields
        )
        .sort("created_at", -1)
    )
    docs = await cursor.to_list(length=500)
    return ReportListResponse(
        data=[_to_report_summary(d) for d in docs],
        total=len(docs),
    )


# ─── GET /api/reports/{id} ─────────────────────────────────────────
@reports_router.get(
    "/{report_id}",
    response_model=Report,
)
async def get_report(
    report_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Fetch a single report including all figures + metrics."""
    db = get_db()
    rid = _parse_oid(report_id, "report id")
    doc = await db.reports.find_one(
        {"_id": rid, "doctor_id": current_doctor["_id"]}
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )
    return _to_report(doc)


# ─── DELETE /api/reports/{id} ──────────────────────────────────────
@reports_router.delete(
    "/{report_id}",
    response_model=ReportDeleteResponse,
)
async def delete_report(
    report_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    """Hard-delete a single report. Patient record stays intact."""
    db = get_db()
    rid = _parse_oid(report_id, "report id")
    result = await db.reports.delete_one(
        {"_id": rid, "doctor_id": current_doctor["_id"]}
    )
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )
    return ReportDeleteResponse(deleted_report_id=report_id)
