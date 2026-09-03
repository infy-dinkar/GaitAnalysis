"""Admin user-management endpoints.

Mounted at /api/admin/* by api.py. EVERY route in this file is gated by
Depends(require_admin), which layers on top of get_current_doctor — so
each request is authenticated, the account is confirmed active, and the
role is confirmed to be "admin" before any handler body runs.

Endpoints:
    POST  /api/admin/users              — create a clinician or admin
    GET   /api/admin/users              — list every doctor
    PATCH /api/admin/users/{id}         — change role / activate-deactivate
    POST  /api/admin/users/{id}/password — set a new password

Scope boundary — deliberately narrow:
    Admin is a USER-MANAGEMENT role, not a data-access role. It grants
    visibility of the doctors list only. Patients and reports remain
    strictly owner-scoped (patients_list_by_doctor et al.) and nothing
    here touches that scoping, so an admin still cannot see another
    doctor's patients.

Password handling:
    Hashes are one-way, so an admin can never read an existing password
    — only overwrite it. The reset endpoint takes a new plain password,
    hashes it with the shared bcrypt context (utils.auth_utils), and
    stores only the hash. No response in this module ever carries password_hash:
    every doctor is serialised through DoctorPublic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from models.auth_models import DoctorPublic
from utils.auth_utils import ROLES, hash_password, require_admin
from utils.db import get_db
from utils import repositories as repo

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ─── Schemas ───────────────────────────────────────────────────────
class AdminCreateUserRequest(BaseModel):
    """Body for POST /api/admin/users.

    This is the ONLY account-creation payload in the app (public signup
    was removed). It DOES accept `role`, which is safe precisely
    because the route is admin-gated and the value is validated against
    the ROLES allowlist below before it reaches the repository."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=2, max_length=100)
    role: str = Field(default="clinician")
    specialization: Optional[str] = Field(default=None, max_length=100)
    license_number: Optional[str] = Field(default=None, max_length=50)

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        if v not in ROLES:
            raise ValueError(f"role must be one of {list(ROLES)}")
        return v

    @field_validator("name", "specialization", "license_number")
    @classmethod
    def _strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v


class AdminUpdateUserRequest(BaseModel):
    """Body for PATCH /api/admin/users/{id}. Both fields optional — send
    either or both."""

    role: Optional[str] = None
    is_active: Optional[bool] = None

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ROLES:
            raise ValueError(f"role must be one of {list(ROLES)}")
        return v


class AdminSetPasswordRequest(BaseModel):
    """Body for POST /api/admin/users/{id}/password."""

    new_password: str = Field(..., min_length=8, max_length=128)


class AdminUserListResponse(BaseModel):
    success: bool = True
    data: list[DoctorPublic]
    total: int


# ─── Helpers ───────────────────────────────────────────────────────
def _to_public(doc: dict) -> DoctorPublic:
    """Raw doctor row/document → DoctorPublic. The single serialisation
    point for this module, so password_hash can never leak: DoctorPublic
    has no such field and Pydantic drops unknown keys."""
    return DoctorPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        name=doc["name"],
        specialization=doc.get("specialization"),
        license_number=doc.get("license_number"),
        role=doc.get("role", "clinician"),
        is_active=doc.get("is_active", True),
        created_at=doc["created_at"],
    )


def _parse_object_id(raw: str) -> ObjectId:
    """Doctor ids are ObjectId on Mongo and the same hex string in the
    Postgres TEXT id column, so one parse works for both backends."""
    try:
        return ObjectId(raw)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user id",
        )


# ─── POST /api/admin/users ─────────────────────────────────────────
@router.post(
    "/users",
    response_model=DoctorPublic,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    payload: AdminCreateUserRequest,
    _admin: dict = Depends(require_admin),
):
    """Create a clinician or admin account."""
    db = get_db()
    email = payload.email.lower().strip()

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
        # payload.role already passed the ROLES allowlist in the
        # validator above; new accounts always start active.
        "role": payload.role,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    new_id = await repo.doctors_insert(db, doc)
    doc["_id"] = new_id
    return _to_public(doc)


# ─── GET /api/admin/users ──────────────────────────────────────────
@router.get(
    "/users",
    response_model=AdminUserListResponse,
)
async def list_users(_admin: dict = Depends(require_admin)):
    """Every doctor, newest first. Password hashes are excluded at the
    query level (DOCTORS_COLS_NO_PW / Mongo projection), not just at
    serialisation."""
    db = get_db()
    docs = await repo.doctors_list_all(db)
    users = [_to_public(d) for d in docs]
    return AdminUserListResponse(data=users, total=len(users))


# ─── PATCH /api/admin/users/{id} ───────────────────────────────────
@router.patch(
    "/users/{user_id}",
    response_model=DoctorPublic,
)
async def update_user(
    user_id: str,
    payload: AdminUpdateUserRequest,
    admin: dict = Depends(require_admin),
):
    """Change a doctor's role and/or active flag.

    Two guards, both of which exist to prevent an irrecoverable state:

      1. An admin cannot deactivate themselves — an easy way to lock
         yourself out mid-session by mis-clicking your own row.
      2. The LAST ACTIVE ADMIN cannot be demoted or deactivated. There
         is no public signup and no self-service role change, so zero
         active admins means nobody can reach /api/admin/* again — and
         no new account can be created at all. Recovery would require
         direct DB access.

    Both are reversible operations otherwise, so there is no confirm
    step — the guards are the safety net.
    """
    if payload.role is None and payload.is_active is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nothing to update — provide role and/or is_active",
        )

    db = get_db()
    uid = _parse_object_id(user_id)
    target = await repo.doctors_find_one_by_id(db, uid, include_password=False)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    is_self = str(target["_id"]) == str(admin["_id"])

    # Guard 1 — self-deactivation.
    if is_self and payload.is_active is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own account",
        )

    # Guard 2 — last active admin. Only relevant when the target is
    # CURRENTLY an active admin and the change would strip that.
    target_is_active_admin = (
        target.get("role") == "admin" and target.get("is_active", True)
    )
    losing_admin = payload.role is not None and payload.role != "admin"
    losing_active = payload.is_active is False
    if target_is_active_admin and (losing_admin or losing_active):
        active_admins = await repo.doctors_count_active_admins(db)
        if active_admins <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "This is the last active admin. Promote another admin "
                    "first, otherwise nobody could manage users."
                ),
            )

    updated = target
    # Sequential single-field writes: role and is_active each have their
    # own dedicated repo function, keeping doctors_set_role the sole
    # write path for the role column.
    if payload.role is not None:
        updated = await repo.doctors_set_role(db, uid, payload.role)
    if payload.is_active is not None:
        updated = await repo.doctors_set_active(db, uid, payload.is_active)

    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return _to_public(updated)


# ─── POST /api/admin/users/{id}/password ───────────────────────────
@router.post(
    "/users/{user_id}/password",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def set_user_password(
    user_id: str,
    payload: AdminSetPasswordRequest,
    _admin: dict = Depends(require_admin),
):
    """Overwrite a doctor's password.

    The admin supplies a new plain password; only its bcrypt hash is
    stored. Existing passwords are never readable — hashes are one-way,
    so this endpoint can set but not reveal. 204 with no body so the
    new password is never echoed back in a response.
    """
    db = get_db()
    uid = _parse_object_id(user_id)
    target = await repo.doctors_find_one_by_id(db, uid, include_password=False)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    await repo.doctors_set_password_hash(
        db, uid, hash_password(payload.new_password)
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
