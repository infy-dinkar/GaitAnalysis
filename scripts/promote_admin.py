#!/usr/bin/env python
"""Promote an existing doctor to admin.

    python scripts/promote_admin.py <email>

Inside the deployed container:

    docker exec -it motionlens-api python scripts/promote_admin.py you@clinic.com

Why a script and not an endpoint
--------------------------------
This is the bootstrap for the FIRST admin — the one that cannot be
created through /api/admin/users, because that route already requires an
admin. Doing it as a CLI means:

  • it needs shell + DB access, so it adds no network attack surface,
  • there is no dormant "create the first admin" route left reachable
    forever afterwards,
  • no admin password ever has to live in an env var or .env file.

It only PROMOTES an existing account — it never creates one and never
touches passwords. Sign up normally first (public signup is still
enabled), then promote that account here.

Both DB backends are supported: it reuses the app's own connect/
repository layer, so DB_BACKEND, PG_* and MONGODB_URI are read from
exactly the same environment the API uses.
"""
from __future__ import annotations

import asyncio
import os
import sys

# Allow `python scripts/promote_admin.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import db as db_module            # noqa: E402
from utils import repositories as repo       # noqa: E402
from utils.auth_utils import ROLES           # noqa: E402


def _fmt(doc: dict) -> str:
    return (
        f"id={doc['_id']}  email={doc['email']}  name={doc.get('name')!r}  "
        f"role={doc.get('role', 'clinician')}  "
        f"is_active={doc.get('is_active', True)}"
    )


async def _run(email: str) -> int:
    email = email.lower().strip()

    # Same startup path the API uses, so the script talks to whichever
    # backend DB_BACKEND selects.
    await db_module.connect()
    await db_module.postgres_healthcheck()
    db = db_module.get_db()

    try:
        doc = await repo.doctors_find_one_by_email(db, email)
        if doc is None:
            print(f"ERROR: no doctor found with email {email!r}.", file=sys.stderr)
            print(
                "       Sign up through the app first, then re-run this script.",
                file=sys.stderr,
            )
            return 1

        print("BEFORE:", _fmt(doc))

        if doc.get("role") == "admin" and doc.get("is_active", True):
            print("Already an active admin — nothing to do.")
            return 0

        assert "admin" in ROLES  # guard against a typo drifting from the allowlist
        await repo.doctors_set_role(db, doc["_id"], "admin")
        updated = await repo.doctors_set_active(db, doc["_id"], True)

        if updated is None:
            updated = await repo.doctors_find_one_by_id(
                db, doc["_id"], include_password=False
            )
        print("AFTER: ", _fmt(updated))
        print(f"\n{email} is now an admin. Sign out and back in to refresh the UI.")
        return 0
    finally:
        await db_module.disconnect()
        await db_module.postgres_disconnect()


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    return asyncio.run(_run(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
