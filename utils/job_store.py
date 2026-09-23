"""utils/job_store.py — file-based job store for long analyses.

Deliberately NOT a database table. Job state is throwaway: it exists
only between the POST that starts an analysis and the poll that
collects the result. Putting it in Postgres would need DDL from a
superuser (the app connects as a non-owner — see utils/db.py) and would
leave operational rows in a clinical database. The container filesystem
is the right lifetime for it.

Two gunicorn workers share this filesystem, so it also doubles as the
cross-worker coordination point:

  • job files      — one JSON per job, written atomically
  • RUN_LOCK       — an advisory lock held for the duration of an
                     analysis, so only one runs machine-wide
  • DISABLED       — kill switch; its presence turns the job endpoint
                     off without a restart

Every write is temp-file + os.replace, which is atomic on POSIX and on
Windows (since Python 3.3), so a reader never sees a half-written file
even if a worker dies mid-write.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Locking primitive ─────────────────────────────────────────────
# Production is Linux (Dockerfile.aws), where fcntl.flock is the right
# tool: it is advisory, it blocks, and critically the OS releases it if
# the holding process dies — a crashed worker cannot wedge the queue.
# Windows has no fcntl, so local development falls back to msvcrt,
# which gives the same three properties.
try:                                     # pragma: no cover - platform
    import fcntl

    def _lock_exclusive(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

    _LOCK_IMPL = "fcntl.flock"
except ImportError:                      # pragma: no cover - platform
    import msvcrt

    def _lock_exclusive(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    def _unlock(fd: int) -> None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

    _LOCK_IMPL = "msvcrt.locking"

JOBS_DIR = os.environ.get("MOTIONLENS_JOBS_DIR", os.path.join(tempfile.gettempdir(), "motionlens_jobs"))
DISABLED_FLAG = "DISABLED"
RUN_LOCK = ".run.lock"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

#: A running job must refresh its heartbeat more often than this or it
#: is presumed dead. Generous against a fully saturated 2-vCPU box.
STALE_AFTER_SEC = 60
#: Heartbeat interval used by the runner.
HEARTBEAT_EVERY_SEC = 10
#: Finished job files are swept after this long.
JOB_TTL_SEC = 2 * 60 * 60

INTERRUPTED_MESSAGE = "Analysis was interrupted — please upload again"


def _ensure_dir() -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)


def _path(job_id: str) -> str:
    # Job ids are generated here (uuid4 hex), never taken from a client
    # without passing through is_valid_id() first.
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def is_valid_id(job_id: str) -> bool:
    """Reject anything that is not one of our own ids.

    The id reaches us from a URL path, so this is what stops
    `../../etc/passwd` becoming a file read.
    """
    return (
        isinstance(job_id, str)
        and len(job_id) == 32
        and all(c in "0123456789abcdef" for c in job_id)
    )


def is_disabled() -> bool:
    """Kill switch. Checked per request so it needs no restart."""
    return os.path.exists(os.path.join(JOBS_DIR, DISABLED_FLAG))


def _write_atomic(job_id: str, payload: dict[str, Any]) -> None:
    _ensure_dir()
    fd, tmp = tempfile.mkstemp(dir=JOBS_DIR, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _path(job_id))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read(job_id: str) -> Optional[dict[str, Any]]:
    """Load a job, or None if it does not exist / is unreadable.

    Never raises: a poll must not 500 because a file was mid-replace on
    a filesystem where that is not atomic.
    """
    if not is_valid_id(job_id):
        return None
    try:
        with open(_path(job_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def create(kind: str, meta: Optional[dict[str, Any]] = None) -> str:
    """Create a queued job and return its id."""
    _ensure_dir()
    sweep()
    job_id = uuid.uuid4().hex
    now = time.time()
    _write_atomic(job_id, {
        "id": job_id,
        "kind": kind,
        "status": STATUS_QUEUED,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "heartbeat_at": None,
        "error": None,
        "result": None,
        "meta": meta or {},
    })
    return job_id


def update(job_id: str, **fields: Any) -> None:
    """Merge fields into a job. Silently ignores a vanished job."""
    cur = read(job_id)
    if cur is None:
        return
    cur.update(fields)
    _write_atomic(job_id, cur)


def heartbeat(job_id: str) -> None:
    update(job_id, heartbeat_at=time.time())


def mark_running(job_id: str) -> None:
    now = time.time()
    update(job_id, status=STATUS_RUNNING, started_at=now, heartbeat_at=now)


def mark_done(job_id: str, result: Any) -> None:
    update(
        job_id,
        status=STATUS_DONE,
        result=result,
        error=None,
        finished_at=time.time(),
    )


def mark_error(job_id: str, message: str) -> None:
    update(
        job_id,
        status=STATUS_ERROR,
        error=message,
        result=None,
        finished_at=time.time(),
    )


def public_view(job: dict[str, Any]) -> dict[str, Any]:
    """The shape the poll endpoint returns.

    A job left `running` with a stale heartbeat is reported as an error
    without rewriting the file: the worker that owned it is gone, so
    nobody is going to finish it, and the browser needs to stop polling.
    """
    status = job.get("status")
    error = job.get("error")
    if status == STATUS_RUNNING:
        hb = job.get("heartbeat_at") or job.get("started_at") or 0
        if time.time() - float(hb) > STALE_AFTER_SEC:
            status = STATUS_ERROR
            error = INTERRUPTED_MESSAGE
    return {
        "job_id": job.get("id"),
        "status": status,
        "error": error,
        "result": job.get("result") if status == STATUS_DONE else None,
    }


def sweep(ttl_sec: int = JOB_TTL_SEC) -> int:
    """Delete job files older than `ttl_sec`. Returns how many went.

    Called on job creation rather than from a scheduler — there is no
    scheduler in this service, and a sweep on create is bounded, cheap
    and happens exactly when new files appear.
    """
    _ensure_dir()
    cutoff = time.time() - ttl_sec
    removed = 0
    try:
        names = os.listdir(JOBS_DIR)
    except OSError:
        return 0
    for name in names:
        if not name.endswith(".json"):
            continue
        p = os.path.join(JOBS_DIR, name)
        try:
            if os.path.getmtime(p) < cutoff:
                os.unlink(p)
                removed += 1
        except OSError:
            continue
    if removed:
        log.info("job sweep removed %d expired job file(s)", removed)
    return removed


#: Serialises threads WITHIN one process. The file lock below handles
#: the cross-process half, but relying on it for threads too is a trap:
#: fcntl.flock is per open-file-description, so it happens to work on
#: Linux, while Windows' msvcrt.locking is per-process and does not —
#: a local test had a second job wait on the lock forever. One extra
#: in-process lock makes the behaviour identical on both.
_THREAD_SLOT = threading.Lock()


class RunSlot:
    """Machine-wide 'one analysis at a time' gate.

    Two layers, because there are two kinds of concurrency:

      • threading.Lock — other threads in THIS gunicorn worker
      • file lock      — the OTHER gunicorn worker, via the shared
                         container filesystem

    Jobs waiting here stay `queued`, which is the honest status — they
    have not started.

    The file lock is advisory and OS-held: if a worker is killed while
    running, the kernel drops it and the next job proceeds. Nothing has
    to time out or clean up.
    """

    def __init__(self) -> None:
        _ensure_dir()
        self._path = os.path.join(JOBS_DIR, RUN_LOCK)
        self._fd: Optional[int] = None
        self._held_thread_lock = False

    def __enter__(self) -> "RunSlot":
        _THREAD_SLOT.acquire()
        self._held_thread_lock = True
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        # Needs a byte to lock on for the msvcrt fallback; harmless on
        # POSIX, where flock locks the whole file regardless.
        try:
            if os.path.getsize(self._path) == 0:
                os.write(self._fd, b"\0")
            os.lseek(self._fd, 0, os.SEEK_SET)
        except OSError:
            pass
        _lock_exclusive(self._fd)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._fd is not None:
                try:
                    _unlock(self._fd)
                finally:
                    try:
                        os.close(self._fd)
                    except OSError:
                        pass
                    self._fd = None
        finally:
            if self._held_thread_lock:
                self._held_thread_lock = False
                _THREAD_SLOT.release()
