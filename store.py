"""
store.py
--------
Persistent sync state, backed by SQLite.

The flat text caches this replaces could only record "seen", which conflated
three very different outcomes: the track was synced, the track genuinely does
not exist on the other service, and the lookup failed for a transient reason.
Because a failed search wrote the same line as a successful one, a momentary
API error permanently retired a track — a 451 during one run meant that track
was never looked at again.

Here each entry carries a status, an attempt counter and timestamps, so
transient failures retry, genuine misses retry a few times over the following
weeks, and successes are recorded together with the IDs on both sides.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Sync directions.
SP_TO_YA = "sp2ya"
YA_TO_SP = "ya2sp"

# Outcomes.
STATUS_SYNCED = "synced"
STATUS_NOT_FOUND = "not_found"
STATUS_ERROR = "error"

# A track that could not be found is retried this many times, spaced this far
# apart, before it is given up on. Catalogues do change.
NOT_FOUND_RETRY_DAYS = 7
NOT_FOUND_MAX_ATTEMPTS = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_state (
    direction    TEXT NOT NULL,
    key          TEXT NOT NULL,
    status       TEXT NOT NULL,
    src_id       TEXT,
    dst_id       TEXT,
    score        REAL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    first_seen   TEXT NOT NULL,
    last_attempt TEXT NOT NULL,
    detail       TEXT,
    PRIMARY KEY (direction, key)
);
CREATE INDEX IF NOT EXISTS idx_sync_state_status ON sync_state (direction, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SyncStore:
    """Records what has been synced, in which direction, and how it went."""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "SyncStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- queries ---------------------------------------------------------

    def get(self, direction: str, key: str) -> Optional[sqlite3.Row]:
        cur = self._db.execute(
            "SELECT * FROM sync_state WHERE direction = ? AND key = ?", (direction, key)
        )
        return cur.fetchone()

    def should_attempt(self, direction: str, key: str) -> bool:
        """
        Whether this track is worth looking up on this run.

        Successes are final. Errors always retry, because the failure said
        nothing about the track. Misses retry on a schedule and eventually
        stop, so a track that truly is not in the other catalogue does not
        cost a search forever.
        """
        row = self.get(direction, key)
        if row is None:
            return True
        if row["status"] == STATUS_SYNCED:
            return False
        if row["status"] == STATUS_ERROR:
            return True

        if row["attempts"] >= NOT_FOUND_MAX_ATTEMPTS:
            return False
        try:
            last = datetime.fromisoformat(row["last_attempt"])
        except (TypeError, ValueError):
            return True
        return datetime.now(timezone.utc) - last >= timedelta(days=NOT_FOUND_RETRY_DAYS)

    def counts(self, direction: Optional[str] = None) -> dict[str, int]:
        """Row counts per status, for logging a summary."""
        if direction:
            cur = self._db.execute(
                "SELECT status, COUNT(*) AS n FROM sync_state WHERE direction = ? GROUP BY status",
                (direction,),
            )
        else:
            cur = self._db.execute(
                "SELECT status, COUNT(*) AS n FROM sync_state GROUP BY status"
            )
        return {row["status"]: row["n"] for row in cur.fetchall()}

    # -- mutations -------------------------------------------------------

    def _upsert(self, direction: str, key: str, status: str, *,
                src_id: Optional[str] = None, dst_id: Optional[str] = None,
                score: Optional[float] = None, detail: Optional[str] = None,
                bump_attempts: bool = True) -> None:
        now = _now()
        self._db.execute(
            """
            INSERT INTO sync_state
                (direction, key, status, src_id, dst_id, score,
                 attempts, first_seen, last_attempt, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(direction, key) DO UPDATE SET
                status       = excluded.status,
                src_id       = COALESCE(excluded.src_id, sync_state.src_id),
                dst_id       = COALESCE(excluded.dst_id, sync_state.dst_id),
                score        = COALESCE(excluded.score, sync_state.score),
                attempts     = sync_state.attempts + ?,
                last_attempt = excluded.last_attempt,
                detail       = excluded.detail
            """,
            (direction, key, status, src_id, dst_id, score,
             1 if bump_attempts else 0, now, now, detail,
             1 if bump_attempts else 0),
        )
        self._db.commit()

    def mark_synced(self, direction: str, key: str, *, src_id: Optional[str] = None,
                    dst_id: Optional[str] = None, score: Optional[float] = None,
                    detail: Optional[str] = None) -> None:
        self._upsert(direction, key, STATUS_SYNCED, src_id=src_id, dst_id=dst_id,
                     score=score, detail=detail)

    def mark_not_found(self, direction: str, key: str, *,
                       best_score: Optional[float] = None,
                       detail: Optional[str] = None) -> None:
        self._upsert(direction, key, STATUS_NOT_FOUND, score=best_score, detail=detail)

    def mark_error(self, direction: str, key: str, detail: str) -> None:
        """Record a transient failure. Never blocks a later retry."""
        self._upsert(direction, key, STATUS_ERROR, detail=detail[:500])

    # -- migration -------------------------------------------------------

    def import_legacy_cache(self, direction: str, path: str) -> int:
        """
        Import a v2 flat text cache, one key per line.

        Existing rows win, so re-running this is harmless and never downgrades
        a richer record back to a bare migrated one.
        """
        if not os.path.exists(path):
            return 0

        with open(path, "r", encoding="utf-8") as fh:
            keys = [line.strip() for line in fh if line.strip()]
        if not keys:
            return 0

        now = _now()
        self._db.executemany(
            """
            INSERT OR IGNORE INTO sync_state
                (direction, key, status, attempts, first_seen, last_attempt, detail)
            VALUES (?, ?, ?, 0, ?, ?, 'migrated from flat cache')
            """,
            [(direction, key, STATUS_SYNCED, now, now) for key in keys],
        )
        self._db.commit()
        imported = self._db.execute(
            "SELECT COUNT(*) FROM sync_state WHERE direction = ?", (direction,)
        ).fetchone()[0]
        logger.info("Store: imported '%s' — %d entries now in direction '%s'.",
                    path, imported, direction)
        return imported

    def bulk_mark_synced(self, direction: str, keys: Iterable[str],
                         detail: str = "baseline snapshot") -> int:
        """Record a set of keys as already synced, without touching existing rows."""
        now = _now()
        rows = [(direction, key, STATUS_SYNCED, now, now, detail) for key in keys]
        if not rows:
            return 0
        self._db.executemany(
            """
            INSERT OR IGNORE INTO sync_state
                (direction, key, status, attempts, first_seen, last_attempt, detail)
            VALUES (?, ?, ?, 0, ?, ?, ?)
            """,
            rows,
        )
        self._db.commit()
        return len(rows)

    def is_empty(self, direction: str) -> bool:
        cur = self._db.execute(
            "SELECT 1 FROM sync_state WHERE direction = ? LIMIT 1", (direction,)
        )
        return cur.fetchone() is None
