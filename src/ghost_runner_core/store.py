"""Minimal persistent store for M0: the kv table (turn counter, settings) and
the turns table. Schema follows docs/architecture_design.html §A8.3/§A8.6 —
the memory tables (events/facts/…) arrive in M2 as migrations on this file.

Synchronous sqlite3 by design: M0 writes are single rows on the event loop,
microseconds each. If profiling ever shows otherwise, this moves behind an
executor without changing callers.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY,
  origin TEXT NOT NULL CHECK (origin IN ('balloon','voice','proactive','tool')),
  owner_session TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  status TEXT NOT NULL CHECK (status IN ('active','done','cancelled','error')),
  cancel_reason TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  turn_id INTEGER NOT NULL REFERENCES turns(id),
  role TEXT NOT NULL CHECK (role IN ('user','assistant','tool')),
  text TEXT NOT NULL,
  ts INTEGER NOT NULL
);
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


class CoreStore:
    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
        cur = self._db.execute("SELECT version FROM schema_meta")
        row = cur.fetchone()
        if row is None:
            self._db.execute("INSERT INTO schema_meta (version) VALUES (1)")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- kv ------------------------------------------------------------------

    def kv_get(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def kv_set(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._db.commit()

    # -- turns ---------------------------------------------------------------

    def next_turn_id(self) -> int:
        """Monotonic turn counter, persisted so restarts never reuse ids (§A4.4)."""
        cur = self._db.execute(
            "INSERT INTO kv (key, value) VALUES ('turn_counter', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
            "RETURNING CAST(value AS INTEGER)"
        )
        turn_id = cur.fetchone()[0]
        self._db.commit()
        return turn_id

    def record_turn(self, turn_id: int, origin: str, owner_session: str) -> None:
        self._db.execute(
            "INSERT INTO turns (id, origin, owner_session, started_at, status) "
            "VALUES (?, ?, ?, ?, 'active')",
            (turn_id, origin, owner_session, _now_ms()),
        )
        self._db.commit()

    def finish_turn(self, turn_id: int, status: str, cancel_reason: str | None = None) -> None:
        if status not in ("done", "cancelled", "error"):
            raise ValueError(f"invalid terminal turn status {status!r}")
        self._db.execute(
            "UPDATE turns SET status = ?, ended_at = ?, cancel_reason = ? WHERE id = ?",
            (status, _now_ms(), cancel_reason, turn_id),
        )
        self._db.commit()

    def append_message(self, turn_id: int, role: str, text: str) -> None:
        self._db.execute(
            "INSERT INTO messages (turn_id, role, text, ts) VALUES (?, ?, ?, ?)",
            (turn_id, role, text, _now_ms()),
        )
        self._db.commit()

    def recent_messages(self, limit: int) -> list[tuple[str, str]]:
        """Newest-last (role, text) pairs for prompt assembly."""
        rows = self._db.execute(
            "SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(r[0], r[1]) for r in reversed(rows)]
