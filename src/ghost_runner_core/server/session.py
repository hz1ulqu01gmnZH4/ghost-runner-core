"""ClientSession, SessionManager, ReplayLog (§A4.2, §A5.5, §A7.1).

Durable notifications (state/turn/…) get a server-assigned monotonic seq and
land in the replay log; on resume the tail since last_seen_seq is replayed.
M0 note: every durable notification broadcasts to all sessions, so one global
log is equivalent to the spec's per-session logs; this changes when directed
messages (elicitations to consent-capable devices) arrive in M4/M5.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field

from websockets.exceptions import ConnectionClosed

from ..envelope import DURABLE_TYPES, Envelope

log = logging.getLogger(__name__)

REPLAY_MAX_ENTRIES = 500
REPLAY_MAX_AGE_MS = 24 * 3600 * 1000
RESUME_KEY_TTL_MS = 24 * 3600 * 1000


class ReplayLog:
    def __init__(self) -> None:
        self._entries: deque[Envelope] = deque()
        self._last_seq = 0        # highest seq ever appended
        self._pruned_below = 0    # every seq <= this has been pruned away

    def append(self, env: Envelope) -> None:
        assert env.seq is not None, "only seq-bearing envelopes belong in the replay log"
        self._last_seq = env.seq
        self._entries.append(env)
        self._prune()

    def _prune(self) -> None:
        cutoff = int(time.time() * 1000) - REPLAY_MAX_AGE_MS
        while self._entries and (
                len(self._entries) > REPLAY_MAX_ENTRIES
                or (self._entries[0].ts or 0) < cutoff):
            dropped = self._entries.popleft()
            self._pruned_below = dropped.seq

    def tail(self, since_seq: int) -> list[Envelope] | None:
        """Entries with seq > since_seq, or None if the log can no longer prove
        completeness (client is older than the pruned horizon → refuse resume;
        an honest fresh session beats a silent gap — §A5.5)."""
        self._prune()
        if since_seq >= self._last_seq:
            return []                      # client is fully current
        if since_seq < self._pruned_below:
            return None                    # gap falls in pruned territory
        return [e for e in self._entries if e.seq > since_seq]


@dataclass
class ClientSession:
    session_id: str
    resume_key: str
    ws: object | None = field(default=None, repr=False)  # websockets connection
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def connected(self) -> bool:
        return self.ws is not None

    async def send(self, env: Envelope) -> None:
        if self.ws is None:
            raise ConnectionError(f"session {self.session_id} is not connected")
        env.session = self.session_id
        await self.ws.send(env.encode())


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ClientSession] = {}
        self._by_resume_key: dict[str, str] = {}
        self._seq = 0
        self.replay = ReplayLog()
        self._state_snapshot: Envelope | None = None
        # S→C binary stream ids: even, never reused for the core's lifetime
        # (§A7.4 parity split — client-assigned C→S ids are odd).
        self._next_stream_id = 0

    def next_stream_id(self) -> int:
        self._next_stream_id += 2
        return self._next_stream_id

    # -- session lifecycle -----------------------------------------------------

    def create_session(self, ws: object) -> ClientSession:
        session = ClientSession(
            session_id=f"s-{secrets.token_urlsafe(8)}",
            resume_key=secrets.token_urlsafe(24),
            ws=ws,
        )
        self._sessions[session.session_id] = session
        self._by_resume_key[session.resume_key] = session.session_id
        return session

    def resume_session(self, resume_key: str, ws: object) -> ClientSession | None:
        """Reattach a socket to an existing session, or None if unknown/expired."""
        session_id = self._by_resume_key.get(resume_key)
        if session_id is None:
            return None
        session = self._sessions[session_id]
        if int(time.time() * 1000) - session.created_at_ms > RESUME_KEY_TTL_MS:
            self.drop_session(session_id)
            return None
        if session.ws is not None:
            # The old socket is a zombie (client reconnected before we noticed).
            log.info("resume displaces a live socket for %s", session_id)
        session.ws = ws
        return session

    def detach(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.ws = None  # kept for resume until TTL

    def drop_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._by_resume_key.pop(session.resume_key, None)

    # -- notification fan-out ----------------------------------------------------

    @property
    def state_snapshot(self) -> Envelope | None:
        return self._state_snapshot

    async def broadcast(self, env: Envelope) -> None:
        """Send to every connected session; durable types get seq + replay-log
        entry (§A7.1). State snapshots track the last NON-transient state so
        resume never replays a SUCCESS/ERROR flash (§A5.5)."""
        if env.type in DURABLE_TYPES:
            self._seq += 1
            env.seq = self._seq
            if env.ts is None:
                env.ts = int(time.time() * 1000)
            self.replay.append(env)
            if env.type == "state" and env.payload.get("state") not in ("SUCCESS", "ERROR"):
                self._state_snapshot = env
        for session in list(self._sessions.values()):
            if not session.connected:
                continue
            try:
                await session.send(env)
            except (ConnectionClosed, ConnectionError, OSError) as exc:
                # A dying socket must not break the fan-out to other sessions.
                # Anything else (e.g. an encode TypeError) is a bug and propagates.
                log.warning("send to %s failed (%s); detaching", session.session_id, exc)
                self.detach(session.session_id)

    async def broadcast_binary(self, frame: bytes) -> None:
        """Fan a pre-encoded binary frame (§A7.4) out to every connected session.

        Binary streams are transient by design: no seq, no replay-log entry — a
        resumed client gets a fresh audio_meta and audio resumes from the
        current chunk, it never rewinds (§A5.5).
        """
        for session in list(self._sessions.values()):
            if not session.connected:
                continue
            try:
                await session.ws.send(frame)  # bytes → binary WS frame
            except (ConnectionClosed, ConnectionError, OSError) as exc:
                log.warning("binary send to %s failed (%s); detaching",
                            session.session_id, exc)
                self.detach(session.session_id)

    async def send_to(self, session_id: str, env: Envelope) -> None:
        session = self._sessions.get(session_id)
        if session is None or not session.connected:
            raise ConnectionError(
                f"send_to {session_id}: session absent/disconnected; "
                f"cannot deliver {env.type}")
        await session.send(env)

    def next_seq_for_resend(self) -> int:
        """A fresh seq for re-sending the state snapshot at resume time."""
        self._seq += 1
        return self._seq
