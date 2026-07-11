"""Protocol v1 envelope: encode/decode/validate.

Wire contract: docs/architecture_design.html §A7 in the ghost-runner repo.
Decode failures raise ProtocolError('bad_envelope', ...) — the server replies
with an error envelope and drops the frame; it never guesses at intent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

PROTOCOL_VERSION = 1

# Message types the core knows. Additive-only within v1.
# audio_meta (M1): transient S→C stream open/close for binary TTS audio (§A7.2);
# unsequenced like token — streams resync by fresh meta, not by replay.
VALID_TYPES = frozenset({
    "hello", "hello_ack",
    "state", "turn", "token", "asr_partial", "audio_meta",
    "command", "ack", "error",
    "ping", "pong",
})

# Durable S->C notifications get a monotonic seq and land in the replay log (§A7.1).
DURABLE_TYPES = frozenset({"state", "turn", "perception.status", "suggestion", "elicitation"})

# Error codes (§A7.5).
E_BAD_ENVELOPE = "bad_envelope"
E_UNSUPPORTED_VERSION = "unsupported_version"
E_AUTH_FAILED = "auth_failed"
E_UNKNOWN_COMMAND = "unknown_command"
E_TURN_OBSOLETE = "turn_obsolete"
E_POLICY_DENIED = "policy_denied"
E_NOT_OWNER = "not_owner"
E_CONFLICT = "conflict"
E_ALREADY_RESOLVED = "already_resolved"
E_UNAVAILABLE = "unavailable"
E_INTERNAL = "internal"


class ProtocolError(Exception):
    """A wire-level violation. `code` is one of the E_* constants."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class Envelope:
    type: str
    payload: dict = field(default_factory=dict)
    id: str | None = None
    seq: int | None = None
    turn: int | None = None
    session: str | None = None
    ts: int | None = None  # None = stamp at encode time

    def encode(self) -> str:
        obj: dict = {"v": PROTOCOL_VERSION, "type": self.type,
                     "ts": self.ts if self.ts is not None else now_ms()}
        if self.id is not None:
            obj["id"] = self.id
        if self.seq is not None:
            obj["seq"] = self.seq
        if self.session is not None:
            obj["session"] = self.session
        if self.turn is not None:
            obj["turn"] = self.turn
        obj["payload"] = self.payload
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def decode(text: str | bytes) -> Envelope:
    """Parse + validate an inbound text frame. Raises ProtocolError, never returns junk."""
    if isinstance(text, bytes):
        raise ProtocolError(E_BAD_ENVELOPE, "binary frames are not accepted in M0")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(E_BAD_ENVELOPE, f"unparseable JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(E_BAD_ENVELOPE, "envelope must be a JSON object")

    v = obj.get("v")
    if v != PROTOCOL_VERSION:
        raise ProtocolError(E_UNSUPPORTED_VERSION, f"unsupported protocol version {v!r}")

    mtype = obj.get("type")
    if not isinstance(mtype, str) or mtype not in VALID_TYPES:
        raise ProtocolError(E_BAD_ENVELOPE, f"unknown message type {mtype!r}")

    payload = obj.get("payload", {})
    if not isinstance(payload, dict):
        raise ProtocolError(E_BAD_ENVELOPE, "payload must be an object")

    env_id = obj.get("id")
    if env_id is not None and not isinstance(env_id, str):
        raise ProtocolError(E_BAD_ENVELOPE, "id must be a string")

    turn = obj.get("turn")
    if turn is not None and not isinstance(turn, int):
        raise ProtocolError(E_BAD_ENVELOPE, "turn must be an integer")

    ts = obj.get("ts", 0)
    if not isinstance(ts, int):
        raise ProtocolError(E_BAD_ENVELOPE, "ts must be an integer")

    seq = obj.get("seq")
    if seq is not None and not isinstance(seq, int):
        raise ProtocolError(E_BAD_ENVELOPE, "seq must be an integer")

    session = obj.get("session")
    if session is not None and not isinstance(session, str):
        raise ProtocolError(E_BAD_ENVELOPE, "session must be a string")

    return Envelope(type=mtype, payload=payload, id=env_id, seq=seq,
                    turn=turn, session=session, ts=ts)


def error_envelope(code: str, message: str, reply_to: str | None = None,
                   data: dict | None = None) -> Envelope:
    payload: dict = {"code": code, "message": message}
    if data:
        payload["data"] = data
    return Envelope(type="error", payload=payload, id=reply_to)
