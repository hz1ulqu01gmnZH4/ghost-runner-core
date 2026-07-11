"""WsServer: protocol v1 over websockets (§A7).

Auth happens at the HTTP upgrade (Bearer header) when a token is configured;
loopback binds may run header-less (§A7.6). The first frame must be `hello`
within HELLO_TIMEOUT_S; afterwards only `command`/`ping`/`pong` are legal from
a client. Repeated protocol violations close the socket with 4400.
"""

from __future__ import annotations

import asyncio
import http
import logging
import secrets
from collections.abc import Awaitable, Callable

import websockets
from websockets.asyncio.server import Request, Response, ServerConnection

from ..envelope import (
    E_BAD_ENVELOPE,
    E_INTERNAL,
    E_UNSUPPORTED_VERSION,
    Envelope,
    ProtocolError,
    decode,
    error_envelope,
)
from .session import ClientSession, SessionManager

log = logging.getLogger(__name__)

HELLO_TIMEOUT_S = 5.0
MAX_VIOLATIONS = 3
CLOSE_PROTOCOL_VIOLATION = 4400
CLOSE_AUTH_FAILED = 4401
# Largest binary frame the core will send (announced in hello_ack.limits).
# TTS PCM is sliced to PCM_FRAME_BYTES (voice/tts_pipeline.py) well below this.
MAX_BIN_FRAME = 262_144

# command handler: (session_id, command_name, args) -> ack result payload
CommandHandler = Callable[[str, str, dict], Awaitable[dict]]
# stream resync: (session_id) -> None; re-sends active token-stream snapshots
StreamResync = Callable[[str], Awaitable[None]]


class WsServer:
    def __init__(self, bind: str, port: int, sessions: SessionManager,
                 handle_command: CommandHandler, auth_token: str | None,
                 resync_streams: StreamResync | None = None) -> None:
        self._bind = bind
        self._port = port
        self._sessions = sessions
        self._handle_command = handle_command
        self._auth_token = auth_token
        self._resync_streams = resync_streams
        self._server: websockets.asyncio.server.Server | None = None

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handler, self._bind, self._port,
            process_request=self._check_auth,
            # Default max_size is 1 MiB, which a base64 asr.transcribe frame
            # exceeds after ~4 s of 44.1 kHz stereo PTT audio. Sized to the
            # orchestrator's 10 MB audio cap (~13.4 MB as base64) plus headroom.
            max_size=16 * 1024 * 1024,
        )
        log.info("listening on ws://%s:%d", self._bind, self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def port(self) -> int:
        """The actually-bound port (differs from the configured one when 0)."""
        if self._server is None or not self._server.sockets:
            raise RuntimeError("server not started")
        return self._server.sockets[0].getsockname()[1]

    # -- upgrade-time auth (§A7.6) ---------------------------------------------

    def _check_auth(self, connection: ServerConnection,
                    request: Request) -> Response | None:
        if self._auth_token is None:
            return None  # loopback profile: config.py enforces token for non-loopback binds
        auth = request.headers.get("Authorization", "")
        if secrets.compare_digest(auth, f"Bearer {self._auth_token}"):
            return None
        log.warning("rejected connection: bad or missing Authorization header")
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, "unauthorized\n")

    # -- connection lifecycle ----------------------------------------------------

    async def _handler(self, ws: ServerConnection) -> None:
        session: ClientSession | None = None
        try:
            session = await self._handshake(ws)
            if session is None:
                return
            await self._serve_session(ws, session)
        except websockets.ConnectionClosed:
            pass
        finally:
            if session is not None:
                self._sessions.detach(session.session_id)
                log.info("session %s detached", session.session_id)

    async def _handshake(self, ws: ServerConnection) -> ClientSession | None:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=HELLO_TIMEOUT_S)
        except TimeoutError:
            await ws.close(CLOSE_PROTOCOL_VIOLATION, "hello timeout")
            return None
        try:
            env = decode(raw)
        except ProtocolError as exc:
            await ws.send(error_envelope(exc.code, exc.message).encode())
            await ws.close(CLOSE_PROTOCOL_VIOLATION, exc.code)
            return None
        if env.type != "hello":
            await ws.send(error_envelope(
                E_BAD_ENVELOPE, "first message must be hello").encode())
            await ws.close(CLOSE_PROTOCOL_VIOLATION, "no hello")
            return None
        accept_v = env.payload.get("accept_v")
        if not isinstance(accept_v, list) or 1 not in accept_v:
            await ws.send(error_envelope(
                E_UNSUPPORTED_VERSION, f"no common protocol version in {accept_v!r}").encode())
            await ws.close(CLOSE_PROTOCOL_VIOLATION, "version mismatch")
            return None

        session, replayed = await self._attach(ws, env)
        log.info("session %s attached (replay=%s)", session.session_id, replayed)
        return session

    async def _attach(self, ws: ServerConnection, hello: Envelope) -> tuple[ClientSession, bool]:
        resume = hello.payload.get("resume")
        session: ClientSession | None = None
        replay_tail: list[Envelope] | None = None
        if isinstance(resume, dict):
            key = resume.get("resume_key")
            last_seen = resume.get("last_seen_seq")
            if isinstance(key, str) and isinstance(last_seen, int):
                session = self._sessions.resume_session(key, ws)
                if session is not None:
                    replay_tail = self._sessions.replay.tail(last_seen)
                    if replay_tail is None:
                        # Past the pruned horizon: honest fresh session (§A5.5).
                        self._sessions.drop_session(session.session_id)
                        session = None
        if session is None:
            session = self._sessions.create_session(ws)
            replay_tail = None

        ack = Envelope(type="hello_ack", id=hello.id, session=session.session_id, payload={
            "session": session.session_id,
            "v": 1,
            "resume_key": session.resume_key,
            "replay": replay_tail is not None,
            "caps": {"consent": True},
            "emits": ["state", "turn", "token", "audio_meta"],
            "viseme_scheme": "none",
            "limits": {"max_bin_frame": MAX_BIN_FRAME},
            "rev": {"perception": 0, "settings": 0},
        })
        await ws.send(ack.encode())

        if replay_tail is not None:
            for entry in replay_tail:
                await session.send(entry)
        # Always re-emit the current lifecycle state so the avatar resyncs even
        # on a fresh session (§A5.5 item 3).
        snapshot = self._sessions.state_snapshot
        if snapshot is not None:
            resend = Envelope(type="state", payload=dict(snapshot.payload),
                              turn=snapshot.turn, seq=self._sessions.next_seq_for_resend())
            await session.send(resend)
        # Active streams resync by full snapshot (§A5.5 item 4) — a client that
        # dropped mid-stream gets its balloon text back without asking.
        if self._resync_streams is not None:
            await self._resync_streams(session.session_id)
        return session, replay_tail is not None

    async def _serve_session(self, ws: ServerConnection, session: ClientSession) -> None:
        violations = 0
        async for raw in ws:
            try:
                env = decode(raw)
            except ProtocolError as exc:
                violations += 1
                await session.send(error_envelope(exc.code, exc.message))
                if violations >= MAX_VIOLATIONS:
                    await ws.close(CLOSE_PROTOCOL_VIOLATION, "too many protocol violations")
                    return
                continue

            if env.type == "ping":
                await session.send(Envelope(type="pong", id=env.id))
            elif env.type == "pong":
                pass  # client answered nothing we sent; harmless
            elif env.type == "command":
                await self._dispatch_command(session, env)
            else:
                violations += 1
                await session.send(error_envelope(
                    E_BAD_ENVELOPE, f"unexpected client message type {env.type!r}", env.id))
                if violations >= MAX_VIOLATIONS:
                    await ws.close(CLOSE_PROTOCOL_VIOLATION, "too many protocol violations")
                    return

    async def _dispatch_command(self, session: ClientSession, env: Envelope) -> None:
        if env.id is None:
            await session.send(error_envelope(E_BAD_ENVELOPE, "command requires an id"))
            return
        name = env.payload.get("name")
        args = env.payload.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            await session.send(error_envelope(
                E_BAD_ENVELOPE, "command payload needs string name + object args", env.id))
            return
        try:
            result = await self._handle_command(session.session_id, name, args)
        except ProtocolError as exc:
            await session.send(error_envelope(exc.code, exc.message, env.id))
            return
        except Exception:
            log.exception("command %s failed", name)
            await session.send(error_envelope(
                E_INTERNAL, f"command {name} failed; see core log", env.id))
            return
        await session.send(Envelope(type="ack", id=env.id, payload={"ok": True, "result": result}))
