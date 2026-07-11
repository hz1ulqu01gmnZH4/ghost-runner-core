"""Orchestrator: the only writer of lifecycle state (§A4.4).

State transitions follow the animate_vrm AgentDriver contract mechanically:
chat arrives → THINKING, first token → RESPONDING, done → SUCCESS (transient)
→ IDLE. Confidence is never decorated: M0 has no confidence model, so every
state carries the documented driver default (0.6) rather than an invented
number (behavior B7 arrives with the confidence model in M4).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time

from ..asr.client import AsrError, WhisperClient
from ..config import LlmConfig
from ..envelope import (
    E_BAD_ENVELOPE,
    E_INTERNAL,
    E_NOT_OWNER,
    E_UNAVAILABLE,
    E_UNKNOWN_COMMAND,
    Envelope,
    ProtocolError,
)
from ..llm.client import LlamaClient, LlmError
from ..llm.scheduler import PRIO_CHAT, LlmScheduler, stream_via_scheduler
from ..server.session import SessionManager
from ..skills.library import SkillError, SkillLibrary
from ..skills.sandbox import SkillExecutionError, run_skill
from ..store import CoreStore
from ..tts import IrodoriClient, TtsClient, TtsError
from ..voice.tts_pipeline import TurnSpeech
from .turns import Turn, TurnManager

log = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = 0.6  # the documented AgentDriver fallback, not an estimate
TRANSIENT_HOLD_S = 1.5    # SUCCESS/ERROR display hold before decaying to IDLE
MAX_AUDIO_BYTES = 10_000_000  # ~5 min of 16 kHz mono PCM16; PTT utterances are seconds


class Orchestrator:
    def __init__(self, store: CoreStore, sessions: SessionManager,
                 scheduler: LlmScheduler, llama: LlamaClient, llm_cfg: LlmConfig,
                 asr: WhisperClient | None = None,
                 skills: SkillLibrary | None = None,
                 tts: TtsClient | IrodoriClient | None = None) -> None:
        self._store = store
        self._sessions = sessions
        self._scheduler = scheduler
        self._llama = llama
        self._cfg = llm_cfg
        self._asr = asr
        self._skills = skills
        self._tts = tts
        self.turns = TurnManager(store, self._notify_turn)
        # Active token stream for stream.sync resync: (stream_id, parts, next_seq, turn_id)
        self._stream: dict | None = None

    # -- state & turn notifications ------------------------------------------------

    async def set_state(self, state: str, *, turn: int | None = None,
                        message: str = "", attention: int = 0) -> None:
        await self._sessions.broadcast(Envelope(type="state", turn=turn, payload={
            "state": state,
            "confidence": DEFAULT_CONFIDENCE,
            "attention": attention,
            "target": None,
            "message": message,
        }))

    async def _notify_turn(self, turn: Turn, event: str, reason: str | None) -> None:
        payload: dict = {"event": event, "origin": turn.origin,
                         "owner_session": turn.owner_session}
        if reason is not None:
            payload["reason"] = reason
        await self._sessions.broadcast(Envelope(type="turn", turn=turn.id, payload=payload))
        if event == "cancelled":
            self._scheduler.cancel_turn(turn.id)

    async def startup(self) -> None:
        await self.set_state("IDLE")

    # -- command routing (§A7.3, M0 subset) ---------------------------------------

    async def handle_command(self, session_id: str, name: str, args: dict) -> dict:
        if name == "chat.send":
            return await self._cmd_chat_send(session_id, args)
        if name == "turn.cancel":
            return await self._cmd_turn_cancel(session_id, args)
        if name == "stream.sync":
            return await self._cmd_stream_sync(session_id, args)
        if name == "asr.transcribe":
            return await self._cmd_asr_transcribe(session_id, args)
        if name == "skill.list":
            return self._cmd_skill_list()
        if name == "skill.run":
            return await self._cmd_skill_run(session_id, args)
        raise ProtocolError(E_UNKNOWN_COMMAND, f"unknown command {name!r}")

    async def _cmd_chat_send(self, session_id: str, args: dict) -> dict:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ProtocolError(E_BAD_ENVELOPE, "chat.send needs a non-empty text argument")
        turn = await self.turns.begin("balloon", session_id)
        turn.task = asyncio.get_running_loop().create_task(
            self._run_chat_turn(turn, text), name=f"turn-{turn.id}")
        return {"turn": turn.id}

    async def _cmd_turn_cancel(self, session_id: str, args: dict) -> dict:
        turn_id = args.get("turn")
        if not isinstance(turn_id, int):
            raise ProtocolError(E_BAD_ENVELOPE, "turn.cancel needs an integer turn argument")
        try:
            cancelled = await self.turns.cancel(turn_id, "user_cancel", by_session=session_id)
        except PermissionError as exc:
            raise ProtocolError(E_NOT_OWNER, str(exc)) from exc
        return {"cancelled": cancelled}  # False = already finished; idempotent (§A7.3)

    async def _cmd_stream_sync(self, session_id: str, args: dict) -> dict:
        stream_id = args.get("stream")
        if not isinstance(stream_id, str):
            raise ProtocolError(E_BAD_ENVELOPE, "stream.sync needs a string stream argument")
        stream = self._stream
        if stream is None or stream["id"] != stream_id:
            return {"active": False}  # stream finished or unknown: nothing to resync
        await self._send_stream_snapshot(session_id, stream)
        return {"active": True}

    async def _cmd_asr_transcribe(self, session_id: str, args: dict) -> dict:
        """Push-to-talk transcription (M1 voice loop, first slice). Stateless
        request/response: the client decides what to do with the transcript
        (it shows and sends it as a normal chat.send, keeping one turn path)."""
        if self._asr is None:
            raise ProtocolError(
                E_UNAVAILABLE, "asr is not configured on this core (no [asr] config section)")
        audio_b64 = args.get("audio")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise ProtocolError(E_BAD_ENVELOPE, "asr.transcribe needs a base64 audio argument")
        try:
            wav = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolError(
                E_BAD_ENVELOPE, f"asr.transcribe audio is not valid base64: {exc}") from exc
        if len(wav) > MAX_AUDIO_BYTES:
            raise ProtocolError(
                E_BAD_ENVELOPE,
                f"asr.transcribe audio too large ({len(wav)} > {MAX_AUDIO_BYTES} bytes)")
        if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            raise ProtocolError(
                E_BAD_ENVELOPE, "asr.transcribe audio must be a RIFF/WAVE file")
        try:
            text = await self._asr.transcribe(wav)
        except AsrError as exc:
            log.error("asr.transcribe for %s failed: %s", session_id, exc)
            raise ProtocolError(E_UNAVAILABLE, _truncate(f"ASR error: {exc}")) from exc
        return {"text": text}

    def _cmd_skill_list(self) -> dict:
        """F3 slice 1: enumerate the promoted skill library."""
        if self._skills is None:
            raise ProtocolError(
                E_UNAVAILABLE, "skills are not configured on this core (no [skills] section)")
        return {"skills": [
            {"name": s.name, "version": s.version, "description": s.description}
            for s in self._skills.list()
        ]}

    async def _cmd_skill_run(self, session_id: str, args: dict) -> dict:
        """Run one promoted skill in the bwrap sandbox. Stateless like
        asr.transcribe: no turn, no state broadcast — the WORKING-pose tie-in
        arrives when the LLM invokes skills inside a turn (M5 proper)."""
        if self._skills is None:
            raise ProtocolError(
                E_UNAVAILABLE, "skills are not configured on this core (no [skills] section)")
        name = args.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError(E_BAD_ENVELOPE, "skill.run needs a string name argument")
        skill_args = args.get("args", {})
        if not isinstance(skill_args, dict):
            raise ProtocolError(E_BAD_ENVELOPE, "skill.run args must be an object")
        skill = self._skills.get(name)
        if skill is None:
            raise ProtocolError(E_BAD_ENVELOPE, f"unknown skill {name!r}")
        try:
            result = await run_skill(self._skills, skill, skill_args)
        except SkillError as exc:
            # Integrity failure (hash drift, sandbox gone): operator problem.
            log.error("skill.run %s for %s refused: %s", name, session_id, exc)
            raise ProtocolError(E_UNAVAILABLE, _truncate(str(exc))) from exc
        except SkillExecutionError as exc:
            log.error("skill.run %s for %s failed: %s", name, session_id, exc)
            raise ProtocolError(E_INTERNAL, _truncate(str(exc))) from exc
        return {"result": result}

    async def resync_streams(self, session_id: str) -> None:
        """Server-initiated stream resync at (re)attach (§A5.5 item 4): a client
        that dropped mid-stream gets the active stream's full text back."""
        stream = self._stream
        if stream is not None:
            await self._send_stream_snapshot(session_id, stream)

    async def _send_stream_snapshot(self, session_id: str, stream: dict) -> None:
        await self._sessions.send_to(session_id, Envelope(
            type="token", turn=stream["turn"], payload={
                "stream": stream["id"],
                "seq": stream["seq"],
                "delta": "",
                "done": False,
                "snapshot": {"seq_base": stream["seq"], "text": "".join(stream["parts"])},
            }))

    # -- the chat turn itself -----------------------------------------------------

    def _prompt(self, text: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self._cfg.system_prompt}]
        for role, msg in self._store.recent_messages(self._cfg.history_messages):
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": msg})
        messages.append({"role": "user", "content": text})
        return messages

    async def _run_chat_turn(self, turn: Turn, text: str) -> None:
        stream_id = f"t{turn.id}"
        speech: TurnSpeech | None = None
        started = time.monotonic()
        first_token_ms: int | None = None
        try:
            self._store.append_message(turn.id, "user", text)
            messages = self._prompt(text)
            await self.set_state("THINKING", turn=turn.id)
            if self._tts is not None:
                speech = TurnSpeech(self._tts, self._sessions, turn.id)

            state = {"id": stream_id, "parts": [], "seq": 0, "turn": turn.id}
            self._stream = state
            first = True

            async def on_delta(delta: str) -> None:
                nonlocal first, first_token_ms
                # Defense in depth (§A4.4): cancellation already stops the
                # scheduler job, but a delta racing the cancel must not leak out.
                if self.turns.is_obsolete(turn.id):
                    return
                if first:
                    first = False
                    first_token_ms = int((time.monotonic() - started) * 1000)
                    await self.set_state("RESPONDING", turn=turn.id)
                await self._sessions.broadcast(Envelope(type="token", turn=turn.id, payload={
                    "stream": stream_id, "seq": state["seq"], "delta": delta, "done": False,
                }))
                state["parts"].append(delta)
                state["seq"] += 1
                if speech is not None:
                    speech.feed(delta)

            await stream_via_scheduler(
                self._scheduler, PRIO_CHAT, turn.id,
                lambda: self._llama.chat_stream(messages), on_delta,
                label=f"chat-t{turn.id}")

            full_text = "".join(state["parts"])
            await self._sessions.broadcast(Envelope(type="token", turn=turn.id, payload={
                "stream": stream_id, "seq": state["seq"], "delta": "",
                "done": True, "finish": "stop",
            }))
            self._stream = None
            self._store.append_message(turn.id, "assistant", full_text)

            if speech is not None:
                # The turn stays open (RESPONDING) until speech drains — speech
                # is part of the reply, and barge-in must find it cancellable
                # through this turn (§A5.2).
                await speech.finish()
                # §8.2 latency harness: one honest line per voice turn, both
                # measured from turn start (first_audio None = silent reply).
                log.info("turn %d latency: first_token_ms=%s first_audio_ms=%s",
                         turn.id, first_token_ms, speech.first_audio_ms)
                speech = None

            await self.set_state("SUCCESS", turn=turn.id)
            await self.turns.finish(turn.id, "done")
            await self._decay_to_idle()

        except asyncio.CancelledError:
            # TurnManager.cancel already notified everyone; just stop cleanly —
            # including in-flight synthesis (abort ≤150 ms, §A5.2).
            if speech is not None:
                await speech.cancel()
            self._stream = None
            raise
        except TtsError as exc:
            # The text reply already streamed; the voice failing is still a
            # loud turn error, never a silent mute (no-fallback policy).
            log.error("turn %d TTS failure: %s", turn.id, exc)
            if speech is not None:
                # §8.2 harness must sample failed voice turns too.
                log.info("turn %d latency: first_token_ms=%s first_audio_ms=%s (failed)",
                         turn.id, first_token_ms, speech.first_audio_ms)
            self._stream = None
            await self.set_state("ERROR", turn=turn.id,
                                 message=_truncate(f"TTS error: {exc}"))
            await self.turns.finish(turn.id, "error")
            await self._decay_to_idle()
        except LlmError as exc:
            log.error("turn %d LLM failure: %s", turn.id, exc)
            if speech is not None:
                await speech.cancel()
            self._stream = None
            await self.set_state("ERROR", turn=turn.id,
                                 message=_truncate(f"LLM error: {exc}"))
            await self.turns.finish(turn.id, "error")
            await self._decay_to_idle()
        except Exception:
            log.exception("turn %d failed", turn.id)
            if speech is not None:
                await speech.cancel()
            self._stream = None
            await self.set_state("ERROR", turn=turn.id, message="internal error; see core log")
            await self.turns.finish(turn.id, "error")
            await self._decay_to_idle()

    async def _decay_to_idle(self) -> None:
        """SUCCESS/ERROR are transient (MascotState contract): hold, then IDLE —
        unless a newer turn already owns the foreground."""
        await asyncio.sleep(TRANSIENT_HOLD_S)
        if self.turns.current is None:
            await self.set_state("IDLE")


def _truncate(message: str, limit: int = 120) -> str:
    return message if len(message) <= limit else message[: limit - 1] + "…"
