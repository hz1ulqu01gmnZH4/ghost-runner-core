"""Turn-scoped speech synthesis (§A4.3 TtsPipeline, §A5.2).

One TurnSpeech per chat turn: token deltas feed the sentence chunker, a single
worker task synthesizes sentence by sentence and streams the result as ONE
binary PCM stream (§A7.4 kind 0x01) wrapped in audio_meta open/close envelopes.
Synthesis lags the token stream by design — the worker drains its sentence
queue long after the LLM finished — so the turn stays open (RESPONDING) until
speech is done, and barge-in cancels speech through the same turn.cancel path
that kills everything else.

audio_meta is sent right before the first PCM frame, not at turn start: a
reply whose synthesis fails outright must not open a stream it can never fill.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Protocol

from ..binframe import KIND_TTS_PCM, encode_frame
from ..envelope import Envelope
from ..server.session import SessionManager
from .chunker import SentenceChunker


# One fish-speech segment can be a whole sentence of PCM (seconds of audio,
# potentially megabytes) delivered as a single chunk; frames on the wire stay
# small so the client starts playing — and can stop on barge-in — mid-segment.
PCM_FRAME_BYTES = 65_536


class SynthClient(Protocol):
    """What TurnSpeech needs from a TTS client (see tts.client.TtsClient)."""

    @property
    def sample_rate(self) -> int: ...

    def synth_stream(self, text: str) -> AsyncIterator[bytes]: ...


class TurnSpeech:
    """Speech output for one turn. feed() during token streaming, then either
    await finish() (drains the queue, closes the stream) or cancel()."""

    def __init__(self, tts: SynthClient, sessions: SessionManager, turn_id: int) -> None:
        self._tts = tts
        self._sessions = sessions
        self._turn_id = turn_id
        self._stream_id = sessions.next_stream_id()
        self._chunker = SentenceChunker()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._seq = 0
        self._opened = False
        self._started_at = time.monotonic()
        self.first_audio_ms: int | None = None  # §8.2 latency instrumentation
        self._worker = asyncio.get_running_loop().create_task(
            self._run(), name=f"speech-t{turn_id}")

    def feed(self, delta: str) -> None:
        for sentence in self._chunker.feed(delta):
            self._queue.put_nowait(sentence)

    async def finish(self) -> None:
        """Token stream is over: speak what remains, then close the stream.
        Raises whatever the worker raised (TtsError included) — a silent
        half-spoken reply is not a success."""
        for sentence in self._chunker.flush():
            self._queue.put_nowait(sentence)
        self._queue.put_nowait(None)
        await self._worker

    async def cancel(self) -> None:
        """Barge-in/cancel: stop synthesis now. No closing audio_meta — the
        turn is obsolete and the client drops the stream on that signal."""
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            sentence = await self._queue.get()
            if sentence is None:
                break
            async for pcm in self._tts.synth_stream(sentence):
                if not self._opened:
                    # First audio of the turn: open the stream. sample_rate was
                    # pinned by the client's startup format probe.
                    await self._send_meta(last=False)
                    self._opened = True
                    self.first_audio_ms = int(
                        (time.monotonic() - self._started_at) * 1000)
                for off in range(0, len(pcm), PCM_FRAME_BYTES):
                    await self._sessions.broadcast_binary(encode_frame(
                        KIND_TTS_PCM, self._stream_id, self._seq,
                        pcm[off:off + PCM_FRAME_BYTES]))
                    self._seq += 1
        if self._opened:
            await self._send_meta(last=True)

    async def _send_meta(self, *, last: bool) -> None:
        await self._sessions.broadcast(Envelope(
            type="audio_meta", turn=self._turn_id, payload={
                "stream": self._stream_id,
                "codec": "pcm_s16le",
                "sample_rate": self._tts.sample_rate,
                "channels": 1,
                "last": last,
            }))
